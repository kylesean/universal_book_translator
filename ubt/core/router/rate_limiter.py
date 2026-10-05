"""Adaptive token bucket rate limiter with AIMD dynamic throttling.

Providers rate-limit across requests per minute (RPM) and tokens per minute
(TPM). Dual buckets refill continuously and are consumed by each ``acquire``
call, adapting via AIMD on HTTP 429 signals.

:class:`AdaptiveTokenBucket` is in-process and serves single-process callers
(the CLI, an in-process API). :class:`SqliteTokenBucket` keeps the same bucket
state in a WAL SQLite file so several ``ubt worker`` *processes* sharing one API
credential split a single budget instead of each minting a full bucket and
multiplying the request rate — and the bill — by the process count.
"""

import asyncio
import logging
import random
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: AIMD TPM ceiling as a multiple of ``initial_tpm`` for a bare bucket (no
#: explicit ``max_tpm``). Matches UBTConfig's 600k / 100k default.
_DEFAULT_MAX_TPM_MULTIPLIER = 6


class AdaptiveTokenBucket:
    """Adaptive dual token bucket (RPM + TPM) with AIMD on HTTP 429s."""

    def __init__(
        self,
        initial_rpm: int = 60,
        initial_tpm: int = 100_000,
        min_rpm: int | None = None,
        max_rpm: int = 240,
        min_tpm: int | None = None,
        max_tpm: int | None = None,
        backoff_cooldown_sec: float = 0.0,
    ) -> None:
        # --- RPM bucket (requests) ---
        # The default floor must never exceed the configured rate, or an explicit
        # low ``rate_limit_rpm`` (e.g. 2) is silently raised to the floor (5)
        # from t=0 — a deliberate budget cap the user set is overwritten. An
        # explicit ``min_rpm`` is honoured as a floor.
        self.min_rpm = min_rpm if min_rpm is not None else min(5, initial_rpm)
        self.max_rpm = max_rpm
        # ``report_429`` computes ``max(min_rpm, capacity * 0.5)``, so starting
        # at or above the floor means the first 429 halves the rate rather than
        # *raising* it (the original reverse-speedup bug).
        rpm = max(initial_rpm, self.min_rpm)
        self.capacity: float = float(rpm)
        self.tokens: float = float(rpm)
        self.fill_rate: float = float(rpm) / 60.0  # tokens per second
        # --- TPM bucket (tokens) — --
        self.min_tpm = min_tpm if min_tpm is not None else min(1_000, initial_tpm)
        tpm = max(initial_tpm, self.min_tpm)
        self.initial_tpm = tpm
        self.tpm_capacity: float = float(tpm)
        self.tpm_tokens: float = float(tpm)
        self.tpm_fill_rate: float = float(tpm) / 60.0  # tokens per second
        # Default the TPM ceiling to 6x the starting budget (matching UBTConfig's
        # 600k/100k). Capping it at ``initial_tpm`` meant a bare bucket could
        # never grow its TPM side past the starting value after a 429 halved it;
        # production paths pass an explicit ``max_tpm``.
        self.max_tpm = max_tpm if max_tpm is not None else tpm * _DEFAULT_MAX_TPM_MULTIPLIER
        # --- shared state ---
        self.backoff_cooldown_sec = backoff_cooldown_sec
        self.last_backoff_monotonic = 0.0
        self.last_update = time.monotonic()
        self._lock: asyncio.Lock | None = None
        self._thread_lock = threading.Lock()
        self.consecutive_429 = 0
        self._waiters: int = 0

    @property
    def lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    @lock.setter
    def lock(self, val: asyncio.Lock) -> None:
        self._lock = val

    async def acquire(self, estimated_tokens: int = 500) -> None:
        """Acquire one RPM token and ``estimated_tokens`` TPM tokens.

        The estimated token cost is consumed from the TPM bucket. A request larger
        than total TPM capacity is clamped so it drains the bucket instead of deadlocking.
        """
        raw_need = float(max(1, estimated_tokens))
        while True:
            sleep_duration: float = 0.0
            async with self.lock:
                with self._thread_lock:
                    now = time.monotonic()
                    elapsed = now - self.last_update
                    self.last_update = now
                    self.tokens = min(self.capacity, self.tokens + elapsed * self.fill_rate)
                    self.tpm_tokens = min(
                        self.tpm_capacity, self.tpm_tokens + elapsed * self.tpm_fill_rate
                    )

                    # Re-clamp need against current tpm_capacity inside lock so that
                    # capacity reductions from report_429() do not permanently deadlock.
                    need = min(raw_need, self.tpm_capacity)

                    if self.tokens >= 1.0 and self.tpm_tokens >= need:
                        self.tokens -= 1.0
                        self.tpm_tokens -= need
                        return

                    # Wait for whichever constraint binds the request.
                    wait_rpm = (1.0 - self.tokens) / self.fill_rate if self.tokens < 1.0 else 0.0
                    wait_tpm = (
                        (need - self.tpm_tokens) / self.tpm_fill_rate
                        if self.tpm_tokens < need
                        else 0.0
                    )
                    wait_time = max(wait_rpm, wait_tpm)
                    self._waiters += 1
                    # Stagger jitter proportionally to the number of waiters to prevent thundering herd
                    jitter = random.uniform(0.01, 0.05) + min(0.5, self._waiters * 0.02)
                    sleep_duration = max(0.01, min(wait_time + jitter, 5.0))

            try:
                await asyncio.sleep(sleep_duration)
            finally:
                # A cancellation during the sleep (job cancel/shutdown) must
                # still release the waiter slot, or every later acquire pays
                # stale-waiter jitter for the rest of the process.
                with self._thread_lock:
                    self._waiters = max(0, self._waiters - 1)

    def report_429(self) -> None:
        """Multiplicative decrease on rate-limit exhaustion, once per episode:
        a 429 burst from concurrent requests in flight must cost one halving,
        not one per report, so a repeat while ``consecutive_429`` is non-zero
        only counts and the first success ends the episode.
        """
        with self._thread_lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.last_update = now
            self.tokens = min(self.capacity, self.tokens + elapsed * self.fill_rate)
            self.tpm_tokens = min(self.tpm_capacity, self.tpm_tokens + elapsed * self.tpm_fill_rate)
            self.consecutive_429 += 1
            if (now - self.last_backoff_monotonic) < self.backoff_cooldown_sec:
                return
            if self.backoff_cooldown_sec == 0.0 and self.consecutive_429 > 1:
                return
            self.last_backoff_monotonic = now
            new_rpm = max(float(self.min_rpm), self.capacity * 0.5)
            self.capacity = new_rpm
            self.fill_rate = self.capacity / 60.0
            # Deplete current tokens to prevent burst
            self.tokens = min(self.tokens, 0.0)
            # 429s usually name the binding quota — assume both, since
            # a token-heavy burst is the common trigger for the TPM axis.
            self.tpm_capacity = max(float(self.min_tpm), self.tpm_capacity * 0.5)
            self.tpm_fill_rate = self.tpm_capacity / 60.0
            self.tpm_tokens = min(self.tpm_tokens, 0.0)
            if self.capacity <= self.min_rpm or self.tpm_capacity <= self.min_tpm:
                logger.warning(
                    "Rate limiter reached its floor after a 429 episode (%.0f rpm, "
                    "%.0f tpm); the run stays clamped until successes start again",
                    self.capacity,
                    self.tpm_capacity,
                )

    def report_success(self) -> None:
        """Trigger additive increase (AI) on successful API responses."""
        with self._thread_lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.last_update = now
            self.tokens = min(self.capacity, self.tokens + elapsed * self.fill_rate)
            self.tpm_tokens = min(self.tpm_capacity, self.tpm_tokens + elapsed * self.tpm_fill_rate)
            if self.consecutive_429 > 0:
                self.consecutive_429 = 0
            # Standard congestion-avoidance AIMD: capacity increases additively by at most
            # 1.0 RPM per full window (1 / capacity per request), avoiding bursty saw-tooth oscillations
            # under concurrent completions.
            increment = min(1.0, max(0.05, 1.0 / max(1.0, self.capacity)))
            new_rpm = min(float(self.max_rpm), self.capacity + increment)
            self.capacity = new_rpm
            self.fill_rate = self.capacity / 60.0
            # Recover TPM capacity by one second's worth of refill per
            # success — slow climb back, never above the configured ceiling.
            if self.tpm_capacity < self.max_tpm:
                self.tpm_capacity = min(
                    float(self.max_tpm), self.tpm_capacity + self.initial_tpm / 60.0
                )
                self.tpm_fill_rate = self.tpm_capacity / 60.0

    async def report_429_async(self) -> None:
        self.report_429()

    async def report_success_async(self) -> None:
        self.report_success()


class NullRateLimiter(AdaptiveTokenBucket):
    """A limiter that never gates, for traffic that cannot be rate limited.

    A mock provider has no credential to protect and no 429s to learn from, so
    an offline full-book run must not be throttled against a real bucket's
    budget — it only wastes wall clock.
    """

    async def acquire(self, estimated_tokens: int = 500) -> None:
        return None

    def report_429(self) -> None:
        return None

    def report_success(self) -> None:
        return None


class _BucketState:
    """A single instant of the shared dual-bucket, read from the WAL row."""

    __slots__ = (
        "rpm_capacity",
        "rpm_tokens",
        "tpm_capacity",
        "tpm_tokens",
        "consecutive_429",
        "last_backoff",
    )

    def __init__(self, row: sqlite3.Row) -> None:
        self.rpm_capacity = float(row["rpm_capacity"])
        self.rpm_tokens = float(row["rpm_tokens"])
        self.tpm_capacity = float(row["tpm_capacity"])
        self.tpm_tokens = float(row["tpm_tokens"])
        self.consecutive_429 = int(row["consecutive_429"])
        self.last_backoff = (
            float(row["last_backoff"]) if "last_backoff" in tuple(row.keys()) else 0.0
        )


class SqliteTokenBucket(AdaptiveTokenBucket):
    """Cross-process dual token bucket backed by a shared WAL SQLite file.

    Subclasses :class:`AdaptiveTokenBucket` so it satisfies every annotation and
    call site, but overrides the three mutating methods to operate on a shared
    row under ``BEGIN IMMEDIATE``. Wall-clock ``time.time()`` is used for refill
    because ``time.monotonic()`` is not comparable across processes.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        bucket_key: str = "global",
        initial_rpm: int = 60,
        initial_tpm: int = 100_000,
        min_rpm: int | None = None,
        max_rpm: int = 240,
        min_tpm: int | None = None,
        max_tpm: int | None = None,
        busy_timeout: float = 30.0,
        backoff_cooldown_sec: float = 3.0,
    ) -> None:
        super().__init__(
            initial_rpm=initial_rpm,
            initial_tpm=initial_tpm,
            min_rpm=min_rpm,
            max_rpm=max_rpm,
            min_tpm=min_tpm,
            max_tpm=max_tpm,
            backoff_cooldown_sec=backoff_cooldown_sec,
        )
        # Clamp to the AIMD floor so a seeded row can never start below it (see
        # AdaptiveTokenBucket.__init__); the key keeps the pre-clamp ints' shape.
        eff_rpm = max(int(initial_rpm), int(self.min_rpm))
        eff_tpm = int(max(initial_tpm, self.min_tpm))
        self._initial_rpm = float(eff_rpm)
        self._initial_tpm = float(eff_tpm)
        self._closed = False
        # The configured rates are part of the row identity, so a config change
        # (e.g. raising UBT_RATE_LIMIT_RPM) starts a clean bucket instead of an
        # existing row outranking this process's settings and inheriting an old
        # episode's capacity. The abandoned rows are a few bytes each.
        self._key = f"{bucket_key}@{eff_rpm}rpm/{eff_tpm}tpm"
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self._path),
            timeout=busy_timeout,
            isolation_level=None,  # we drive BEGIN/COMMIT manually
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS token_bucket (
                key TEXT PRIMARY KEY,
                rpm_capacity REAL NOT NULL,
                rpm_tokens REAL NOT NULL,
                tpm_capacity REAL NOT NULL,
                tpm_tokens REAL NOT NULL,
                last_update REAL NOT NULL,
                consecutive_429 INTEGER NOT NULL DEFAULT 0,
                last_backoff REAL NOT NULL DEFAULT 0.0
            )
            """
        )
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(token_bucket)").fetchall()}
        if "last_backoff" not in cols:
            self._conn.execute(
                "ALTER TABLE token_bucket ADD COLUMN last_backoff REAL NOT NULL DEFAULT 0.0"
            )
        now = time.time()
        self._conn.execute(
            """
            INSERT OR IGNORE INTO token_bucket (
                key, rpm_capacity, rpm_tokens, tpm_capacity, tpm_tokens, last_update,
                consecutive_429, last_backoff
            ) VALUES (?, ?, ?, ?, ?, ?, 0, 0.0)
            """,
            (
                self._key,
                self._initial_rpm,
                self._initial_rpm,
                self._initial_tpm,
                self._initial_tpm,
                now,
            ),
        )

    @contextmanager
    def _txn(self) -> Iterator[sqlite3.Connection]:
        """Serialize the read-modify-write across processes."""
        with self._thread_lock:
            self._conn.execute("BEGIN IMMEDIATE;")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK;")
                raise
            else:
                self._conn.execute("COMMIT;")

    @staticmethod
    def _refill(state: _BucketState, now: float, last_update: float) -> None:
        """Add elapsed refill to both buckets, clamped to their capacities."""
        elapsed = max(0.0, now - last_update)
        state.rpm_tokens = min(
            state.rpm_capacity, state.rpm_tokens + elapsed * (state.rpm_capacity / 60.0)
        )
        state.tpm_tokens = min(
            state.tpm_capacity, state.tpm_tokens + elapsed * (state.tpm_capacity / 60.0)
        )

    def _load(self, conn: sqlite3.Connection) -> tuple[_BucketState, float]:
        row = conn.execute("SELECT * FROM token_bucket WHERE key = ?", (self._key,)).fetchone()
        if row is None:  # another process reset the file; reseed lazily
            now = time.time()
            conn.execute(
                """
                INSERT OR IGNORE INTO token_bucket (
                    key, rpm_capacity, rpm_tokens, tpm_capacity, tpm_tokens, last_update,
                    consecutive_429
                ) VALUES (?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    self._key,
                    self._initial_rpm,
                    self._initial_rpm,
                    self._initial_tpm,
                    self._initial_tpm,
                    now,
                ),
            )
            row = conn.execute("SELECT * FROM token_bucket WHERE key = ?", (self._key,)).fetchone()
            if row is None:
                raise RuntimeError(
                    f"token bucket row {self._key!r} vanished and could not be reseeded"
                )
        return _BucketState(row), float(row["last_update"])

    def _acquire_step(self, raw_need: float) -> tuple[bool, float]:
        """One serialized read-modify-write against the shared row.

        Runs on a worker thread (see :meth:`acquire`): ``BEGIN IMMEDIATE`` can
        wait out the whole ``busy_timeout`` under contention, and holding the
        event loop through that is the same blocking-the-loop failure the shared
        bucket exists to avoid. ``_txn`` already takes ``_thread_lock``, so this
        stays mutually exclusive with the synchronous ``report_*`` mutators even
        once ``acquire`` moves off the loop.
        """
        with self._thread_lock:
            if self._closed:
                return True, 0.0
        with self._txn() as conn:
            state, last_update = self._load(conn)
            now = time.time()
            self._refill(state, now, last_update)
            need = min(raw_need, state.tpm_capacity)
            if state.rpm_tokens >= 1.0 and state.tpm_tokens >= need:
                conn.execute(
                    "UPDATE token_bucket SET rpm_tokens=?, tpm_tokens=?, last_update=? WHERE key=?",
                    (state.rpm_tokens - 1.0, state.tpm_tokens - need, now, self._key),
                )
                return True, 0.0
            wait_rpm = (
                (1.0 - state.rpm_tokens) / (state.rpm_capacity / 60.0)
                if state.rpm_tokens < 1.0
                else 0.0
            )
            wait_tpm = (
                (need - state.tpm_tokens) / (state.tpm_capacity / 60.0)
                if state.tpm_tokens < need
                else 0.0
            )
            conn.execute(
                "UPDATE token_bucket SET rpm_tokens=?, tpm_tokens=?, last_update=? WHERE key=?",
                (state.rpm_tokens, state.tpm_tokens, now, self._key),
            )
            wait_time = max(wait_rpm, wait_tpm)
            jitter = random.uniform(0.01, 0.05)
            return False, max(0.01, min(wait_time + jitter, 5.0))

    async def acquire(self, estimated_tokens: int = 500) -> None:
        """Acquire one RPM token and ``estimated_tokens`` TPM from the shared bucket.

        The lock intentionally spans the ``to_thread`` step: it serializes the
        read-modify-write of the token row, which is a sub-millisecond single
        statement. Waiting-for-refill sleeps happen OUTSIDE the lock, so a
        throttled caller never blocks the process's other acquirers; do not
        "optimize" the lock away to save one statement — concurrent updates
        would last-write-wins the refill accounting.
        """
        raw_need = float(max(1, estimated_tokens))
        while not self._closed:
            async with self.lock:
                if self._closed:
                    return
                acquired, sleep_duration = await asyncio.to_thread(self._acquire_step, raw_need)
            if acquired or self._closed:
                return
            await asyncio.sleep(sleep_duration)

    def report_429(self) -> None:
        """Multiplicative decrease on the shared bucket, once per episode.

        ``consecutive_429`` lives in the row precisely so the merge is
        cross-process: ten workers behind one credential see the same 429 burst,
        and it must cost one halving rather than ten.
        """
        with self._txn() as conn:
            state, last_update = self._load(conn)
            now = time.time()
            self._refill(state, now, last_update)
            state.consecutive_429 += 1
            if (now - state.last_backoff) < self.backoff_cooldown_sec:
                self._store(conn, state, now)
                return
            if self.backoff_cooldown_sec == 0.0 and state.consecutive_429 > 1:
                self._store(conn, state, now)
                return
            state.last_backoff = now
            state.rpm_capacity = max(float(self.min_rpm), state.rpm_capacity * 0.5)
            state.rpm_tokens = min(state.rpm_tokens, 0.0)
            state.tpm_capacity = max(float(self.min_tpm), state.tpm_capacity * 0.5)
            state.tpm_tokens = min(state.tpm_tokens, 0.0)
            if state.rpm_capacity <= self.min_rpm or state.tpm_capacity <= self.min_tpm:
                logger.warning(
                    "Shared rate limiter reached its floor after a 429 episode (%.0f rpm, "
                    "%.0f tpm in %s); the run stays clamped until successes start again",
                    state.rpm_capacity,
                    state.tpm_capacity,
                    self._path,
                )
            self._store(conn, state, now)

    def report_success(self) -> None:
        """Additive increase on the shared bucket."""
        with self._txn() as conn:
            state, last_update = self._load(conn)
            now = time.time()
            self._refill(state, now, last_update)
            if state.consecutive_429 > 0:
                state.consecutive_429 = 0
            # Same AIMD increment law as the in-process bucket: additive, at
            # most 1.0 RPM per window (1/capacity per success). A flat +1.0
            # let N workers reporting concurrently climb N times faster than
            # one worker, defeating the shared bucket's rate ceiling.
            increment = min(1.0, max(0.05, 1.0 / max(1.0, float(state.rpm_capacity))))
            state.rpm_capacity = min(float(self.max_rpm), state.rpm_capacity + increment)
            if state.tpm_capacity < self.max_tpm:
                state.tpm_capacity = min(
                    float(self.max_tpm), state.tpm_capacity + self.initial_tpm / 60.0
                )
            self._store(conn, state, now)

    async def report_429_async(self) -> None:
        """Offload synchronous SQLite transaction to thread pool."""
        await asyncio.to_thread(self.report_429)

    async def report_success_async(self) -> None:
        """Offload synchronous SQLite transaction to thread pool."""
        await asyncio.to_thread(self.report_success)

    def _store(self, conn: sqlite3.Connection, state: _BucketState, now: float) -> None:
        conn.execute(
            "UPDATE token_bucket SET rpm_capacity=?, rpm_tokens=?, tpm_capacity=?, "
            "tpm_tokens=?, last_update=?, consecutive_429=?, last_backoff=? WHERE key=?",
            (
                state.rpm_capacity,
                state.rpm_tokens,
                state.tpm_capacity,
                state.tpm_tokens,
                now,
                state.consecutive_429,
                state.last_backoff,
                self._key,
            ),
        )

    def close(self) -> None:
        """Release the SQLite handle (each instance owns its own connection)."""
        with self._thread_lock:
            self._closed = True
            with suppress(sqlite3.Error):
                self._conn.close()


def build_rate_limiter(
    config: Any, *, shared_path: str | Path | None = None
) -> AdaptiveTokenBucket:
    """Construct the right limiter for the process topology.

    A non-``None`` ``shared_path`` yields a cross-process bucket for the worker;
    otherwise an in-process bucket serves single-process CLI/API runs.
    """
    params: dict[str, Any] = {
        "initial_rpm": config.rate_limit_rpm,
        "initial_tpm": config.rate_limit_tpm,
        "max_rpm": config.rate_limit_max_rpm,
        # Without an explicit ceiling this defaulted to initial_tpm, so the AIMD
        # TPM side could never grow past the starting budget (see config field).
        "max_tpm": getattr(config, "rate_limit_max_tpm", None),
        "backoff_cooldown_sec": getattr(config, "rate_limit_backoff_cooldown_sec", 3.0),
    }
    if shared_path is not None:
        return SqliteTokenBucket(shared_path, **params)
    return AdaptiveTokenBucket(**params)
