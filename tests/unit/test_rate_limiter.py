"""Unit tests for AdaptiveTokenBucket AIMD rate limiter."""

import asyncio
import threading
import time
from pathlib import Path

import pytest

from ubt.core.router.provider import MockModelProvider
from ubt.core.router.rate_limiter import AdaptiveTokenBucket, SqliteTokenBucket


@pytest.mark.asyncio
async def test_sqlite_bucket_shares_budget_across_instances(tmp_path: Path) -> None:
    """Two handles on one file split a single RPM budget (the worker-process case)."""
    path = tmp_path / "rl.sqlite"
    b1 = SqliteTokenBucket(path, initial_rpm=3, initial_tpm=1_000_000, min_rpm=1)
    b2 = SqliteTokenBucket(path, initial_rpm=3, initial_tpm=1_000_000, min_rpm=1)
    try:
        # Drain the shared bucket through b1: 3 acquires fit, the 4th must wait.
        for _ in range(3):
            await b1.acquire(estimated_tokens=1)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(b1.acquire(estimated_tokens=1), timeout=0.1)
        # A *separate* handle proves the state is shared, not per-instance:
        # b2 sees b1's drained bucket and also blocks.
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(b2.acquire(estimated_tokens=1), timeout=0.1)
    finally:
        b1.close()
        b2.close()


@pytest.mark.asyncio
async def test_sqlite_bucket_isolated_across_files(tmp_path: Path) -> None:
    """Different files are different credentials and must not share state."""
    a = SqliteTokenBucket(tmp_path / "a.sqlite", initial_rpm=2, initial_tpm=1_000_000, min_rpm=1)
    b = SqliteTokenBucket(tmp_path / "b.sqlite", initial_rpm=2, initial_tpm=1_000_000, min_rpm=1)
    try:
        await a.acquire(estimated_tokens=1)
        await a.acquire(estimated_tokens=1)
        # a is drained...
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(a.acquire(estimated_tokens=1), timeout=0.1)
        # ...but b still has its own full budget.
        await asyncio.wait_for(b.acquire(estimated_tokens=1), timeout=0.1)
    finally:
        a.close()
        b.close()


@pytest.mark.asyncio
async def test_sqlite_bucket_aimd_is_shared(tmp_path: Path) -> None:
    """A 429 seen by one handle halves the capacity the other handle observes."""
    path = tmp_path / "rl.sqlite"
    b1 = SqliteTokenBucket(path, initial_rpm=100, initial_tpm=1_000_000, min_rpm=1)
    b2 = SqliteTokenBucket(path, initial_rpm=100, initial_tpm=1_000_000, min_rpm=1)
    try:
        b1.report_429()
        row = b2._conn.execute("SELECT rpm_capacity FROM token_bucket").fetchone()
        assert row["rpm_capacity"] == pytest.approx(50.0)
    finally:
        b1.close()
        b2.close()


def test_build_rate_limiter_selects_backend(tmp_path: Path) -> None:
    from ubt.core.router.rate_limiter import build_rate_limiter

    class _Cfg:
        rate_limit_rpm = 60
        rate_limit_tpm = 100_000
        rate_limit_max_rpm = 240

    in_proc = build_rate_limiter(_Cfg())
    shared = build_rate_limiter(_Cfg(), shared_path=tmp_path / "rl.sqlite")
    assert type(in_proc) is AdaptiveTokenBucket
    assert isinstance(shared, SqliteTokenBucket)
    shared.close()


@pytest.mark.asyncio
async def test_token_bucket_initialization_and_acquire() -> None:
    bucket = AdaptiveTokenBucket(initial_rpm=600)  # 10 tokens/sec
    assert bucket.capacity == 600.0
    assert bucket.tokens == 600.0

    start = time.monotonic()
    await bucket.acquire(estimated_tokens=100)
    duration = time.monotonic() - start
    # Should acquire immediately since bucket starts full
    assert duration < 0.1
    assert bucket.tokens < 600.0


@pytest.mark.asyncio
async def test_token_bucket_aimd_multiplicative_decrease_on_429() -> None:
    bucket = AdaptiveTokenBucket(initial_rpm=60, min_rpm=10)
    initial_cap = bucket.capacity

    # Trigger 429
    bucket.report_429()
    assert bucket.consecutive_429 == 1
    assert bucket.capacity == initial_cap * 0.5
    assert bucket.tokens <= 0.0

    # A second 429 belongs to the same episode: it counts, but the reaction was
    # already taken (halving twice for one burst is what collapsed runs).
    bucket.report_429()
    assert bucket.consecutive_429 == 2
    assert bucket.capacity == initial_cap * 0.5


@pytest.mark.asyncio
async def test_token_bucket_aimd_additive_increase_on_success() -> None:
    bucket = AdaptiveTokenBucket(initial_rpm=20, max_rpm=30)
    bucket.report_429()
    reduced_cap = bucket.capacity

    bucket.report_success()
    assert bucket.consecutive_429 == 0
    assert bucket.capacity == reduced_cap + 1.0


@pytest.mark.asyncio
async def test_token_bucket_backoff_cooldown_prevents_oscillation() -> None:
    bucket = AdaptiveTokenBucket(initial_rpm=100, min_rpm=10, backoff_cooldown_sec=2.0)
    initial_cap = bucket.capacity

    # First 429 episode halves capacity to 50
    bucket.report_429()
    assert bucket.capacity == initial_cap * 0.5

    # An interleaved success happens immediately (+1 RPM)
    bucket.report_success()
    expected_cap = (initial_cap * 0.5) + 1.0
    assert bucket.capacity == expected_cap

    # Another 429 arrives within 2s cooldown window:
    # Must NOT halve again to 25.5!
    bucket.report_429()
    assert bucket.capacity == expected_cap


@pytest.mark.asyncio
async def test_token_bucket_throttling_when_empty() -> None:
    # 2 tokens/sec, capacity 1
    bucket = AdaptiveTokenBucket(initial_rpm=120)
    bucket.tokens = 0.0  # Force empty

    start = time.monotonic()
    await bucket.acquire(estimated_tokens=50)
    elapsed = time.monotonic() - start

    # Should have waited for token replenishment (~0.5s)
    assert elapsed >= 0.2


# ---------------------------------------------------------------------------
# Dual-bucket (RPM + TPM) enforcement and AIMD on both axes.
# ---------------------------------------------------------------------------


@pytest.mark.slow  # real-time throttle waits (~24s)
@pytest.mark.asyncio
async def test_tpm_bucket_throttles_token_heavy_requests() -> None:
    """A token-heavy request beyond the TPM refill rate must wait, even with
    plenty of RPM tokens left (the old limiter ignored estimated_tokens)."""
    bucket = AdaptiveTokenBucket(initial_rpm=6000, initial_tpm=1_000)
    bucket.tpm_tokens = 0.0  # drain TPM; refill is 1000/60 ≈ 16.7 tokens/s
    start = time.monotonic()
    await bucket.acquire(estimated_tokens=400)  # needs ~24s of refill
    elapsed = time.monotonic() - start
    assert elapsed >= 1.0, "TPM bucket did not throttle a token-heavy request"
    assert bucket.tpm_tokens < 1_000.0  # tokens were actually consumed


@pytest.mark.asyncio
async def test_tpm_bucket_consumes_estimated_tokens() -> None:
    bucket = AdaptiveTokenBucket(initial_rpm=60, initial_tpm=100_000)
    before = bucket.tpm_tokens
    await bucket.acquire(estimated_tokens=2_500)
    # Elapsed refill is negligible on a fast call; consumption dominates.
    assert bucket.tpm_tokens <= before - 2_000


@pytest.mark.asyncio
async def test_oversized_request_clamped_never_deadlocks() -> None:
    """estimated_tokens above total TPM capacity is clamped instead of
    waiting forever for an unreachable threshold."""
    bucket = AdaptiveTokenBucket(initial_rpm=6000, initial_tpm=1_000)
    start = time.monotonic()
    await bucket.acquire(estimated_tokens=999_999)
    assert time.monotonic() - start < 2.0


@pytest.mark.asyncio
async def test_aimd_applies_to_both_buckets() -> None:
    bucket = AdaptiveTokenBucket(initial_rpm=60, initial_tpm=100_000)
    bucket.report_429()
    assert bucket.tpm_capacity == 50_000.0
    assert bucket.tpm_tokens <= 0.0
    # Additive increase recovers TPM capacity but never above the ceiling.
    for _ in range(5):
        bucket.report_success()
    assert bucket.tpm_capacity <= 100_000.0
    assert bucket.tpm_capacity > 50_000.0


def test_max_rpm_respected_in_additive_increase() -> None:
    """The additive-increase ceiling is configurable (was hardcoded)."""
    bucket = AdaptiveTokenBucket(initial_rpm=30, max_rpm=35)
    for _ in range(20):
        bucket.report_success()
    assert bucket.capacity == 35.0


def test_429_burst_collapses_capacity_once_per_episode() -> None:
    """One TPM-limit hit arrives as ``max_concurrency`` simultaneous 429s.

    Halving per report turned that single event into 60 rpm -> 5 rpm (the floor)
    and ~25k tpm, and the additive climb of +1 rpm per success needed 55
    successes at 5 rpm to get back — over ten minutes of crawling for an
    ordinary limit (2026-09 review).
    """
    bucket = AdaptiveTokenBucket(initial_rpm=60, initial_tpm=100_000, min_rpm=5)

    for _ in range(10):
        bucket.report_429()

    assert bucket.capacity == pytest.approx(30.0)
    assert bucket.tpm_capacity == pytest.approx(50_000.0)

    # A success closes the episode, so a later 429 is real new congestion.
    bucket.report_success()
    bucket.report_429()
    assert bucket.capacity == pytest.approx(15.5)


@pytest.mark.asyncio
async def test_shared_bucket_merges_a_429_burst_across_processes(tmp_path: Path) -> None:
    """The merge must be cross-process, since the burst is across processes."""
    path = tmp_path / "rl.sqlite"
    b1 = SqliteTokenBucket(path, initial_rpm=60, initial_tpm=1_000_000, min_rpm=5)
    b2 = SqliteTokenBucket(path, initial_rpm=60, initial_tpm=1_000_000, min_rpm=5)
    try:
        for handle in (b1, b2) * 5:
            handle.report_429()
        row = b1._conn.execute("SELECT rpm_capacity FROM token_bucket").fetchone()
        assert row["rpm_capacity"] == pytest.approx(30.0)
    finally:
        b1.close()
        b2.close()


def test_raising_the_configured_rate_starts_a_clean_shared_bucket(
    tmp_path: Path,
) -> None:
    """An existing row used to outrank this process's settings indefinitely."""
    path = tmp_path / "rl.sqlite"
    small = SqliteTokenBucket(path, initial_rpm=60, initial_tpm=100_000)
    small.report_429()
    small.close()

    raised = SqliteTokenBucket(path, initial_rpm=240, initial_tpm=100_000)
    try:
        row = raised._conn.execute(
            "SELECT rpm_capacity FROM token_bucket WHERE key = ?", (raised._key,)
        ).fetchone()
        assert row["rpm_capacity"] == pytest.approx(240.0)
    finally:
        raised.close()


@pytest.mark.asyncio
async def test_sqlite_acquire_runs_its_transaction_off_the_event_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§10.1: BEGIN IMMEDIATE can wait the whole busy_timeout; not on the loop.

    The shared bucket used to run its read-modify-write inline in the
    coroutine, so a contended write-lock froze every other coroutine on the
    event loop — the same blocking->lease-stall class of failure it exists to
    prevent. Pin the fix: while acquire's BEGIN is blocked in a worker thread,
    an unrelated coroutine must keep ticking.
    """
    bucket = SqliteTokenBucket(tmp_path / "rl.sqlite", initial_rpm=100, initial_tpm=1_000_000)
    real_load = bucket._load
    started = threading.Event()
    release = threading.Event()

    def blocked_load(conn):  # type: ignore[no-untyped-def]
        # Hold the transaction open (as a contended write lock would) until the
        # test releases it — the point is to observe the loop *while* it blocks.
        started.set()
        release.wait(5.0)
        return real_load(conn)

    monkeypatch.setattr(bucket, "_load", blocked_load)
    ticks = 0

    async def heartbeat() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    try:
        hb = asyncio.create_task(heartbeat())
        acquire = asyncio.create_task(bucket.acquire(estimated_tokens=1))
        await asyncio.to_thread(started.wait, 5.0)  # txn is now blocked off-loop
        before = ticks
        await asyncio.sleep(0.1)
        assert ticks > before, "event loop froze while the shared transaction ran"
        release.set()
        await asyncio.wait_for(acquire, timeout=5.0)
    finally:
        release.set()
        hb.cancel()
        bucket.close()


@pytest.mark.asyncio
async def test_mock_provider_gets_a_limiter_that_never_gates() -> None:
    """An offline drill must not queue behind a budget it cannot exhaust.

    The full-book matrix run spent 20+ minutes at ~41 calls/min because the
    default 60 rpm / 100k tpm shape was charged call for call against a mock
    provider. The refill maths is what makes this assertable: a real bucket
    cannot serve 200 x 100k tokens in under a second, this one must.
    """
    from ubt.core.router.rate_limiter import NullRateLimiter
    from ubt.core.router.router import ModelRouter

    router = ModelRouter(provider=MockModelProvider(), draft_model="mock")
    assert isinstance(router.rate_limiter, NullRateLimiter)
    await asyncio.wait_for(
        asyncio.gather(*(router.rate_limiter.acquire(100_000) for _ in range(200))),
        timeout=1.0,
    )
    # AIMD feedback is a no-op too: a stray 429 must not clamp a drill.
    router.rate_limiter.report_429()
    assert router.rate_limiter.capacity == 60.0


def test_explicit_limiter_wins_even_for_mock() -> None:
    """Callers that hand in a bucket mean it — retry/AIMD tests pin one."""
    from ubt.core.router.rate_limiter import NullRateLimiter
    from ubt.core.router.router import ModelRouter

    limiter = AdaptiveTokenBucket(initial_rpm=60, initial_tpm=1_000)
    router = ModelRouter(provider=MockModelProvider(), rate_limiter=limiter)
    assert router.rate_limiter is limiter
    assert not isinstance(router.rate_limiter, NullRateLimiter)


@pytest.mark.fast
def test_sqlite_token_bucket_reseed_when_row_deleted(tmp_path: Path) -> None:
    db_file = tmp_path / "bucket.sqlite"
    bucket = SqliteTokenBucket(db_file, initial_rpm=60, initial_tpm=1000)

    # Verify initial load works
    with bucket._txn() as conn:
        state, last_update = bucket._load(conn)
        assert state.rpm_capacity == 60

    # Delete the row to simulate external reset / row vanishing
    with bucket._txn() as conn:
        conn.execute("DELETE FROM token_bucket WHERE key = ?", (bucket._key,))

    # Should reseed lazily instead of raising RuntimeError
    with bucket._txn() as conn:
        state, last_update = bucket._load(conn)
        assert state.rpm_capacity == 60
        assert state.rpm_tokens == 60


@pytest.mark.fast
def test_sqlite_token_bucket_preserves_negative_debt_on_429(tmp_path: Path) -> None:
    db_path = tmp_path / "rate_limiter.sqlite"
    bucket = SqliteTokenBucket(
        path=db_path,
        initial_rpm=60,
        initial_tpm=100000,
        min_rpm=10,
        min_tpm=1000,
    )
    # Put bucket into debt
    with bucket._txn() as conn:
        state, last_update = bucket._load(conn)
        state.rpm_tokens = -15.0
        state.tpm_tokens = -5000.0
        bucket._store(conn, state, last_update)

    bucket.report_429()

    with bucket._txn() as conn:
        state, _ = bucket._load(conn)
        assert state.rpm_tokens <= -10.0 or state.rpm_tokens < 0.0, (
            f"Expected rpm_tokens to retain negative debt, got {state.rpm_tokens}"
        )
        assert state.tpm_tokens <= -2000.0 or state.tpm_tokens < 0.0, (
            f"Expected tpm_tokens to retain negative debt, got {state.tpm_tokens}"
        )
