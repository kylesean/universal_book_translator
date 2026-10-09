"""Translation Memory: exact-hit skip + fuzzy few-shot.

Accepted translations for identical sources skip the LLM entirely,
and near-identical sources (>= threshold, same polarity) are injected as
terminology-only few-shot references so the model reuses consistent renderings
instead of re-translating from scratch. A fuzzy neighbour that diverges on a
polarity/antonym cue is dropped rather than shown (see ``_polarity_diverges``).

Storage lives in a dedicated shared database (``{db_dir}/tm.sqlite``) rather
than the per-job ledger, because the memory's value comes from reuse across
books and jobs. All methods are synchronous and thread-safe; async callers
must invoke them through ``asyncio.to_thread`` (same discipline as the job
ledger).

Fuzzy matching scans the in-memory normalized source pool for the language
pair with ``rapidfuzz``. At book scale (thousands of segments per pair) a
C-level scan is faster than any index round-trip -- see the measurement on
``FTS_PREFILTER_MIN_POOL`` -- so pools below that threshold keep the direct
scan. Above it, an FTS5 trigram index prefilters to ``FTS_PREFILTER_LIMIT``
candidates sharing tokens with the query, and ``rapidfuzz`` reranks only those
with the identical scorer/cutoff.

Why trigram (not unicode61): trigram containment matches compounds, spacing
variants (``riverbank`` vs ``river bank``), intra-word typos, and CJK runs
uniformly, and folds ASCII case — all verified against the bundled SQLite.
The prefilter can only cause a miss (one extra LLM call), never a wrong hit.
"""

import hashlib
import logging
import random
import re
import sqlite3
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz, process

from ubt.core.fs_perms import ensure_private_dir, restrict_sqlite_family

logger = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_TOKEN_RE = re.compile(r"[a-z]+")
# Typographic apostrophes (and the acute/backtick stand-ins) must fold to ASCII
# before the ``n't`` contraction check: real books use U+2019, and
# ``"n't" in "isn’t"`` is false, so the polarity guard missed every curly-quoted
# negation and injected an opposite-polarity fuzzy reference.
_APOSTROPHE_RE = re.compile(r"[\u2018\u2019\u02bc\u0060\u00b4]")

# ---------------------------------------------------------------------------
# Fuzzy few-shot polarity guard
# ---------------------------------------------------------------------------
# ``fuzz.ratio`` is a whole-string Indel ratio, so a one-morpheme semantic
# inversion ("... exothermic." vs "... endothermic.") still clears the 0.85
# fuzzy threshold; swapping the scorer does not fix it (``token_ratio`` scores
# the "This is the case."/"This is not the case." pair 1.000, i.e. worse). The
# guard below therefore drops the reference whenever query and candidate
# diverge on a closed set of polarity cues, and the surviving reference is
# presented as terminology-only.
#
# Conservative by design: a dropped reference costs one LLM draft, while a
# wrong reference is the most direct route for a wrong sentence into a book.
_NEGATION_CUES = frozenset(
    {"not", "no", "never", "without", "none", "neither", "nor", "nothing", "nobody", "cannot"}
)
# CJK negation travels as characters, not [a-z]+ tokens, so the cue set above
# never fired for zh/ja sources and every fuzzy reference passed the guard.
# Unambiguous negators only: 未/非 are omitted on purpose — they are substrings
# of 未来/非常 — and a false cue costs one dropped reference (a re-draft), never
# a wrong sentence.
_CJK_NEGATION_CUES = frozenset({"不", "没", "没有", "别", "无", "勿", "ない", "ず", "ぬ"})
# Mutually exclusive antonym cue groups: both sides carrying the SAME group
# member agrees in polarity; a missing or different member is a divergence.
_ANTONYM_CUE_GROUPS: tuple[tuple[str, ...], ...] = (
    ("increase", "decrease"),
    ("left", "right"),
    ("exothermic", "endothermic"),
    ("more", "less"),
    ("above", "below"),
)
_ANTONYM_CUE_RES = tuple(
    re.compile(r"\b(?:" + "|".join(group) + r")(?:s|es|ed|d|ing)?\b", re.IGNORECASE)
    for group in _ANTONYM_CUE_GROUPS
)


def _polarity_signature(text: str) -> tuple[str, ...]:
    """One slot per cue group: the cue the text carries, or ``""`` when absent.

    Slot 0 is negation (any cue, plus the ``n't`` contraction tail that
    tokenization splits off); every antonym group gets its own slot holding the
    group's canonical cue, so inflections ("increases"/"increased") agree.
    Word-level membership keeps "no" out of "north" and "not" out of "nothing".
    """
    lowered = _APOSTROPHE_RE.sub("'", text.lower())
    tokens = set(_TOKEN_RE.findall(lowered))
    negated = (
        bool(tokens & _NEGATION_CUES)
        or "n't" in lowered
        or any(cue in lowered for cue in _CJK_NEGATION_CUES)
    )
    slots = ["neg" if negated else ""]
    for group, regex in zip(_ANTONYM_CUE_GROUPS, _ANTONYM_CUE_RES, strict=True):
        match = regex.search(text)
        # Canonicalise the inflected surface ("increased" -> "increase") so a
        # tense change does not read as an antonym divergence.
        surface = match.group(0).lower() if match else ""
        slots.append(next((cue for cue in group if surface.startswith(cue)), ""))
    return tuple(slots)


def _polarity_diverges(query_source: str, reference_source: str) -> bool:
    """True when the two sources disagree on any polarity/antonym cue."""
    return _polarity_signature(query_source) != _polarity_signature(reference_source)


# Scale-up knobs for the FTS5 prefilter path.
#
# The rapidfuzz scan is linear in the pool, while the trigram prefilter's price
# is dominated by its fixed 64-term OR MATCH (it does not shrink with the pool),
# so the scan is the cheaper path until the pool is genuinely large. A too-low
# threshold picks a path slower than the one it replaces on every block; the
# previous 20_000 contradicted this comment's own crossover claim. Recalibrate
# with scripts/knob_sweep.py rather than trusting a hand-written number.
FTS_PREFILTER_MIN_POOL = 1_000_000
FTS_PREFILTER_LIMIT = 200
_FTS_QUERY_MAX_TERMS = 64
_FTS_SCHEMA_VERSION = 1

# Prompt topology version baked into the TM exact-hit identity. Bump
# whenever the draft prompt layout, glossary injection format, or MT tier
# changes meaningfully — stale translations must never silently skip the LLM.
PROMPT_VERSION = "v1"


def compute_tm_context(
    prompt_version: str,
    profile_name: str,
    glossary_table: str,
    src_lang: str,
    tgt_lang: str,
    abbreviation_table: str = "",
    model: str = "",
) -> str:
    """Fingerprint the translation context an exact TM hit is valid under.

    ``model`` is the draft-tier model that produced the stored draft. It is an
    identity axis on purpose: without it, a rerun under a different draft model
    would short-circuit on a same-content hit from another model and never call
    the new one. The content-addressed draft cache already keys on ``model``;
    this is the same axis, deliberately kept in lockstep.

    ``glossary_table`` and ``abbreviation_table`` are the two prompt-visible
    terminology channels (the rendered global term table and abbreviation
    table); both must be hashed, else a prompt-visible change is invisible to
    the identity and a stale exact hit survives it.
    """
    h = hashlib.sha256()
    for part in (
        prompt_version,
        model,
        profile_name,
        glossary_table,
        src_lang,
        tgt_lang,
        abbreviation_table,
    ):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:32]


# provenance values
PROVENANCE_MACHINE = "machine"
PROVENANCE_HUMAN_PE = "human_pe"


def normalize_for_tm(text: str) -> str:
    """Whitespace-collapsed, stripped normalization used as the TM exact key."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def _source_hash(normalized_source: str) -> str:
    return hashlib.sha256(normalized_source.encode("utf-8")).hexdigest()


def _read_one(conn: sqlite3.Connection, sql: str, params: Sequence[Any] = ()) -> Any:
    """Read a single row and release the statement.

    A cursor left half-stepped (the usual ``fetchone()`` on a one-row result)
    keeps its WAL read snapshot open, so every later read on that connection
    answers from a database state that predates another process's commit. The
    shared tm.sqlite has several readers at once, so aggregation queries have to
    close their cursor before the next one starts.
    """
    cursor = conn.execute(sql, params)
    try:
        return cursor.fetchone()
    finally:
        cursor.close()


def _fts_query_terms(normalized: str) -> list[str]:
    """Split normalized text into FTS5 trigram OR-terms (3-codepoint shingles).

    Whole runs would demand every trigram to match (one variant word kills the
    term); shingles degrade gracefully — a variant only kills its neighboring
    windows while the rest still match and ``ORDER BY rank`` floats the true
    near-duplicate to the top. Runs under 3 codepoints yield no trigram and
    are skipped; near-identical matches on such fragments are covered by the
    exact path.
    """
    terms: list[str] = []
    seen: set[str] = set()
    for run in _WORD_RE.findall(normalized):
        for i in range(len(run) - 2):
            shingle = run[i : i + 3]
            if shingle not in seen:
                seen.add(shingle)
                terms.append(shingle)
            if len(terms) >= _FTS_QUERY_MAX_TERMS:
                return terms
    return terms


def _fts_phrase(term: str) -> str:
    """Quote one term as an FTS5 phrase, doubling embedded quotes."""
    return '"' + term.replace('"', '""') + '"'


@dataclass(frozen=True)
class TMHit:
    """A translation memory hit (exact similarity=1.0, fuzzy below)."""

    source_text: str
    target_text: str
    similarity: float
    provenance: str
    #: Serialized target-side emphasis runs (see ``ubt.core.ir.emphasis``); empty
    #: when the stored translation carried no preserved emphasis.
    runs_json: str = ""


@dataclass(frozen=True)
class TMPendingEntry:
    """One source/target pair queued for writeback."""

    src_lang: str
    tgt_lang: str
    source_text: str
    target_text: str
    provenance: str = PROVENANCE_MACHINE
    domain: str | None = None
    context_hash: str = ""  # must match the lookup context to ever hit
    runs_json: str = ""


@dataclass(frozen=True)
class TMStoredEntry:
    """One persisted TM row, for auditing and correction."""

    id: int
    src_lang: str
    tgt_lang: str
    source_text: str
    target_text: str
    provenance: str
    domain: str | None
    use_count: int


def format_few_shot_reference(hit: TMHit, query_source: str) -> str:
    """Render a fuzzy TM hit as a prompt few-shot reference block.

    Injected into the dynamic tail of draft prompts (never
    into the static prefix). Returns ``""`` when ``query_source`` and the
    hit's source diverge on a polarity/antonym cue (:func:`_polarity_diverges`)
    — a fuzzy neighbour that negates or antonymizes the sentence under
    translation must never be shown as an example.

    The surviving reference is presented as **terminology only**. Whole-string
    similarity cannot see meaning, so the block states explicitly that it is a
    different sentence which must not be copied or used to infer content.
    """
    if _polarity_diverges(query_source, hit.source_text):
        return ""
    similarity_pct = round(hit.similarity * 100)
    return (
        "### Reference Translation (fuzzy match from translation memory, "
        f"{similarity_pct}% similar — terminology reference ONLY)\n"
        "The reference below is a DIFFERENT sentence, kept only as evidence of "
        "how shared terms were rendered. It may differ in meaning, polarity, or "
        "figures from the sentence you are translating: do NOT copy its wording "
        "verbatim and do NOT infer the sentence's content from it.\n"
        f"Reference source: {hit.source_text}\n"
        f"Reference translation: {hit.target_text}"
    )


class TranslationMemory:
    """SQLite-backed sentence-level translation memory shared across jobs."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        prefilter_min_pool: int = FTS_PREFILTER_MIN_POOL,
        prefilter_limit: int = FTS_PREFILTER_LIMIT,
        busy_timeout_sec: float = 30.0,
    ) -> None:
        self.db_path = Path(db_path)
        self._busy_timeout_sec = float(busy_timeout_sec)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        # (src_lang, tgt_lang) -> (normalized_sources, ids); lazily populated.
        self._pool_cache: dict[tuple[str, str], tuple[list[str], list[int]]] = {}
        self._entry_count_cache: dict[tuple[str, str] | None, int] = {}
        # The pair generation each cached per-pair view was read at. One dict per
        # cache: the pool and the count are usually filled on different calls,
        # and sharing a generation between them lets the fresher view mark the
        # staler one valid.
        self._pool_gen: dict[tuple[str, str], int] = {}
        self._count_gen: dict[tuple[str, str], int] = {}
        self._last_data_version: int | None = None
        self._prefilter_min_pool = prefilter_min_pool
        self._prefilter_limit = prefilter_limit
        self._fts_ok = False
        self._init_db()
        # tm.sqlite accumulates the bilingual sentence pool of every book this
        # machine has touched, and SQLite creates it 0666&~umask -- world
        # readable on a shared host. The parent is the operator's ``db_dir``:
        # only tighten it when UBT creates it, so an existing shared directory
        # is not silently rewritten (matches the ledger; see ensure_private_dir).
        ensure_private_dir(self.db_path.parent)
        restrict_sqlite_family(self.db_path)

    # ------------------------------------------------------------------
    # Connection & schema
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            conn = sqlite3.connect(
                self.db_path, check_same_thread=False, timeout=self._busy_timeout_sec
            )
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(f"PRAGMA busy_timeout={int(self._busy_timeout_sec * 1000)};")
            conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn = conn
        return self._conn

    def _init_db(self) -> None:
        with self._lock, self._get_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tm_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    src_lang TEXT NOT NULL,
                    tgt_lang TEXT NOT NULL,
                    src_hash TEXT NOT NULL,
                    src_text TEXT NOT NULL,
                    tgt_text TEXT NOT NULL,
                    provenance TEXT NOT NULL DEFAULT 'machine',
                    domain TEXT,
                    use_count INTEGER NOT NULL DEFAULT 0,
                    context_hash TEXT NOT NULL DEFAULT '',
                    runs_json TEXT NOT NULL DEFAULT '',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS tm_exact "
                "ON tm_entries(src_lang, tgt_lang, src_hash, context_hash)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS tm_pair ON tm_entries(src_lang, tgt_lang)")
            # Per-language-pair write counter, so a write to one pair does not
            # invalidate every other pair's cached pool (see _pool_gen).
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tm_generation (
                    src_lang TEXT NOT NULL,
                    tgt_lang TEXT NOT NULL,
                    gen INTEGER NOT NULL,
                    PRIMARY KEY (src_lang, tgt_lang)
                )
                """
            )
            # Context-gated exact identity. Gated on column existence so a
            # pre-existing database gains the column idempotently; rows without a
            # context keep context_hash='' and match default lookups or the
            # human_pe fallback below.
            cols = {row[1] for row in conn.execute("PRAGMA table_info(tm_entries)").fetchall()}
            if "context_hash" not in cols:
                try:
                    conn.execute(
                        "ALTER TABLE tm_entries ADD COLUMN context_hash TEXT NOT NULL DEFAULT ''"
                    )
                except sqlite3.OperationalError as exc:
                    # Two processes can open the same legacy tm.sqlite at once
                    # (shared WAL store); the loser sees the column the winner
                    # just added. Only that race is tolerated.
                    if "duplicate column" not in str(exc).lower():
                        raise
                conn.execute("DROP INDEX IF EXISTS tm_exact")
                conn.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS tm_exact "
                    "ON tm_entries(src_lang, tgt_lang, src_hash, context_hash)"
                )
            # Target-side emphasis runs, so an exact hit can restore the bold the
            # marker mechanism preserved instead of serving a plain translation.
            # Gated on column existence so a pre-existing database gains it idempotently.
            if "runs_json" not in cols:
                try:
                    conn.execute(
                        "ALTER TABLE tm_entries ADD COLUMN runs_json TEXT NOT NULL DEFAULT ''"
                    )
                except sqlite3.OperationalError as exc:
                    if "duplicate column" not in str(exc).lower():
                        raise
            try:
                conn.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS tm_fts USING fts5("
                    "src_text, src_lang, tgt_lang, "
                    "content='tm_entries', content_rowid='id', "
                    "tokenize='trigram')"
                )
                conn.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS tm_fts_ai AFTER INSERT ON tm_entries BEGIN
                        INSERT INTO tm_fts(rowid, src_text, src_lang, tgt_lang)
                        VALUES (new.id, new.src_text, new.src_lang, new.tgt_lang);
                    END
                    """
                )
                conn.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS tm_fts_ad AFTER DELETE ON tm_entries BEGIN
                        INSERT INTO tm_fts(tm_fts, rowid, src_text, src_lang, tgt_lang)
                        VALUES ('delete', old.id, old.src_text, old.src_lang, old.tgt_lang);
                    END
                    """
                )
                conn.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS tm_fts_au AFTER UPDATE ON tm_entries BEGIN
                        INSERT INTO tm_fts(tm_fts, rowid, src_text, src_lang, tgt_lang)
                        VALUES ('delete', old.id, old.src_text, old.src_lang, old.tgt_lang);
                        INSERT INTO tm_fts(rowid, src_text, src_lang, tgt_lang)
                        VALUES (new.id, new.src_text, new.src_lang, new.tgt_lang);
                    END
                    """
                )
                # One-time backfill for databases created before the FTS index.
                # (COUNT(*) on an external-content table reads the content
                # table, so it cannot detect an empty index — gate on a schema
                # version marker instead. user_version is unused on tm.sqlite.)
                version = conn.execute("PRAGMA user_version").fetchone()[0]
                if version < _FTS_SCHEMA_VERSION:
                    conn.execute("INSERT INTO tm_fts(tm_fts) VALUES('rebuild')")
                    conn.execute(f"PRAGMA user_version = {_FTS_SCHEMA_VERSION}")
                self._fts_ok = True
            except sqlite3.OperationalError as exc:
                # SQLite builds without FTS5: degrade to full-pool scans.
                logger.warning("TM FTS5 index unavailable, using full scans: %s", exc)
                self._fts_ok = False
            self._sync_data_version(conn)

    def _sync_data_version(self, conn: sqlite3.Connection) -> None:
        """Refresh the one cache that cannot be versioned per pair: the all-pairs count.

        ``PRAGMA data_version`` counts writes to the whole database file, and the
        shared tm.sqlite takes a write on every TM *hit* (the use_count bump), so
        using it to drop the per-pair pools rescanned and re-normalized every
        cached language pair on nearly every lookup. Pairs
        carry their own generation in ``tm_generation`` now.
        """
        try:
            row = _read_one(conn, "PRAGMA data_version;")
            if row is not None:
                current_version = int(row[0])
                if (
                    self._last_data_version is not None
                    and current_version != self._last_data_version
                ):
                    self._entry_count_cache.pop(None, None)
                self._last_data_version = current_version
        except sqlite3.Error:
            pass

    @staticmethod
    def _bump_generations(conn: sqlite3.Connection, pairs: set[tuple[str, str]]) -> None:
        """Record that these pairs' entry sets changed, in the writer's transaction.

        Counts set changes, not content changes: an upsert that only rewrites
        ``tgt_text`` leaves the source pool (and so the cached pool) valid, but
        reloading it occasionally is cheaper than proving which write mattered.
        """
        conn.executemany(
            "INSERT INTO tm_generation (src_lang, tgt_lang, gen) VALUES (?, ?, 1) "
            "ON CONFLICT(src_lang, tgt_lang) DO UPDATE SET gen = gen + 1",
            sorted(pairs),
        )

    @staticmethod
    def _generation(conn: sqlite3.Connection, pair: tuple[str, str]) -> int:
        """A pair's write counter, or 0 while nobody has written to it."""
        row = _read_one(
            conn,
            "SELECT gen FROM tm_generation WHERE src_lang = ? AND tgt_lang = ?",
            pair,
        )
        return int(row[0]) if row is not None else 0

    def close(self) -> None:
        """Close the underlying connection."""
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                finally:
                    self._conn = None

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def lookup_exact(
        self,
        src_lang: str,
        tgt_lang: str,
        source_text: str,
        context_hash: str = "",
        domain: str | None = None,
    ) -> TMHit | None:
        """Return the stored translation for a byte-identical (normalized) source.

        The hit must carry the caller's ``context_hash``, have ``human_pe``
        provenance, or share the caller's ``domain``. Hits increment the
        entry's ``use_count``.
        """
        normalized = normalize_for_tm(source_text)
        if not normalized:
            return None
        key = _source_hash(normalized)
        with self._lock, self._get_conn() as conn:
            cursor = conn.execute(
                """
                SELECT id, src_text, tgt_text, provenance, context_hash, domain, runs_json
                FROM tm_entries
                WHERE src_lang = ? AND tgt_lang = ? AND src_hash = ?
                """,
                (src_lang, tgt_lang, key),
            )
            rows = cursor.fetchall()
            if not rows:
                return None

            valid_rows = [
                row
                for row in rows
                if not self._row_rejected(str(row[3]), str(row[1]), str(row[2]))
                and self._hit_allowed(
                    str(row[3]),
                    str(row[4] or ""),
                    row[5],
                    context_hash,
                    domain,
                )
            ]
            if not valid_rows:
                return None

            def _candidate_rank(r: tuple[Any, ...]) -> tuple[int, int, int, int]:
                # human_pe outranks context. A PE import carries no context
                # hash, so ranking context first would let an exact-context
                # machine row outrank the reviewed rendering, contradicting the
                # writeback contract that human review is the highest-trust
                # signal in the pipeline.
                r_id = int(r[0])
                prov = str(r[3])
                r_ctx = str(r[4] or "")
                r_domain = r[5]
                is_human = 1 if prov == PROVENANCE_HUMAN_PE else 0
                exact_ctx = (
                    1
                    if (context_hash and r_ctx == context_hash) or (not context_hash and not r_ctx)
                    else 0
                )
                same_domain = 1 if domain and r_domain == domain else 0
                return (is_human, exact_ctx, same_domain, r_id)

            best_row = max(valid_rows, key=_candidate_rank)
            hit_id = int(best_row[0])
            self._bump_use_count(conn, hit_id)

        return TMHit(
            source_text=str(best_row[1]),
            target_text=str(best_row[2]),
            similarity=1.0,
            provenance=str(best_row[3]),
            runs_json=str(best_row[6] or ""),
        )

    def lookup_fuzzy(
        self,
        src_lang: str,
        tgt_lang: str,
        source_text: str,
        threshold: float = 0.85,
        context_hash: str = "",
        domain: str | None = None,
    ) -> TMHit | None:
        """Return the best fuzzy TM hit at or above ``threshold`` (0..1).

        fuzzy hits carry the same context gate as the exact path.
        A stored translation is reusable only under the caller's
        ``context_hash``; ``human_pe`` provenance and same-``domain`` rows
        are exempt so vetted translations stay reusable across books.
        """
        normalized = normalize_for_tm(source_text)
        if not normalized:
            return None
        if self._fts_path_active(src_lang, tgt_lang):
            # Large pool: FTS prefilter owns the lookup (may internally fall
            # back to a scan if the index errors mid-query).
            return self._lookup_fuzzy_fts(
                src_lang, tgt_lang, normalized, threshold, context_hash, domain
            )
        return self._lookup_fuzzy_scan(
            src_lang, tgt_lang, normalized, threshold, context_hash, domain
        )

    def _fts_path_active(self, src_lang: str, tgt_lang: str) -> bool:
        """Whether the FTS prefilter owns this pair (large pool, index healthy)."""
        return bool(
            self._fts_ok and self.entry_count(src_lang, tgt_lang) >= self._prefilter_min_pool
        )

    def _lookup_fuzzy_fts(
        self,
        src_lang: str,
        tgt_lang: str,
        normalized: str,
        threshold: float,
        context_hash: str = "",
        domain: str | None = None,
    ) -> TMHit | None:
        """FTS5 trigram prefilter + rapidfuzz rerank. None = miss (no full-scan fallback).

        Skipping the fallback is deliberate: on a large pool, a >=threshold hit
        without trigram overlap is near-impossible (sibling terms protect
        multi-term queries; single-term typo variants are the only known gap),
        and falling back would reintroduce the full scan exactly in the
        no-overlap case the prefilter exists to avoid. A miss only costs one
        LLM call — it can never produce a wrong translation, since the rerank
        enforces the same cutoff.
        """
        terms = _fts_query_terms(normalized)
        if not terms:
            # Punctuation-only queries have no indexable tokens; the exact
            # path (lookup_exact) already covers identical matches.
            return None
        match_expr = " OR ".join(_fts_phrase(t) for t in terms)
        try:
            with self._lock, self._get_conn() as conn:
                rows = conn.execute(
                    "SELECT rowid FROM tm_fts WHERE tm_fts MATCH ? "
                    "AND src_lang = ? AND tgt_lang = ? ORDER BY rank LIMIT ?",
                    (match_expr, src_lang, tgt_lang, self._prefilter_limit),
                ).fetchall()
                if not rows:
                    return None
                texts = conn.execute(
                    f"SELECT id, src_text FROM tm_entries WHERE id IN "
                    f"({','.join('?' * len(rows))})",
                    [int(r[0]) for r in rows],
                ).fetchall()
        except sqlite3.Error as exc:
            logger.warning("TM FTS prefilter failed, falling back to scan: %s", exc)
            return self._lookup_fuzzy_scan(
                src_lang, tgt_lang, normalized, threshold, context_hash, domain
            )
        candidates = [(int(r[0]), normalize_for_tm(str(r[1]))) for r in texts]
        matches = process.extract(
            normalized,
            [c[1] for c in candidates],
            scorer=fuzz.ratio,
            score_cutoff=max(threshold, 0.01) * 100,
            limit=10,
        )
        for _, score, index in matches:
            hit = self._fetch_hit_if_allowed(
                candidates[index][0], score / 100.0, context_hash, domain
            )
            if hit is not None:
                return hit
        return None

    def _lookup_fuzzy_scan(
        self,
        src_lang: str,
        tgt_lang: str,
        normalized: str,
        threshold: float,
        context_hash: str = "",
        domain: str | None = None,
    ) -> TMHit | None:
        """Full-pool rapidfuzz scan (small pools, or FTS failure fallback)."""
        pool = self._pool(src_lang, tgt_lang)
        normalized_sources, ids = pool
        if not normalized_sources:
            return None
        matches = process.extract(
            normalized,
            normalized_sources,
            scorer=fuzz.ratio,
            score_cutoff=max(threshold, 0.01) * 100,
            limit=10,
        )
        for _, score, index in matches:
            hit = self._fetch_hit_if_allowed(ids[index], score / 100.0, context_hash, domain)
            if hit is not None:
                return hit
        return None

    @staticmethod
    def _bump_use_count(conn: sqlite3.Connection, entry_id: int) -> None:
        """Best-effort reuse counter: audit metadata must never fail a read.

        A concurrent writer (another worker process on the shared tm.sqlite) can
        hold the WAL write lock past ``busy_timeout``; the hit is already read
        and chosen, so losing one count beats failing the translation.
        """
        try:
            conn.execute(
                "UPDATE tm_entries SET use_count = use_count + 1 WHERE id = ?",
                (entry_id,),
            )
        except sqlite3.OperationalError as exc:
            logger.debug("tm: use_count bump skipped for entry %s: %s", entry_id, exc)

    @staticmethod
    def _row_rejected(provenance: str, src_text: str, tgt_text: str) -> bool:
        """True for a stored row whose target is just the source carried over.

        A row can enter the store with an untranslated value (an MT pass-through
        that cleared the write-side filter). Serving it back as a finished
        translation converts stored noise into a shortcut that bypasses
        drafting, so reads refuse identity rows; they stay in the table
        untouched. normalize_for_tm never folds case, so a machine row differing
        only by case ("Introduction" -> "introduction") is the same echo; a
        human PE restyle may re-case deliberately and only a verbatim echo is
        noise.
        """
        src = normalize_for_tm(src_text)
        tgt = normalize_for_tm(tgt_text)
        if provenance == PROVENANCE_HUMAN_PE:
            return src == tgt
        return src.casefold() == tgt.casefold()

    @staticmethod
    def _hit_allowed(
        provenance: str,
        row_context: str,
        row_domain: str | None,
        context_hash: str,
        domain: str | None,
    ) -> bool:
        """Gate: context equality, human_pe exemption, same-domain pass."""
        if provenance == PROVENANCE_HUMAN_PE:
            return True
        if row_context == context_hash:
            return True
        return bool(domain and row_domain and row_domain == domain)

    def _fetch_hit_if_allowed(
        self,
        entry_id: int,
        similarity: float,
        context_hash: str,
        domain: str | None,
    ) -> TMHit | None:
        """Fetch a fuzzy candidate only when the context gate passes."""
        with self._lock, self._get_conn() as conn:
            cursor = conn.execute(
                "SELECT src_text, tgt_text, provenance, context_hash, domain "
                "FROM tm_entries WHERE id = ?",
                (entry_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            provenance = str(row[2])
            if self._row_rejected(str(row[2]), str(row[0]), str(row[1])):
                return None
            if not self._hit_allowed(provenance, str(row[3] or ""), row[4], context_hash, domain):
                return None
            self._bump_use_count(conn, entry_id)
        return TMHit(
            source_text=str(row[0]),
            target_text=str(row[1]),
            similarity=similarity,
            provenance=provenance,
        )

    def _pool(self, src_lang: str, tgt_lang: str) -> tuple[list[str], list[int]]:
        """Lazily load and cache the normalized source pool for a language pair."""
        cache_key = (src_lang, tgt_lang)
        with self._lock:
            conn = self._get_conn()
            generation = self._generation(conn, cache_key)
            cached = self._pool_cache.get(cache_key)
            if cached is not None and self._pool_gen.get(cache_key) == generation:
                return cached
            cursor = conn.execute(
                "SELECT id, src_text FROM tm_entries WHERE src_lang = ? AND tgt_lang = ?",
                (src_lang, tgt_lang),
            )
            rows = cursor.fetchall()
            normalized_sources = [normalize_for_tm(str(r[1])) for r in rows]
            ids = [int(r[0]) for r in rows]
            pool = (normalized_sources, ids)
            self._pool_cache[cache_key] = pool
            self._pool_gen[cache_key] = generation
            return pool

    def _invalidate_pool(self, src_lang: str, tgt_lang: str) -> None:
        """Drop one language pair's cached views, in the writer's own process.

        The generation bump is what tells *other* processes; this spare drop is
        what keeps the writer from paying even the staleness query on its next
        lookup. It runs after the commit, so a rollback cannot leave a cache that
        is both fresh-checked and stale.
        """
        with self._lock:
            pair = (src_lang, tgt_lang)
            self._pool_cache.pop(pair, None)
            self._pool_gen.pop(pair, None)
            self._count_gen.pop(pair, None)
            self._entry_count_cache.pop(pair, None)
            self._entry_count_cache.pop(None, None)

    # ------------------------------------------------------------------
    # Writeback
    # ------------------------------------------------------------------

    def writeback(self, entries: list[TMPendingEntry], max_retries: int = 5) -> int:
        """Upsert translated pairs. Returns the number of entries stored.

        Conflict policy: the latest translation wins, but ``human_pe``
        provenance is never downgraded by machine writeback — human review
        is the highest-trust signal in the pipeline.
        """
        if not entries:
            return 0
        for attempt in range(max_retries):
            try:
                touched: set[tuple[str, str]] = set()
                stored = 0
                with self._lock, self._get_conn() as conn:
                    for entry in entries:
                        normalized = normalize_for_tm(entry.source_text)
                        if not normalized or not normalize_for_tm(entry.target_text):
                            continue
                        conn.execute(
                            """
                            INSERT INTO tm_entries (
                                src_lang, tgt_lang, src_hash, src_text, tgt_text,
                                provenance, domain, context_hash, runs_json
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT(src_lang, tgt_lang, src_hash, context_hash) DO UPDATE SET
                                tgt_text = excluded.tgt_text,
                                -- Keep the stored label honest: the conflict
                                -- target does not include ``domain``, so without
                                -- this a rewrite under another domain would leave
                                -- a stale ``domain`` mismatched with tgt_text.
                                domain = excluded.domain,
                                provenance = CASE
                                    WHEN tm_entries.provenance = 'human_pe' THEN 'human_pe'
                                    ELSE excluded.provenance
                                END,
                                -- The stored emphasis must track the stored text,
                                -- or a hit would bold a span the new text no
                                -- longer has.
                                runs_json = excluded.runs_json,
                                -- a writeback means "Stored again", not
                                -- "served again", so use_count is intentionally NOT
                                -- bumped here. The former `use_count + 1` conflated
                                -- writes with lookups and polluted the reuse audit
                                -- data that scan()/evict_ids exist for.
                                updated_at = CURRENT_TIMESTAMP
                            """,
                            (
                                entry.src_lang,
                                entry.tgt_lang,
                                _source_hash(normalized),
                                entry.source_text,
                                entry.target_text,
                                entry.provenance,
                                entry.domain,
                                entry.context_hash,
                                entry.runs_json,
                            ),
                        )
                        stored += 1
                        touched.add((entry.src_lang, entry.tgt_lang))
                    if touched:
                        self._bump_generations(conn, touched)
                for pair in touched:
                    self._invalidate_pool(*pair)
                return stored
            except sqlite3.OperationalError as exc:
                if "locked" in str(exc).lower() and attempt < max_retries - 1:
                    delay = min(2.0, 0.05 * (2**attempt)) + random.uniform(0.01, 0.05)
                    time.sleep(delay)
                    continue
                raise
        return 0

    def scan(self) -> list[TMStoredEntry]:
        """Return every stored entry (id, pair, texts, provenance, use_count).

        Read-only companion to :meth:`evict_ids`: reusable memory that can only
        grow and never be corrected is a liability, so auditing has to be part
        of the public surface rather than a hand-written SQL query.
        """
        with self._lock, self._get_conn() as conn:
            rows = conn.execute(
                "SELECT id, src_lang, tgt_lang, src_text, tgt_text, provenance, "
                "domain, use_count FROM tm_entries ORDER BY id"
            ).fetchall()
        return [
            TMStoredEntry(
                id=int(r[0]),
                src_lang=str(r[1]),
                tgt_lang=str(r[2]),
                source_text=str(r[3]),
                target_text=str(r[4]),
                provenance=str(r[5]),
                domain=None if r[6] is None else str(r[6]),
                use_count=int(r[7]),
            )
            for r in rows
        ]

    def evict_ids(self, ids: list[int]) -> int:
        """Delete the given entry ids. Returns the number of rows removed.

        Correction path for entries that reached the store under a gate that has
        since been tightened — or that never should have passed one. A poisoned
        entry is worse than a missing one: :meth:`lookup_exact` serves it
        verbatim on every later run, so a single bad generation becomes the
        permanent, authoritative translation. The ``tm_fts_ad`` trigger keeps the
        trigram index in sync.
        """
        if not ids:
            return 0
        touched: set[tuple[str, str]] = set()
        removed = 0
        with self._lock, self._get_conn() as conn:
            for start in range(0, len(ids), 500):
                chunk = ids[start : start + 500]
                placeholders = ",".join("?" * len(chunk))
                rows = conn.execute(
                    f"SELECT DISTINCT src_lang, tgt_lang FROM tm_entries "
                    f"WHERE id IN ({placeholders})",
                    chunk,
                ).fetchall()
                touched.update((str(r[0]), str(r[1])) for r in rows)
                cursor = conn.execute(f"DELETE FROM tm_entries WHERE id IN ({placeholders})", chunk)
                removed += cursor.rowcount if cursor.rowcount and cursor.rowcount > 0 else 0
            if touched:
                self._bump_generations(conn, touched)
        for pair in touched:
            self._invalidate_pool(*pair)
        self._entry_count_cache.pop(None, None)
        return removed

    def entry_count(self, src_lang: str | None = None, tgt_lang: str | None = None) -> int:
        """Number of stored TM entries, optionally filtered by language pair."""
        with self._lock:
            conn = self._get_conn()
            pair = (src_lang, tgt_lang) if (src_lang and tgt_lang) else None
            generation: int | None = None
            query: str
            params: Sequence[Any]
            if pair is None:
                # A count over every pair has no generation of its own, so it
                # rides the database-wide signal and its whole-cache reset.
                self._sync_data_version(conn)
                query = "SELECT COUNT(*) FROM tm_entries"
                params = ()
            else:
                generation = self._generation(conn, pair)
                cached = self._entry_count_cache.get(pair)
                if cached is not None and self._count_gen.get(pair) == generation:
                    return cached
                query = "SELECT COUNT(*) FROM tm_entries WHERE src_lang = ? AND tgt_lang = ?"
                params = pair
            row = _read_one(conn, query, params)
            count = int(row[0]) if row else 0
            self._entry_count_cache[pair] = count
            if pair is not None and generation is not None:
                self._count_gen[pair] = generation
            return count

    def get_use_count(self, src_lang: str, tgt_lang: str, source_text: str) -> int:
        """Return the maximum use_count for a normalized source in a language pair."""
        normalized = normalize_for_tm(source_text)
        if not normalized:
            return 0
        key = _source_hash(normalized)
        with self._lock, self._get_conn() as conn:
            row = _read_one(
                conn,
                "SELECT MAX(use_count) FROM tm_entries "
                "WHERE src_lang = ? AND tgt_lang = ? AND src_hash = ?",
                (src_lang, tgt_lang, key),
            )
        return int(row[0]) if row and row[0] is not None else 0
