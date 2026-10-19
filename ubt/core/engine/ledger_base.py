"""SQLite WAL ledger foundation: connection, transactions, schema.

Holds the connection/transaction core (:class:`LedgerBase`) plus module-level
constants and helpers shared by domain mixins in
:mod:`ubt.core.engine.ledger_mixins`. Public classes and helpers remain
importable from the :mod:`ubt.core.engine.ledger` facade.

Schema policy: a **single flat baseline**. The historical 14-step incremental
migration ladder (v2..v14) was collapsed into one authoritative
``_SCHEMA_SQL`` that creates the final shape directly, with the composite
``(job_id, block_id)`` primary key and no dead columns. A database written by
an older build (``user_version`` below :data:`SCHEMA_VERSION`) is not migrated
in place: it is refused with instructions, because silently rewriting a
half-migrated legacy file is the exact data-loss shape the ladder used to risk.
"""

import json
import logging
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any, Self

from ubt.core.exceptions import LedgerError
from ubt.core.fs_perms import ensure_private_dir, restrict_sqlite_family
from ubt.core.ir.models import (
    TERMINAL_STATUSES,
    BlockProvenance,
    BlockStatus,
    BlockType,
    BoundingBox,
    FlowID,
    IRBlock,
    StyleMeta,
    make_element,
)
from ubt.model.ast import Confidence, RegionKind
from ubt.model.span import CompositeSpan, boxes_from_provenance

#: The single, flattened schema version. Bumped only when ``_SCHEMA_SQL``
#: changes; there is no longer a migration ladder behind it.
#:
#: Deliberately a high sentinel, not ``1``: the retired ladder stamped real
#: databases with ``user_version`` 1..14, so ``1`` would collide with the old
#: ladder's first rung and a genuine legacy file could pass the "already at the
#: baseline" check. A value the old code could never have written makes the
#: fresh/legacy discrimination unambiguous.
SCHEMA_VERSION = 1000

#: Backwards-compatible alias. Older callers/modules import
#: ``TARGET_SCHEMA_VERSION``; it now names the flat baseline.
TARGET_SCHEMA_VERSION = SCHEMA_VERSION

#: The authoritative schema. Fresh databases are created from this directly;
#: it is the *only* definition of the tables (no incremental ALTERs).
_SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS job_meta (
        job_id TEXT PRIMARY KEY,
        doc_id TEXT NOT NULL,
        source_path TEXT NOT NULL,
        target_lang TEXT NOT NULL,
        total_blocks INTEGER NOT NULL,
        status TEXT NOT NULL,
        metadata_json TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS blocks (
        block_id TEXT NOT NULL,
        job_id TEXT NOT NULL,
        flow_id TEXT NOT NULL,
        spine_index INTEGER NOT NULL,
        block_type TEXT NOT NULL DEFAULT 'narrative',
        bbox_json TEXT,
        style_json TEXT,
        source_text TEXT NOT NULL,
        draft_text TEXT,
        target_text TEXT,
        status TEXT NOT NULL,
        skip_translate INTEGER DEFAULT 0,
        glossary_hits_json TEXT,
        mtqe_score REAL,
        repair_rounds INTEGER DEFAULT 0,
        error_flags_json TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        tm_hit INTEGER DEFAULT 0,
        mqm_severity TEXT DEFAULT NULL,
        mqm_spans_json TEXT DEFAULT NULL,
        layout_role TEXT DEFAULT NULL,
        policy_translate INTEGER DEFAULT NULL,
        policy_reason TEXT DEFAULT NULL,
        provenance_json TEXT DEFAULT NULL,
        level INTEGER DEFAULT NULL,
        marker TEXT DEFAULT NULL,
        confidence TEXT DEFAULT NULL,
        decorative INTEGER DEFAULT NULL,
        span_chars_json TEXT DEFAULT NULL,
        PRIMARY KEY (job_id, block_id),
        FOREIGN KEY(job_id) REFERENCES job_meta(job_id)
    );

    CREATE TABLE IF NOT EXISTS batch_jobs (
        batch_id TEXT PRIMARY KEY,
        job_id TEXT NOT NULL,
        idempotency_key TEXT NOT NULL,
        status TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Cross-process single-writer guard, replacing the old msvcrt/fcntl
    -- file-offset lock (which could not work on NFS/K8s PVCs and crashed under
    -- Docker PID 1). One row per job; ``owner_pid`` + ``lease_expires_at`` let a
    -- crashed writer's lock be reclaimed instead of wedging the job forever.
    CREATE TABLE IF NOT EXISTS job_locks (
        job_id TEXT PRIMARY KEY,
        owner_pid INTEGER NOT NULL,
        owner_token TEXT NOT NULL,
        lease_expires_at REAL NOT NULL,
        acquired_at REAL NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_block_status ON blocks(job_id, status);
    CREATE INDEX IF NOT EXISTS idx_block_flow ON blocks(job_id, flow_id, spine_index);
    CREATE INDEX IF NOT EXISTS idx_batch_jobs_idem ON batch_jobs(idempotency_key, status);
    CREATE INDEX IF NOT EXISTS idx_blocks_job_mtqe ON blocks(job_id, mtqe_score);
    CREATE INDEX IF NOT EXISTS idx_blocks_pending ON blocks(job_id, status, spine_index, block_id);
    CREATE INDEX IF NOT EXISTS idx_blocks_rollup ON blocks(job_id, skip_translate, mtqe_score, status);
"""

# Every column ``_row_to_block`` reads by name. Kept as a literal (not derived
# from the CREATE statement) to guarantee all required schema fields are verified
# during startup; see the check in ``_init_db_with_migrations``.
_REQUIRED_BLOCK_COLUMNS = frozenset(
    {
        "block_id",
        "flow_id",
        "spine_index",
        "block_type",
        "bbox_json",
        "style_json",
        "source_text",
        "draft_text",
        "target_text",
        "status",
        "skip_translate",
        "tm_hit",
        "glossary_hits_json",
        "mtqe_score",
        "repair_rounds",
        "error_flags_json",
        "mqm_severity",
        "mqm_spans_json",
        "layout_role",
        "policy_translate",
        "policy_reason",
        "provenance_json",
        "level",
        "marker",
        "confidence",
        "decorative",
        "span_chars_json",
    }
)

# The ledger's historic logger name is kept explicit (not __name__) so moving
# methods into these modules cannot change which logger their records go to.
logger = logging.getLogger("ubt.core.engine.ledger")

# Statuses that mean "work not finished"; any block left in one of these states at
# export time is stale work that silently leaked past a crashed/failed stage.
# Derived from the IR's single TERMINAL_STATUSES definition.
NON_TERMINAL_STATUSES: tuple[str, ...] = tuple(
    sorted(s.value for s in BlockStatus if s not in TERMINAL_STATUSES)
)


def _upsert_blocks_batch(cursor: sqlite3.Cursor, job_id: str, blocks: Sequence[IRBlock]) -> None:
    """Batch upsert IRBlock entities into the blocks table using ON CONFLICT update.

    One *append* must carry unique ids: a duplicate within the batch means the
    extractor produced colliding ids, and the ``ON CONFLICT`` below would
    silently keep whichever row happened to be written last. That is how a TOC
    renumbering bug shipped a PDF with no title page — the contents rows
    overwrote the front matter block-for-block with no error anywhere. Refusing
    here turns that class of parse defect into a loud failure at ingest.
    (Across appends, re-upserting a known id is the supported resume path and
    stays allowed.)
    """
    if not blocks:
        return
    seen_ids: set[str] = set()
    duplicates: set[str] = set()
    for b in blocks:
        if b.id in seen_ids:
            duplicates.add(b.id)
        seen_ids.add(b.id)
    if duplicates:
        sample = ", ".join(sorted(duplicates)[:5])
        raise LedgerError(
            f"duplicate block id(s) in one append for job '{job_id}': {sample}"
            + (f" (+{len(duplicates) - 5} more)" if len(duplicates) > 5 else "")
            + ". The extractor assigned the same id to different content; "
            "fix the id assignment before translating (a silent overwrite would drop blocks).",
            details={"job_id": job_id, "duplicate_ids": sorted(duplicates)},
        )
    blocks_data = [
        (
            b.id,
            job_id,
            b.flow_id.value,
            b.spine_index,
            b.block_type.value,
            b.bbox.model_dump_json() if b.bbox else None,
            b.style.model_dump_json() if b.style else None,
            b.source_text,
            b.draft_text,
            b.target_text,
            b.status.value,
            1 if b.skip_translate else 0,
            json.dumps(b.glossary_hits, ensure_ascii=False),
            b.mtqe_score,
            b.repair_rounds,
            json.dumps(b.error_flags, ensure_ascii=False),
            b.region.value if b.region else None,
            (1 if b.policy_translate else 0) if b.policy_translate is not None else None,
            b.policy_reason,
            json.dumps(
                b.provenance.model_dump(exclude_none=True)
                if hasattr(b.provenance, "model_dump")
                else b.provenance,
                ensure_ascii=False,
            ),
            b.mqm_severity,
            json.dumps(b.mqm_spans, ensure_ascii=False) if b.mqm_spans else None,
            getattr(b.element, "level", None),
            getattr(b.element, "marker", None),
            b.element.confidence.value,
            1 if b.element.decorative else 0,
            # The element's canonical ``Span.chars`` as ``[start, end]``. Stored
            # so a resumed block slices the canonical stream again instead of
            # falling back to its (coarser) carried text.
            json.dumps(list(b.element.span.chars)) if b.element.span.chars else None,
        )
        for b in blocks
    ]

    cursor.executemany(
        """
        INSERT INTO blocks (
            block_id, job_id, flow_id, spine_index, block_type,
            bbox_json, style_json, source_text, draft_text, target_text,
            status, skip_translate, glossary_hits_json, mtqe_score,
            repair_rounds, error_flags_json,
            layout_role,
            policy_translate, policy_reason, provenance_json,
            mqm_severity, mqm_spans_json,
            level, marker, confidence, decorative,
            span_chars_json,
            updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(job_id, block_id) DO UPDATE SET
            flow_id = excluded.flow_id,
            spine_index = excluded.spine_index,
            block_type = excluded.block_type,
            bbox_json = excluded.bbox_json,
            style_json = excluded.style_json,
            source_text = excluded.source_text,
            skip_translate = excluded.skip_translate,
            -- Skip-flag transitions must drive status/target, or a resumed job
            -- keeps stale state in both directions:
            --  0 -> 1 (block NEWLY verbatim, e.g. skip rules tightened): the
            --    excluded row already carries source as its target, so adopt it.
            --  1 -> 0 (block NEWLY translatable, e.g. --translate-chrome): the
            --    stale source-as-target and any MTQE_PASSED status would make
            --    the block look finished and it would never be re-drafted, so
            --    reset it to pending and discard the source echo. The export
            --    completion floor only checks target-is-non-empty and does not
            --    catch this.
            status = CASE
                WHEN excluded.skip_translate = 0 AND blocks.skip_translate = 1
                THEN 'pending'
                ELSE blocks.status
            END,
            target_text = CASE
                WHEN excluded.skip_translate = 1 AND blocks.skip_translate = 0
                THEN excluded.target_text
                WHEN excluded.skip_translate = 0 AND blocks.skip_translate = 1
                THEN NULL
                ELSE blocks.target_text
            END,
            draft_text = CASE
                WHEN excluded.skip_translate = 0 AND blocks.skip_translate = 1
                THEN NULL
                ELSE blocks.draft_text
            END,
            mtqe_score = CASE
                WHEN excluded.skip_translate = 0 AND blocks.skip_translate = 1
                THEN NULL
                ELSE blocks.mtqe_score
            END,
            error_flags_json = CASE
                WHEN excluded.skip_translate = 0 AND blocks.skip_translate = 1
                THEN '[]'
                ELSE blocks.error_flags_json
            END,
            layout_role = excluded.layout_role,
            policy_translate = excluded.policy_translate,
            policy_reason = excluded.policy_reason,
            provenance_json = excluded.provenance_json,
            level = excluded.level,
            marker = excluded.marker,
            confidence = excluded.confidence,
            decorative = excluded.decorative,
            -- Structural, so adopt the freshly-ingested range; COALESCE only to
            -- keep a previously-stored range if an update omits it (an old
            -- writer that does not carry the column).
            span_chars_json = COALESCE(excluded.span_chars_json, blocks.span_chars_json),
            mqm_severity = COALESCE(excluded.mqm_severity, blocks.mqm_severity),
            mqm_spans_json = COALESCE(excluded.mqm_spans_json, blocks.mqm_spans_json),
            updated_at = CURRENT_TIMESTAMP
        """,
        blocks_data,
    )


class LedgerBase:
    """Connection, thread-safety, pragma and schema-migration foundation of the ledger.

    Domain methods live in :mod:`ubt.core.engine.ledger_mixins`; the assembled
    public class is :class:`ubt.core.engine.ledger.SQLiteJobLedger`.
    """

    def __init__(
        self,
        db_path: Path,
        timeout: float = 30.0,
        *,
        read_only: bool = False,
        fresh: bool = False,
    ) -> None:
        self.db_path = db_path
        self.timeout = timeout
        self.read_only = read_only
        # ``--fresh`` means "discard this job's prior state and start over". A
        # ledger file written by an older build is reset to the baseline rather
        # than refused, which is exactly what the operator asked for; without the
        # flag an old schema is refused (never silently migrated).
        self._fresh = fresh
        if not read_only:
            # Only a directory UBT itself creates is tightened to owner-only;
            # an existing user-chosen db_dir keeps its mode (see ensure_private_dir).
            ensure_private_dir(self.db_path.parent)
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        # Set by close(). A stage generator that finalizes *after* the run's
        # finally closed the ledger used to silently reopen a fresh connection
        # (``_get_conn`` rebuilt one whenever ``_conn`` was None), which wrote
        # past the writer-lock release the finally had just performed. Once
        # closed, the ledger refuses to reopen.
        self._closed = False
        # Bumped by every committed write to a block row. :class:`StageContext`
        # caches a block snapshot against this counter, so a later reader reloads
        # without having to remember to manually invalidate. Valid
        # within one process: a job's writer lock keeps other processes from
        # writing its ledger underneath it.
        self._blocks_seq = 0
        # Owning job of this ledger file, set by ``init_job_from_manifest``. It
        # scopes block_id-keyed writes: block ids are content-derived and repeat
        # across jobs, so an unscoped write on a multi-job file could hit the
        # wrong job's row.
        self._job_scope: str | None = None
        self._init_connection()
        if not read_only:
            self._init_db_with_migrations()
            # The ledger holds every source and target paragraph of the book, and
            # SQLite creates the file 0666&~umask (0644 under a normal umask).
            restrict_sqlite_family(self.db_path)

    def _init_connection(self) -> None:
        """Establish a persistent, thread-safe connection configured with WAL and busy timeout once."""
        try:
            if self.read_only:
                conn = sqlite3.connect(
                    f"file:{self.db_path.resolve()}?mode=ro",
                    uri=True,
                    timeout=self.timeout,
                    isolation_level=None,
                    check_same_thread=False,
                )
            else:
                conn = sqlite3.connect(
                    str(self.db_path),
                    timeout=self.timeout,
                    isolation_level=None,  # Autocommit mode for explicit manual transactions
                    check_same_thread=False,
                )
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
            conn.row_factory = sqlite3.Row
            conn.execute(f"PRAGMA busy_timeout={int(self.timeout * 1000)};")
            self._conn = conn
        except sqlite3.Error as err:
            raise LedgerError(
                f"Failed to initialize SQLite connection: {err}",
                details={"db_path": str(self.db_path), "error": str(err)},
            ) from err

    def _safe_rollback(self) -> None:
        """Roll back any open manual transaction, tolerating a missing/closed connection."""
        if self._conn is not None:
            with suppress(sqlite3.Error):
                self._conn.execute("ROLLBACK;")
            with suppress(sqlite3.Error):
                self._conn.rollback()

    @property
    def blocks_seq(self) -> int:
        """Revision counter for block-row writes; see :meth:`_mark_blocks_changed`."""
        return self._blocks_seq

    def _mark_blocks_changed(self) -> None:
        """Note that block rows were committed, invalidating cached snapshots.

        Called inside ``_get_conn()``, which holds ``self._lock`` for the whole
        transaction, so the increment needs no guard of its own.
        """
        self._blocks_seq += 1

    @contextmanager
    def _get_conn(self) -> Iterator[sqlite3.Connection]:
        """Provides thread-synchronized access to the persistent database connection."""
        with self._lock:
            if self._closed:
                raise LedgerError(
                    "ledger is closed; refusing to reopen a connection",
                    details={"db_path": str(self.db_path)},
                )
            if self._conn is None:
                self._init_connection()
            if self._conn is None:
                # Explicit, not assert: -O would strip the check and yield None.
                raise RuntimeError("ledger connection failed to initialize")
            try:
                yield self._conn
            except sqlite3.Error as err:
                # Roll back any open manual transaction so the persistent connection is
                # never left "within a transaction" (which would poison every later
                # BEGIN IMMEDIATE).
                self._safe_rollback()
                raise LedgerError(
                    f"SQLite transaction or execution error: {err}",
                    details={"db_path": str(self.db_path), "error": str(err)},
                ) from err
            except BaseException:
                # Non-SQLite exceptions (KeyError, ValueError, ...) raised inside a transaction
                # must also roll back, otherwise the connection stays poisoned.
                self._safe_rollback()
                raise

    def close(self) -> None:
        """Close the underlying persistent connection (idempotent).

        Also marks the ledger closed: a later ``_get_conn`` raises rather than
        silently reopening, so a stage that finalizes after the run's teardown
        cannot write past the writer-lock release.
        """
        with self._lock:
            self._closed = True
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    logger.warning("Error while closing ledger connection %s", self.db_path)
                finally:
                    self._conn = None

    # ``Self`` preserves the historic ``-> "SQLiteJobLedger"`` contract at every
    # call site (the context manager is only ever used on the assembled class).
    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def _init_db_with_migrations(self) -> None:
        """Create the flat baseline schema, or refuse a legacy database.

        Fresh databases are created from :data:`_SCHEMA_SQL` and stamped
        ``user_version = SCHEMA_VERSION``. A database from an older build
        (``user_version`` above 0 but below the baseline, or at the retired
        14-step ladder's version) is refused with instructions rather than
        migrated in place: silently rewriting a half-migrated legacy file is the
        data-loss shape the old ladder risked, and the ledger holds a run's only
        copy of its translations.

        The method name is kept for call-site stability even though there are no
        longer any incremental migrations.
        """
        with self._get_conn() as conn:
            current_version: int = conn.execute("PRAGMA user_version;").fetchone()[0]

            if current_version == SCHEMA_VERSION:
                # Already at the flat baseline; verify the columns are actually
                # present (a version stamp is not evidence the schema is right).
                self._verify_block_columns(conn, current_version)
                return

            if current_version == 0:
                # Fresh (or fully empty) database: create the final shape
                # directly. ``executescript`` implicitly commits, so the stamp is
                # written alongside the DDL.
                conn.executescript(_SCHEMA_SQL)
                conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION};")
            elif self._fresh:
                # ``--fresh`` explicitly discards the job's prior state, so a
                # legacy schema is reset to the baseline instead of refused —
                # that is the operator's stated intent. Drop every table this
                # build owns (a stale ladder could have left extra columns/tables)
                # and recreate the flat baseline. This is the *only* path that
                # rewrites an old ledger, and only because the user asked to.
                self._reset_to_baseline(conn)
                logger.warning(
                    "Reset ledger %s from schema version %s to the baseline "
                    "(version %s) for this --fresh run.",
                    self.db_path,
                    current_version,
                    SCHEMA_VERSION,
                )
            else:
                # A legacy (ladder-migrated) database without --fresh. Refuse
                # rather than silently downgrade or half-migrate: the operator
                # chooses (a new --job-id, or --fresh to discard and restart).
                raise LedgerError(
                    f"Ledger at {self.db_path} uses schema version {current_version}, "
                    f"but this build ships the flattened baseline version "
                    f"{SCHEMA_VERSION}. Incremental migration was removed, and "
                    "rewriting a legacy ledger in place risks the run's only copy "
                    "of its translations. Start a fresh job (a new --job-id, or "
                    "re-run with --fresh to discard this ledger and restart), or "
                    "restore a backup if you need the old data with an older build.",
                    details={
                        "db_version": current_version,
                        "baseline_version": SCHEMA_VERSION,
                    },
                )

            self._verify_block_columns(conn, SCHEMA_VERSION)

    def _reset_to_baseline(self, conn: sqlite3.Connection) -> None:
        """Drop this build's tables and recreate the flat baseline schema.

        Only reached under ``--fresh``. Drops only the tables UBT owns so an
        unrelated table in the same file is untouched, then recreates the
        baseline and stamps it.

        ``job_locks`` is deliberately **not** dropped. The pipeline acquires the
        writer lock *before* constructing this ledger (so the whole run, ingest
        included, is under the guard), which means this reset runs while the
        caller already holds a row in that table. Dropping it deleted the
        caller's own lock: any second process could then acquire the same job
        immediately and the two would write one ledger concurrently — each
        paying the provider, each overwriting the other's checkpoints. The
        table's shape is part of the baseline schema and is recreated by
        ``_SCHEMA_SQL`` with ``IF NOT EXISTS``, so leaving the live row in place
        keeps both the schema and the guard intact.
        """
        for table in ("blocks", "job_meta", "batch_jobs", "blocks_v10", "blocks_v11"):
            conn.execute(f"DROP TABLE IF EXISTS {table};")
        conn.executescript(_SCHEMA_SQL)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION};")

    def _verify_block_columns(self, conn: sqlite3.Connection, version: int) -> None:
        """Fail loudly if the ``blocks`` table is missing a column we read.

        The version stamp alone is not evidence the schema is right: a table
        created by a hand-edited file or a partial write can be at the baseline
        version with columns missing. Checking here fails at startup instead of
        deep inside ``_row_to_block``.
        """
        columns = {
            str(row["name"]) for row in conn.execute("PRAGMA table_info(blocks);").fetchall()
        }
        missing = sorted(_REQUIRED_BLOCK_COLUMNS - columns)
        if missing:
            raise LedgerError(
                f"Ledger at {self.db_path} is missing required blocks columns: "
                f"{', '.join(missing)}. The file is corrupt or was written by an "
                "incompatible build. Delete it to start fresh, or restore a backup.",
                details={"missing_columns": missing, "db_version": version},
            )

    def _block_scope(self, conn: sqlite3.Connection, job_id: str | None) -> str | None:
        """Resolve the job that scopes a block_id-keyed write.

        An explicit ``job_id`` (which may be a doc_id alias) resolves first.
        Otherwise the ledger's own job is used. A file holding several jobs with
        no explicit scope is refused rather than guessed: block ids repeat
        across jobs, so an unscoped write could corrupt another job's row.
        """
        if job_id:
            return self._resolve_actual_job_id(conn, job_id)
        if self._job_scope is not None:
            return self._job_scope
        rows = conn.execute("SELECT DISTINCT job_id FROM blocks LIMIT 2").fetchall()
        if len(rows) == 1:
            self._job_scope = str(rows[0]["job_id"])
            return self._job_scope
        if len(rows) > 1:
            raise LedgerError(
                "blocks table holds multiple jobs; pass job_id to scope the write",
                details={"db_path": str(self.db_path)},
            )
        return None

    def _resolve_actual_job_id(self, conn: sqlite3.Connection, job_id: str) -> str:
        """Resolve exact job_id, falling back to latest job_id for doc_id without cross-job mixing."""
        row = conn.execute("SELECT job_id FROM job_meta WHERE job_id = ?", (job_id,)).fetchone()
        if row:
            return str(row["job_id"])
        # If not found directly, check if it was a doc_id, picking only the most recent job:
        doc_row = conn.execute(
            "SELECT job_id FROM job_meta WHERE doc_id = ? ORDER BY created_at DESC, rowid DESC LIMIT 1",
            (job_id,),
        ).fetchone()
        if doc_row:
            return str(doc_row["job_id"])
        return job_id

    def _row_to_block(self, row: sqlite3.Row) -> IRBlock:
        """Convert a SQLite row to an IRBlock instance."""
        bbox = BoundingBox.model_validate_json(row["bbox_json"]) if row["bbox_json"] else None
        style = StyleMeta.model_validate_json(row["style_json"]) if row["style_json"] else None
        glossary_hits = json.loads(row["glossary_hits_json"]) if row["glossary_hits_json"] else []
        error_flags = json.loads(row["error_flags_json"]) if row["error_flags_json"] else []
        mqm_spans = json.loads(row["mqm_spans_json"]) if row["mqm_spans_json"] else []
        try:
            provenance: dict[str, Any] = (
                json.loads(row["provenance_json"]) if row["provenance_json"] else {}
            )
        except (ValueError, TypeError):
            provenance = {}
        if not isinstance(provenance, dict):
            provenance = {}
        policy_raw = row["policy_translate"]

        region = RegionKind(row["layout_role"]) if row["layout_role"] else None
        confidence_raw = row["confidence"]
        confidence = Confidence(confidence_raw) if confidence_raw else Confidence.INFERRED
        level_raw = row["level"]
        level = int(level_raw) if level_raw is not None else 1
        marker = row["marker"] or ""
        decorative = bool(row["decorative"])
        chars: tuple[int, int] | None = None
        raw_chars = row["span_chars_json"]
        if raw_chars:
            try:
                start, end = json.loads(raw_chars)
                chars = (int(start), int(end))
            except (ValueError, TypeError):
                chars = None
        span: CompositeSpan | None = None
        boxes = boxes_from_provenance(provenance.get("physical_boxes"))
        if boxes:
            span = CompositeSpan(boxes=boxes, chars=chars)

        return IRBlock(
            element=make_element(
                id=row["block_id"],
                spine_index=row["spine_index"],
                block_type=BlockType(row["block_type"]),
                flow_id=FlowID(row["flow_id"]),
                region=region,
                source_text=row["source_text"],
                bbox=bbox,
                span=span,
                skip_translate=bool(row["skip_translate"]),
                confidence=confidence,
                decorative=decorative,
                level=level,
                marker=marker,
                chars=chars,
            ),
            style=style,
            draft_text=row["draft_text"],
            target_text=row["target_text"],
            status=BlockStatus(row["status"]),
            tm_hit=bool(row["tm_hit"]),
            glossary_hits=glossary_hits,
            mtqe_score=row["mtqe_score"],
            repair_rounds=row["repair_rounds"],
            error_flags=error_flags,
            mqm_severity=row["mqm_severity"],
            mqm_spans=mqm_spans,
            policy_translate=bool(policy_raw) if policy_raw is not None else None,
            policy_reason=row["policy_reason"],
            provenance=BlockProvenance(**provenance),
        )
