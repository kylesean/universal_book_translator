"""SQLite WAL ledger foundation: connection, transactions, schema migrations.

Holds the connection/transaction/migration core (:class:`LedgerBase`)
plus module-level constants and helpers shared by domain mixins in
:mod:`ubt.core.engine.ledger_mixins`. Public classes and helpers remain
importable from the :mod:`ubt.core.engine.ledger` facade.
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
from ubt.core.fs_perms import restrict_dir_to_owner, restrict_sqlite_family
from ubt.core.ir.models import (
    TERMINAL_STATUSES,
    BlockStatus,
    BlockType,
    BoundingBox,
    FlowID,
    IRBlock,
    LayoutRole,
    SemanticRole,
    StructureRole,
    StyleMeta,
)

TARGET_SCHEMA_VERSION = 11

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
        "semantic_role",
        "structure_role",
        "policy_translate",
        "policy_reason",
        "provenance_json",
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
    """Batch upsert IRBlock entities into the blocks table using ON CONFLICT update."""
    if not blocks:
        return
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
            b.layout_role.value if b.layout_role else None,
            b.semantic_role.value if b.semantic_role else None,
            b.structure_role.value if b.structure_role else None,
            (1 if b.policy_translate else 0) if b.policy_translate is not None else None,
            b.policy_reason,
            json.dumps(b.provenance, ensure_ascii=False),
            b.mqm_severity,
            json.dumps(b.mqm_spans, ensure_ascii=False) if b.mqm_spans else None,
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
            layout_role, semantic_role, structure_role,
            policy_translate, policy_reason, provenance_json,
            mqm_severity, mqm_spans_json,
            updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
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
            semantic_role = excluded.semantic_role,
            structure_role = excluded.structure_role,
            policy_translate = excluded.policy_translate,
            policy_reason = excluded.policy_reason,
            provenance_json = excluded.provenance_json,
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

    def __init__(self, db_path: Path, timeout: float = 30.0, *, read_only: bool = False) -> None:
        self.db_path = db_path
        self.timeout = timeout
        self.read_only = read_only
        if not read_only:
            restrict_dir_to_owner(self.db_path.parent)
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
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
        """Close the underlying persistent connection (idempotent)."""
        with self._lock:
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
        """Idempotently initialize schema and run PRAGMA user_version schema migrations."""
        with self._get_conn() as conn:
            cursor = conn.execute("PRAGMA user_version;")
            current_version: int = cursor.fetchone()[0]

            if current_version == 0:
                conn.executescript("""
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
                        semantic_role TEXT DEFAULT NULL,
                        structure_role TEXT DEFAULT NULL,
                        policy_translate INTEGER DEFAULT NULL,
                        policy_reason TEXT DEFAULT NULL,
                        provenance_json TEXT DEFAULT NULL,
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

                    CREATE INDEX IF NOT EXISTS idx_block_status ON blocks(job_id, status);
                    CREATE INDEX IF NOT EXISTS idx_block_flow ON blocks(job_id, flow_id, spine_index);
                    CREATE INDEX IF NOT EXISTS idx_batch_jobs_idem ON batch_jobs(idempotency_key, status);
                    CREATE INDEX IF NOT EXISTS idx_blocks_job_mtqe ON blocks(job_id, mtqe_score);
                    CREATE INDEX IF NOT EXISTS idx_blocks_pending ON blocks(job_id, status, spine_index, block_id);
                    CREATE INDEX IF NOT EXISTS idx_blocks_rollup ON blocks(job_id, skip_translate, mtqe_score, status);
                    PRAGMA user_version = 11;
                """)
                current_version = TARGET_SCHEMA_VERSION

            # Migration to Version 2: Add worker lease columns and claim index
            if current_version < 2:
                conn.executescript("""
                    ALTER TABLE blocks ADD COLUMN owner_id TEXT DEFAULT NULL;
                    ALTER TABLE blocks ADD COLUMN lease_expires_at REAL DEFAULT NULL;
                    CREATE INDEX IF NOT EXISTS idx_block_claim ON blocks(job_id, status, lease_expires_at, spine_index);
                    PRAGMA user_version = 2;
                """)

            # Migration to Version 3: Translation Memory provenance column
            # (exact TM hits skip the LLM and are auditable)
            if current_version < 3:
                conn.executescript("""
                    ALTER TABLE blocks ADD COLUMN tm_hit INTEGER DEFAULT 0;
                    PRAGMA user_version = 3;
                """)

            # Migration to Version 4: MQM severity triage columns
            # (severity tier + serialized spans survive restarts and feed
            # the human PE queue exporters)
            if current_version < 4:
                conn.executescript("""
                    ALTER TABLE blocks ADD COLUMN mqm_severity TEXT DEFAULT NULL;
                    ALTER TABLE blocks ADD COLUMN mqm_spans_json TEXT DEFAULT NULL;
                    PRAGMA user_version = 4;
                """)

            # Migration to Version 5: document-v1 contract columns
            # role layers + policy verdict + provenance survive restarts so
            # render routing and verdict audits work post-resume)
            if current_version < 5:
                conn.executescript("""
                    ALTER TABLE blocks ADD COLUMN layout_role TEXT DEFAULT NULL;
                    ALTER TABLE blocks ADD COLUMN semantic_role TEXT DEFAULT NULL;
                    ALTER TABLE blocks ADD COLUMN structure_role TEXT DEFAULT NULL;
                    ALTER TABLE blocks ADD COLUMN policy_translate INTEGER DEFAULT NULL;
                    ALTER TABLE blocks ADD COLUMN policy_reason TEXT DEFAULT NULL;
                    ALTER TABLE blocks ADD COLUMN provenance_json TEXT DEFAULT NULL;
                    PRAGMA user_version = 5;
                """)

            # Migration to Version 6: Batch API job persistence (
            # batch_id survives restarts, idempotency key = sha1 over the
            # submitted custom_id set prevents duplicate-billed resubmits)
            if current_version < 6:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS batch_jobs (
                        batch_id TEXT PRIMARY KEY,
                        job_id TEXT NOT NULL,
                        idempotency_key TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE INDEX IF NOT EXISTS idx_batch_jobs_idem
                        ON batch_jobs(idempotency_key, status);
                    PRAGMA user_version = 6;
                """)

            # Migration to Version 7: drop the unused worker-lease columns.
            # The single-process engine drains work through
            # ``fetch_pending_blocks`` and claims no block-level lease; rows
            # stranded in 'claimed' by an interrupted run go back to
            # 'pending' so they are re-drafted.
            if current_version < 7:
                columns = {
                    str(row["name"])
                    for row in conn.execute("PRAGMA table_info(blocks);").fetchall()
                }
                conn.execute("UPDATE blocks SET status = 'pending' WHERE status = 'claimed';")
                conn.execute("DROP INDEX IF EXISTS idx_block_claim;")
                if "owner_id" in columns:
                    conn.execute("ALTER TABLE blocks DROP COLUMN owner_id;")
                if "lease_expires_at" in columns:
                    conn.execute("ALTER TABLE blocks DROP COLUMN lease_expires_at;")
                conn.execute("PRAGMA user_version = 7;")

            # Migration to Version 8: composite index on (job_id, mtqe_score) for distribution queries
            if current_version < 8:
                columns = {
                    str(row["name"])
                    for row in conn.execute("PRAGMA table_info(blocks);").fetchall()
                }
                if "mtqe_score" in columns:
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_blocks_job_mtqe ON blocks(job_id, mtqe_score);"
                    )
                conn.execute("PRAGMA user_version = 8;")

            # Migration to Version 9: composite index for the draft keyset
            # pagination. ``fetch_pending_blocks`` filters (job_id, status) and
            # orders/seek on (spine_index, block_id); without this the batch
            # loop re-scanned and re-sorted all remaining pending rows every
            # batch (O(N^2/batch_limit) on a 10k-block book).
            if current_version < 9:
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_blocks_pending "
                    "ON blocks(job_id, status, spine_index, block_id);"
                )
                conn.execute("PRAGMA user_version = 9;")

            # Migration to Version 10: covering index for get_job_stats. The
            # aggregates (COUNT/SUM over status, AVG over the QE-scored
            # population) live in one SELECT whose columns sit AFTER the text
            # columns in the row layout, so every progress event walked the
            # whole book's overflow chain — O(book bytes) per event, 35 ms on a
            # 20k-block ledger. The index carries all
            # four referenced columns, making both stats queries index-only.
            if current_version < 10:
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_blocks_rollup "
                    "ON blocks(job_id, skip_translate, mtqe_score, status);"
                )
                conn.execute("PRAGMA user_version = 10;")

            # Migration to Version 11: composite primary key (job_id, block_id).
            # ``block_id`` is content-derived (``ch_001#b0001``), so two jobs over
            # the same document produce identical block ids. With block_id as the
            # sole PRIMARY KEY, ingesting the second job's blocks overwrote the
            # first job's rows and left it with none of its own. SQLite cannot
            # alter a primary key, so the table is rebuilt (create-copy-drop-
            # rename); every block_id-keyed write is now scoped by job_id.
            if current_version < 11:
                conn.executescript("""
                    ALTER TABLE blocks RENAME TO blocks_v10;
                    CREATE TABLE blocks (
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
                        semantic_role TEXT DEFAULT NULL,
                        structure_role TEXT DEFAULT NULL,
                        policy_translate INTEGER DEFAULT NULL,
                        policy_reason TEXT DEFAULT NULL,
                        provenance_json TEXT DEFAULT NULL,
                        PRIMARY KEY (job_id, block_id),
                        FOREIGN KEY(job_id) REFERENCES job_meta(job_id)
                    );
                    INSERT INTO blocks (
                        block_id, job_id, flow_id, spine_index, block_type,
                        bbox_json, style_json, source_text, draft_text, target_text,
                        status, skip_translate, glossary_hits_json, mtqe_score,
                        repair_rounds, error_flags_json, updated_at, tm_hit,
                        mqm_severity, mqm_spans_json, layout_role, semantic_role,
                        structure_role, policy_translate, policy_reason, provenance_json
                    )
                    SELECT
                        block_id, job_id,
                        COALESCE(flow_id, 'main_story'),
                        COALESCE(spine_index, 0),
                        COALESCE(block_type, 'narrative'),
                        bbox_json, style_json,
                        COALESCE(source_text, ''),
                        draft_text, target_text, status,
                        COALESCE(skip_translate, 0),
                        glossary_hits_json, mtqe_score,
                        COALESCE(repair_rounds, 0),
                        error_flags_json, updated_at,
                        COALESCE(tm_hit, 0),
                        mqm_severity, mqm_spans_json, layout_role, semantic_role,
                        structure_role, policy_translate, policy_reason, provenance_json
                    FROM blocks_v10;
                    DROP TABLE blocks_v10;
                    CREATE INDEX IF NOT EXISTS idx_block_status ON blocks(job_id, status);
                    CREATE INDEX IF NOT EXISTS idx_block_flow ON blocks(job_id, flow_id, spine_index);
                    CREATE INDEX IF NOT EXISTS idx_blocks_job_mtqe ON blocks(job_id, mtqe_score);
                    CREATE INDEX IF NOT EXISTS idx_blocks_pending ON blocks(job_id, status, spine_index, block_id);
                    CREATE INDEX IF NOT EXISTS idx_blocks_rollup ON blocks(job_id, skip_translate, mtqe_score, status);
                    PRAGMA user_version = 11;
                """)

            # Guard against TARGET_SCHEMA_VERSION drift: migrations above must
            # land exactly on the declared target, otherwise future restarts
            # silently skip new migrations.
            final_version: int = conn.execute("PRAGMA user_version;").fetchone()[0]
            if final_version != TARGET_SCHEMA_VERSION:
                raise LedgerError(
                    f"Ledger schema version mismatch: db={final_version} "
                    f"target={TARGET_SCHEMA_VERSION}",
                    details={
                        "db_version": final_version,
                        "target": TARGET_SCHEMA_VERSION,
                    },
                )

            # The version stamp alone is not evidence the schema is right. Existing
            # tables may have skipped migrations or created with incomplete columns.
            # Check the actual columns so the mismatch fails here, not later
            # inside ``_row_to_block``.
            columns = {
                str(row["name"]) for row in conn.execute("PRAGMA table_info(blocks);").fetchall()
            }
            missing = sorted(_REQUIRED_BLOCK_COLUMNS - columns)
            if missing:
                raise LedgerError(
                    f"Ledger at {self.db_path} predates this schema and cannot be "
                    f"migrated: blocks is missing {', '.join(missing)}. Delete the "
                    f"file to start fresh, or restore a backup of it.",
                    details={"missing_columns": missing, "db_version": final_version},
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

        return IRBlock(
            id=row["block_id"],
            flow_id=FlowID(row["flow_id"]),
            spine_index=row["spine_index"],
            block_type=BlockType(row["block_type"]),
            bbox=bbox,
            style=style,
            source_text=row["source_text"],
            draft_text=row["draft_text"],
            target_text=row["target_text"],
            status=BlockStatus(row["status"]),
            skip_translate=bool(row["skip_translate"]),
            tm_hit=bool(row["tm_hit"]),
            glossary_hits=glossary_hits,
            mtqe_score=row["mtqe_score"],
            repair_rounds=row["repair_rounds"],
            error_flags=error_flags,
            mqm_severity=row["mqm_severity"],
            mqm_spans=mqm_spans,
            layout_role=LayoutRole(row["layout_role"]) if row["layout_role"] else None,
            semantic_role=SemanticRole(row["semantic_role"]) if row["semantic_role"] else None,
            structure_role=StructureRole(row["structure_role"]) if row["structure_role"] else None,
            policy_translate=bool(policy_raw) if policy_raw is not None else None,
            policy_reason=row["policy_reason"],
            provenance=provenance,
        )
