"""Domain partitions for the SQLite job ledger: job lifecycle/metadata/usage,
block reads/writes, and Batch API task persistence.

These are *mixins*, not subclasses: they add methods to the host and expect
:class:`~ubt.core.engine.ledger_base.LedgerBase` to supply the connection and
scope. ``SQLiteJobLedger`` is the only class that inherits
``LedgerBase`` -- the previous shape had every mixin extend ``LedgerBase`` and
then re-combine, which was a God Object cut into three files wearing a mixin
costume (the diamond was linear, but the type said "is-a base" four times).
"""

import json
import sqlite3
from collections.abc import Iterable
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from ubt.core.engine.delivery_freshness import RENDERED_TEXT_DIGEST_KEY
from ubt.core.engine.ledger_base import (
    NON_TERMINAL_STATUSES,
    _upsert_blocks_batch,
    logger,
)
from ubt.core.exceptions import LedgerError
from ubt.core.ir.models import (
    TERMINAL_STATUSES,
    BlockStatus,
    BlockType,
    BookManifest,
    ChapterIR,
    FlowID,
    IRBlock,
)
from ubt.core.qe.defect_taxonomy import (
    DRAFTING_ERROR_PREFIX,
    TRANSIENT_FAILURE_PREFIXES,
    has_triage_verdict,
    is_repair_only_transient_failure,
    is_transient_failure,
)
from ubt.core.qe.score_policy import QE_SCORED_SQL

if TYPE_CHECKING:
    # Methods reach into the host's connection/scope; at runtime the host is
    # ``SQLiteJobLedger`` (LedgerBase + these mixins), at type-check time the
    # mixin methods type against LedgerBase's API.
    from ubt.core.engine.ledger_base import LedgerBase as _LedgerHost

    class _LedgerHostWithMetadata(_LedgerHost):
        def get_job_metadata_value(self, job_id: str, key: str) -> Any | None: ...
else:
    _LedgerHost = object
    _LedgerHostWithMetadata = object

#: Metadata key holding the Agent translation session (leases and package
#: state). Kept in ``job_meta.metadata_json`` rather than a new table: the
#: session is a small, job-scoped state machine and must commit atomically with
#: the block rows it owns.
AGENT_SESSION_KEY = "agent_session"

#: Metadata key holding the deterministic request fingerprint used by Agent
#: prepare. A repeated prepare with the same id and different inputs must not
#: silently reuse the prior ledger.
AGENT_REQUEST_FINGERPRINT_KEY = "agent_request_fingerprint"

#: Terminal ``job_meta.status`` values. A ``completed`` finalize must not
#: overwrite any of these (the string mirror of the queue's terminal set).


def _style_json(style: Any) -> str | None:
    """Serialize a checkpoint's style for the ``style_json`` column.

    Accepts a ``StyleMeta`` (the draft stage's parsed target runs), a raw JSON
    string, or ``None``; ``None`` leaves the stored column untouched via the
    caller's ``COALESCE``.
    """
    if style is None:
        return None
    if isinstance(style, str):
        return style
    dump = getattr(style, "model_dump_json", None)
    if dump is not None:
        return cast(str, dump())
    return json.dumps(style, ensure_ascii=False)


def _chapter_match(chapter_id: str) -> tuple[str, tuple[str, ...]]:
    """SQL fragment + params selecting every block that belongs to one chapter.

    Blocks carry ids shaped ``<chapter_id>#<n>`` (or ``<chapter_id>:<n>``).
    ``chapter_id`` is parser-derived and routinely contains ``_`` (``ch_001``,
    ``docx_main``), which is a ``LIKE`` wildcard: interpolating it unescaped let
    ``ch_001`` match ``chX001`` and made a chapter-scoped read or reset touch
    foreign rows. Escape ``\\``, ``%`` and ``_`` and declare ``ESCAPE`` so the
    prefix match is literal.
    """
    escaped = chapter_id.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    clause = " AND (block_id = ? OR block_id LIKE ? ESCAPE '\\' OR block_id LIKE ? ESCAPE '\\')"
    return clause, (chapter_id, f"{escaped}#%", f"{escaped}:%")


def _as_usage_totals(raw: Any) -> dict[str, dict[str, int]]:
    """Coerce a stored or in-flight usage mapping into ``{model: {counter: int}}``.

    A non-numeric counter is dropped rather than guessed at, because these totals
    are what the job is priced from.
    """
    totals: dict[str, dict[str, int]] = {}
    if not isinstance(raw, dict):
        return totals
    for model, counters in raw.items():
        if not isinstance(counters, dict):
            continue
        clean = {
            str(key): int(value)
            for key, value in counters.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        if clean:
            totals[str(model)] = clean
    return totals


def merge_usage_totals(
    lifetime: dict[str, dict[str, int]], delta: dict[str, dict[str, int]]
) -> dict[str, dict[str, int]]:
    """Key-wise sum of two per-model usage maps; ``lifetime`` is not modified."""
    merged: dict[str, dict[str, int]] = {
        model: dict(counters) for model, counters in lifetime.items()
    }
    for model, counters in delta.items():
        target = merged.setdefault(model, {})
        for key, value in counters.items():
            target[key] = int(target.get(key, 0)) + int(value)
    return merged


class LedgerJobsMixin(_LedgerHost):
    """Job lifecycle, metadata, usage accounting and reporting reads."""

    def init_job_placeholder(
        self,
        job_id: str,
        source_path: str,
        target_lang: str,
        title: str | None = None,
        submitted_request: dict[str, Any] | None = None,
    ) -> None:
        """Persist an accepted job before its manifest is available.

        The API accepts a job and only then starts the pipeline. A restart in
        that window used to erase the only record of the job (the in-memory
        manager), so ``/status`` returned 404 even though the client held a
        valid accepted id. This placeholder is replaced by
        :meth:`init_job_from_manifest` once ingest knows the real document.

        ``submitted_request`` is the resolved, non-sensitive request snapshot
        the API accepted. It is what lets ``/resume`` recreate the in-memory
        record after a process restart without guessing model, profile or
        output choices. It deliberately excludes provider credentials and
        endpoints; those remain operator configuration.
        """
        self._job_scope = job_id
        with self._get_conn() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            row = conn.execute(
                "SELECT metadata_json FROM job_meta WHERE job_id = ?", (job_id,)
            ).fetchone()
            if row is not None:
                try:
                    meta = json.loads(row["metadata_json"] or "{}")
                except (TypeError, ValueError):
                    meta = {}
                if isinstance(meta, dict) and meta.get("accepted_placeholder") is True:
                    if submitted_request is not None:
                        meta["submitted_request"] = submitted_request
                    conn.execute(
                        """
                        UPDATE job_meta
                           SET source_path = ?, target_lang = ?, metadata_json = ?,
                               updated_at = CURRENT_TIMESTAMP
                         WHERE job_id = ?
                        """,
                        (
                            source_path,
                            target_lang,
                            json.dumps(meta, ensure_ascii=False),
                            job_id,
                        ),
                    )
                conn.execute("COMMIT;")
                return

            metadata: dict[str, Any] = {
                "title": title or Path(source_path).name,
                "accepted_placeholder": True,
            }
            if submitted_request is not None:
                metadata["submitted_request"] = submitted_request
            conn.execute(
                """
                INSERT INTO job_meta (
                    job_id, doc_id, source_path, target_lang, total_blocks, status, metadata_json
                ) VALUES (?, ?, ?, ?, 0, 'initialized', ?)
                """,
                (
                    job_id,
                    job_id,
                    source_path,
                    target_lang,
                    json.dumps(metadata, ensure_ascii=False),
                ),
            )
            conn.execute("COMMIT;")

    def init_job_from_manifest(self, job_id: str, manifest: BookManifest) -> None:
        """Atomically initialize job metadata from a lightweight BookManifest."""
        self._job_scope = job_id
        with self._get_conn() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            cursor = conn.cursor()
            cursor.execute("SELECT metadata_json FROM job_meta WHERE job_id = ?", (job_id,))
            existing = cursor.fetchone()
            if existing is not None:
                try:
                    existing_meta = json.loads(existing["metadata_json"] or "{}")
                except (TypeError, ValueError):
                    existing_meta = {}
                if not (
                    isinstance(existing_meta, dict)
                    and existing_meta.get("accepted_placeholder") is True
                ):
                    conn.execute("COMMIT;")
                    return
                metadata = {
                    "title": manifest.title,
                    "source_lang": manifest.source_lang,
                    "chapters": [c.model_dump() for c in manifest.chapters],
                    **manifest.run.to_metadata_dict(),
                    **manifest.metadata,
                }
                # The accepted request is intentionally not part of the
                # manifest. Preserve it across the placeholder replacement so
                # a restart after ingest can still resume with the exact
                # operator choices instead of a reconstructed approximation.
                submitted_request = existing_meta.get("submitted_request")
                if isinstance(submitted_request, dict):
                    metadata["submitted_request"] = submitted_request
                # Keep block-derived counts and metadata written by a run that
                # already progressed. Replacing the accepted placeholder is an
                # idempotent ingest step, not a fresh run: zeroing these here
                # would make a crash-rebuilt ingest report "0 blocks" while the
                # translated rows (and their terminal states) are still present.
                current_total = int(
                    conn.execute(
                        "SELECT total_blocks FROM job_meta WHERE job_id = ?", (job_id,)
                    ).fetchone()["total_blocks"]
                    or 0
                )
                cursor.execute(
                    """
                    UPDATE job_meta
                       SET doc_id = ?, source_path = ?, target_lang = ?,
                           total_blocks = ?, status = 'initialized', metadata_json = ?,
                           updated_at = CURRENT_TIMESTAMP
                     WHERE job_id = ?
                    """,
                    (
                        manifest.doc_id,
                        manifest.source_path,
                        manifest.target_lang,
                        current_total,
                        json.dumps(metadata, ensure_ascii=False),
                        job_id,
                    ),
                )
                conn.execute("COMMIT;")
                return

            cursor.execute(
                """
                INSERT INTO job_meta (
                    job_id, doc_id, source_path, target_lang, total_blocks, status, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    manifest.doc_id,
                    manifest.source_path,
                    manifest.target_lang,
                    0,
                    "initialized",
                    json.dumps(
                        {
                            "title": manifest.title,
                            "source_lang": manifest.source_lang,
                            "chapters": [c.model_dump() for c in manifest.chapters],
                            **manifest.run.to_metadata_dict(),
                            **manifest.metadata,
                        },
                        ensure_ascii=False,
                    ),
                ),
            )
            conn.execute("COMMIT;")

    def get_job_fingerprint(self, job_id: str) -> str | None:
        """Source-file fingerprint stored at ingest; None only when definitely absent.

        Read failures raise on purpose: the ingest resume path reads a missing
        fingerprint as "crashed mid-ingest" and clears every block in the job,
        so a swallowed sqlite error or a corrupt metadata_json must never
        masquerade as absence (that combination destroys fully translated
        books on resume).
        """
        with self._get_conn() as conn:
            actual_id = self._resolve_actual_job_id(conn, job_id)
            row = conn.execute(
                "SELECT metadata_json FROM job_meta WHERE job_id = ?", (actual_id,)
            ).fetchone()
            if not row:
                return None
            meta = json.loads(row["metadata_json"] or "{}")
            if not isinstance(meta, dict):
                raise LedgerError(f"job_meta.metadata_json for '{actual_id}' is not a JSON object")
            fp = meta.get("source_fingerprint")
            return str(fp) if fp else None

    def set_job_fingerprint(self, job_id: str, fingerprint: str) -> None:
        """Record the source-file fingerprint after a successful ingest."""
        self.set_job_metadata_value(job_id, "source_fingerprint", fingerprint)

    def get_job_metadata_value(self, job_id: str, key: str) -> Any | None:
        """Read one top-level key from the job's metadata_json (None when absent)."""
        with self._get_conn() as conn:
            try:
                actual_id = self._resolve_actual_job_id(conn, job_id)
            except sqlite3.Error as exc:
                # A missing/locked job_meta table reads as "no metadata"; that is
                # the intended None contract, but log it so a real DB error is
                # not indistinguishable from an absent key.
                logger.warning(
                    "get_job_metadata_value(%r): job_id resolution failed: %s", job_id, exc
                )
                return None
            row = conn.execute(
                "SELECT metadata_json FROM job_meta WHERE job_id = ?", (actual_id,)
            ).fetchone()
            if not row:
                return None
            try:
                meta = json.loads(row["metadata_json"] or "{}")
            except (TypeError, ValueError):
                return None
            if not isinstance(meta, dict):
                return None
            return meta.get(key)

    def get_job_metadata(self, job_id: str) -> dict[str, Any]:
        """Return a defensive copy of the job's top-level metadata object."""
        with self._get_conn() as conn:
            try:
                actual_id = self._resolve_actual_job_id(conn, job_id)
            except sqlite3.Error as exc:
                logger.warning("get_job_metadata(%r): job_id resolution failed: %s", job_id, exc)
                return {}
            row = conn.execute(
                "SELECT metadata_json FROM job_meta WHERE job_id = ?", (actual_id,)
            ).fetchone()
            if not row:
                return {}
            try:
                meta = json.loads(row["metadata_json"] or "{}")
            except (TypeError, ValueError):
                return {}
            if not isinstance(meta, dict):
                return {}
            return {str(key): value for key, value in meta.items()}

    def set_job_metadata_value(self, job_id: str, key: str, value: Any) -> None:
        """Write one top-level key into the job's metadata_json.

        The read-modify-write runs inside ``BEGIN IMMEDIATE`` using SQLite's native
        ``json_set`` to eliminate Python-level full deserialization/serialization
        amplification across multi-megabyte job metadata.

        A failed write is surfaced, not swallowed: ``source_fingerprint`` in
        particular must never be left silently missing, because the next resume
        reads an absent fingerprint as "ingest never finished" and clears every
        already-billed block to re-ingest it.
        """
        val_json = json.dumps(value, ensure_ascii=False)
        with self._get_conn() as conn:
            actual_id = self._resolve_actual_job_id(conn, job_id)
            conn.execute("BEGIN IMMEDIATE;")
            row = conn.execute(
                "SELECT metadata_json FROM job_meta WHERE job_id = ?", (actual_id,)
            ).fetchone()
            if not row:
                conn.execute("ROLLBACK;")
                raise LedgerError(
                    f"cannot set metadata key '{key}': job '{actual_id}' has no "
                    "job_meta row (was the job initialized?)"
                )
            raw = row["metadata_json"]
            if raw:
                try:
                    meta = json.loads(raw)
                    if not isinstance(meta, dict):
                        conn.execute("ROLLBACK;")
                        raise LedgerError(
                            f"job_meta.metadata_json for '{actual_id}' is not a JSON object"
                        )
                except (TypeError, ValueError) as exc:
                    conn.execute("ROLLBACK;")
                    raise LedgerError(
                        f"job_meta.metadata_json for '{actual_id}' is corrupt; "
                        "refusing to rewrite it (resume with a restored ledger "
                        "or start over with --fresh)"
                    ) from exc
            path = "$." + json.dumps(key)
            conn.execute(
                "UPDATE job_meta "
                "   SET metadata_json = json_set(COALESCE(NULLIF(metadata_json, ''), '{}'), ?, json(?)) "
                " WHERE job_id = ?",
                (path, val_json, actual_id),
            )
            conn.commit()

    def get_job_usage(self, job_id: str) -> dict[str, dict[str, int]]:
        """Per-model token totals billed over the whole life of the job.

        Empty when nothing has been recorded. This is the only place a resumed
        job's earlier sittings are still counted: provider counters live in
        process memory and start at zero on every launch, so without it the
        report of a book finished in N runs priced only the N-th one.
        """
        return _as_usage_totals(self.get_job_metadata_value(job_id, "usage_totals"))

    def record_job_usage(self, job_id: str, totals_by_model: dict[str, dict[str, int]]) -> None:
        """Store the job's lifetime usage totals (absolute, not incremental).

        Written once per priced progress event, so a run that dies mid-book
        still left what it spent on disk. Absolute on purpose: an incremental
        add would double-count as soon as the same event was emitted twice (a
        retry after a cancelled write), and the caller already holds the exact
        figure it wants recorded.
        """
        self.set_job_metadata_value(job_id, "usage_totals", _as_usage_totals(totals_by_model))

    def atomic_increment_job_usage(
        self, job_id: str, increment: dict[str, dict[str, int]]
    ) -> dict[str, dict[str, int]]:
        """Atomically merge increment into job_meta.usage_totals within a single transaction."""
        with self._get_conn() as conn:
            actual_id = self._resolve_actual_job_id(conn, job_id)
            conn.execute("BEGIN IMMEDIATE;")
            row = conn.execute(
                "SELECT metadata_json FROM job_meta WHERE job_id = ?", (actual_id,)
            ).fetchone()
            if not row:
                conn.execute("ROLLBACK;")
                raise LedgerError(
                    f"cannot increment usage: job '{actual_id}' has no "
                    "job_meta row (was the job initialized?)"
                )
            raw = row["metadata_json"]
            prior: dict[str, dict[str, int]] = {}
            if raw:
                try:
                    meta = json.loads(raw)
                    if not isinstance(meta, dict):
                        conn.execute("ROLLBACK;")
                        raise LedgerError(
                            f"job_meta.metadata_json for '{actual_id}' is not a JSON object"
                        )
                    if "usage_totals" in meta:
                        prior = _as_usage_totals(meta["usage_totals"])
                except (TypeError, ValueError) as exc:
                    conn.execute("ROLLBACK;")
                    raise LedgerError(
                        f"job_meta.metadata_json for '{actual_id}' is corrupt; "
                        "refusing to rewrite it"
                    ) from exc
            lifetime = merge_usage_totals(prior, increment)
            val_json = json.dumps(_as_usage_totals(lifetime), ensure_ascii=False)
            conn.execute(
                "UPDATE job_meta "
                "   SET metadata_json = json_set(COALESCE(NULLIF(metadata_json, ''), '{}'), '$.usage_totals', json(?)) "
                " WHERE job_id = ?",
                (val_json, actual_id),
            )
            conn.commit()
            return lifetime

    def get_job_status(self, job_id: str) -> str | None:
        """The raw ``job_meta.status`` column, or ``None`` when never written.

        The lifecycle is ``initialized`` (written at ingest) → a terminal
        ``completed``/``failed``/``cancelled`` (written by ``finalize_job``, the
        pipeline's abort handler, or the worker's terminal write). Note there is
        no ``running`` value: a row stays ``initialized`` for the whole run, so
        this column alone cannot tell a live job from one whose process died
        mid-run. Every outward surface therefore normalizes before answering —
        REST's disk fallback downgrades a non-terminal row to ``failed``
        (embedded mode has no reconciler), and MCP's ``ubt_status`` downgrades
        only after the ledger goes quiet (``_ledger_is_stale``), so a live run
        in another process still reads as running. Callers that need the
        normalized vocabulary should follow those precedents rather than
        inventing a fourth mapping.

        ``None`` for unknown or never-finalized jobs — lets disk-fallback status
        readers distinguish a genuinely failed job from one still running
        (otherwise a FAILED job could read as 'running' forever).
        """
        with self._get_conn() as conn:
            try:
                actual_id = self._resolve_actual_job_id(conn, job_id)
            except sqlite3.Error as exc:
                logger.warning("get_job_status(%r): job_id resolution failed: %s", job_id, exc)
                return None
            row = conn.execute(
                "SELECT status FROM job_meta WHERE job_id = ?", (actual_id,)
            ).fetchone()
            if not row:
                return None
            stored = row["status"]
            return str(stored) if stored else None

    def get_job_target_lang(self, job_id: str) -> str | None:
        """Target language stored at ingest (None if unspecified)."""
        with self._get_conn() as conn:
            try:
                actual_id = self._resolve_actual_job_id(conn, job_id)
            except sqlite3.Error as exc:
                logger.warning("get_job_target_lang(%r): job_id resolution failed: %s", job_id, exc)
                return None
            row = conn.execute(
                "SELECT target_lang FROM job_meta WHERE job_id = ?", (actual_id,)
            ).fetchone()
            if not row:
                return None
            stored = row["target_lang"]
            return str(stored) if stored else None

    def get_job_source_path(self, job_id: str) -> str | None:
        """Source document path stored at ingest (None if unspecified).

        The delivery render reads this PDF for figure/formula crops, so the L3
        single-page preview needs the same path back from the ledger.
        """
        with self._get_conn() as conn:
            try:
                actual_id = self._resolve_actual_job_id(conn, job_id)
            except sqlite3.Error as exc:
                logger.warning("get_job_source_path(%r): job_id resolution failed: %s", job_id, exc)
                return None
            row = conn.execute(
                "SELECT source_path FROM job_meta WHERE job_id = ?", (actual_id,)
            ).fetchone()
            if not row:
                return None
            stored = row["source_path"]
            return str(stored) if stored else None

    def resolve_job_id(self, job_id: str) -> str | None:
        """The stored job_id this handle resolves ``job_id`` to (exact id, or the
        latest job for a doc_id), or ``None`` when no such job exists here.

        Used by PE re-import to verify a queue file belongs to this ledger.
        """
        with self._get_conn() as conn:
            row = conn.execute("SELECT job_id FROM job_meta WHERE job_id = ?", (job_id,)).fetchone()
            if row:
                return str(row["job_id"])
            doc_row = conn.execute(
                "SELECT job_id FROM job_meta WHERE doc_id = ? "
                "ORDER BY created_at DESC, rowid DESC LIMIT 1",
                (job_id,),
            ).fetchone()
            return str(doc_row["job_id"]) if doc_row else None

    def clear_job_blocks(self, job_id: str) -> int:
        """Delete all blocks of a job for a --fresh re-ingest; returns the count."""
        with self._get_conn() as conn:
            try:
                actual_id = self._resolve_actual_job_id(conn, job_id)
            except sqlite3.Error as exc:
                logger.warning("clear_job_blocks(%r): job_id resolution failed: %s", job_id, exc)
                return 0
            conn.execute("BEGIN IMMEDIATE;")
            cursor = conn.execute("DELETE FROM blocks WHERE job_id = ?", (actual_id,))
            removed = cursor.rowcount or 0
            conn.execute(
                "UPDATE job_meta SET status = 'initialized', total_blocks = 0 WHERE job_id = ?",
                (actual_id,),
            )
            # Fresh re-ingest invalidates derived per-job caches: block texts may
            # change, so a cached bible / memory state must not be assumed to
            # describe them. The artifact paths belong in the same list --
            # they are metadata written by the previous run, and the API's
            # ``_artifact_path`` fallback serves any ``output_file`` it finds on
            # disk, so leaving them would let ``--fresh`` hand out the PDF whose
            # provenance it just deleted.
            #
            # ``source_fingerprint`` is the load-bearing one. The resume guard in
            # ingest treats a *present* fingerprint as proof that a previous
            # ingest ran to completion, so a ``--fresh`` run that dies partway
            # through re-parsing would leave the completed run's hash beside a
            # half-loaded block set: no mismatch to raise, no incomplete-ingest
            # clear to trigger, and parsing skipped because total != 0 -- a
            # truncated book exported as finished. Dropping it here restores the
            # invariant that a missing fingerprint reliably means "ingest never
            # finished", which is exactly what the next resume then re-ingests.
            for cache_key in (
                "bible_cache",
                "memory_state",
                "output_file",
                "report_file",
                "visual_report",
                "visual_report_file",
                "source_fingerprint",
            ):
                conn.execute(
                    """
                    UPDATE job_meta
                       SET metadata_json = json_remove(metadata_json, ?)
                     WHERE job_id = ?
                    """,
                    (f"$.{cache_key}", actual_id),
                )
            conn.commit()
            self._mark_blocks_changed()
            return removed

    def get_job_stats(self, job_id: str) -> dict[str, Any]:
        """Calculate aggregated progress statistics strictly for a single job.

        Reads on its own connection (see ``_get_read_conn``): this is the query
        behind every progress event — 666 of them on a 20k-block run at the
        default ``batch_limit`` — and running it behind the writer's lock made
        each one wait out an in-flight full scan (3 ms uncontended, 384 ms p95
        under contention, measured).
        """
        with self._get_read_conn() as conn:
            actual_id = self._resolve_actual_job_id(conn, job_id)
            # One read snapshot across both queries: without it a concurrent
            # checkpoint write can make total/completed disagree with the scores.
            conn.execute("BEGIN;")
            try:
                cursor = conn.execute(
                    f"""
                    SELECT
                        COUNT(*) as total_all,
                        SUM(CASE WHEN COALESCE(skip_translate, 0) = 1 THEN 1 ELSE 0 END) as skipped,
                        SUM(CASE WHEN COALESCE(skip_translate, 0) = 0 THEN 1 ELSE 0 END) as total,
                        SUM(CASE WHEN status IN ('mtqe_passed', 'repaired')
                                  AND COALESCE(skip_translate, 0) = 0 THEN 1 ELSE 0 END) as completed,
                        SUM(CASE WHEN status = 'drafted' THEN 1 ELSE 0 END) as drafted,
                        SUM(CASE WHEN status = 'repaired' THEN 1 ELSE 0 END) as repaired,
                        SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                        SUM(CASE WHEN status = 'needs_human' THEN 1 ELSE 0 END) as needs_human,
                        SUM(CASE WHEN status = 'blocked_human' THEN 1 ELSE 0 END) as blocked_human,
                        AVG(CASE WHEN {QE_SCORED_SQL} AND COALESCE(skip_translate, 0) = 0
                                 THEN mtqe_score ELSE NULL END) as avg_score
                    FROM blocks
                    WHERE job_id = ?
                    """,  # noqa: S608 — policy constant, not user input
                    (actual_id,),
                )
                row = cursor.fetchone()
                b15_avg = 0.0
                if row and row["avg_score"] is not None:
                    scores_cursor = conn.execute(
                        f"""
                        SELECT mtqe_score FROM blocks
                        WHERE job_id = ? AND {QE_SCORED_SQL}
                        ORDER BY mtqe_score ASC
                        """,  # noqa: S608 — policy constant, not user input
                        (actual_id,),
                    )
                    scores = [
                        r["mtqe_score"]
                        for r in scores_cursor.fetchall()
                        if r["mtqe_score"] is not None
                    ]
                    if scores:
                        cutoff = max(1, int(len(scores) * 0.15))
                        b15_avg = round(sum(scores[:cutoff]) / cutoff, 4)
            finally:
                # Always end the read snapshot, including on a query error: a
                # connection left "within a transaction" pins the WAL and poisons
                # the next ``BEGIN`` on this thread. A read-only transaction is
                # released by either statement; COMMIT is the success path, and
                # ROLLBACK covers the failure one. Suppressed so a rollback
                # failure cannot mask the error that caused it.
                with suppress(sqlite3.Error):
                    conn.execute("ROLLBACK;" if conn.in_transaction else "COMMIT;")
            return {
                # ``total``/``completed`` count only blocks that could be
                # translated. Policy-preserved blocks (images, bibliography
                # entries, bylines, short headings — 154 of 259 on the reference
                # job) are marked ``mtqe_passed`` by the skip rule without any
                # work, so counting them inflated both sides of the fraction
                # equally: the panel reported "154/259 = 59.5%" for a job with 90
                # real translated blocks out of 105, and different books got
                # different-looking progress for identical work depending on how
                # many images they contained.
                "total": row["total"] or 0,
                "total_all": row["total_all"] or 0,
                "skipped": row["skipped"] or 0,
                "completed": row["completed"] or 0,
                "drafted": row["drafted"] or 0,
                "repaired": row["repaired"] or 0,
                "failed": row["failed"] or 0,
                "needs_human": row["needs_human"] or 0,
                "blocked_human": row["blocked_human"] or 0,
                "avg_qe_score": round(row["avg_score"], 4) if row["avg_score"] is not None else 0.0,
                # The avg above is a 0.0 SENTINEL when nothing was scored
                # (heuristic QE engine: skips and FastPass passes never reach a
                # scorer). Consumers must read this flag before displaying the
                # number — the status table and CLI summary both do — or a
                # placeholder zero reads as a measured failing score.
                "qe_measured": row["avg_score"] is not None,
                "bottom_15_avg_qe": b15_avg,
            }

    def get_total_blocks(self, job_id: str) -> int:
        """Return total number of content blocks recorded for this job."""
        return int(self.get_job_stats(job_id).get("total", 0) or 0)

    def get_job_snapshot(self, job_id: str) -> dict[str, Any] | None:
        """Return job_meta plus block stats, or None if the job is not in this ledger."""
        with self._get_conn() as conn:
            actual_id = self._resolve_actual_job_id(conn, job_id)
            meta = conn.execute(
                """
                SELECT job_id, doc_id, source_path, target_lang, status, created_at, updated_at
                FROM job_meta WHERE job_id = ?
                """,
                (actual_id,),
            ).fetchone()
            if meta is None:
                return None
            snapshot: dict[str, Any] = {
                "job_id": str(meta["job_id"]),
                "doc_id": str(meta["doc_id"]),
                "source_path": str(meta["source_path"]),
                "target_lang": str(meta["target_lang"]),
                "status": str(meta["status"]),
                "created_at": meta["created_at"],
                "updated_at": meta["updated_at"],
            }
        snapshot.update(self.get_job_stats(job_id))
        return snapshot

    def finalize_job(
        self,
        job_id: str,
        status: str = "completed",
        *,
        rendered_digest: str | None = None,
    ) -> None:
        """Mark job metadata as completed or terminated.

        Fail-closed on ``completed``: refuses when the job row is missing, when
        non-terminal blocks remain, or when the job is already terminal (a
        concurrent cancel/lease-loss must not be rewritten to a false success).
        ``failed``/``cancelled`` stay permissive by design — the abort path must
        land a terminal write even mid-flight.

        ``rendered_digest`` is written **in the same transaction** as the status
        flip. It used to be a separate ``set_job_metadata_value`` call *after*
        this one, which left a window where the job read ``completed`` with no
        digest at all. Nothing self-heals that state: ``is_delivery_stale``
        treats a missing digest as "not stale" (it has to, for jobs that
        predate the marker), so the download endpoint would hand out a file
        rendered from text the ledger no longer holds. Both writes now commit
        together or not at all.
        """
        with self._get_conn() as conn:
            actual_id = self._resolve_actual_job_id(conn, job_id)
            conn.execute("BEGIN IMMEDIATE;")
            row = conn.execute(
                "SELECT job_id, status FROM job_meta WHERE job_id = ?", (actual_id,)
            ).fetchone()
            if row is None:
                conn.execute("ROLLBACK;")
                raise LedgerError(f"finalize_job: unknown job '{job_id}'")
            existing = str(row["status"])
            if status == "failed" and existing in ("cancelled",):
                # A cancel is authoritative against a late abort: an export or
                # worker failure that arrives afterwards must not rewrite it.
                # Idempotent no-op (not an error) so stages do not crash after
                # losing the race.
                logger.warning(
                    "finalize_job('failed') ignored for '%s': already cancelled", actual_id
                )
                conn.execute("ROLLBACK;")
                return
            if status == "completed" and existing == "cancelled":
                # A rerun the user explicitly started after a cancel may finish:
                # keeping the job cancelled forever stranded delivered artifacts
                # under a dead status. Cancel stays authoritative only while
                # nothing new completed.
                logger.info(
                    "finalize_job('completed') supersedes a cancellation for '%s'", actual_id
                )
            elif status == "completed" and existing == "failed":
                # ``failed`` is not authoritative against a real completion: a
                # job marked failed by a stale owner (its lease was reclaimed)
                # or by a previous attempt that the user resumed can still reach
                # export. Refusing here left delivered jobs reading ``failed``
                # and made every resume re-run the paid export. To get this far
                # the pipeline must have passed the non-terminal-block check
                # below, i.e. the work really is done.
                logger.info("finalize_job('completed') supersedes a failure for '%s'", actual_id)
            elif status in ("failed", "cancelled") and existing == "completed":
                # The mirror race: a stale owner's abort must not re-mark a job
                # the reclaiming worker already completed. ``failed``/``cancelled``
                # stay permissive only against non-terminal rows.
                logger.warning(
                    "finalize_job(%r) ignored for '%s': already completed",
                    status,
                    actual_id,
                )
                conn.execute("ROLLBACK;")
                return
            if status == "completed":
                placeholders = ",".join("?" * len(NON_TERMINAL_STATUSES))
                stale = conn.execute(
                    f"""
                    SELECT block_id FROM blocks
                    WHERE job_id = ? AND status IN ({placeholders})
                    LIMIT 11
                    """,
                    (actual_id, *NON_TERMINAL_STATUSES),
                ).fetchall()
                if stale:
                    sample = ", ".join(str(r["block_id"]) for r in stale[:10])
                    raise LedgerError(
                        f"finalize_job refused 'completed' for '{actual_id}': "
                        f"{len(stale)} non-terminal block(s) remain (e.g. {sample}); "
                        "run export (fail_non_terminal_blocks) or resume first"
                    )
            cursor = conn.execute(
                "UPDATE job_meta SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE job_id = ?",
                (status, actual_id),
            )
            if cursor.rowcount == 0:
                raise LedgerError(f"finalize_job: unknown job '{job_id}'")
            if rendered_digest is not None:
                # Same transaction as the status flip: json_set keeps the rest
                # of metadata_json untouched, and a failed digest write rolls
                # the completion back rather than leaving it undeliverable.
                conn.execute(
                    "UPDATE job_meta "
                    "   SET metadata_json = json_set("
                    "       COALESCE(NULLIF(metadata_json, ''), '{}'), ?, json(?)) "
                    " WHERE job_id = ?",
                    (
                        "$." + json.dumps(RENDERED_TEXT_DIGEST_KEY),
                        json.dumps(rendered_digest, ensure_ascii=False),
                        actual_id,
                    ),
                )
            conn.execute("COMMIT;")

    def record_visual_report(self, job_id: str, report: dict[str, Any]) -> None:
        """Persist visual gate results into job_meta.metadata_json."""
        # One writer per metadata key: go through set_job_metadata_value, which
        # reads and writes inside BEGIN IMMEDIATE (reading the preimage outside
        # its own transaction is the lost-update shape that guard prevents).
        self.set_job_metadata_value(job_id, "visual_report", report)

    def get_visual_report(self, job_id: str) -> dict[str, Any] | None:
        """Retrieve visual gate report from job_meta.metadata_json if present."""
        with self._get_conn() as conn:
            actual_id = self._resolve_actual_job_id(conn, job_id)
            cursor = conn.execute(
                "SELECT metadata_json FROM job_meta WHERE job_id = ?", (actual_id,)
            )
            row = cursor.fetchone()
            if row and row["metadata_json"]:
                try:
                    meta = json.loads(row["metadata_json"])
                    if isinstance(meta, dict) and "visual_report" in meta:
                        res = meta["visual_report"]
                        if isinstance(res, dict):
                            return cast(dict[str, Any], res)
                except Exception as exc:
                    # Corrupt metadata must not break the read, but a torn
                    # value and an absent visual report should not look
                    # identical to whoever is debugging the job.
                    logger.debug("Unreadable visual_report metadata for %s: %s", job_id, exc)
                    return None
            return None


class LedgerBlocksMixin(_LedgerHost):
    """Block writes, checkpoints and status/type/chapter reads."""

    def claim_next_agent_work(
        self,
        job_id: str,
        *,
        package_id: str,
        lease_token_hash: str,
        expires_at: float,
        now: float,
        limit: int,
        package_state: dict[str, Any] | None = None,
    ) -> tuple[list[IRBlock], dict[str, Any]]:
        """Select and lease the next Agent package in one transaction.

        Selection and the lease must share one transaction: two concurrent
        claims could otherwise read the same pending rows before either write
        committed. This method performs the read and the ``BEGIN IMMEDIATE``
        write under one connection lock, so at most one claim can win a block.

        Repair candidates take precedence over fresh blocks, and a package is
        homogeneous: when any repair is pending the claim contains only repairs;
        otherwise it contains only fresh blocks. Mixing the two would make a
        single prompt kind and ``repair_rounds`` accounting wrong for part of
        the package. The returned blocks are the rows as they were before being
        marked ``drafted``; their status is persisted as ``expected_statuses``
        so lease expiry can restore the correct pre-claim state.
        """
        if limit < 1:
            raise LedgerError("Agent claim limit must be positive")
        with self._get_conn() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            actual_id = self._resolve_actual_job_id(conn, job_id)
            row = conn.execute(
                "SELECT metadata_json FROM job_meta WHERE job_id = ?", (actual_id,)
            ).fetchone()
            if row is None:
                conn.execute("ROLLBACK;")
                raise LedgerError(f"cannot claim work: unknown job '{job_id}'")
            try:
                meta = json.loads(row["metadata_json"] or "{}")
            except (TypeError, ValueError) as exc:
                conn.execute("ROLLBACK;")
                raise LedgerError(
                    f"job_meta.metadata_json for '{actual_id}' is corrupt; "
                    "refusing to claim Agent work"
                ) from exc
            if not isinstance(meta, dict):
                conn.execute("ROLLBACK;")
                raise LedgerError(f"job_meta.metadata_json for '{actual_id}' is not a JSON object")
            session = meta.get(AGENT_SESSION_KEY)
            if isinstance(session, dict) and session.get("state") == "finishing":
                conn.execute("ROLLBACK;")
                raise LedgerError(
                    f"Agent job {job_id} is finishing; wait for ubt_agent_finish to return"
                )
            status = BlockStatus.REPAIR_PENDING
            candidates = conn.execute(
                """
                SELECT * FROM blocks
                 WHERE job_id = ? AND skip_translate = 0 AND status = ?
                 ORDER BY spine_index ASC, block_id ASC
                 LIMIT ?
                """,
                (actual_id, status.value, int(limit)),
            ).fetchall()
            if not candidates:
                status = BlockStatus.PENDING
                candidates = conn.execute(
                    """
                    SELECT * FROM blocks
                     WHERE job_id = ? AND skip_translate = 0 AND status = ?
                     ORDER BY spine_index ASC, block_id ASC
                     LIMIT ?
                    """,
                    (actual_id, status.value, int(limit)),
                ).fetchall()
            blocks = [self._row_to_block(r) for r in candidates]
            if not blocks:
                conn.execute("ROLLBACK;")
                return [], {}
            expected_by_block = {block.id: block.status.value for block in blocks}
            if not isinstance(session, dict):
                session = {}
            packages = session.get("packages")
            if not isinstance(packages, dict):
                packages = {}
            package = {
                "package_id": package_id,
                "lease_token_hash": lease_token_hash,
                "expires_at": float(expires_at),
                "claimed_at": float(now),
                "block_ids": [block.id for block in blocks],
                "expected_statuses": expected_by_block,
                "kind": "repair" if status is BlockStatus.REPAIR_PENDING else "translate",
                "state": "claimed",
                **(package_state or {}),
            }
            packages[package_id] = package
            session["packages"] = packages
            session["state"] = "awaiting_agent"
            session["updated_at"] = float(now)
            meta[AGENT_SESSION_KEY] = session
            conn.execute(
                "UPDATE job_meta SET metadata_json = ?, updated_at = CURRENT_TIMESTAMP "
                "WHERE job_id = ?",
                (json.dumps(meta, ensure_ascii=False), actual_id),
            )
            placeholders = ",".join("?" for _ in blocks)
            conn.execute(
                f"UPDATE blocks SET status = ?, updated_at = CURRENT_TIMESTAMP "
                f"WHERE job_id = ? AND block_id IN ({placeholders})",
                (BlockStatus.DRAFTED.value, actual_id, *[block.id for block in blocks]),
            )
            conn.execute("COMMIT;")
            self._mark_blocks_changed()
            return blocks, package

    def submit_agent_work(
        self,
        job_id: str,
        *,
        package_id: str,
        lease_token_hash: str,
        now: float,
        updates: list[dict[str, Any]],
        package_state: dict[str, Any] | None = None,
        submission_hash: str | None = None,
    ) -> dict[str, Any]:
        """Atomically accept one Agent submission.

        The lease is checked against the stored digest; every block must belong
        to the package and still be ``drafted`` (the claim marker). The package
        becomes ``submitted`` in the same transaction as the block updates, so a
        crash can never leave translated blocks attached to a still-claimable
        package.

        Repeating an identical submission is idempotent: the package is already
        ``submitted`` and the method returns its stored state. A different
        submission for that package is a conflict.
        """
        with self._get_conn() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            actual_id = self._resolve_actual_job_id(conn, job_id)
            row = conn.execute(
                "SELECT metadata_json FROM job_meta WHERE job_id = ?", (actual_id,)
            ).fetchone()
            if row is None:
                conn.execute("ROLLBACK;")
                raise LedgerError(f"cannot submit work: unknown job '{job_id}'")
            try:
                meta = json.loads(row["metadata_json"] or "{}")
            except (TypeError, ValueError) as exc:
                conn.execute("ROLLBACK;")
                raise LedgerError(
                    f"job_meta.metadata_json for '{actual_id}' is corrupt; "
                    "refusing to accept Agent work"
                ) from exc
            if not isinstance(meta, dict):
                conn.execute("ROLLBACK;")
                raise LedgerError(f"job_meta.metadata_json for '{actual_id}' is not a JSON object")
            session = meta.get(AGENT_SESSION_KEY)
            packages = session.get("packages") if isinstance(session, dict) else None
            package = packages.get(package_id) if isinstance(packages, dict) else None
            if not isinstance(package, dict):
                conn.execute("ROLLBACK;")
                raise LedgerError(f"unknown Agent package: {package_id}")
            if not isinstance(session, dict):
                conn.execute("ROLLBACK;")
                raise LedgerError("Agent session metadata is not an object")
            if not isinstance(packages, dict):
                conn.execute("ROLLBACK;")
                raise LedgerError("Agent package map is not an object")
            state = str(package.get("state") or "")
            if session.get("state") == "finishing" and state != "submitted":
                conn.execute("ROLLBACK;")
                raise LedgerError(
                    f"Agent job {job_id} is finishing; this package can no longer be submitted"
                )
            if package.get("lease_token_hash") != lease_token_hash:
                conn.execute("ROLLBACK;")
                raise LedgerError("Agent lease token does not match this package")
            if state == "submitted":
                if submission_hash and package.get("submission_hash") != submission_hash:
                    conn.execute("ROLLBACK;")
                    raise LedgerError(
                        f"Agent package {package_id} was already submitted with different text"
                    )
                conn.execute("ROLLBACK;")
                return package
            if state != "claimed":
                conn.execute("ROLLBACK;")
                raise LedgerError(
                    f"Agent package {package_id} is {state or 'unknown'}, not claimable for submit"
                )
            expires_at = float(package.get("expires_at") or 0.0)
            if now > expires_at:
                conn.execute("ROLLBACK;")
                raise LedgerError(
                    f"Agent package {package_id} lease expired at {expires_at:.0f}; "
                    "claim a fresh package"
                )
            package_blocks = [str(b) for b in package.get("block_ids", [])]
            update_ids = [str(item.get("block_id")) for item in updates]
            if sorted(update_ids) != sorted(package_blocks):
                conn.execute("ROLLBACK;")
                raise LedgerError(
                    "Agent submission must update exactly the package's blocks "
                    f"(expected {len(package_blocks)}, got {len(update_ids)})"
                )
            placeholders = ",".join("?" for _ in package_blocks)
            rows = conn.execute(
                f"SELECT block_id, status FROM blocks "
                f"WHERE job_id = ? AND block_id IN ({placeholders})",
                (actual_id, *package_blocks),
            ).fetchall()
            statuses = {str(r["block_id"]): str(r["status"]) for r in rows}
            wrong = {
                block_id: status
                for block_id, status in statuses.items()
                if status != BlockStatus.DRAFTED.value
            }
            if wrong:
                conn.execute("ROLLBACK;")
                raise LedgerError(
                    "Agent package blocks changed while it was leased: "
                    + ", ".join(f"{b}={s}" for b, s in sorted(wrong.items())[:10])
                )
            for item in updates:
                fields = ["status = ?", "target_text = ?", "updated_at = CURRENT_TIMESTAMP"]
                params: list[Any] = [
                    (
                        item["status"].value
                        if isinstance(item["status"], BlockStatus)
                        else str(item["status"])
                    ),
                    item.get("target_text"),
                ]
                optional = {
                    "draft_text": "draft_text = ?",
                    "mtqe_score": "mtqe_score = ?",
                    "repair_rounds": "repair_rounds = ?",
                    "mqm_severity": "mqm_severity = ?",
                }
                for key, clause in optional.items():
                    if key in item:
                        fields.append(clause)
                        params.append(item[key])
                if "error_flags" in item:
                    fields.append("error_flags_json = ?")
                    params.append(json.dumps(item["error_flags"], ensure_ascii=False))
                if "mqm_spans" in item:
                    fields.append("mqm_spans_json = ?")
                    params.append(json.dumps(item["mqm_spans"], ensure_ascii=False))
                if "style" in item:
                    fields.append("style_json = ?")
                    params.append(_style_json(item["style"]))
                params.extend([item["block_id"], actual_id])
                conn.execute(
                    f"UPDATE blocks SET {', '.join(fields)} WHERE block_id = ? AND job_id = ?",
                    params,
                )
            package.update(package_state or {})
            if submission_hash:
                package["submission_hash"] = submission_hash
            package["state"] = "submitted"
            package["submitted_at"] = float(now)
            packages[package_id] = package
            session["packages"] = packages
            session["state"] = "awaiting_agent"
            session["updated_at"] = float(now)
            meta[AGENT_SESSION_KEY] = session
            conn.execute(
                "UPDATE job_meta SET metadata_json = ?, updated_at = CURRENT_TIMESTAMP "
                "WHERE job_id = ?",
                (json.dumps(meta, ensure_ascii=False), actual_id),
            )
            conn.execute("COMMIT;")
            self._mark_blocks_changed()
            return package

    def agent_session_packages(self, job_id: str) -> dict[str, dict[str, Any]]:
        """Return the Agent package map as a defensive copy."""
        raw = cast("_LedgerHostWithMetadata", self).get_job_metadata_value(
            job_id, AGENT_SESSION_KEY
        )
        if not isinstance(raw, dict):
            return {}
        packages = raw.get("packages")
        if not isinstance(packages, dict):
            return {}
        return {
            str(package_id): dict(package)
            for package_id, package in packages.items()
            if isinstance(package, dict)
        }

    def expire_agent_packages(self, job_id: str, *, now: float) -> list[str]:
        """Release expired, unsubmitted Agent leases and re-pend their blocks.

        Expiry is a maintenance operation, not a state transition the Agent can
        perform: a package that was leased but never submitted must become
        claimable again. The block reset and lease release commit together.
        """
        expired: list[str] = []
        with self._get_conn() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            actual_id = self._resolve_actual_job_id(conn, job_id)
            row = conn.execute(
                "SELECT metadata_json FROM job_meta WHERE job_id = ?", (actual_id,)
            ).fetchone()
            if row is None:
                conn.execute("ROLLBACK;")
                return []
            try:
                meta = json.loads(row["metadata_json"] or "{}")
            except (TypeError, ValueError):
                conn.execute("ROLLBACK;")
                return []
            if not isinstance(meta, dict):
                conn.execute("ROLLBACK;")
                return []
            session = meta.get(AGENT_SESSION_KEY)
            packages = session.get("packages") if isinstance(session, dict) else None
            if not isinstance(packages, dict):
                conn.execute("ROLLBACK;")
                return []
            if not isinstance(session, dict):
                conn.execute("ROLLBACK;")
                return []
            for package_id, package in list(packages.items()):
                if not isinstance(package, dict) or package.get("state") != "claimed":
                    continue
                if float(package.get("expires_at") or 0.0) >= now:
                    continue
                block_ids = [str(b) for b in package.get("block_ids", [])]
                if block_ids:
                    expected = package.get("expected_statuses")
                    for block_id in block_ids:
                        restore = (
                            str(expected.get(block_id))
                            if isinstance(expected, dict) and expected.get(block_id)
                            else BlockStatus.PENDING.value
                        )
                        conn.execute(
                            "UPDATE blocks SET status = ?, updated_at = CURRENT_TIMESTAMP "
                            "WHERE job_id = ? AND block_id = ? AND status = ?",
                            (restore, actual_id, block_id, BlockStatus.DRAFTED.value),
                        )
                package["state"] = "expired"
                package["expired_at"] = float(now)
                expired.append(package_id)
            if expired:
                session["packages"] = packages
                session["updated_at"] = float(now)
                meta[AGENT_SESSION_KEY] = session
                conn.execute(
                    "UPDATE job_meta SET metadata_json = ?, updated_at = CURRENT_TIMESTAMP "
                    "WHERE job_id = ?",
                    (json.dumps(meta, ensure_ascii=False), actual_id),
                )
                conn.execute("COMMIT;")
                self._mark_blocks_changed()
                return expired
            conn.execute("ROLLBACK;")
            return []

    def begin_agent_finish(self, job_id: str, *, now: float) -> list[str]:
        """Expire stale leases and atomically enter the finishing state.

        Returns the ids of packages that are still actively leased. When that
        list is non-empty the session stays ``awaiting_agent`` and no caller may
        screen or render its drafted blocks. When it is empty the session becomes
        ``finishing`` in the same transaction that released every expired
        package, so a concurrent claim cannot slip in between the check and the
        quality screen.
        """
        active: list[str] = []
        with self._get_conn() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            actual_id = self._resolve_actual_job_id(conn, job_id)
            row = conn.execute(
                "SELECT metadata_json FROM job_meta WHERE job_id = ?", (actual_id,)
            ).fetchone()
            if row is None:
                conn.execute("ROLLBACK;")
                raise LedgerError(f"cannot finish: unknown job '{job_id}'")
            try:
                meta = json.loads(row["metadata_json"] or "{}")
            except (TypeError, ValueError) as exc:
                conn.execute("ROLLBACK;")
                raise LedgerError(
                    f"job_meta.metadata_json for '{actual_id}' is corrupt; "
                    "refusing to finish Agent work"
                ) from exc
            if not isinstance(meta, dict):
                conn.execute("ROLLBACK;")
                raise LedgerError(f"job_meta.metadata_json for '{actual_id}' is not a JSON object")
            session = meta.get(AGENT_SESSION_KEY)
            if not isinstance(session, dict):
                conn.execute("ROLLBACK;")
                raise LedgerError(f"Agent job {job_id} has no Agent session")
            packages = session.get("packages")
            if not isinstance(packages, dict):
                conn.execute("ROLLBACK;")
                raise LedgerError(f"Agent job {job_id} has a malformed package map")
            for package_id, package in list(packages.items()):
                if not isinstance(package, dict) or package.get("state") != "claimed":
                    continue
                if float(package.get("expires_at") or 0.0) >= now:
                    active.append(str(package_id))
                    continue
                block_ids = [str(block_id) for block_id in package.get("block_ids", [])]
                expected = package.get("expected_statuses")
                for block_id in block_ids:
                    restore = (
                        str(expected.get(block_id))
                        if isinstance(expected, dict) and expected.get(block_id)
                        else BlockStatus.PENDING.value
                    )
                    conn.execute(
                        "UPDATE blocks SET status = ?, updated_at = CURRENT_TIMESTAMP "
                        "WHERE job_id = ? AND block_id = ? AND status = ?",
                        (restore, actual_id, block_id, BlockStatus.DRAFTED.value),
                    )
                package["state"] = "expired"
                package["expired_at"] = float(now)
            session["packages"] = packages
            session["state"] = "awaiting_agent" if active else "finishing"
            session["updated_at"] = float(now)
            meta[AGENT_SESSION_KEY] = session
            conn.execute(
                "UPDATE job_meta SET metadata_json = ?, updated_at = CURRENT_TIMESTAMP "
                "WHERE job_id = ?",
                (json.dumps(meta, ensure_ascii=False), actual_id),
            )
            conn.execute("COMMIT;")
            self._mark_blocks_changed()
            return active

    def set_agent_session_state(self, job_id: str, state: str, *, now: float) -> None:
        """Update only the durable Agent state, preserving packages and leases."""
        with self._get_conn() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            actual_id = self._resolve_actual_job_id(conn, job_id)
            row = conn.execute(
                "SELECT metadata_json FROM job_meta WHERE job_id = ?", (actual_id,)
            ).fetchone()
            if row is None:
                conn.execute("ROLLBACK;")
                raise LedgerError(f"cannot update Agent state: unknown job '{job_id}'")
            try:
                meta = json.loads(row["metadata_json"] or "{}")
            except (TypeError, ValueError) as exc:
                conn.execute("ROLLBACK;")
                raise LedgerError(f"job_meta.metadata_json for '{actual_id}' is corrupt") from exc
            if not isinstance(meta, dict):
                conn.execute("ROLLBACK;")
                raise LedgerError(f"job_meta.metadata_json for '{actual_id}' is not a JSON object")
            session = meta.get(AGENT_SESSION_KEY)
            if not isinstance(session, dict):
                session = {}
            session["state"] = state
            session["updated_at"] = float(now)
            meta[AGENT_SESSION_KEY] = session
            conn.execute(
                "UPDATE job_meta SET metadata_json = ?, updated_at = CURRENT_TIMESTAMP "
                "WHERE job_id = ?",
                (json.dumps(meta, ensure_ascii=False), actual_id),
            )
            conn.execute("COMMIT;")

    def append_chapter(self, job_id: str, chapter: ChapterIR) -> None:
        """Atomically append a ChapterIR partition using non-destructive ON CONFLICT update."""
        if not chapter.blocks:
            return

        with self._get_conn() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            cursor = conn.cursor()
            actual_id = self._resolve_actual_job_id(conn, job_id)

            _upsert_blocks_batch(cursor, actual_id, chapter.blocks)

            cursor.execute(
                """
                UPDATE job_meta
                SET total_blocks = (SELECT COUNT(*) FROM blocks WHERE job_id = ?),
                    updated_at = CURRENT_TIMESTAMP
                WHERE job_id = ?
                """,
                (actual_id, actual_id),
            )
            conn.execute("COMMIT;")
            self._mark_blocks_changed()

    def fetch_pending_blocks(
        self,
        job_id: str,
        limit: int = 50,
        after_spine_index: int | None = None,
        after_block_id: str | None = None,
        chapter_id: str | None = None,
    ) -> list[IRBlock]:
        """Fetch pending blocks using Keyset Pagination (spine_index, block_id).

        Ordering is total on ``(spine_index, block_id)``. When both
        ``after_spine_index`` and ``after_block_id`` are provided, SQLite row-value
        comparison ``(spine_index, block_id) > (?, ?)`` is used, avoiding batch
        limit blowout while never skipping blocks that share the same spine_index.
        When chapter_id is specified, only pending blocks belonging to that chapter are fetched.
        """
        base = "job_id = ? AND status = 'pending' AND skip_translate = 0"
        chapter_clause = ""
        chapter_params: tuple[Any, ...] = ()
        if chapter_id is not None:
            chapter_clause, chapter_params = _chapter_match(chapter_id)
        full_base = f"{base}{chapter_clause}"
        with self._get_conn() as conn:
            actual_id = self._resolve_actual_job_id(conn, job_id)
            if after_spine_index is not None and after_block_id is not None:
                rows = conn.execute(
                    f"""
                    SELECT * FROM blocks
                    WHERE {full_base} AND (spine_index, block_id) > (?, ?)
                    ORDER BY spine_index ASC, block_id ASC
                    LIMIT ?
                    """,
                    (actual_id, *chapter_params, after_spine_index, after_block_id, limit),
                ).fetchall()
            elif after_spine_index is not None:
                rows = conn.execute(
                    f"""
                    SELECT * FROM blocks
                    WHERE {full_base} AND spine_index > ?
                    ORDER BY spine_index ASC, block_id ASC
                    LIMIT ?
                    """,
                    (actual_id, *chapter_params, after_spine_index, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    f"""
                    SELECT * FROM blocks
                    WHERE {full_base}
                    ORDER BY spine_index ASC, block_id ASC
                    LIMIT ?
                    """,
                    (actual_id, *chapter_params, limit),
                ).fetchall()

            return [self._row_to_block(r) for r in rows]

    def fail_non_terminal_blocks(self, job_id: str, reason: str) -> list[str]:
        """Force every non-terminal block of a job into FAILED status.

        Export-time safety net: blocks left in pending/
        drafted/repair_pending after all stages ran mean a stage crashed
        mid-flight. Rendering them as untranslated source silently
        is the failure mode this guard eliminates.
        """
        placeholders = ",".join("?" * len(NON_TERMINAL_STATUSES))
        with self._get_conn() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            actual_id = self._resolve_actual_job_id(conn, job_id)
            cursor = conn.execute(
                f"""
                SELECT block_id, error_flags_json FROM blocks
                WHERE job_id = ? AND status IN ({placeholders})
                """,
                (actual_id, *NON_TERMINAL_STATUSES),
            )
            rows = cursor.fetchall()
            failed_ids: list[str] = []
            for row in rows:
                flags: list[str] = []
                raw = row["error_flags_json"]
                if raw:
                    with suppress(json.JSONDecodeError):
                        loaded = json.loads(raw)
                        if isinstance(loaded, list):
                            flags = [str(f) for f in loaded]
                flags.append(reason)
                conn.execute(
                    """
                    UPDATE blocks
                    SET status = 'failed',
                        error_flags_json = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE block_id = ? AND job_id = ?
                    """,
                    (json.dumps(flags, ensure_ascii=False), row["block_id"], actual_id),
                )
                failed_ids.append(str(row["block_id"]))
            conn.execute("COMMIT;")
            self._mark_blocks_changed()
            return failed_ids

    def get_preceding_text_tail(
        self,
        job_id: str,
        flow_id: FlowID,
        before_spine_index: int,
        max_chars: int = 300,
    ) -> str:
        """Fetch the preceding excerpt in the same flow_id (cross-chapter bridge).

        The prose before a block is the prose the model is continuing, so this
        side carries the *translation* once one exists (reading ``source_text``
        here would hand the model source-language context for text it had
        already rendered, so register, terminology and sentence rhythm get
        re-decided per block instead of carried forward). The *following*
        excerpt stays source-side on purpose -- that text has not been
        translated yet, and it is only there to disambiguate a sentence cut in
        half.
        """
        # ``job_id`` may be a doc_id alias (the ledger resolves aliases to the
        # latest job for that document). Every other read resolves first, and
        # this one feeds the model's cross-chapter continuation prompt — an
        # unresolved alias here would splice another job's prose into it.
        with self._get_conn() as conn:
            actual_id = self._resolve_actual_job_id(conn, job_id)
            query = """
                SELECT COALESCE(NULLIF(target_text, ''), source_text) AS context_text
                  FROM blocks
                 WHERE job_id = ? AND flow_id = ? AND spine_index < ?
                 ORDER BY spine_index DESC
                 LIMIT 5
            """
            cursor = conn.execute(query, (actual_id, flow_id.value, before_spine_index))
            rows = cursor.fetchall()
            if not rows:
                return ""

            collected: list[str] = []
            char_count = 0
            for r in rows:
                text = r["context_text"]
                collected.insert(0, text)
                char_count += len(text)
                if char_count >= max_chars:
                    break

            combined = " ".join(collected)
            return combined[-max_chars:].strip() if len(combined) > max_chars else combined.strip()

    def get_following_text_head(
        self,
        job_id: str,
        flow_id: FlowID,
        after_spine_index: int,
        chapter_id: str | None = None,
        max_chars: int = 300,
    ) -> str:
        """Fetch following excerpt in the same flow_id (seamless cross-batch bridge within chapter)."""
        with self._get_conn() as conn:
            actual_id = self._resolve_actual_job_id(conn, job_id)
            if chapter_id:
                chapter_clause, chapter_params = _chapter_match(chapter_id)
                query = f"""
                    SELECT source_text FROM blocks
                    WHERE job_id = ? AND flow_id = ? AND spine_index > ?{chapter_clause}
                    ORDER BY spine_index ASC
                    LIMIT 5
                """
                params: tuple[Any, ...] = (
                    actual_id,
                    flow_id.value,
                    after_spine_index,
                    *chapter_params,
                )
            else:
                query = """
                    SELECT source_text FROM blocks
                    WHERE job_id = ? AND flow_id = ? AND spine_index > ?
                    ORDER BY spine_index ASC
                    LIMIT 5
                """
                params = (actual_id, flow_id.value, after_spine_index)

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            if not rows:
                return ""

            collected: list[str] = []
            char_count = 0
            for r in rows:
                text = r["source_text"]
                collected.append(text)
                char_count += len(text)
                if char_count >= max_chars:
                    break

            combined = " ".join(collected)
            return combined[:max_chars].strip() if len(combined) > max_chars else combined.strip()

    def save_checkpoint(
        self,
        block_id: str,
        status: BlockStatus,
        target_text: str | None = None,
        draft_text: str | None = None,
        mtqe_score: float | None = None,
        repair_rounds: int | None = None,
        error_flags: list[str] | None = None,
        *,
        tm_hit: bool | None = None,
        mqm_severity: str | None = None,
        mqm_spans: list[dict[str, Any]] | None = None,
        style: Any | None = None,
        job_id: str | None = None,
    ) -> bool:
        """Atomically commit progress checkpoint for a single block."""
        with self._get_conn() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            scope = self._block_scope(conn, job_id)
            updates: list[str] = [
                "status = ?",
                "updated_at = CURRENT_TIMESTAMP",
            ]
            params: list[Any] = [status.value]

            if target_text is not None:
                updates.append("target_text = ?")
                params.append(target_text)
            if draft_text is not None:
                updates.append("draft_text = ?")
                params.append(draft_text)
            if mtqe_score is not None:
                updates.append("mtqe_score = ?")
                params.append(mtqe_score)
            if repair_rounds is not None:
                updates.append("repair_rounds = ?")
                params.append(repair_rounds)
            if error_flags is not None:
                updates.append("error_flags_json = ?")
                params.append(json.dumps(error_flags, ensure_ascii=False))
            if tm_hit is not None:
                updates.append("tm_hit = ?")
                params.append(1 if tm_hit else 0)
            if mqm_severity is not None:
                updates.append("mqm_severity = ?")
                params.append(mqm_severity)
            if mqm_spans is not None:
                updates.append("mqm_spans_json = ?")
                params.append(json.dumps(mqm_spans, ensure_ascii=False))
            if style is not None:
                updates.append("style_json = ?")
                params.append(_style_json(style))

            where = "WHERE block_id = ?"
            if scope is not None:
                where += " AND job_id = ?"
            cursor = conn.execute(
                f"UPDATE blocks SET {', '.join(updates)} {where}",
                [*params, block_id, *([scope] if scope is not None else [])],
            )
            conn.execute("COMMIT;")
            self._mark_blocks_changed()
            return cursor.rowcount > 0

    def save_checkpoints_batch(
        self,
        updates: list[dict[str, Any]],
        *,
        clear_verdict_for: list[str] | None = None,
        job_id: str | None = None,
        allow_terminal_override: bool = False,
    ) -> int:
        """Atomically update multiple blocks in a single transactional write.

        Returns the number of rows actually updated (callers counting
        promotions must not trust a buffered write's success).

        ``clear_verdict_for`` nulls the machine verdict columns of those block
        ids in the *same* transaction. Human PE import needs both, and doing them
        as two calls left a window where a crash kept the new human text next to
        a stale "critical" severity — the exact state the clear exists to
        prevent.

        ``allow_terminal_override`` lets a *deliberate* non-terminal write land
        on a row already in a terminal state. The triage stage's paid escalated
        repair returns ``REPAIR_PENDING`` for a block that entered as ``FAILED``
        (terminal); without this flag the terminal guard silently dropped the
        result (``rowcount == 0``) and resume re-escalated — and re-billed — the
        same block. Every other caller leaves it off so a late duplicate write
        still cannot resurrect a terminal row.

        Note on omitted keys: ``repair_rounds``
        and ``error_flags`` are read with ``item.get(...)`` — a *missing* key
        yields ``None`` and ``COALESCE`` then preserves the stored value.
        Callers that omit them (quality_gate, export, ctext) keep the stored
        repair history and defect markers intact.

        Passing an explicit empty list for ``error_flags`` still clears the
        stored flags — omit the key to preserve.
        """
        if not updates and not clear_verdict_for:
            return 0

        with self._get_conn() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            scope = self._block_scope(conn, job_id)
            cursor = conn.cursor()
            updated = 0
            for item in updates:
                block_id = item["block_id"]
                status = item["status"]
                target_text = item.get("target_text")
                draft_text = item.get("draft_text")
                mtqe_score = item.get("mtqe_score")
                repair_rounds = item.get("repair_rounds")
                # Default None (not ) so an omitted key preserves
                # the stored flags instead of overwriting them with "[]".
                error_flags = item.get("error_flags")
                tm_hit = item.get("tm_hit")
                mqm_severity = item.get("mqm_severity")
                mqm_spans = item.get("mqm_spans")
                glossary_hits = item.get("glossary_hits")
                # Omitted (repair/consistency/export checkpoints) -> COALESCE
                # preserves the stored style, so a target-side run the draft
                # parsed survives a later text rewrite.
                style = item.get("style")

                status_val = status.value if isinstance(status, BlockStatus) else str(status)

                # A *non-terminal* write must not resurrect a row
                # already in a terminal state. A terminal -> terminal re-save (PE
                # import, export) is legitimate and must still apply. The triage
                # stage's paid escalated repair is the one caller that must write
                # a non-terminal ``REPAIR_PENDING`` over a terminal ``FAILED``
                # row, so it opts in via ``allow_terminal_override``.
                terminal_values = sorted(s.value for s in TERMINAL_STATUSES)
                query = """
                    UPDATE blocks
                    SET status = ?,
                        target_text = COALESCE(?, target_text),
                        draft_text = COALESCE(?, draft_text),
                        mtqe_score = COALESCE(?, mtqe_score),
                        repair_rounds = COALESCE(?, repair_rounds),
                        error_flags_json = COALESCE(?, error_flags_json),
                        tm_hit = COALESCE(?, tm_hit),
                        mqm_severity = COALESCE(?, mqm_severity),
                        mqm_spans_json = COALESCE(?, mqm_spans_json),
                        glossary_hits_json = COALESCE(?, glossary_hits_json),
                        style_json = COALESCE(?, style_json),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE block_id = ?
                """
                params: list[Any] = [
                    status_val,
                    target_text,
                    draft_text,
                    mtqe_score,
                    repair_rounds,
                    json.dumps(error_flags, ensure_ascii=False)
                    if error_flags is not None
                    else None,
                    (1 if tm_hit else 0) if tm_hit is not None else None,
                    mqm_severity,
                    json.dumps(mqm_spans, ensure_ascii=False) if mqm_spans is not None else None,
                    json.dumps(glossary_hits, ensure_ascii=False)
                    if glossary_hits is not None
                    else None,
                    _style_json(style),
                    block_id,
                ]
                if status_val not in terminal_values and not allow_terminal_override:
                    terminal_placeholders = ",".join("?" * len(terminal_values))
                    query += f" AND status NOT IN ({terminal_placeholders})"
                    params.extend(terminal_values)
                if scope is not None:
                    query += " AND job_id = ?"
                    params.append(scope)
                cursor.execute(query, params)
                updated += cursor.rowcount
            if clear_verdict_for:
                placeholders = ",".join("?" * len(clear_verdict_for))
                scope_clause = " AND job_id = ?" if scope is not None else ""
                conn.execute(
                    f"""
                    UPDATE blocks
                       SET mtqe_score = NULL, mqm_severity = NULL,
                           mqm_spans_json = NULL, updated_at = CURRENT_TIMESTAMP
                     WHERE block_id IN ({placeholders}){scope_clause}
                    """,
                    [*clear_verdict_for, *([scope] if scope is not None else [])],
                )
            conn.execute("COMMIT;")
            self._mark_blocks_changed()
            return updated

    def reset_blocks_to_pending(self, block_ids: list[str], job_id: str | None = None) -> int:
        """Re-queue blocks by clearing their translation state. Returns rows changed.

        Resets translation state when a target needs re-derivation:
        when a target was produced under a gate that has since been tightened,
        re-rendering cannot fix it — the wrong text is already stored, and
        :meth:`save_checkpoint` deliberately treats ``None`` as "leave alone so
        the value can never be cleared. Re-queueing is the only way to make such
        a block re-derivable; the next run re-drafts it (and, because the TM
        writeback guard rejects the same content, cannot restore the previous text).

        ``tm_hit`` is cleared as well: leaving it set would record the *next*
        draft as a memory hit even though the poisoned entry is gone. The MQM
        triage columns are cleared too, or a re-queued block would keep the
        stale severity/spans of the defective draft that is being discarded.
        """
        if not block_ids:
            return 0
        with self._get_conn() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            scope = self._block_scope(conn, job_id)
            placeholders = ",".join("?" * len(block_ids))
            scope_clause = " AND job_id = ?" if scope is not None else ""
            cursor = conn.execute(
                f"""
                UPDATE blocks
                   SET status = ?, target_text = NULL, draft_text = NULL,
                       tm_hit = 0, mtqe_score = NULL, error_flags_json = NULL,
                       mqm_severity = NULL, mqm_spans_json = NULL,
                       repair_rounds = 0, updated_at = CURRENT_TIMESTAMP
                 WHERE block_id IN ({placeholders}){scope_clause}
                """,
                [BlockStatus.PENDING.value, *block_ids, *([scope] if scope is not None else [])],
            )
            conn.execute("COMMIT;")
            self._mark_blocks_changed()
            return int(cursor.rowcount)

    def reset_blocks_to_repair(self, block_ids: list[str], job_id: str | None = None) -> list[str]:
        """Re-queue blocks for *repair*, preserving their already-paid draft.

        A transient failure in the repair stage leaves a perfectly good (and
        already billed) ``target_text`` in the row — ``stages/repair.py``
        persists it on purpose. Sending such a block through
        :meth:`reset_blocks_to_pending` would NULL that text, so the next run
        re-drafts the same paragraph and pays for it a second time. These go
        back to ``REPAIR_PENDING`` with their text intact instead, which is the
        state the repair stage already consumes.

        Only ids that actually carry text are requeued; the returned list is the
        subset that was, so the caller can PENDING-reset the rest.
        """
        if not block_ids:
            return []
        with self._get_conn() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            scope = self._block_scope(conn, job_id)
            placeholders = ",".join("?" * len(block_ids))
            scope_clause = " AND job_id = ?" if scope is not None else ""
            rows = conn.execute(
                f"""
                SELECT block_id, error_flags_json FROM blocks
                 WHERE block_id IN ({placeholders}){scope_clause}
                   AND (target_text IS NOT NULL OR draft_text IS NOT NULL)
                """,
                [*block_ids, *([scope] if scope is not None else [])],
            ).fetchall()
            requeued: list[str] = []
            updates: list[tuple[Any, ...]] = []
            for row in rows:
                b_id = str(row["block_id"])
                requeued.append(b_id)
                raw = row["error_flags_json"]
                new_flags_json = None
                if raw:
                    with suppress(json.JSONDecodeError):
                        loaded = json.loads(raw)
                        if isinstance(loaded, list):
                            new_flags = [
                                str(f)
                                for f in loaded
                                if not any(
                                    str(f).startswith(prefix)
                                    for prefix in TRANSIENT_FAILURE_PREFIXES
                                )
                            ]
                            if new_flags:
                                new_flags_json = json.dumps(new_flags, ensure_ascii=False)
                param_row = (
                    BlockStatus.REPAIR_PENDING.value,
                    new_flags_json,
                    b_id,
                    *([scope] if scope is not None else []),
                )
                updates.append(param_row)
            if updates:
                conn.executemany(
                    f"""
                    UPDATE blocks
                       SET status = ?, error_flags_json = ?,
                           updated_at = CURRENT_TIMESTAMP
                     WHERE block_id = ?{scope_clause}
                    """,
                    updates,
                )
            conn.execute("COMMIT;")
            self._mark_blocks_changed()
        return requeued

    def reset_transient_failures(self, job_id: str, chapter_id: str | None = None) -> list[str]:
        """Re-queue blocks stranded by transient drafting/repair errors; returns reset ids.

        Resume semantics: without this reset an API outage would permanently
        strand blocks in FAILED/NEEDS_HUMAN — terminal states every resume
        query skips, turning a temporary provider error into a manual work
        order.
        Only blocks whose error flags carry the transient markers written by
        the draft/repair retry paths (``Drafting error:`` / ``Repair error:``)
        are reset; quality escalations (``needs_human_review``, BLOCKED_HUMAN,
        MQM spans) stay terminal so gated defects cannot loop back into
        automatic shipping.

        Repair-only markers are split out: the draft survived that failure, so
        the block returns to ``REPAIR_PENDING`` with its paid ``target_text``
        rather than being nulled and re-drafted (see
        :meth:`reset_blocks_to_repair`).

        ``chapter_id`` scopes the reset to one chapter. Chapter-streaming
        calls the draft stage once per chapter, and a job-wide reset there
        would re-PENDING (and NULL the paid ``target_text`` of) transiently
        failed blocks from *earlier* chapters whose draft pass is already
        over — destroying their best translation mid-run. Whole-book runs
        keep the job-wide default.

        Triage verdict exemption: triage keeps the original
        transient marker when upgrading to NEEDS_HUMAN, so the prefix alone
        cannot tell "never reviewed" from "reviewed, awaiting a human". Rows
        carrying a triage verdict (``needs_human_review`` /
        ``mqm_critical_blocked`` flags or a non-null ``mqm_severity``) are
        never reset — the PE queue must not silently lose members on resume.
        """
        chapter_clause = ""
        chapter_params: tuple[Any, ...] = ()
        if chapter_id is not None:
            chapter_clause, chapter_params = _chapter_match(chapter_id)
        with self._get_conn() as conn:
            actual_id = self._resolve_actual_job_id(conn, job_id)
            # Route by what the row actually carries, not by prefix
            # archaeology: a transient row whose paid text survived and whose
            # failure does not impugn the draft (export's ``untranslated:``
            # sweep marks DRAFTED rows FAILED without touching their text)
            # goes back to repair with the text intact — PENDING would NULL it
            # and re-bill the same draft on the next run.
            # A ``Drafting error:`` row is the exception: its text is a partial
            # or untrusted attempt, so it keeps the full PENDING reset even
            # when text is present. Textless rows always fall through to the
            # flag split: a repair-only failure without text still re-drafts
            # rather than stranding in REPAIR_PENDING.
            cursor = conn.execute(
                f"""
                SELECT block_id, error_flags_json, mqm_severity,
                       (target_text IS NOT NULL OR draft_text IS NOT NULL) AS has_text
                FROM blocks
                WHERE job_id = ? AND status IN ('failed', 'needs_human'){chapter_clause}
                """,
                (actual_id, *chapter_params),
            )
            rows = cursor.fetchall()
        reset_ids: list[str] = []
        repair_only_ids: set[str] = set()
        for row in rows:
            raw = row["error_flags_json"]
            if not raw:
                continue
            with suppress(json.JSONDecodeError):
                loaded = json.loads(raw)
                if not isinstance(loaded, list):
                    continue
                flags = [str(f) for f in loaded]
                if has_triage_verdict(flags):
                    continue
                if row["mqm_severity"] is not None:
                    continue
                if is_transient_failure(flags):
                    block_id = str(row["block_id"])
                    reset_ids.append(block_id)
                    # Keep paid text only when the failure does not impugn the
                    # draft itself; the full routing rules
                    # live on the SELECT comment above.
                    drafting_failed = any(
                        str(flag).startswith(DRAFTING_ERROR_PREFIX) for flag in flags
                    )
                    if is_repair_only_transient_failure(flags) or (
                        row["has_text"] and not drafting_failed
                    ):
                        repair_only_ids.add(block_id)
        if reset_ids:
            # Split by what actually failed. A repair-only failure kept its paid
            # draft, so it goes back to the repair queue with its text intact; a
            # drafting/untranslated failure never produced usable text, so
            # nulling and re-drafting is correct for it.
            requeued = self.reset_blocks_to_repair(
                [block_id for block_id in reset_ids if block_id in repair_only_ids],
                job_id,
            )
            requeued_set = set(requeued)
            pending_ids = [block_id for block_id in reset_ids if block_id not in requeued_set]
            if pending_ids:
                self.reset_blocks_to_pending(pending_ids, job_id)
            logger.info(
                "Resume recovery for %s: re-queued %d block(s) stranded by transient "
                "errors (%d for repair keeping their draft, %d for re-draft)",
                job_id,
                len(reset_ids),
                len(requeued),
                len(pending_ids),
            )
        return reset_ids

    def get_block(self, block_id: str, job_id: str | None = None) -> IRBlock | None:
        """Fetch a single block by ID (scoped to ``job_id`` when the file holds several)."""
        with self._get_conn() as conn:
            scope = self._block_scope(conn, job_id)
            if scope is not None:
                cursor = conn.execute(
                    "SELECT * FROM blocks WHERE block_id = ? AND job_id = ?", (block_id, scope)
                )
            else:
                cursor = conn.execute("SELECT * FROM blocks WHERE block_id = ?", (block_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_block(row)

    def get_all_blocks(self, job_id: str) -> list[IRBlock]:
        """Fetch all blocks for a job in strict spine order with zero cross-job mixing.

        Reads on a connection of their own (see ``_get_read_conn``) and
        converts **after** releasing it: a full scan is 346 ms on a 20k-block
        book and 88 % of that is :class:`IRBlock` construction, so holding the
        writer's lock across the parse stalled every other ledger user (a
        progress event went from 3 ms to 384 ms p95).
        """
        with self._get_read_conn() as conn:
            actual_id = self._resolve_actual_job_id(conn, job_id)
            cursor = conn.execute(
                "SELECT * FROM blocks WHERE job_id = ? ORDER BY spine_index ASC",
                (actual_id,),
            )
            rows = cursor.fetchall()
        return [self._row_to_block(r) for r in rows]

    def fetch_blocks_by_status(
        self, job_id: str, status: BlockStatus | str, chapter_id: str | None = None
    ) -> list[IRBlock]:
        """Fetch blocks for a job matching a specific lifecycle status."""
        status_val = status.value if isinstance(status, BlockStatus) else str(status)
        chapter_clause = ""
        chapter_params: tuple[Any, ...] = ()
        if chapter_id is not None:
            chapter_clause, chapter_params = _chapter_match(chapter_id)
        with self._get_conn() as conn:
            actual_id = self._resolve_actual_job_id(conn, job_id)
            cursor = conn.execute(
                f"SELECT * FROM blocks WHERE job_id = ? AND status = ?{chapter_clause} ORDER BY spine_index ASC",
                (actual_id, status_val, *chapter_params),
            )
            rows = cursor.fetchall()
            return [self._row_to_block(r) for r in rows]

    def fetch_blocks_by_type(
        self, job_id: str, block_type: BlockType | str, chapter_id: str | None = None
    ) -> list[IRBlock]:
        """Fetch blocks for a job matching a specific block type (e.g. formula)."""
        type_val = block_type.value if isinstance(block_type, BlockType) else str(block_type)
        chapter_clause = ""
        chapter_params: tuple[Any, ...] = ()
        if chapter_id is not None:
            chapter_clause, chapter_params = _chapter_match(chapter_id)
        with self._get_conn() as conn:
            actual_id = self._resolve_actual_job_id(conn, job_id)
            cursor = conn.execute(
                f"SELECT * FROM blocks WHERE job_id = ? AND block_type = ?{chapter_clause} ORDER BY spine_index ASC",
                (actual_id, type_val, *chapter_params),
            )
            rows = cursor.fetchall()
            return [self._row_to_block(r) for r in rows]

    def fetch_blocks_by_statuses(
        self, job_id: str, statuses: Iterable[BlockStatus | str]
    ) -> list[IRBlock]:
        """Fetch blocks for a job matching any of the specified statuses."""
        status_vals = tuple(s.value if isinstance(s, BlockStatus) else str(s) for s in statuses)
        if not status_vals:
            return []
        placeholders = ", ".join("?" for _ in status_vals)
        with self._get_conn() as conn:
            actual_id = self._resolve_actual_job_id(conn, job_id)
            cursor = conn.execute(
                f"SELECT * FROM blocks WHERE job_id = ? AND status IN ({placeholders}) ORDER BY spine_index ASC",
                (actual_id, *status_vals),
            )
            rows = cursor.fetchall()
            return [self._row_to_block(r) for r in rows]

    def fetch_source_texts(self, job_id: str) -> list[str]:
        """Fetch raw source texts for a job in spine order with zero Pydantic model overhead."""
        with self._get_read_conn() as conn:
            actual_id = self._resolve_actual_job_id(conn, job_id)
            cursor = conn.execute(
                "SELECT source_text FROM blocks WHERE job_id = ? ORDER BY spine_index ASC",
                (actual_id,),
            )
            return [str(row[0] or "") for row in cursor.fetchall()]

    def fetch_repair_eligible_blocks(
        self, job_id: str, max_rounds: int = 2, chapter_id: str | None = None
    ) -> list[IRBlock]:
        """Fetch blocks eligible for repair (non-finalized, not skipped, rounds < max_rounds)."""
        # Derive from the IR's TERMINAL_STATUSES (single source of truth).
        terminal_statuses = tuple(s.value for s in sorted(TERMINAL_STATUSES, key=lambda x: x.value))
        chapter_clause = ""
        chapter_params: tuple[Any, ...] = ()
        if chapter_id is not None:
            chapter_clause, chapter_params = _chapter_match(chapter_id)
        query = f"""
            SELECT * FROM blocks
            WHERE job_id = ?
              AND skip_translate = 0
              AND status NOT IN ({", ".join("?" * len(terminal_statuses))})
              AND repair_rounds < ?
              {chapter_clause}
            ORDER BY spine_index ASC
        """
        with self._get_conn() as conn:
            actual_id = self._resolve_actual_job_id(conn, job_id)
            cursor = conn.execute(
                query, (actual_id, *terminal_statuses, max_rounds, *chapter_params)
            )
            rows = cursor.fetchall()
            return [self._row_to_block(r) for r in rows]


class LedgerBatchMixin(_LedgerHost):
    """Batch API job persistence and create-reservation."""

    # ------------------------------------------------------------------
    # Batch API job persistence
    # ------------------------------------------------------------------

    # Batch statuses that mean "still consuming the provider's job queue";
    # anything else is terminal and must not be resumed.
    LIVE_BATCH_STATUSES: frozenset[str] = frozenset(
        {"submitted", "validating", "in_progress", "finalizing"}
    )

    # Statuses a restart must *resume* rather than re-create. ``completed`` is
    # the provider's terminal status but the batch is not yet consumed: results
    # have not been fetched and persisted (the router writes ``completed``
    # during polling and ``consumed`` only after it reaps the output file). A
    # crash in that window that re-created the batch would re-submit the same
    # payload and double-bill it, so ``completed`` resumes and the poll loop
    # re-fetches.
    RESUMABLE_BATCH_STATUSES: frozenset[str] = LIVE_BATCH_STATUSES | {"completed"}

    def register_batch_job(
        self,
        batch_id: str,
        job_id: str,
        idempotency_key: str,
        status: str = "submitted",
    ) -> None:
        """Persist a newly created Batch API job (survives restarts)."""
        with self._get_conn() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            conn.execute(
                """
                INSERT INTO batch_jobs (batch_id, job_id, idempotency_key, status)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(batch_id) DO UPDATE SET
                    status = excluded.status,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (batch_id, job_id, idempotency_key, status),
            )
            conn.execute("COMMIT;")

    def find_live_batch_by_idempotency_key(self, idempotency_key: str) -> str | None:
        """Return the batch_id of an unfinished job with this key, or None.

        The idempotency key is derived from the submitted block set, so a
        restart that re-proposes the same batch resumes polling the original
        provider job instead of paying for a duplicate submission.
        """
        placeholders = ",".join("?" * len(self.RESUMABLE_BATCH_STATUSES))
        with self._get_conn() as conn:
            row = conn.execute(
                f"""
                SELECT batch_id FROM batch_jobs
                WHERE idempotency_key = ? AND status IN ({placeholders})
                ORDER BY created_at ASC
                LIMIT 1
                """,
                (idempotency_key, *sorted(self.RESUMABLE_BATCH_STATUSES)),
            ).fetchone()
            return str(row["batch_id"]) if row else None

    def find_live_batch_for_job(self, job_id: str, *, exclude_key: str) -> str | None:
        """Return the batch_id of another unfinished batch for ``job_id``.

        A restart whose payload changed (a glossary edit, a new block set) gets a
        fresh idempotency key, so ``reserve_batch_job`` creates a *new* provider
        batch while the superseded one keeps billing until it is cancelled.
        The router uses this to abandon the superseded batch first.
        """
        placeholders = ",".join("?" * len(self.RESUMABLE_BATCH_STATUSES))
        with self._get_conn() as conn:
            row = conn.execute(
                f"""
                SELECT batch_id FROM batch_jobs
                WHERE job_id = ? AND idempotency_key != ? AND status IN ({placeholders})
                ORDER BY created_at ASC
                LIMIT 1
                """,
                (job_id, exclude_key, *sorted(self.RESUMABLE_BATCH_STATUSES)),
            ).fetchone()
            return str(row["batch_id"]) if row else None

    def update_batch_job_status(self, batch_id: str, status: str) -> None:
        """Record a Batch API status transition (terminal or intermediate)."""
        with self._get_conn() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            conn.execute(
                """
                UPDATE batch_jobs
                SET status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE batch_id = ?
                """,
                (status, batch_id),
            )
            conn.execute("COMMIT;")

    def is_batch_live(self, batch_id: str) -> bool:
        """Whether the provider batch may still consume its queue.

        A missing row answers ``True``: with no recorded status the safe action
        is to cancel (it may be billing) rather than assume it finished.
        """
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT status FROM batch_jobs WHERE batch_id = ?", (batch_id,)
            ).fetchone()
        if row is None:
            return True
        return str(row["status"]) in self.LIVE_BATCH_STATUSES

    # A batch_id is only known *after* the provider call returns, so a crash
    # between ``create_batch_job`` and persisting that id would re-submit (and
    # re-bill) the whole batch on restart. To close the gap, a worker first
    # reserves the right to create by inserting a deterministic sentinel row
    # keyed on the idempotency key; the PK on ``batch_id`` makes concurrent
    # reservations mutually exclusive. ``finalize_batch_job`` then atomically
    # promotes the sentinel to the real id. ``creating`` is deliberately NOT a
    # live status, so a still-in-flight reservation is never mistaken for a
    # pollable batch.
    BATCH_CREATING_LEASE_SECONDS: float = 900.0

    def reserve_batch_job(
        self, idempotency_key: str, job_id: str, *, lease_seconds: float | None = None
    ) -> tuple[str, str | None]:
        """Claim the right to create the batch for ``idempotency_key``.

        Returns ``(outcome, batch_id)``:

        - ``("resume", batch_id)`` — a live or completed-but-unconsumed batch
          already exists; poll (and, when completed, fetch) it instead of
          re-creating and re-billing the same payload.
        - ``("create", None)`` — the caller owns the create (fresh reservation,
          or a stale one handed over from a worker that died mid-create).
        - ``("pending", None)`` — another worker holds a fresh reservation and
          is submitting right now; do not create a duplicate.

        The whole check-and-reserve runs in one ``BEGIN IMMEDIATE`` transaction
        so two workers can never both conclude "nobody created this batch yet".
        ``lease_seconds`` overrides the staleness horizon (test seam).
        """
        sentinel = f"creating:{idempotency_key}"
        horizon = self.BATCH_CREATING_LEASE_SECONDS if lease_seconds is None else lease_seconds
        live_placeholders = ",".join("?" * len(self.RESUMABLE_BATCH_STATUSES))
        with self._get_conn() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            try:
                live = conn.execute(
                    f"""
                    SELECT batch_id FROM batch_jobs
                    WHERE idempotency_key = ? AND status IN ({live_placeholders})
                    ORDER BY created_at ASC
                    LIMIT 1
                    """,
                    (idempotency_key, *sorted(self.RESUMABLE_BATCH_STATUSES)),
                ).fetchone()
                if live is not None:
                    conn.execute("COMMIT;")
                    return ("resume", str(live["batch_id"]))
                claimed = conn.execute(
                    """
                    INSERT INTO batch_jobs (batch_id, job_id, idempotency_key, status)
                    VALUES (?, ?, ?, 'creating')
                    ON CONFLICT(batch_id) DO NOTHING
                    """,
                    (sentinel, job_id, idempotency_key),
                )
                if claimed.rowcount == 1:
                    conn.execute("COMMIT;")
                    return ("create", None)
                # Sentinel exists: adopt it only if its owner is stale.
                stale = conn.execute(
                    """
                    UPDATE batch_jobs
                    SET job_id = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE batch_id = ?
                      AND updated_at < datetime('now', ?)
                    """,
                    (job_id, sentinel, f"{-horizon} seconds"),
                )
                conn.execute("COMMIT;")
                if stale.rowcount == 1:
                    # The dead owner may have paid for a batch whose id never
                    # reached the ledger (crash between the provider call and
                    # ``finalize_batch_job``). The id is unknowable now, so the
                    # only honest record is a loud one: this submission may be
                    # a second charge for the same payload. Warning here, not
                    # at the call site: the sentinel is consumed by this
                    # handover, and a successful ``finalize_batch_job`` would
                    # overwrite the adopted row -- this is the one point where
                    # the takeover is observable.
                    logger.warning(
                        "Adopted stale batch reservation %s (job %s): the previous "
                        "owner died mid-create and a batch it submitted may remain "
                        "unrecorded (paid but uncancellable)",
                        sentinel,
                        job_id,
                    )
                    return ("create", None)
                return ("pending", None)
            except BaseException:
                self._safe_rollback()
                raise

    def finalize_batch_job(
        self, idempotency_key: str, batch_id: str, status: str = "submitted"
    ) -> None:
        """Promote the reservation sentinel to the real ``batch_id``.

        Run immediately after ``create_batch_job`` returns; a single atomic
        UPDATE means a crash either leaves the reclaimable sentinel or a fully
        registered live batch — never a paid-for batch with no recorded id.

        If the sentinel is gone (a slow worker's reservation was reclaimed by a
        second worker that already promoted its own batch), the UPDATE matches
        no row. Dropping the id here would leave a **paid** batch unrecorded and
        therefore uncancellable and invisible to ``find_live_batch_for_job``, so
        the batch is registered as its own row instead.
        """
        sentinel = f"creating:{idempotency_key}"
        with self._get_conn() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            try:
                promoted = conn.execute(
                    """
                    UPDATE batch_jobs
                    SET batch_id = ?, status = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE batch_id = ?
                    """,
                    (batch_id, status, sentinel),
                )
                if promoted.rowcount == 0:
                    # Record the already-paid batch anyway (job_id from the
                    # sentinel's original claim if it survived, else empty).
                    owner = conn.execute(
                        "SELECT job_id FROM batch_jobs WHERE batch_id = ?", (sentinel,)
                    ).fetchone()
                    conn.execute(
                        """
                        INSERT INTO batch_jobs (batch_id, job_id, idempotency_key, status)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(batch_id) DO NOTHING
                        """,
                        (batch_id, str(owner["job_id"]) if owner else "", idempotency_key, status),
                    )
                conn.execute("COMMIT;")
            except BaseException:
                self._safe_rollback()
                raise
