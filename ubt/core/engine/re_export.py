"""Re-render a finished job's deliverable from its ledger (human-edit re-export).

The L3 review workbench lets a reviewer revise a segment's target text; the edit
lands in the ledger (``apply_human_edit``), but the *delivered file* was rendered
before the edit and nothing re-rendered it. The reviewer saw the correction in
the single-page preview, and the customer kept downloading the stale machine
draft. This module closes that gap: it rebuilds the run's inputs from the
ledger, re-runs only the export stage (render + reports + contract), and writes
the artifact the caller downloads.

Only the export stage runs. Every block is already terminal (a human edit stamps
``REPAIRED``), so no drafting, QE or repair is invoked and no token is spent —
the re-export is a pure re-render. The render mode is read back from the ledger
where the first export persisted it, so a bilingual book stays bilingual.

This is deliberately *not* a second pipeline: the block set, the glossary, and
the render mode all come from the ledger, so a re-export cannot diverge from what
the run actually produced.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ubt.core.engine.events import EventType, TranslationProgressEvent
from ubt.core.engine.facts import Terminology
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.stage_context import StageContext
from ubt.core.engine.stages.bible import _BIBLE_CACHE_KEY
from ubt.core.engine.stages.export import run_export_stage
from ubt.core.engine.writer_lock import LedgerWriterLock, keep_lease_fresh
from ubt.core.ir.models import BookManifest, ChapterMeta
from ubt.core.ir.render_plan import RenderPlan
from ubt.core.ports import resolve_adapter
from ubt.core.qe.fast_pass import FastPassFilter

if TYPE_CHECKING:
    from ubt.core.config import UBTConfig
    from ubt.core.engine.services import RunServices
    from ubt.core.qe.base import BaseQERunner
    from ubt.core.router.router import ModelRouter

logger = logging.getLogger(__name__)


class ReExportError(RuntimeError):
    """A job cannot be re-exported (no ledger, no source, no adapter, no output)."""


def _manifest_from_ledger(ledger: SQLiteJobLedger, job_id: str) -> BookManifest:
    """Rebuild the run's manifest from the ledger's persisted metadata.

    ``init_job_from_manifest`` stored doc_id, source_path, target_lang, title,
    source_lang, the chapter list and the run metadata, so the manifest a
    re-export renders from is the one the original run used — not a
    reconstruction that could disagree with it.
    """
    source_path = ledger.get_job_source_path(job_id) or ""
    target_lang = ledger.get_job_target_lang(job_id) or "zh"
    title = str(ledger.get_job_metadata_value(job_id, "title") or "")
    source_lang = str(ledger.get_job_metadata_value(job_id, "source_lang") or "en")
    raw_chapters = ledger.get_job_metadata_value(job_id, "chapters") or []
    chapters: list[ChapterMeta] = []
    if isinstance(raw_chapters, list):
        for entry in raw_chapters:
            if isinstance(entry, dict):
                try:
                    chapters.append(ChapterMeta.model_validate(entry))
                except Exception:  # a malformed chapter row must not sink the export
                    logger.debug("Skipping malformed chapter metadata for %s: %r", job_id, entry)
    return BookManifest(
        doc_id=job_id,
        title=title,
        source_path=source_path,
        source_lang=source_lang,
        target_lang=target_lang,
        chapters=chapters,
    )


def _render_plan_from_ledger(ledger: SQLiteJobLedger, job_id: str) -> RenderPlan:
    """The render decision the first export persisted, or a monolingual default.

    A job that predates the persistence (or never reached export) has no stored
    plan; a re-export then renders monolingual, which is the safe direction (it
    never invents a bilingual pairing the original did not ship).
    """
    stored = ledger.get_job_metadata_value(job_id, "render_plan")
    if not isinstance(stored, dict):
        return RenderPlan(bilingual_mode="monolingual", effective_dual_mode="mono")
    return RenderPlan(
        bilingual_mode=stored.get("bilingual_mode") or "monolingual",
        effective_dual_mode=stored.get("effective_dual_mode") or "mono",
        emit_secondary_mode=stored.get("emit_secondary_mode"),
    )


def _terminology_from_ledger(ledger: SQLiteJobLedger, job_id: str) -> Terminology:
    """The glossary the run enforced, from its persisted bible cache.

    Read from ``bible_cache`` rather than re-reading the config's glossary file,
    so the re-export enforces exactly the terms the quality gate judged — a fix
    cannot disagree with the flag the reviewer saw.
    """
    payload = ledger.get_job_metadata_value(job_id, _BIBLE_CACHE_KEY)
    if not isinstance(payload, dict):
        return Terminology()
    return Terminology(
        glossary_dicts=[
            dict(entry) for entry in payload.get("glossary_dicts", []) if isinstance(entry, dict)
        ],
        abbreviation_entries=[
            dict(entry)
            for entry in payload.get("abbreviation_entries", [])
            if isinstance(entry, dict)
        ],
    )


def _build_services(
    config: UBTConfig,
    router: ModelRouter,
    qe_runner: BaseQERunner,
    manifest: BookManifest,
) -> RunServices:
    """The minimal per-run services the export stage reads.

    The export stage touches only ``html_validator``, ``adaptive_policy`` and
    ``concurrency_sem``; the QE runner and repair loop are built for completeness
    but never invoked, because every block is already terminal.

    The policy is constructed directly rather than re-derived from a route probe:
    a re-export re-renders existing blocks, so ``visual_blocking`` (whether the
    post-render gate may refuse delivery) and ``deterministic_glossary`` (the
    export-time term enforcement) are the only fields it reads, and both are
    properties of the *config*, not of a fresh document probe.
    """
    from ubt.core.engine.repair_loop import RepairLoop
    from ubt.core.engine.services import RunServices
    from ubt.core.policy.adaptive_policy import AdaptivePolicy

    adaptive_policy = AdaptivePolicy(
        fast_lane_bible=False,
        visual_blocking=config.visual_blocking_gate_enabled,
        # Same source of truth as the original run's export: when the config
        # opts the long route into deterministic glossary enforcement, a
        # re-export must enforce it too — the old hard False produced a
        # second, weaker delivery from the same ledger (enforced_spans N -> 0)
        # with nothing flagging the difference.
        deterministic_glossary=config.deterministic_glossary_long_route,
        reason="re-export (render-only; no route probe)",
    )
    return RunServices(
        fast_pass=FastPassFilter(
            source_lang=manifest.source_lang, target_lang=manifest.target_lang
        ),
        qe_runner=qe_runner,
        repair_loop=RepairLoop(router=router, qe_runner=qe_runner),
        adaptive_policy=adaptive_policy,
        concurrency_sem=asyncio.Semaphore(config.max_concurrency),
    )


def _make_event_factory(ledger: SQLiteJobLedger, job_id: str) -> Any:
    """A minimal event factory: the export stage builds one terminal event.

    The real orchestrator's factory folds live ledger stats and billing; a
    re-export only needs the terminal event to carry the artifact path, so this
    reads the stats cheaply and leaves the spend fields at their defaults (a
    re-export spends nothing).
    """

    async def _create(
        event_type: EventType,
        job_id_arg: str,
        ledger_arg: SQLiteJobLedger,
        message: str = "",
        active_block_id: str | None = None,
        artifact_path: str | None = None,
    ) -> TranslationProgressEvent:
        stats = await asyncio.to_thread(ledger_arg.get_job_stats, job_id_arg)
        return TranslationProgressEvent(
            event_type=event_type,
            job_id=job_id_arg,
            total_blocks=int(stats.get("total", 0)),
            completed_blocks=int(stats.get("completed", 0)),
            message=message,
            active_block_id=active_block_id,
            artifact_path=artifact_path,
        )

    return _create


def _default_collaborators(
    config: UBTConfig,
    router: ModelRouter | None,
    qe_runner: BaseQERunner | None,
) -> tuple[ModelRouter, BaseQERunner]:
    """Build the router / QE runner the run would have used, from ``config``.

    Reuses :class:`PipelineOrchestrator`'s own construction (provider, rate
    limiter, QE engine selection, LLM-judge wrap) rather than re-deriving it, so
    a re-export's collaborators cannot drift from a real run's. Only the missing
    half is built: a caller holding a process-wide router (the API server) passes
    it and keeps the shared rate-limit bucket.
    """
    from ubt.core.engine.pipeline import PipelineOrchestrator

    orchestrator = PipelineOrchestrator(config=config, router=router, qe_runner=qe_runner)
    return orchestrator.router, orchestrator.qe_runner


async def re_export_job(
    *,
    db_path: Path,
    job_id: str,
    output_path: Path | None = None,
    config: UBTConfig,
    router: ModelRouter | None = None,
    qe_runner: BaseQERunner | None = None,
) -> TranslationProgressEvent:
    """Re-render a finished job's deliverable from its current ledger text.

    ``output_path`` defaults to the path the original run wrote (from the
    ledger's ``output_file``), so a re-export overwrites the deliverable in place
    — the file the customer downloads is the one that gains the human edits.
    Returns the terminal export event (its ``artifact_path`` names the written
    file).

    ``router``/``qe_runner`` are optional: when omitted, the orchestrator's own
    defaults are built from ``config`` (the same provider the run used). A
    caller that already holds a process-wide router (the API server) passes it
    so the re-export shares the one rate-limit bucket.

    Raises :class:`ReExportError` when the job has no ledger, no source document,
    or no resolvable output path. Raises
    :class:`ubt.core.exceptions.LedgerWriterLockConflictError` when another
    writer (a running resume, an interactive edit) holds the job's lock — the
    same single-writer invariant a live run and ``apply_human_edit`` observe.
    """
    if not db_path.exists():
        raise ReExportError(f"no ledger for job {job_id}")
    # Same invariant as apply_human_edit: an interactive re-render cannot race
    # a concurrent resume. Without this, a re-export forced another run's
    # in-flight blocks to FAILED (they are already terminal-block-checked now)
    # and rewrote the job status behind the running owner's back.
    # Lock is acquired BEFORE creating SQLiteJobLedger to prevent connection leaks
    # or concurrent DDL execution on locked databases.
    lock = LedgerWriterLock(db_path, job_id)
    lock.acquire()
    ledger: SQLiteJobLedger | None = None
    try:
        ledger = SQLiteJobLedger(db_path)
        # Resolve every precondition BEFORE building collaborators: building the
        # router needs a configured provider, and a job with no source or no
        # output path must fail with its own reason rather than a "no draft
        # model" error from an unnecessary provider construction.
        manifest = _manifest_from_ledger(ledger, job_id)
        source_path = Path(manifest.source_path) if manifest.source_path else None
        if source_path is None or not source_path.exists():
            raise ReExportError(
                f"job {job_id} source document is missing ({manifest.source_path!r}); "
                "cannot re-render"
            )
        target = output_path
        if target is None:
            stored_output = ledger.get_job_metadata_value(job_id, "output_file")
            if not stored_output:
                raise ReExportError(
                    f"job {job_id} has no recorded output path; pass one explicitly"
                )
            target = Path(str(stored_output))

        if router is None or qe_runner is None:
            built_router, built_qe = _default_collaborators(config, router, qe_runner)
            router = router or built_router
            qe_runner = qe_runner or built_qe

        adapter = resolve_adapter(source_path, pdf_engine=config.pdf_engine)
        render_plan = _render_plan_from_ledger(ledger, job_id)
        services = _build_services(config, router, qe_runner, manifest)
        ctx = StageContext(
            config=config,
            router=router,
            ledger=ledger,
            manifest=manifest,
            adapter=adapter,
            job_id=job_id,
            input_path=source_path,
            output_path=target,
            source_lang=manifest.source_lang,
            target_lang=manifest.target_lang,
            profile_name="general",
            create_event=_make_event_factory(ledger, job_id),
        )
        # Renew the lease for the life of the re-export. A whole-book Typst render
        # plus the visual gate routinely outlives the 300 s default lease, and
        # without renewal another process may legally reclaim the job mid-render:
        # both then write the same ledger and the same output file, the later
        # writer silently winning. The live run has always renewed; a re-export
        # is the same kind of long holder and must renew on the same rule.
        lease_lost = asyncio.Event()
        renew_task = asyncio.create_task(
            keep_lease_fresh(lock, on_lost=lambda _reason: lease_lost.set()),
            name=f"renew-lock:{job_id}",
        )
        try:
            terminal: TranslationProgressEvent | None = None
            async for event in run_export_stage(
                ctx,
                services,
                render_plan,
                _terminology_from_ledger(ledger, job_id),
                claims_finality=False,
            ):
                if lease_lost.is_set():
                    # Another writer owns the ledger now. Stop rather than write
                    # over it; the report above is the operator's explanation.
                    raise ReExportError(
                        f"re-export of {job_id} stopped: the writer lease was "
                        "reclaimed by another process"
                    )
                if event.event_type is EventType.EXPORT_COMPLETED:
                    terminal = event
            if terminal is None:
                raise ReExportError(f"re-export of {job_id} produced no artifact")
            return terminal
        finally:
            renew_task.cancel()
            with suppress(asyncio.CancelledError):
                await renew_task
    finally:
        lock.release()
        if ledger is not None:
            ledger.close()
