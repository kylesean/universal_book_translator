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
the re-export is a pure re-render. That zero-spend contract is enforced, not
assumed: the export stage's visual gate reads the config bound on the ctx, and
an operator with ``visual_judge_enabled`` would otherwise make paid VLM calls a
path that binds no usage sink never bills — so the re-export runs on a config
copy with the judge closed, the same closure the rehearsal path applies. The
deterministic T0/T1 visual checks still run. The render mode is read back from
the ledger where the first export persisted it, so a bilingual book stays
bilingual.

This is deliberately *not* a second pipeline: the block set, the glossary, and
the render mode all come from the ledger, so a re-export cannot diverge from what
the run actually produced.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

from ubt.core.engine.events import EventType, TranslationProgressEvent
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.ledger_reading import (
    ledger_event_factory,
    manifest_from_ledger,
    require_output_suffix,
    terminology_from_ledger,
)
from ubt.core.engine.stage_context import StageContext
from ubt.core.engine.stages.export import run_export_stage
from ubt.core.engine.writer_lock import LedgerWriterLock, keep_lease_fresh
from ubt.core.exceptions import LedgerWriterLeaseLostError
from ubt.core.ir.models import BookManifest
from ubt.core.ir.render_plan import RenderPlan
from ubt.core.ports import adapter_runtime_config, apply_runtime_config, resolve_adapter
from ubt.core.qe.fast_pass import FastPassFilter

if TYPE_CHECKING:
    from ubt.core.config import UBTConfig
    from ubt.core.engine.services import RunServices
    from ubt.core.qe.base import BaseQERunner
    from ubt.core.router.router import ModelRouter

logger = logging.getLogger(__name__)


class ReExportError(RuntimeError):
    """A job cannot be re-exported (no ledger, no source, no adapter, no output)."""


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
        dual_mode_downgraded=stored.get("dual_mode_downgraded"),
        translate_chrome=stored.get("translate_chrome"),
        bilingual_advisory=stored.get("bilingual_advisory"),
        emit_secondary_mode=stored.get("emit_secondary_mode"),
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

    Ownership: everything this builds travels out through the returned router /
    QE runner (the discarded orchestrator keeps nothing they need), so the
    *caller* owns their lifecycle — a long-lived caller that is done
    re-exporting closes the router. Today's only build-both caller is a
    one-shot CLI process, which is why nothing has leaked yet.
    """
    from ubt.core.engine.pipeline import PipelineOrchestrator

    orchestrator = PipelineOrchestrator(config=config, router=router, qe_runner=qe_runner)
    return orchestrator.router, orchestrator.qe_runner


def _apply_delivery_override(
    manifest: BookManifest,
    requested: str | None,
    *,
    allow_degrade: bool = False,
) -> str | None:
    """Re-target one re-export at another delivery contract, in place.

    Writes the same metadata keys the pipeline writes before a live run
    (``delivery_mode``, ``delivery_plan``, requested/effective/legacy), so the
    renderer and the artifact contract read one plan instead of each guessing.
    Returns the effective legacy render mode, or ``None`` when no override was
    requested.

    ``plan_for_document`` is the same resolver the pipeline uses, so a request
    the router would refuse ("faithful is not feasible for this document")
    refuses here too — a replay must not be able to promise something a live run
    would not.
    """
    if requested is None:
        return None
    from ubt.core.ports import classify_pdf_structure, probe_pdf_pages
    from ubt.delivery.planning import plan_for_document

    metadata = dict(manifest.metadata or {})
    source_path = Path(manifest.source_path) if manifest.source_path else Path()
    is_pdf = source_path.suffix.lower() == ".pdf"

    has_scan = bool(metadata.get("has_scan", False))
    page_count = int(metadata.get("page_count") or 0)
    raw_damage = metadata.get("confirmed_font_damage_pages", metadata.get("font_damage_pages"))
    if isinstance(raw_damage, list):
        font_damage_pages = len(raw_damage)
    elif isinstance(raw_damage, int):
        font_damage_pages = raw_damage
    else:
        font_damage_pages = 0

    multicolumn_page_share = float(metadata.get("multicolumn_page_share") or 0.0)
    structural_page_share = float(metadata.get("structural_page_share") or 0.0)
    formula_heavy = bool(metadata.get("formula_heavy", False))

    if (
        is_pdf
        and source_path.exists()
        and (
            requested in ("faithful", "auto")
            or ("has_scan" not in metadata and "confirmed_font_damage_pages" not in metadata)
        )
    ):
        try:
            facts = classify_pdf_structure(source_path)
            has_scan = bool(facts.has_scan)
            font_damage_pages = int(facts.confirmed_font_damage_pages)
            multicolumn_page_share = float(facts.multicolumn_page_share)
            structural_page_share = float(facts.structural_page_share)
            formula_heavy = bool(facts.formula_heavy)
            if page_count <= 0 and font_damage_pages > 0:
                page_count = probe_pdf_pages(source_path)[0]
        except Exception as exc:  # noqa: BLE001 - census fallback stays fail-closed for faithful
            logger.debug("Re-export delivery census unavailable for %s: %s", source_path, exc)
            if requested == "faithful" and font_damage_pages == 0:
                font_damage_pages = 1

    decision = plan_for_document(
        requested,
        document_mode=str(metadata.get("document_mode") or ""),
        is_pdf=is_pdf,
        has_scan=has_scan,
        font_damage_pages=font_damage_pages,
        page_count=page_count,
        multicolumn_page_share=multicolumn_page_share,
        structural_page_share=structural_page_share,
        formula_heavy=formula_heavy,
        allow_degrade=allow_degrade,
    )
    if decision.effective_mode is None:
        raise ReExportError(
            f"delivery request {requested!r} is not feasible for job {manifest.doc_id}; "
            "the renderer has no contract it could honour"
        )
    metadata["delivery_mode"] = decision.effective_mode
    metadata["delivery_plan"] = decision.plan.to_manifest()
    metadata["delivery_mode_requested"] = requested
    metadata["delivery_mode_effective"] = decision.effective_mode
    metadata["delivery_mode_legacy"] = decision.legacy
    manifest.metadata = metadata
    return decision.effective_mode


async def re_export_job(
    *,
    db_path: Path,
    job_id: str,
    output_path: Path | None = None,
    config: UBTConfig,
    router: ModelRouter | None = None,
    qe_runner: BaseQERunner | None = None,
    delivery: str | None = None,
) -> TranslationProgressEvent:
    """Re-render a finished job's deliverable from its current ledger text.

    ``output_path`` defaults to the path the original run wrote (from the
    ledger's ``output_file``), so a re-export overwrites the deliverable in place
    — the file the customer downloads is the one that gains the human edits.
    Returns the terminal export event (its ``artifact_path`` names the written
    file).

    ``delivery`` re-renders the same translations under a different contract
    (``auto``/``readable``/``faithful``) without calling a model. It exists so
    G1 can compare the two paths on one ledger — a readable replay of a faithful
    job — instead of paying for a second translation run just to change the
    renderer. The override is resolved through the same
    :func:`ubt.delivery.planning.plan_for_document` the pipeline uses, so an
    infeasible request fails here exactly as it would fail before a live run.

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
    # Off-loop like every other sqlite call: a losing claimant parks on the
    # lock's busy_timeout, and this coroutine runs on the API server's loop
    # where a stall misses queue heartbeats.
    await asyncio.to_thread(lock.acquire)
    ledger: SQLiteJobLedger | None = None
    try:
        ledger = await asyncio.to_thread(SQLiteJobLedger, db_path)
        # Resolve every precondition BEFORE building collaborators: building the
        # router needs a configured provider, and a job with no source or no
        # output path must fail with its own reason rather than a "no draft
        # model" error from an unnecessary provider construction.
        manifest = manifest_from_ledger(ledger, job_id)
        delivery_mode = _apply_delivery_override(
            manifest,
            delivery,
            allow_degrade=bool(getattr(config, "allow_degrade", False)),
        )
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
                submitted = ledger.get_job_metadata_value(job_id, "submitted_request")
                if isinstance(submitted, dict) and submitted.get("output_path"):
                    stored_output = submitted["output_path"]
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
        # The adapter decides the artifact format, the same rule a first run
        # enforces before spending anything: a contradicting output suffix must
        # be refused before the render, not after a broken file is written (a
        # PDF engine into ``.md`` would also leak quarantine ``<mark>`` wrappers
        # verbatim into the deliverable).
        require_output_suffix(
            adapter,
            target,
            subject="re-exporting to",
            remedy="Drop --output (the ledger's own path is kept) or choose a supported suffix.",
        )
        # The zero-spend contract, enforced: the visual gate is driven by the
        # config bound on the ctx, and this path bills nothing (it binds no
        # usage sink), so a paid judge here would spend invisibly to every
        # budget surface.
        export_config = config.model_copy(update={"visual_judge_enabled": False})
        if delivery_mode is not None:
            # The adapter's renderer reads ``config.typst.delivery_mode``; the
            # manifest metadata alone would leave it on the original mode.
            export_config = export_config.model_copy(
                update={
                    "typst": export_config.typst.model_copy(update={"delivery_mode": delivery_mode})
                }
            )
        # The runtime config must be applied here too, not only at live-run
        # start: without it the adapter keeps its constructor fallback
        # (``facsimile``), so a ``readable`` replay re-rendered page-locked and
        # the artifact contract audited a contract the file was not rendered
        # under. ``delivery_mode`` is the ledger's or ``--delivery``'s effective
        # mode; when a ledger predates the delivery plan and records nothing, the
        # override is skipped and the adapter's own default stands — the
        # renderer still writes the mode it actually used into
        # ``manifest.metadata`` for the contract to read.
        recorded_mode = str(
            (manifest.metadata or {}).get("delivery_mode")
            or (manifest.metadata or {}).get("delivery_mode_effective")
            or ""
        )
        if delivery_mode is not None or recorded_mode:
            apply_runtime_config(
                adapter,
                adapter_runtime_config(export_config, delivery_mode=delivery_mode or recorded_mode),
            )
        render_plan = _render_plan_from_ledger(ledger, job_id)
        services = _build_services(export_config, router, qe_runner, manifest)
        # Created before the ctx so the export stage's own boundary checks can
        # observe a reclaimed lease mid-render, not only at the two event
        # boundaries (a whole-book render plus the visual gate routinely
        # outlives the default lease).
        lease_lost = asyncio.Event()
        ctx = StageContext(
            config=export_config,
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
            create_event=ledger_event_factory(ledger, job_id),
            lease_lost=lease_lost,
        )
        # Renew the lease for the life of the re-export. A whole-book Typst render
        # plus the visual gate routinely outlives the default lease, and
        # without renewal another process may legally reclaim the job mid-render:
        # both then write the same ledger and the same output file, the later
        # writer silently winning. The live run has always renewed; a re-export
        # is the same kind of long holder and must renew on the same rule.
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
                terminology_from_ledger(ledger, job_id),
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
        except LedgerWriterLeaseLostError as exc:
            # The export stage's own boundary check saw the reclaim mid-render
            # (the ctx is wired to ``lease_lost`` now). Same stop, expressed in
            # this module's documented error so callers need no new handler.
            raise ReExportError(f"re-export of {job_id} stopped: {exc}") from exc
        finally:
            renew_task.cancel()
            with suppress(asyncio.CancelledError):
                await renew_task
    finally:
        lock.release()
        if ledger is not None:
            ledger.close()
