"""The export stage: build the delivered file and validate it on the way out.

Everything that can only be judged against the finished artifact happens here —
integrity guards on the text about to be written, the render-quality and skip
flags recorded back onto the blocks, the paid post-render artifact gate, and the
usage billing the report reads — so a run is never declared complete on
evidence collected before the render existed.
"""

from __future__ import annotations

import asyncio
import contextlib
import html
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from ubt.core.cleaners.cjk_spacing import normalize_publishing_cjk
from ubt.core.config import DualMode
from ubt.core.engine.adapter_runtime import (
    DocumentAdapter,
    get_last_render_reconstructions,
)
from ubt.core.engine.artifact_services import check_artifact  # noqa: F401 -- patch seam
from ubt.core.engine.delivery_freshness import record_rendered_digest, rendered_digest_for
from ubt.core.engine.document_services import (
    optimize_pdf_deliverable,
)
from ubt.core.engine.events import EventType, TranslationProgressEvent
from ubt.core.engine.ledger import NON_TERMINAL_STATUSES, SQLiteJobLedger
from ubt.core.engine.stage_context import StageContext
from ubt.core.engine.stages.export_companions import (
    _artifact_payload,
    _delivery_translations,
    _write_artifact_check,
    _write_epub_view,
    _write_html_view,
    _write_spill_appendix,
    _write_table_appendix,
    _write_xliff_companion,
)
from ubt.core.engine.stages.export_gates import (
    RENDER_QUALITY_FLAGS,
    _apply_render_skip_ledger_pass,
    _check_completion_ratio,
    _check_needs_human_gate,
    _deliver_contract,
    _delivery_contract_evidence,
    _enforce_blocking_gate,
    _evaluate_delivery,
    _faithful_expected_pages,
    _is_faithful_delivery,
    _partition_render_skip_counts,
    _persist_crashed_gate_report,
    _run_artifact_gate,
    apply_render_quality_flags,
    apply_render_skip_flags,
)
from ubt.core.engine.stages.export_reports import (
    _build_reports,
    _drop_stale_run_reports,
    _font_delivery_for,
)
from ubt.core.exceptions import (
    IntegrityViolationError,
)
from ubt.core.ir.models import BlockStatus, BookManifest, IRBlock
from ubt.core.job_options import (
    CompanionArtifactResult,
    CompanionExportSummary,
    artifact_and_report_paths,
    resolve_target_output,
    sidecar_path,
)
from ubt.core.policy.bilingual_advisor import SECONDARY_SUFFIX

if TYPE_CHECKING:
    from ubt.core.job_options import DeliveredArtifact
from ubt.core.engine.facts import RenderPlan, Terminology
from ubt.core.engine.services import RunServices
from ubt.core.qe.defect_taxonomy import UNTRANSLATED_PREFIX
from ubt.core.validators.consistency import GlossaryConsistencyValidator
from ubt.core.validators.glossary_enforcer import DeterministicGlossaryEnforcer
from ubt.core.validators.html_delta import HTMLDeltaValidator
from ubt.core.validators.math_guard import apply_math_guards

logger = logging.getLogger(__name__)

#: Spellings of "target only" in both vocabularies (see :func:`_is_mono`).
_MONO_VALUES = frozenset({"mono", "monolingual"})


def _is_mono(value: object) -> bool:
    """True when ``value`` explicitly names the target-only delivery.

    Two vocabularies reach this stage — the advisor's ``DualMode``
    (``mono``/``dual``) and the adapter's ``bilingual_mode``
    (``monolingual``/``bilingual``). ``None`` means "the renderer reported no
    outcome", which is *not* an assertion of mono: a non-PDF target has no
    bilingual form but also no reported mode, so callers decide that case
    themselves rather than inheriting a guess.
    """
    return value is not None and str(value) in _MONO_VALUES


async def _render_adapter_output(
    adapter: DocumentAdapter,
    manifest: BookManifest,
    blocks: list[IRBlock],
    target_lang: str,
    output_path: Path,
    bilingual_mode: str | None = None,
    render_plan: RenderPlan | None = None,
) -> Path:
    """Render via ``render_blocks``.

    ``render_plan`` is the render decision the advisories made (compiler
    render plan protocol); ``bilingual_mode`` is read from it when not passed
    explicitly.
    """
    if bilingual_mode is None and render_plan is not None:
        # Same single mapping as preflight; an explicit bilingual_mode argument
        # (e.g. the secondary companion) still wins.
        bilingual_mode = render_plan.adapter_mode()

    if adapter.engine_name is not None:
        return await adapter.render_blocks(
            manifest=manifest,
            blocks=blocks,
            target_lang=target_lang,
            output_path=output_path,
            bilingual_mode=bilingual_mode,
            render_plan=render_plan,
        )
    return await adapter.render_blocks(
        manifest=manifest,
        blocks=blocks,
        target_lang=target_lang,
        output_path=output_path,
        bilingual_mode=bilingual_mode,
    )


def _terminology_and_structure_pass(
    blocks: list[IRBlock],
    *,
    glossary_enforcer: DeterministicGlossaryEnforcer | None,
    glossary_validator: GlossaryConsistencyValidator,
    html_validator: HTMLDeltaValidator,
    target_lang: str,
    source_lang: str = "en",
) -> tuple[list[dict[str, Any]], int]:
    """Edit every final block once and return the checkpoints its edits changed.

    Runs in a worker thread: over a book's worth of blocks this is the longest
    synchronous stretch of the export, and the loop it would otherwise hold also
    serves SSE subscribers and other concurrent jobs. Safe to
    move because the blocks are this stage's own copies from
    ``ledger.get_all_blocks``, and nothing else reads them mid-pass.
    """
    modified_checkpoints: list[dict[str, Any]] = []
    enforced_spans = 0
    enforcer_cache: dict[str, DeterministicGlossaryEnforcer | None] = {}
    validator_cache: dict[str, GlossaryConsistencyValidator] = {}

    def _enforcer_for(block: IRBlock) -> DeterministicGlossaryEnforcer | None:
        if glossary_enforcer is None:
            return None
        from ubt.core.memory.rolling_summary import extract_chapter_id
        from ubt.core.memory.scoped_glossary import (
            glossary_has_block_scope,
            resolve_effective_glossary,
            scope_context_for_block,
        )

        source_entries = list(getattr(glossary_enforcer, "source_glossary", ()))
        key = block.id if glossary_has_block_scope(source_entries) else extract_chapter_id(block.id)
        if key not in enforcer_cache:
            effective = resolve_effective_glossary(
                source_entries,
                scope_context_for_block(block),
            )
            enforcer_cache[key] = DeterministicGlossaryEnforcer(
                glossary=effective,
                target_lang=glossary_enforcer.target_lang,
                source_lang=getattr(glossary_enforcer, "source_lang", source_lang),
            )
        return enforcer_cache[key]

    def _validator_for(block: IRBlock) -> GlossaryConsistencyValidator:
        from ubt.core.memory.rolling_summary import extract_chapter_id
        from ubt.core.memory.scoped_glossary import (
            glossary_has_block_scope,
            resolve_effective_glossary,
            scope_context_for_block,
        )

        entries = list(glossary_validator.glossary)
        key = block.id if glossary_has_block_scope(entries) else extract_chapter_id(block.id)
        if key not in validator_cache:
            validator_cache[key] = GlossaryConsistencyValidator(
                glossary=resolve_effective_glossary(entries, scope_context_for_block(block))
            )
        return validator_cache[key]

    for fb in blocks:
        # Verbatim-skipped blocks (watermarks, bibliography, symbol
        # debris) ship byte-identical: glossary substitution inside them
        # creates Chinglish mash ("A. Vaswani et al. 注意力 Is All You
        # Need"), so both enforcement and validation skip them.
        if fb.target_text and not fb.skip_translate:
            # 0. Glossary enforcement (opt-in)
            # Establish canonical terminology first so subsequent publishing polish formats it properly.
            # A quarantined block ships the source verbatim, so substituting
            # target-language terms into it (and later validating it against the
            # glossary) can only produce Chinglish mash and false drift.
            enforcer = _enforcer_for(fb) if fb.status is not BlockStatus.BLOCKED_HUMAN else None
            if enforcer is not None:
                enforced_text, records, quarantined = enforcer.enforce_audited(fb.target_text)
                if records:
                    enforced_spans += len(records)
                    fb.target_text = enforced_text
                    modified_checkpoints.append(
                        {
                            "block_id": fb.id,
                            "target_text": fb.target_text,
                            "status": fb.status,
                            "error_flags": fb.error_flags,
                        }
                    )
                if quarantined:
                    for q in quarantined:
                        flag = (
                            f"glossary_quarantine_compound: {q.original_span}->{q.corrected_span}"
                        )
                        if flag not in fb.error_flags:
                            fb.error_flags.append(flag)
                    modified_checkpoints.append(
                        {
                            "block_id": fb.id,
                            "target_text": fb.target_text,
                            "status": fb.status,
                            "error_flags": fb.error_flags,
                        }
                    )

            # 1. CJK spacing and publishing punctuation normalization
            spaced = normalize_publishing_cjk(fb.target_text, target_lang=target_lang)
            if spaced != fb.target_text:
                fb.target_text = spaced
                modified_checkpoints.append(
                    {
                        "block_id": fb.id,
                        "target_text": fb.target_text,
                        "status": fb.status,
                        "error_flags": fb.error_flags,
                    }
                )

            if fb.status is not BlockStatus.BLOCKED_HUMAN:
                # Use the canonical marker QE writes, not a private string. The
                # export stage used to write "glossary_inconsistency: …" while
                # the QE path wrote "Glossary term violation: …", and the
                # workbench groups issues by matching the *latter* substring --
                # so drift discovered here reached no ribbon, no filter and no
                # PE queue row. One producer, one marker.
                from ubt.core.qe.heuristic import glossary_violation_flag

                canonical = glossary_violation_flag(
                    fb.source_text, fb.target_text, _validator_for(fb)
                )
                if canonical is not None:
                    flag = canonical
                    if flag not in fb.error_flags:
                        fb.error_flags.append(flag)
                        modified_checkpoints.append(
                            {
                                "block_id": fb.id,
                                "target_text": fb.target_text,
                                "status": fb.status,
                                "error_flags": fb.error_flags,
                            }
                        )

            # 2. Structural HTML delta validation (preserve draft with warning mark)
            html_res = html_validator.validate(fb.source_text, fb.target_text)
            if not html_res.is_valid:
                # Dedup like every other flag append here — without
                # it a resume/re-export double-appends the flag and nests the
                # <mark> warning wrap around an already-wrapped target.
                flag = f"html_structure_mismatch: {html_res.message}"
                if flag not in fb.error_flags:
                    fb.error_flags.append(flag)
                    clean_msg = html.escape((html_res.message or "").replace('"', "'"), quote=True)
                    fb.target_text = (
                        f'<mark class="ubt-failed-draft" title="html_structure_mismatch: {clean_msg}">'
                        f"{fb.target_text}</mark>"
                    )
                    # Keep the in-memory block in step with the checkpoint: the
                    # checkpoint records FAILED, so leaving ``fb.status`` at its
                    # prior value made the ledger and ``final_blocks`` disagree
                    # for the rest of the stage.
                    fb.status = BlockStatus.FAILED
                    modified_checkpoints.append(
                        {
                            "block_id": fb.id,
                            "target_text": fb.target_text,
                            "status": BlockStatus.FAILED,
                            "error_flags": fb.error_flags,
                        }
                    )
            else:
                # Advisory tier: emphasis-tag drift / dropped anchors are
                # normal in fluent translation, so they get a review flag and
                # a checkpoint -- never a <mark> wrap and never FAILED.
                for warning in html_res.details.get("formatting_warnings", []):
                    wflag = f"html_formatting_drift: {warning}"
                    if wflag not in fb.error_flags:
                        fb.error_flags.append(wflag)
                        modified_checkpoints.append(
                            {
                                "block_id": fb.id,
                                "target_text": fb.target_text,
                                "status": fb.status,
                                "error_flags": fb.error_flags,
                            }
                        )

    return modified_checkpoints, enforced_spans


async def _force_stale_blocks_terminal(
    ledger: SQLiteJobLedger, actual_job_id: str, *, claims_finality: bool = True
) -> list[IRBlock]:
    """Fail every still-non-terminal block, then return the job's blocks.

    Any block in a non-terminal status here means a stage crashed mid-flight.
    Rendering them as-is would ship untranslated source while the job reports
    "completed"; forcing FAILED first surfaces them in stats, the quality
    report, and logs.

    ``claims_finality=False`` (re-export) flips the contract: a re-export must
    not rewrite another run's in-flight blocks to FAILED — its only legitimate
    input is an already-terminal book. Non-terminal blocks are a hard error
    there, so the stale state stays exactly as the interrupted run left it
    (resume still sees ``pending``/``drafted`` rows to requeue).
    """
    if not claims_finality:
        placeholders = ",".join("?" * len(NON_TERMINAL_STATUSES))
        stale_ids: list[str] = []
        with ledger._get_conn() as conn:
            rows = conn.execute(
                f"""
                SELECT block_id FROM blocks
                WHERE job_id = ? AND status IN ({placeholders})
                LIMIT 11
                """,
                (actual_job_id, *NON_TERMINAL_STATUSES),
            ).fetchall()
            stale_ids = [str(row["block_id"]) for row in rows]
        if stale_ids:
            sample = ", ".join(stale_ids[:10])
            raise IntegrityViolationError(
                f"Re-export blocked for job {actual_job_id}: "
                f"{len(stale_ids)} block(s) are still non-terminal ({sample}). "
                "A re-export renders only an already-final book — it must not "
                "force another run's in-flight blocks to FAILED. Resume the job "
                "instead, and re-export once every block is terminal."
            )
        final_blocks: list[IRBlock] = await asyncio.to_thread(ledger.get_all_blocks, actual_job_id)
        return final_blocks
    stale_block_ids = await asyncio.to_thread(
        ledger.fail_non_terminal_blocks,
        actual_job_id,
        f"{UNTRANSLATED_PREFIX} block was still non-terminal when export started",
    )
    if stale_block_ids:
        logger.warning(
            "Export forced %d stale block(s) to FAILED for job %s: %s",
            len(stale_block_ids),
            actual_job_id,
            ", ".join(stale_block_ids[:10]),
        )
    final_blocks = await asyncio.to_thread(ledger.get_all_blocks, actual_job_id)
    return final_blocks


async def _render_complementary_artifact(
    ctx: StageContext,
    adapter: DocumentAdapter,
    final_blocks: list[IRBlock],
    artifact: DeliveredArtifact,
    render: RenderPlan,
) -> tuple[Path | None, str | None]:
    """Render the requested complementary dual-mode or secondary-engine PDF.

    Returns ``(path, None)`` on success, ``(None, None)`` when no secondary
    artifact was requested, and ``(None, reason)`` when the render failed.
    The failure is *never fatal* — a secondary render must not sink the
    delivered primary artifact — but it is returned to the caller so the
    companion audit record says "failed" instead of silently omitting the
    artifact from the result set.
    """
    manifest = ctx.manifest
    secondary_render = str(render.emit_secondary_mode or "")
    secondary_path: Path | None = None
    if secondary_render:
        # The mode the primary *actually* shipped (the renderer's outcome).
        primary_outcome = getattr(adapter, "last_render_outcome", None)
        primary_mode = str(
            (primary_outcome.effective_dual_mode if primary_outcome else None)
            or render.effective_dual_mode
            or "dual"
        )
        suffix = SECONDARY_SUFFIX.get(cast(DualMode, primary_mode), "_secondary")
        candidate = artifact.sibling(suffix)
        try:
            secondary_path = await _render_adapter_output(
                adapter=adapter,
                manifest=manifest,
                blocks=final_blocks,
                target_lang=ctx.target_lang,
                output_path=candidate,
                bilingual_mode=secondary_render,
                render_plan=render,
            )
            logger.info("Dual output rendered complementary artifact: %s", secondary_path)
        except Exception as exc:
            logger.warning("Complementary dual render failed (non-fatal): %s", exc)
            return None, f"{type(exc).__name__}: {exc}"
    return secondary_path, None


def _prepare_blocks_for_reflow(
    blocks: list[IRBlock],
    native_adapter: DocumentAdapter,
) -> list[IRBlock]:
    """Strip/resolve native-codec inline anchors before handing blocks to a reflow renderer.

    ``NativePDFCodec`` encodes inline formulas as ``<f1/>`` and style/link spans
    as ``<b1>...</b1>``/``<c1>...</c1>``. Reflow renderers (Typst/Docling) do not
    interpret those anchors and would otherwise print them verbatim into the PDF.
    """
    from ubt.core.validators.inline_tags import resolve_inline_tags_for_reflow

    native_units = getattr(native_adapter, "last_units", ()) or ()
    raw_by_unit: dict[str, dict[str, str]] = {}
    for unit in native_units:
        uid = str(getattr(unit, "unit_id", "") or "")
        if not uid:
            continue
        tok_map: dict[str, str] = {}
        for tok in getattr(unit, "tokens", ()) or ():
            if getattr(tok, "kind", None) == "rigid":
                tid = str(getattr(tok, "tag_id", "") or "").lower()
                raw = str((getattr(tok, "payload", None) or {}).get("raw_text", "") or "")
                if tid and raw:
                    tok_map[tid] = raw
        if tok_map:
            raw_by_unit[uid] = tok_map

    prepared: list[IRBlock] = []
    for blk in blocks:
        raw_map = raw_by_unit.get(blk.id)
        updates: dict[str, Any] = {}
        src = blk.source_text or ""
        if "<" in src:
            clean_src = resolve_inline_tags_for_reflow(src, raw_map)
            if clean_src != src:
                updates["source_text"] = clean_src
        if blk.draft_text and "<" in blk.draft_text:
            clean_draft = resolve_inline_tags_for_reflow(blk.draft_text, raw_map)
            if clean_draft != blk.draft_text:
                updates["draft_text"] = clean_draft
        if blk.target_text and "<" in blk.target_text:
            clean_tgt = resolve_inline_tags_for_reflow(blk.target_text, raw_map)
            if clean_tgt != blk.target_text:
                updates["target_text"] = clean_tgt
        prepared.append(blk.model_copy(update=updates) if updates else blk)
    return prepared


async def run_export_stage(
    ctx: StageContext,
    services: RunServices,
    render: RenderPlan,
    terminology: Terminology,
    *,
    claims_finality: bool = True,
) -> AsyncIterator[TranslationProgressEvent]:
    """Perform integrity checks, apply glossary enforcement (opt-in), and render output document.

    ``claims_finality`` (default True) is the full run's contract: stale
    non-terminal blocks are forced FAILED and the job is finalized
    ``completed``. A re-export passes False — it renders an already-final book
    without owning any block's fate or the job's terminal status.
    """
    ledger = ctx.ledger
    actual_job_id = ctx.job_id
    # NOTE: the run's usage snapshot is taken after the export stages so the
    # quality report reflects every billed call (including overflow compaction).
    manifest = ctx.manifest
    adapter = ctx.require_adapter()
    output_path = ctx.output_path
    input_path = ctx.input_path
    target_lang = ctx.target_lang
    source_lang = ctx.source_lang
    glossary_dicts = terminology.glossary_dicts
    html_validator = services.html_validator
    create_event_fn = ctx.create_event
    # Read the short-chain-derived policy from its single owner (AdaptivePolicy,
    # also what manifest.run.adaptive_policy reports) instead of re-deriving
    # from ctx.short_chain here — a second derivation could silently drift from
    # the value the report already published.
    artifact_blocking_gate_enabled = services.adaptive_policy.artifact_blocking
    min_completion_ratio = ctx.config.export_min_completion_ratio
    deterministic_glossary_enforce = services.adaptive_policy.deterministic_glossary
    ctx.check_cancelled()
    final_blocks = await _force_stale_blocks_terminal(
        ledger, actual_job_id, claims_finality=claims_finality
    )
    _check_completion_ratio(actual_job_id, final_blocks, min_completion_ratio)
    _check_needs_human_gate(actual_job_id, final_blocks, allow=ctx.config.export_allow_needs_human)

    # A re-export's whole contract is "re-typeset what the ledger already holds".
    # Every pass below can rewrite a block's delivered text or its status:
    # deterministic glossary substitution, CJK normalization, the HTML structure
    # checker (FAILED + <mark> wrap), the math guard (target reset to source),
    # and the length policy (NEEDS_HUMAN). Those belong to the run that *owns*
    # the blocks. Applying them here made a formatting-only re-render change
    # what the ledger says, silently rewriting a human edit or quarantining a
    # block that a later resume would no longer pick up. Skip the lot.
    owning_run = claims_finality

    glossary_enforcer = (
        DeterministicGlossaryEnforcer(
            glossary=glossary_dicts,
            target_lang=target_lang,
            source_lang=source_lang,
        )
        if deterministic_glossary_enforce and owning_run
        else None
    )
    glossary_validator = GlossaryConsistencyValidator(glossary=glossary_dicts)
    modified_checkpoints: list[dict[str, Any]] = []
    enforced_spans = 0
    if owning_run:
        modified_checkpoints, enforced_spans = await asyncio.to_thread(
            _terminology_and_structure_pass,
            final_blocks,
            glossary_enforcer=glossary_enforcer,
            glossary_validator=glossary_validator,
            html_validator=html_validator,
            target_lang=target_lang,
            source_lang=source_lang,
        )

    if modified_checkpoints:
        await asyncio.to_thread(ledger.save_checkpoints_batch, modified_checkpoints)

    # Gate 3 (math closed loop, CAT translate="no" + QA): formula targets must
    # be source LaTeX verbatim; FAILED math-debris blocks render source instead
    # of hallucinated drafts. Counted + flagged, statuses untouched.
    if owning_run:
        math_checkpoints, math_counts = await asyncio.to_thread(apply_math_guards, final_blocks)
    else:
        math_checkpoints, math_counts = (
            [],
            {
                "formula_invariant_repairs": 0,
                "math_debris_fallbacks": 0,
                "c_text_accepted": 0,
            },
        )
    if math_checkpoints:
        await asyncio.to_thread(ledger.save_checkpoints_batch, math_checkpoints)
    if (
        math_counts["formula_invariant_repairs"]
        or math_counts["math_debris_fallbacks"]
        or math_counts["c_text_accepted"]
    ):
        logger.warning(
            "Math guard for job %s: formula_invariant_repairs=%d "
            "math_debris_fallbacks=%d c_text_accepted=%d (formula pages should "
            "use docling; see pdf_parser_engine in job manifest)",
            actual_job_id,
            math_counts["formula_invariant_repairs"],
            math_counts["math_debris_fallbacks"],
            math_counts["c_text_accepted"],
        )

    # Default output is isolated under tmp/output/ so a forgotten -o
    # never pollutes the input directory. All derived artifacts
    # (secondary render, quality/visual reports, .typ sidecar, PE queue)
    # hang off target_output and follow it automatically. A target-only primary
    # gets the honest ``_mono`` default name; an explicit -o is respected as
    # given.
    is_monolingual_output = _is_mono(render.effective_dual_mode or render.bilingual_mode)
    # The page scope must reach the default name: a --pages run otherwise
    # writes the full document's filename and silently replaces a complete
    # delivery with a partial one. An explicit -o is left exactly as given.
    target_output = resolve_target_output(
        output_path,
        input_path,
        monolingual=is_monolingual_output,
        pages=ctx.config.get_selected_pages(),
    )
    target_output.parent.mkdir(parents=True, exist_ok=True)

    ctx.check_cancelled()
    effective_bilingual_mode = render.adapter_mode()
    # Persist the render decision before rendering: a re-export (a human edit
    # re-rendered into the delivered file) has no advisory stage to re-derive it
    # from, so the mode a re-export reproduces is read back from here. Without
    # it a re-export would default to monolingual and silently drop the
    # bilingual pairing the first run shipped.
    await asyncio.to_thread(
        ledger.set_job_metadata_value,
        actual_job_id,
        "render_plan",
        {
            "bilingual_mode": render.bilingual_mode,
            "effective_dual_mode": render.effective_dual_mode,
            "dual_mode_downgraded": render.dual_mode_downgraded,
            "translate_chrome": render.translate_chrome,
            "bilingual_advisory": render.bilingual_advisory,
            "emit_secondary_mode": render.emit_secondary_mode,
        },
    )
    event = await create_event_fn(
        EventType.RENDER_STARTED,
        actual_job_id,
        ledger,
        message=f"Rendering document with {adapter.engine_name or 'typst'} for {len(final_blocks)} blocks",
    )
    yield event
    rendered_path = await _render_adapter_output(
        adapter=adapter,
        manifest=manifest,
        blocks=final_blocks,
        target_lang=target_lang,
        output_path=target_output,
        bilingual_mode=effective_bilingual_mode,
        render_plan=render,
    )

    # Bounded LLM overflow compaction for 1:1 native PDF delivery when text
    # still exceeds its source container at the font floor.
    native_overflows = getattr(adapter, "last_overflows", ())
    native_units = getattr(adapter, "last_units", ())
    if (
        owning_run
        and not ctx.is_mock_run
        and getattr(adapter, "engine_name", None) == "native"
        and native_overflows
        and native_units
        and getattr(ctx, "router", None) is not None
    ):
        from ubt.core.engine.overflow_compactor import compact_overflowing_units

        units_by_id = {u.unit_id: u for u in native_units}
        tgt_map = {
            b.id: b.target_text
            for b in final_blocks
            if b.target_text
            and not b.skip_translate
            and b.status not in (BlockStatus.FAILED, BlockStatus.BLOCKED_HUMAN)
            and "<mark" not in b.target_text
        }
        compaction = await compact_overflowing_units(
            overflows=native_overflows,
            units_by_id=units_by_id,
            translations=tgt_map,
            complete_fn=ctx.router.complete_raw,
            target_lang=target_lang,
        )
        if compaction.compacted:
            by_id = {b.id: b for b in final_blocks}
            compact_ckpts: list[dict[str, Any]] = []
            for uid, tighter in compaction.compacted.items():
                blk = by_id.get(uid)
                if blk is None:
                    continue
                blk.target_text = tighter
                compact_ckpts.append(
                    {
                        "block_id": blk.id,
                        "target_text": tighter,
                        "status": blk.status,
                        "error_flags": blk.error_flags,
                    }
                )
            if compact_ckpts:
                await asyncio.to_thread(ledger.save_checkpoints_batch, compact_ckpts)
                rendered_path = await _render_adapter_output(
                    adapter=adapter,
                    manifest=manifest,
                    blocks=final_blocks,
                    target_lang=target_lang,
                    output_path=target_output,
                    bilingual_mode=effective_bilingual_mode,
                    render_plan=render,
                )

    # Render skip pass-through + length conservation, before the report is
    # built (see _apply_render_skip_ledger_pass). A re-export applies them in
    # memory only.
    await _apply_render_skip_ledger_pass(ctx, adapter, final_blocks, persist=owning_run)
    _check_completion_ratio(actual_job_id, final_blocks, min_completion_ratio)
    _check_needs_human_gate(actual_job_id, final_blocks, allow=ctx.config.export_allow_needs_human)

    # The renderer is allowed to change a block's disposition (a preserved
    # table's caption/legend is kept source-visible, a caption label is
    # normalized, a formula degrades). Build the placed-target map *after*
    # that decision: building it before the render made the artifact check
    # demand the pre-render translation of a block the renderer had just
    # decided to keep in the source (measured: caption pdf_main#b0038).
    translations: dict[str, str] = await asyncio.to_thread(_delivery_translations, final_blocks)

    # Clean up any stale quality/metrics reports from prior runs so that
    # the persisted report files always reflect the current render.
    _drop_stale_run_reports(rendered_path)

    # Delivery contract: the content-graph reconciliation, written beside the
    # artifact and embedded in the quality report; an opt-in hard gate
    # (UBT_STRICT_CONTRACT) aborts a knowingly-broken delivery. The renderer's
    # per-asset reconstruction ledger goes in, so a table rebuilt from its
    # translated grid is recorded RECONSTRUCTED (verified/corrupt), not opaque.
    contract = _deliver_contract(
        ctx, final_blocks, rendered_path, get_last_render_reconstructions(adapter)
    )
    if ctx.config.strict_contract and not contract.passed:
        raise IntegrityViolationError(
            f"Export blocked for job {ctx.job_id}: delivery contract failed with "
            f"{len(contract.errors)} error(s) - {contract.summary_line()}. The "
            "content/asset ledgers do not balance (dropped text or lost assets). "
            "Fix the loss, or unset UBT_STRICT_CONTRACT to ship knowingly."
        )
    delivery_contract = contract.model_dump(mode="json")

    # Bilingual XLIFF companion, HTML/EPUB views, and Attestation shadow.
    # Read-only, best-effort companions with structured audit contract.
    xliff_res = await asyncio.to_thread(_write_xliff_companion, ctx, rendered_path, final_blocks)
    html_res = await asyncio.to_thread(
        _write_html_view, ctx, rendered_path, translations, final_blocks
    )
    epub_res = await asyncio.to_thread(
        _write_epub_view, ctx, rendered_path, translations, final_blocks
    )
    attestation_res = await asyncio.to_thread(
        _write_artifact_check, ctx, rendered_path, translations, final_blocks
    )

    companion_results: list[CompanionArtifactResult] = [
        xliff_res,
        html_res,
        epub_res,
        attestation_res,
    ]
    if getattr(ctx.config, "spill_policy", "error") in ("appendix", "warn", "source_kept_warn"):
        spill_path = await asyncio.to_thread(
            _write_spill_appendix, ctx, rendered_path, final_blocks, contract
        )
        if spill_path is not None:
            companion_results.append(
                CompanionArtifactResult(
                    kind="spill_appendix",
                    status="generated",
                    path=spill_path,
                    size_bytes=spill_path.stat().st_size if spill_path.exists() else None,
                )
            )

    companion_summary = CompanionExportSummary(results=companion_results)
    companion_summary_dict = companion_summary.model_dump(mode="json")

    # Persist companion audit record to ledger job metadata
    await asyncio.to_thread(
        ledger.set_job_metadata_value, actual_job_id, "companion_artifacts", companion_summary_dict
    )
    if hasattr(ctx.manifest, "metadata") and isinstance(ctx.manifest.metadata, dict):
        ctx.manifest.metadata["companion_artifacts"] = companion_summary_dict

    comp_event = await create_event_fn(
        EventType.COMPANIONS_EXPORTED,
        actual_job_id,
        ledger,
        message=f"Companion artifacts exported: {companion_summary.summary_line()}",
    )
    yield comp_event

    # Enforcement is deferred until after _build_reports (below): a refusal
    # must still leave the quality/metrics reports on disk for audit.
    rendered_path, artifact_report_path, gate = await _run_artifact_gate(
        ctx, adapter, final_blocks, rendered_path, render, claims_finality=owning_run
    )

    # Page geometry: what the renderer measured off the source and what it
    # actually delivered, per page. Persisted because this fact existed only
    # inside the build until now, so "why is page 12 at 7.5pt" became
    # unanswerable once the process exited — and the QA report, the artifact gate
    # and the operator all ask that question about a shipped artifact.
    #
    # Written from the adapter's side channel rather than re-measured: the
    # solver already made these decisions, and a second measurement could
    # disagree with what actually shipped. Empty for a reflow render,
    # which measures no page — recorded as nothing rather than as zeroes.
    page_geometry = list(getattr(adapter, "last_page_geometry", None) or [])
    if page_geometry:
        await asyncio.to_thread(ctx.ledger.record_page_geometry, actual_job_id, page_geometry)
        if hasattr(manifest, "metadata") and isinstance(manifest.metadata, dict):
            manifest.metadata["page_geometry_pages"] = len(page_geometry)

    # Miniaturization & stream optimization pass for final delivered PDF
    if rendered_path.suffix.lower() == ".pdf" and getattr(
        ctx.config, "enable_pdf_optimization", True
    ):
        opt_prof = getattr(ctx.config, "pdf_optimize_profile", "lossless")
        opt_res = await asyncio.to_thread(optimize_pdf_deliverable, rendered_path, opt_prof)
        if opt_res.status == "optimized":
            if hasattr(manifest, "metadata") and isinstance(manifest.metadata, dict):
                manifest.metadata["pdf_optimization"] = opt_res.model_dump()
            await asyncio.to_thread(
                ledger.set_job_metadata_value,
                actual_job_id,
                "pdf_optimization",
                opt_res.model_dump(),
            )

    # The two artifact identities (requested target vs returned file, which the
    # artifact gate may have rewritten) named once, where the render returns.
    from ubt.core.job_options import delivered_artifact

    artifact = delivered_artifact(rendered_path, target_output)
    secondary_path, secondary_error = await _render_complementary_artifact(
        ctx, adapter, final_blocks, artifact, render
    )
    if secondary_error is not None:
        # Keep the failure in the same audit record operators already read for
        # HTML/EPUB/XLIFF companions. The job stays completed (the primary
        # deliverable is sound), but "the secondary artifact is missing" is no
        # longer indistinguishable from "no secondary artifact was requested".
        companion_results.append(
            CompanionArtifactResult(
                kind="complementary_pdf",
                status="failed",
                error=secondary_error,
            )
        )
    elif secondary_path is not None:
        companion_results.append(
            CompanionArtifactResult(
                kind="complementary_pdf",
                status="generated",
                path=secondary_path,
                size_bytes=secondary_path.stat().st_size if secondary_path.exists() else None,
            )
        )
    if secondary_path is not None or secondary_error is not None:
        companion_summary = CompanionExportSummary(results=companion_results)
        companion_summary_dict = companion_summary.model_dump(mode="json")
        await asyncio.to_thread(
            ledger.set_job_metadata_value,
            actual_job_id,
            "companion_artifacts",
            companion_summary_dict,
        )
        if hasattr(ctx.manifest, "metadata") and isinstance(ctx.manifest.metadata, dict):
            ctx.manifest.metadata["companion_artifacts"] = companion_summary_dict
    if (
        secondary_path is not None
        and secondary_path.suffix.lower() == ".pdf"
        and getattr(ctx.config, "enable_pdf_optimization", True)
    ):
        opt_prof = getattr(ctx.config, "pdf_optimize_profile", "lossless")
        await asyncio.to_thread(optimize_pdf_deliverable, secondary_path, opt_prof)

    # Append translated table appendix and generate companion files when tables were translated
    table_mode = getattr(ctx.config, "table_render_mode", "auto")
    if table_mode in ("auto", "appendix"):
        table_path = await asyncio.to_thread(
            _write_table_appendix, ctx, rendered_path, final_blocks
        )
        if table_path is not None:
            companion_results.append(
                CompanionArtifactResult(
                    kind="table_appendix",
                    status="generated",
                    path=table_path,
                    size_bytes=table_path.stat().st_size if table_path.exists() else None,
                )
            )
            companion_summary = CompanionExportSummary(results=companion_results)
            companion_summary_dict = companion_summary.model_dump(mode="json")
            await asyncio.to_thread(
                ledger.set_job_metadata_value,
                actual_job_id,
                "companion_artifacts",
                companion_summary_dict,
            )
            if hasattr(ctx.manifest, "metadata") and isinstance(ctx.manifest.metadata, dict):
                ctx.manifest.metadata["companion_artifacts"] = companion_summary_dict

    # Persist this run's spend to the ledger before the report reads it.
    await ctx.bill_run_usage()
    run_usage = ctx.measure_run_usage()
    report, report_path = await _build_reports(
        ctx,
        final_blocks,
        rendered_path=rendered_path,
        artifact_report_path=artifact_report_path,
        glossary_dicts=glossary_dicts,
        enforced_spans=enforced_spans,
        run_usage=run_usage,
        delivery_contract=delivery_contract,
        mode_advisory=render.bilingual_advisory,
        companion_artifacts=companion_summary_dict["results"],
    )

    # Artifact-level delivery contract: invariants read from the PDF
    # itself. Written after the reports so a refusal keeps its audit trail.
    # Non-PDF deliverables are out of scope for the PDF geometry contract.
    delivery_report = (
        await asyncio.to_thread(
            _evaluate_delivery,
            ctx,
            adapter,
            final_blocks,
            rendered_path,
            delivery_contract,
            page_geometry,
        )
        if rendered_path.suffix.lower() == ".pdf"
        else None
    )
    allow_major = bool(getattr(ctx.config, "allow_major", False))
    contract_ok = (
        (delivery_report.passed if allow_major else delivery_report.deliverable)
        if delivery_report is not None
        else True
    )
    if delivery_report is not None and not contract_ok and not ctx.is_mock_run:
        plan_meta = (getattr(ctx.manifest, "metadata", {}) or {}).get("delivery_plan")
        can_degrade = False
        if isinstance(plan_meta, dict):
            can_degrade = str(plan_meta.get("degrade_policy", "")).lower() == "allow_degrade" or (
                str(plan_meta.get("contract", "")).lower() == "faithful"
                and bool(getattr(ctx.config, "allow_degrade", False))
            )
        else:
            can_degrade = bool(getattr(ctx.config, "allow_degrade", False))

        # Only degrade a native 1:1 artifact to reflow when there is a *critical*
        # contract failure (``not delivery_report.passed``: missing/clipped/overlapping
        # text, page-count mismatch, or leaked machinery). Major-only overflow
        # advisories must never throw away an intact 1:1 PDF with all figures and
        # equations to substitute a figure-less reflow copy.
        if (
            can_degrade
            and not delivery_report.passed
            and getattr(delivery_report, "contract", "") == "faithful"
            and getattr(adapter, "engine_name", None) == "native"
        ):
            logger.warning(
                "Export contract failed critically for faithful mode (%s); degrading to reflow reading copy as permitted by policy",
                delivery_report.summary_line(),
            )
            from ubt.core.engine.adapter_runtime import (
                adapter_runtime_config,
                apply_runtime_config,
            )
            from ubt.core.engine.document_services import resolve_adapter

            reflow_adapter: DocumentAdapter | None = None
            reflow_blocks = _prepare_blocks_for_reflow(final_blocks, adapter)
            candidate_path = rendered_path.with_name(
                f".{rendered_path.stem}.reflow_candidate{rendered_path.suffix}"
            )
            degraded_rendered = False
            try:
                reflow_adapter = resolve_adapter(ctx.input_path, pdf_engine="docling")
                apply_runtime_config(reflow_adapter, adapter_runtime_config(ctx.config))
                degraded_out = await _render_adapter_output(
                    adapter=reflow_adapter,
                    manifest=ctx.manifest,
                    blocks=reflow_blocks,
                    target_lang=ctx.target_lang,
                    output_path=candidate_path,
                    bilingual_mode=effective_bilingual_mode,
                    render_plan=render,
                )
                if degraded_out.exists() and degraded_out != rendered_path:
                    degraded_out.replace(rendered_path)
                degraded_rendered = True
            except Exception as degrade_exc:
                logger.warning(
                    "Faithful->readable degradation to docling reflow failed for job %s: %s",
                    actual_job_id,
                    degrade_exc,
                )
            finally:
                with contextlib.suppress(OSError):
                    if candidate_path.exists():
                        candidate_path.unlink()
                if reflow_adapter is not None:
                    close_fn = getattr(reflow_adapter, "close", None)
                    if callable(close_fn):
                        with contextlib.suppress(Exception):
                            await asyncio.to_thread(close_fn)

            if degraded_rendered:
                if hasattr(ctx.manifest, "metadata") and isinstance(ctx.manifest.metadata, dict):
                    ctx.manifest.metadata["delivery_mode"] = "readable"
                    ctx.manifest.metadata["delivery_mode_effective"] = "readable"
                    if "delivery_plan" in ctx.manifest.metadata and isinstance(
                        ctx.manifest.metadata["delivery_plan"], dict
                    ):
                        ctx.manifest.metadata["delivery_plan"]["contract"] = "readable"
                        ctx.manifest.metadata["delivery_plan"]["effective_mode"] = "readable"
                        ctx.manifest.metadata["delivery_plan"]["degraded"] = True
                        reasons = list(ctx.manifest.metadata["delivery_plan"].get("reasons", []))
                        reasons.append(
                            f"faithful delivery contract failed ({delivery_report.summary_line()}); degraded to reflow reading copy"
                        )
                        ctx.manifest.metadata["delivery_plan"]["reasons"] = reasons
                        await asyncio.to_thread(
                            ledger.set_job_metadata_value,
                            actual_job_id,
                            "delivery_plan",
                            ctx.manifest.metadata["delivery_plan"],
                        )

                await asyncio.to_thread(
                    ledger.set_job_metadata_value,
                    actual_job_id,
                    "delivery_mode",
                    "readable",
                )
                await asyncio.to_thread(
                    ledger.set_job_metadata_value,
                    actual_job_id,
                    "delivery_mode_effective",
                    "readable",
                )

                eval_adapter = reflow_adapter or adapter
                rendered_path, artifact_report_path, gate = await _run_artifact_gate(
                    ctx,
                    eval_adapter,
                    reflow_blocks,
                    rendered_path,
                    render,
                    claims_finality=owning_run,
                )
                delivery_contract = _deliver_contract(
                    ctx,
                    reflow_blocks,
                    rendered_path,
                    get_last_render_reconstructions(eval_adapter),
                ).model_dump(mode="json")
                delivery_report = await asyncio.to_thread(
                    _evaluate_delivery,
                    ctx,
                    eval_adapter,
                    reflow_blocks,
                    rendered_path,
                    delivery_contract,
                    [],
                )
                contract_ok = (
                    (delivery_report.passed if allow_major else delivery_report.deliverable)
                    if delivery_report is not None
                    else True
                )
                report, report_path = await _build_reports(
                    ctx,
                    reflow_blocks,
                    rendered_path=rendered_path,
                    artifact_report_path=artifact_report_path,
                    glossary_dicts=glossary_dicts,
                    enforced_spans=enforced_spans,
                    run_usage=run_usage,
                    delivery_contract=delivery_contract,
                    mode_advisory=render.bilingual_advisory,
                    companion_artifacts=companion_summary_dict["results"],
                )

        if not contract_ok:
            # The report is optional in this scope; calling through it directly
            # would raise AttributeError inside the failure path and hide the
            # contract failure behind a formatting error.
            contract_summary = (
                delivery_report.summary_line()
                if delivery_report is not None
                else "no delivery report was produced"
            )
            if getattr(ctx.config, "strict_contract", False):
                raise IntegrityViolationError(
                    f"Export blocked for job {ctx.job_id}: delivery contract failed — "
                    f"{contract_summary}. See "
                    f"{sidecar_path(rendered_path, 'delivery_contract.json')}."
                )
            logger.warning(
                "Delivery contract finding(s) for job %s: %s; delivering best-effort artifact (strict_contract is false). See %s.",
                ctx.job_id,
                contract_summary,
                sidecar_path(rendered_path, "delivery_contract.json"),
            )

    # Reader-visible machinery is never shippable, even in the default advisory
    # mode: ``#h(0pt, weak: true)`` inside a reference URL or ``⟦ω=0.5⟧`` in a
    # caption is not degraded prose a human can repair, it is the renderer's own
    # syntax leaking into the book. Recorded by the contract just above; blocked
    # here before the artifact is finalized (and therefore downloadable).
    if delivery_report is not None and not ctx.is_mock_run:
        reader_leaks = tuple(
            finding
            for finding in delivery_report.findings
            if finding.code in {"reader_machinery_leak", "placeholder_leak"}
        )
        if reader_leaks:
            detail = "; ".join(finding.message for finding in reader_leaks)
            raise IntegrityViolationError(
                f"Export blocked for job {ctx.job_id}: reader text contains "
                f"unshippable machinery — {detail}. See "
                f"{sidecar_path(rendered_path, 'delivery_contract.json')}."
            )

    # Enforce AFTER the reports are on disk so a refused job keeps its audit
    # trail, and downgrade to a warning for rehearsal (mock) runs — a mock
    # draft echoes source text, so the sparse-target finding would fail every
    # --dry-run before it could prove the render path.
    _enforce_blocking_gate(
        actual_job_id,
        gate,
        artifact_report_path,
        artifact_blocking_gate_enabled,
        rehearsal=ctx.is_mock_run,
    )

    # Mark the delivered files as current. Recorded here — after the primary
    # artifact, every companion, the artifact gate's possible rewrite and the
    # table appendix are all on disk — so a render that raised above leaves the
    # previous digest in place and the job still reads stale. The download
    # endpoint refuses a stale delivery, and a human edit is precisely what
    # makes one stale.
    #
    # The digest is computed *before* the terminal write and handed to
    # ``finalize_job`` so both land in one transaction: the previous order
    # (finalize, then digest) left a window where the job read ``completed``
    # with no digest, and a missing digest is the one state the freshness check
    # has to treat as "not stale".
    # The last cooperative boundary before the terminal write. The artifact gate
    # checks once after its own run, but a lease reclaimed during the companion
    # render, report writing or blocking-gate enforcement would otherwise let
    # the old owner stamp ``completed`` over the new owner's ledger.
    ctx.check_cancelled()
    deliverable_digest = await asyncio.to_thread(rendered_digest_for, ledger, actual_job_id)
    if claims_finality:
        # The download pointers commit with the status. Persisting them from the
        # export hook afterwards left a crash window where the job read
        # ``completed`` with no ``output_file`` — undownloadable, and every
        # surface reported success. Derived from the same helper the hook uses,
        # so the two cannot name different files.
        artifact_output, artifact_report, artifact_visual = artifact_and_report_paths(
            str(rendered_path)
        )
        await asyncio.to_thread(
            ledger.finalize_job,
            actual_job_id,
            status="completed",
            rendered_digest=deliverable_digest,
            artifacts={
                "output_file": str(artifact_output),
                "report_file": str(artifact_report),
                "artifact_report_file": str(artifact_visual),
            },
        )
    else:
        # A re-export does not own the job's terminal status, but this artifact
        # is now the current one, so the freshness marker must move with it.
        await asyncio.to_thread(record_rendered_digest, ledger, actual_job_id, deliverable_digest)

    dual_suffix = f" | Complementary artifact: {secondary_path}" if secondary_path else ""
    # What the artifact actually is: the renderer's outcome when it rendered,
    # else the plan (a non-PDF target never downgrades).
    render_outcome = getattr(adapter, "last_render_outcome", None)
    effective_mode = (
        render_outcome.effective_dual_mode if render_outcome else None
    ) or render.effective_dual_mode
    mode_value = (
        render_outcome.bilingual_mode if render_outcome else None
    ) or render.bilingual_mode
    # An unreported mode counts as monolingual here: the label must not promise
    # a bilingual pairing the renderer never claimed to produce.
    is_bilingual = (
        effective_mode is not None and not _is_mono(effective_mode) and not _is_mono(mode_value)
    )
    doc_label = "Bilingual document" if is_bilingual else "Translated document"
    event = await create_event_fn(
        EventType.EXPORT_COMPLETED,
        actual_job_id,
        ledger,
        message=f"{doc_label} rendered: {rendered_path}{dual_suffix}",
        artifact_path=str(rendered_path),
    )

    yield event


__all__ = [
    "RENDER_QUALITY_FLAGS",
    "_apply_render_skip_ledger_pass",
    "_artifact_payload",
    "_build_reports",
    "_check_completion_ratio",
    "_check_needs_human_gate",
    "_deliver_contract",
    "_delivery_contract_evidence",
    "_delivery_translations",
    "_drop_stale_run_reports",
    "_enforce_blocking_gate",
    "_evaluate_delivery",
    "_faithful_expected_pages",
    "_font_delivery_for",
    "_force_stale_blocks_terminal",
    "_is_faithful_delivery",
    "_is_mono",
    "_partition_render_skip_counts",
    "_persist_crashed_gate_report",
    "_render_adapter_output",
    "_render_complementary_artifact",
    "_run_artifact_gate",
    "_terminology_and_structure_pass",
    "_write_artifact_check",
    "_write_epub_view",
    "_write_html_view",
    "_write_spill_appendix",
    "_write_table_appendix",
    "_write_xliff_companion",
    "apply_render_quality_flags",
    "apply_render_skip_flags",
    "check_artifact",
    "run_export_stage",
]
