"""Stages 1.4 / 1.5 / 5.6: extraction witness and the two bilingual advisories.

These three ran inline in ``PipelineOrchestrator.run()``, which is why the
generator was 800+ lines: none of them is a translation step, all three are
zero-token judgements about the artifact, and they read only post-ingest IR.

- :func:`run_extraction_witness_stage` (D2') flags pages whose math was already
  broken by the extractor's font encoding, before anything is translated.
- :func:`run_mode_advisory_stage` scores the requested render mode against the
  layout the document actually has, applies the rigid-engine downgrade, and
  publishes the run-policy keys the renderer reads.
- :func:`run_difficulty_advisory_stage` is the post-repair half: live repair
  burden can step the mode down one notch, but only under ``auto``.
- :func:`apply_layout_tradeoff_advisory` records the formula-dense-on-rigid
  tradeoff in the delivered report (a plain function, not a stage: it runs
  before ingest, from facts the route probe already produced).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Literal

from ubt.core.config import RIGID_ENGINES, DualMode, canonical_render_engine
from ubt.core.engine.events import EventType, TranslationProgressEvent
from ubt.core.engine.stage_context import StageContext
from ubt.core.ir.models import BookManifest
from ubt.core.policy.adaptive_policy import (
    resolve_pdf_engine,
    resolve_render_engine_from_signals,
)
from ubt.core.policy.bilingual_advisor import (
    RENDER_MODE_VALUE,
    Advisory,
    advise_layout,
    assess_difficulty,
    resolve_effective_mode,
    secondary_mode,
)
from ubt.core.ports import (
    detect_figure_pages,
    flag_font_encoding_damage,
    inspect_font_encoding_damage,
    summarize_font_encoding_damage,
)

logger = logging.getLogger(__name__)


async def run_extraction_witness_stage(ctx: StageContext) -> None:
    """Mark font-encoding damage on the ingested blocks. Yields no events.

    Math-bearing fonts with no ``/ToUnicode`` make every library guess the same
    wrong thing, so cross-channel agreement *hides* the damage; this looks for it
    directly and flags the confirmed pages while the text is still cheap to
    re-do (docs/design/PDF_SKILL_BORROWINGS.md D2').
    """
    if ctx.source_pdf_path is None:
        return
    ledger = ctx.ledger
    try:
        witness_verdicts = await asyncio.to_thread(inspect_font_encoding_damage, ctx.input_path)
        stats = summarize_font_encoding_damage(witness_verdicts)
        ctx.manifest.run.extraction_witness = stats
        # force_refresh: this is the stage that *fills* the shared blocks cache,
        # and it mutates error_flags on disk right after the read. Handing the
        # freshly-populated snapshot to the next stage ensures mode advisory
        # reflects the post-witness state.
        witness_flagged = flag_font_encoding_damage(
            await ctx.current_blocks(force_refresh=True), witness_verdicts
        )
        if witness_flagged:
            await asyncio.to_thread(
                ledger.save_checkpoints_batch,
                [
                    {
                        "block_id": b.id,
                        "status": b.status,
                        "error_flags": b.error_flags,
                    }
                    for b in witness_flagged
                ],
            )
        if stats["confirmed_pages"]:
            logger.warning(
                "Extraction witness for job %s: %d/%d pages carry "
                "font-encoding damage (%d residue chars); formulas in "
                "source text are already wrong before translation — "
                "see docs/design/PDF_SKILL_BORROWINGS.md D2'",
                ctx.job_id,
                stats["confirmed_pages"],
                stats["pages"],
                stats["residue_chars"],
            )
    except Exception as exc:
        logger.debug("Extraction witness skipped for %s: %s", ctx.job_id, exc)


async def run_mode_advisory_stage(ctx: StageContext) -> AsyncIterator[TranslationProgressEvent]:
    """Publish the run policy and advise on the requested bilingual mode."""
    config = ctx.config
    manifest = ctx.manifest
    requested_mode: DualMode = (
        config.dual_mode if config.dual_mode in RENDER_MODE_VALUE else "inline"
    )
    auto_mode = config.dual_mode == "auto"
    tier_basis: DualMode = "inline" if auto_mode else requested_mode
    enforcement: Literal["advise", "auto"] = "auto" if auto_mode else "advise"
    ctx.tier_basis = tier_basis
    ctx.enforcement = enforcement

    # Run-policy keys: the renderer and the report read these off the manifest,
    # so they are published before the first stage that could fail.
    manifest.run.render_engine = config.render_engine
    manifest.run.translate_chrome = config.translate_chrome
    manifest.run.cover_mode = config.cover_mode
    manifest.run.ocr_mode = config.ocr_mode
    if config.ocr_endpoint:
        manifest.run.ocr_endpoint = config.ocr_endpoint

    figure_pages: set[int] = set()
    if ctx.source_pdf_path is not None:
        try:
            figure_pages = await asyncio.to_thread(
                detect_figure_pages, ctx.input_path, await ctx.current_blocks()
            )
        except Exception as exc:
            logger.debug("Figure page detection skipped for %s: %s", ctx.job_id, exc)
    # force_refresh on both reads: this stage runs after repair/consistency/
    # triage have mutated block statuses on disk — the ingest-era snapshot would score stale repair states.
    current_blocks = await ctx.current_blocks(force_refresh=True)
    advisory: Advisory = await asyncio.to_thread(
        advise_layout,
        current_blocks,
        tier_basis,
        profile=ctx.profile_name,
        figure_pages=figure_pages,
    )
    ctx.advisory = advisory
    effective_mode: DualMode = (
        advisory.recommended if auto_mode and advisory.tier == "discourage" else tier_basis
    )
    # Rigid typesetting is monolingual: downgrade and record here so the
    # quality report and the render adapter agree on the artifact.
    engine_advisory_msg: str | None = None
    effective_engine: str
    if config.render_engine in RIGID_ENGINES:
        effective_engine = config.render_engine
    else:
        effective_engine = (
            getattr(getattr(ctx, "adaptive_policy", None), "render_engine", None)
            or getattr(getattr(ctx, "adapter", None), "render_engine", None)
            or config.render_engine
        )
        if effective_engine == "auto":
            effective_engine = resolve_pdf_engine("auto", current_blocks, manifest=manifest)
    if (
        ctx.source_pdf_path is not None
        and effective_engine in RIGID_ENGINES
        and effective_mode != "monolingual"
    ):
        manifest.run.dual_mode_downgraded = effective_mode
        effective_mode = "monolingual"
        engine_name = effective_engine
        # Both rigid names reflow onto the source page, so the way back to a
        # bilingual artifact is `reflow` — the old text sent 'overlay' users to
        # 'publication', which is that same engine under its other name.
        engine_advisory_msg = (
            f"Render-engine advisory: '{engine_name}' is monolingual; the requested dual "
            f"mode '{manifest.run.dual_mode_downgraded}' was downgraded to "
            f"'monolingual'. Use --render-engine reflow (or --preset publication) "
            "for bilingual output."
        )
        logger.warning("Job %s: %s", ctx.job_id, engine_advisory_msg)
    elif (
        ctx.source_pdf_path is not None
        and canonical_render_engine(config.render_engine) == "publication"
        and (
            config.emit_companion_rigid
            or (
                resolve_pdf_engine("auto", current_blocks, manifest=manifest) == "rigid"
                and advisory.tier == "discourage"
            )
        )
    ):
        manifest.run.emit_secondary_engine = "rigid"
        engine_advisory_msg = (
            f"Layout tradeoff advisory: '--render-engine {config.render_engine}' was explicitly forced "
            f"on a formula/structure-dense PDF where auto routing selects 'rigid'. "
            "Proceeding with reflow bilingual rendering as requested, and scheduling a zero-cost "
            "companion '*_rigid.pdf' artifact for layout-faithful verification."
        )
        logger.warning("Job %s: %s", ctx.job_id, engine_advisory_msg)
    adv_dict = dict(advisory.to_dict())
    adv_dict["effective"] = effective_mode
    adv_dict["rendered_modes"] = [effective_mode]
    manifest.run.bilingual_advisory = adv_dict
    manifest.run.dual_enforcement = enforcement
    manifest.run.bilingual_mode = RENDER_MODE_VALUE[effective_mode]
    manifest.run.effective_dual_mode = effective_mode
    manifest.run.facing_spread = config.facing_spread or effective_mode in (
        "facing",
        "facing_spread",
    )
    manifest.run.render_engine_advisory = engine_advisory_msg
    if advisory.tier != "ok":
        logger.warning(
            "Bilingual advisory for job %s: requested '%s' is '%s' "
            "(recommended '%s', enforcement '%s'). Reasons: %s",
            ctx.job_id,
            tier_basis,
            advisory.tier,
            advisory.recommended,
            enforcement,
            "; ".join(advisory.reasons),
        )
    advise_msg = (
        f"Render-mode advisory: requested '{tier_basis}' is '{advisory.tier}'; "
        f"effective '{effective_mode}' (recommended '{advisory.recommended}')"
    )
    if engine_advisory_msg:
        advise_msg += f" | {engine_advisory_msg}"
    event = await ctx.create_event(
        EventType.MODE_ADVISED, ctx.job_id, ctx.ledger, message=advise_msg
    )
    yield event


async def run_difficulty_advisory_stage(ctx: StageContext) -> None:
    """Step the render mode down on live repair burden. Yields no events.

    Only ``auto`` enforcement may move the user's choice, and only one notch
    (inline -> alternating -> monolingual). The short chain fixed its mode in
    phase 1, so difficulty is recorded for the report but cannot downgrade.
    """
    config = ctx.config
    manifest = ctx.manifest
    advisory = ctx.advisory
    if advisory is None:  # pragma: no cover - phase 1 always runs first
        return
    post_stats = await asyncio.to_thread(ctx.ledger.get_job_stats, ctx.job_id)
    difficulty = assess_difficulty(
        total=int(post_stats.get("total", 0)),
        repaired=int(post_stats.get("repaired", 0)),
        failed=int(post_stats.get("failed", 0)),
    )
    pre_downgrade = str(manifest.run.effective_dual_mode or ctx.tier_basis)
    effective_mode: DualMode = ctx.tier_basis
    if not ctx.short_chain:
        effective_mode = resolve_effective_mode(
            ctx.tier_basis, advisory, difficulty, enforcement=ctx.enforcement
        )
    else:
        effective_mode = pre_downgrade  # type: ignore[assignment]
    effective_engine: str
    if config.render_engine in RIGID_ENGINES:
        effective_engine = config.render_engine
    else:
        effective_engine = (
            getattr(getattr(ctx, "adaptive_policy", None), "render_engine", None)
            or getattr(getattr(ctx, "adapter", None), "render_engine", None)
            or config.render_engine
        )
        if effective_engine == "auto":
            current_blocks = await ctx.current_blocks()
            effective_engine = resolve_pdf_engine("auto", current_blocks, manifest=manifest)
    if (
        ctx.source_pdf_path is not None
        and effective_engine in RIGID_ENGINES
        and effective_mode != "monolingual"
    ):
        manifest.run.dual_mode_downgraded = effective_mode
        effective_mode = "monolingual"
    if effective_mode != pre_downgrade:
        logger.warning(
            "Bilingual advisory downgrade for job %s: '%s' -> '%s' (%s)",
            ctx.job_id,
            pre_downgrade,
            effective_mode,
            "; ".join(difficulty.reasons),
        )
    manifest.run.bilingual_mode = RENDER_MODE_VALUE[effective_mode]
    manifest.run.effective_dual_mode = effective_mode
    manifest.run.facing_spread = config.facing_spread or effective_mode in (
        "facing",
        "facing_spread",
    )
    advisory_dict = dict(advisory.to_dict())
    advisory_dict["effective"] = effective_mode
    advisory_dict["enforcement"] = ctx.enforcement
    advisory_dict["difficulty"] = {
        "hard": difficulty.hard,
        "repair_share": difficulty.repair_share,
        "reasons": list(difficulty.reasons),
    }
    if effective_mode != pre_downgrade:
        advisory_dict["difficulty_downgrade"] = f"{pre_downgrade} -> {effective_mode}"
    secondary: DualMode | None = secondary_mode(effective_mode) if config.emit_both else None
    advisory_dict["rendered_modes"] = (
        [effective_mode] if secondary is None else [effective_mode, secondary]
    )
    manifest.run.bilingual_advisory = advisory_dict
    manifest.run.emit_secondary_mode = RENDER_MODE_VALUE[secondary] if secondary is not None else ""


def apply_layout_tradeoff_advisory(
    manifest: BookManifest,
    *,
    input_name: str,
    formula_heavy: bool,
    requested_engine: str,
) -> None:
    """Surface the rigid-on-formula-dense tradeoff in the delivered report.

    The rigid engine keeps figures and multi-row tables exactly where they are
    (the reflow engine shatters them — arXiv 2609.20519 lost 4/6 figures and
    Table 1), at the cost of imperfect inline-math typography. ``auto`` selects
    rigid for these documents on purpose, so this is an advisory the reader can
    weigh, never an UNSUITABLE stop.

    The criterion resolves the engine the run will actually use:
    ``requested_engine`` is still ``auto`` here, so comparing it to ``rigid``
    directly would silently skip the advisory on auto-routed documents — exactly
    when the reader did not choose the tradeoff and most needs to see it.
    """
    active = resolve_render_engine_from_signals(
        requested_engine, has_math=formula_heavy, struct_share=0.0, has_geometry=True
    )
    if not (formula_heavy and active == "rigid"):
        return
    advisory_msg = (
        f"Document '{input_name}' is formula-dense and is being "
        "rendered with the overlay (rigid) engine. This preserves the "
        "source page geometry — figures and complex tables stay intact — "
        "but inline math is typeset into the source's text boxes, so "
        "dense equations may render with imperfect spacing or sizing. "
        "For native math typography at the cost of figure/table fidelity, "
        "re-run with --render-engine reflow."
    )
    logger.info("Pre-flight Layout Advisory: %s", advisory_msg)
    manifest.run.delivery_status = (
        "LAYOUT_TRADEOFF_ADVISORY (formula-dense document rendered "
        "with the overlay engine: figures/tables preserved, inline "
        "math typography may be imperfect)"
    )
    manifest.run.delivery_warning = advisory_msg
