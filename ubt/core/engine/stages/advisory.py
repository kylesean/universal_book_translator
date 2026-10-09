"""Stages 1.4 / 1.5 / 5.6: extraction witness and the two bilingual advisories.

These three ran inline in ``PipelineOrchestrator.run()``, which is why the
generator was 800+ lines: none of them is a translation step, all three are
zero-token judgements about the artifact, and they read only post-ingest IR.

- :func:`run_extraction_witness_stage` flags pages whose math was already
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

from ubt.core.config import DualMode
from ubt.core.engine.blocks import BlockReader
from ubt.core.engine.events import EventType, TranslationProgressEvent
from ubt.core.engine.facts import LayoutAdvisory, RenderPlan
from ubt.core.engine.services import RunServices
from ubt.core.engine.stage_context import StageContext
from ubt.core.ir.models import BookManifest
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


async def run_extraction_witness_stage(ctx: StageContext, blocks: BlockReader) -> None:
    """Mark font-encoding damage on the ingested blocks. Yields no events.

    Math-bearing fonts with no ``/ToUnicode`` make every library guess the same
    wrong thing, so cross-channel agreement *hides* the damage; this looks for it
    directly and flags the confirmed pages while the text is still cheap to re-do.
    """
    if ctx.source_pdf_path is None:
        return
    ledger = ctx.ledger
    try:
        witness_verdicts = await asyncio.to_thread(inspect_font_encoding_damage, ctx.input_path)
        stats = summarize_font_encoding_damage(witness_verdicts)
        # force_refresh: this is the stage that *fills* the shared blocks cache,
        # and it mutates error_flags on disk right after the read. Handing the
        # freshly-populated snapshot to the next stage ensures mode advisory
        # reflects the post-witness state.
        witness_flagged = flag_font_encoding_damage(
            await blocks.current_blocks(force_refresh=True), witness_verdicts
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
                "source text are already wrong before translation — treat the "
                "affected pages as visual-fallback candidates",
                ctx.job_id,
                stats["confirmed_pages"],
                stats["pages"],
                stats["residue_chars"],
            )
    except Exception as exc:
        logger.debug("Extraction witness skipped for %s: %s", ctx.job_id, exc)


async def run_mode_advisory_stage(
    ctx: StageContext,
    layout: LayoutAdvisory,
    render: RenderPlan,
    blocks: BlockReader,
    services: RunServices,
) -> AsyncIterator[TranslationProgressEvent]:
    """Publish the run policy and advise on the requested bilingual mode."""
    config = ctx.config
    requested_mode: DualMode = (
        config.dual_mode if config.dual_mode in RENDER_MODE_VALUE else "inline"
    )
    auto_mode = config.dual_mode == "auto"
    tier_basis: DualMode = "inline" if auto_mode else requested_mode
    enforcement: Literal["advise", "auto"] = "auto" if auto_mode else "advise"
    layout.tier_basis = tier_basis
    layout.enforcement = enforcement

    # Run-policy keys: the renderer reads these off the plan (compiler
    # render plan protocol). They are passed as arguments, not posted on manifest.run.
    render.translate_chrome = config.translate_chrome
    render.cover_mode = config.cover_mode

    figure_pages: set[int] = set()
    if ctx.source_pdf_path is not None:
        try:
            figure_pages = await asyncio.to_thread(
                detect_figure_pages, ctx.input_path, await blocks.current_blocks()
            )
        except Exception as exc:
            logger.debug("Figure page detection skipped for %s: %s", ctx.job_id, exc)
    # force_refresh on both reads: this stage runs after repair/consistency/
    # triage have mutated block statuses on disk — the ingest-era snapshot would score stale repair states.
    current_blocks = await blocks.current_blocks(force_refresh=True)
    advisory: Advisory = await asyncio.to_thread(
        advise_layout,
        current_blocks,
        tier_basis,
        profile=ctx.profile_name,
        figure_pages=figure_pages,
    )
    layout.advisory = advisory
    effective_mode: DualMode = (
        advisory.recommended if auto_mode and advisory.tier == "discourage" else tier_basis
    )
    adv_dict = dict(advisory.to_dict())
    adv_dict["effective"] = effective_mode
    adv_dict["rendered_modes"] = [effective_mode]
    render.bilingual_advisory = adv_dict
    render.bilingual_mode = RENDER_MODE_VALUE[effective_mode]
    render.effective_dual_mode = effective_mode
    render.facing_spread = config.facing_spread or effective_mode in (
        "facing",
        "facing_spread",
    )
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
    event = await ctx.create_event(
        EventType.MODE_ADVISED, ctx.job_id, ctx.ledger, message=advise_msg
    )
    yield event


async def run_difficulty_advisory_stage(
    ctx: StageContext,
    layout: LayoutAdvisory,
    render: RenderPlan,
    blocks: BlockReader,
    services: RunServices,
) -> None:
    """Step the render mode down on live repair burden. Yields no events.

    Only ``auto`` enforcement may move the user's choice, and only one notch
    (inline -> alternating -> monolingual). The short chain fixed its mode in
    phase 1, so difficulty is recorded for the report but cannot downgrade.
    """
    config = ctx.config
    advisory = layout.advisory
    if advisory is None:  # pragma: no cover - phase 1 always runs first
        return
    post_stats = await asyncio.to_thread(ctx.ledger.get_job_stats, ctx.job_id)
    difficulty = assess_difficulty(
        total=int(post_stats.get("total", 0)),
        repaired=int(post_stats.get("repaired", 0)),
        failed=int(post_stats.get("failed", 0)),
    )
    pre_downgrade = str(render.effective_dual_mode or layout.tier_basis)
    effective_mode: DualMode = layout.tier_basis
    if not ctx.short_chain:
        effective_mode = resolve_effective_mode(
            layout.tier_basis, advisory, difficulty, enforcement=layout.enforcement
        )
    else:
        effective_mode = pre_downgrade  # type: ignore[assignment]
    # A source-canvas engine serves every mode, so no requested mode downgrades.
    if effective_mode != pre_downgrade:
        logger.warning(
            "Bilingual advisory downgrade for job %s: '%s' -> '%s' (%s)",
            ctx.job_id,
            pre_downgrade,
            effective_mode,
            "; ".join(difficulty.reasons),
        )
    render.bilingual_mode = RENDER_MODE_VALUE[effective_mode]
    render.effective_dual_mode = effective_mode
    render.facing_spread = config.facing_spread or effective_mode in (
        "facing",
        "facing_spread",
    )
    advisory_dict = dict(advisory.to_dict())
    advisory_dict["effective"] = effective_mode
    advisory_dict["enforcement"] = layout.enforcement
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
    render.bilingual_advisory = advisory_dict
    render.emit_secondary_mode = RENDER_MODE_VALUE[secondary] if secondary is not None else ""


def apply_layout_tradeoff_advisory(
    manifest: BookManifest,
    *,
    input_name: str,
    formula_heavy: bool,
) -> None:
    """Surface the overlay-on-formula-dense tradeoff in the delivered report.

    The source-canvas composition keeps figures and multi-row tables exactly
    where they are, at the cost of imperfect inline-math typography. When a
    formula-dense document is rendered, this advisory lets the reader weigh the
    tradeoff; it is never an UNSUITABLE stop.
    """
    if not formula_heavy:
        return
    advisory_msg = (
        f"Document '{input_name}' is formula-dense and is being "
        "rendered with the overlay engine. This preserves the "
        "source page geometry — figures and complex tables stay intact — "
        "but inline math is typeset into the source's text boxes, so "
        "dense equations may render with imperfect spacing or sizing."
    )
    logger.info("Pre-flight Layout Advisory: %s", advisory_msg)
    manifest.run.delivery_status = (
        "LAYOUT_TRADEOFF_ADVISORY (formula-dense document rendered "
        "with the overlay engine: figures/tables preserved, inline "
        "math typography may be imperfect)"
    )
    manifest.run.delivery_warning = advisory_msg
