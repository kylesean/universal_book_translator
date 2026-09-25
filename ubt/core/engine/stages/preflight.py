"""Stages 2.9 / 2.95: prove the artifact can render and the run can be paid for.

Both are zero-token and both exist to fail *before* the first draft call buys a
book that cannot be delivered:

- :func:`run_render_preflight_stage` compiles a source sample through the real
  PDF renderer: without it a missing Typst binary surfaces only as a
  translated book that dies at stage 6, after the spend.
- :func:`run_cost_preflight_stage` prices the cheapest plausible draft pass and
  refuses the job when it already exceeds ``--budget-usd``.
"""

from __future__ import annotations

import logging

from ubt.core.engine.cost_estimate import estimate_draft_cost, measure_prefix_tokens
from ubt.core.engine.render_preflight import run_render_preflight
from ubt.core.engine.stage_context import StageContext
from ubt.core.exceptions import UBTError

logger = logging.getLogger(__name__)


async def run_render_preflight_stage(ctx: StageContext) -> None:
    """Compile a source sample with the configured renderer. Yields no events."""
    if not ctx.config.render_preflight_enabled or ctx.is_mock_run:
        # A mock run rehearses the renderer through its own pipeline already.
        return
    await run_render_preflight(
        adapter=ctx.require_adapter(),
        manifest=ctx.manifest,
        blocks=await ctx.current_blocks(),
        target_lang=ctx.target_lang,
    )


async def run_cost_preflight_stage(ctx: StageContext) -> None:
    """Refuse a job whose cheapest pass breaks the budget. Yields no events."""
    prefix_tokens = measure_prefix_tokens(
        ctx.router, target_lang=ctx.target_lang, source_lang=ctx.source_lang
    )
    run_estimate = (
        None
        if prefix_tokens is None
        else estimate_draft_cost(
            await ctx.current_blocks(),
            draft_model=ctx.config.draft_model,
            prefix_tokens=prefix_tokens,
            base_url=ctx.config.base_url,
        )
    )
    if run_estimate is None or not run_estimate.billable_blocks:
        return
    floor = run_estimate.cost_usd_cached
    logger.info("Job %s: %s", ctx.job_id, run_estimate.describe())
    if floor is not None and ctx.config.budget_usd is not None and floor > ctx.config.budget_usd:
        raise UBTError(
            f"Refused before the first request: the cheapest plausible draft pass "
            f"(${floor:.6g}) already exceeds --budget-usd "
            f"(${ctx.config.budget_usd:.6g}). Raise the budget, or unset it "
            "to run without a ceiling."
        )
