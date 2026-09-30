"""The stage plan: order and gating, stated once (ADR-0001 ``pipeline/run.py``).

The orchestrator owns the run's resources -- the writer lock, ledger, adapter,
router, and the per-run collaborators -- and this module owns *what runs in what
order*. Keeping the two apart is the first slice of the ADR's orchestration
convergence: the plan becomes one readable sequence with named gates instead of a
stretch of the resource lifecycle, so a step's position and its gate are visible
without the ``try/finally`` around them. It is also where the pure steps of
:mod:`ubt.pipeline.steps` land as the shared
:class:`~ubt.core.engine.stage_context.StageContext` is retired.

The plan is deliberately a function, not a table of callables: the
chapter-streaming branch and the export terminal hook are real control flow, and
encoding them as data would hide the order rather than state it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

from ubt.core.engine.events import EventType, TranslationProgressEvent
from ubt.core.engine.stage_context import StageContext
from ubt.core.engine.stages import (
    run_bible_stage,
    run_c_text_stage,
    run_chapter_streaming_pipeline,
    run_consistency_stage,
    run_cost_preflight_stage,
    run_difficulty_advisory_stage,
    run_draft_stage,
    run_export_stage,
    run_extraction_witness_stage,
    run_ingest_stage,
    run_mode_advisory_stage,
    run_quality_gate_stage,
    run_render_preflight_stage,
    run_repair_stage,
    run_triage_stage,
)
from ubt.pipeline.facts import RunFacts

#: Called with the terminal export event, *before* it is yielded.
ExportHook = Callable[[TranslationProgressEvent], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class RunGates:
    """The run's gating decisions, resolved before the plan runs.

    Each one picks a branch of the plan. Resolving them up front (from the config
    and the manifest) keeps the plan free of ``config`` reads, so the sequence
    states *what* runs, not *how* the decision was reached.
    """

    chapter_streaming: bool
    c_text: bool
    consistency: bool


async def run_stages(
    ctx: StageContext,
    gates: RunGates,
    facts: RunFacts,
    *,
    on_export_completed: ExportHook | None = None,
) -> AsyncIterator[TranslationProgressEvent]:
    """Run the stage plan, yielding every progress event in order.

    ``facts`` carries the values a stage produces and a later one consumes (see
    :mod:`ubt.pipeline.facts`); the plan threads the value each stage reads, so
    the inter-stage flow is an explicit parameter rather than a field on the
    shared context.

    ``on_export_completed`` runs on the terminal export event *before* it is
    yielded, so a caller that stops iterating immediately does not lose the
    terminal persistence (TM writeback, finalize hook) to ``GeneratorExit``.
    """
    async for event in run_ingest_stage(ctx):
        yield event
    await run_extraction_witness_stage(ctx)
    async for event in run_mode_advisory_stage(ctx, facts.layout):
        yield event
    # Zero-token preflights BEFORE the bible: the bible stage's skeleton
    # extraction and abbreviation backfill are billable calls, and the
    # render/cost preflight exists to fail before any spend.
    await run_render_preflight_stage(ctx)
    await run_cost_preflight_stage(ctx)
    async for event in run_bible_stage(ctx, facts.terminology):
        yield event
    if gates.chapter_streaming:
        async for event in run_chapter_streaming_pipeline(ctx, facts.terminology, facts.scoring):
            yield event
    else:
        async for event in run_draft_stage(ctx, facts.terminology):
            yield event
        if gates.c_text:
            async for event in run_c_text_stage(ctx):
                yield event
        async for event in run_quality_gate_stage(ctx, facts.terminology, facts.scoring):
            yield event
        async for event in run_repair_stage(ctx, facts.terminology, facts.scoring):
            yield event
    if gates.consistency:
        async for event in run_consistency_stage(ctx, facts.terminology, facts.scoring):
            yield event
    async for event in run_triage_stage(ctx, facts.terminology, facts.scoring):
        yield event
    await run_difficulty_advisory_stage(ctx, facts.layout)
    async for event in run_export_stage(ctx, facts.terminology):
        if event.event_type is EventType.EXPORT_COMPLETED and on_export_completed is not None:
            await on_export_completed(event)
        yield event


__all__ = ["ExportHook", "RunGates", "run_stages"]
