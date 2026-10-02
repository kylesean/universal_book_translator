"""One object carries a run's shared state into each pipeline stage.

Before this, every stage restated the same handful of arguments: ``ledger``,
``actual_job_id``, ``manifest``, ``target_lang``, ``source_lang``,
``create_event_fn``, ``concurrency_sem``… ``run_export_stage``
took 24 keyword arguments and ``run_draft_stage`` 21, and ``run()`` paid the cost
of that in ~180 lines of argument lists where a reader had to work out, per
argument, whether the value was per-run state, a config field, or a number a
previous stage produced. Two stages also shared a value only because both read
the same config field, so "what flows between stages" and "what the config holds"
were indistinguishable.

The split this enforces:

- :attr:`StageContext.config` / :attr:`~StageContext.router` /
  :attr:`~StageContext.manifest` — the run's inputs; a stage reads the config
  field it needs instead of receiving it as a parameter.
- What one stage *produces* and a later one consumes lives in
  :mod:`ubt.pipeline.facts` now, not here: the plan owns the value and hands it
  to the stage that reads it (``terminology`` to the draft/repair/triage/… stages,
  ``layout`` to the two advisories), so the inter-stage flow is an explicit,
  typed parameter.
- The collaborators a run *builds* for its stages (the QE runner, repair loop,
  FastPass filter, adaptive policy, in-flight semaphore, TM, HTML validator) live
  in :class:`~ubt.pipeline.services.RunServices`, which the plan constructs once
  and threads, so "what the run *was configured with*" is separate from "the
  per-run services it made for itself".
- What a *single* stage needs for itself (``mode`` and ``max_repairs`` for
  consistency, the ``visual_*`` knobs for export) stays in that stage's own
  signature, so it remains greppable as a stage-specific tunable.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ubt.core.config import UBTConfig
from ubt.core.engine.events import (
    EventType,
    TranslationProgressEvent,
)
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.exceptions import JobInterruptedError
from ubt.core.ir.models import BookManifest
from ubt.core.ports import DocumentAdapter
from ubt.core.router.router import ModelRouter


async def _noop_bill() -> None:
    """Default ``StageContext.bill_run_usage`` for tests that build no driver."""
    return None


class EventFactory(Protocol):
    """The orchestrator's progress-event builder, bound to the live job.

    Every stage yields events through this instead of reaching back into the
    orchestrator, which is what kept ``run()`` and the stages mutually coupled.
    """

    async def __call__(
        self,
        event_type: EventType,
        job_id: str,
        ledger: SQLiteJobLedger,
        message: str = "",
        active_block_id: str | None = None,
        artifact_path: str | None = None,
    ) -> TranslationProgressEvent: ...


@dataclass(frozen=True)
class StageContext:
    """The immutable inputs and per-run collaborators a pipeline stage reads.

    Frozen on purpose (unified pipeline stage context): no stage
    writes a field, and the run's one piece of mutable state -- the block
    snapshot -- now lives in :class:`~ubt.pipeline.blocks.BlockReader`, owned by
    the plan. ``frozen=True`` makes "a stage cannot mutate the run" a
    compile/runtime invariant rather than a convention.
    """

    # --- The run's inputs -------------------------------------------------
    config: UBTConfig
    router: ModelRouter
    ledger: SQLiteJobLedger
    manifest: BookManifest
    job_id: str
    input_path: Path
    source_lang: str
    target_lang: str
    profile_name: str

    # --- Driver callbacks -------------------------------------------------
    # The orchestrator reaching *into* a stage (progress events, and further
    # below billing and cancellation), not a collaborator the run built for
    # itself; these stay on the context.
    create_event: EventFactory

    # --- Per-run services -------------------------------------------------
    # The collaborators this run builds for its stages (QE runner, repair loop,
    # FastPass filter, adaptive policy, in-flight semaphore, TM, HTML validator)
    # live in :class:`~ubt.pipeline.services.RunServices`, constructed by the
    # plan and handed to the stages that use them.

    # --- Optional, and defaulted so a stage test can omit them -------------
    #: Only the stages that read or write the artifact need this; the QE,
    #: repair and triage stages never touch the adapter, and a test driving one
    #: of them should not have to invent an adapter to say so.
    adapter: DocumentAdapter | None = None
    output_path: Path | None = None

    # --- Decided before staging, read by several stages -------------------
    short_chain: bool = False
    #: Chapter window from the CLI: where a book resumes / stops, applied by
    #: ingest. ``start_chapter`` is 1-based and ``max_chapters`` counts from it.
    start_chapter: int = 1
    max_chapters: int | None = None
    #: Seed-only translation bible (short born-digital PDFs). Resolved during
    #: setup because it is a property of the input file, not of any stage.
    fast_lane: bool = False
    #: The raw-completion channel the bible backfill asks the provider for.
    #: Overridable so a test can make backfill a no-op without a fake router.
    complete_raw_fn: Callable[..., Awaitable[str]] | None = None
    #: Set by the driver so the export stage can bill exactly this run, at the
    #: moment it renders. A callable, not a snapshot: the numbers must be read
    measure_run_usage: Callable[[], dict[str, dict[str, int]]] = lambda: {}
    #: Persist this run's usage to the ledger on demand. The export stage calls
    #: it after the paid post-render visual gate so a report built from the
    #: ledger does not omit spend made after the last progress event. The
    #: callback may return the billed ``JobBill`` (the orchestrator's
    #: ``_bill_run`` does); the export stage ignores the return, so the field is
    #: ``Awaitable[object]`` rather than forcing a discarding wrapper.
    bill_run_usage: Callable[[], Awaitable[object]] = _noop_bill
    #: Cancellation token for cooperative early termination of long stages.
    cancel_token: asyncio.Event | None = None

    def check_cancelled(self) -> None:
        """Raise JobInterruptedError if cooperative cancellation was signaled.

        A *lost lease* is signalled separately (through ``JobWorker``'s lease
        check between events), never through this token: conflating them made a
        reclaimed job read ``cancelled`` and refuse the new owner's
        ``finalize_job('completed')``.
        """
        if self.cancel_token is not None and self.cancel_token.is_set():
            raise JobInterruptedError(f"Job {self.job_id} cancelled cooperatively")

    # --- Produced by an earlier stage -------------------------------------
    # Values one stage produces and a later one consumes live in
    # :mod:`ubt.pipeline.facts` now: the plan owns them and passes the value to
    # the stage that reads it, so the inter-stage flow is an explicit, typed
    # parameter instead of a field on this shared object. The block snapshot
    # lives in :class:`~ubt.pipeline.blocks.BlockReader`, owned by the plan.

    @property
    def is_mock_run(self) -> bool:
        """Whether the provider produces simulated text.

        Read off the provider rather than stored, so a caller that swaps the
        router's provider (``--dry-run``) cannot leave a stale flag behind.
        """
        return self.router.provider.is_mock

    def require_adapter(self) -> DocumentAdapter:
        """The adapter, or a clear error from the one stage kind that needs it.

        ``adapter`` is optional on the context because the QE / repair / triage
        stages never touch it; ingest and export are the stages that do, and they
        should fail with a sentence that names the problem rather than an
        ``AttributeError`` on ``None``.
        """
        if self.adapter is None:
            raise RuntimeError(
                f"stage for job {self.job_id} needs a document adapter, none was set"
            )
        return self.adapter

    @property
    def raw_completion(self) -> Callable[..., Awaitable[str]]:
        """The injected backfill channel, or the router's own."""
        return self.complete_raw_fn or self.router.complete_raw

    @property
    def source_pdf_path(self) -> Path | None:
        """The source PDF for the stages that need pixel-level evidence."""
        return self.input_path if self.input_path.suffix.lower() == ".pdf" else None
