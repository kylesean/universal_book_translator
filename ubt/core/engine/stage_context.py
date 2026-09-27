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
- The ``glossary_dicts`` / ``abbreviation_entries`` / ``block_count`` fields —
  written by the stage that produces them and read by the stages after it. That
  is the actual stage-to-stage data flow, now visible in one place.
- What a *single* stage needs for itself (``mode`` and ``max_repairs`` for
  consistency, the ``visual_*`` knobs for export) stays in that stage's own
  signature, so it remains greppable as a stage-specific tunable.

Tests drive one stage by building a context with the fields that stage reads; see
``tests/stage_ctx_factory.py``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from ubt.core.cleaners.citation_masker import CitationMasker
from ubt.core.cleaners.code_masker import CodeMasker
from ubt.core.cleaners.math_masker import MathMasker
from ubt.core.cleaners.soup_math import SoupMathMasker
from ubt.core.config import DualMode, UBTConfig
from ubt.core.engine.events import (
    EventType,
    TranslationProgressEvent,
)
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.repair_loop import RepairLoop
from ubt.core.exceptions import JobInterruptedError
from ubt.core.ir.models import BookManifest, IRBlock
from ubt.core.memory.bible import BookBible
from ubt.core.memory.tm import TranslationMemory
from ubt.core.policy.adaptive_policy import AdaptivePolicy
from ubt.core.policy.bilingual_advisor import Advisory
from ubt.core.ports import DocumentAdapter
from ubt.core.qe.base import BaseQERunner
from ubt.core.qe.fast_pass import FastPassFilter
from ubt.core.router.router import ModelRouter
from ubt.core.validators.html_delta import HTMLDeltaValidator


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


@dataclass
class StageContext:
    """State a pipeline stage reads, and the few values stages hand forward."""

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

    # --- Per-run collaborators (language-bound, so not orchestrator-owned) --
    fast_pass: FastPassFilter
    qe_runner: BaseQERunner
    repair_loop: RepairLoop
    adaptive_policy: AdaptivePolicy
    concurrency_sem: asyncio.Semaphore
    create_event: EventFactory

    # --- Optional, and defaulted so a stage test can omit them -------------
    #: Only the stages that read or write the artifact need this; the QE,
    #: repair and triage stages never touch the adapter, and a test driving one
    #: of them should not have to invent an adapter to say so.
    adapter: DocumentAdapter | None = None
    output_path: Path | None = None
    tm: TranslationMemory | None = None
    html_validator: HTMLDeltaValidator = field(default_factory=HTMLDeltaValidator)
    code_masker: CodeMasker = field(default_factory=CodeMasker)
    citation_masker: CitationMasker = field(default_factory=CitationMasker)
    math_masker: MathMasker = field(default_factory=MathMasker)
    soup_masker: SoupMathMasker = field(default_factory=SoupMathMasker)

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
    #: Whether the orchestrator was handed a QE runner / repair loop (only a
    #: runner this pipeline built itself may be re-bound to the run's
    #: terminology, and an injected loop must stay exactly as it was given).
    qe_runner_explicit: bool = False
    repair_loop_explicit: bool = False
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
        """Raise JobInterruptedError if cooperative cancellation was signaled."""
        if self.cancel_token is not None and self.cancel_token.is_set():
            raise JobInterruptedError(f"Job {self.job_id} cancelled cooperatively")

    # --- Produced by an earlier stage, consumed by a later one ------------
    #: The whole-book translation bible the bible stage assembled.
    bible: BookBible | None = None
    glossary_dicts: list[dict[str, Any]] = field(default_factory=list)
    abbreviation_entries: list[dict[str, Any]] = field(default_factory=list)
    block_count: int = 0
    #: Bilingual advisory state: phase 1 (layout) writes it, phase 2
    #: (post-repair difficulty) reads it to decide whether to downgrade a step.
    advisory: Advisory | None = None
    tier_basis: DualMode = "inline"
    enforcement: Literal["advise", "auto"] = "advise"

    # Backing store for :meth:`current_blocks`: the snapshot plus the ledger
    # block revision it was read at.
    _blocks: tuple[list[IRBlock], int] | None = field(default=None, repr=False)

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

    def rebuild_repair_loop(self) -> RepairLoop:
        """Rebuild the repair loop around a re-bound QE runner.

        The loop shares the runner so the closed-loop re-score after a repair
        uses the identical policy; an explicitly injected loop is the caller's
        object and is handed back untouched.
        """
        if self.repair_loop_explicit:
            return self.repair_loop
        config = self.config
        self.repair_loop = RepairLoop(
            router=self.router,
            qe_runner=self.qe_runner,
            max_rounds=config.max_repair_rounds,
            qe_threshold=config.qe_threshold,
            bottom_percentile=config.bottom_percentile,
            rerank_k=config.rerank_k,
        )
        return self.repair_loop

    @property
    def raw_completion(self) -> Callable[..., Awaitable[str]]:
        """The injected backfill channel, or the router's own."""
        return self.complete_raw_fn or self.router.complete_raw

    def invalidate_blocks_cache(self) -> None:
        """Drop the cached block snapshot so the next read comes from the ledger."""
        self._blocks = None

    async def current_blocks(self, force_refresh: bool = False) -> list[IRBlock]:
        """This job's blocks, read from SQLite.

        The snapshot is reused only while the ledger reports no block write, so a
        stage cannot be handed an outdated pre-write view of the book by forgetting to
        refresh.
        ``force_refresh`` and :meth:`invalidate_blocks_cache` remain for callers
        that changed the blocks in memory without a ledger write to report.
        """
        # Read the revision before the query: a write that lands mid-read leaves
        # the cached pair below looking older than it is, which costs one more
        # reload. Reading it afterwards would certify a snapshot that missed it.
        revision = self.ledger.blocks_seq
        if force_refresh or self._blocks is None or self._blocks[1] != revision:
            # Same thread discipline as the event factory: a full-table read
            # belongs off the loop an SSE server shares across jobs.
            blocks = await asyncio.to_thread(self.ledger.get_all_blocks, self.job_id)
            self._blocks = (blocks, revision)
        return self._blocks[0]

    @property
    def source_pdf_path(self) -> Path | None:
        """The source PDF for the stages that need pixel-level evidence."""
        return self.input_path if self.input_path.suffix.lower() == ".pdf" else None
