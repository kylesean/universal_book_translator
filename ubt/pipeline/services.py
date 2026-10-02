"""The per-run services a stage reads (unified pipeline orchestration).

The shared :class:`~ubt.core.engine.stage_context.StageContext` used to carry
these as fields, which mixed "what the run *was configured with*" together with
"the collaborators the run *built for itself*". They are now one named value the
plan constructs once and hands to the stages that use them -- the same move as
:mod:`ubt.pipeline.facts` for inter-stage values and
:class:`~ubt.pipeline.blocks.BlockReader` for the run's mutable state.

A service is per-run and language-bound: a QE runner scored against the run's
target language, a repair loop sharing that runner, the FastPass filter, the
in-flight semaphore, the adaptive policy, the translation memory, the HTML delta
validator. The ``*_explicit`` flags stay beside the collaborator they describe
(whether the orchestrator *built* it or the caller *injected* it), so the
question "may this run re-bind the runner to its glossary?" is answered by the
same object that holds the runner.

The driver callbacks (``create_event``, ``measure_run_usage``, ``bill_run_usage``,
``cancel_token``, ``complete_raw_fn``) deliberately stay on ``StageContext``: they
are the orchestrator reaching *into* a stage, not collaborators the run built.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ubt.core.validators.html_delta import HTMLDeltaValidator

if TYPE_CHECKING:
    from ubt.core.config import UBTConfig
    from ubt.core.engine.repair_loop import RepairLoop
    from ubt.core.memory.tm import TranslationMemory
    from ubt.core.policy.adaptive_policy import AdaptivePolicy
    from ubt.core.qe.base import BaseQERunner
    from ubt.core.qe.fast_pass import FastPassFilter
    from ubt.core.router.router import ModelRouter


@dataclass(frozen=True)
class RunServices:
    """The collaborators a run builds for its stages (frozen: the plan owns them).

    ``frozen=True`` mirrors :class:`~ubt.core.engine.stage_context.StageContext`:
    a stage cannot swap a service out from under a later stage. The values the
    fields *reference* (the semaphore, the memory) remain mutable in the usual
    way; what is frozen is the binding.
    """

    fast_pass: FastPassFilter
    qe_runner: BaseQERunner
    repair_loop: RepairLoop
    adaptive_policy: AdaptivePolicy
    concurrency_sem: asyncio.Semaphore
    tm: TranslationMemory | None = None
    html_validator: HTMLDeltaValidator = field(default_factory=HTMLDeltaValidator)
    #: Whether the orchestrator built the runner/loop itself, or the caller
    #: injected one. Only a runner this pipeline built may be re-bound to the
    #: run's terminology; an injected loop must stay exactly as it was given.
    qe_runner_explicit: bool = False
    repair_loop_explicit: bool = False

    def repair_loop_for(
        self, qe_runner: BaseQERunner, *, config: UBTConfig, router: ModelRouter
    ) -> RepairLoop:
        """A repair loop around ``qe_runner``; an injected loop is handed back as-is.

        The loop shares the runner so the closed-loop re-score after a repair
        uses the identical policy. The result is *returned*, not written back:
        the run's own ``repair_loop`` stays what the caller supplied, and the
        terminology-bound one is a value the plan threads (see
        :mod:`ubt.pipeline.facts`).
        """
        if self.repair_loop_explicit:
            return self.repair_loop
        from ubt.core.engine.repair_loop import RepairLoop

        return RepairLoop(
            router=router,
            qe_runner=qe_runner,
            max_rounds=config.max_repair_rounds,
            qe_threshold=config.qe_threshold,
            bottom_percentile=config.bottom_percentile,
            rerank_k=config.rerank_k,
        )


__all__ = ["RunServices"]
