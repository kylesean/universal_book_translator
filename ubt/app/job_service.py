"""The application-layer entry point for building a job's pipeline.

Both job executors (the embedded API manager and the durable queue worker) and
the CLI run the same pipeline; this is the one place their orchestrator is
constructed. Keeping it in ``ubt.app`` is what stops ``ubt.core`` from growing a
second, drifting copy of the wiring, and it is the seam a future ``JobService``
surface (CLI/API/MCP) shares.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ubt.core.config import UBTConfig
from ubt.core.engine.dry_run import create_dry_run_orchestrator
from ubt.core.engine.events import TranslationProgressEvent
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.pipeline import PipelineOrchestrator
from ubt.core.qe import BaseQERunner
from ubt.core.router.rate_limiter import AdaptiveTokenBucket
from ubt.core.router.router import ModelRouter

#: The orchestrator's completion hook: called inside the writer-lock block with
#: the live ledger, so the caller's final artifact-triple write carries the
#: same single-writer guard as every stage write.
FinalizeJobHook = Callable[[TranslationProgressEvent, SQLiteJobLedger], None]


@dataclass(frozen=True)
class JobService:
    """Builds the one orchestrator a job executes on.

    ``rehearse`` selects the zero-token wiring (echo provider, mocked QE, page
    egress closed — see :func:`create_dry_run_orchestrator`); everything else is
    identical between the two. ``config`` owns the delivery decision; the
    pipeline resolves it before it builds an adapter.
    """

    config: UBTConfig
    router: ModelRouter | None = None
    qe_runner: BaseQERunner | None = None
    rate_limiter: AdaptiveTokenBucket | None = None
    finalize_job: FinalizeJobHook | None = None

    def orchestrator(self, *, rehearse: bool) -> PipelineOrchestrator:
        from ubt.app.bootstrap import install_services

        install_services()
        if rehearse:
            return create_dry_run_orchestrator(self.config, finalize_job=self.finalize_job)
        return PipelineOrchestrator(
            config=self.config,
            router=self.router,
            qe_runner=self.qe_runner,
            rate_limiter=self.rate_limiter,
            finalize_job=self.finalize_job,
        )


__all__ = ["FinalizeJobHook", "JobService"]
