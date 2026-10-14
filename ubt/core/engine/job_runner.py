"""The one place both job executors build their orchestrator.

Two executors run jobs: the embedded :class:`ubt.api.manager.JobManager` (an
in-process asyncio task) and the queue :class:`ubt.core.engine.job_worker.JobWorker`
(one process draining the durable queue). They are *not* two implementations of
one job — each deployment mode drives exactly one of them — but they built the
orchestrator with the same twelve-line block, and that block has drifted
before: the manager's hand-written copy silently dropped the quality preset
until the shared ``overrides_from_request`` mapping replaced it.

This factory is that block, once. The *decision* of whether to rehearse stays
with each executor (the API honours its own process's ``dry_run``; the worker
recomputes from its own key, overriding an API auto-decision — see
``JobWorker._should_rehearse``), and is passed in.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

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


def build_job_orchestrator(
    config: UBTConfig,
    *,
    router: ModelRouter | None,
    qe_runner: BaseQERunner | None,
    rate_limiter: AdaptiveTokenBucket | None,
    finalize_job: FinalizeJobHook | None,
    rehearse: bool,
) -> PipelineOrchestrator:
    """Build the orchestrator a job executes on (real or rehearsal).

    ``rehearse`` selects the zero-token wiring (echo provider, mocked QE, page
    egress closed — see :func:`create_dry_run_orchestrator`); everything else
    is identical between the two, which is the point of this function.
    """
    if rehearse:
        return create_dry_run_orchestrator(config, finalize_job=finalize_job)
    return PipelineOrchestrator(
        config=config,
        router=router,
        qe_runner=qe_runner,
        rate_limiter=rate_limiter,
        finalize_job=finalize_job,
    )


def resolve_job_paths(payload: dict[str, Any]) -> tuple[Path, Path | None]:
    """The input/output paths every executor reads from a job payload."""
    input_path = Path(str(payload["input_path"]))
    raw_output = payload.get("output_path")
    return input_path, Path(str(raw_output)) if raw_output else None


__all__ = ["FinalizeJobHook", "build_job_orchestrator", "resolve_job_paths"]
