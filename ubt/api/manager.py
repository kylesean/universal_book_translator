"""In-memory job tracking and subscriber fan-out management for the UBT API."""

import asyncio
import logging
import uuid
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

from ubt.api.models import JobSubmitRequest
from ubt.core.config import UBTConfig
from ubt.core.engine.dry_run import create_dry_run_orchestrator
from ubt.core.engine.events import TranslationProgressEvent
from ubt.core.engine.job_queue import TERMINAL_JOB_STATUSES, JobStatus
from ubt.core.engine.pipeline import PipelineOrchestrator
from ubt.core.engine.progress import ProgressSnapshot, persist_progress_metadata
from ubt.core.exceptions import UBTError
from ubt.core.job_options import (
    apply_config_overrides,
    overrides_from_request,
    run_kwargs_from_request,
)
from ubt.core.qe import BaseQERunner
from ubt.core.router.rate_limiter import AdaptiveTokenBucket
from ubt.core.router.router import ModelRouter

logger = logging.getLogger(__name__)


class JobRecord:
    """Internal in-memory tracking structure for an active job."""

    def __init__(self, job_id: str, request: JobSubmitRequest) -> None:
        self.job_id = job_id
        self.request = request
        # A str-valued status from the shared JobStatus vocabulary; annotated so
        # callers (and tests) may assign the plain string value too.
        self.status: str = JobStatus.SUBMITTED
        self.created_at = datetime.now(UTC)
        # Counters, metrics and artifact paths: replaced wholesale by each
        # folded event, never patched field by field.
        self.progress = ProgressSnapshot()
        self.error: str | None = None

        self.subscribers: list[asyncio.Queue[TranslationProgressEvent | None]] = []
        self.task: asyncio.Task[None] | None = None


class JobManager:
    """Manages active job tasks and broadcast event fan-out with bounded memory retention."""

    _SUBSCRIBER_QUEUE_MAXSIZE = 256
    _MAX_SUBSCRIBERS_PER_JOB = 32

    def __init__(
        self,
        router: ModelRouter | None = None,
        qe_runner: BaseQERunner | None = None,
        max_retained_jobs: int = 100,
        max_running_jobs: int = 8,
        rate_limiter: AdaptiveTokenBucket | None = None,
    ) -> None:
        self.jobs: dict[str, JobRecord] = {}
        self.router = router
        self.qe_runner = qe_runner
        self.max_retained_jobs = max_retained_jobs
        self.max_running_jobs = max_running_jobs
        # One bucket for the process: `rate_limit_rpm` is a per-credential
        # budget, so a per-job bucket would let N concurrent jobs multiply the
        # request rate (and the bill) against the same API key.
        self.rate_limiter = rate_limiter

    def create_job(self, request: JobSubmitRequest, job_id: str | None = None) -> JobRecord:
        # Cap concurrent running jobs — each job is a full pipeline with
        # provider connections; an unbounded intake would exhaust memory and
        # hammer the translation provider's rate limits.
        running = sum(
            1
            for rec in self.jobs.values()
            if rec.status in (JobStatus.SUBMITTED, JobStatus.RUNNING)
        )
        if running >= self.max_running_jobs:
            raise UBTError(
                f"Server at capacity: {running} jobs active (max {self.max_running_jobs})",
                details={"running": running, "max": self.max_running_jobs},
            )
        self._prune_old_jobs()
        record = JobRecord(job_id=job_id or f"job_{uuid.uuid4().hex[:12]}", request=request)
        self.jobs[record.job_id] = record
        return record

    def _prune_old_jobs(self) -> None:
        """Evict oldest terminal jobs when in-memory job map exceeds max retention limit."""
        if len(self.jobs) < self.max_retained_jobs:
            return
        finished = [
            (jid, rec) for jid, rec in self.jobs.items() if rec.status in TERMINAL_JOB_STATUSES
        ]
        finished.sort(key=lambda item: item[1].created_at)
        to_remove = len(self.jobs) - self.max_retained_jobs + 1
        for jid, _ in finished[: max(0, to_remove)]:
            del self.jobs[jid]

    @classmethod
    def _enqueue(
        cls,
        sub_q: asyncio.Queue[TranslationProgressEvent | None],
        event: TranslationProgressEvent | None,
    ) -> None:
        """Non-blocking fan-out with drop-oldest backpressure.

        A stalled SSE consumer must never block the translation pipeline nor
        grow its queue without bound: when the subscriber's queue is full the
        oldest buffered event is discarded to make room for the new one.
        """
        while True:
            try:
                sub_q.put_nowait(event)
                return
            except asyncio.QueueFull:
                with suppress(asyncio.QueueEmpty):  # pragma: no cover - race guard
                    sub_q.get_nowait()

    def get_job(self, job_id: str) -> JobRecord | None:
        return self.jobs.get(job_id)

    async def execute_job(self, record: JobRecord, config: UBTConfig) -> None:
        record.status = JobStatus.RUNNING
        try:
            # Inside the try on purpose: an unparseable request field (a preset
            # or enum the caller invented) must mark the job failed and release
            # its SSE subscribers. Out here it escaped into the asyncio task,
            # leaving the record at "running" and the stream never terminating.
            input_file = Path(record.request.input_path)
            output_file = Path(record.request.output_path) if record.request.output_path else None

            # One shared request→config mapping (also applies the quality preset the
            # former hand-written block silently ignored).
            request_payload = record.request.model_dump()
            request_payload["job_id"] = record.job_id
            job_config = apply_config_overrides(
                config, overrides_from_request(request_payload, allow_provider_keys=False)
            )

            if record.request.dry_run:
                # Rehearsal: deterministic echo provider, mocked QE, no spend.
                # finalize_job still runs inside the orchestrator's writer-lock
                # block so the artifact triple persists exactly as in a real run.
                orchestrator = create_dry_run_orchestrator(
                    job_config,
                    finalize_job=self._persist_final_metadata(job_config, record),
                )
                logger.info("Job %s running as a zero-token rehearsal", record.job_id)
            else:
                orchestrator = PipelineOrchestrator(
                    config=job_config,
                    router=self.router,
                    qe_runner=self.qe_runner,
                    rate_limiter=self.rate_limiter,
                    # Persist the final artifact triple from inside the
                    # orchestrator's writer-lock block instead of re-opening the
                    # ledger here after run() released the lock.
                    finalize_job=self._persist_final_metadata(job_config, record),
                )
            async for event in orchestrator.run(
                input_path=input_file,
                output_path=output_file,
                **run_kwargs_from_request(request_payload),
            ):
                # Update state: one fold, one owner (ubt.core.engine.progress).
                record.progress = ProgressSnapshot.from_event(event)

                # Fan-out event to active subscribers (non-blocking)
                for sub_q in list(record.subscribers):
                    self._enqueue(sub_q, event)
            record.status = JobStatus.COMPLETED
        except asyncio.CancelledError:
            # CancelledError is a BaseException, so the `except Exception`
            # below never saw it and an externally-cancelled task left the
            # record at "running" — still counted against max_running_jobs.
            record.status = JobStatus.CANCELLED
            logger.warning("Job %s task cancelled", record.job_id)
            raise
        except Exception as exc:
            record.status = JobStatus.FAILED
            # Do not echo the raw exception: provider error bodies and parse
            # failures can contain source book text or host paths. The full
            # detail is logged server-side; the client gets a correlation id.
            logger.exception("Job %s failed: %s", record.job_id, exc)
            record.error = f"{type(exc).__name__} (see server logs; job_id={record.job_id})"
        finally:
            # Broadcast termination sentinel (non-blocking)
            for sub_q in list(record.subscribers):
                self._enqueue(sub_q, None)

    @staticmethod
    def _persist_final_metadata(
        job_config: UBTConfig, record: JobRecord
    ) -> Callable[[TranslationProgressEvent], None]:
        """Build the orchestrator's in-lock completion hook.

        ``PipelineOrchestrator.run`` invokes the returned callable *inside* its
        writer-lock block, so this write carries the same single-writer guard as
        every stage write. ``ProgressSnapshot.from_event`` is the same fold the
        SSE record uses, so the persisted artifact triple cannot drift from what
        subscribers were told.
        """

        def _persist(event: TranslationProgressEvent) -> None:
            persist_progress_metadata(
                event, Path(job_config.db_dir) / f"{record.job_id}.sqlite", record.job_id
            )

        return _persist
