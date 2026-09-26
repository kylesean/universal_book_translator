"""Worker runner for the durable job queue.

``ubt worker`` (and the API in embedded mode) drains :class:`JobQueue` claims
by running each job through the existing :class:`PipelineOrchestrator` — the
pipeline is reused verbatim, not reimplemented. The worker owns the
cross-process concerns: it heartbeats the job lease while the pipeline
streams, persists progress so the API's SSE can serve it from another process,
honours a cancel request between events, and writes a terminal status
(completed / failed / cancelled) back to the queue.

The pipeline event source is injectable so the worker can be tested without a
provider, a PDF, or a ledger.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator, Callable
from contextlib import aclosing, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ubt.core.config import UBTConfig
from ubt.core.engine.dry_run import create_dry_run_orchestrator
from ubt.core.engine.events import TranslationProgressEvent
from ubt.core.engine.job_queue import JobQueue, JobStatus, QueuedJob
from ubt.core.engine.pipeline import PipelineOrchestrator
from ubt.core.engine.progress import ProgressSnapshot
from ubt.core.exceptions import (
    JobInterruptedError,
    LeaseLostError,
    LedgerWriterLockConflictError,
)
from ubt.core.job_options import (
    apply_config_overrides,
    overrides_from_request,
    run_kwargs_from_request,
)

if TYPE_CHECKING:
    from ubt.core.qe.base import BaseQERunner
    from ubt.core.router.rate_limiter import AdaptiveTokenBucket
    from ubt.core.router.router import ModelRouter

logger = logging.getLogger(__name__)

EventSource = Callable[[QueuedJob, UBTConfig], AsyncGenerator[TranslationProgressEvent, None]]


class JobWorker:
    """One process that drains queued jobs with ``concurrency`` slots."""

    def __init__(
        self,
        queue: JobQueue,
        config: UBTConfig,
        *,
        worker_id: str,
        concurrency: int = 1,
        lease_seconds: float = 60.0,
        poll_interval: float = 2.0,
        router: ModelRouter | None = None,
        qe_runner: BaseQERunner | None = None,
        rate_limiter: AdaptiveTokenBucket | None = None,
        event_source: EventSource | None = None,
    ) -> None:
        self.queue = queue
        self.config = config
        self.worker_id = worker_id
        self.concurrency = max(1, concurrency)
        self.lease_seconds = lease_seconds
        self.poll_interval = poll_interval
        self.router = router
        self.qe_runner = qe_runner
        # Shared across this worker's slots: `rate_limit_rpm` is a
        # per-credential budget, so a per-job bucket would multiply it by
        # `concurrency`.
        self.rate_limiter = rate_limiter
        self._event_source = event_source or self._default_event_source
        self._heartbeat_interval = max(1.0, lease_seconds / 3.0)
        self._stop = asyncio.Event()
        self.failed_jobs: int = 0

    def stop(self) -> None:
        """Ask every slot to exit after its current job."""
        self._stop.set()

    # -- pipeline wiring ------------------------------------------------------
    async def _default_event_source(
        self,
        job: QueuedJob,
        job_config: UBTConfig,
        cancel_token: asyncio.Event | None = None,
    ) -> AsyncGenerator[TranslationProgressEvent, None]:
        payload = {**job.payload, "job_id": job.job_id}
        input_path = Path(str(payload["input_path"]))
        output_path = Path(str(payload["output_path"])) if payload.get("output_path") else None
        if payload.get("dry_run"):
            # Rehearsal row (auto-set at intake when no key, or explicitly
            # requested): deterministic echo provider, mocked QE, no spend.
            orchestrator = create_dry_run_orchestrator(job_config)
        else:
            orchestrator = PipelineOrchestrator(
                config=job_config,
                router=self.router,
                qe_runner=self.qe_runner,
                rate_limiter=self.rate_limiter,
            )
        async for event in orchestrator.run(
            input_path=input_path,
            output_path=output_path,
            cancel_token=cancel_token,
            **run_kwargs_from_request(payload),
        ):
            yield event

    @staticmethod
    async def _q[T](fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Run a blocking ``JobQueue`` call off the event loop.

        The queue is plain ``sqlite3`` with a 30s ``busy_timeout``; calling it
        inline inside a coroutine stalls every slot's loop for that long, so
        heartbeats miss their window and ``reclaim_stale`` hands a still-running
        job to a second worker — the book is then billed twice and both workers
        checkpoint the same rows. Offloading keeps the lease fresh.
        """
        return await asyncio.to_thread(fn, *args, **kwargs)

    async def _write_abort_ledger(
        self,
        job: QueuedJob,
        job_config: UBTConfig | None,
        *,
        status: str,
        reason: str,
    ) -> None:
        """Land the abort's terminal ledger status, distinct from the queue's.

        The pipeline can only mark a bare ``failed`` when its generator is
        closed early — it cannot tell a user cancel from a lost lease. The
        worker does know, so it rewrites the ledger here: cancel ->
        ``cancelled``; lease loss -> ``failed`` with an ``abort_reason``. The
        queue row stays the source of truth for slot accounting; this is
        best-effort and never raises.
        """
        cfg = job_config or self.config
        db_path = Path(cfg.db_dir) / f"{job.job_id}.sqlite"

        def _write() -> None:
            from ubt.core.engine.ledger import SQLiteJobLedger

            if not db_path.exists():
                return
            try:
                with SQLiteJobLedger(db_path) as ledger:
                    if ledger.get_job_status(job.job_id) == "completed":
                        return
                    ledger.set_job_metadata_value(job.job_id, "abort_reason", reason)
                    ledger.finalize_job(job.job_id, status=status)
            except Exception as exc:  # best-effort terminal write
                logger.debug("Could not write abort ledger for %s: %s", job.job_id, exc)

        await asyncio.to_thread(_write)

    async def _heartbeat_loop(
        self,
        job_id: str,
        worker_id: str,
        lost: asyncio.Event,
        cancel_token: asyncio.Event | None = None,
    ) -> None:
        """Renew the lease; set ``lost`` when the queue refuses to, or ``cancel_token`` when cancel requested."""
        try:
            while True:
                await asyncio.sleep(self._heartbeat_interval)
                ok = await self._q(
                    self.queue.heartbeat, job_id, worker_id, lease_seconds=self.lease_seconds
                )
                if not ok:
                    # Logging alone would leave the job running: ``reclaim_stale``
                    # hands it to a second worker while the first keeps drafting,
                    # so the book is paid for twice and both workers checkpoint
                    # the same rows.
                    logger.warning("Job %s: lease lost, stopping heartbeats", job_id)
                    lost.set()
                    return
                if cancel_token is not None and await self._q(
                    self.queue.is_cancel_requested, job_id, worker_id
                ):
                    if not cancel_token.is_set():
                        logger.info(
                            "Job %s: cancellation requested; signalling cancel_token", job_id
                        )
                        cancel_token.set()
                    # Keep renewing the lease: the pipeline observes the token and
                    # unwinds, but until it does the job is still ours and must not
                    # be reclaimed mid-teardown (which would bill it twice).
                    continue
        except asyncio.CancelledError:
            return
        except Exception as exc:  # heartbeat failure is lease loss
            # A transient SQLite/OS error must not leave the heartbeat task dead
            # while execute() continues billing a lease another worker may
            # reclaim. Treat every unexpected heartbeat failure as lease loss;
            # the main loop will stop the generator at its next event.
            lost.set()
            logger.warning(
                "Job %s: heartbeat failed; treating lease as lost: %s",
                job_id,
                exc,
                exc_info=True,
            )

    # -- execution ------------------------------------------------------------
    async def execute(self, job: QueuedJob, worker_id: str | None = None) -> None:
        """Run one claimed job to a terminal queue state.

        ``worker_id`` must be the id that claimed the job (a slot id under
        concurrency); it defaults to this worker's own id.
        """
        import inspect

        owner = worker_id or self.worker_id
        lease_lost = asyncio.Event()
        cancel_token = asyncio.Event()
        heartbeat = asyncio.create_task(
            self._heartbeat_loop(job.job_id, owner, lease_lost, cancel_token)
        )
        progress: dict[str, Any] = {}
        job_config: UBTConfig | None = None
        try:
            # Config construction is inside the try on purpose (same shape as
            # api/manager and mcp/server): the queue payload persists across
            # versions, so one drifted enum value must FAILED the job, not
            # escape execute() and crash-loop the whole worker process.
            payload = {**job.payload, "job_id": job.job_id}
            job_config = apply_config_overrides(
                self.config, overrides_from_request(payload, allow_provider_keys=False)
            )
            # Inspect if custom event_source implementation accepts cancel_token.
            kwargs: dict[str, Any] = {}
            if "cancel_token" in inspect.signature(self._event_source).parameters:
                kwargs["cancel_token"] = cancel_token

            # ``aclosing`` guarantees the orchestrator's ``finally`` (ledger/
            # router teardown) runs the moment we leave the loop early on a lost
            # lease or cancel — not later, when the generator is GC'd — so no
            # further blocks get drafted (and billed) against a job we no longer
            # own.
            async with aclosing(self._event_source(job, job_config, **kwargs)) as events:
                async for event in events:
                    progress = ProgressSnapshot.from_event(event).to_payload()
                    await self._q(self.queue.update_progress, job.job_id, owner, progress)
                    if lease_lost.is_set():
                        raise LeaseLostError(f"job {job.job_id} lost its lease to another worker")
                    if cancel_token.is_set() or await self._q(
                        self.queue.is_cancel_requested, job.job_id, owner
                    ):
                        raise JobInterruptedError(f"job {job.job_id} cancelled by request")
            completed = await self._q(
                self.queue.complete,
                job.job_id,
                owner,
                status=JobStatus.COMPLETED,
                progress=progress,
            )
            if completed:
                logger.info("Job %s completed", job.job_id)
            else:
                # The lease had already been reclaimed (worker_id reset by the
                # stale-lease reaper), so this write matched 0 rows. The ledger
                # artifact was still finalized inside run(), but the queue row
                # now belongs to another state — surface the divergence rather
                # than log a false "completed".
                logger.warning(
                    "Job %s finished but its queue lease was already lost; "
                    "terminal write skipped (ledger holds the artifact)",
                    job.job_id,
                )
        except asyncio.CancelledError:
            # Task is tearing down; shield the terminal write so it still lands
            # off-loop rather than being cancelled mid-flight and leaving the job
            # stuck in ``running``.
            await asyncio.shield(
                self._q(
                    self.queue.complete,
                    job.job_id,
                    owner,
                    status=JobStatus.CANCELLED,
                    progress=progress,
                )
            )
            with suppress(Exception):
                await asyncio.shield(
                    self._write_abort_ledger(
                        job, job_config, status="cancelled", reason="cancelled_by_task"
                    )
                )
            logger.info("Job %s cancelled (task)", job.job_id)
            raise
        except LeaseLostError:
            # The new owner is still running this job, so the QUEUE row must get
            # no terminal write (that worker's complete() owns it). The LEDGER,
            # though, was marked a bare ``failed`` by the pipeline's early-close
            # handler; rewrite it with the real reason so infrastructure
            # failure is distinguishable from a user cancel.
            with suppress(Exception):
                await self._write_abort_ledger(
                    job, job_config, status="failed", reason="lease_lost"
                )
            logger.warning("Job %s: lease lost, stopping without a queue terminal write", job.job_id)
        except JobInterruptedError:
            await self._q(
                self.queue.complete,
                job.job_id,
                owner,
                status=JobStatus.CANCELLED,
                progress=progress,
            )
            with suppress(Exception):
                await self._write_abort_ledger(
                    job, job_config, status="cancelled", reason="cancelled_by_request"
                )
            logger.info("Job %s cancelled by request", job.job_id)
        except LedgerWriterLockConflictError as exc:
            # Another process currently holds the writer lock on disk (e.g. former worker
            # lost lease due to heartbeat jitter/GC pause but hasn't exited or released lock yet).
            # Do NOT mark the job as FAILED! Release the queue slot so it remains QUEUED,
            # allowing the previous or future worker to handle it without wasting attempts.
            logger.warning(
                "Job %s: writer lock held by another process; yielding slot back to QUEUED: %s",
                job.job_id,
                exc,
            )
            await self._q(
                self.queue.release_claim,
                job.job_id,
                owner,
                error=f"Yielded on writer lock contention: {exc}",
                decrement_attempt=True,
            )
            # Back off before the slot re-claims. The job is QUEUED again and
            # ``_slot`` claims immediately, so without this the same worker
            # grabs it, hits the same still-held lock, and spins: attempts are
            # never consumed (decrement_attempt=True), so ``run_until_idle``
            # never returns and the job is executed a second time the moment
            # the lock frees.
            await asyncio.sleep(self.poll_interval)
        except Exception as exc:  # a job failure is terminal, not fatal
            self.failed_jobs += 1
            logger.warning("Job %s failed: %s", job.job_id, exc, exc_info=True)
            await self._q(
                self.queue.complete,
                job.job_id,
                owner,
                status=JobStatus.FAILED,
                error=str(exc),
                progress=progress,
            )
        finally:
            heartbeat.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat

    async def process_one(self) -> bool:
        """Claim and run one job; ``False`` when the queue has nothing to give."""
        job = await self._q(
            self.queue.claim, self.worker_id, lease_seconds=self.lease_seconds, now=time.time()
        )
        if job is None:
            return False
        await self.execute(job)
        return True

    # -- loops ----------------------------------------------------------------
    async def _slot(self, worker_id: str, *, drain: bool) -> int:
        processed = 0
        while not self._stop.is_set():
            job = await self._q(
                self.queue.claim, worker_id, lease_seconds=self.lease_seconds, now=time.time()
            )
            if job is None:
                if drain:
                    return processed
                with suppress(TimeoutError):
                    await asyncio.wait_for(self._stop.wait(), timeout=self.poll_interval)
                continue
            await self.execute(job, worker_id)
            processed += 1
        return processed

    async def run_until_idle(self) -> int:
        """Drain queued jobs with all slots, then return the processed count."""
        slots = [
            asyncio.create_task(self._slot(f"{self.worker_id}-{i}", drain=True))
            for i in range(self.concurrency)
        ]
        try:
            results = await asyncio.gather(*slots)
        except BaseException:
            # A slot may fail outside execute() (queue/claim bugs, custom slot
            # hooks). Do not orphan sibling slots that would otherwise keep
            # claiming and billing jobs after the worker call has failed.
            for slot in slots:
                if not slot.done():
                    slot.cancel()
            await asyncio.gather(*slots, return_exceptions=True)
            raise
        return sum(results)

    async def run_forever(self) -> None:
        """Run until :meth:`stop` is called (or the task is cancelled)."""
        slots = [
            asyncio.create_task(self._slot(f"{self.worker_id}-{i}", drain=False))
            for i in range(self.concurrency)
        ]
        try:
            await asyncio.gather(*slots)
        except BaseException:
            self._stop.set()
            for slot in slots:
                if not slot.done():
                    slot.cancel()
            await asyncio.gather(*slots, return_exceptions=True)
            raise
