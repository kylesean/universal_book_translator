"""Shared per-app state for the UBT API route registrars.

``create_app`` builds one of these and hands it to each ``_register_*`` function
in :mod:`ubt.api.app`. It is parameter packing (config + manager + the job/artifact
helpers), not a service layer: the handlers keep their original local names via
the registrar's binding preamble.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ubt.api.app import JobManager
    from ubt.core.config import UBTConfig
    from ubt.core.engine.job_queue import JobQueue
    from ubt.core.ir.models import IRBlock


class ApiScope:
    """The app-wide collaborators and path helpers every registrar needs."""

    def __init__(
        self,
        *,
        config: UBTConfig,
        manager: JobManager,
        job_queue: JobQueue | None,
        assess_semaphore: asyncio.Semaphore,
    ) -> None:
        self.config = config
        self.manager = manager
        self.job_queue = job_queue
        self.assess_semaphore = assess_semaphore

    def tenant_allows(self, job_id: str, tenant: str) -> bool:
        """False when a queued job exists but belongs to another tenant.

        Tenant is a starvation/isolation boundary, not authentication — the API
        key still grants whole-service access. Scoping reads and cancels stops
        one tenant from reading or cancelling another tenant's job. Embedded
        (non-queue) mode has no tenant dimension, so it always allows.
        """
        if self.job_queue is None:
            return True
        job = self.job_queue.get(job_id)
        return job is None or job.tenant_id == tenant

    async def tenant_allows_async(self, job_id: str, tenant: str) -> bool:
        if self.job_queue is None:
            return True
        job = await asyncio.to_thread(self.job_queue.get, job_id)
        return job is None or job.tenant_id == tenant

    def artifact_path(self, valid_id: str, key: str) -> str | None:
        """Disk fallback for an artifact path a completed job persisted.

        The in-memory ``JobRecord`` is pruned/evicted across restarts, so the
        path is read back from whichever durable store actually holds it:
        ``output_file``/``report_file``/``visual_report_file`` live in the job
        ledger for an embedded run and in the queue row's payload for
        ``job_mode="queue"`` (the worker never writes ledger metadata). Before
        both were consulted a queued job 404'd its own download forever.
        """
        from ubt.core.engine.ledger import SQLiteJobLedger as _Ledger

        db_path = self.config.db_dir / f"{valid_id}.sqlite"
        if db_path.exists():
            try:
                with _Ledger(db_path, read_only=True) as ledger:
                    value = ledger.get_job_metadata_value(valid_id, key)
            except Exception:
                value = None
            if value:
                return str(value)
        if self.job_queue is not None:
            job = self.job_queue.get(valid_id)
            if job is not None:
                queued = (job.progress or {}).get(key)
                if queued:
                    return str(queued)
        return None

    async def artifact_path_async(self, valid_id: str, key: str) -> str | None:
        return await asyncio.to_thread(self.artifact_path, valid_id, key)

    async def primary_output_file(self, valid_id: str) -> str | None:
        """The primary artifact path for a job, from memory then durable stores."""
        record = self.manager.get_job(valid_id)
        output_file = record.progress.output_file if record else None
        if not output_file:
            output_file = await self.artifact_path_async(valid_id, "output_file")
        return output_file

    def job_db_path(self, valid_id: str) -> Path:
        return self.config.db_dir / f"{valid_id}.sqlite"

    async def job_is_running(self, valid_id: str) -> bool:
        """True when the job is actively running (an interactive edit must not race it)."""
        from ubt.core.engine.job_queue import JobStatus as _JobStatus

        record = self.manager.get_job(valid_id)
        if record is not None and record.status == _JobStatus.RUNNING:
            return True
        if self.job_queue is not None:
            job = await asyncio.to_thread(self.job_queue.get, valid_id)
            if job is not None and job.status == _JobStatus.RUNNING:
                return True
        return False

    async def read_job_blocks(self, valid_id: str) -> list[IRBlock] | None:
        """All blocks for a job, or ``None`` when the ledger does not exist."""
        from ubt.core.engine.ledger import SQLiteJobLedger as _Ledger

        db_path = self.job_db_path(valid_id)
        if not db_path.exists():
            return None

        def _read() -> list[IRBlock]:
            with _Ledger(db_path, read_only=True) as ledger:
                return ledger.get_all_blocks(valid_id)

        return await asyncio.to_thread(_read)

    def managed_dir(self, name: str) -> Path:
        """A UBT-managed subtree under ``db_dir`` (or the operator allowlist).

        ``db_dir`` is an implicit sandbox base, so managed trees (uploaded
        sources, derived deliverables) pass :func:`resolve_secure_path` with no
        operator configuration. When an operator allowlist excludes ``db_dir``,
        placing files there would 403 on the very next request — fall back to
        the first allowlisted base instead.
        """
        from ubt.api.security import effective_allowed_bases

        db_sub = (self.config.db_dir / name).resolve()
        bases = effective_allowed_bases(self.config)
        if any(db_sub == b or b in db_sub.parents for b in bases):
            return db_sub
        if bases:
            return (bases[0] / name).resolve()
        return db_sub

    def uploads_dir(self) -> Path:
        """Staging area for documents uploaded through the web console."""
        return self.managed_dir("uploads")
