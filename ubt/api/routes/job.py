"""The jobs surface: assess, upload, submit, list, cancel, resume, delete, status."""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import (
    FastAPI,
    File,
    Header,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse, JSONResponse

from ubt.api.job_catalog import list_job_summaries, page_scope_label, summarize_job
from ubt.api.models import (
    JobAssessRequest,
    JobAssessResponse,
    JobDeleteResponse,
    JobListResponse,
    JobStatusResponse,
    JobSubmitRequest,
    JobSubmitResponse,
    JobUploadResponse,
)
from ubt.api.routes.shared import (
    cross_tenant_404,
    is_own_prior_output,
    output_path_claimant,
    public_progress,
)
from ubt.api.scope import ApiScope
from ubt.api.security import resolve_secure_path, validate_job_id
from ubt.api.uploads import UPLOAD_MAX_BYTES
from ubt.core.config import credential_required, has_real_credential
from ubt.core.engine.job_queue import TERMINAL_JOB_STATUSES, JobStatus
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.progress import ProgressSnapshot
from ubt.core.engine.writer_lock import LedgerWriterLock
from ubt.core.exceptions import (
    BudgetExceededError,
    DocumentParseError,
    QueueDepthExceededError,
    ServerCapacityError,
    UBTError,
    UnsupportedDocumentFormatError,
)
from ubt.core.fs_perms import ensure_private_dir
from ubt.core.job_options import clean_source_stem, resolve_target_output
from ubt.core.router.router import ModelRouter

# Strong references to fire-and-forget tasks: the event loop holds only weak
# references, so an unreferenced task can be garbage-collected mid-flight and
# its work silently never happen.
background_tasks: set[asyncio.Task[None]] = set()

logger = logging.getLogger(__name__)


def _ledger_is_rehearsal(ledger: Any, job_id: str) -> bool:
    """Whether this job ran on the echo provider, read from its own ledger.

    ``run_identity.mock_run`` is recorded by the ingest stage on every run, so
    the ledger answers for a job no process currently tracks. A missing or
    malformed record means "not a rehearsal": every pre-identity ledger is a
    real run, and defaulting the other way would badge real deliveries as
    rehearsals.
    """
    identity = ledger.get_job_metadata_value(job_id, "run_identity")
    return bool(isinstance(identity, dict) and identity.get("mock_run"))


def _ledger_page_scope(ledger: Any, job_id: str) -> str | None:
    """The run's page scope (``"1-6"``) from the ledger, or ``None`` for a full run."""
    return page_scope_label(ledger.get_job_metadata_value(job_id, "selected_pages"))


def _progress_from_ledger(db_dir: Path, job_id: str) -> dict[str, Any] | None:
    """A terminal job's progress rebuilt from its ledger, or ``None``.

    The in-memory snapshot a finished job carries was folded from the events it
    emitted while running, so it preserves whatever counters that code produced
    at the time — including a denominator that counted skipped blocks. A job
    that finished before a fix kept reporting the old numbers forever, because
    nothing recomputed them. The ledger is the durable record and the query
    behind it is current, so a terminal job is rebuilt from there.

    Returns ``None`` for anything that cannot be read (no ledger, a malformed
    row): the endpoint must still answer, and a stale-but-valid snapshot beats a
    500.
    """
    from ubt.core.engine.ledger import SQLiteJobLedger
    from ubt.core.engine.progress import ProgressSnapshot

    db_path = db_dir / f"{job_id}.sqlite"
    if not db_path.exists():
        return None
    try:
        with SQLiteJobLedger(db_path, read_only=True) as ledger:
            snap = ledger.get_job_snapshot(job_id)
            if not snap:
                return None
            return ProgressSnapshot.from_ledger(
                snap, lambda key: ledger.get_job_metadata_value(job_id, key)
            ).model_dump()
    except Exception:  # a broken ledger must not break /status
        return None


#: The console's built index.html: the SPA shell the browser-navigation branch
#: of ``GET /jobs`` serves. Resolved from the package root (``ubt/api/static``),
#: not this file's directory -- the route modules live one level below since the
#: app split.
_STATIC_INDEX = Path(__file__).resolve().parent.parent / "static" / "index.html"


def register_job_routes(
    api_app: FastAPI, scope: ApiScope, router: ModelRouter | None = None
) -> None:
    """Routes for the job surface."""
    app_config = scope.config
    manager = scope.manager
    job_queue = scope.job_queue
    assess_semaphore = scope.assess_semaphore
    _tenant_allows = scope.tenant_allows
    _tenant_allows_async = scope.tenant_allows_async
    _artifact_path = scope.artifact_path
    _artifact_path_async = scope.artifact_path_async
    _primary_output_file = scope.primary_output_file
    _job_db_path = scope.job_db_path
    _job_is_running = scope.job_is_running
    _read_job_blocks = scope.read_job_blocks
    _managed_dir = scope.managed_dir
    _uploads_dir = scope.uploads_dir

    @api_app.post(
        "/jobs/assess",
        response_model=JobAssessResponse,
        tags=["Jobs"],
        summary="Assess a document quote, risks and route without spend",
    )
    async def assess_job(req: JobAssessRequest) -> dict[str, Any]:
        from ubt.core.assess import AssessmentError, assess_document_async

        # ``scope.config`` is a live property (it re-reads the file when that
        # changed); ``app_config`` is the value it returned at startup. Reading the
        # snapshot here made a freshly configured provider invisible: assess
        # reported ``draft_model: ""`` and priced nothing, while the panel beside it
        # named the model that would actually run.
        cfg = scope.config
        # Price the job the caller is about to submit, not the one the operator's
        # file defaults to. ``apply_config_overrides`` is the same mapping the
        # submit path uses (manager.execute_job / job_worker), so the quote and the
        # run read one provider block, one OCR mode and one vision setting.
        #
        # Without the provider this quoted ``[defaults]``'s models against the
        # dropdown's selection: a user who picked a cheap local block still saw the
        # expensive cloud block's dollar figure. ``ocr_mode`` mattered for the same
        # reason — cloud OCR adds a vision line, and the quote never learned the
        # mode, so it silently priced 0 page calls.
        #
        # ``pages`` is deliberately NOT taken from the override chain: it rides in
        # as an explicit argument (``assess_document_async(pages=...)``) and the
        # quote scales itself by the selected slice. Applying it as a config
        # override too would scale the block count a second time.
        from ubt.core.job_options import apply_config_overrides, overrides_from_request

        assess_request = req.model_dump(exclude={"pages"})
        try:
            cfg = apply_config_overrides(
                cfg, overrides_from_request(assess_request, allow_provider_keys=False)
            )
        except UBTError as exc:
            # An unknown provider name is a caller error naming the block, not a
            # 500 — the submit path already answers this way for the same input.
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

        resolved_in = resolve_secure_path(req.input_path, must_exist=True, config=cfg)

        async def _assess() -> dict[str, Any]:
            try:
                report = await assess_document_async(
                    resolved_in,
                    cfg,
                    deep=req.deep,
                    target_lang=req.target_lang,
                    source_lang=req.source_lang,
                    pages=req.pages,
                )
                payload = report.to_dict()
                # The quote echoed the source file's absolute path, handing every
                # caller that can reach this (unauthenticated-by-default) port the
                # host's directory layout. Every other surface reduces to a
                # basename; do the same here (the quote needs the file *name*, not
                # where it lives).
                if isinstance(payload.get("path"), str):
                    payload["path"] = Path(payload["path"]).name
                return payload
            except AssessmentError as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={"code": exc.code, "message": str(exc)},
                ) from exc

        # Every assess — deep or not — runs under the same slots. A "shallow"
        # assess is not a cheap manifest read: a PDF still runs the full pdfium
        # page census + font-encoding witness (assess._pdf_facts), and every
        # other format opens its container and probes its archetype
        # (analyze_archetype → zip member reads for EPUB/DOCX) before
        # decide_route. The old code gated only deep and PDF requests and let a
        # non-PDF shallow request run completely unbounded, so a burst of them
        # piled concurrent container work onto the process with no ceiling.
        # Only the deep path fast-fails when saturated (it is minutes long);
        # a shallow request waits rather than reject.
        if req.deep and assess_semaphore.locked():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Server at capacity: too many deep assessments in flight",
            )
        async with assess_semaphore:
            return await _assess()

    @api_app.post(
        "/jobs/upload",
        response_model=JobUploadResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["Jobs"],
        summary="Stage a source document server-side; returns the input_path to submit",
    )
    async def upload_source_document(
        file: Annotated[UploadFile, File()],
    ) -> JobUploadResponse:
        # Browsers cannot send a usable filesystem path (File.path is an
        # Electron-only property), so the wizard uploads the bytes here and
        # submits the returned server-side path instead of a client guess.
        from ubt.adapters.factory import supported_suffixes

        original_name = Path(file.filename or "").name.strip()
        if not original_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing file name",
            )
        suffix = Path(original_name).suffix.lower()
        known = supported_suffixes()
        if suffix not in known:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported document format '{suffix or '(none)'}'. Supported: {', '.join(known)}.",
            )

        # Collisions cannot clobber a prior upload: every staged file gets a
        # timestamped, uuid-prefixed name. The readable stem is kept so the
        # operator can still tell the staged copies apart on disk.
        stem = (
            re.sub(r"[^\w.\- ]+", "_", Path(original_name).stem, flags=re.UNICODE).strip()
            or "upload"
        )
        uploads_dir = ensure_private_dir(_uploads_dir())
        dest = (
            uploads_dir / f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}-{stem}{suffix}"
        )

        size = 0
        try:
            with dest.open("wb") as out:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > UPLOAD_MAX_BYTES:
                        raise HTTPException(
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail=(
                                f"Uploaded file exceeds the {UPLOAD_MAX_BYTES // (1024 * 1024)} MB limit"
                            ),
                        )
                    out.write(chunk)
        except HTTPException:
            dest.unlink(missing_ok=True)
            raise
        except OSError as exc:
            dest.unlink(missing_ok=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to store upload: {exc}",
            ) from exc
        finally:
            await file.close()

        return JobUploadResponse(file_path=str(dest), file_name=original_name, size_bytes=size)

    @api_app.post(
        "/jobs/submit",
        response_model=JobSubmitResponse,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["Jobs"],
    )
    async def submit_job(
        req: JobSubmitRequest,
        x_ubt_tenant: str | None = Header(default=None),
    ) -> JobSubmitResponse:
        resolved_in = resolve_secure_path(req.input_path, must_exist=True, config=app_config)
        requested_id = validate_job_id(req.job_id) if req.job_id else None
        # Resolve the id once (client-supplied or server-generated) so the
        # default deliverable directory can be derived per-job before enqueue.
        submit_id = requested_id or f"job_{uuid.uuid4().hex[:12]}"

        # Submit idempotency: if this job_id is already known and still live (or
        # completed), return it before checking output collisions (a completed
        # run's output file already exists). A failed/cancelled id falls through
        # so ``enqueue`` re-runs it instead of handing back a dead job.
        if job_queue is not None and requested_id is not None:
            existing_queued = await asyncio.to_thread(job_queue.get, requested_id)
            if existing_queued is not None:
                # Tenant isolation: a cross-tenant probe must not learn the job
                # exists (the read routes already return 404, but idempotency
                # here returned it outright — an existence/status oracle).
                if existing_queued.tenant_id != scope.tenant_from_header(x_ubt_tenant):
                    raise cross_tenant_404(requested_id)
                if existing_queued.status not in (
                    JobStatus.FAILED,
                    JobStatus.CANCELLED,
                ):
                    return JobSubmitResponse(
                        job_id=existing_queued.job_id,
                        status=existing_queued.status.value,
                        stream_url=f"/jobs/{existing_queued.job_id}/stream",
                        status_url=f"/jobs/{existing_queued.job_id}/status",
                        rehearsal=bool(existing_queued.payload.get("dry_run", False)),
                        reused=True,
                    )
        if requested_id is not None:
            existing = manager.get_job(requested_id)
            # Embedded parity with the queue branch above: a failed/cancelled id
            # falls through so a fresh job replaces the dead record instead of
            # handing back a task that will never run.
            if existing is not None and existing.status not in (
                JobStatus.FAILED,
                JobStatus.CANCELLED,
            ):
                return JobSubmitResponse(
                    job_id=existing.job_id,
                    status=existing.status,
                    stream_url=f"/jobs/{existing.job_id}/stream",
                    status_url=f"/jobs/{existing.job_id}/status",
                    rehearsal=existing.request.dry_run,
                    reused=True,
                )

        if req.output_path:
            resolved_out = resolve_secure_path(req.output_path, must_exist=False, config=app_config)
            target_candidate = (
                resolve_target_output(resolved_out, resolved_in)
                if resolved_out is not None
                else None
            )
            if (
                target_candidate is not None
                and target_candidate.exists()
                and not is_own_prior_output(
                    requested_id,
                    target_candidate,
                    job_queue,
                    manager,
                    input_path=resolved_in,
                )
            ):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="output_path already exists; refusing to overwrite it",
                )
        else:
            # Server-derived default: deliverables live in their own per-job
            # directory under the managed ``outputs`` tree — never beside the
            # staged upload, which would mix sources with translations. Passing
            # the directory allows the export stage to derive the honest
            # deliverable name (_mono or _bilingual) dynamically based on the
            # resolved profile and bilingual_mode, rather than pre-allocating
            # a hardcoded filename.
            default_out_dir = ensure_private_dir(_managed_dir("outputs") / submit_id)
            resolved_out = resolve_secure_path(default_out_dir, must_exist=False, config=app_config)
            stem = clean_source_stem(resolved_in)
            for candidate in (
                default_out_dir / f"{stem}_mono{resolved_in.suffix}",
                default_out_dir / f"{stem}_bilingual{resolved_in.suffix}",
            ):
                if candidate.exists() and not is_own_prior_output(
                    requested_id, candidate, job_queue, manager, input_path=resolved_in
                ):
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail="output_path already exists; refusing to overwrite it",
                    )

        # Reject when another live job already claimed this output path. The
        # filesystem exists() check above cannot see an un-written target, so
        # the job maps are the only guard against two concurrent pipelines
        # last-writer-wins-ing the same deliverable.
        if resolved_out is not None:
            # Off-loop: the durable-queue half of the claim scan reads up to 1000
            # queue rows synchronously; on the event loop that stalls every other
            # request (SSE, other submits) for the duration.
            claimant = await asyncio.to_thread(
                output_path_claimant, manager, job_queue, submit_id, resolved_out
            )
            if claimant is not None:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"output_path already claimed by live job {claimant}",
                )

        resolved_glossary: str | None = None
        if req.glossary:
            # Same sandbox as input_path: a glossary is a filesystem path, so it
            # must clear resolve_secure_path before the job may read it.
            resolved_glossary = str(
                resolve_secure_path(req.glossary, must_exist=True, config=app_config)
            )

        safe_req = req.model_copy(
            update={
                "input_path": str(resolved_in),
                "output_path": str(resolved_out) if resolved_out is not None else None,
                "glossary": resolved_glossary,
            }
        )
        # No usable credential and no injected router → a hosted endpoint can
        # only be a mock: auto-set rehearsal so every downstream surface (submit
        # response, status, queue payload, worker) labels it instead of silently
        # reporting a mock delivery as a real one.
        #
        # A loopback endpoint is exempt. It needs no credential, so requiring one
        # turned a correctly configured local model into a rehearsal the operator
        # never asked for — the same defect the worker had.
        auto_rehearsal = False
        # Live config, not the startup snapshot: a provider saved from the console
        # must be usable by the very next submit, which is the whole point of
        # writing it to a file the engine re-reads.
        live = scope.config
        if (
            not safe_req.dry_run
            and router is None
            and credential_required(live)
            and not has_real_credential(live)
        ):
            safe_req = safe_req.model_copy(update={"dry_run": True})
            auto_rehearsal = True
            logger.warning(
                "No API key configured for a remote endpoint: auto-setting dry_run "
                "rehearsal for input %s",
                safe_req.input_path,
            )
        if job_queue is not None:
            queued_id = submit_id
            queued_payload = safe_req.model_dump()
            if auto_rehearsal:
                # The worker recomputes rehearsal from its OWN key; this
                # marker says the dry_run was our auto-decision, not a user
                # request, so a keyed worker runs for real.
                queued_payload["rehearsal_auto"] = True
            try:
                # ``enqueue`` runs ``BEGIN IMMEDIATE`` with a 30s busy_timeout;
                # calling it inline on the event loop stalls every other request
                # under write contention.
                job = await asyncio.to_thread(
                    job_queue.enqueue,
                    queued_id,
                    queued_payload,
                    tenant_id=scope.tenant_from_header(x_ubt_tenant),
                    priority=int(req.priority),
                )
            except QueueDepthExceededError as exc:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=str(exc),
                ) from exc
            except UBTError as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(exc),
                ) from exc
            return JobSubmitResponse(
                job_id=job.job_id,
                status=job.status.value,
                stream_url=f"/jobs/{job.job_id}/stream",
                status_url=f"/jobs/{job.job_id}/status",
                rehearsal=safe_req.dry_run,
            )
        try:
            record = manager.create_job(safe_req, job_id=submit_id)
        except (ServerCapacityError, QueueDepthExceededError) as exc:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=str(exc),
            ) from exc
        except UnsupportedDocumentFormatError as exc:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=str(exc),
            ) from exc
        except DocumentParseError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        except BudgetExceededError as exc:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail=str(exc),
            ) from exc
        except (RuntimeError, UBTError) as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

        # ``scope.config``, not the boot snapshot: the worker must translate with
        # the provider the operator just saved, not the one that existed when
        # the server started.
        record.task = asyncio.create_task(manager.execute_job(record, scope.config))

        return JobSubmitResponse(
            job_id=record.job_id,
            status=JobStatus.SUBMITTED,
            stream_url=f"/jobs/{record.job_id}/stream",
            status_url=f"/jobs/{record.job_id}/status",
            rehearsal=safe_req.dry_run,
        )

    @api_app.get("/jobs", response_model=JobListResponse, tags=["Jobs"])
    async def list_jobs(
        request: Request,
        limit: int = 200,
        x_ubt_tenant: str | None = Header(default=None),
    ) -> Any:
        """The job queue: every ledger in ``db_dir``, newest first.

        Reads the durable store (one ``{job_id}.sqlite`` per job) so a restarted
        console still lists finished jobs, and overlays live jobs with the
        manager's in-memory progress (fresher than the ``job_meta`` row a run
        only finalizes at the end).

        ``GET /jobs`` is also the console's Mission Control URL, so a browser
        navigation (``Accept: text/html``) is answered with the SPA shell and
        the client router resolves the screen; API clients get the JSON queue.
        """
        if "text/html" in request.headers.get("accept", ""):
            index = _STATIC_INDEX
            if index.exists():
                return FileResponse(index, media_type="text/html")

        def _live_overrides() -> dict[str, dict[str, Any]]:
            overrides: dict[str, dict[str, Any]] = {}
            for job_id, record in manager.jobs.items():
                progress = record.progress
                overrides[job_id] = {
                    "status": record.status,
                    "total_blocks": progress.total_blocks,
                    "completed_blocks": progress.completed_blocks,
                    "failed_blocks": progress.failed_blocks,
                    "needs_human_blocks": progress.needs_human_blocks,
                    "progress_percent": progress.progress_percent,
                    "estimated_cost_usd": progress.estimated_cost_usd,
                }
            return overrides

        def _catalog() -> list[dict[str, Any]]:
            # Queue rows first: a job submitted but not yet claimed has no ledger,
            # and listing only ledgers made a queue of ten look like an empty
            # console until a worker started. ``list_job_summaries`` merges these
            # by job_id and prefers the ledger's own numbers where both exist.
            pending = job_queue.list_jobs(limit=min(max(limit, 1), 1000)) if job_queue else []
            return list_job_summaries(
                app_config.db_dir,
                live=_live_overrides(),
                limit=min(max(limit, 1), 1000),
                queued=pending,
            )

        summaries = await asyncio.to_thread(_catalog)
        tenant = scope.tenant_from_header(x_ubt_tenant)
        if tenant is not None and job_queue is not None:
            # Match the per-job endpoints' rule exactly: a job still in the queue
            # is visible only to its tenant; a job the queue no longer tracks
            # (finished and pruned) is visible to any authenticated caller.
            def _visible() -> list[dict[str, Any]]:
                kept: list[dict[str, Any]] = []
                for item in summaries:
                    queued = job_queue.get(item["job_id"])
                    if queued is None or queued.tenant_id == tenant:
                        kept.append(item)
                return kept

            summaries = await asyncio.to_thread(_visible)
        return {"jobs": summaries}

    @api_app.post(
        "/jobs/{job_id}/cancel",
        tags=["Jobs"],
    )
    async def cancel_job(
        job_id: str, x_ubt_tenant: str | None = Header(default=None)
    ) -> dict[str, str]:
        """Cancel an in-flight job (idempotent: terminal jobs return as-is)."""
        valid_id = validate_job_id(job_id)
        if not await _tenant_allows_async(valid_id, scope.tenant_from_header(x_ubt_tenant)):
            raise cross_tenant_404(valid_id)
        if job_queue is not None:
            job = await asyncio.to_thread(job_queue.request_cancel, valid_id)
            if job is None:
                raise HTTPException(status_code=404, detail=f"Job not found: {valid_id}")
            return {"job_id": valid_id, "status": job.status.value}
        record = manager.get_job(valid_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Job not found: {valid_id}")
        if record.status not in (JobStatus.SUBMITTED, JobStatus.RUNNING):
            return {"job_id": valid_id, "status": record.status}
        record.status = JobStatus.CANCELLED
        task = record.task
        if task is not None and not task.done():
            task.cancel()

        def _finalize_cancelled_ledger() -> None:
            try:
                db_path = app_config.db_dir / f"{valid_id}.sqlite"
                if db_path.exists():
                    import time

                    from ubt.core.exceptions import LedgerWriterLockConflictError

                    # The pipeline holds the job-level writer lock for the whole
                    # run; taking it here too is what stops a cancellation from
                    # clobbering a still-writing pipeline's checkpoints (and vice
                    # versa). Wait briefly for the cancelled task to release it,
                    # then finalize; if it is still held, the pipeline itself
                    # will land the terminal status.
                    lock = LedgerWriterLock(db_path, valid_id)
                    deadline = time.monotonic() + 2.0
                    while True:
                        try:
                            lock.acquire()
                            break
                        except LedgerWriterLockConflictError:
                            if time.monotonic() >= deadline:
                                logger.debug(
                                    "Cancellation for %s: writer lock still held; "
                                    "the running pipeline will finalize it.",
                                    valid_id,
                                )
                                return
                            time.sleep(0.1)
                    try:
                        with SQLiteJobLedger(db_path) as ledger:
                            # Prevent rewriting a finished job to CANCELLED if it
                            # already reached a terminal status.
                            if ledger.get_job_status(valid_id) in TERMINAL_JOB_STATUSES:
                                return
                            ledger.finalize_job(valid_id, status=JobStatus.CANCELLED)
                    finally:
                        lock.release()
            except Exception as exc:
                logger.debug("Could not persist cancellation for %s: %s", valid_id, exc)

        # Pin the task: without a strong reference the event loop may collect
        # it before the thread pool finishes, losing the terminal ledger write.
        task = asyncio.create_task(asyncio.to_thread(_finalize_cancelled_ledger))
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)
        return {"job_id": valid_id, "status": JobStatus.CANCELLED}

    @api_app.post("/jobs/{job_id}/resume", tags=["Jobs"])
    async def resume_job(
        job_id: str, x_ubt_tenant: str | None = Header(default=None)
    ) -> JSONResponse:
        """Re-run a failed/cancelled job from its ledger checkpoints (PRD §4.2.4).

        The pipeline resumes from the blocks still pending in the ledger, so no
        finished segment is re-translated. Resume reuses the *original* request
        (kept on the in-memory record), which keeps the run-identity guard happy
        — a reconstructed request would silently drop knobs and could be refused
        as a different profile/engine. That makes this a same-session recovery
        (the network-drop case); a job whose record is gone after a restart must
        be resubmitted instead.
        """
        valid_id = validate_job_id(job_id)
        if not await _tenant_allows_async(valid_id, scope.tenant_from_header(x_ubt_tenant)):
            raise cross_tenant_404(valid_id)
        if job_queue is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "Resume is available in embedded mode; in queue mode resubmit "
                    "the job to have a worker drain its pending blocks."
                ),
            )
        record = manager.get_job(valid_id)
        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"Job {valid_id} is not tracked by this server session; "
                    "resubmit it to resume from its ledger."
                ),
            )
        if record.status not in (JobStatus.FAILED, JobStatus.CANCELLED):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Job is {record.status}; only a failed or cancelled job can resume.",
            )
        # The resume path obeys the same concurrency cap as a fresh submit: a
        # batch of failed jobs restarted together would otherwise open N
        # pipelines on a server configured for one. Checked before the record
        # flips to SUBMITTED, so a refusal leaves it resumable.
        try:
            manager.ensure_capacity()
        except ServerCapacityError as exc:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=str(exc),
            ) from exc

        # Re-check the output-path claim that submit enforces: while this job
        # sat FAILED/CANCELLED its path is up for grabs, so a job submitted in
        # the meantime may already own it. Resuming anyway would run two
        # pipelines into one deliverable and the last writer wins. Reject
        # before the flip, like the capacity check above.
        if record.request.output_path:
            claimed = Path(record.request.output_path).resolve()
            for other in manager.jobs.values():
                if other.job_id == valid_id or other.status in (
                    JobStatus.FAILED,
                    JobStatus.CANCELLED,
                ):
                    continue
                if (
                    other.request.output_path
                    and Path(other.request.output_path).resolve() == claimed
                ):
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail=f"output_path already claimed by live job {other.job_id}",
                    )

        # Never re-fresh (that would discard the checkpoints) and clear the
        # previous failure so the record restarts clean.
        record.request = record.request.model_copy(update={"fresh": False})
        record.error = None
        record.progress = ProgressSnapshot()
        record.status = JobStatus.SUBMITTED
        # ``scope.config``, not the boot snapshot: the worker must translate with
        # the provider the operator just saved, not the one that existed when
        # the server started.
        record.task = asyncio.create_task(manager.execute_job(record, scope.config))
        return JSONResponse(
            {
                "job_id": valid_id,
                "status": record.status,
                "stream_url": f"/jobs/{valid_id}/stream",
                "status_url": f"/jobs/{valid_id}/status",
            }
        )

    @api_app.post("/jobs/{job_id}/reparse", tags=["Jobs"])
    async def reparse_job(
        job_id: str, x_ubt_tenant: str | None = Header(default=None)
    ) -> JSONResponse:
        """Throw away a finished job's parse and rebuild it from the source file.

        The one thing neither resume nor re-export can do. ``resume`` explicitly
        refuses to re-fresh (it would discard the checkpoints it exists to keep),
        and ``re-export`` only re-runs the export stage over the text already in
        the ledger — so a fix to how the *source document* is parsed (block
        splitting, table reconstruction, formula and figure crops) never reaches
        a job that has finished. The delivered file keeps the old artefacts no
        matter how many times it is re-rendered, because the ledger still names
        the PNGs the old code cut.

        This is expensive and destructive, and the API says so by requiring an
        explicit call: every block is discarded and the whole book is
        re-translated, incurring the provider cost of a first run. Human
        revisions in the ledger are lost with it.

        Only an in-session, terminal job can be re-parsed. Rebuilding the
        request from the ledger alone would silently drop the model, preset and
        output choices the run was made with — the same trap ``resume``
        documents — so a job whose record is gone must be resubmitted.
        """
        valid_id = validate_job_id(job_id)
        if not await _tenant_allows_async(valid_id, scope.tenant_from_header(x_ubt_tenant)):
            raise cross_tenant_404(valid_id)
        if job_queue is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "Re-parse is available in embedded mode; in queue mode "
                    "resubmit the job with fresh=true."
                ),
            )
        record = manager.get_job(valid_id)
        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"Job {valid_id} is not tracked by this server session; "
                    "resubmit it with fresh=true to re-parse from the source."
                ),
            )
        if record.status not in TERMINAL_JOB_STATUSES:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Job is {record.status}; only a finished job can be re-parsed.",
            )
        try:
            manager.ensure_capacity()
        except ServerCapacityError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

        # ``fresh`` discards the blocks; clearing the error/progress mirrors the
        # resume path so the row does not show the previous run's failure while
        # the new one is still starting.
        record.request = record.request.model_copy(update={"fresh": True})
        record.error = None
        record.progress = ProgressSnapshot()
        record.status = JobStatus.SUBMITTED
        record.task = asyncio.create_task(manager.execute_job(record, scope.config))
        return JSONResponse(
            {
                "job_id": valid_id,
                "status": record.status,
                "stream_url": f"/jobs/{valid_id}/stream",
                "status_url": f"/jobs/{valid_id}/status",
            }
        )

    @api_app.delete(
        "/jobs/{job_id}",
        response_model=JobDeleteResponse,
        tags=["Jobs"],
        summary="Remove a finished job's ledger and deliverables from the console",
    )
    async def delete_job(
        job_id: str, x_ubt_tenant: str | None = Header(default=None)
    ) -> JobDeleteResponse:
        """Delete one job's history: the db_dir ledger plus its deliverable dir.

        A running job must reach a terminal status first — deletion is history
        management, not a stop button. A non-terminal queue-mode record is
        likewise refused: the durable queue owns its rows (``prune_terminal``
        is their lifecycle) and removing the ledger underneath a worker would
        desynchronize its view.
        """
        valid_id = validate_job_id(job_id)
        if not await _tenant_allows_async(valid_id, scope.tenant_from_header(x_ubt_tenant)):
            raise cross_tenant_404(valid_id)

        record = manager.get_job(valid_id)
        if record is not None and record.status not in TERMINAL_JOB_STATUSES:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"Job is {record.status}; cancel it and wait for a "
                    "terminal status before deleting."
                ),
            )
        if job_queue is not None:
            queued = await asyncio.to_thread(job_queue.get, valid_id)
            if queued is not None and queued.status not in TERMINAL_JOB_STATUSES:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=(
                        f"Queue record is {queued.status.value}; only a "
                        "terminal job can be deleted."
                    ),
                )

        def _is_verified_job_ledger() -> bool:
            """Whether ``db_dir/{valid_id}.sqlite`` is a real job ledger for this id.

            ``validate_job_id`` constrains the *charset*, not the meaning:
            ``tm`` and ``job_queue`` are legal ids that name the shared
            translation memory (``db_dir/tm.sqlite``) and the live queue
            database (``db_dir/job_queue.sqlite``). An unconstrained
            ``glob(f"{valid_id}.sqlite*")`` therefore unlinked a service
            database — ``DELETE /jobs/tm`` destroyed every human_pe pair, and
            ``DELETE /jobs/job_queue`` destroyed the queue the API holds open.
            A job ledger is one whose ``job_meta`` table carries a row for this
            id; that is exactly the test the queue listing already runs
            (``job_catalog.summarize_job`` returns ``None`` for a non-ledger
            sqlite file, never raising).
            """
            ledger_path = app_config.db_dir / f"{valid_id}.sqlite"
            if not ledger_path.exists():
                return False
            summary = summarize_job(ledger_path)
            return summary is not None and str(summary.get("job_id")) == valid_id

        def _remove() -> tuple[bool, bool]:
            # Only a verified job ledger is unlinked, and only then are its
            # sidecars (the glob is literal: ``-shm``/``-wal``) removed with it.
            # A file that merely shares the id's name — a shared ``tm.sqlite``,
            # the ``job_queue.sqlite``, an operator's ``backup_2026.sqlite`` —
            # is left untouched.
            ledger_gone = False
            if _is_verified_job_ledger():
                for path in sorted(app_config.db_dir.glob(f"{valid_id}.sqlite*")):
                    path.unlink(missing_ok=True)
                    ledger_gone = True
            # The deliverable dir is namespaced by job id under the managed
            # ``outputs`` tree, so it is removed only when this id is (or was)
            # a real job — never for an id that merely collided with a service
            # file name.
            outputs_gone = False
            if ledger_gone or record is not None:
                outputs_dir = _managed_dir("outputs") / valid_id
                outputs_gone = outputs_dir.is_dir()
                if outputs_gone:
                    shutil.rmtree(outputs_dir, ignore_errors=True)
            return ledger_gone, outputs_gone

        ledger_gone, outputs_gone = await asyncio.to_thread(_remove)
        # Drop the queue row too. Leaving it makes the job reappear in /jobs as
        # COMPLETED with a download that no longer resolves, and makes re-submitting
        # the same id a no-op ("already completed") that never runs again.
        queued_gone = False
        if job_queue is not None:
            queued_gone = bool(await asyncio.to_thread(job_queue.delete, valid_id))
        manager.forget_job(valid_id)
        if not ledger_gone and not outputs_gone and not queued_gone and record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job not found: {valid_id}",
            )
        return JobDeleteResponse(
            job_id=valid_id, removed_ledger=ledger_gone, removed_outputs=outputs_gone
        )

    @api_app.get(
        "/jobs/{job_id}/status",
        response_model=JobStatusResponse,
        tags=["Jobs"],
    )
    def get_status(
        job_id: str, x_ubt_tenant: str | None = Header(default=None)
    ) -> JobStatusResponse:
        valid_id = validate_job_id(job_id)
        if not _tenant_allows(valid_id, scope.tenant_from_header(x_ubt_tenant)):
            raise cross_tenant_404(valid_id)

        def _ledger_facts() -> tuple[bool | None, str | None]:
            """``(rehearsal, page_scope)`` from the job's ledger, once it exists.

            The ledger is written by the process that actually ran the job, so
            it answers the "was this a rehearsal / a fragment?" question
            authoritatively. The queue row's payload records only what the API
            *requested*: a keyless API auto-sets ``dry_run`` while a keyed worker
            runs the job for real, and the API also explicitly defers to the
            worker (``rehearsal_auto``). Reporting the payload made ``/status``
            disagree with ``/jobs``, which reads the ledger. ``None`` means "no
            ledger yet" — the caller falls back to the request.
            """
            db_path = app_config.db_dir / f"{valid_id}.sqlite"
            if not db_path.exists():
                return None, None
            try:
                with SQLiteJobLedger(db_path, read_only=True) as ledger:
                    return (
                        _ledger_is_rehearsal(ledger, valid_id),
                        _ledger_page_scope(ledger, valid_id),
                    )
            except Exception as exc:
                logger.debug("Ledger facts unavailable for job %s: %s", valid_id, exc)
                return None, None

        ledger_rehearsal, ledger_page_scope = _ledger_facts()

        if job_queue is not None:
            job = job_queue.get(valid_id)
            if job is not None:
                return JobStatusResponse(
                    **public_progress(ProgressSnapshot.model_validate(job.progress).model_dump()),
                    job_id=job.job_id,
                    status=job.status.value,
                    created_at=datetime.fromtimestamp(job.enqueued_at, UTC),
                    error=job.error,
                    queue_position=job_queue.queue_position(valid_id),
                    # Before the first ledger write (still queued) the request is
                    # the only evidence; afterwards the run's own record wins.
                    rehearsal=(
                        bool(job.payload.get("dry_run", False))
                        if ledger_rehearsal is None
                        else ledger_rehearsal
                    ),
                    page_scope=ledger_page_scope,
                )
        record = manager.get_job(valid_id)
        if record:
            # A terminal job's in-memory snapshot was folded from the events it
            # emitted at the time, so it carries whatever denominator the code
            # had while it ran — a finished job kept reporting the old
            # ``244/259`` long after the query behind it had been corrected. A
            # terminal record has a ledger, and the ledger is the source of
            # truth, so rebuild from it.
            progress_payload = record.progress.model_dump()
            if record.status in TERMINAL_JOB_STATUSES:
                ledger_progress = _progress_from_ledger(app_config.db_dir, valid_id)
                if ledger_progress is not None:
                    progress_payload = ledger_progress
            return JobStatusResponse(
                **public_progress(progress_payload),
                job_id=record.job_id,
                status=record.status,
                created_at=record.created_at,
                error=record.error,
                rehearsal=(
                    record.request.dry_run if ledger_rehearsal is None else ledger_rehearsal
                ),
                page_scope=ledger_page_scope,
            )

        # Fallback: inspect persisted SQLite ledger on disk
        db_path = app_config.db_dir / f"{valid_id}.sqlite"
        if db_path.exists():
            try:
                with SQLiteJobLedger(db_path, read_only=True) as ledger:
                    snap = ledger.get_job_snapshot(valid_id)
                    if snap:
                        created_str = str(snap.get("created_at", ""))
                        try:
                            created_dt = datetime.fromisoformat(created_str)
                        except Exception:
                            # Malformed created_at in a corrupt/legacy ledger row:
                            # report a present-time placeholder rather than a
                            # fabricated earlier date — clients act on status,
                            # not on this field.
                            created_dt = datetime.now(UTC)

                        progress = ProgressSnapshot.from_ledger(
                            snap, lambda key: ledger.get_job_metadata_value(valid_id, key)
                        )
                        status_value = str(snap["status"])
                        error_note: str | None = None
                        if job_queue is None and status_value not in TERMINAL_JOB_STATUSES:
                            # Embedded mode has no durable worker and no startup
                            # reconcile, so a row a crash left mid-run will never
                            # move again. Reporting "initialized" forever keeps
                            # every polling client waiting on a job that no
                            # longer exists.
                            #
                            # But a non-terminal row is NOT evidence of a crash:
                            # the CLI and the API share ``.ubt/ledgers``, so a
                            # ``ubt translate`` running in another terminal is a
                            # perfectly live writer of exactly this ledger.
                            # Declaring it FAILED ("the server restarted") told the
                            # operator to re-run a job that was working, and
                            # following that advice collides with the writer lock
                            # or pays the provider twice. Ask the lease instead:
                            # a live holder means running, an expired or missing
                            # one means the writer is gone.
                            holder = LedgerWriterLock.lease_holder(db_path, valid_id)
                            if holder is None:
                                status_value = JobStatus.FAILED
                                error_note = (
                                    "Job was interrupted before it reached a terminal "
                                    "state (the server restarted); re-run it to resume."
                                )
                            else:
                                status_value = JobStatus.RUNNING
                        return JobStatusResponse(
                            **public_progress(progress.model_dump()),
                            job_id=str(snap["job_id"]),
                            status=status_value,
                            created_at=created_dt,
                            error=error_note,
                            # The ledger is the authority on what this run actually
                            # was. ``run_identity.mock_run`` is written by the
                            # ingest stage on every run, so a rehearsal reads as a
                            # rehearsal from any process; this branch used to leave
                            # ``rehearsal`` at its False default and a CLI
                            # rehearsal showed up as a real delivery.
                            rehearsal=_ledger_is_rehearsal(ledger, valid_id),
                            page_scope=_ledger_page_scope(ledger, valid_id),
                        )
            except Exception as exc:
                logger.exception("Failed to inspect ledger on disk for job %s: %s", valid_id, exc)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=(
                        "Database error inspecting job ledger; see server logs "
                        f"(job_id={valid_id})."
                    ),
                ) from exc

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {valid_id}",
        )
