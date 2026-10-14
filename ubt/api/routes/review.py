"""The review surface: segments, human edits, term propagation, re-export, previews."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Query, status
from fastapi.responses import JSONResponse, Response

from ubt.api.models import SegmentEditRequest, TermPropagationRequest
from ubt.api.review import (
    ISSUE_KINDS,
    ReviewBlockNotFound,
    ReviewEditConflict,
    ReviewEditError,
    apply_human_edit,
    apply_term_propagation,
    segment_issue_kinds,
    segment_matches_filter,
    serialize_segment,
    term_cascade_report,
)
from ubt.api.routes.shared import UNPROCESSABLE_ENTITY, cross_tenant_404
from ubt.api.scope import ApiScope
from ubt.api.security import resolve_secure_path, validate_job_id
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.re_export import ReExportError, re_export_job
from ubt.render.page_preview import (
    PagePreviewUnavailable,
    render_page_preview,
    render_source_page_png,
)


def register_review_routes(api_app: FastAPI, scope: ApiScope) -> None:
    """Routes for the review surface."""
    app_config = scope.config
    _tenant_allows = scope.tenant_allows
    _tenant_allows_async = scope.tenant_allows_async
    _artifact_path = scope.artifact_path
    _artifact_path_async = scope.artifact_path_async
    _primary_output_file = scope.primary_output_file
    _job_db_path = scope.job_db_path
    _job_is_running = scope.job_is_running
    _read_job_blocks = scope.read_job_blocks
    _managed_dir = scope.managed_dir

    # -- L3 Review Workbench (segments + fault ribbon + human edit) -----------
    @api_app.get("/jobs/{job_id}/segments", tags=["Jobs"])
    async def list_segments(
        job_id: str,
        status_filter: str | None = Query(default=None, alias="status"),
        block_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
        x_ubt_tenant: str | None = Header(default=None),
    ) -> JSONResponse:
        """A page of a job's finalized blocks for the workbench grid.

        ``status=issues`` returns everything the fault ribbon counts plus the
        PE-queue members; a concrete ``BlockStatus`` value filters by lifecycle
        status. Blocks are projected with their source/target, QE score, flags
        and grouped issue kinds.
        """
        valid_id = validate_job_id(job_id)
        if not await _tenant_allows_async(valid_id, scope.tenant_from_header(x_ubt_tenant)):
            raise cross_tenant_404(valid_id)
        blocks = await _read_job_blocks(valid_id)
        if blocks is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Job not found: {valid_id}"
            )

        filtered = [b for b in blocks if segment_matches_filter(b, status_filter)]
        if block_type:
            filtered = [b for b in filtered if b.block_type.value == block_type]
        total = len(filtered)
        start = max(offset, 0)
        window = filtered[start : start + min(max(limit, 1), 500)]
        return JSONResponse(
            {
                "job_id": valid_id,
                "total": total,
                "segments": [serialize_segment(b) for b in window],
            }
        )

    @api_app.get("/jobs/{job_id}/issues", tags=["Jobs"])
    async def list_issues(
        job_id: str, x_ubt_tenant: str | None = Header(default=None)
    ) -> JSONResponse:
        """Fault-ribbon counts by issue kind, plus terminology drift detail."""
        valid_id = validate_job_id(job_id)
        if not await _tenant_allows_async(valid_id, scope.tenant_from_header(x_ubt_tenant)):
            raise cross_tenant_404(valid_id)
        blocks = await _read_job_blocks(valid_id)
        if blocks is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Job not found: {valid_id}"
            )

        counts: dict[str, int] = dict.fromkeys(ISSUE_KINDS, 0)
        status_counts = {"needs_human": 0, "blocked_human": 0, "failed": 0}
        for block in blocks:
            for kind in segment_issue_kinds(block):
                counts[kind] += 1
            if block.status.value in status_counts:
                status_counts[block.status.value] += 1

        term_drift: list[Any] = []
        report_file = await _artifact_path_async(valid_id, "report_file")
        if report_file:
            try:
                report_path = resolve_secure_path(report_file, must_exist=True, config=app_config)
                report_data = json.loads(
                    await asyncio.to_thread(report_path.read_text, encoding="utf-8")
                )
                term_drift = report_data.get("entity_consistency", {}).get("top_drifted", []) or []
            except (HTTPException, ValueError, OSError):
                term_drift = []

        return JSONResponse(
            {
                "job_id": valid_id,
                "counts": counts,
                "status": status_counts,
                "total_issues": sum(counts.values()),
                "term_drift": term_drift,
            }
        )

    @api_app.post("/jobs/{job_id}/segments/{block_id}", tags=["Jobs"])
    async def edit_segment(
        job_id: str,
        block_id: str,
        req: SegmentEditRequest,
        x_ubt_tenant: str | None = Header(default=None),
    ) -> JSONResponse:
        """Apply one human revision (ledger + shared TM), the L3 feedback path."""
        valid_id = validate_job_id(job_id)
        if not await _tenant_allows_async(valid_id, scope.tenant_from_header(x_ubt_tenant)):
            raise cross_tenant_404(valid_id)
        if not _job_db_path(valid_id).exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Job not found: {valid_id}"
            )
        if await _job_is_running(valid_id):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Job is running; stop it before editing segments.",
            )

        try:
            result = await asyncio.to_thread(
                apply_human_edit,
                _job_db_path(valid_id),
                valid_id,
                block_id,
                req.target_text,
                tm_path=app_config.db_dir / "tm.sqlite",
            )
        except ReviewBlockNotFound as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except ReviewEditConflict as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        except ReviewEditError as exc:
            raise HTTPException(status_code=UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

        updated = None
        blocks = await _read_job_blocks(valid_id)
        if blocks is not None:
            match = next((b for b in blocks if b.id == block_id), None)
            updated = serialize_segment(match) if match is not None else None
        return JSONResponse({**result, "job_id": valid_id, "segment": updated})

    @api_app.post("/jobs/{job_id}/re-export", tags=["Jobs"])
    async def re_export(
        job_id: str, x_ubt_tenant: str | None = Header(default=None)
    ) -> JSONResponse:
        """Re-render a finished job's deliverable from its current ledger text.

        A human edit lands in the ledger, but the delivered file was rendered
        before it — the reviewer saw the fix in the preview while the download
        stayed stale. This re-runs the export stage over the current text and
        overwrites the deliverable in place. No token is spent: every block is
        already terminal, so only render + reports run.
        """
        valid_id = validate_job_id(job_id)
        if not await _tenant_allows_async(valid_id, scope.tenant_from_header(x_ubt_tenant)):
            raise cross_tenant_404(valid_id)
        if not _job_db_path(valid_id).exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Job not found: {valid_id}"
            )
        if await _job_is_running(valid_id):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Job is running; stop it before re-exporting.",
            )

        try:
            event = await re_export_job(
                db_path=_job_db_path(valid_id),
                job_id=valid_id,
                config=app_config,
                router=scope.manager.router,
                qe_runner=scope.manager.qe_runner,
            )
        except ReExportError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        return JSONResponse(
            {
                "job_id": valid_id,
                "artifact_path": event.artifact_path,
                "message": event.message,
            }
        )

    @api_app.get("/jobs/{job_id}/segments/{block_id}/terms", tags=["Jobs"])
    async def segment_terms(
        job_id: str,
        block_id: str,
        x_ubt_tenant: str | None = Header(default=None),
    ) -> JSONResponse:
        """Terminology findings for one block, with the cascade size each implies.

        Each finding carries the offending ``surface``, the ``expected``
        rendering, and how many *other* blocks in the job carry the same
        error — ``cascade_all`` for the whole book, ``cascade_subsequent`` for
        blocks at or after this one. This is what the "fix all N" checkbox
        counts.
        """
        valid_id = validate_job_id(job_id)
        if not await _tenant_allows_async(valid_id, scope.tenant_from_header(x_ubt_tenant)):
            raise cross_tenant_404(valid_id)
        if not _job_db_path(valid_id).exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Job not found: {valid_id}"
            )

        report = await asyncio.to_thread(
            term_cascade_report, _job_db_path(valid_id), valid_id, block_id
        )
        if report is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown block: {block_id}"
            )
        return JSONResponse({"job_id": valid_id, **report})

    @api_app.post("/jobs/{job_id}/term-propagation", tags=["Jobs"])
    async def propagate_term(
        job_id: str,
        req: TermPropagationRequest,
        x_ubt_tenant: str | None = Header(default=None),
    ) -> JSONResponse:
        """Replace one offending term surface across the job (PRD §5.2.2).

        Rewrites the selected block and, per ``scope``, every matching block;
        each rewritten block is promoted to a human revision and its pair fed
        back to the shared TM, exactly like a manual edit.
        """
        valid_id = validate_job_id(job_id)
        if not await _tenant_allows_async(valid_id, scope.tenant_from_header(x_ubt_tenant)):
            raise cross_tenant_404(valid_id)
        if not _job_db_path(valid_id).exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Job not found: {valid_id}"
            )
        if await _job_is_running(valid_id):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Job is running; stop it before editing segments.",
            )

        try:
            result = await asyncio.to_thread(
                apply_term_propagation,
                _job_db_path(valid_id),
                valid_id,
                req.block_id,
                req.surface,
                req.expected,
                scope=req.scope,
                tm_path=app_config.db_dir / "tm.sqlite",
            )
        except ReviewBlockNotFound as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except ReviewEditConflict as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        except ReviewEditError as exc:
            raise HTTPException(status_code=UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

        return JSONResponse({"job_id": valid_id, **result})

    @api_app.get("/jobs/{job_id}/pages/{page}/preview", tags=["Jobs"])
    async def preview_page(
        job_id: str,
        page: int,
        dpi: int = 110,
        bilingual: bool = False,
        x_ubt_tenant: str | None = Header(default=None),
    ) -> Response:
        """Re-compose one source page with the current ledger text, as PNG.

        Reuses the delivery renderer for a single page,
        so a human edit can be previewed without re-rendering the book. 503 when
        the page has nothing to compose or the rasterizer is unavailable.
        """
        valid_id = validate_job_id(job_id)
        if not await _tenant_allows_async(valid_id, scope.tenant_from_header(x_ubt_tenant)):
            raise cross_tenant_404(valid_id)
        if page < 1:
            raise HTTPException(status_code=UNPROCESSABLE_ENTITY, detail="page must be >= 1")
        blocks = await _read_job_blocks(valid_id)
        if blocks is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Job not found: {valid_id}"
            )

        def _meta() -> tuple[str | None, str | None]:
            with SQLiteJobLedger(_job_db_path(valid_id), read_only=True) as ledger:
                return (
                    ledger.get_job_source_path(valid_id),
                    ledger.get_job_target_lang(valid_id),
                )

        source_path, target_lang = await asyncio.to_thread(_meta)
        if not source_path:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Job has no source document to preview against.",
            )
        resolved_source = resolve_secure_path(source_path, must_exist=True, config=app_config)

        with tempfile.TemporaryDirectory(prefix="ubt_page_preview_") as tmp:
            try:
                png = await asyncio.to_thread(
                    render_page_preview,
                    source_pdf=resolved_source,
                    blocks=blocks,
                    target_lang=target_lang or "zh",
                    page=page,
                    workdir=Path(tmp),
                    bilingual=bilingual,
                    dpi=min(max(dpi, 40), 200),
                )
            except PagePreviewUnavailable as exc:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
                ) from exc
        return Response(content=png, media_type="image/png")

    @api_app.get("/jobs/{job_id}/pages/{page}/source", tags=["Jobs"])
    async def source_page(
        job_id: str,
        page: int,
        dpi: int = 110,
        x_ubt_tenant: str | None = Header(default=None),
    ) -> Response:
        """Rasterize the *source* PDF's page ``page`` as PNG.

        The "before" half of the L3 pixel-witness view; pairs with
        ``/pages/{page}/preview`` (the composed "after"). 503 when the source is
        missing or the rasterizer is unavailable.
        """
        valid_id = validate_job_id(job_id)
        if not await _tenant_allows_async(valid_id, scope.tenant_from_header(x_ubt_tenant)):
            raise cross_tenant_404(valid_id)
        if page < 1:
            raise HTTPException(status_code=UNPROCESSABLE_ENTITY, detail="page must be >= 1")
        if not _job_db_path(valid_id).exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Job not found: {valid_id}"
            )

        def _meta() -> str | None:
            with SQLiteJobLedger(_job_db_path(valid_id), read_only=True) as ledger:
                return ledger.get_job_source_path(valid_id)

        source_path = await asyncio.to_thread(_meta)
        if not source_path:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Job has no source document to preview against.",
            )
        resolved_source = resolve_secure_path(source_path, must_exist=True, config=app_config)
        try:
            png = await asyncio.to_thread(
                render_source_page_png,
                resolved_source,
                page,
                min(max(dpi, 40), 200),
            )
        except PagePreviewUnavailable as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
            ) from exc
        return Response(content=png, media_type="image/png")
