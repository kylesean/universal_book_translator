"""The artifact surface: a finished job's on-disk outputs, as bytes.

Everything here resolves a stored path through the artifact sandbox and returns
the file or its metadata, so the surface's rule is about *how* a stored file may
reach a caller: attached for saving, or rendered in place when the caller asks
for an inline read, with the media type taken from the file's own suffix and an
opaque-origin sandbox because deliverable HTML is generated from untrusted book
content. Re-rendering is not here — a page preview recomputes one page against
the live ledger instead of reading a stored file, which is review-surface work.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse

from ubt.api.models import DeliverableListResponse
from ubt.api.routes.shared import (
    DELIVERABLE_LABELS,
    cross_tenant_404,
    deliverable_paths,
    public_report,
)
from ubt.api.scope import ApiScope
from ubt.api.security import resolve_secure_path, validate_job_id
from ubt.core.engine.delivery_freshness import is_delivery_stale
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.job_options import delivery_verdict_from_summary
from ubt.delivery.router import MIN_FAITHFUL_TEXT_TRUST

logger = logging.getLogger(__name__)

#: Suffix-derived media types for inline preview. ``primary``/``secondary``
#: register as ``application/octet-stream`` — a type browsers refuse to render,
#: so an iframe pointing at the download URL would fall back to downloading
#: even with ``Content-Disposition`` removed. The file suffix knows better.
_INLINE_SUFFIX_TYPES = {
    ".pdf": "application/pdf",
    ".html": "text/html; charset=utf-8",
    ".htm": "text/html; charset=utf-8",
}


def _preview_media_type(path: Path, registered: str) -> str:
    """Media type for an ``?inline=1`` render of one deliverable."""
    if registered != "application/octet-stream":
        return registered
    return _INLINE_SUFFIX_TYPES.get(path.suffix.lower(), registered)


#: Effective modes that still satisfy an explicit ``faithful`` request. The
#: planner maps ``faithful`` onto the legacy ``facsimile`` renderer, so a
#: non-degraded run reports either name depending on which layer wrote the key.
_FAITHFUL_EFFECTIVE_MODES: frozenset[str] = frozenset({"faithful", "facsimile", "page_locked"})

#: Fallback copy for a console that does not know a code. The console localizes
#: ``degrade_reason_codes`` with the measured params first; this prose is the
#: last resort, following the ``AssessWarning.detail_zh`` convention.
_DEGRADE_COPY: dict[str, str] = {
    "text_trust_low": (
        "原文档文本层可信度 {text_trust} 低于 {threshold}，通常是字体编码损坏，"
        "无法保证页级 1:1 锁定"
    ),
    "scanned_source": "源文档以扫描/图像页为主，文本层不可信，无法保证页级 1:1 锁定",
    "source_overlap": "源文档存在重叠文本层，无法保证页级 1:1 锁定",
    "page_lock_infeasible": "源文档无法解出页级 1:1 版面（page-lock 不可行）",
    "faithful_ineligible": "该文档不满足 1:1 物理复本的交付条件",
}


def _as_mode(value: object) -> str | None:
    """One delivery-mode string, normalized, or ``None`` when absent."""
    return value.strip().lower() if isinstance(value, str) and value.strip() else None


def _as_count(value: object) -> int | None:
    """A page counter from ledger metadata (bools are not counts)."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _degrade_codes(plan: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    """Structured reasons a ``faithful`` request was not honored.

    Codes (not sentences) so the bilingual console can render them from its own
    catalogue; ``params`` carries the measured values. Derived from the plan's
    signals rather than its prose ``reasons`` so a copy edit cannot silently
    change what the console reports.
    """
    signals = plan.get("signals")
    signals = signals if isinstance(signals, dict) else {}
    codes: list[str] = []
    params: dict[str, Any] = {"threshold": MIN_FAITHFUL_TEXT_TRUST}
    trust = signals.get("text_trust")
    if (
        isinstance(trust, (int, float))
        and not isinstance(trust, bool)
        and float(trust) < MIN_FAITHFUL_TEXT_TRUST
    ):
        codes.append("text_trust_low")
        params["text_trust"] = float(trust)
    page_kind = signals.get("page_kind")
    if isinstance(page_kind, str) and page_kind != "native":
        codes.append("scanned_source")
        params["page_kind"] = page_kind
    if signals.get("source_overlap") is True:
        codes.append("source_overlap")
    if signals.get("page_lock_feasible") is False:
        codes.append("page_lock_infeasible")
    if not codes:
        codes.append("faithful_ineligible")
    return codes, params


def _degrade_message(codes: list[str], params: dict[str, Any]) -> str:
    """The Chinese fallback sentence for the codes above."""
    known = [code for code in codes if code in _DEGRADE_COPY] or ["faithful_ineligible"]
    fmt = dict(params)
    for key in ("text_trust", "threshold"):
        value = fmt.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            fmt[key] = f"{float(value):.2f}"
    clauses = [_DEGRADE_COPY[code].format(**fmt) for code in known]
    return "；".join(clauses) + "。已按「1:1 不可行时自动降级」改为流式重排交付。"


def _delivery_report(meta: dict[str, Any]) -> dict[str, Any]:
    """What contract the files on disk actually satisfy.

    ``delivery_mode`` was read from a key no planned downgrade writes
    (``delivery_mode``), and ``degraded`` required the plan's contract to still
    be ``faithful`` -- which it is not once the planner has already re-routed a
    faithful request to readable. A job that asked for 1:1 and received a
    reflow copy therefore showed neither a mode badge nor a degradation notice.
    Compare the requested/effective pair the pipeline actually persists.
    """
    plan = meta.get("delivery_plan")
    plan = plan if isinstance(plan, dict) else {}
    effective = (
        _as_mode(meta.get("delivery_mode_effective"))
        or _as_mode(meta.get("delivery_mode"))
        or _as_mode(plan.get("effective_mode"))
        or _as_mode(plan.get("contract"))
    )
    requested = _as_mode(meta.get("delivery_mode_requested")) or _as_mode(
        plan.get("requested_mode")
    )
    degraded = bool(
        requested == "faithful"
        and effective is not None
        and effective not in _FAITHFUL_EFFECTIVE_MODES
    )
    codes: list[str] = []
    params: dict[str, Any] = {}
    reason: str | None = None
    if degraded:
        codes, params = _degrade_codes(plan)
        page_count = _as_count(meta.get("page_count"))
        if page_count is not None:
            params["page_count"] = page_count
        damaged = _as_count(meta.get("confirmed_font_damage_pages"))
        if damaged is None:
            route = meta.get("route_decision")
            if isinstance(route, dict):
                damaged = _as_count(route.get("confirmed_font_damage_pages"))
        if damaged is not None:
            params["damaged_pages"] = damaged
        reason = _degrade_message(codes, params)
    return {
        "delivery_mode": effective,
        "requested_delivery_mode": requested,
        "planned_delivery_mode": _as_mode(plan.get("contract")),
        "degraded": degraded,
        "degrade_reason": reason,
        "degrade_reason_codes": codes,
        "degrade_reason_params": params,
    }


#: Deliverables whose contents are *for the reader*: the translated book, its
#: companion render, and the reflowable EPUB. These are what a ``blocked``
#: verdict is about — the reader meets untranslated source in them.
#:
#: Everything else in :data:`DELIVERABLE_LABELS` is an audit or handoff
#: artifact (the quality report, the delivery contract, the XLIFF, the run
#: metrics, the visual report). Those stay downloadable from a blocked job on
#: purpose: they are the only way an operator can find out *why* it was
#: blocked, and refusing them would turn a recoverable refusal into a dead end.
_READER_FACING_DELIVERABLES = frozenset({"primary", "secondary", "epub"})


def register_artifact_routes(api_app: FastAPI, scope: ApiScope) -> None:
    """Routes for the artifact surface."""
    app_config = scope.config
    manager = scope.manager
    job_queue = scope.job_queue
    _tenant_allows = scope.tenant_allows
    _tenant_allows_async = scope.tenant_allows_async
    _artifact_path = scope.artifact_path
    _artifact_path_async = scope.artifact_path_async
    _primary_output_file = scope.primary_output_file
    _job_db_path = scope.job_db_path
    _job_is_running = scope.job_is_running
    _read_job_blocks = scope.read_job_blocks
    _managed_dir = scope.managed_dir
    _is_job_ledger = scope.is_job_ledger

    def _delivery_is_stale(job_id: str) -> bool:
        """Whether this job's delivered files predate its ledger text.

        Never raises: this runs on the download path, and a freshness probe
        that failed here would replace every download with a 500. An
        unreadable ledger reads as "not stale" — the download then behaves
        exactly as it did before this marker existed, which is the right
        degradation for a check whose only job is to catch a specific
        mistake.

        The ledger is opened **read-only**, and only for a real job ledger.
        ``validate_job_id`` constrains the charset, not the meaning: ``tm`` and
        ``job_queue`` are legal ids naming the shared translation memory and the
        live queue database. Opening one of those read-write through
        ``SQLiteJobLedger`` runs the whole ledger DDL and stamps the service
        database with UBT's tables — verified: an empty ``job_queue.sqlite``
        gained ``job_meta``, ``blocks``, ``batch_jobs`` and ``job_locks``. The
        review routes already refuse that; this probe was the remaining door.
        """
        db_path = _job_db_path(job_id)
        if not db_path.exists():
            return False
        try:
            if not _is_job_ledger_sync(job_id):
                return False
            with SQLiteJobLedger(db_path, read_only=True) as ledger:
                return is_delivery_stale(ledger, job_id)
        except Exception as exc:  # noqa: BLE001 - a probe must not break delivery
            logger.warning("delivery freshness probe failed for %s: %s", job_id, exc)
            return False

    def _is_job_ledger_sync(job_id: str) -> bool:
        """Synchronous twin of ``scope.is_job_ledger`` for the thread-run probe.

        ``summarize_job`` opens the file read-only and returns ``None`` for a
        database that is not a job ledger, so this is a pure predicate.
        """
        from ubt.api.job_catalog import summarize_job

        summary = summarize_job(_job_db_path(job_id))
        return summary is not None and str(summary.get("job_id")) == job_id

    def _delivery_verdict_sync(job_id: str) -> str | None:
        """The shared three-state verdict for this job, or ``None`` if unknown.

        ``None`` means "no quality report on disk" — the run has not reached the
        export stage, or its report was never written. Callers must treat that
        as *not blocked*: a job whose report is missing is a job with nothing to
        judge, and refusing its download would break every in-flight poll.

        Never raises, for the same reason ``_delivery_is_stale`` never does: this
        runs on the download path, and a probe that threw would replace every
        download with a 500. A corrupt report reads as ``None`` and the download
        behaves as it did before this check existed.

        Deliberately re-reads the report sidecar instead of caching: the verdict
        changes when a reviewer clears a quarantined block, and a stale cached
        "blocked" would keep refusing a delivery that is now clean.
        """
        report_file = None
        record = manager.get_job(job_id)
        if record is not None:
            report_file = record.progress.report_file
        if not report_file:
            db_path = _job_db_path(job_id)
            if not db_path.exists() or not _is_job_ledger_sync(job_id):
                return None
            try:
                with SQLiteJobLedger(db_path, read_only=True) as ledger:
                    report_file = ledger.get_job_metadata_value(job_id, "report_file")
            except Exception as exc:  # noqa: BLE001 - a probe must not break delivery
                logger.warning("delivery verdict probe failed for %s: %s", job_id, exc)
                return None
        if not report_file:
            return None
        try:
            path = Path(str(report_file))
            if not path.is_file():
                return None
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - see the docstring
            logger.warning("quality report unreadable for %s: %s", job_id, exc)
            return None
        summary = data.get("summary") if isinstance(data, dict) else None
        if not isinstance(summary, dict):
            return None
        return delivery_verdict_from_summary(summary)

    @api_app.get(
        "/jobs/{job_id}/report",
        tags=["Jobs"],
    )
    async def get_report(
        job_id: str, x_ubt_tenant: str | None = Header(default=None)
    ) -> JSONResponse:
        valid_id = validate_job_id(job_id)
        if not await _tenant_allows_async(valid_id, scope.tenant_from_header(x_ubt_tenant)):
            raise cross_tenant_404(valid_id)
        record = manager.get_job(valid_id)
        report_file = record.progress.report_file if record else None
        if not report_file:
            report_file = await _artifact_path_async(valid_id, "report_file")
        if not report_file or not Path(report_file).exists():
            in_queue = (
                job_queue is not None
                and (await asyncio.to_thread(job_queue.get, valid_id)) is not None
            )
            known = (
                record is not None
                or in_queue
                or (await _artifact_path_async(valid_id, "output_file")) is not None
                or (app_config.db_dir / f"{valid_id}.sqlite").exists()
            )
            raise HTTPException(
                status_code=404 if not known else 409,
                detail=(
                    f"Job not found: {valid_id}"
                    if not known
                    else "Quality report is not yet generated or available."
                ),
            )

        try:
            report_path = resolve_secure_path(report_file, must_exist=True, config=app_config)
            content = await asyncio.to_thread(report_path.read_text, encoding="utf-8")
            report_data: dict[str, Any] = json.loads(content)
            payload = public_report(report_data)
            # The shared three-state verdict (CLI banner reads the same rule):
            # the web gate used to derive PASS from ``failed == 0`` alone, so a
            # delivery the CLI calls BLOCKED (quarantined blocks shipping
            # source) read "PRODUCTION READY" here.
            summary = report_data.get("summary")
            if isinstance(summary, dict):
                payload["delivery_verdict"] = delivery_verdict_from_summary(summary)
            return JSONResponse(content=payload)
        except HTTPException:
            raise
        except (ValueError, FileNotFoundError) as err:
            raise HTTPException(
                status_code=400,
                detail="Quality report is not yet generated or available.",
            ) from err
        except Exception as exc:
            logger.exception("Failed to read quality report for job %s: %s", valid_id, exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=(f"Failed to read quality report; see server logs (job_id={valid_id})."),
            ) from exc

    @api_app.get(
        "/jobs/{job_id}/visual-report",
        tags=["Jobs"],
    )
    async def get_visual_report(
        job_id: str, x_ubt_tenant: str | None = Header(default=None)
    ) -> JSONResponse:
        valid_id = validate_job_id(job_id)
        if not await _tenant_allows_async(valid_id, scope.tenant_from_header(x_ubt_tenant)):
            raise cross_tenant_404(valid_id)
        record = manager.get_job(valid_id)
        vis_file = record.progress.visual_report_file if record else None
        if not vis_file:
            vis_file = await _artifact_path_async(valid_id, "visual_report_file")
        if vis_file:
            try:
                vis_path = resolve_secure_path(vis_file, must_exist=True, config=app_config)
            except HTTPException:
                # ``resolve_secure_path`` signals a rejected or missing path with
                # HTTPException, not ValueError/FileNotFoundError — catching the
                # latter let it escape, so the ledger fallback below (which holds
                # the report that is actually persisted) was unreachable.
                pass
            else:
                if vis_path.exists():
                    try:
                        content = await asyncio.to_thread(vis_path.read_text, encoding="utf-8")
                        visual_data: dict[str, Any] = json.loads(content)
                        return JSONResponse(content=visual_data)
                    except Exception as exc:
                        logger.exception(
                            "Failed to parse visual report JSON for job %s: %s", valid_id, exc
                        )
                        raise HTTPException(
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=(
                                f"Error reading visual report; see server logs (job_id={valid_id})."
                            ),
                        ) from exc

        db_path = app_config.db_dir / f"{valid_id}.sqlite"
        if db_path.exists():
            try:

                def _fetch_visual_report() -> Any:
                    with SQLiteJobLedger(db_path, read_only=True) as ledger:
                        return ledger.get_visual_report(valid_id)

                snap_report = await asyncio.to_thread(_fetch_visual_report)
                if snap_report is not None:
                    return JSONResponse(content=snap_report)
            except Exception as exc:
                logger.exception(
                    "Failed to query visual report from ledger for job %s: %s", valid_id, exc
                )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=(
                        "Database error reading visual report; see server logs "
                        f"(job_id={valid_id})."
                    ),
                ) from exc

        if not record and not db_path.exists():
            raise HTTPException(status_code=404, detail=f"Job not found: {valid_id}")
        raise HTTPException(
            status_code=400,
            detail="Visual report is not yet generated or available.",
        )

    @api_app.get(
        "/jobs/{job_id}/delivery-contract",
        tags=["Jobs"],
        summary="Fetch the artifact delivery contract (11-point QA, CELS distortion, and layout invariants)",
    )
    async def get_delivery_contract(
        job_id: str, x_ubt_tenant: str | None = Header(default=None)
    ) -> JSONResponse:
        valid_id = validate_job_id(job_id)
        if not await _tenant_allows_async(valid_id, scope.tenant_from_header(x_ubt_tenant)):
            raise cross_tenant_404(valid_id)
        output_file = await _primary_output_file(valid_id)
        if output_file:
            candidate = deliverable_paths(output_file).get("delivery_contract")
            if candidate is not None:
                try:
                    resolved = resolve_secure_path(candidate, must_exist=True, config=app_config)
                    if resolved.is_file():
                        content = await asyncio.to_thread(resolved.read_text, encoding="utf-8")
                        return JSONResponse(content=json.loads(content))
                except Exception as exc:
                    logger.warning("Could not read delivery contract for %s: %s", valid_id, exc)

        record = manager.get_job(valid_id)
        db_path = app_config.db_dir / f"{valid_id}.sqlite"
        if not record and not db_path.exists():
            raise HTTPException(status_code=404, detail=f"Job not found: {valid_id}")
        raise HTTPException(
            status_code=400,
            detail="Delivery contract is not yet generated or available.",
        )

    @api_app.get(
        "/jobs/{job_id}/download",
        tags=["Jobs"],
    )
    async def download_file(
        job_id: str, x_ubt_tenant: str | None = Header(default=None)
    ) -> FileResponse:
        valid_id = validate_job_id(job_id)
        if not await _tenant_allows_async(valid_id, scope.tenant_from_header(x_ubt_tenant)):
            raise cross_tenant_404(valid_id)
        record = manager.get_job(valid_id)
        output_file = record.progress.output_file if record else None
        if not output_file:
            output_file = await _artifact_path_async(valid_id, "output_file")
        if not output_file:
            db_path = app_config.db_dir / f"{valid_id}.sqlite"
            in_queue = (
                job_queue is not None
                and (await asyncio.to_thread(job_queue.get, valid_id)) is not None
            )
            job_exists = (record is not None) or in_queue or db_path.exists()
            raise HTTPException(
                status_code=409 if job_exists else 404,
                detail=(
                    "Translated file is not ready for download."
                    if job_exists
                    else f"Job not found: {valid_id}"
                ),
            )

        output_path = resolve_secure_path(output_file, must_exist=False, config=app_config)
        if not output_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Translated file not found for job: {valid_id}",
            )
        return FileResponse(
            path=output_path,
            filename=output_path.name,
            media_type="application/octet-stream",
        )

    @api_app.get(
        "/jobs/{job_id}/deliverables",
        response_model=DeliverableListResponse,
        tags=["Jobs"],
    )
    async def list_deliverables(
        job_id: str, x_ubt_tenant: str | None = Header(default=None)
    ) -> JSONResponse:
        """List the deliverables a finished job actually left on disk.

        A run may emit a complementary dual/mono render, an EPUB and the JSON
        sidecars depending on its flags; the UI must render only what exists
        rather than offer four fixed buttons. Keys match
        :data:`DELIVERABLE_LABELS`.
        """
        valid_id = validate_job_id(job_id)
        if not await _tenant_allows_async(valid_id, scope.tenant_from_header(x_ubt_tenant)):
            raise cross_tenant_404(valid_id)
        output_file = await _primary_output_file(valid_id)
        if not output_file:
            record = manager.get_job(valid_id)
            in_queue = (
                job_queue is not None
                and (await asyncio.to_thread(job_queue.get, valid_id)) is not None
            )
            known = (
                record is not None
                or in_queue
                or (app_config.db_dir / f"{valid_id}.sqlite").exists()
            )
            raise HTTPException(
                status_code=409 if known else 404,
                detail=(
                    "Deliverables are not ready yet." if known else f"Job not found: {valid_id}"
                ),
            )

        items: list[dict[str, Any]] = []
        for key, candidate in deliverable_paths(output_file).items():
            try:
                resolved = resolve_secure_path(candidate, must_exist=False, config=app_config)
            except HTTPException:
                continue
            if resolved.is_file():
                label, media_type = DELIVERABLE_LABELS[key]
                items.append(
                    {
                        "key": key,
                        "label": label,
                        "filename": resolved.name,
                        "size_bytes": resolved.stat().st_size,
                        "media_type": media_type,
                    }
                )
        companion_artifacts: list[dict[str, Any]] | None = None
        delivery = _delivery_report({})
        db_path = _job_db_path(valid_id)
        if db_path.exists() and _is_job_ledger_sync(valid_id):
            try:
                with SQLiteJobLedger(db_path, read_only=True) as ledger:
                    meta = ledger.get_job_metadata(valid_id)
                    companion = meta.get("companion_artifacts")
                    if (
                        isinstance(companion, dict)
                        and "results" in companion
                        and isinstance(companion["results"], list)
                    ):
                        companion_artifacts = companion["results"]
                    delivery = _delivery_report(meta)
            except Exception as exc:  # metadata is optional: the artifact list still answers
                logger.debug("Could not read artifact metadata for %s: %s", valid_id, exc)

        return JSONResponse(
            {
                "job_id": valid_id,
                "deliverables": items,
                "companion_artifacts": companion_artifacts,
                # Lets the console offer "re-render first" instead of opening a
                # download that the guard below will refuse.
                "stale": await asyncio.to_thread(_delivery_is_stale, valid_id),
                # The same three-state verdict the report carries, surfaced here
                # so the download button can warn *before* the click rather than
                # 409-ing after it. ``None`` = no report yet, which is not a
                # refusal.
                "delivery_verdict": await asyncio.to_thread(_delivery_verdict_sync, valid_id),
                **delivery,
            }
        )

    @api_app.get(
        "/jobs/{job_id}/download/{key}",
        tags=["Jobs"],
    )
    async def download_deliverable(
        job_id: str,
        key: str,
        inline: bool = False,
        force: bool = False,
        x_ubt_tenant: str | None = Header(default=None),
    ) -> FileResponse:
        """Serve one deliverable by key (see :data:`DELIVERABLE_LABELS`).

        ``?inline=1`` renders the file in place (the console's preview pane)
        instead of attaching it: no ``Content-Disposition: attachment``, and a
        media type the browser can actually display. The response is served
        under a CSP ``sandbox`` directive either way — the HTML deliverable is
        generated from *book* content, which is untrusted input, so even a
        fetched-directly preview runs scriptless in an opaque origin rather
        than in the console's origin.

        ``?force=1`` takes a delivery the quality gate blocked. It exists for
        the operator who has read the report and decided the partial book is
        still worth shipping; without it a ``blocked`` verdict refuses the
        reader-facing deliverables (see the check below).
        """
        valid_id = validate_job_id(job_id)
        if not await _tenant_allows_async(valid_id, scope.tenant_from_header(x_ubt_tenant)):
            raise cross_tenant_404(valid_id)
        if key not in DELIVERABLE_LABELS:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Unknown deliverable: {key}",
            )
        output_file = await _primary_output_file(valid_id)
        if not output_file:
            record = manager.get_job(valid_id)
            in_queue = (
                job_queue is not None
                and (await asyncio.to_thread(job_queue.get, valid_id)) is not None
            )
            known = (
                record is not None
                or in_queue
                or (app_config.db_dir / f"{valid_id}.sqlite").exists()
            )
            raise HTTPException(
                status_code=409 if known else 404,
                detail=(
                    "Deliverables are not ready yet." if known else f"Job not found: {valid_id}"
                ),
            )

        candidate = deliverable_paths(output_file).get(key)
        if candidate is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Deliverable not available: {key}",
            )
        resolved = resolve_secure_path(candidate, must_exist=True, config=app_config)
        if not resolved.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Deliverable not available: {key}",
            )
        # Refuse a delivery the ledger has moved past. A human edit lands in the
        # ledger; the file on disk was written by an earlier export. Serving it
        # anyway is the worst failure mode this endpoint has: the customer
        # receives a pre-edit draft, the reviewer believes their fix shipped,
        # and nothing anywhere reports a problem. Failing closed turns that into
        # a retryable error, and the console turns the error into a re-render.
        #
        # A job whose ledger predates the marker is not stale (see
        # ``is_delivery_stale``); only a *recorded* mismatch blocks.
        stale = await asyncio.to_thread(_delivery_is_stale, valid_id)
        if stale:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "The delivered files predate the current text: this job has "
                    "edits that were not re-rendered. Re-render the deliverable "
                    "before downloading it."
                ),
            )

        # Refuse a reader-facing deliverable whose quality gate said BLOCKED.
        #
        # ``blocked`` means blocks were quarantined (MQM Critical unresolved) or
        # failed: they ship as a source-only placeholder, not a translation, so
        # the reader meets English in a Chinese book. Every other surface
        # already says so — the CLI banner, the console's gate screen, Mission
        # Control's status line — and this endpoint served the file anyway,
        # which made the whole verdict advisory at the one moment it matters.
        #
        # Three deliberate narrowings:
        #   * only reader-facing keys (``_READER_FACING_DELIVERABLES``); the
        #     audit sidecars stay open, or an operator could not read the report
        #     explaining the refusal;
        #   * ``?inline=1`` is exempt — that is the console's own preview pane,
        #     not a delivery, and the operator needs to look at the artifact to
        #     decide what to do about it;
        #   * ``?force=1`` overrides, mirroring ``strict_contract`` and
        #     ``allow_unpriced_budget``: the safe behaviour is the default and
        #     loosening it must be asked for explicitly.
        if key in _READER_FACING_DELIVERABLES and not inline and not force:
            verdict = await asyncio.to_thread(_delivery_verdict_sync, valid_id)
            if verdict == "blocked":
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=(
                        "The quality gate blocked this delivery: some blocks were "
                        "quarantined or failed and ship the source text instead of a "
                        "translation. Inspect the quality report, or re-download with "
                        "?force=1 to take it knowingly."
                    ),
                )

        _, media_type = DELIVERABLE_LABELS[key]
        if inline:
            return FileResponse(
                path=resolved,
                media_type=_preview_media_type(resolved, media_type),
                headers={
                    # Opaque origin, no script execution — for engine-generated
                    # HTML built out of untrusted book text.
                    "Content-Security-Policy": "sandbox",
                    "X-Content-Type-Options": "nosniff",
                },
            )
        return FileResponse(
            path=resolved,
            filename=resolved.name,
            media_type=media_type,
        )
