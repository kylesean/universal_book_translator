"""The asset surface: glossary and translation-memory CRUD, model profiles."""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, status
from fastapi.responses import JSONResponse

from ubt.api.assets import (
    GlossaryFormatError,
    TMImportError,
    add_glossary_term,
    detect_glossary_conflicts,
    evict_tm_entries,
    import_tm_entries,
    list_tm_entries,
    parse_tm_payload,
    read_glossary_terms,
    remove_glossary_term,
)
from ubt.api.models import (
    GlossaryTermRequest,
    TMevictRequest,
    TMImportRequest,
)
from ubt.api.routes.shared import UNPROCESSABLE_ENTITY
from ubt.api.scope import ApiScope
from ubt.api.security import resolve_secure_path, verify_api_key
from ubt.core.router import ModelProfile, get_default_registry


def register_asset_routes(api_app: FastAPI, scope: ApiScope) -> None:
    """Routes for the asset surface."""
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

    def _resolved_glossary_path() -> Path:
        configured = app_config.glossary_path
        if configured is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "No glossary file is configured. Set glossary_path in ubt.toml "
                    "or pass --glossary."
                ),
            )
        return resolve_secure_path(configured, must_exist=True, config=app_config)

    def _tm_path() -> Path:
        return app_config.db_dir / "tm.sqlite"

    @api_app.get(
        "/api/v1/model-profiles",
        response_model=list[ModelProfile],
        tags=["Model Profiles"],
    )
    @api_app.get(
        "/model-profiles",
        response_model=list[ModelProfile],
        tags=["Model Profiles"],
        include_in_schema=False,
    )
    async def list_model_profiles() -> list[ModelProfile]:
        """List all currently registered model capability profiles."""
        return get_default_registry().list_profiles()

    @api_app.post(
        "/api/v1/model-profiles",
        response_model=ModelProfile,
        tags=["Model Profiles"],
    )
    @api_app.post(
        "/model-profiles",
        response_model=ModelProfile,
        tags=["Model Profiles"],
        include_in_schema=False,
    )
    async def register_model_profile(
        profile: ModelProfile,
        x_api_key: str | None = Header(default=None),
    ) -> ModelProfile:
        """Register a custom model capability profile (built-in profiles cannot be overridden)."""
        verify_api_key(x_api_key, _config=app_config)
        try:
            get_default_registry().register(profile, override=False)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc
        return profile

    @api_app.get("/assets/glossary", tags=["Assets"])
    async def get_glossary() -> JSONResponse:
        path = _resolved_glossary_path()
        try:
            terms = await asyncio.to_thread(read_glossary_terms, path)
        except GlossaryFormatError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        return JSONResponse({"path": path.name, "terms": terms})

    @api_app.post("/assets/glossary", tags=["Assets"])
    async def upsert_glossary_term(req: GlossaryTermRequest) -> JSONResponse:
        if not req.target.strip():
            raise HTTPException(
                status_code=UNPROCESSABLE_ENTITY,
                detail="target is required when adding a term",
            )
        path = _resolved_glossary_path()
        try:
            await asyncio.to_thread(add_glossary_term, path, req.source, req.target)
            terms = await asyncio.to_thread(read_glossary_terms, path)
        except GlossaryFormatError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        return JSONResponse({"path": path.name, "terms": terms})

    @api_app.delete("/assets/glossary", tags=["Assets"])
    async def delete_glossary_term(source: str) -> JSONResponse:
        path = _resolved_glossary_path()
        try:
            removed = await asyncio.to_thread(remove_glossary_term, path, source)
            terms = await asyncio.to_thread(read_glossary_terms, path)
        except GlossaryFormatError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        return JSONResponse({"path": path.name, "removed": removed, "terms": terms})

    @api_app.get("/assets/glossary/conflicts", tags=["Assets"])
    async def get_glossary_conflicts() -> JSONResponse:
        """Sources configured with more than one target rendering (PRD §4.4.2)."""
        path = _resolved_glossary_path()
        try:
            terms = await asyncio.to_thread(read_glossary_terms, path)
        except GlossaryFormatError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        return JSONResponse({"conflicts": detect_glossary_conflicts(terms)})

    @api_app.get("/assets/tm", tags=["Assets"])
    async def get_translation_memory(
        limit: int = 200,
        offset: int = 0,
        src_lang: str | None = None,
        tgt_lang: str | None = None,
    ) -> JSONResponse:
        data = await asyncio.to_thread(
            list_tm_entries,
            _tm_path(),
            limit=min(max(limit, 1), 1000),
            offset=max(offset, 0),
            src_lang=src_lang,
            tgt_lang=tgt_lang,
        )
        return JSONResponse(data)

    @api_app.post("/assets/tm/evict", tags=["Assets"])
    async def evict_translation_memory(req: TMevictRequest) -> JSONResponse:
        removed = await asyncio.to_thread(evict_tm_entries, _tm_path(), req.ids)
        return JSONResponse({"removed": removed})

    @api_app.post("/assets/tm/import", tags=["Assets"])
    async def import_translation_memory(req: TMImportRequest) -> JSONResponse:
        """Import TMX/JSON pairs into the shared TM (machine provenance by default).

        An import is unverified, so the default ``machine`` provenance cannot
        downgrade an existing ``human_pe`` row (the store's writeback enforces
        that). Parsing is strict: a malformed payload is a 422, not a silent
        empty import.
        """
        try:
            rows = await asyncio.to_thread(parse_tm_payload, req.content, req.format)
            imported = await asyncio.to_thread(
                import_tm_entries,
                _tm_path(),
                rows,
                default_src_lang=req.src_lang,
                default_tgt_lang=req.tgt_lang,
                provenance=req.provenance,
            )
        except TMImportError as exc:
            raise HTTPException(status_code=UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
        return JSONResponse({"parsed": len(rows), "imported": imported})
