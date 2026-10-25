"""Agent-driven translation: prepare, claim, submit and finish.

This is the zero-model-call counterpart to :class:`PipelineOrchestrator`. The
calling Agent (Codex, Claude Code, or any MCP host) supplies the translation
text; UBT owns everything deterministic around it:

* document parsing into the shared IR and ledger;
* terminology, section context, neighbour excerpts and masked placeholders;
* atomic work-package leases, expiry and idempotent submission;
* FastPass / glossary validation and the repair queue;
* rendering, reports and the delivered artifact.

No provider or network model call is made on this path. A ``NoopProvider`` is
installed in the router so an accidental future model call fails loudly instead
of silently spending the operator's API quota.

The four public operations are deliberately small and MCP-friendly. The service
is synchronous/blocking (like the ledger) and its async wrappers in the MCP
layer run it off the event loop; long prepare/finish operations hold the
existing writer lease.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import secrets
import time
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ubt.core.config import UBTConfig
from ubt.core.engine.events import EventType, TranslationProgressEvent
from ubt.core.engine.facts import Terminology
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.ledger_mixins import (
    AGENT_REQUEST_FINGERPRINT_KEY,
    AGENT_SESSION_KEY,
)
from ubt.core.engine.repair_loop import RepairLoop
from ubt.core.engine.services import RunServices
from ubt.core.engine.stage_context import StageContext
from ubt.core.engine.stages.export import run_export_stage
from ubt.core.engine.stages.ingest import run_ingest_stage
from ubt.core.engine.stages.quality_gate import screen_drafted_blocks
from ubt.core.engine.writer_lock import LedgerWriterLock, keep_lease_fresh
from ubt.core.exceptions import (
    LedgerError,
    UBTError,
    UnsupportedDocumentFormatError,
)
from ubt.core.ir.document_tree import section_context_for
from ubt.core.ir.emphasis import parse_emphasis
from ubt.core.ir.models import (
    TERMINAL_STATUSES,
    BlockStatus,
    BookManifest,
    IRBlock,
    StyleMeta,
)
from ubt.core.ir.render_plan import RenderPlan
from ubt.core.memory.glossary_table import (
    build_chunk_glossary_table,
    build_global_glossary_table,
)
from ubt.core.memory.neighbor_window import NeighborContextBuilder
from ubt.core.policy.adaptive_policy import AdaptivePolicy
from ubt.core.ports import (
    AdapterRuntimeConfig,
    DocumentAdapter,
    apply_runtime_config,
    resolve_adapter,
)
from ubt.core.qe.fast_pass import FastPassFilter
from ubt.core.qe.heuristic import HeuristicQERunner
from ubt.core.router.prompts import build_rich_draft_prompt, build_rich_repair_prompt
from ubt.core.router.provider import BaseModelProvider
from ubt.core.router.rate_limiter import NullRateLimiter
from ubt.core.router.router import ModelRouter
from ubt.core.validators.consistency import GlossaryConsistencyValidator
from ubt.segment.placeholders import PlaceholderEngine, default_placeholder_engine
from ubt.translate.engine import TranslationEngine

logger = logging.getLogger(__name__)

AGENT_MODE = "agent"
_AGENT_ENGINE_SIGNATURE = "agent"
_DEFAULT_PACKAGE_SIZE = 8
_MAX_PACKAGE_SIZE = 64
_DEFAULT_LEASE_SECONDS = 900.0
_MIN_LEASE_SECONDS = 30.0
_MAX_LEASE_SECONDS = 86_400.0


class _NoopProvider(BaseModelProvider):
    """A provider that exists only to make accidental model calls impossible."""

    @property
    def provider_name(self) -> str:
        return "agent-noop"

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = 0.3,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        raise UBTError(
            "Agent translation mode made an unexpected model call; "
            "the host Agent must supply translations via ubt_agent_submit."
        )


def _agent_router() -> ModelRouter:
    provider = _NoopProvider()
    return ModelRouter(
        provider=provider,
        rate_limiter=NullRateLimiter(),
        draft_model="agent-host",
        repair_model="agent-host",
    )


def _now() -> float:
    return time.time()


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _request_fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _submission_hash(blocks: Sequence[AgentBlockSubmission]) -> str:
    payload = sorted(
        ({"block_id": item.block_id, "translation": item.translation} for item in blocks),
        key=lambda item: item["block_id"],
    )
    return _request_fingerprint({"blocks": payload})


def _source_hash(block: IRBlock) -> str:
    return hashlib.sha256(block.source_text.encode("utf-8")).hexdigest()


def _as_json_metadata(value: Any) -> Any:
    """Best-effort JSON-safe projection for a ledger metadata value."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): _as_json_metadata(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_json_metadata(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class AgentPrepareRequest(BaseModel):
    """Inputs for creating an Agent translation job."""

    model_config = ConfigDict(extra="forbid")

    input_path: str = Field(
        min_length=1,
        max_length=4096,
        description="Source document path inside the MCP filesystem allowlist.",
    )
    output_path: str | None = Field(
        default=None,
        max_length=4096,
        description="Optional deliverable path; defaults beside the input or in the allowed base.",
    )
    target_lang: str = Field(
        default="zh",
        min_length=2,
        max_length=16,
        description="Target language code (for example zh, en, ja).",
    )
    source_lang: str = Field(
        default="en",
        min_length=2,
        max_length=16,
        description="Source language code (for example en, zh, ja).",
    )
    profile: str = Field(
        default="auto",
        min_length=1,
        max_length=64,
        description="Domain profile such as auto, general, textbook or paper.",
    )
    job_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=80,
        description="Optional stable idempotency key for the job.",
    )
    glossary: str | None = Field(
        default=None,
        max_length=4096,
        description="Optional external glossary path inside the allowlist.",
    )
    pages: str | None = Field(
        default=None,
        max_length=256,
        description="Optional 1-based page range, for example 1-10,15.",
    )
    domain: str | None = Field(
        default=None,
        max_length=200,
        description="Optional subject descriptor used in prompts and terminology.",
    )
    dual_mode: Literal["auto", "dual", "mono"] = Field(
        default="auto",
        description="Delivered layout: aligned to the source, bilingual, or target-only.",
    )
    layout_columns: Literal[1, 2] | None = Field(
        default=None,
        description="Optional delivered column count; None aligns to the source.",
    )
    package_size: int = Field(
        default=_DEFAULT_PACKAGE_SIZE,
        ge=1,
        le=_MAX_PACKAGE_SIZE,
        description="Maximum blocks in one claim package.",
    )
    lease_seconds: float = Field(
        default=_DEFAULT_LEASE_SECONDS,
        ge=_MIN_LEASE_SECONDS,
        le=_MAX_LEASE_SECONDS,
        description="Lease duration for a claimed package before it can be reclaimed.",
    )
    fresh: bool = Field(
        default=False,
        description="Discard this job's existing blocks and parse the source again.",
    )


class AgentPrepareResult(BaseModel):
    """Result of preparing an Agent job."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    status: Literal["awaiting_agent"] = "awaiting_agent"
    total_blocks: int = Field(ge=0)
    pending_blocks: int = Field(ge=0)
    skipped_blocks: int = Field(ge=0)
    output_path: str
    source_lang: str
    target_lang: str
    profile: str
    next_action: Literal["ubt_agent_claim"] = "ubt_agent_claim"
    model_calls: Literal[0] = 0


class AgentBlockWork(BaseModel):
    """One block the host Agent must translate or repair."""

    model_config = ConfigDict(extra="forbid")

    block_id: str
    spine_index: int = Field(ge=0)
    block_type: str
    masked_source: str
    source_hash: str
    user_prompt: str
    draft_text: str | None = None
    error_flags: list[str] = Field(default_factory=list)
    neighbor_context: str = ""
    section_context: str = ""
    glossary_table: str = ""


class AgentWorkPackage(BaseModel):
    """A leased, self-contained unit of Agent work."""

    model_config = ConfigDict(extra="forbid")

    package_id: str
    lease_token: str
    job_id: str
    kind: Literal["translate", "repair"]
    expires_at: float
    target_lang: str
    source_lang: str
    profile: str
    system_prompt: str
    global_glossary: str = ""
    blocks: list[AgentBlockWork] = Field(min_length=1)
    instructions: str
    model_calls: Literal[0] = 0


class AgentClaimRequest(BaseModel):
    """Request one work package from an Agent job."""

    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(min_length=1, max_length=80)
    max_blocks: int | None = Field(
        default=None,
        ge=1,
        le=_MAX_PACKAGE_SIZE,
        description="Optional per-claim cap; defaults to the prepare-time package size.",
    )


class AgentClaimResult(BaseModel):
    """Result of a claim; ``package`` is absent when no work remains."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    package: AgentWorkPackage | None = None
    status: Literal["awaiting_agent", "completed", "needs_attention"]
    pending_blocks: int = Field(ge=0)
    next_action: Literal["ubt_agent_submit", "ubt_agent_finish", "none"]
    message: str
    active_packages: int = Field(default=0, ge=0)


class AgentBlockSubmission(BaseModel):
    """The host Agent's result for one block."""

    model_config = ConfigDict(extra="forbid")

    block_id: str = Field(min_length=1, max_length=256)
    translation: str = Field(min_length=1, max_length=500_000)


class AgentSubmitRequest(BaseModel):
    """Submit all block results for one leased package."""

    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(min_length=1, max_length=80)
    package_id: str = Field(min_length=1, max_length=128)
    lease_token: str = Field(min_length=16, max_length=512)
    blocks: list[AgentBlockSubmission] = Field(min_length=1, max_length=_MAX_PACKAGE_SIZE)


class AgentSubmitResult(BaseModel):
    """Result of accepting a package submission."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    package_id: str
    accepted_blocks: int = Field(ge=0)
    rejected_blocks: int = Field(ge=0)
    pending_blocks: int = Field(ge=0)
    status: Literal["awaiting_agent", "needs_attention"]
    next_action: Literal["ubt_agent_claim", "ubt_agent_finish"]
    idempotent_replay: bool = False
    model_calls: Literal[0] = 0
    message: str = ""


class AgentFinishRequest(BaseModel):
    """Request quality gating and rendering of a prepared Agent job."""

    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(min_length=1, max_length=80)


class AgentIssue(BaseModel):
    """One unresolved issue reported by finish."""

    model_config = ConfigDict(extra="forbid")

    block_id: str
    status: str
    error_flags: list[str] = Field(default_factory=list)


class AgentFinishResult(BaseModel):
    """Terminal or actionable result of finishing an Agent job."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    status: Literal[
        "completed",
        "repair_required",
        "needs_attention",
        "failed",
        "awaiting_agent",
    ]
    output_path: str | None = None
    report_path: str | None = None
    pending_blocks: int = Field(default=0, ge=0)
    repair_blocks: int = Field(default=0, ge=0)
    issues: list[AgentIssue] = Field(default_factory=list)
    next_action: Literal["ubt_agent_claim", "ubt_agent_finish", "none"]
    message: str
    model_calls: Literal[0] = 0


class AgentTranslationService:
    """Core Agent translation workflow with no model/provider dependency."""

    def __init__(
        self,
        *,
        config: UBTConfig,
        placeholder_engine: PlaceholderEngine | None = None,
    ) -> None:
        self.config = config
        self.placeholders = placeholder_engine or default_placeholder_engine()
        self.engine = TranslationEngine(
            placeholders=self.placeholders,
            model="agent-host",
            prompt_version="agent-v1",
        )
        self._lease_lost: asyncio.Event | None = None

    # ------------------------------------------------------------------
    # prepare
    # ------------------------------------------------------------------
    async def prepare(self, request: AgentPrepareRequest) -> AgentPrepareResult:
        """Parse the source and create a resumable, zero-model Agent job."""
        input_path = Path(request.input_path).expanduser().resolve()
        if not input_path.exists():
            raise UBTError(f"input document does not exist: {input_path}")
        config = await self._config_for_prepare(request)
        adapter = resolve_adapter(input_path, pdf_engine=config.pdf_engine)
        try:
            apply_runtime_config(
                adapter,
                AdapterRuntimeConfig(
                    ocr_mode=config.ocr_mode,
                    ocr_endpoint=config.ocr_endpoint,
                    ocr_api_key=config.ocr_api_key.get_secret_value(),
                    ocr_model=config.ocr_model,
                    formula_enrichment=config.formula_enrichment,
                    docling_device=config.docling_device,
                    formula_crop_margin=config.formula_crop_margin,
                    toc_mode=config.toc_mode,
                    font_family=config.font_family,
                    allow_page_upload=False,
                    cache_dir=str(config.cache_dir) if config.cache_enabled else "",
                ),
            )
            manifest = await adapter.extract_manifest(input_path)
            manifest.source_lang = request.source_lang
            manifest.target_lang = request.target_lang
            manifest.run.layout_columns = request.layout_columns
            output_path = self._resolve_output(request.output_path, input_path, manifest)
            job_id = request.job_id or self._derive_job_id(
                manifest=manifest,
                target_lang=request.target_lang,
                source_lang=request.source_lang,
                profile=request.profile,
                pages=request.pages,
            )
            db_path = config.db_dir / f"{job_id}.sqlite"
            fingerprint = _request_fingerprint(
                {
                    "mode": AGENT_MODE,
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "source_lang": request.source_lang,
                    "target_lang": request.target_lang,
                    "profile": request.profile,
                    "glossary": str(Path(request.glossary).expanduser().resolve())
                    if request.glossary
                    else None,
                    "pages": request.pages,
                    "domain": request.domain,
                    "dual_mode": request.dual_mode,
                    "layout_columns": request.layout_columns,
                }
            )
            lock = LedgerWriterLock(db_path, job_id)
            await asyncio.to_thread(lock.acquire)
            lease_lost = asyncio.Event()
            renew_task = asyncio.create_task(
                keep_lease_fresh(lock, on_lost=lambda _reason: lease_lost.set()),
                name=f"renew-agent-prepare:{job_id}",
            )
            try:
                ledger = await asyncio.to_thread(SQLiteJobLedger, db_path, fresh=request.fresh)
                try:
                    await self._prepare_in_ledger(
                        ledger=ledger,
                        job_id=job_id,
                        manifest=manifest,
                        input_path=input_path,
                        output_path=output_path,
                        config=config,
                        adapter=adapter,
                        request=request,
                        fingerprint=fingerprint,
                        lease_lost=lease_lost,
                    )
                    stats = await asyncio.to_thread(ledger.get_job_stats, job_id)
                    pending = (
                        int(stats.get("total", 0) or 0)
                        - int(stats.get("completed", 0) or 0)
                        - int(stats.get("drafted", 0) or 0)
                        - int(stats.get("failed", 0) or 0)
                        - int(stats.get("needs_human", 0) or 0)
                        - int(stats.get("blocked_human", 0) or 0)
                    )
                    return AgentPrepareResult(
                        job_id=job_id,
                        total_blocks=int(stats.get("total_all", 0) or 0),
                        pending_blocks=max(0, pending),
                        skipped_blocks=int(stats.get("skipped", 0) or 0),
                        output_path=str(output_path),
                        source_lang=request.source_lang,
                        target_lang=request.target_lang,
                        profile=request.profile,
                    )
                finally:
                    await asyncio.to_thread(ledger.close)
            finally:
                renew_task.cancel()
                with suppress(asyncio.CancelledError):
                    await renew_task
                await asyncio.to_thread(lock.release)
        finally:
            close = getattr(adapter, "close", None)
            if callable(close):
                with suppress(Exception):
                    await asyncio.to_thread(close)

    async def _config_for_prepare(self, request: AgentPrepareRequest) -> UBTConfig:
        from ubt.core.job_options import apply_config_overrides, resolve_profile_name

        profile = resolve_profile_name(request.profile, request.input_path)
        request.profile = profile
        overrides: dict[str, Any] = {
            "pages": request.pages,
            "domain": request.domain,
            "visual_judge_enabled": False,
            "ocr_mode": "off",
            "allow_page_upload": False,
            "fresh": request.fresh,
            "dual_mode": request.dual_mode,
            "layout_columns": request.layout_columns,
        }
        if request.glossary:
            glossary = Path(request.glossary).expanduser().resolve()
            if not glossary.exists():
                raise UBTError(f"glossary does not exist: {glossary}")
            overrides["glossary_path"] = glossary
        config = apply_config_overrides(self.config, overrides)
        config.db_dir.mkdir(parents=True, exist_ok=True)
        return config

    def _resolve_output(self, output: str | None, input_path: Path, manifest: BookManifest) -> Path:
        from ubt.core.job_options import resolve_target_output

        raw = Path(output).expanduser() if output else None
        resolved = resolve_target_output(
            raw,
            input_path,
            monolingual=self.config.dual_mode == "mono",
            pages=self.config.get_selected_pages(),
        )
        if resolved.resolve() == input_path.resolve():
            raise UBTError("output_path equals the input document; refusing to overwrite it")
        return resolved

    def _derive_job_id(
        self,
        *,
        manifest: BookManifest,
        target_lang: str,
        source_lang: str,
        profile: str,
        pages: str | None,
    ) -> str:
        from ubt.core.engine.pipeline import derive_job_id

        return derive_job_id(
            doc_id=manifest.doc_id,
            target_lang=target_lang,
            pages=pages,
            start_chapter=1,
            max_chapters=None,
            mock_run=False,
            profile_name=profile,
            engine_signature=_AGENT_ENGINE_SIGNATURE,
            source_lang=source_lang,
        )

    async def _prepare_in_ledger(
        self,
        *,
        ledger: SQLiteJobLedger,
        job_id: str,
        manifest: BookManifest,
        input_path: Path,
        output_path: Path,
        config: UBTConfig,
        adapter: DocumentAdapter,
        request: AgentPrepareRequest,
        fingerprint: str,
        lease_lost: asyncio.Event,
    ) -> None:
        router = _agent_router()
        ctx = StageContext(
            config=config,
            router=router,
            ledger=ledger,
            manifest=manifest,
            adapter=adapter,
            job_id=job_id,
            input_path=input_path,
            output_path=output_path,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            profile_name=request.profile,
            create_event=self._event_factory(ledger, job_id),
            fast_lane=False,
            lease_lost=lease_lost,
        )
        stored_fingerprint = await asyncio.to_thread(
            ledger.get_job_metadata_value, job_id, AGENT_REQUEST_FINGERPRINT_KEY
        )
        persisted = await asyncio.to_thread(ledger.get_job_status, job_id)
        metadata = await asyncio.to_thread(ledger.get_job_metadata, job_id)
        stored_mode = metadata.get("translation_mode")
        if stored_mode is not None and stored_mode != AGENT_MODE and not request.fresh:
            raise UBTError(
                f"job {job_id} is a {stored_mode} translation job; use a different job_id"
            )
        if persisted in ("completed", "failed", "cancelled") and not request.fresh:
            raise UBTError(
                f"job {job_id} is already {persisted}; use a different job_id or "
                "fresh=true to replace its ledger"
            )
        if request.fresh and metadata.get("agent_session"):
            session = metadata.get("agent_session")
            if isinstance(session, dict):
                packages = session.get("packages")
                if isinstance(packages, dict) and any(
                    isinstance(package, dict) and package.get("state") == "claimed"
                    for package in packages.values()
                ):
                    raise UBTError(
                        f"job {job_id} has an active Agent lease; finish or let it expire "
                        "before using fresh=true"
                    )
        if (
            stored_fingerprint is not None
            and stored_fingerprint != fingerprint
            and not request.fresh
        ):
            raise UBTError(
                f"job {job_id} already exists with different Agent prepare inputs; "
                "use another job_id or fresh=true to discard the prior ledger"
            )
        # Overwrite authorization is per deliverable, not per job id: an existing
        # file may be replaced only when it is the exact deliverable this job's
        # own ledger recorded (a completed run's artifact, or a partial file a
        # crashed render left behind). Anything else at the resolved path — a
        # first run, another job's or a human's file — is refused here, after the
        # terminal/fingerprint guards so a completed job's replay still answers
        # with the terminal reason, and before any new parse work begins.
        if output_path.exists():
            stored_output = await asyncio.to_thread(
                ledger.get_job_metadata_value, job_id, "output_file"
            )
            owns_output = stored_output is not None and Path(str(stored_output)).resolve() == (
                output_path.resolve()
            )
            if not owns_output:
                raise UBTError(
                    f"output_path already exists; refusing to overwrite it: {output_path}"
                )
        existing = await asyncio.to_thread(ledger.get_job_stats, job_id)
        if int(existing.get("total_all", 0) or 0) == 0 or request.fresh:
            async for _event in run_ingest_stage(ctx):
                pass
        else:
            await asyncio.to_thread(ledger.init_job_from_manifest, job_id, manifest)
        await asyncio.to_thread(
            ledger.set_job_metadata_value, job_id, "translation_mode", AGENT_MODE
        )
        await asyncio.to_thread(
            ledger.set_job_metadata_value, job_id, AGENT_REQUEST_FINGERPRINT_KEY, fingerprint
        )
        await asyncio.to_thread(
            ledger.set_job_metadata_value, job_id, "output_file", str(output_path)
        )
        await asyncio.to_thread(
            ledger.set_job_metadata_value, job_id, "source_lang", request.source_lang
        )
        await asyncio.to_thread(
            ledger.set_job_metadata_value,
            job_id,
            "agent_config",
            {
                "package_size": request.package_size,
                "lease_seconds": request.lease_seconds,
                "profile": request.profile,
                "domain": request.domain,
                "target_lang": request.target_lang,
                "source_lang": request.source_lang,
                "dual_mode": request.dual_mode,
                "layout_columns": request.layout_columns,
                "glossary_path": str(config.glossary_path) if config.glossary_path else None,
                "pages": request.pages,
                "max_repair_rounds": config.max_repair_rounds,
                "export_min_completion_ratio": config.export_min_completion_ratio,
            },
        )
        # Deterministic terminology only. The provider path's skeleton/backfill
        # channels are model calls and are intentionally not invoked here.
        terminology = await asyncio.to_thread(self._build_deterministic_terminology, ctx)
        await asyncio.to_thread(
            ledger.set_job_metadata_value,
            job_id,
            "bible_cache",
            {
                "version": "agent-deterministic-v1",
                "source_lang": request.source_lang,
                "target_lang": request.target_lang,
                "profile_name": request.profile,
                "glossary_dicts": terminology.glossary_dicts,
                "abbreviation_entries": terminology.abbreviation_entries,
            },
        )
        render_plan = await asyncio.to_thread(self._render_plan, manifest, config)
        config_snapshot = {
            "draft_model": "agent-host",
            "repair_model": "agent-host",
            "qe_engine": config.qe_engine,
            "qe_judge_enabled": False,
            "qe_threshold": config.qe_threshold,
            "max_repair_rounds": config.max_repair_rounds,
            "prompt_strategy": "agent",
        }
        await asyncio.to_thread(
            ledger.set_job_metadata_value,
            job_id,
            "config_snapshot",
            config_snapshot,
        )
        await asyncio.to_thread(
            ledger.set_job_metadata_value,
            job_id,
            "render_plan",
            {
                "bilingual_mode": render_plan.bilingual_mode,
                "effective_dual_mode": render_plan.effective_dual_mode,
                "dual_mode_downgraded": render_plan.dual_mode_downgraded,
                "translate_chrome": render_plan.translate_chrome,
                "bilingual_advisory": render_plan.bilingual_advisory,
                "emit_secondary_mode": render_plan.emit_secondary_mode,
            },
        )
        if not isinstance(metadata.get(AGENT_SESSION_KEY), dict):
            await asyncio.to_thread(
                ledger.set_job_metadata_value,
                job_id,
                "agent_session",
                {
                    "mode": AGENT_MODE,
                    "state": "awaiting_agent",
                    "packages": {},
                    "created_at": _now(),
                    "updated_at": _now(),
                },
            )

    def _build_deterministic_terminology(self, ctx: StageContext) -> Terminology:
        from ubt.core.engine.stages.bible import _split_bible_entries
        from ubt.core.memory.abbreviation_miner import mine_abbreviations_stream
        from ubt.core.memory.bible import (
            BibleEntry,
            BookBible,
            clean_bible_entry,
            merge_bible_entries,
        )
        from ubt.core.memory.character_miner import mine_characters_stream
        from ubt.core.memory.seed_glossary import load_external_glossary, seed_entries_for_profile

        entries: list[BibleEntry] = []
        if ctx.config.glossary_path:
            entries.extend(load_external_glossary(ctx.config.glossary_path))
        entries.extend(
            seed_entries_for_profile(
                ctx.profile_name,
                source_lang=ctx.source_lang,
                target_lang=ctx.target_lang,
            )
        )
        chapter_translations = ctx.manifest.metadata.get("chapter_translations", {})
        for chapter in ctx.manifest.chapters:
            if chapter.title and chapter.title in chapter_translations:
                translated = str(chapter_translations[chapter.title]).strip()
                if translated and translated != chapter.title:
                    entry = clean_bible_entry(
                        source=chapter.title, translation=translated, kind="term"
                    )
                    if entry:
                        entries.append(entry)
        blocks = ctx.ledger.get_all_blocks(ctx.job_id)
        texts = [block.source_text for block in blocks if block.source_text]
        mined_abbreviations = mine_abbreviations_stream(texts)
        entries.extend(
            BibleEntry(
                source=item["source"],
                translation="",
                aliases=list(item["aliases"]),
                kind="term",
                frequency=int(item.get("frequency", 0)),
            )
            for item in mined_abbreviations
        )
        is_fiction = ctx.profile_name.lower() in ("fiction", "novel", "literature", "classic")
        entries.extend(
            BibleEntry(
                source=item["source"],
                translation="",
                aliases=list(item["aliases"]),
                kind="person",
                frequency=int(item.get("frequency", 0)),
            )
            for item in mine_characters_stream(
                texts, source_lang=ctx.source_lang, allow_bare_tokens=is_fiction
            )
        )
        merged = merge_bible_entries(entries)
        bible = BookBible(doc_id=ctx.manifest.doc_id, language=ctx.target_lang, glossary=merged)
        glossary, abbreviations = _split_bible_entries(bible.glossary)
        return Terminology(glossary_dicts=glossary, abbreviation_entries=abbreviations)

    def _render_plan(self, manifest: BookManifest, config: UBTConfig) -> RenderPlan:
        from ubt.core.router_mode import decide

        route = decide(
            manifest.source_path,
            short_max_pages=config.short_max_pages,
            exec_mode=config.exec_mode,
        )
        manifest.run.route_decision = route.to_dict()
        manifest.run.route_mode = route.mode
        policy = AdaptivePolicy(
            fast_lane_bible=False,
            visual_blocking=config.visual_blocking_gate_enabled,
            deterministic_glossary=config.deterministic_glossary_long_route,
            reason="agent translation (deterministic render policy)",
        )
        manifest.run.adaptive_policy = policy.to_dict()
        if config.layout_columns is not None:
            manifest.run.layout_columns = config.layout_columns
        return RenderPlan(
            bilingual_mode="bilingual" if config.dual_mode == "dual" else "monolingual",
            effective_dual_mode=config.dual_mode if config.dual_mode != "auto" else "mono",
        )

    @staticmethod
    def _event_factory(ledger: SQLiteJobLedger, job_id: str) -> Any:
        async def _create(
            event_type: EventType,
            job_id_arg: str,
            ledger_arg: SQLiteJobLedger,
            message: str = "",
            active_block_id: str | None = None,
            artifact_path: str | None = None,
        ) -> TranslationProgressEvent:
            stats = await asyncio.to_thread(ledger_arg.get_job_stats, job_id_arg)
            return TranslationProgressEvent(
                event_type=event_type,
                job_id=job_id_arg,
                total_blocks=int(stats.get("total", 0) or 0),
                skipped_blocks=int(stats.get("skipped", 0) or 0),
                completed_blocks=int(stats.get("completed", 0) or 0),
                drafted_blocks=int(stats.get("drafted", 0) or 0),
                repaired_blocks=int(stats.get("repaired", 0) or 0),
                failed_blocks=int(stats.get("failed", 0) or 0),
                needs_human_blocks=int(stats.get("needs_human", 0) or 0),
                blocked_human_blocks=int(stats.get("blocked_human", 0) or 0),
                message=message,
                active_block_id=active_block_id,
                artifact_path=artifact_path,
            )

        return _create

    # ------------------------------------------------------------------
    # claim / submit
    # ------------------------------------------------------------------
    async def claim(self, request: AgentClaimRequest) -> AgentClaimResult:
        db_path, ledger = await self._open_job(request.job_id)
        try:
            await self._assert_agent_job(ledger, request.job_id)
            expired = await asyncio.to_thread(
                ledger.expire_agent_packages, request.job_id, now=_now()
            )
            if expired:
                logger.info(
                    "Released %d expired Agent package(s) for %s", len(expired), request.job_id
                )
            return await self._claim_locked(ledger, request)
        finally:
            await asyncio.to_thread(ledger.close)
            del db_path

    async def _claim_locked(
        self, ledger: SQLiteJobLedger, request: AgentClaimRequest
    ) -> AgentClaimResult:
        config_meta = await asyncio.to_thread(
            ledger.get_job_metadata_value, request.job_id, "agent_config"
        )
        package_size = int(
            request.max_blocks
            or (config_meta.get("package_size") if isinstance(config_meta, dict) else None)
            or _DEFAULT_PACKAGE_SIZE
        )
        lease_seconds = float(
            (config_meta.get("lease_seconds") if isinstance(config_meta, dict) else None)
            or _DEFAULT_LEASE_SECONDS
        )
        package_id = f"pkg_{secrets.token_hex(12)}"
        lease_token = secrets.token_urlsafe(32)
        expires_at = _now() + lease_seconds
        # Selection and the lease must share one transaction; reading first and
        # writing later lets two concurrent claims lease the same blocks.
        candidates, package_meta = await asyncio.to_thread(
            ledger.claim_next_agent_work,
            request.job_id,
            package_id=package_id,
            lease_token_hash=_hash_token(lease_token),
            expires_at=expires_at,
            now=_now(),
            limit=max(1, min(package_size, _MAX_PACKAGE_SIZE)),
        )
        if not candidates:
            stats = await asyncio.to_thread(ledger.get_job_stats, request.job_id)
            packages = await asyncio.to_thread(ledger.agent_session_packages, request.job_id)
            active = sum(
                1
                for package in packages.values()
                if package.get("state") == "claimed"
                and float(package.get("expires_at") or 0.0) >= _now()
            )
            message = (
                f"No unleased blocks remain; {active} package(s) are still leased. "
                "Submit each active package before finishing."
                if active
                else (
                    f"No claimable blocks remain ({stats.get('total', 0)} total). "
                    "Call ubt_agent_finish to run deterministic gates and render."
                )
            )
            return AgentClaimResult(
                job_id=request.job_id,
                status="awaiting_agent",
                pending_blocks=0,
                next_action="ubt_agent_submit" if active else "ubt_agent_finish",
                message=message,
                active_packages=active,
            )
        kind_value = package_meta.get("kind")
        kind: Literal["translate", "repair"] = "repair" if kind_value == "repair" else "translate"
        all_blocks = await asyncio.to_thread(ledger.get_all_blocks, request.job_id)
        terminology = await self._terminology(ledger, request.job_id)
        global_glossary = build_global_glossary_table(terminology.glossary_dicts, max_entries=80)
        work_blocks = [
            self._block_work(
                block,
                kind=kind,
                terminology=terminology,
                all_blocks=all_blocks,
                target_lang=str((config_meta or {}).get("target_lang", "zh")),
                source_lang=str((config_meta or {}).get("source_lang", "en")),
            )
            for block in candidates
        ]
        system_prompt, _ = build_rich_draft_prompt(
            source_text="",
            target_lang=str((config_meta or {}).get("target_lang", "zh")),
            source_lang=str((config_meta or {}).get("source_lang", "en")),
            genre_profile=str((config_meta or {}).get("profile", "general")),
            global_glossary=global_glossary,
            domain=(config_meta or {}).get("domain"),
        )
        if kind == "repair":
            system_prompt, _ = build_rich_repair_prompt(
                source_text="",
                draft_text="",
                error_flags=[],
                target_lang=str((config_meta or {}).get("target_lang", "zh")),
                source_lang=str((config_meta or {}).get("source_lang", "en")),
                glossary_table=global_glossary,
            )
        instructions = (
            "Repair each block's draft_text against masked_source and error_flags. "
            "Preserve every ⟦..._MASK_...⟧ placeholder exactly; they are restored and "
            "checksum-verified by UBT. Return one corrected translation per block through "
            "ubt_agent_submit. Do not edit the ledger or invent block ids."
            if kind == "repair"
            else (
                "Translate each block's masked_source into target_lang. Preserve every "
                "⟦..._MASK_...⟧ placeholder exactly; they are restored and checksum-verified "
                "by UBT. Return one translation per block through ubt_agent_submit. Do not "
                "edit the ledger or invent block ids."
            )
        )
        package = AgentWorkPackage(
            package_id=package_id,
            lease_token=lease_token,
            job_id=request.job_id,
            kind=kind,
            expires_at=expires_at,
            target_lang=str((config_meta or {}).get("target_lang", "zh")),
            source_lang=str((config_meta or {}).get("source_lang", "en")),
            profile=str((config_meta or {}).get("profile", "general")),
            system_prompt=system_prompt,
            global_glossary=global_glossary,
            blocks=work_blocks,
            instructions=instructions,
        )
        stats = await asyncio.to_thread(ledger.get_job_stats, request.job_id)
        return AgentClaimResult(
            job_id=request.job_id,
            package=package,
            status="awaiting_agent",
            pending_blocks=int(stats.get("total", 0) or 0)
            - int(stats.get("completed", 0) or 0)
            - int(stats.get("drafted", 0) or 0)
            - int(stats.get("failed", 0) or 0)
            - int(stats.get("needs_human", 0) or 0)
            - int(stats.get("blocked_human", 0) or 0),
            next_action="ubt_agent_submit",
            message=f"Leased {len(candidates)} block(s); lease expires at {expires_at:.0f}.",
        )

    def _block_work(
        self,
        block: IRBlock,
        *,
        kind: str,
        terminology: Terminology,
        all_blocks: Sequence[IRBlock],
        target_lang: str,
        source_lang: str,
    ) -> AgentBlockWork:
        from ubt.core.engine.stages.draft import _source_with_bold_markers

        masked = self.engine.mask(_source_with_bold_markers(block))
        glossary_table = build_chunk_glossary_table(
            terminology.glossary_dicts,
            terminology.abbreviation_entries,
            block.source_text,
        )
        neighbor = NeighborContextBuilder().extract_from_blocks(block, list(all_blocks))
        section = section_context_for(all_blocks, [block])
        global_glossary = build_global_glossary_table(terminology.glossary_dicts, max_entries=80)
        if kind == "repair":
            _, user_prompt = build_rich_repair_prompt(
                source_text=masked.text,
                draft_text=block.target_text or block.draft_text or "",
                error_flags=block.error_flags,
                glossary_table=glossary_table,
                target_lang=target_lang,
                source_lang=source_lang,
            )
        else:
            _, user_prompt = build_rich_draft_prompt(
                source_text=masked.text,
                glossary_table=glossary_table,
                neighbor_context=neighbor,
                target_lang=target_lang,
                source_lang=source_lang,
                global_glossary=global_glossary,
                block_type=str(block.block_type),
            )
        if section:
            user_prompt = f"### Section Context\n{section}\n\n{user_prompt}"
        return AgentBlockWork(
            block_id=block.id,
            spine_index=block.spine_index,
            block_type=str(block.block_type),
            masked_source=masked.text,
            source_hash=_source_hash(block),
            user_prompt=user_prompt,
            draft_text=(block.target_text or block.draft_text) if kind == "repair" else None,
            error_flags=list(block.error_flags) if kind == "repair" else [],
            neighbor_context=neighbor,
            section_context=section,
            glossary_table=glossary_table,
        )

    async def submit(self, request: AgentSubmitRequest) -> AgentSubmitResult:
        _, ledger = await self._open_job(request.job_id)
        try:
            await self._assert_agent_job(ledger, request.job_id)
            package_meta = await asyncio.to_thread(
                ledger.get_job_metadata_value, request.job_id, AGENT_SESSION_KEY
            )
            package = (
                package_meta.get("packages", {}).get(request.package_id)
                if isinstance(package_meta, dict) and isinstance(package_meta.get("packages"), dict)
                else None
            )
            if not isinstance(package, dict):
                raise UBTError(f"unknown Agent package: {request.package_id}")
            submission_hash = _submission_hash(request.blocks)
            if package.get("state") == "submitted":
                stored_hash = package.get("submission_hash")
                if stored_hash is not None and stored_hash != submission_hash:
                    raise UBTError(
                        f"package {request.package_id} was already submitted with different text"
                    )
                stats = await asyncio.to_thread(ledger.get_job_stats, request.job_id)
                return AgentSubmitResult(
                    job_id=request.job_id,
                    package_id=request.package_id,
                    accepted_blocks=len(package.get("block_ids", [])),
                    rejected_blocks=0,
                    pending_blocks=max(
                        0,
                        int(stats.get("total", 0) or 0)
                        - int(stats.get("completed", 0) or 0)
                        - int(stats.get("drafted", 0) or 0)
                        - int(stats.get("failed", 0) or 0)
                        - int(stats.get("needs_human", 0) or 0)
                        - int(stats.get("blocked_human", 0) or 0),
                    ),
                    status="awaiting_agent",
                    next_action="ubt_agent_finish",
                    idempotent_replay=True,
                    message="This package was already accepted; returning its original result.",
                )
            blocks = await asyncio.to_thread(ledger.get_all_blocks, request.job_id)
            by_id = {block.id: block for block in blocks}
            expected = [str(b) for b in package.get("block_ids", [])]
            submitted_ids = [item.block_id for item in request.blocks]
            if sorted(submitted_ids) != sorted(expected):
                raise UBTError(
                    "submission must contain exactly the leased block ids: "
                    + ", ".join(sorted(expected))
                )
            updates: list[dict[str, Any]] = []
            package_kind = str(package.get("kind") or "")
            for item in request.blocks:
                block = by_id.get(item.block_id)
                if block is None:
                    raise UBTError(f"unknown block: {item.block_id}")
                if block.status is not BlockStatus.DRAFTED:
                    raise UBTError(
                        f"block {item.block_id} is {block.status.value}, not a leased draft"
                    )
                masked = self.engine.mask(self._source_for_block(block))
                raw = self._clean_submission(item.translation)
                result = await asyncio.to_thread(self.engine.resolve, raw, masked)
                clean_text, target_runs = parse_emphasis(result.text)
                if block.block_type.value == "heading":
                    from ubt.core.engine.stages.draft import trim_heading_expansion

                    clean_text = trim_heading_expansion(block.source_text, clean_text)
                flags = list(block.error_flags)
                for label, report in result.dirty:
                    from ubt.core.qe.defect_taxonomy import format_defect_flag

                    # Same encoding the draft stage's five token-corrupt
                    # families carry, with the severity declared: these are
                    # untrustworthy spans (MQM Critical), so the repair gate
                    # must re-queue them exactly like the provider path does.
                    detail = format_defect_flag(
                        label,
                        "critical",
                        f"missing={report.missing} mismatched={report.mismatched} "
                        f"mutated={report.mutated} reordered={report.reordered} "
                        f"duplicated={report.duplicated}",
                    )
                    flags.append(detail)
                style = (block.style or StyleMeta()).model_copy(update={"target_runs": target_runs})
                updates.append(
                    {
                        "block_id": block.id,
                        "target_text": clean_text,
                        "draft_text": clean_text,
                        "status": BlockStatus.DRAFTED
                        if result.clean
                        else BlockStatus.REPAIR_PENDING,
                        "error_flags": flags,
                        "repair_rounds": (
                            block.repair_rounds + 1
                            if package_kind == "repair"
                            else block.repair_rounds
                        ),
                        "style": style,
                    }
                )
            try:
                await asyncio.to_thread(
                    ledger.submit_agent_work,
                    request.job_id,
                    package_id=request.package_id,
                    lease_token_hash=_hash_token(request.lease_token),
                    now=_now(),
                    updates=updates,
                    package_state={"submitted_blocks": len(updates)},
                    submission_hash=submission_hash,
                )
            except LedgerError as exc:
                raise UBTError(str(exc)) from exc
            stats = await asyncio.to_thread(ledger.get_job_stats, request.job_id)
            pending = max(
                0,
                int(stats.get("total", 0) or 0)
                - int(stats.get("completed", 0) or 0)
                - int(stats.get("drafted", 0) or 0)
                - int(stats.get("failed", 0) or 0)
                - int(stats.get("needs_human", 0) or 0)
                - int(stats.get("blocked_human", 0) or 0),
            )
            return AgentSubmitResult(
                job_id=request.job_id,
                package_id=request.package_id,
                accepted_blocks=len(updates),
                rejected_blocks=0,
                pending_blocks=pending,
                status="awaiting_agent",
                next_action="ubt_agent_finish" if pending == 0 else "ubt_agent_claim",
            )
        finally:
            await asyncio.to_thread(ledger.close)

    def _source_for_block(self, block: IRBlock) -> str:
        from ubt.core.engine.stages.draft import _source_with_bold_markers

        return _source_with_bold_markers(block)

    @staticmethod
    def _clean_submission(text: str) -> str:
        from ubt.core.router.capabilities import ExtractionStrategy
        from ubt.core.router.extractor import TranslationOutputExtractor

        return TranslationOutputExtractor.extract(text, ExtractionStrategy.AUTO).strip()

    # ------------------------------------------------------------------
    # finish
    # ------------------------------------------------------------------
    async def finish(self, request: AgentFinishRequest) -> AgentFinishResult:
        db_path, ledger = await self._open_job(request.job_id)
        lock = LedgerWriterLock(db_path, request.job_id)
        await asyncio.to_thread(lock.acquire)
        lease_lost = asyncio.Event()
        renew_task = asyncio.create_task(
            keep_lease_fresh(lock, on_lost=lambda _reason: lease_lost.set()),
            name=f"renew-agent-finish:{request.job_id}",
        )
        session_state = "awaiting_agent"
        try:
            await self._assert_agent_job(ledger, request.job_id)
            persisted = await asyncio.to_thread(ledger.get_job_status, request.job_id)
            if persisted == "completed":
                output_path = await asyncio.to_thread(
                    ledger.get_job_metadata_value, request.job_id, "output_file"
                )
                report_path = await asyncio.to_thread(
                    ledger.get_job_metadata_value, request.job_id, "report_file"
                )
                return AgentFinishResult(
                    job_id=request.job_id,
                    status="completed",
                    output_path=str(output_path) if output_path else None,
                    report_path=str(report_path) if report_path else None,
                    next_action="none",
                    message="Agent translation is already completed; returning its artifacts.",
                )
            active_packages = await asyncio.to_thread(
                ledger.begin_agent_finish, request.job_id, now=_now()
            )
            if active_packages:
                session_state = "awaiting_agent"
                return AgentFinishResult(
                    job_id=request.job_id,
                    status="awaiting_agent",
                    next_action="ubt_agent_claim",
                    message=(
                        f"{len(active_packages)} Agent package(s) are still actively leased. "
                        "Submit them before finishing; expired leases were released."
                    ),
                )
            self._lease_lost = lease_lost
            manifest = await asyncio.to_thread(self._manifest_from_ledger, ledger, request.job_id)
            config = await asyncio.to_thread(self._config_from_agent_metadata, manifest)
            blocks = await asyncio.to_thread(ledger.get_all_blocks, request.job_id)
            # Submitted packages leave blocks in DRAFTED so a later finish can
            # distinguish "Agent returned text" from "already passed the gate".
            # Run the deterministic screen first; only PENDING/REPAIR_PENDING
            # blocks are unfinished work at this point.
            if any(
                block.status in (BlockStatus.DRAFTED, BlockStatus.REPAIR_PENDING)
                for block in blocks
            ):
                await self._quality_screen(ledger, request.job_id, manifest, config)
                blocks = await asyncio.to_thread(ledger.get_all_blocks, request.job_id)
            non_terminal = [b for b in blocks if b.status not in TERMINAL_STATUSES]
            if non_terminal:
                pending = [b for b in non_terminal if b.status is BlockStatus.PENDING]
                repairs = [b for b in non_terminal if b.status is BlockStatus.REPAIR_PENDING]
                status: Literal["repair_required", "needs_attention"] = (
                    "repair_required" if repairs else "needs_attention"
                )
                session_state = status
                await asyncio.to_thread(
                    ledger.set_agent_session_state,
                    request.job_id,
                    status,
                    now=_now(),
                )
                result = AgentFinishResult(
                    job_id=request.job_id,
                    status=status,
                    pending_blocks=len(pending),
                    repair_blocks=len(repairs),
                    issues=[
                        AgentIssue(
                            block_id=b.id, status=b.status.value, error_flags=list(b.error_flags)
                        )
                        for b in non_terminal[:20]
                    ],
                    next_action="ubt_agent_claim",
                    message=(
                        f"{len(non_terminal)} block(s) are not terminal; claim and submit them "
                        "before finishing."
                    ),
                )
                if lease_lost.is_set():
                    raise UBTError("Agent finish stopped: the writer lease was reclaimed")
                return result
            event = await self._render(ledger, db_path, request.job_id, manifest, config)
            if lease_lost.is_set():
                raise UBTError("Agent finish stopped: the writer lease was reclaimed")
            report_path = await asyncio.to_thread(
                ledger.get_job_metadata_value, request.job_id, "report_file"
            )
            session_state = "completed"
            await asyncio.to_thread(
                ledger.set_agent_session_state,
                request.job_id,
                "completed",
                now=_now(),
            )
            self._lease_lost = None
            return AgentFinishResult(
                job_id=request.job_id,
                status="completed",
                output_path=str(event.artifact_path) if event.artifact_path else None,
                report_path=str(report_path) if report_path else None,
                next_action="none",
                message="Agent translation passed deterministic checks and was rendered.",
            )
        finally:
            if lease_lost.is_set():
                logger.error("Agent finish for %s lost its writer lease", request.job_id)
            elif session_state != "completed":
                try:
                    await asyncio.to_thread(
                        ledger.set_agent_session_state,
                        request.job_id,
                        session_state,
                        now=_now(),
                    )
                except Exception:
                    # Best effort only: a failed reset must not mask the error
                    # that made finish fail, and the next finish call re-derives
                    # the session state from the blocks.
                    logger.exception(
                        "Agent finish for %s could not reset session state to %s",
                        request.job_id,
                        session_state,
                    )
            self._lease_lost = None
            renew_task.cancel()
            with suppress(asyncio.CancelledError):
                await renew_task
            await asyncio.to_thread(ledger.close)
            await asyncio.to_thread(lock.release)

    async def _quality_screen(
        self,
        ledger: SQLiteJobLedger,
        job_id: str,
        manifest: BookManifest,
        config: UBTConfig,
    ) -> None:
        terminology = await self._terminology(ledger, job_id)
        blocks = await asyncio.to_thread(ledger.get_all_blocks, job_id)
        candidates = [
            b for b in blocks if b.status in (BlockStatus.DRAFTED, BlockStatus.REPAIR_PENDING)
        ]
        if not candidates:
            return
        fast_pass = FastPassFilter(
            source_lang=manifest.source_lang, target_lang=manifest.target_lang
        )
        glossary_validator = (
            GlossaryConsistencyValidator(glossary=terminology.glossary_dicts)
            if terminology.glossary_dicts
            else None
        )
        suspicious, passed = await asyncio.to_thread(
            screen_drafted_blocks,
            candidates,
            fast_pass=fast_pass,
            glossary_validator=glossary_validator,
            glossary_terms=[str(t.get("source")) for t in terminology.glossary_dicts],
        )
        updates = list(passed)
        for block in suspicious:
            max_rounds = max(0, int(config.max_repair_rounds))
            if block.repair_rounds >= max_rounds:
                # The Agent repair budget is exhausted. Do not silently render a
                # known-bad draft; route it to the same human terminal states the
                # provider pipeline uses.
                from ubt.core.qe.defect_taxonomy import has_critical_defect

                critical = has_critical_defect(block.error_flags)
                updates.append(
                    {
                        "block_id": block.id,
                        "target_text": block.target_text or "",
                        "status": (
                            BlockStatus.BLOCKED_HUMAN if critical else BlockStatus.NEEDS_HUMAN
                        ),
                        "error_flags": block.error_flags,
                    }
                )
            else:
                updates.append(
                    {
                        "block_id": block.id,
                        "target_text": block.target_text or "",
                        "status": BlockStatus.REPAIR_PENDING,
                        "error_flags": block.error_flags,
                    }
                )
        await asyncio.to_thread(ledger.save_checkpoints_batch, updates, job_id=job_id)

    async def _render(
        self,
        ledger: SQLiteJobLedger,
        db_path: Path,
        job_id: str,
        manifest: BookManifest,
        config: UBTConfig,
    ) -> TranslationProgressEvent:
        render_config = config.model_copy(
            update={"visual_judge_enabled": False, "ocr_mode": "off", "allow_page_upload": False}
        )
        router = _agent_router()
        fast_pass = FastPassFilter(
            source_lang=manifest.source_lang, target_lang=manifest.target_lang
        )
        qe = HeuristicQERunner(
            target_lang=manifest.target_lang,
            source_lang=manifest.source_lang,
            fast_pass=fast_pass,
        )
        services = RunServices(
            fast_pass=fast_pass,
            qe_runner=qe,
            repair_loop=RepairLoop(
                router=router,
                qe_runner=qe,
                max_rounds=config.max_repair_rounds,
                qe_threshold=config.qe_threshold,
                fast_pass=fast_pass,
            ),
            adaptive_policy=AdaptivePolicy(
                fast_lane_bible=False,
                visual_blocking=render_config.visual_blocking_gate_enabled,
                deterministic_glossary=render_config.deterministic_glossary_long_route,
                reason="agent finish (render-only)",
            ),
            concurrency_sem=asyncio.Semaphore(1),
        )
        terminology = await self._terminology(ledger, job_id)
        source_path = Path(manifest.source_path)
        output_path = Path(
            str(
                ledger.get_job_metadata_value(job_id, "output_file")
                or (source_path.with_name(f"{source_path.stem}_translated{source_path.suffix}"))
            )
        )
        adapter = resolve_adapter(source_path, pdf_engine=render_config.pdf_engine)
        allowed: frozenset[str] = getattr(type(adapter), "output_suffixes", frozenset())
        if allowed and output_path.suffix and output_path.suffix.lower() not in allowed:
            raise UnsupportedDocumentFormatError(
                f"{type(adapter).__name__} writes {'/'.join(sorted(allowed))}, not {output_path.suffix}"
            )
        render_plan = await asyncio.to_thread(self._render_plan, manifest, config)
        ctx = StageContext(
            config=render_config,
            router=router,
            ledger=ledger,
            manifest=manifest,
            adapter=adapter,
            job_id=job_id,
            input_path=source_path,
            output_path=output_path,
            source_lang=manifest.source_lang,
            target_lang=manifest.target_lang,
            profile_name=str(
                (ledger.get_job_metadata_value(job_id, "agent_config") or {}).get(
                    "profile", "general"
                )
            ),
            create_event=self._event_factory(ledger, job_id),
            lease_lost=self._lease_lost,
        )
        terminal: TranslationProgressEvent | None = None
        async for event in run_export_stage(
            ctx,
            services,
            render_plan,
            terminology,
            claims_finality=True,
        ):
            if event.event_type is EventType.EXPORT_COMPLETED:
                terminal = event
        if terminal is None:
            raise UBTError("Agent render produced no terminal export event")
        from ubt.core.engine.progress import persist_progress_metadata

        await asyncio.to_thread(persist_progress_metadata, terminal, ledger, job_id)
        return terminal

    async def _terminology(self, ledger: SQLiteJobLedger, job_id: str) -> Terminology:
        raw = await asyncio.to_thread(ledger.get_job_metadata_value, job_id, "bible_cache")
        if not isinstance(raw, dict):
            return Terminology()
        return Terminology(
            glossary_dicts=[dict(e) for e in raw.get("glossary_dicts", []) if isinstance(e, dict)],
            abbreviation_entries=[
                dict(e) for e in raw.get("abbreviation_entries", []) if isinstance(e, dict)
            ],
        )

    def _config_from_agent_metadata(self, manifest: BookManifest) -> UBTConfig:
        """Reconstruct the prepare-time config that affects finish/render."""
        raw = manifest.metadata.pop("_agent_config", None)
        if not isinstance(raw, dict):
            return self.config
        overrides = {
            key: raw.get(key)
            for key in (
                "target_lang",
                "source_lang",
                "profile",
                "domain",
                "dual_mode",
                "layout_columns",
                "glossary_path",
                "pages",
                "max_repair_rounds",
                "export_min_completion_ratio",
            )
            if raw.get(key) is not None
        }
        if overrides.get("profile") == "auto":
            overrides.pop("profile", None)
        from ubt.core.job_options import apply_config_overrides

        return apply_config_overrides(self.config, overrides)

    @staticmethod
    def _manifest_from_ledger(ledger: SQLiteJobLedger, job_id: str) -> BookManifest:
        from ubt.core.ir.models import ChapterMeta
        from ubt.core.ir.run_metadata import RunMetadata

        snapshot = ledger.get_job_snapshot(job_id)
        if snapshot is None:
            raise UBTError(f"no such job: {job_id}")
        chapters_raw = ledger.get_job_metadata_value(job_id, "chapters") or []

        chapters = []
        if isinstance(chapters_raw, list):
            for entry in chapters_raw:
                if isinstance(entry, dict):
                    with suppress(Exception):
                        chapters.append(ChapterMeta.model_validate(entry))
        metadata = ledger.get_job_metadata(job_id)
        artifact_metadata = {
            key: value
            for key, value in metadata.items()
            if key
            in {
                "author",
                "chapter_translations",
                "opf_path",
                "pdf_parser_engine",
                "boilerplate_footers",
                "is_page_slice_epub",
                "page_kinds",
                "typst_version",
            }
        }
        run = RunMetadata()
        for field in RunMetadata.model_fields:
            if field in metadata:
                setattr(run, field, metadata[field])
        manifest = BookManifest(
            doc_id=str(snapshot["doc_id"]),
            title=str(ledger.get_job_metadata_value(job_id, "title") or ""),
            source_path=str(snapshot["source_path"]),
            source_lang=str(ledger.get_job_metadata_value(job_id, "source_lang") or "en"),
            target_lang=str(snapshot["target_lang"]),
            chapters=chapters,
            metadata=artifact_metadata,
            run=run,
        )
        agent_config = metadata.get("agent_config")
        if isinstance(agent_config, dict):
            manifest.metadata["_agent_config"] = agent_config
        return manifest

    async def _open_job(self, job_id: str) -> tuple[Path, SQLiteJobLedger]:
        db_path = self.config.db_dir / f"{job_id}.sqlite"
        if not db_path.exists():
            raise UBTError(f"no such job: {job_id}")
        ledger = await asyncio.to_thread(SQLiteJobLedger, db_path)
        return db_path, ledger

    @staticmethod
    async def _assert_agent_job(ledger: SQLiteJobLedger, job_id: str) -> None:
        mode = await asyncio.to_thread(ledger.get_job_metadata_value, job_id, "translation_mode")
        if mode != AGENT_MODE:
            raise UBTError(f"job {job_id} is not an Agent translation job")


__all__ = [
    "AGENT_MODE",
    "AgentBlockSubmission",
    "AgentBlockWork",
    "AgentClaimRequest",
    "AgentClaimResult",
    "AgentFinishRequest",
    "AgentFinishResult",
    "AgentIssue",
    "AgentPrepareRequest",
    "AgentPrepareResult",
    "AgentSubmitRequest",
    "AgentSubmitResult",
    "AgentTranslationService",
    "AgentWorkPackage",
]
