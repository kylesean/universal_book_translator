"""Pydantic models for the UBT REST API."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ubt.core.config import (
    CoverMode,
    DualMode,
    ExecMode,
    FormulaEnrichment,
    FormulaMode,
    FormulaRender,
    MathBackend,
    OcrMode,
    PdfEngine,
    PromptStrategyName,
    QeEngine,
    parse_page_ranges,
)
from ubt.core.engine.progress import ProgressSnapshot
from ubt.core.job_options import LANG_CODE_PATTERN, PROFILE_NAME_PATTERN
from ubt.core.language_profile import is_supported_lang, supported_lang_codes
from ubt.core.presets import Preset


def _require_supported_target_lang(value: str) -> str:
    """Reject a target language the engine has no profile for, at parse time.

    ``LANG_CODE_PATTERN`` only checks the *shape*, so ``pt-BR`` used to pass the
    API and then die inside the pipeline after a full ingest. Region/script tags
    of a supported base language (``zh-CN``) stay accepted.
    """
    if not is_supported_lang(value):
        raise ValueError(
            f"Unsupported target language {value!r}. Supported base languages: "
            f"{', '.join(supported_lang_codes())} (region tags such as 'zh-CN' are accepted)."
        )
    return value


class JobSubmitRequest(BaseModel):
    """Payload to initiate an asynchronous translation job."""

    input_path: str = Field(..., description="Path to source document file")
    output_path: str | None = Field(
        default=None, description="Optional target path for bilingual file"
    )
    target_lang: str = Field(
        default="zh",
        description="Target ISO language code",
        pattern=LANG_CODE_PATTERN,
    )
    source_lang: str = Field(
        default="en",
        description="Source ISO language code",
        pattern=LANG_CODE_PATTERN,
    )
    # NOTE: source_lang is shape-validated only, on purpose. The shared gate
    # (``lang_pair_validation_error``) and ``get_pair_policy`` tolerate an
    # unknown source by falling back to the target profile's defaults, so
    # requiring a source profile here would make the API stricter than the CLI
    # and reject jobs the engine runs fine. Only the *target* needs a profile.
    profile: str = Field(
        default="general",
        description="Domain profile (general, textbook, paper)",
        pattern=PROFILE_NAME_PATTERN,
    )
    draft_model: str | None = Field(default=None, description="Override draft model tier")
    repair_model: str | None = Field(default=None, description="Override repair model tier")
    preset: Preset | None = Field(
        default=None, description="Quality preset (publication, standard, preview)"
    )
    pdf_engine: PdfEngine | None = Field(
        default=None,
        description="PDF extraction/parser engine (docling, pdfium, auto)",
    )
    dual_mode: DualMode | None = Field(
        default=None,
        description="Bilingual mode (inline, alternating, facing, monolingual, auto)",
    )
    pages: str | None = Field(default=None, description="Page range filter (e.g. 1-10)")

    @field_validator("pages")
    @classmethod
    def _validate_pages(cls, value: str | None) -> str | None:
        # Reject a malformed range at parse time (422) instead of failing the
        # job asynchronously after a 202, and bound the raw string before it is
        # split (a multi-MB ``pages`` field was a body-size DoS).
        if value is None:
            return value
        try:
            parse_page_ranges(value)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        return value

    glossary: str | None = Field(
        default=None,
        description="Path to an external glossary file to enforce (CLI/MCP parity)",
    )
    exec_mode: ExecMode | None = Field(
        default=None, description="Execution mode (auto, short, long)"
    )
    formula_mode: FormulaMode | None = Field(
        default=None, description="Formula handling (strict, readable)"
    )
    # Engine knobs. The CLI builds 40+ request keys and the shared mapping
    # (``ubt.core.job_options.overrides_from_request``) accepts any of them that
    # names a ``UBTConfig`` field — so a key absent from *this* model is not
    # "ignored", it is a 422 before a job ever exists. A narrower model meant a
    # REST client could not cap spend, raise the concurrency ceiling, pick the
    # OCR engine or pin the formula policy at all. Provider credentials and
    # endpoints are deliberately NOT fields here: they may only come from the
    # operator's environment (defence in depth with ``allow_provider_keys=False``).
    #
    # Bounds below mirror the config's own (``gt=0``, ``le=30``). They exist for
    # a fast 422; the authoritative check still happens when the override is
    # assigned onto ``UBTConfig`` at job start, so a config-side tightening is
    # enforced there rather than silently diverging here.
    #
    # Cost and throughput ceilings.
    budget_usd: float | None = Field(
        default=None,
        gt=0,
        description="Hard USD cap for this job across every resume; the run fails once priced cost exceeds it",
    )
    max_concurrency: int | None = Field(
        default=None, gt=0, description="Concurrent LLM requests for this job"
    )
    batch_limit: int | None = Field(default=None, gt=0, description="Blocks per draft batch")
    macro_chunk_size: int | None = Field(
        default=None,
        ge=1,
        le=30,
        description="Consecutive micro-blocks packed into one draft request (1 = single block)",
    )
    short_max_pages: int | None = Field(
        default=None, gt=0, description="Page ceiling for the short execution chain"
    )
    # Long-chain behaviour.
    enable_rolling_summary: bool | None = Field(
        default=None, description="Carry a rolling chapter summary across draft batches"
    )
    chapter_streaming_enabled: bool | None = Field(
        default=None, description="Translate chapter-by-chapter as discovery lands"
    )
    offline_batch_enabled: bool | None = Field(
        default=None, description="Route drafting through the provider's offline batch API"
    )
    qe_engine: QeEngine | None = Field(
        default=None,
        description="QE runner (heuristic, comet, cometkiwi, neural, subprocess, tiered)",
    )
    # Quality gates.
    visual_judge_enabled: bool | None = Field(
        default=None, description="Enable the VLM visual judge over rendered pages"
    )
    visual_judge_model: str | None = Field(
        default=None, description="Model the visual judge runs on"
    )
    prompt_strategy: PromptStrategyName | None = Field(
        default=None, description="Prompt depth (auto, minimal, hybrid, rich)"
    )
    # Output shape.
    translate_chrome: bool | None = Field(
        default=None, description="Translate running heads, footers and page chrome"
    )
    facing_spread: bool | None = Field(
        default=None, description="Lay the bilingual output out as facing pages"
    )
    emit_both: bool | None = Field(
        default=None, description="Emit both the reflow and the rigid deliverable"
    )
    emit_companion_rigid: bool | None = Field(
        default=None,
        description="Emit a zero-cost '*_rigid.pdf' source-fidelity companion alongside the reflow output",
    )
    cover_mode: CoverMode | None = Field(
        default=None, description="Cover handling (auto, always, never)"
    )
    # Formula pipeline.
    formula_enrichment: FormulaEnrichment | None = Field(
        default=None, description="Formula enrichment (auto, on, off)"
    )
    formula_render: FormulaRender | None = Field(
        default=None, description="Formula rendering (native, image, witness)"
    )
    math_backend: MathBackend | None = Field(
        default=None,
        description="Retired knob: recorded in the engine signature but no renderer "
        "consumes it; formula fidelity is controlled by formula_render.",
    )
    # OCR engine selection. ``ocr_endpoint`` / ``ocr_api_key`` stay operator-only.
    ocr_mode: OcrMode | None = Field(
        default=None, description="OCR engine (auto, sidecar, cloud, vlm, rapidocr, off)"
    )
    # Content selection.
    domain: str | None = Field(
        default=None, description="Domain descriptor used for glossary and chrome hints"
    )
    # Idempotency / resume: resubmitting the same job_id returns the existing
    # job instead of starting (and billing) a duplicate run.
    job_id: str | None = Field(default=None, description="Optional stable job id (idempotency key)")
    # Queue mode scheduling: higher runs sooner; ignored by embedded mode.
    priority: int = Field(default=0, ge=0, le=100, description="Queue priority (higher = sooner)")
    # Rehearsal: run the full pipeline against the deterministic echo provider
    # (zero token spend). Explicit dry_run=True required for mock translation.
    dry_run: bool = Field(default=False, description="Zero-token rehearsal run (echo provider)")
    # Resume controls (UBTConfig fields, forwarded by overrides_from_request).
    # ``None`` (not ``False``) so a client that omits the field leaves the
    # operator's UBT_FRESH in force: overrides_from_request only applies non-None
    # values, and a concrete False silently erased the environment setting.
    fresh: bool | None = Field(
        default=None,
        description="Discard prior ledger state instead of resuming (unset follows UBT_FRESH)",
    )
    start_chapter: int | None = Field(
        default=None, ge=1, description="First chapter of the window (run-only key)"
    )
    max_chapters: int | None = Field(
        default=None, ge=1, description="Chapter-window length (run-only key)"
    )
    # Reject unknown fields to fail fast on typos and prevent unwanted parameter injection.
    model_config = ConfigDict(extra="forbid")

    @field_validator("target_lang")
    @classmethod
    def _validate_target_lang(cls, value: str) -> str:
        return _require_supported_target_lang(value)


class JobAssessRequest(BaseModel):
    """Payload to assess a cold document without translating."""

    model_config = ConfigDict(extra="forbid")

    input_path: str = Field(..., description="Path to source document file")
    deep: bool = Field(default=False, description="Run deep real adapter ingest for exact counts")
    target_lang: str = Field(
        default="zh",
        description="Target ISO language code",
        pattern=LANG_CODE_PATTERN,
    )
    source_lang: str = Field(
        default="en",
        description="Source ISO language code",
        pattern=LANG_CODE_PATTERN,
    )
    preset: Preset | None = Field(
        default=None, description="Quality preset (publication, standard, preview)"
    )
    pages: str | None = Field(default=None, description="Page range filter (e.g. 1-10)")

    @field_validator("pages")
    @classmethod
    def _validate_pages(cls, value: str | None) -> str | None:
        if value is None:
            return value
        try:
            parse_page_ranges(value)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        return value

    @field_validator("target_lang")
    @classmethod
    def _validate_target_lang(cls, value: str) -> str:
        return _require_supported_target_lang(value)


# --------------------------------------------------------------------------- #
# Assessment report (mirrors ubt.core.assess.AssessmentReport so the OpenAPI
# schema — and therefore the generated TS types — is real instead of a bare
# ``additionalProperties: true`` object).
# --------------------------------------------------------------------------- #


class AssessDocumentFacts(BaseModel):
    """Structural facts about the source document."""

    model_config = ConfigDict(extra="ignore")

    file_name: str
    file_size_bytes: int
    format_ext: str
    pages: int
    chapters: int
    source_chars: int
    estimated_tokens: int
    category: str
    detected_domain: str
    domain_confidence: float
    math_density: str
    is_scanned: bool
    primary_engine: str | None = None
    has_vector_diagrams: bool | None = None
    has_multicolumn: bool | None = None
    has_formulas: bool | None = None
    text_layer_coverage: float | None = None
    scan_page_share: float | None = None
    selected_pages: int | None = None


class AssessRoute(BaseModel):
    """The engine's recommended routing for this document."""

    model_config = ConfigDict(extra="ignore")

    mode: str
    reason: str
    recommended_preset: str
    recommended_dual_mode: str
    recommended_profile: str
    confidence: float
    confidence_basis: str


class AssessCost(BaseModel):
    """The cost quote, split by component so the UI can show the drivers."""

    model_config = ConfigDict(extra="ignore")

    draft_model: str
    repair_model: str
    prefix_tokens_per_call: int | None = None
    billable_blocks: int
    billable_blocks_is_exact: bool
    prompt_tokens: int
    completion_tokens: int
    draft_cost_usd_cached: float | None = None
    draft_cost_usd_uncached: float | None = None
    repair_blocks: int
    repair_cost_usd: float | None = None
    qe_calls: int
    qe_cost_usd: float | None = None
    vlm_page_calls: int
    ocr_page_calls: int
    vision_cost_usd: float | None = None
    rollup_calls: int
    total_cost_usd: float | None = None


class AssessRuntime(BaseModel):
    """A heuristic wall-clock estimate (no per-stage history is persisted)."""

    model_config = ConfigDict(extra="ignore")

    heuristic: bool
    est_seconds_low: float
    est_seconds_high: float
    basis: str


class AssessWarning(BaseModel):
    """One degraded probe: a stable machine code plus Chinese human copy."""

    model_config = ConfigDict(extra="ignore")

    code: str
    level: str
    detail_zh: str


class JobAssessResponse(BaseModel):
    """The full pre-flight assessment report."""

    model_config = ConfigDict(extra="ignore")

    schema_version: int
    status: str
    path: str
    deep: bool
    document: AssessDocumentFacts
    route: AssessRoute
    cost: AssessCost
    runtime: AssessRuntime
    quality_signals: list[str]
    warnings: list[AssessWarning]
    next_step_command: str
    meta: dict[str, Any]


class JobSummary(BaseModel):
    """One row of the operator console's job queue."""

    model_config = ConfigDict(extra="ignore")

    job_id: str
    file_name: str
    source_path: str
    target_lang: str
    status: str
    total_blocks: int
    completed_blocks: int
    failed_blocks: int
    needs_human_blocks: int
    progress_percent: float
    estimated_cost_usd: float | None = None
    created_at: str | None = None
    updated_at: str | None = None
    has_output: bool


class JobListResponse(BaseModel):
    """The job queue, newest first."""

    model_config = ConfigDict(extra="ignore")

    jobs: list[JobSummary]


class SystemInfoResponse(BaseModel):
    """The console's security-boundary panel: where this server is reachable and
    what it is allowed to touch."""

    model_config = ConfigDict(extra="ignore")

    version: str
    host: str = Field(description="Host this request reached (client-visible)")
    is_loopback: bool = Field(description="True when the request arrived on a loopback host")
    auth_enabled: bool = Field(description="True when an API key gate is configured")
    allowed_bases: list[str] = Field(description="Filesystem roots the server will read/write")
    db_dir: str
    job_mode: str = Field(description="embedded (in-process) or queue (worker-drained)")
    disk_free_gb: float | None = Field(
        default=None, description="Free disk space in GB on the db_dir volume"
    )
    wal_status: str | None = Field(default=None, description="SQLite WAL health status string")


class SegmentEditRequest(BaseModel):
    """A human post-edit of one block's target text (L3 workbench)."""

    model_config = ConfigDict(extra="forbid")

    target_text: str = Field(..., min_length=1, description="Revised target text")


class GlossaryTermRequest(BaseModel):
    """One glossary term to add to (or remove from) the configured glossary file."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(..., min_length=1, description="Source term")
    target: str = Field(default="", description="Target translation (required when adding)")


class TermPropagationRequest(BaseModel):
    """Replace one offending term surface with its canonical rendering (L3)."""

    model_config = ConfigDict(extra="forbid")

    block_id: str = Field(..., min_length=1, description="Block the action was invoked from")
    surface: str = Field(..., min_length=1, description="Offending surface to replace")
    expected: str = Field(..., min_length=1, description="Canonical rendering to write")
    scope: Literal["block", "subsequent", "all"] = Field(
        default="all",
        description="block = only this block, subsequent = this and later blocks, all = whole book",
    )


class TMevictRequest(BaseModel):
    """Translation-memory rows to evict, by id."""

    model_config = ConfigDict(extra="forbid")

    # Bounded so one request cannot submit an unbounded delete set (the store
    # deletes in batches, but the parsed list is held whole first).
    ids: list[int] = Field(
        default_factory=list, max_length=10_000, description="TM entry ids to delete"
    )


class TMImportRequest(BaseModel):
    """Import translation-memory pairs from a TMX or JSON payload (L4 assets)."""

    model_config = ConfigDict(extra="forbid")

    format: Literal["tmx", "json"] = Field(..., description="Payload format")
    content: str = Field(..., min_length=1, description="Raw TMX XML or JSON text")
    src_lang: str = Field(
        ..., min_length=1, description="Fallback source language for rows without one"
    )
    tgt_lang: str = Field(
        ..., min_length=1, description="Fallback target language for rows without one"
    )
    provenance: Literal["machine", "human_pe"] = Field(
        default="machine",
        description="machine = unverified import (cannot downgrade existing human rows)",
    )


class JobUploadResponse(BaseModel):
    """Result of staging an uploaded source document on the server."""

    file_path: str = Field(..., description="Server-side path to pass as input_path")
    file_name: str = Field(..., description="Original client-side file name")
    size_bytes: int = Field(..., ge=0, description="Stored file size in bytes")


class JobDeleteResponse(BaseModel):
    """Result of removing a finished job's history from the console."""

    job_id: str
    removed_ledger: bool = Field(..., description="The db_dir ledger file(s) were deleted")
    removed_outputs: bool = Field(..., description="The per-job deliverable directory was deleted")


class JobSubmitResponse(BaseModel):
    """Response returned upon successful job enqueueing."""

    job_id: str
    status: str
    stream_url: str
    status_url: str
    # True when this job will run as a zero-token rehearsal (echo provider)
    # instead of a billed translation — set explicitly via dry_run=true or
    # auto-set at intake when no provider key is configured, so a keyless
    # server can never report a mock run as a real delivery.
    rehearsal: bool = False


class JobStatusResponse(ProgressSnapshot):
    """Real-time job status and metric counters.

    The telemetry fields are the shared :class:`ProgressSnapshot` -- one
    definition for the queue row, the in-memory record and this response.
    """

    job_id: str
    # One vocabulary (ubt.core.engine.job_queue.JobStatus): "submitted"
    # is the in-memory manager's accepted state; queue mode reports "queued".
    status: str
    created_at: datetime
    error: str | None = None
    # Queue mode only: 1-based position among queued jobs (None = not queued).
    queue_position: int | None = None
    # True when the job ran (is running) as a zero-token rehearsal instead of a
    # billed translation — surfaced so a keyless server can never report a mock
    # run as a finished delivery without saying so.
    rehearsal: bool = False
