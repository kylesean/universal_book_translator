"""Pydantic models for the UBT REST API."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ubt.api.provider_editor import ProviderEditModel
from ubt.core.config import (
    DualMode,
    OcrMode,
    PdfEngine,
    parse_page_ranges,
)
from ubt.core.engine.progress import ProgressSnapshot
from ubt.core.job_options import LANG_CODE_PATTERN, PROFILE_NAME_PATTERN
from ubt.core.language_profile import is_supported_lang, supported_lang_codes
from ubt.core.presets import Preset


def _require_supported_target_lang(value: str) -> str:
    """Reject a target language the engine has no profile for, at parse time.

    ``LANG_CODE_PATTERN`` only checks the *shape*, so a shape-valid but
    unsupported code (``pt-BR``) would otherwise pass the API and die inside the
    pipeline after a full ingest. Region/script tags of a supported base
    language (``zh-CN``) stay accepted.
    """
    if not is_supported_lang(value):
        raise ValueError(
            f"Unsupported target language {value!r}. Supported base languages: "
            f"{', '.join(supported_lang_codes())} (region tags such as 'zh-CN' are accepted)."
        )
    return value


def _require_valid_page_ranges(value: str | None) -> str | None:
    """Reject a malformed page range at parse time.

    Failing here (422) beats failing the job asynchronously after a 202, and it
    bounds the raw string before it is split (a multi-MB ``pages`` field would
    otherwise be a body-size DoS).
    """
    if value is None:
        return value
    try:
        parse_page_ranges(value)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
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
    #: Which declared ``[providers.<name>]`` block this job runs against.
    #:
    #: A *name*, not an endpoint: the URL, the wire protocol and the credential all
    #: still come from the operator's file. Selecting a different provider per job
    #: is therefore not a credential-injection path — see the reasoning in
    #: ``ubt.core.job_options.overrides_from_request``. The name is validated
    #: against the declared set, so an unknown one is a 400 naming it.
    provider: str | None = Field(
        default=None,
        description=(
            "Provider block to run this job against (must be declared in ubt.toml "
            "or be a built-in wire protocol). Endpoint and credentials still come "
            "from that block."
        ),
    )
    draft_model: str | None = Field(default=None, description="Override draft model tier")
    repair_model: str | None = Field(default=None, description="Override repair model tier")
    preset: Preset | None = Field(
        default=None,
        description="Quality preset (publication, standard, publication_math, preview, fast)",
    )
    pdf_engine: PdfEngine | None = Field(
        default=None,
        description="PDF extraction/parser engine (docling, pdfium, auto)",
    )
    dual_mode: DualMode | None = Field(
        default=None,
        description="What the reader receives (dual, mono, auto)",
    )
    pages: str | None = Field(default=None, description="Page range filter (e.g. 1-10)")

    @field_validator("pages")
    @classmethod
    def _validate_pages(cls, value: str | None) -> str | None:
        return _require_valid_page_ranges(value)

    glossary: str | None = Field(
        default=None,
        description="Path to an external glossary file to enforce (CLI/MCP parity)",
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
    # Quality gates.
    visual_judge_enabled: bool | None = Field(
        default=None, description="Enable the VLM visual judge over rendered pages"
    )
    visual_judge_model: str | None = Field(
        default=None, description="Model the visual judge runs on"
    )
    # Output shape.
    translate_chrome: bool | None = Field(
        default=None, description="Translate running heads, footers and page chrome"
    )
    emit_both: bool | None = Field(
        default=None,
        description="Emit both the primary and complementary dual_mode deliverables (e.g. mono and bilingual)",
    )
    # Formula pipeline.
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
    layout_columns: int | None = Field(
        default=None,
        description=(
            "Physical column count for a reflowed PDF (1 single, 2 dual). Unset "
            "aligns to the source layout; --profile never forces a column count."
        ),
    )
    # Reject unknown fields to fail fast on typos and prevent unwanted parameter injection.
    model_config = ConfigDict(extra="forbid")

    @field_validator("target_lang")
    @classmethod
    def _validate_target_lang(cls, value: str) -> str:
        return _require_supported_target_lang(value)

    @field_validator("layout_columns")
    @classmethod
    def _validate_layout_columns(cls, value: int | None) -> int | None:
        if value is not None and value not in (1, 2):
            raise ValueError("layout_columns must be 1 or 2")
        return value


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
        default=None,
        description="Quality preset (publication, standard, publication_math, preview, fast)",
    )
    pages: str | None = Field(default=None, description="Page range filter (e.g. 1-10)")
    # Every knob below must price the same work ``JobSubmitRequest`` will run.
    # The quote used to be built from ``[defaults]`` alone, so the wizard showed
    # a dollar figure for whichever provider the *file* named while the dropdown
    # above it selected a different one — and an operator picking cloud OCR saw a
    # quote with no vision line at all, because assess never learned the mode.
    # The names — not their values — are what matters: a provider is a declared
    # block, and the endpoint/credential still come from that block.
    provider: str | None = Field(
        default=None,
        description=(
            "Provider block to quote against (must be declared in ubt.toml or be a "
            "built-in wire protocol). Endpoint and credentials still come from it."
        ),
    )
    ocr_mode: OcrMode | None = Field(
        default=None, description="OCR engine the job will use (affects the vision line)"
    )
    visual_judge_enabled: bool | None = Field(
        default=None, description="Quote the VLM visual judge over rendered pages"
    )
    max_concurrency: int | None = Field(
        default=None, gt=0, description="Concurrent LLM requests (drives the runtime estimate)"
    )

    @field_validator("pages")
    @classmethod
    def _validate_pages(cls, value: str | None) -> str | None:
        return _require_valid_page_ranges(value)

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
    """One degraded probe: a stable machine code plus Chinese human copy.

    ``params`` carries the values interpolated into ``detail_zh`` so a
    non-Chinese console can render the warning from its own catalogue; it falls
    back to ``detail_zh`` for a code it does not know.
    """

    model_config = ConfigDict(extra="ignore")

    code: str
    level: str
    detail_zh: str
    params: dict[str, Any] = Field(default_factory=dict)


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
    target_lang: str
    status: str
    total_blocks: int
    #: Drafted but not yet finalized. A live-work counter, so the queue row can
    #: show motion during a long draft stage without claiming the job is done.
    drafted_blocks: int = 0
    completed_blocks: int
    failed_blocks: int
    needs_human_blocks: int
    progress_percent: float
    estimated_cost_usd: float | None = None
    created_at: str | None = None
    updated_at: str | None = None
    has_output: bool
    #: True when the job ran on the echo provider (``--dry-run``, or an API
    #: server with no LLM key). Its artifact is ``[模拟翻译]`` source text, so a
    #: queue row that does not say so reads exactly like a real delivery — the
    #: console listed rehearsals and real runs side by side with no distinction.
    rehearsal: bool = False
    #: The page scope of a ``--pages`` run (e.g. ``"1-6"``), else ``None``. A
    #: sample run and a full delivery were otherwise indistinguishable in the
    #: queue.
    page_scope: str | None = None
    #: Last safe terminal error persisted in the ledger. The list view uses it
    #: for a failed job after the in-memory record is gone.
    last_error: str | None = None


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
    port: int | None = Field(default=None, description="Port this request reached (client-visible)")
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
    # True when an idempotency key matched a job that already exists and
    # nothing new was started. A retried submit (double-click, a flaky network
    # retry, a page reload resending the form) must not bill a second run, and
    # the console has to be able to tell "I started this" from "this already
    # existed" — otherwise a reused id looks like a fresh start and the user
    # wonders why the progress bar is already at 60%.
    reused: bool = False


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
    # The run's page selection ("1-6"), or None for a whole-book run. The list
    # endpoint carried this but the detail endpoint did not, so a --pages
    # fragment looked identical to a full delivery once the operator opened it.
    page_scope: str | None = None


# --------------------------------------------------------------------------- #
# Response models for the workbench/asset endpoints.
#
# These routes returned bare JSONResponse dicts, so the OpenAPI schema could
# not describe them and the web client typed the payloads as Record<string,
# any> — the frontend then re-declared each field by hand and could not tell
# when the server changed one. Declaring the shapes here makes the generated
# types the contract (2026-10-14 review, W4-5).
# --------------------------------------------------------------------------- #


class SegmentItem(BaseModel):
    """One finalized block as the workbench grid shows it.

    Mirrors :func:`ubt.api.review.serialize_segment`; that function is the
    runtime owner, this is the published shape.
    """

    model_config = ConfigDict(extra="ignore")

    block_id: str
    spine_index: int
    page: int | None = None
    block_type: str
    status: str
    source_text: str
    target_text: str
    mtqe_score: float | None = None
    error_flags: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    human_verified: bool = False
    tm_hit: bool = False
    repair_rounds: int = 0
    glossary_hits: list[str] = Field(default_factory=list)
    mqm_severity: str | None = None
    mqm_spans: list[dict[str, Any]] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)


class SegmentListResponse(BaseModel):
    """A page of segments plus the filtered total."""

    model_config = ConfigDict(extra="ignore")

    job_id: str
    total: int
    segments: list[SegmentItem]


class IssueCountsResponse(BaseModel):
    """The fault ribbon: counts by issue kind and by blocking status."""

    model_config = ConfigDict(extra="ignore")

    job_id: str
    counts: dict[str, int]
    status: dict[str, int]
    total_issues: int
    term_drift: list[dict[str, Any]] = Field(default_factory=list)


class DeliverableItem(BaseModel):
    """One file a finished job actually left on disk."""

    model_config = ConfigDict(extra="ignore")

    key: str
    label: str
    filename: str
    size_bytes: int
    media_type: str


class DeliverableListResponse(BaseModel):
    """The deliverables endpoint's payload."""

    model_config = ConfigDict(extra="ignore")

    job_id: str
    deliverables: list[DeliverableItem]
    #: True when the files on disk predate the job's current ledger text, i.e.
    #: someone edited segments without re-rendering. The download endpoint
    #: refuses such a job (409); this flag is how the console knows *before* it
    #: opens a download the user would only see fail.
    stale: bool = False


class GlossaryResponse(BaseModel):
    """The shared glossary's current contents."""

    model_config = ConfigDict(extra="ignore")

    path: str
    terms: list[dict[str, str]]


class GlossaryDeleteResponse(GlossaryResponse):
    """A delete also reports whether the source was present."""

    removed: bool


class GlossaryConflictsResponse(BaseModel):
    """Sources with more than one configured target rendering."""

    model_config = ConfigDict(extra="ignore")

    conflicts: list[dict[str, Any]]


class TMEntryItem(BaseModel):
    """One translation-memory row as the console lists it."""

    model_config = ConfigDict(extra="ignore")

    id: int
    src_lang: str
    tgt_lang: str
    source_text: str
    target_text: str
    provenance: str
    domain: str
    use_count: int


class TMListResponse(BaseModel):
    """A page of the shared translation memory."""

    model_config = ConfigDict(extra="ignore")

    total: int
    entries: list[TMEntryItem]


class RuntimeActiveModel(BaseModel):
    """The provider and models the engine will actually call, right now."""

    model_config = ConfigDict(extra="ignore")

    provider: str = Field(description="Selected provider name (UBT_PROVIDER / --provider)")
    draft_model: str = Field(description="Model used for drafting; empty when unconfigured")
    repair_model: str = Field(description="Model used for repair; empty when unconfigured")
    base_url: str = Field(description="Endpoint the selected provider resolves to")
    api_mode: str = Field(description="Wire protocol (openai-chat, openai-responses, …)")
    matched_pattern: str | None = Field(
        default=None,
        description=(
            "Capability profile this model resolved to, or null when it fell through "
            "to the generic default"
        ),
    )
    prompt_strategy: str = Field(description="Prompt strategy the resolved profile selects")
    display_name: str = Field(description="Human-readable name of the resolved profile")
    declared_capability: list[str] = Field(
        default_factory=list,
        description=(
            "Capability flags the provider block states outright. Every other flag is "
            "the conservative default, not a measurement of the endpoint."
        ),
    )
    supports_temperature: bool | None = Field(
        default=None, description="Whether a temperature is sent"
    )
    supports_reasoning_effort: bool | None = Field(
        default=None, description="Whether a reasoning effort is sent"
    )
    supports_system_prompt: bool | None = Field(
        default=None, description="Whether the system turn is sent as its own message"
    )
    supports_vision: bool | None = Field(
        default=None, description="Whether a page image may be sent to this model"
    )
    extraction_strategy: str = Field(
        default="", description="How the translation is recovered from the completion"
    )
    can_translate: bool = Field(
        default=False,
        description=(
            "True when a submit would translate for real. False means the server "
            "would downgrade it to a zero-token rehearsal (no model, or no real "
            "credential) — the wizard says so before the button is pressed"
        ),
    )
    allow_page_upload: bool = Field(
        default=False,
        description=(
            "True when rendered page images may leave this machine. The visual "
            "judge and cloud/vlm OCR both require it; with it false the pipeline "
            "skips them, so the console must say so instead of offering a switch "
            "that silently does nothing"
        ),
    )


class RuntimeProvider(BaseModel):
    """One declared provider, with credentials deliberately omitted."""

    model_config = ConfigDict(extra="ignore")

    name: str
    active: bool = Field(description="True for the provider currently selected")
    is_default: bool = Field(
        default=False,
        description=(
            "True when [defaults] provider names this block. Distinct from "
            "``active``: UBT_PROVIDER / --provider outrank the file, so the panel "
            "must be able to show what the file says separately from what runs"
        ),
    )
    available: bool = Field(description="False when the block could not be loaded")
    declared: bool = Field(
        default=False, description="True when the operator declared it in [providers.*]"
    )
    is_alias: bool = Field(
        default=False, description="True for a bare protocol alias (chat, claude, responses)"
    )
    alias_of: str | None = Field(
        default=None,
        description=(
            "The canonical wire protocol an alias stands for (``claude`` -> "
            "``anthropic-messages``), or null for a protocol or a declared block"
        ),
    )
    is_protocol: bool = Field(
        default=False,
        description=(
            "True for the four canonical wire protocols (openai-chat, …). They are "
            "the names a [providers.*] block should carry; an alias is a shortcut "
            "to one of them"
        ),
    )
    base_url: str | None = None
    api_mode: str | None = None
    draft_model: str | None = None
    repair_model: str | None = None
    is_free: bool = Field(default=False, description="Declared free (no billable spend)")
    credential_issue: str | None = Field(
        default=None,
        description=(
            "Why this provider cannot authenticate, as a stable code "
            "('unresolved_variable' or 'missing'), or null when it can. The "
            "variable name is not included: it belongs on the single-provider "
            "endpoint the operator opens to fix it"
        ),
    )
    supports_temperature: bool | None = None
    supports_reasoning_effort: bool | None = None
    supports_system_prompt: bool | None = None
    supports_vision: bool | None = None
    extraction_strategy: str | None = None
    deployment_backend: str | None = None


class RuntimeConfigResponse(BaseModel):
    """Read-only model/provider configuration actually in force.

    Credentials are never included: provider blocks are filtered through an
    allow-list before they reach this model.
    """

    model_config = ConfigDict(extra="ignore")

    active: RuntimeActiveModel
    providers: list[RuntimeProvider]
    provider_config_source: str = Field(
        description="Where provider settings live (ubt.toml or the UBT_* environment)"
    )
    credential_source: str = Field(
        default="",
        description="Where the credential comes from when the console may not write it",
    )
    can_write_credential: bool = Field(
        default=False,
        description=(
            "True on a loopback, non-strict deployment. False means the console "
            "must not offer a credential field at all, and says why"
        ),
    )
    frozen_fields_changed: list[str] = Field(
        default_factory=list,
        description=(
            "Deployment-bound fields edited in the file that need a restart. They "
            "are decided once at boot, so reporting them is the difference between "
            "a control that works and one that silently does nothing"
        ),
    )


class ProviderEditRequest(ProviderEditModel):
    """One provider block edit from the console.

    ``extra="forbid"`` so a client that sends a credential gets a 422 naming the
    field instead of a silently ignored key. Every field is optional: an edit
    submits only what changed, and an omitted field is left untouched in the file.
    """


class ProviderDetailResponse(BaseModel):
    """One provider block as the console may see it.

    ``has_credential`` / ``credential_in_file`` are booleans on purpose: whether a
    key is configured decides whether the block can run at all, and a boolean is
    the most that may be revealed about it.

    **"Configured" means "resolves", not "is written".** A block may carry
    ``api_key = "${OPENCODE_API_KEY}"`` while that variable is unset; the loader
    then hands the router an empty string and every call fails, so a form that
    reported "credential set" for it promised an authentication the next job could
    not perform. ``credential_issue`` names the two failure shapes by code, and
    ``unresolved_vars`` says which exports are missing — names only, never values.
    """

    model_config = ConfigDict(extra="ignore")

    name: str
    declared: bool = Field(description="True when the operator declared it in [providers.*]")
    has_credential: bool = Field(
        default=False,
        description=(
            "Whether this block's own credential resolves to a non-empty value. "
            "False for an unset ${VAR}: present in the file, unusable on the wire"
        ),
    )
    authenticates: bool = Field(
        default=False,
        description=(
            "Whether a call from this block can authenticate at all, counting "
            "UBT_LLM_API_KEY. Distinct from has_credential: a keyless block on a "
            "deployment that exports the generic variable does run"
        ),
    )
    credential_issue: str | None = Field(
        default=None,
        description=(
            "Stable code for why the credential is unusable: 'unresolved_variable' "
            "when a ${VAR} reference has no value, 'missing' when nothing supplies "
            "one, null when it resolves"
        ),
    )
    unresolved_vars: list[str] = Field(
        default_factory=list,
        description="Names of ${VAR} references in the block that resolve to nothing",
    )
    credential_shadowed_by_env: bool = Field(
        default=False,
        description=(
            "True when this block holds a usable credential that UBT_LLM_API_KEY "
            "nevertheless outranks. A save is then inert until that variable is "
            "unset, and the form must say so rather than promise the next job"
        ),
    )
    credential_hint: str | None = Field(
        default=None,
        description=(
            "A fixed mask when a credential is configured, never the value: the "
            "console needs to render 'already set', not to read the secret back"
        ),
    )
    credential_in_file: bool = Field(
        default=False, description="True when the block itself carries the credential"
    )
    base_url: str | None = None
    api_mode: str | None = None
    draft_model: str | None = None
    repair_model: str | None = None
    draft_reasoning_effort: str | None = None
    repair_reasoning_effort: str | None = None
    reasoning_dialect: str | None = None
    supports_temperature: bool | None = None
    supports_reasoning_effort: bool | None = None
    supports_system_prompt: bool | None = None
    supports_vision: bool | None = None
    extraction_strategy: str | None = None
    deployment_backend: str | None = None
    is_free: bool | None = None
    api_timeout: float | None = None
    fallback_models: list[str] | None = None
