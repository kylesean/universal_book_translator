"""Pydantic models for the UBT REST API."""

from datetime import datetime

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
    PromptStrategyName,
    QeEngine,
    RenderEngine,
)
from ubt.core.engine.progress import ProgressSnapshot
from ubt.core.job_options import LANG_CODE_PATTERN, PROFILE_NAME_PATTERN
from ubt.core.language_profile import is_supported_lang
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
            "zh, en, ja, ko, fr, de, es, ru (region tags such as 'zh-CN' are accepted)."
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
    render_engine: RenderEngine | None = Field(
        default=None,
        description="PDF render engine (rigid, reflow, publication, auto)",
    )
    dual_mode: DualMode | None = Field(
        default=None,
        description="Bilingual mode (inline, alternating, facing, monolingual, auto)",
    )
    pages: str | None = Field(default=None, description="Page range filter (e.g. 1-10)")
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
        default=None, description="Math typesetting backend (typst, mathjax, image)"
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

    @field_validator("target_lang")
    @classmethod
    def _validate_target_lang(cls, value: str) -> str:
        return _require_supported_target_lang(value)


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
