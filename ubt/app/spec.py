"""Canonical JobSpec shared across CLI, REST API, and MCP surfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from ubt.core.config import (
    DeliveryMode,
    DualMode,
    OcrMode,
    PdfEngine,
    parse_page_ranges,
)
from ubt.core.job_options import LANG_CODE_PATTERN, PROFILE_NAME_PATTERN
from ubt.core.language_profile import is_supported_lang, supported_lang_codes
from ubt.core.presets import Preset


def _require_supported_target_lang(value: str) -> str:
    if not is_supported_lang(value):
        raise ValueError(
            f"Unsupported target language {value!r}. Supported base languages: "
            f"{', '.join(supported_lang_codes())} (region tags such as 'zh-CN' are accepted)."
        )
    return value


def _require_valid_page_ranges(value: str | None) -> str | None:
    if value is None:
        return value
    try:
        parse_page_ranges(value)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    return value


class JobSpec(BaseModel):
    """Single submission model shared by CLI, REST API and MCP.

    ``input_path`` is the public API field; ``source_path`` is accepted as a
    validation alias for the codec service and for callers that speak in file
    paths. The delivery contract (``delivery_mode``) and the codec layout
    (``layout_mode``/``dual_mode``) are deliberately distinct.
    """

    input_path: str = Field(
        ...,
        validation_alias=AliasChoices("input_path", "source_path"),
        description="Path to source document file",
    )
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
        default="auto",
        description="Domain profile (auto, general, textbook, paper, etc.)",
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
    repair_provider: str | None = Field(
        default=None,
        description=(
            "Provider block for the repair tier (defaults to the primary provider). "
            "A name, validated like `provider`: the named block supplies the repair "
            "endpoint and credential."
        ),
    )
    #: Export-time fail-closed gate for NEEDS_HUMAN blocks (queued for human
    #: post-editing). Default follows config / ``UBT_EXPORT_ALLOW_NEEDS_HUMAN``;
    #: an explicit true ships them knowingly, with the block ids listed in the
    #: quality report's ``summary.needs_human_block_ids``.
    export_allow_needs_human: bool | None = Field(
        default=None,
        description="Allow export to ship NEEDS_HUMAN blocks unverified (default: refuse)",
    )
    draft_model: str | None = Field(default=None, description="Override draft model tier")
    repair_model: str | None = Field(default=None, description="Override repair model tier")
    preset: Preset | None = Field(
        default=None,
        description="Quality preset (high_fidelity, standard, vector_math, preview, fast)",
    )
    pdf_engine: PdfEngine | None = Field(
        default=None,
        description="PDF extraction/parser engine (docling, pdfium, auto)",
    )
    dual_mode: DualMode | None = Field(
        default=None,
        description="What the reader receives (dual, mono, auto)",
    )
    delivery_mode: DeliveryMode | None = Field(
        default=None,
        description=("PDF delivery contract: auto, readable or faithful"),
    )
    allow_degrade: bool | None = Field(
        default=None,
        description=(
            "When a new faithful request is ineligible, re-route to readable "
            "instead of refusing the run (legacy modes are grandfathered)"
        ),
    )
    pages: str | None = Field(default=None, description="Page range filter (e.g. 1-10)")
    allow_overflow_compaction: bool = Field(
        default=True, description="Let the codec compact overflowing units once"
    )
    max_overflow_compaction_units: int = Field(
        default=32, ge=0, le=200, description="Upper bound on compaction units per run"
    )

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
    # OCR engine selection. ``ocr_endpoint`` / ``ocr_api_key`` stay operator-only.
    ocr_mode: OcrMode | None = Field(
        default=None, description="OCR engine (auto, local, cloud, off)"
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
    # Set by the intake when IT forced dry_run (no usable credential), never by
    # the client. Marks the dry_run as an auto-decision: a resume after the
    # operator fixed the key then runs for real, exactly as the queue worker
    # reads the same marker. A client-requested dry_run stays a rehearsal.
    rehearsal_auto: bool = Field(
        default=False,
        description="True when the server (not the client) forced dry_run; resume then runs for real",
    )
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
    #: Codec-level layout for the surgical service (mono/interleaved/side_by_side/
    #: bilingual/reflow). When omitted it is derived from ``dual_mode``.
    layout_mode: Any = Field(default=None, description="Codec layout override")

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

    @property
    def source_path(self) -> Path:
        return Path(self.input_path)

    @property
    def page_set(self) -> set[int] | None:
        if not self.pages:
            return None
        from ubt.core.config import parse_page_ranges

        parsed = parse_page_ranges(self.pages)
        return {p for p in parsed if p >= 1} if parsed else None

    @property
    def codec_mode(self) -> Any:
        if self.layout_mode is not None:
            return self.layout_mode
        from ubt.codecs.protocol import DeliveryMode as CodecDeliveryMode

        return {
            "mono": CodecDeliveryMode.MONO,
            "dual": CodecDeliveryMode.BILINGUAL,
            "auto": CodecDeliveryMode.MONO,
        }.get(str(self.dual_mode or "mono"), CodecDeliveryMode.MONO)

    def resolved_output_path(self) -> Path:
        if self.output_path:
            return Path(self.output_path)
        stem = self.source_path.stem
        suffix = self.source_path.suffix or ".pdf"
        mode = self.codec_mode.value
        pages = self.page_set
        page_tag = ""
        if pages:
            ordered = sorted(pages)
            page_tag = f"_p{ordered[0]}-{ordered[-1]}" if len(ordered) > 1 else f"_p{ordered[0]}"
        return self.source_path.with_name(f"{stem}_{self.target_lang}_{mode}{page_tag}{suffix}")

    @field_validator("input_path", mode="before")
    @classmethod
    def _path_to_string(cls, value: Any) -> Any:
        return str(value)

    @field_validator("pages", mode="before")
    @classmethod
    def _normalise_pages(cls, value: Any) -> Any:
        if isinstance(value, list):
            return ",".join(str(v) for v in value)
        return value

    @field_validator("job_id")
    @classmethod
    def _validate_job_id(cls, value: str | None) -> str | None:
        from ubt.core.job_options import job_id_is_valid

        if value is not None and not job_id_is_valid(value):
            raise ValueError(f"Invalid or reserved job_id: {value!r}")
        return value
