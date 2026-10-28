"""Canonical UBT configuration (single-key design).

Design contract:
- ONE canonical env var per field: ``UBT_<FIELD_NAME>`` (upper-snake of the
  field name), resolved declaratively by ``pydantic-settings``. A few fields
  also accept a documented legacy alias via ``AliasChoices`` (``UBT_TIMEOUT``,
  ``UBT_PAGE_RANGE``); there is no second *role* name. The role-based names
  (``draft_*`` / ``repair_*``, matching :class:`ModelRouter` parameters and CLI
  flags) are the only names.
- Distinct credentials never share an env name: ``UBT_LLM_API_KEY`` is the
  outbound LLM key, ``UBT_API_KEY`` the inbound HTTP gate, and
  ``UBT_OCR_API_KEY`` the OCR/vision key. Sharing a name across these three
  silently couples the inbound gate to the outbound provider secret (a security
  defect — see ``_check_invariants``).
- Third-party vendor names (``OPENAI_API_KEY``, ``ANTHROPIC_BASE_URL``, ...) are
  NOT field aliases. Credentials come directly from the provider block's ``api_key``
  (with optional ``${VAR}`` expansion) or the generic ``UBT_LLM_API_KEY`` override.
  The settings source accepts only ``UBT_*`` names and pins ``api_key`` /
  ``service_api_key`` to their one declared variable, so a bare or third-party
  name in the process environment can never hijack the outbound credential, the
  endpoint, or the inbound gate.
- No credential is ever scraped from another program: an outbound key resolves
  only from ``UBT_LLM_API_KEY``, a configured provider's ``api_key``, or the
  ``"mock-key"`` placeholder for dry-run / tests. Explicit env always wins — the
  settings layer resolves env before the factory ever runs, so factories never
  peek at other fields' env vars.
- Closed value sets are :data:`Literal` types (fail fast with a clear error
  instead of silently accepting typos); cross-field invariants live in one
  ``model_validator``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal
from urllib.parse import urlsplit

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

if TYPE_CHECKING:
    # Import-time cost and import direction: ``config`` is imported by everything,
    # and the router package must not be pulled in just to type a return value.
    from ubt.core.router.capabilities import ModelProfile

#: The four LLM wire protocols UBT speaks. A provider is just an endpoint that
#: speaks one of them — vendor identity never selects the protocol.
ApiMode = Literal["openai-chat", "openai-responses", "anthropic-messages", "gemini-native"]
# Deliberately open, unlike its Literal neighbours: the valid set is
# _PDF_ENGINE_REGISTRY (plus "auto"), which in-process registrations
# extend at runtime. Enumerating it here would be a second source of truth to drift.
# Membership is enforced where the truth lives, by
# ubt.adapters.factory._resolve_pdf_adapter.
PdfEngine = str
# "heuristic" emits a discrete defect-class proxy (12 fixed bands),
# NOT a calibrated quality score — see QE_DEFECT_CLASS_LEGEND in
# ubt/core/qe/heuristic.py. A calibrated neural score (CometKiwi) was retired
# for commercial-license compliance; the tiered mix adds an opt-in LLM judge.
QeEngine = Literal["heuristic", "tiered"]
#: What the reader receives: target only, or source beside target.
#: ``dual`` is a single layout — each block's source line above its translation —
#: because the renderer rebuilds the page and a page-zip against the
#: source is impossible (it changes the page count). ``auto`` lets the advisor
#: pick from the document's structure.
DualMode = Literal["mono", "dual", "auto"]
#: Document delivery channel: 'reflow' (responsive Typst reflow), 'facsimile' (1:1 visual canvas isomorphism), or 'page_locked' (1:1 style-cloned reflow).
DeliveryMode = Literal["reflow", "facsimile", "page_locked"]

#: What the export stage does when a block's text cannot be placed (a layout
#: spill). ``error`` fails the delivery contract; the other three downgrade it
#: to a warning and write the overflow companion. ``source_kept_warn`` is the
#: same behaviour as ``warn`` under a name that says what the warning is
#: about (a block shipped its source); it exists because the export stage's
#: contract check distinguishes the two reasons in its own vocabulary.
SpillPolicy = Literal["error", "warn", "appendix", "source_kept_warn"]

#: Accelerator device Docling uses for its layout/table/formula models.
#: ``auto`` (default) keeps Docling's own probe: CUDA when a GPU is present.
#: ``cpu`` keeps Docling off the GPU entirely — the escape hatch for a local
#: LLM sharing one card, where Docling's CUDA footprint (measured on arXiv
#: 2610.02185: ~388 MiB allocated, ~1.4 GiB reserved by the caching allocator,
#: ~1.6 GiB as nvidia-smi reports the process) otherwise makes the LLM fail to
#: load with cudaMalloc OOM. ``cuda`` forces the GPU. Note the footprint is
#: only paid on a cache MISS: Docling loads its models lazily in
#: ``convert()``, so a cached conversion uses zero GPU.
DoclingDevice = Literal["auto", "cpu", "cuda"]


def resolve_repair_model(
    draft_model: str, repair_model: str, *, repair_is_independent: bool
) -> str:
    """Return the repair model, following the draft unless repair was chosen alone.

    "Alone" means configured independently of the draft: an explicit repair
    value, a provider block whose repair differs from its draft, or a base
    config whose repair already differs from its draft. When repair only ever
    inherited the draft, moving the draft must move repair with it. This is the
    single owner of the draft->repair default; the env constructor, the request
    override path and the model validator all call it.
    """
    return repair_model if repair_is_independent else draft_model


PromptStrategyName = Literal["auto", "minimal", "hybrid", "rich"]
#: How to recover the translation from a raw completion. ``auto`` (the default)
#: tries the XML tag first, then strips code fences and conversational lead-ins.
ExtractionStrategyName = Literal["auto", "xml_tag", "raw", "cp"]
OcrMode = Literal["auto", "local", "vlm", "off"]
ExecMode = Literal["auto", "short", "long"]
DocumentMode = Literal["auto", "paper", "book"]
FormulaEnrichment = Literal["auto", "on", "off"]
TocMode = Literal["auto", "on", "off"]
#: Table handling for PDF. ``auto`` translates every table (complex ones also
#: get a translated appendix companion); ``appendix`` always writes the
#: companion; ``skip`` never translates tables. ``overlay`` is not a value:
#: every table is rebuilt from the IR, so a kept table is simply absent from the
#: document. A stale ``UBT_TABLE_RENDER_MODE=overlay`` is rejected rather than
#: silently reinterpreted.
TableRenderMode = Literal["auto", "appendix", "skip"]


#: Valid provider protocol/strategy vocabularies
VALID_API_MODES = frozenset(
    {"openai-chat", "openai-responses", "anthropic-messages", "gemini-native"}
)
VALID_EXTRACTION_STRATEGIES = frozenset({"auto", "xml_tag", "raw", "cp"})
VALID_REASONING_DIALECTS = frozenset({"nested", "flat"})


class ProviderConfig(BaseModel):
    """Strongly typed model for a provider configuration block or [defaults] block."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
    )

    api_key: str | SecretStr | None = None
    base_url: str | None = None
    api_mode: str | None = None
    draft_model: str | None = None
    repair_model: str | None = None
    draft_reasoning_effort: str | None = None
    repair_reasoning_effort: str | None = None
    fallback_models: list[str] | None = None
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)
    extra_headers: dict[str, str] = Field(default_factory=dict)
    api_timeout: float | None = Field(default=None, gt=0)
    prompt_caching_enabled: bool | None = None
    is_free: bool | None = None
    cost_per_mtok: tuple[float, float] | None = None
    supports_temperature: bool | None = None
    supports_reasoning_effort: bool | None = None
    supports_system_prompt: bool | None = None
    supports_vision: bool | None = None
    extraction_strategy: str | None = None
    deployment_backend: str | None = None
    reasoning_dialect: str | None = None
    repair_provider: str | None = None
    qe_judge_provider: str | None = None
    ocr_api_key: str | SecretStr | None = None
    db_dir: str | Path | None = None
    provider: str | None = None
    qe_engine: str | None = None
    qe_judge_enabled: bool | None = None
    qe_judge_pass_sample: float | None = Field(default=None, ge=0.0, le=1.0)
    qe_judge_sentences: bool | None = None
    qe_judge_model: str | None = None
    qe_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    max_repair_rounds: int | None = Field(default=None, ge=0)
    consistency_enforce: Literal["off", "report", "repair"] | None = None
    consistency_max_repairs: int | None = Field(default=None, ge=0)
    #: Shared glossary file. This is global deployment state, not a provider
    #: capability: ``[defaults]`` may set it, while provider blocks may not.
    glossary_path: Path | None = None

    @field_validator(
        "supports_temperature",
        "supports_reasoning_effort",
        "supports_system_prompt",
        "supports_vision",
        "is_free",
        "prompt_caching_enabled",
        "qe_judge_enabled",
        "qe_judge_sentences",
        mode="before",
    )
    @classmethod
    def _validate_bool_fields(cls, v: Any, info: Any) -> bool | None:
        if v is None:
            return None
        if not isinstance(v, bool):
            raise ValueError(f"{info.field_name} must be a boolean")
        return v

    @field_validator("api_timeout", mode="before")
    @classmethod
    def _validate_api_timeout(cls, v: Any) -> float | None:
        if v is None:
            return None
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0:
            raise ValueError("api_timeout must be a positive number of seconds")
        return float(v)

    @field_validator("fallback_models", mode="before")
    @classmethod
    def _validate_fallback_models(cls, v: Any) -> list[str] | None:
        if v is None:
            return None
        if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
            raise ValueError("fallback_models must be a list of model ids")
        return [x for x in v if x.strip()]

    @field_validator(
        "api_key",
        "ocr_api_key",
        "base_url",
        "api_mode",
        "draft_model",
        "repair_model",
        "draft_reasoning_effort",
        "repair_reasoning_effort",
        "reasoning_dialect",
        "extraction_strategy",
        "deployment_backend",
        "repair_provider",
        "qe_judge_provider",
        "provider",
        "qe_judge_model",
        mode="before",
    )
    @classmethod
    def _validate_string_fields(cls, v: Any, info: Any) -> Any:
        if v is None or isinstance(v, SecretStr):
            return v
        if not isinstance(v, str):
            raise ValueError(f"{info.field_name} must be a string")
        cleaned = v.strip()
        field = info.field_name
        if field == "api_mode":
            lowered = cleaned.lower()
            if lowered not in VALID_API_MODES:
                raise ValueError(f"api_mode must be one of {', '.join(sorted(VALID_API_MODES))}")
            return lowered
        if field == "extraction_strategy":
            lowered = cleaned.lower()
            if lowered not in VALID_EXTRACTION_STRATEGIES:
                raise ValueError(
                    f"extraction_strategy must be one of {', '.join(sorted(VALID_EXTRACTION_STRATEGIES))}"
                )
            return lowered
        if field == "reasoning_dialect":
            lowered = cleaned.lower()
            if lowered not in VALID_REASONING_DIALECTS:
                raise ValueError(
                    f"reasoning_dialect must be one of {', '.join(sorted(VALID_REASONING_DIALECTS))}"
                )
            return lowered
        if field == "base_url" and cleaned and not cleaned.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        return cleaned

    @field_validator("qe_engine", mode="before")
    @classmethod
    def _validate_qe_engine(cls, v: Any) -> str | None:
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError("qe_engine must be a string")
        cleaned = v.strip().lower()
        legacy = {
            "comet": "heuristic",
            "cometkiwi": "heuristic",
            "neural": "heuristic",
            "subprocess": "heuristic",
        }
        if cleaned in legacy:
            return legacy[cleaned]
        if cleaned not in {"heuristic", "tiered"}:
            raise ValueError("qe_engine must be one of heuristic, tiered")
        return cleaned


DefaultsConfig = ProviderConfig


class UBTConfigFile(BaseModel):
    """Strongly typed model representing the entire ubt.toml configuration document."""

    model_config = ConfigDict(extra="ignore")

    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)


#: Default endpoint when no provider is selected and no UBT_BASE_URL is set.
#: The provider registry (``ubt/core/providers.py``) carries the vendor-specific
#: endpoints; this one keeps a bare ``UBTConfig()`` usable against the OpenAI API.
_OPENAI_BASE_URL = "https://api.openai.com/v1"

# Placeholder used when no real credential is configured (dry-run / tests).
MOCK_API_KEY = "mock-key"

logger = logging.getLogger(__name__)


def _default_api_key() -> SecretStr:
    """``mock-key`` stands in when no credential is configured.

    This factory never reaches into other programs' credential stores — an
    opencode ``auth.json``, ``~/.config/deepseek_key``, or the like — because
    doing so would translate a whole book through a third-party account the
    user never selected and never saw named (``doctor`` prints only "API key
    OK", correctly refusing to echo a key fragment). A credential comes only
    from ``UBT_LLM_API_KEY`` or a configured provider's ``api_key``; see the
    ``api_key`` field.
    """
    return SecretStr(MOCK_API_KEY)


_MAX_PAGE_RANGE = 100_000
#: Cap on the raw ``pages`` specification string itself, before it is split.
_MAX_PAGES_SPEC_LEN = 10_000


def parse_page_ranges(pages_str: str | None) -> set[int] | None:
    """Parse page specification strings like '1-3,5,7-9' into a set of 1-based page numbers.

    Returns None if pages_str is None or empty.
    Raises ValueError on invalid syntax or non-positive integers.
    """
    if not pages_str or not pages_str.strip():
        return None
    # Bound the raw string before splitting it: a multi-megabyte ``pages`` field
    # (or UBT_PAGES) otherwise materializes a millions-long list first.
    if len(pages_str) > _MAX_PAGES_SPEC_LEN:
        raise ValueError(
            f"Page specification too long ({len(pages_str)} chars); limit is {_MAX_PAGES_SPEC_LEN}"
        )
    pages: set[int] = set()
    parts = [p.strip() for p in pages_str.split(",") if p.strip()]
    for part in parts:
        if "-" in part:
            bounds = [b.strip() for b in part.split("-", 1)]
            if not bounds[0].isdigit() or not bounds[1].isdigit():
                raise ValueError(f"Invalid page range specification: '{part}'")
            start, end = int(bounds[0]), int(bounds[1])
            if start < 1 or end < 1:
                raise ValueError(f"Page numbers must be >= 1, got '{part}'")
            if start > end:
                raise ValueError(f"Invalid page range: start ({start}) > end ({end}) in '{part}'")
            # Bound the materialized set: a config like UBT_PAGES=1-999999999
            # would otherwise allocate ~1e9 ints.
            if end - start + 1 > _MAX_PAGE_RANGE:
                raise ValueError(
                    f"Page range too large ({end - start + 1} pages); limit is {_MAX_PAGE_RANGE}"
                )
            pages.update(range(start, end + 1))
        else:
            if not part.isdigit():
                raise ValueError(f"Invalid page number specification: '{part}'")
            val = int(part)
            if val < 1:
                raise ValueError(f"Page numbers must be >= 1, got '{val}'")
            pages.add(val)
    # Cap the whole materialized set, not just one contiguous span: a comma list
    # of millions of distinct integers would otherwise allocate them all.
    if len(pages) > _MAX_PAGE_RANGE:
        raise ValueError(f"Too many pages requested ({len(pages)}); limit is {_MAX_PAGE_RANGE}")
    return pages


class UBTEnvSettingsSource(EnvSettingsSource):
    """Environment source that honours only UBT-owned names.

    ``populate_by_name=True`` makes pydantic-settings also accept
    ``UBT_<FIELD_NAME>`` for every field, which collides for the credentials:
    ``api_key`` would read ``UBT_API_KEY`` (the inbound gate's name) and
    ``service_api_key`` would read the retired ``UBT_SERVICE_API_KEY``. Those two
    resolve only from their declared alias. Every other field is limited to
    ``UBT_*``, so a bare or third-party name in the process environment can never
    hijack the outbound credential, the endpoint, the inbound gate, or the page
    filter.
    """

    #: Credential fields whose prefix-generated name collides with another name,
    #: pinned to their one legitimate variable.
    _ALIAS_ONLY: ClassVar[dict[str, str]] = {
        "api_key": "UBT_LLM_API_KEY",
        "service_api_key": "UBT_API_KEY",
    }

    def _extract_field_info(self, field: FieldInfo, field_name: str) -> list[tuple[str, str, bool]]:
        info = super()._extract_field_info(field, field_name)
        pinned = self._ALIAS_ONLY.get(field_name)
        if pinned is not None:
            return [(k, env, is_c) for k, env, is_c in info if env.upper() == pinned]
        return [(k, env, is_c) for k, env, is_c in info if env.upper().startswith("UBT_")]


class RouterConfig(BaseModel):
    """Sub-domain configuration for LLM routing, endpoints, rate limits, and model capabilities."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    api_key: SecretStr = Field(default_factory=_default_api_key)
    base_url: str = _OPENAI_BASE_URL
    api_mode: ApiMode = "openai-chat"
    provider: str | None = None
    repair_provider: str | None = None
    qe_judge_provider: str | None = None
    is_free: bool = False
    cost_per_mtok: tuple[float, float] | None = None
    supports_temperature: bool | None = None
    supports_reasoning_effort: bool | None = None
    supports_system_prompt: bool | None = None
    supports_vision: bool | None = None
    extraction_strategy: ExtractionStrategyName = "auto"
    deployment_backend: str = ""
    reasoning_dialect: str = "nested"
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)
    extra_headers: dict[str, str] = Field(default_factory=dict)
    api_timeout: float = Field(default=180.0, gt=0)
    draft_model: str = ""
    repair_model: str = ""
    fallback_models: list[str] = Field(default_factory=list)
    draft_reasoning_effort: str = "low"
    repair_reasoning_effort: str = "high"
    prompt_strategy: PromptStrategyName = "auto"
    rate_limit_rpm: int = Field(default=60, gt=0)
    rate_limit_tpm: int = Field(default=100_000, gt=0)
    rate_limit_max_rpm: int = Field(default=240, gt=0)
    rate_limit_max_tpm: int = Field(default=600_000, gt=0)
    rate_limit_backoff_cooldown_sec: float = Field(default=3.0, ge=0.0)
    draft_max_retries: int = Field(default=0, ge=0)
    draft_retry_base_delay: float = Field(default=1.0, ge=0.0)
    prompt_caching_enabled: bool = True


class TypstConfig(BaseModel):
    """Sub-domain configuration for document rendering, Typst typesetting, and page layouts."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    dual_mode: DualMode = "auto"
    layout_columns: int | None = None
    emit_both: bool = False
    emit_xliff_companion: bool = True
    emit_attestation_shadow: bool = True
    emit_html_companion: bool = True
    emit_epub_companion: bool = True
    strict_contract: bool = False
    spill_policy: SpillPolicy = "error"
    table_render_mode: TableRenderMode = "auto"
    font_family: str | None = None
    translate_chrome: bool = False
    toc_mode: TocMode = "auto"
    enable_layout_auto_heal: bool = True
    layout_auto_heal_max_passes: int = 2
    delivery_mode: DeliveryMode = "reflow"
    enable_pdf_optimization: bool = True
    pdf_optimize_profile: Literal["lossless", "ebook", "screen"] = "lossless"


class QualityConfig(BaseModel):
    """Sub-domain configuration for quality gates, MTQE, defect screening, and consistency."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    qe_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    max_repair_rounds: int = Field(default=2, ge=0)
    bottom_percentile: float = Field(default=0.15, ge=0.0, le=1.0)
    consistency_enforce: Literal["off", "report", "repair"] = "off"
    consistency_max_repairs: int = Field(default=50, ge=0)
    qe_engine: QeEngine = "heuristic"
    qe_judge_enabled: bool = False
    qe_judge_model: str | None = None
    qe_judge_pass_sample: float = Field(default=0.0, ge=0.0, le=1.0)
    qe_judge_base_url: str | None = None
    qe_judge_api_key: SecretStr | None = None
    qe_judge_api_mode: ApiMode | None = None
    qe_judge_sentences: bool = True
    visual_gate_enabled: bool = True
    render_preflight_enabled: bool = True
    visual_judge_enabled: bool = False
    visual_judge_model: str | None = None
    visual_sample_pages: int = Field(default=6, ge=0, le=50)
    visual_max_vlm_pages: int = Field(default=3, ge=0, le=10)
    visual_blocking_gate_enabled: bool = False
    deterministic_glossary_long_route: bool = False
    export_min_completion_ratio: float = Field(default=0.5, ge=0.0, le=1.0)


class EngineConfig(BaseModel):
    """Sub-domain configuration for pipeline execution, storage, databases, and worker queues."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    service_api_key: SecretStr = Field(default_factory=lambda: SecretStr(""))
    trusted_hosts: str = ""
    trust_tenant_header: bool = False
    strict_auth: bool = False
    env: str = "development"
    allowed_dirs: str = ""
    pdf_engine: PdfEngine = "auto"
    docling_device: DoclingDevice = "auto"
    docling_isolate_process: bool = True
    ocr_mode: OcrMode = "auto"
    ocr_endpoint: str = ""
    ocr_model: str = ""
    ocr_api_key: SecretStr = Field(default_factory=lambda: SecretStr(""))
    db_dir: Path = Field(default_factory=lambda: Path(".ubt/ledgers"))
    cache_enabled: bool = True
    cache_dir: Path = Field(default_factory=lambda: Path(".ubt/cache"))
    job_mode: Literal["embedded", "queue"] = "embedded"
    job_queue_path: Path | None = None
    job_max_running: int = Field(default=8, gt=0)
    job_tenant_max_running: int = Field(default=4, gt=0)
    job_max_queued: int = Field(default=1000, gt=0)
    upload_retention_days: float = Field(default=7.0, ge=0.0)
    budget_usd: float | None = Field(default=None, gt=0)
    allow_unpriced_budget: bool = False
    local_endpoints: str = ""
    prices_file: Path | None = None
    allow_page_upload: bool = False
    vlm_trust_remote_code: bool = False
    max_concurrency: int = Field(default=10, gt=0)
    batch_limit: int = Field(default=30, gt=0)
    macro_chunk_size: int = Field(default=8, ge=1, le=30)
    ledger_flush_interval: float = Field(default=0.25, gt=0.0)
    ledger_flush_batch_size: int = Field(default=50, gt=0)
    chapter_streaming_enabled: bool = False
    chapter_streaming_queue_size: int = Field(default=2, gt=0)
    step_chars: int = Field(default=3500, gt=0)
    enable_rolling_summary: bool = True
    memory_sliding_window: bool = True
    memory_max_epochs: int = Field(default=50, gt=0)
    glossary_path: Path | None = None
    glossary_max_global_entries: int = Field(default=100, ge=0)
    domain: str | None = None
    document_mode: DocumentMode = "auto"
    exec_mode: ExecMode = "auto"
    short_max_pages: int = Field(default=30, gt=0)
    formula_enrichment: FormulaEnrichment = "auto"
    formula_crop_margin: float = Field(default=0.08, ge=0.0, le=0.5)
    pages: str | None = None
    fresh: bool = False
    c_text_enabled: bool = False
    tm_enabled: bool = True
    tm_fuzzy_threshold: float = Field(default=0.85, ge=0.0, le=1.0)


class UBTConfig(BaseSettings):
    """Configuration settings governing translation pipeline, models, and thresholds."""

    model_config = SettingsConfigDict(
        env_prefix="UBT_",
        # The process environment is the only external source: there is no
        # dotenv file. Provider selection and its settings live in the TOML
        # provider registry, and credentials come from the environment
        # (``UBT_LLM_API_KEY``), provider direct ``api_key``, or CLI flags.
        # Precedence is explicit init > real environment > field default.
        env_ignore_empty=True,
        extra="ignore",
        populate_by_name=True,
        validate_assignment=True,
    )

    def __init__(
        self,
        *args: Any,
        router: RouterConfig | None = None,
        typst: TypstConfig | None = None,
        quality: QualityConfig | None = None,
        engine: EngineConfig | None = None,
        **kwargs: Any,
    ) -> None:
        for sub_cfg in (router, typst, quality, engine):
            if sub_cfg is not None:
                for k in type(sub_cfg).model_fields:
                    kwargs.setdefault(k, getattr(sub_cfg, k))
        super().__init__(*args, **kwargs)

    @property
    def router(self) -> RouterConfig:
        """Strongly typed sub-domain configuration for LLM routing and providers."""
        return RouterConfig(
            **{k: getattr(self, k) for k in RouterConfig.model_fields if hasattr(self, k)}
        )

    @property
    def typst(self) -> TypstConfig:
        """Strongly typed sub-domain configuration for PDF rendering and typesetter layout."""
        return TypstConfig(
            **{k: getattr(self, k) for k in TypstConfig.model_fields if hasattr(self, k)}
        )

    @property
    def quality(self) -> QualityConfig:
        """Strongly typed sub-domain configuration for quality evaluation and repair gates."""
        return QualityConfig(
            **{k: getattr(self, k) for k in QualityConfig.model_fields if hasattr(self, k)}
        )

    @property
    def engine(self) -> EngineConfig:
        """Strongly typed sub-domain configuration for core engine execution and storage."""
        return EngineConfig(
            **{k: getattr(self, k) for k in EngineConfig.model_fields if hasattr(self, k)}
        )

    def as_subdomains(self) -> tuple[RouterConfig, TypstConfig, QualityConfig, EngineConfig]:
        """Unpack this configuration into its 4 decoupled typed domain sub-configs."""
        return self.router, self.typst, self.quality, self.engine

    @classmethod
    def from_subdomains(
        cls,
        *,
        router: RouterConfig | None = None,
        typst: TypstConfig | None = None,
        quality: QualityConfig | None = None,
        engine: EngineConfig | None = None,
        **extra_overrides: Any,
    ) -> UBTConfig:
        """Compose a unified UBTConfig directly from typed sub-domain configurations."""
        return cls(
            router=router,
            typst=typst,
            quality=quality,
            engine=engine,
            **extra_overrides,
        )

    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> UBTConfig:
        """Create a copy of UBTConfig, supporting both top-level and sub-domain updates."""
        if update:
            upd = dict(update)
            for sub_name, sub_cls in (
                ("router", RouterConfig),
                ("typst", TypstConfig),
                ("quality", QualityConfig),
                ("engine", EngineConfig),
            ):
                if sub_name in upd:
                    sub_val = upd.pop(sub_name)
                    if isinstance(sub_val, sub_cls):
                        for k in type(sub_val).model_fields:
                            upd.setdefault(k, getattr(sub_val, k))
                    elif isinstance(sub_val, dict):
                        for k, v in sub_val.items():
                            upd.setdefault(k, v)
            return super().model_copy(update=upd, deep=deep)
        return super().model_copy(update=update, deep=deep)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            UBTEnvSettingsSource(settings_cls, env_prefix="UBT_"),
            file_secret_settings,
        )

    # -- API credentials and endpoints -------------------------------------
    # One canonical env name per field. The third-party names (OPENAI_API_KEY,
    # ANTHROPIC_BASE_URL, ...) are NOT ambient aliases: credentials come from
    # the provider block's ``api_key`` (with optional ${VAR} expansion) or the
    # generic ``UBT_LLM_API_KEY`` override.
    api_key: SecretStr = Field(
        default_factory=_default_api_key,
        validation_alias="UBT_LLM_API_KEY",
    )
    base_url: str = Field(
        default=_OPENAI_BASE_URL,
        validation_alias="UBT_BASE_URL",
    )
    api_mode: ApiMode = "openai-chat"
    provider: str | None = Field(default=None, validation_alias="UBT_PROVIDER")
    repair_provider: str | None = Field(default=None, validation_alias="UBT_REPAIR_PROVIDER")
    #: Name of a ``[providers.<name>]`` block that answers QE judge calls.
    #:
    #: The judge is a different job from translating and is often better served
    #: by a different model — and by a *hosted* one, since a local model large
    #: enough to notice a dropped noun is slow (measured: 35-85 s per block on a
    #: local 27B against 1.6-3.5 s for a hosted model on the same decisions).
    #:
    #: Pointing at a provider block rather than repeating an endpoint keeps this
    #: standardised: every user already knows how to declare one, credentials
    #: resolve the same way (``${VAR}`` expansion, ``UBT_LLM_API_KEY``), and
    #: nothing machine-specific has to live in the shipped config. Unset means
    #: the judge uses the translation provider.
    qe_judge_provider: str | None = Field(default=None, validation_alias="UBT_QE_JUDGE_PROVIDER")
    is_free: bool = False
    cost_per_mtok: tuple[float, float] | None = None
    # -- Declared model capability ------------------------------------------
    # What the endpoint accepts on the wire. Every field is ``None`` when the
    # provider block does not say, which means "not declared" and resolves to the
    # conservative fallback in ``ubt.core.router.registry.fallback_profile``.
    #
    # These used to be *inferred from the model name* by a fourteen-entry table
    # in the registry. That table could not answer for a locally hosted model
    # (the reference machine's five local models all matched nothing) and it
    # mis-answered for unrelated vendors: a name ending in ``-pro`` matched the
    # ``pro`` entry and lost its output-token cap, while ``claude-sonnet-4``
    # matched the bare ``claude`` entry and had its thinking budget suppressed.
    # Declaring the capability next to the endpoint and model id it describes is
    # the only place the answer is actually known.
    #
    # **Which of these are worth asking.** Only two facts cannot be discovered by
    # the engine, and only those earn a form control:
    #
    # * ``supports_system_prompt`` — a rejected system role is a 400 that the
    #   self-heal rules cannot fix, because the fix (merge the system text into the
    #   user turn) is a decision, not a parameter drop. Measured: one call, no
    #   retry, job fails. This one is required.
    # * ``supports_vision`` — a model that cannot read images fails *every* visual
    #   repair attempt, and the failure is per block. Declaring it avoids a
    #   repeated cost; the engine degrades to text repair but pays each time.
    #
    # ``supports_temperature`` and ``supports_reasoning_effort`` are different: the
    # 400 self-heal drops the parameter and the learned-omission cache stops
    # re-sending it, so the whole cost is one failed call per model per process.
    # Measured over four blocks: five requests (four blocks plus one 400) and the
    # parameter never reappears. Asking an operator to answer a question the engine
    # answers itself was the design error this comment records. They stay as
    # overrides for the one case that matters — skipping that single round-trip, or
    # forcing a parameter a gateway accepts but the engine would otherwise drop.
    supports_temperature: bool | None = None
    supports_reasoning_effort: bool | None = None
    supports_system_prompt: bool | None = None
    supports_vision: bool | None = None
    #: How to recover the translation from the raw completion. ``auto`` tries the
    #: XML tag first, then strips code fences and conversational lead-ins.
    extraction_strategy: ExtractionStrategyName = "auto"
    #: Which self-hosted stack serves this model (``ollama``, ``llama-swap``,
    #: ``vllm``, ...). When set, a gateway that answers 401/404 "model not
    #: supported" earns exactly one retry against the local stack. Unset disables
    #: that retry, because guessing it would send a private request to
    #: ``localhost`` for every hosted model.
    deployment_backend: str = ""
    # How the responses protocol spells a reasoning effort: "nested" is
    # ``reasoning: {effort}`` (OpenAI); "flat" is ``reasoning_effort`` with a
    # nested-minimal fallback (OpenCode Zen). Endpoint data, never host sniffing.
    reasoning_dialect: str = "nested"
    # Chat-template flags forwarded verbatim into the chat request body for
    # llama.cpp / vLLM style servers (e.g. {"enable_thinking": false}). Populated
    # from a provider block's ``chat_template_kwargs`` table.
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)
    extra_headers: dict[str, str] = Field(
        default_factory=dict,
        description="Arbitrary custom headers forwarded to outbound LLM provider calls",
    )
    api_timeout: float = Field(
        default=180.0,
        gt=0,
        validation_alias=AliasChoices("UBT_API_TIMEOUT", "UBT_TIMEOUT"),
        description="HTTP request timeout in seconds for LLM inference calls",
    )

    # -- Inbound service auth (X-API-Key gate, opt-in) -------------------------
    # Distinct from outbound ``api_key`` above (which resolves from
    # UBT_LLM_API_KEY, provider direct api_key, or CLI). The service gate
    # must ONLY honor an explicitly configured UBT_API_KEY — it must never
    # read the outbound LLM credential or un-prefixed keys. Empty = open (development mode).
    service_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias="UBT_API_KEY",
    )

    # -- HTTP Host / Origin trust (UBT_TRUSTED_HOSTS) -----------------------
    # Comma-separated hostnames the API will answer for, beyond loopback and the
    # in-process test client. A request whose ``Host`` header matches none is
    # refused (400) — the DNS-rebinding defense: a browser resolving an attacker
    # domain to 127.0.0.1 sends ``Host: evil.example.com``, which no longer reads
    # as the loopback console. Empty keeps the loopback + testserver defaults;
    # an operator serving the console on a real domain sets this to that domain.
    trusted_hosts: str = Field(default="", validation_alias="UBT_TRUSTED_HOSTS")

    #: Whether ``X-UBT-Tenant`` is honored. Off by default: the header is
    #: client-forgeable, so honoring it lets any caller read or cancel another
    #: tenant's job. Set only when a trusted front proxy authenticates the
    #: tenant and sets the header itself (multi-tenant SaaS deployment).
    trust_tenant_header: bool = Field(default=False, validation_alias="UBT_TRUST_TENANT_HEADER")

    # -- Model routing tiers (role-based canonical names) -------------------
    # No vendor default: a model must come from the environment, a provider
    # preset, or an explicit argument. Empty fails loudly at the call boundary.
    draft_model: str = ""
    repair_model: str = ""
    # Ordered model-level fallback chain. When the draft/repair
    # model fails (fail-fast client errors or exhausted retries), the router
    # retries the request on the next entry instead of failing every block.
    fallback_models: list[str] = Field(default_factory=list)
    draft_reasoning_effort: str = "low"
    repair_reasoning_effort: str = "high"

    # -- PDF engine selection ------------------------------------------------
    # ``auto`` — the default — takes the lossless **docling** engine
    # unconditionally. It does NOT route clean single-column PDFs to pypdfium2:
    # that promise was retired after measurement showed every document in the
    # reference corpus carried structure the pdfium reader cannot represent, and
    # the cheap probe signals read zero exactly on vector-drawn figures. The
    # engine choice is a fixed policy; the probe's output is *evidence about the
    # document* (the assess report shows it), not a vote.
    #
    # Measured on arXiv 2610.02185: pdfium splits the paper into 1801 fragments
    # (0 tables, title shattered into 15 pieces) where docling yields 350
    # structured blocks (19 tables, title intact), and the fragment storm then
    # trips the numeric/structural validators into quarantining ~14% of the
    # book. ``high_fidelity``/``standard`` presets also select ``auto``;
    # ``preview``/``fast`` pin ``pdfium`` (a real opt-in for prose-only input),
    # and ``--pdf-engine`` always wins. A scanned (text-less) PDF is a capability
    # boundary, not a preference: pdfium extracts 0 blocks from one.
    #
    # ``auto`` therefore *tries* docling first and, when the ``pdf`` extra is
    # absent, degrades to the geometric pdfium reader with a warning that names
    # what is lost (``factory._resolve_pdf_adapter``); only an explicit
    # ``--pdf-engine docling`` is refused. See
    # docs/guides/PDF_ENGINE_ROUTING_REDESIGN.md.
    pdf_engine: PdfEngine = "auto"

    # -- Docling accelerator device -------------------------------------------
    # Decouples *whether* Docling runs from *where* it runs. 'auto' keeps
    # Docling's own CUDA probe; 'cpu' keeps it off the GPU so a local LLM
    # sharing the card can still load its weights. See
    # docs/guides/PDF_ENGINE_ROUTING_REDESIGN.md §L4.
    docling_device: DoclingDevice = "auto"

    # -- Docling process isolation & worker pool ------------------------------
    # When enabled (default True), Docling PDF conversions run in an isolated
    # worker process to prevent PyTorch/C++ segfaults or OOM from taking down the
    # main UBT orchestrator process, and to reuse warm model weights across chunks.
    docling_isolate_process: bool = True

    # -- Pluggable OCR Configuration -------------------------------------------
    # 'auto' (default): local measured-box engine first, then VLM.
    # 'local': measured-box engine on this machine (rapidocr or sidecar HTTP).
    # 'vlm': OpenAI-compatible vision endpoint (or in-process unlimited-ocr2).
    # 'off': Explicitly disable OCR (scanned pages preserved as source).
    ocr_mode: OcrMode = "auto"
    ocr_endpoint: str = ""
    # Vision model for the OCR channel (env UBT_OCR_MODEL). Declared here so the
    # spend pre-flight and the assessor can price the channel the driver
    # actually bills against. No vendor default: the vision channel must be told
    # which model to call, and an empty value fails closed (see _check_invariants
    # and CloudOcrDriver) rather than silently billing a shipped model.
    ocr_model: str = ""
    # Plain declarative field: env_prefix maps this to UBT_OCR_API_KEY only, so
    # neither the bare ``OCR_API_KEY`` nor a shared LLM key can reach it.
    ocr_api_key: SecretStr = SecretStr("")

    # -- SQLite ledger storage directory ------------------------------------
    db_dir: Path = Path(".ubt/ledgers")
    # Content-addressed step cache (content-addressed cache layer). The render path's pixel
    # witnesses compile + raster every formula/table, the most expensive pure
    # step of a render; a resumed or re-run job reuses the verdict when the
    # inputs are unchanged. Delete the directory to clear it; UBT_CACHE_ENABLED=0
    # disables it (nothing is read or written).
    cache_enabled: bool = True
    cache_dir: Path = Path(".ubt/cache")
    # -- Service job queue ----------------------------------------------------
    # 'embedded' (default): the REST API runs each job as an in-process task —
    # zero setup, single host. 'queue': the API only enqueues into a durable
    # SQLite queue and ``ubt worker`` processes drain it, so jobs survive
    # restarts and scale across processes. Enterprise/on-prem may keep
    # 'embedded'; multi-tenant SaaS uses 'queue'.
    job_mode: Literal["embedded", "queue"] = "embedded"
    # Defaults to <db_dir>/job_queue.sqlite when unset.
    job_queue_path: Path | None = None
    job_max_running: int = Field(default=8, gt=0)
    job_tenant_max_running: int = Field(default=4, gt=0)
    # Depth cap on QUEUED jobs. ``claim`` caps how many run at once, not how
    # many pile up; refusing at submit is the only place queue growth can be
    # stopped, since workers drain at LLM speed.
    job_max_queued: int = Field(default=1000, gt=0)
    # Staged uploads (the wizard POSTs bytes, the API stores a copy under
    # <db_dir>/uploads and returns its path) are never removed by any other
    # path, so the API sweeps the staging directory at startup and deletes
    # files older than this. An upload a non-terminal queue job still lists as
    # its input_path is always spared. 0 disables the sweep.
    upload_retention_days: float = Field(default=7.0, ge=0.0)

    # -- Rate limiting -------------------------------------------------------
    rate_limit_rpm: int = Field(default=60, gt=0)
    rate_limit_tpm: int = Field(default=100_000, gt=0)
    # AIMD additive-increase ceiling for the RPM bucket.
    rate_limit_max_rpm: int = Field(default=240, gt=0)
    # TPM growth ceiling. Governs AIMD token budget adjustments on successful calls
    # up to provider quota; decreases on 429 rate limit responses.
    rate_limit_max_tpm: int = Field(default=600_000, gt=0)
    rate_limit_backoff_cooldown_sec: float = Field(
        default=3.0,
        ge=0.0,
        description="Cooldown window in seconds to prevent concurrent 429 backoff oscillation",
    )

    # -- Quality control ------------------------------------------------------
    qe_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    max_repair_rounds: int = Field(default=2, ge=0)
    # Quoting only: ``ubt assess`` extrapolates the repair fan-out from this
    # fraction (UBT_BOTTOM_PERCENTILE). The repair loop itself does NOT read it —
    # candidates are selected by ``defect_taxonomy.classify_defect``
    # (DETERMINISTIC_DEFECT only). Changing it changes the estimate and the
    # "Bottom N%" line in ``ubt status``, never which blocks get repaired.
    bottom_percentile: float = Field(default=0.15, ge=0.0, le=1.0)
    # Document-level terminology consistency enforcement, run after repair.
    # "off"    = detect + report only (default; no extra LLM spend)
    # "report" = plan the targeted re-translations and log them, still no spend
    # "repair" = run the constrained re-translations (bounded by the cap below)
    consistency_enforce: Literal["off", "report", "repair"] = "off"
    #: Budget for "repair" mode. Zero with ``consistency_enforce="repair"`` is
    #: rejected: the stage would plan drift repairs and then discard every one,
    #: reading "on" while being a guaranteed no-op.
    consistency_max_repairs: int = Field(default=50, ge=0)
    qe_engine: QeEngine = "heuristic"

    # -- L3 LLM-as-Judge (suspect-only rescoring, off by default) ------------
    qe_judge_enabled: bool = False
    qe_judge_model: str | None = None
    #: The judge is asked about blocks whose deterministic verdict is
    #: *ambiguous* — the ``STRUCTURAL_OTHER`` class (a rejection with no specific
    #: classifier) — plus, when ``pass_sample > 0``, a stable sample of clean
    #: passes. There used to be two more knobs here, ``qe_judge_gray_low`` and
    #: ``qe_judge_gray_high``, defining a score band to send to the judge. With
    #: the heuristic engine they were inert: it emits twelve discrete values, the
    #: band was ``[0.7, 0.8)``, and the only reachable value inside it (0.70) was
    #: already in the ambiguous set. A configuration surface that cannot change
    #: behaviour is worse than none — it invites tuning that silently does
    #: nothing — so the band is gone and the rule is stated here in one line.
    qe_judge_pass_sample: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of clean heuristic passes also sent to the LLM judge. The "
            "heuristic cannot distinguish a good translation from a merely "
            "well-formed one, so this is the only place a paid judge informs "
            "quality; 0 (default) keeps the judge on ambiguous classes only."
        ),
    )
    #: The judge may only LOWER a score — there is no "allow upgrade" switch.
    #: A judge verdict is a probabilistic opinion: it can say "this is worse than
    #: the invariants suggest", which is useful, but never "this is better",
    #: because that would release a draft on an opinion alone. The removed
    #: ``qe_judge_allow_upgrade`` could lift a real STRUCTURAL_OTHER failure over
    #: the pass line, which is precisely what the gate exists to prevent.
    # -- Independent judge endpoint -------------------------------------------
    # The judge is often best served by a different (larger) model than the one
    # doing the translating, and differently again by a hosted API rather than
    # the local gateway. Measured on arXiv 2606.23344: the local 4B judge scores a
    # translation that drops "publicly" at 100/100, and the local 27B needs a
    # specific rubric to separate it — so "whatever the repair model is" is not a
    # safe default for a quality gate.
    #
    # All four are optional; each falls back to the main provider's value, so an
    # existing configuration keeps working unchanged.
    qe_judge_base_url: str | None = Field(
        default=None,
        validation_alias="UBT_QE_JUDGE_BASE_URL",
        description=(
            "Endpoint for judge calls, when the judge runs somewhere other than the "
            "translation provider (a hosted API, a second local server). Defaults to "
            "base_url."
        ),
    )
    qe_judge_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="UBT_QE_JUDGE_API_KEY",
        description="Credential for qe_judge_base_url. Defaults to the main api_key.",
    )
    qe_judge_api_mode: ApiMode | None = Field(
        default=None,
        validation_alias="UBT_QE_JUDGE_API_MODE",
        description=(
            "Wire protocol for the judge endpoint (openai-chat, openai-responses, "
            "anthropic, gemini). Defaults to the main api_mode; set it when the "
            "judge endpoint speaks a different protocol than the translation one."
        ),
    )
    qe_judge_sentences: bool = Field(
        default=True,
        validation_alias="UBT_QE_JUDGE_SENTENCES",
        description=(
            "Compare each source sentence against its target sentence instead of "
            "scoring the whole block. A one-word loss diluted across a long block "
            "reads as a small proportional deduction (measured: the same defect "
            "scored 0.85 block-level, below 0.50 sentence-level), so a block-level "
            "score cannot express 'this clause is wrong'."
        ),
    )

    # -- Post-render visual gate (T0/T1 deterministic always safe; T2 VLM sampled) --
    visual_gate_enabled: bool = True
    # -- Spend-the-tokens gate (pre-flight render before Stage 3 bills) --
    # Compiles a source-text sample through the real PDF renderer so a broken
    # toolchain fails with zero tokens spent instead of at Stage 6 after the
    # whole book is translated. Covers toolchain-shaped failures (missing or
    # wrong-version Typst, unrenderable generated markup); translation-content
    # syntax defects that only appear once target text exists surface later, at
    # the export syntax-fallback gate.
    render_preflight_enabled: bool = True
    visual_judge_enabled: bool = False
    visual_judge_model: str | None = None
    # Hard USD cap for one job, measured across every resume: the ledger keeps the
    # job's lifetime token totals, so restarting a stopped job cannot start a
    # fresh budget. None = uncapped. Exceeding fails the job; blocks already
    # drafted stay in the ledger, so resuming after raising UBT_BUDGET_USD costs
    # nothing extra. A cap is enforced only while spend is measurable: starting
    # a capped run with an unpriced model is refused at startup (see
    # ``allow_unpriced_budget``), because "unknown" usage would silently never
    # trip the cap -- a cap you can bypass by renaming a model is no cap.
    budget_usd: float | None = Field(default=None, gt=0)
    # Explicit opt-out of that startup refusal: bill unpriced models as unknown
    # and keep the (inert) cap anyway. Mirrors the strict_auth pattern -- the
    # safe behavior is the default, looseness must be asked for.
    allow_unpriced_budget: bool = False
    # Mirrors of the pricing helper's endpoint knobs (ubt/core/router/pricing.py
    # reads the environment directly — this decision is made deep in the pricing
    # path, where no config object is in scope). Surfaced here so `ubt config`
    # and the docs-completeness gate can see them. There is no dotenv source
    # (see ``model_config``), so the process environment is the one external
    # source: reading ``os.environ`` there and reading this field resolve the
    # same value.
    #
    # Extra hosts declared self-hosted (a LAN inference box), comma/`os.pathsep`
    # separated. Loopback and the container-host aliases need no declaration.
    local_endpoints: str = ""

    # Path to external prices.toml (UBT_PRICES_FILE). When configured or present
    # in standard search locations, its pricing definitions override shipped rates.
    prices_file: Path | None = None

    def remote_billing_models(self) -> dict[str, str]:
        """Models that bill through an endpoint other than ``base_url``.

        Today only the OCR channel: ``UBT_OCR_MODEL`` is billed through
        ``UBT_OCR_ENDPOINT`` (OpenAI by default) when OCR may reach a cloud
        route — ``auto`` counts only with page egress on, matching
        ``ocr_can_bill_cloud`` in the budget pre-flight. An empty map means
        everything bills through ``base_url``.

        Cost accounting needs this so a local ``base_url`` is not read as "the
        whole run is free": a local LLM with cloud OCR still spends real money.
        """
        ocr_can_bill_cloud = self.ocr_mode == "vlm" or (
            self.ocr_mode == "auto" and self.allow_page_upload
        )
        if not ocr_can_bill_cloud:
            return {}
        return {self.ocr_model: self.ocr_endpoint or ""}

    # Master gate for every route that ships rendered PAGE IMAGES of the book to
    # model endpoints: visual-scalpel repair, VLM judge, and cloud/VLM OCR (auto
    # mode degrades to local engines; explicit --ocr cloud|vlm refuses to
    # start). Text prompts are unaffected.
    # Off by default: the rendered pages of an unpublished manuscript are the
    # sensitive half of the payload, so shipping them must never be the default
    # and requires explicit, disclosed opt-in. Set UBT_ALLOW_PAGE_UPLOAD=true to
    # opt into pixel-level repair and cloud OCR; `ubt doctor` prints the
    # effective egress routing either way.
    allow_page_upload: bool = False
    # Local VLM OCR drivers (DeepSeek-OCR) load model-repo Python through
    # transformers' ``trust_remote_code``: the checkpoint's own code executes
    # in this process. Shipping arbitrary code from a downloaded repo must
    # never be the default (same fail-closed posture as ``allow_page_upload``).
    # Default false refuses model-supplied code; the driver then stops at load
    # time and points at a vetted offline snapshot (HF_HOME) instead. Set
    # UBT_VLM_TRUST_REMOTE_CODE=true to opt in and keep the recipe working.
    vlm_trust_remote_code: bool = False
    visual_sample_pages: int = Field(default=6, ge=0, le=50)
    visual_max_vlm_pages: int = Field(default=3, ge=0, le=10)
    # -- visual blocking gate (opt-in for long docs; short docs fail closed) ---
    # Short docs (<= QA_FULL_GATE_MAX_PAGES) enforce visual blocking fail-closed
    # by default. For longer docs, setting this to true refuses export when
    # CRITICAL visual findings are detected; default false keeps the visual report
    # + NEEDS_HUMAN quarantine without blocking long doc delivery.
    visual_blocking_gate_enabled: bool = False
    # -- deterministic glossary enforcement on the long route ----------------
    # The export-time literal term substitution (word-boundary + protected-span
    # + compound + jieba fail-closed guards, see glossary_enforcer.py) is
    # short-route only by default: the long route leans on the bible + QE
    # defenses. The old "Aho-Corasick splices across word boundaries" concern
    # predates those guards; run the on/off comparison on your own corpus
    # (term_precision vs quarantine rate) before enabling it for books.
    # Re-export reads the same flag, so a re-render enforces what the original
    # run enforced instead of silently dropping the layer.
    deterministic_glossary_long_route: bool = False
    # -- export coverage gate (fail-closed) --------------------------------
    # Share of blocks that must carry a target before export may render.
    # Without it, a run in which every block failed still produced a finished
    # book — the renderer falls back to ``source_text`` on an empty target —
    # and the job finalized as "completed". Set to 0 to disable the gate.
    export_min_completion_ratio: float = Field(default=0.5, ge=0.0, le=1.0)

    # -- Concurrency / pagination ----------------------------------------------
    max_concurrency: int = Field(default=10, gt=0)
    batch_limit: int = Field(default=30, gt=0)
    macro_chunk_size: int = Field(
        default=8,
        ge=1,
        le=30,
        description="Number of consecutive micro-blocks packed into a single LLM draft request. "
        ">1 enables the chapter-level long-context path (blocks packed up to a section "
        "boundary, transmitted as <block id=...> markup for one long generation); "
        "1 = one request per block (the legacy micro-window).",
    )
    prompt_caching_enabled: bool = Field(
        default=True,
        description="Enable provider prompt caching headers and structured prefix alignment",
    )
    ledger_flush_interval: float = Field(
        default=0.25,
        gt=0.0,
        description="Maximum seconds before flushing queued draft checkpoints to SQLite WAL",
    )
    ledger_flush_batch_size: int = Field(
        default=50,
        gt=0,
        description="Maximum checkpoints to accumulate before flushing to SQLite WAL",
    )
    chapter_streaming_enabled: bool = Field(
        default=False,
        description="Enable chapter-level streaming pipeline channel for multi-chapter books",
    )
    chapter_streaming_queue_size: int = Field(
        default=2,
        gt=0,
        description="Maximum chapters buffered between draft and repair in chapter streaming",
    )

    # -- Draft retry (delegated to ModelRouter by default) -------------------
    draft_max_retries: int = Field(
        default=0,
        ge=0,
        description="Stage-level outer retry count (default 0 delegates retries and fallback to ModelRouter)",
    )
    draft_retry_base_delay: float = Field(default=1.0, ge=0.0)

    # -- Hierarchical memory step snapshots --------------------------------------
    step_chars: int = Field(default=3500, gt=0)

    # -- Cross-chapter rolling summary (one bulk call per chapter transition) --
    enable_rolling_summary: bool = True
    memory_sliding_window: bool = Field(
        default=True,
        description="Enable sliding window on L3 compressed history so long-form books retain the most recent epochs within budget",
    )
    memory_max_epochs: int = Field(
        default=50,
        gt=0,
        description="Maximum number of L3 epoch summaries retained in memory",
    )

    # -- Decoupled External Glossary & Domain ----------------------------------
    # Declared here and on ProviderConfig because ``[defaults]`` shares that
    # schema; the provider loader rejects it in a ``[providers.<name>]`` block.
    glossary_path: Path | None = None
    glossary_max_global_entries: int = Field(
        default=100,
        ge=0,
        description=(
            "Cap on the book-level terminology sheet carried by every draft prompt "
            "(0 = send none and rely on per-chunk retrieval + the export enforcer). "
            "An uncapped sheet scales the per-request prompt with the book, not the paragraph."
        ),
    )
    domain: str | None = Field(
        default=None,
        description="Domain or subject field descriptor (e.g. 'semiconductor physics', 'biomedicine')",
    )

    # -- Bilingual render-mode advisory -----------------------------------------
    # 'auto' lets the advisor decide; explicit modes only warn.
    # Default 'auto' allows the advisory layout stage to inspect document signals
    # and automatically adopt monolingual when layout is heavily structured/discouraged.
    dual_mode: DualMode = "auto"
    # Physical column count for a reflowed PDF (1 or 2). ``None`` (default) aligns
    # the delivered layout to the *source* — a two-column paper stays two-column,
    # a single-column one stays single-column. Set 1 or 2 to override explicitly
    # (`--columns`, `UBT_LAYOUT_COLUMNS`). This is a *layout* decision, distinct
    # from ``--profile`` (which only informs the prompt/glossary that the text is
    # academic) — the profile must never force a column count.
    layout_columns: int | None = None
    # Also render the complementary artifact (mono when primary is dual and
    # vice versa), BabelDOC no-dual/no-mono style.
    emit_both: bool = False
    # Emit a translation-ready XLIFF 2.1 companion ('*_<tag>.xliff') beside the
    # deliverable: the native reader turns the source PDF into a typed Document,
    # protected spans are masked, and one segment per element is written. It is
    # a *source* view (no target) for CAT-tool handoff; read-only, CPU-only, and
    # it can never affect the rendered artifact. Toggle off to skip re-reading
    # the source on large books.
    emit_xliff_companion: bool = True

    # Emit an artifact check ('*_<tag>_attestations.json') beside the
    # deliverable: whether the delivered file actually carries each text block's
    # placed translation. Read-only, CPU-only, best-effort.
    emit_attestation_shadow: bool = True

    # Emit a semantic HTML view of the delivery ('*_<tag>.html') beside the
    # artifact, and an EPUB 3 package ('*_<tag>.epub'). These are the delivered
    # blocks lowered to another format: a text block with a placed target shows
    # its translation, one kept in the source its source. Read-only, best-effort
    # companions -- a view failure never sinks the PDF delivery.
    emit_html_companion: bool = True
    emit_epub_companion: bool = True

    # Opt-in hard gate: abort export when the delivery contract has an
    # ERROR-severity violation (dropped text, lost/corrupt asset, or a
    # translation that could not be placed). Off by default so a partial book
    # can still ship knowingly; ``ubt verify`` always reports the same contract
    # regardless of this flag.
    strict_contract: bool = False

    # Policy when translated text cannot fit in layout bounding boxes even after
    # minimum font shrinking (spill / overflow):
    # - 'error': strict delivery contract, space failures are ERROR severity (Axiom B)
    # - 'warn': graceful degradation, space failures are downgraded to WARNING with audit note
    # - 'appendix': graceful degradation, writes an overflow companion (*_spill_appendix.json & .md)
    #   listing all displaced text blocks, and downgrades contract violation to WARNING.
    # - 'source_kept_warn': same behaviour as 'warn', named for the warning it
    #   concerns (a block that shipped its source text).
    spill_policy: SpillPolicy = Field(
        default="error",
        description=("Layout spill policy ('error', 'warn', 'appendix', 'source_kept_warn')."),
    )

    # Adaptive layout auto-healing pass during Typst export: iteratively remediates
    # physical margin spills and gutter overruns by adjusting table scaling and
    # breaking long display formulas across lines.
    enable_layout_auto_heal: bool = True
    layout_auto_heal_max_passes: int = 2
    delivery_mode: DeliveryMode = "reflow"
    enable_pdf_optimization: bool = True
    pdf_optimize_profile: Literal["lossless", "ebook", "screen"] = "lossless"

    # -- Table rendering strategy for PDF ----------------------------------------
    # The publication renderer rebuilds every table from the IR, so a table is
    # either translated into the page or deliberately left in the source.
    # - 'auto': translate every table; complex ones also get a translated
    #   appendix companion so the grid is readable even if the rebuild is lossy.
    # - 'appendix': same as 'auto' — translate every table and always write the
    #   appendix companion.
    # - 'skip': never translate tables (saves 100% table LLM tokens).
    table_render_mode: TableRenderMode = Field(
        default="auto",
        description="Table handling mode for PDF ('auto', 'appendix', 'skip').",
    )

    # -- PDF render engine --------------------------------------------------------
    # There is no engine knob: every PDF is rebuilt from the semantic IR and laid
    # out by Typst. ``font_family`` only overrides the body font.
    font_family: str | None = Field(
        default=None,
        description="Override body font family for typesetter (e.g. 'Noto Serif CJK SC').",
    )

    # -- Running head / footer translation (opt-in) -----------------------------
    # Off by default: chrome stays source-visible (chrome opt-in,
    # page numbers never translated) because a
    # stale or truncated chrome target would strip a legible running head and
    # repaint garbage. When on, ingest lets HEADER/FOOTER roles (never
    # PAGE_NUMBER) reach the translator, and the renderer typesets them as
    # ordinary blocks.
    translate_chrome: bool = False

    # -- Prompt-strategy declaration (local test vs production) -----------------
    # 'auto' (default): keep whatever the resolved capability profile says.
    # Force 'minimal' / 'hybrid' / 'rich' to run every model under one strategy —
    # e.g. production runs against large models with 'rich', while local smoke
    # tests keep 'minimal'. Extraction is declared separately
    # (``extraction_strategy``), because a model can need a different prompt shape
    # and a different output-recovery rule.
    prompt_strategy: PromptStrategyName = "auto"

    # -- Document scenario mode (Paper vs Book core experience decoupling) -----
    # 'auto' (default): probes the document archetype and infers 'paper' (academic paper)
    #   or 'book' (long-form book / literature / general).
    # 'paper': Academic paper scenario — prioritizes scientific rigor, academic Typst layout,
    #   vector math preservation, and suppresses fiction character mining and standalone TOC.
    # 'book': Book scenario — prioritizes literary flow, Sliding Epoch continuous memory,
    #   fictional character tracking, TOC preservation, and reflowable EPUB / book styling.
    document_mode: DocumentMode = "auto"

    # -- Execution-mode router (unified entry, adaptive execution) --------------
    # 'auto' (default): decide() probes the PDF (pages/chars/scan) and routes
    #   <= short_max_pages born-digital pages to the short chain (whole-chapter
    #   rewrite + reflow + full visual gate), everything else to the long
    #   block pipeline (``ubt.core.engine.plan.run_stages`` is the authoritative
    #   stage list). 'short' / 'long' force one chain (short on a 300-page book
    #   raises a clear error instead of burning context).
    exec_mode: ExecMode = "auto"
    short_max_pages: int = Field(default=30, gt=0)
    # Docling math formula enrichment (CodeFormulaV2 VLM): 'auto' (GPU probe with graceful
    # degradation), 'on' (force VLM), or 'off' (bypass VLM for fast ingest).
    formula_enrichment: FormulaEnrichment = "auto"
    # Crop margin (fraction of the formula bbox width/height added on each side)
    # used when the VLM reads a formula. Docling's own default is 0.18, which
    # on a tight academic layout reaches into the equation number and the
    # neighbouring prose line: the model then transcribes that text into the
    # LaTeX (``expansion_factor``, docling#4548). 0.08 keeps the crop around the
    # equation while staying clear of the surrounding block. Only meaningful when
    # ``formula_enrichment`` is ``on``: the default screenshot path never crops
    # with a margin, so this does not affect it.
    formula_crop_margin: float = Field(default=0.08, ge=0.0, le=0.5)
    # Whether to place a table of contents recovered from the source.
    # Academic papers (``genre_profile=paper``) carry no TOC by convention, and
    # the one a PDF reader picks up is often a run of section titles with the
    # *source* page numbers, which the rebuilt (reflowed) document does not
    # preserve — so the default is ``auto``: none for a paper, one for a book
    # that actually has it. ``on`` forces it, ``off`` suppresses it. Only a TOC
    # the reader actually recovered is ever placed; this never invents one.
    toc_mode: TocMode = "auto"

    # -- Page range filter -----------------------------------------------------
    pages: str | None = Field(
        default=None,
        validation_alias=AliasChoices("UBT_PAGES", "UBT_PAGE_RANGE"),
        description="Page range filter for PDF ingestion (e.g. '1-2', '1,3,5', '5-10')",
    )

    # -- Fresh re-ingest -------------------------------------------------------
    # Default resume reuses ledger blocks keyed by block_id, so parse-stage
    # fixes (new blocks, changed sources) never reach an existing job, and a
    # changed source file silently resumes stale translations. --fresh clears
    # the job's blocks and re-ingests; without it, a fingerprint mismatch
    # fails fast with a hint instead of mistranslating the wrong file.
    fresh: bool = False

    # -- Natural-language spans inside math -------------------------
    # Off by default (needs the skeleton invariant + soak). When on,
    # FORMULA blocks with translatable \text{...} phrases route those spans
    # through the draft LLM; the math skeleton must stay byte-identical or
    # the block fails closed to source-verbatim (Gate 3 holds).
    c_text_enabled: bool = False

    # -- Translation Memory (shared across jobs via {db_dir}/tm.sqlite) ---------
    # Exact hits skip the LLM entirely; fuzzy hits become few-shot references.
    tm_enabled: bool = True
    tm_fuzzy_threshold: float = Field(default=0.85, ge=0.0, le=1.0)

    # -- Service hardening / deployment (single source of truth) ----
    # Canonical fields for the deployment knobs: env_prefix already maps
    # STRICT_AUTH -> UBT_STRICT_AUTH, ENV -> UBT_ENV, etc., so no manual
    # ``os.getenv`` is needed. Explicit aliases only where the ecosystem name
    # differs from the field name.
    strict_auth: bool = False
    env: str = "development"
    allowed_dirs: str = ""
    # UBT_MODEL_PROFILES_JSON / UBT_MODEL_PROFILES_FILE are consumed directly
    # by ModelCapabilityRegistry._load_environment_profiles: the registry
    # bootstraps under a lock that a UBTConfig built there must not re-enter.

    # -- QE engine normalization --------------------------------------------------
    # Canonical QE engines are heuristic | tiered. The retired neural names
    # (comet / cometkiwi / neural / subprocess) are accepted for backward
    # compatibility and normalize to the heuristic runner with a warning: the
    # CC-BY-NC-SA CometKiwi weights were removed for commercial-license
    # compliance, and a config that asked for them must not crash.
    _QE_LEGACY_ALIASES: ClassVar[dict[str, str]] = {
        "comet": "heuristic",
        "cometkiwi": "heuristic",
        "neural": "heuristic",
        "subprocess": "heuristic",
    }

    @field_validator(
        "api_mode",
        "pdf_engine",
        "docling_device",
        "ocr_mode",
        "qe_engine",
        "dual_mode",
        "prompt_strategy",
        "exec_mode",
        "document_mode",
        "formula_enrichment",
        "toc_mode",
        "spill_policy",
        "table_render_mode",
        "delivery_mode",
        "env",
        mode="before",
    )
    @classmethod
    def _lower_choice(cls, value: object) -> object:
        if isinstance(value, str):
            lowered = value.strip().lower()
            return lowered
        return value

    @field_validator("qe_engine", mode="before")
    @classmethod
    def _normalize_qe_engine(cls, value: object) -> object:
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in cls._QE_LEGACY_ALIASES:
                logger.warning(
                    "qe_engine=%r is retired (the neural CometKiwi scorer was "
                    "removed for commercial-license compliance); normalized to "
                    "'heuristic'. Use 'tiered' with UBT_QE_JUDGE_ENABLED=true for "
                    "an LLM-judge gray-zone rescore.",
                    value,
                )
                return cls._QE_LEGACY_ALIASES[lowered]
            return lowered
        return value

    # -- Cross-field invariants ------------------------------------------------------
    @model_validator(mode="after")
    def _check_invariants(self) -> UBTConfig:
        # NOTE: qe_engine="tiered" does not silently flip qe_judge_enabled.
        # Callers must opt in explicitly (UBT_QE_JUDGE_ENABLED=true); the
        # pipeline logs a warning when tiered is selected without a judge.
        # AIMD ceiling must not sit below the configured initial rate, or the
        # limiter clamps on the first update and the initial RPM is a lie.
        if self.rate_limit_rpm > self.rate_limit_max_rpm:
            self.rate_limit_max_rpm = self.rate_limit_rpm
        # Same ceiling-floor invariant for TPM: an operator who raises the
        # initial budget above the growth ceiling would otherwise get a bucket
        # that clamps on its first refill.
        if self.rate_limit_tpm > self.rate_limit_max_tpm:
            self.rate_limit_max_tpm = self.rate_limit_tpm
        # Repair mode with a zero budget plans drift repairs and then discards
        # every one (plan_consistency_tasks caps to max_repairs): the stage
        # returns before re-translating a block, so the setting reads "on"
        # while being a guaranteed no-op. "report" is the detect-only mode.
        if self.consistency_enforce == "repair" and self.consistency_max_repairs == 0:
            raise ValueError(
                "consistency_enforce='repair' requires consistency_max_repairs > 0; "
                "use 'report' to detect drift without re-translating."
            )
        # If draft_model was explicitly configured (via env or constructor or copy)
        # and repair_model was NOT explicitly configured, sync repair_model to draft_model.
        # Assign only on an actual change: ``validate_assignment`` re-enters this
        # validator, and re-writing the same value would recurse forever. The
        # derived value must not be sticky — drop the marker after assigning, or
        # a later draft override would read the inherited repair as an explicit
        # choice and refuse to follow (the same latch api_mode had).
        if "draft_model" in self.model_fields_set:
            synced_repair = resolve_repair_model(
                self.draft_model,
                self.repair_model,
                repair_is_independent="repair_model" in self.model_fields_set,
            )
            if synced_repair != self.repair_model:
                self.repair_model = synced_repair
                self.model_fields_set.discard("repair_model")
        # Credential-reuse guard: the inbound gate secret must never be the
        # outbound LLM key. When both read UBT_API_KEY, setting the gate
        # silently handed every API consumer the provider credential.
        raw_service_key = self.service_api_key.get_secret_value()
        if raw_service_key and raw_service_key == self.api_key.get_secret_value():
            raise ValueError(
                "UBT_API_KEY (inbound X-API-Key gate) must differ from the outbound "
                "LLM key (UBT_LLM_API_KEY or provider api_key); "
                "refusing to reuse one secret for both."
            )

        # The base URL selects where the bearer token goes, so a malformed one
        # must fail when the config is parsed — not as an httpx error on the
        # first request, and not as a silent no-op for a non-http scheme.
        # ``hostname`` is None for every shape worth refusing here: a missing
        # scheme, a ``file://`` path, and a bracketless IPv6 literal alike.
        base_split = urlsplit(self.base_url.strip())
        if base_split.scheme not in ("http", "https") or not base_split.hostname:
            raise ValueError(
                f"base_url must be an absolute http(s) URL with a host, got {self.base_url!r} "
                "(it decides which server receives the API key; for a local "
                "llama.cpp/Ollama server use http://127.0.0.1:11434/v1, and write an IPv6 "
                "host in brackets: http://[::1]:11434/v1)."
            )

        # The vision-LLM OCR route bills a model the operator must name: UBT
        # ships no vendor default, so an explicitly requested 'vlm' with no
        # model is unbuildable. 'auto' and 'local' pass (the driver resolves
        # at runtime, not at config parse).
        if self.ocr_mode == "vlm" and not self.ocr_model.strip():
            raise ValueError(
                "ocr_mode='vlm' requires a model (UBT_OCR_MODEL): the vision OCR "
                "channel bills the model you name, and UBT ships no vendor default. "
                "Set UBT_OCR_MODEL, choose a local engine (UBT_VLM_DRIVER=rapidocr), "
                "or set UBT_OCR_MODE=off."
            )

        return self

    def bootstrap_runtime_environment(self) -> None:
        """Register dynamic custom pricing, free endpoints, and model capability profiles configured on this instance.

        Explicit runtime bootstrap so that pure config parsing and validation remain free of process-wide global side effects.
        """
        if self.is_free and self.base_url:
            from ubt.core.router.pricing import declare_custom_free_endpoint

            declare_custom_free_endpoint(self.base_url)
        if self.cost_per_mtok is not None:
            from ubt.core.router.pricing import register_custom_model_pricing

            if self.draft_model:
                register_custom_model_pricing(self.draft_model, self.cost_per_mtok)
            if self.repair_model and self.repair_model != self.draft_model:
                register_custom_model_pricing(self.repair_model, self.cost_per_mtok)

        if self.local_endpoints:
            from ubt.core.router.pricing import declare_custom_free_endpoint

            for ep in self.local_endpoints.replace(os.pathsep, ",").split(","):
                if ep.strip():
                    declare_custom_free_endpoint(ep.strip())

        # The config's own prices file. Without this a path set in a config file
        # was silently ignored while ``UBT_PRICES_FILE`` worked (measured: the
        # model stayed unpriced), so the field read as a working knob.
        from ubt.core.router.pricing import declare_custom_prices_file

        declare_custom_prices_file(self.prices_file)

    def capability_profiles(self) -> list[ModelProfile]:
        """The capability profiles *this* configuration declares.

        Pure: builds and returns a list, registering nothing globally. The router
        consumes it through ``registry_from_config``.

        This replaced a block in :meth:`bootstrap_runtime_environment` that
        registered a profile into the process-wide singleton for every
        ``draft_model``/``repair_model``. Three measured defects came from that:

        * **Cross-job leakage.** ``apply_config_overrides`` returns a fresh
          ``UBTConfig`` per job, and the pipeline bootstrapped it. A job that
          declared a capability for ``shared-model`` made the next job on the same
          model id inherit it — including ``supports_system_prompt=False``, which
          merges the system prompt into the user turn for an endpoint that accepts
          a system role.
        * **Unbounded growth.** Five jobs registering distinct model ids took the
          registry from 14 to 20 entries, never reclaimed.
        * **Field loss.** The old copy listed five fields, so a profile's
          ``supports_vision`` and ``deployment_backend`` were dropped by any
          override — the exact flags the visual-scalpel repair and the local
          self-hosted retry read.

        Every undeclared field comes from :func:`fallback_profile`, never from a
        name table: only what this configuration actually states is asserted.

        **An empty list when nothing is declared.** Returning a profile for the
        model id unconditionally made the router resolve to a profile whose
        pattern was the model name, so a diagnostics screen reported
        ``Capability profile: gemma4-e4b`` — technically true and completely
        uninformative, because it only echoed the model id back. Worse, it made
        "no capability declared" indistinguishable from "capabilities declared",
        which is the one distinction the screen exists to draw. With an empty
        list the generic fallback applies and the resolution reports itself as
        the fallback.
        """
        from ubt.core.router.capabilities import ExtractionStrategy, ModelProfile, PromptStrategy
        from ubt.core.router.registry import fallback_profile

        # Nothing declared: let the fallback speak, rather than minting a
        # profile that says nothing.
        declares_something = (
            self.supports_temperature is not None
            or self.supports_reasoning_effort is not None
            or self.supports_system_prompt is not None
            or self.supports_vision is not None
            or self.prompt_strategy != "auto"
            or self.extraction_strategy != "auto"
            or bool(self.deployment_backend)
        )
        if not declares_something:
            return []

        profiles: list[ModelProfile] = []
        # Both tiers get a profile because a job may route either one, and a
        # dedupe keeps a config whose draft and repair are the same model id from
        # registering it twice.
        seen: set[str] = set()
        for model_name in (self.draft_model, self.repair_model):
            if not model_name or model_name in seen:
                continue
            seen.add(model_name)
            base = fallback_profile(model_name)
            # ``'auto'`` means "no opinion": the profile default stands. Any other
            # value is an explicit operator choice, which is the whole point of
            # the field. The enums are constructed here rather than passed as the
            # config's literal strings, so an out-of-vocabulary value fails at the
            # one place that builds a profile instead of halfway through a run.
            strategy = (
                base.prompt_strategy
                if self.prompt_strategy == "auto"
                else PromptStrategy(self.prompt_strategy)
            )
            extraction = (
                base.extraction_strategy
                if self.extraction_strategy == "auto"
                else ExtractionStrategy(self.extraction_strategy)
            )
            profiles.append(
                ModelProfile(
                    model_pattern=model_name,
                    prompt_strategy=strategy,
                    extraction_strategy=extraction,
                    supports_reasoning_effort=(
                        base.supports_reasoning_effort
                        if self.supports_reasoning_effort is None
                        else self.supports_reasoning_effort
                    ),
                    supports_system_prompt=(
                        base.supports_system_prompt
                        if self.supports_system_prompt is None
                        else self.supports_system_prompt
                    ),
                    supports_temperature=(
                        base.supports_temperature
                        if self.supports_temperature is None
                        else self.supports_temperature
                    ),
                    supports_vision=(
                        base.supports_vision
                        if self.supports_vision is None
                        else self.supports_vision
                    ),
                    deployment_backend=self.deployment_backend,
                    display_name=f"Declared ({model_name})",
                )
            )
        return profiles

    def is_strict_auth(self) -> bool:
        """Whether the API must enforce authentication (strict/production)."""
        return bool(self.strict_auth) or self.env == "production"

    def allowed_base_dirs(self) -> list[Path]:
        """Parse allowed_dirs into resolved base directories.

        Splits on comma and semicolon always, plus the platform path separator
        (``os.pathsep``: ':' on POSIX, ';' on Windows) when it is not already
        one of them. On Windows the pathsep is ';' and is already covered, so a
        drive letter's ':' is never treated as a separator (``C:\\path`` stays
        whole); on POSIX ':' is the pathsep and is added, so a pathsep-joined
        list works.
        """
        import os
        import re

        bases: list[Path] = []
        raw = self.allowed_dirs.strip()
        if raw:
            separators = {",", ";"}
            if os.pathsep:
                separators.add(os.pathsep)
            char_class = "[" + re.escape("".join(sorted(separators))) + "]"
            for part in re.split(char_class, raw):
                if part.strip():
                    p = Path(part.strip()).resolve()
                    if p not in bases:
                        bases.append(p)
        return bases

    def get_selected_pages(self) -> set[int] | None:
        """Parse the configured page range into a set of 1-based integers (or None)."""
        return parse_page_ranges(self.pages)

    @classmethod
    @classmethod
    def from_env(
        cls,
        bootstrap: bool = False,
        custom_path: Path | str | None = None,
        overrides: Mapping[str, Any] | None = None,
        **extra_overrides: Any,
    ) -> UBTConfig:
        """Canonical constructor: ``[defaults]``, provider block, env, then overrides.

        None values are skipped, so optional CLI flags can be passed straight
        through without erasing an environment-provided value. One precedence
        ladder, shared with ``apply_config_overrides``:

        ``overrides > process env > [providers.<name>] > [defaults] > field default``

        **Provider selection** follows the same ladder. An explicit ``provider=``
        or ``UBT_PROVIDER`` wins; otherwise ``[defaults] provider = "<name>"``
        decides. The file form is what lets the console write a complete
        configuration: without it the file could declare ten providers and still
        not say which one runs, so selecting one was a CLI-only operation.
        """
        from ubt.core.providers import (
            load_defaults_config,
            load_layer_config,
            merge_provider_under,
        )

        all_overrides = dict(overrides or {})
        all_overrides.update(extra_overrides)
        clean = {k: v for k, v in all_overrides.items() if v is not None}
        provider_name = clean.get("provider") or os.getenv("UBT_PROVIDER")
        if not provider_name:
            # The file layer names the provider only when nothing more explicit
            # did. Read it from ``[defaults]`` alone: resolving it through
            # ``load_layer`` would need the answer it is trying to produce.
            declared = load_defaults_config(custom_path).provider
            provider_name = str(declared) if declared else None
        layer_cfg = load_layer_config(str(provider_name) if provider_name else None, custom_path)
        # ``load_layer`` already carries ``provider`` when it came from
        # ``[defaults]``, so the selection survives the merge without a separate
        # assignment here — one owner for the key, not two that can disagree.
        clean = merge_provider_under(clean, layer_cfg, env_supplied=_env_supplied_field_names())
        cfg = cls(**clean)
        cfg._anchor_state_dir_to_config_file(custom_path)
        if bootstrap:
            cfg.bootstrap_runtime_environment()
        return cfg

    def _anchor_state_dir_to_config_file(self, custom_path: Path | str | None) -> None:
        """Anchor untouched state-dir defaults to the discovered config file.

        ``find_config_file`` walks parent directories, so a run from a
        subdirectory reads the project's ``ubt.toml`` while the CWD-relative
        ``db_dir``/``cache_dir`` defaults put the ledgers under
        ``<subdir>/.ubt`` — resume from any other directory then cannot find a
        healthy ledger. When no layer set them, the project the config came
        from owns its own state. Explicit values (env, file, overrides) are
        respected verbatim; no config file found keeps the CWD default.
        """
        from ubt.core.providers import find_config_file

        try:
            discovered = find_config_file(custom_path)
        except FileNotFoundError:
            return
        if discovered is None:
            return
        anchor = discovered.parent.resolve()
        try:
            cwd = Path.cwd()
        except OSError:
            return
        if anchor == cwd:
            return
        if self.db_dir == Path(".ubt/ledgers"):
            self.db_dir = anchor / ".ubt" / "ledgers"
        if self.cache_dir == Path(".ubt/cache"):
            self.cache_dir = anchor / ".ubt" / "cache"


def _env_supplied_field_names() -> set[str]:
    """Field names the operator set to a *usable* value in the environment.

    Feeds ``merge_provider_under`` so a provider block's default cannot override
    an explicit ``UBT_*`` setting — the precedence ``from_env`` documents.

    "Supplied" must mean the same thing pydantic-settings means. This model is
    built with ``env_ignore_empty=True``, so an empty ``UBT_BASE_URL=""`` is
    treated as *unset* by the field resolution — but a bare ``var in os.environ``
    test counted it as set. The mismatch let a provider block's ``base_url`` be
    stripped (so it fell back to ``api.openai.com``) while the block's ``api_key``
    was kept: a private-gateway credential then went to the public endpoint.
    Mirror the parser: a variable counts only when present and non-empty.
    """
    supplied: set[str] = set()
    for name in UBTConfig.model_fields:
        if any(os.environ.get(var, "") != "" for var in env_var_names(name)):
            supplied.add(name)
    return supplied


def env_var_names(field_name: str) -> list[str]:
    """Environment variables that set ``field_name``, in resolution order.

    A declared ``validation_alias`` is an absolute env name (pydantic-settings
    uses it as-is, e.g. ``api_key`` -> ``UBT_LLM_API_KEY``); a field without one
    resolves through the ``UBT_`` prefix. Deriving this from the schema keeps the
    introspection command and the docs-completeness test off a hand-maintained
    list that would drift the moment a field is renamed.
    """
    field = UBTConfig.model_fields[field_name]
    alias = field.validation_alias
    choices = getattr(alias, "choices", None)
    if choices:
        names: list[str] = []
        for choice in choices:
            name = str(choice)
            if name not in names:
                names.append(name)
        return names
    if isinstance(alias, str):
        return [alias]
    return [f"UBT_{field_name.upper()}"]


def credential_required(config: UBTConfig) -> bool:
    """Whether this configuration needs a real credential to run for real.

    A **loopback endpoint does not.** llama.cpp, Ollama, vLLM and llama-swap
    accept any bearer — often none at all — and this project's own reference
    configuration points at ``127.0.0.1:9090`` with no ``api_key``. The old rule
    ("no credential means rehearsal") therefore downgraded a correctly configured
    local model to a zero-token mock, and the operator saw a rehearsal they never
    asked for while the model sat there answering ``/v1/models``.

    The distinction that matters is not "is there a key" but "**would sending
    this request to this endpoint bill somebody, or reach a third party**". A
    hosted endpoint without a credential cannot run at all, so it must rehearse
    rather than post ``mock-key`` to ``api.openai.com`` (the failure the sentinel
    exists to prevent). A local one runs fine, so it must not be blocked.

    ``UBT_BILL_LOCAL_ENDPOINT`` is deliberately *not* consulted: that flag says
    "this loopback gateway charges money", which affects pricing, not whether a
    credential is needed to reach it. A paid local gateway still accepts requests
    without a key from the machine that hosts it.

    One predicate, because six call sites each compared against ``MOCK_API_KEY``
    and every one of them got the local case wrong.
    """
    from ubt.core.router.pricing import endpoint_is_local

    return not endpoint_is_local(config.base_url)


def has_real_credential(config: UBTConfig) -> bool:
    """Whether a usable credential is configured — the sentinel is not one.

    ``UBTConfig`` fills an unconfigured ``api_key`` with ``MOCK_API_KEY``
    (``"mock-key"``), so "no credential" is not an empty string. Every caller
    that tested truthiness treated a keyless, air-gapped worker as authenticated
    and built a live provider against the literal sentinel.

    Note this answers "is a key present", not "can this run": a local endpoint
    with no key is runnable (:func:`credential_required` is False) even though
    this returns False. Both questions are real and they are not the same one.
    """
    key = config.api_key.get_secret_value().strip()
    return bool(key) and key != MOCK_API_KEY


def require_api_key(config: UBTConfig | None = None) -> str:
    """Return the configured API key, or fail fast with an actionable message.

    Single source of truth for scripts and one-off tools: resolves through the
    canonical ladder (``UBT_LLM_API_KEY``, provider configured ``api_key``,
    or CLI flag) and refuses to silently run live workloads against the
    ``mock-key`` placeholder. Raises :class:`SystemExit` when unconfigured.

    A loopback endpoint is not exempt: a script that asks for a key wants one,
    and the local case is served by not calling this at all rather than by
    returning the sentinel.
    """
    key = (config or UBTConfig.from_env()).api_key.get_secret_value()
    if not key or key == MOCK_API_KEY:
        raise SystemExit(
            "error: no API key configured — set UBT_LLM_API_KEY, configure "
            "api_key in ubt.toml, or pass --api-key"
        )
    return key
