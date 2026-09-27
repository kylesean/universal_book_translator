"""Canonical UBT configuration (single-key design).

Design contract:
- ONE canonical env var per field: ``UBT_<FIELD_NAME>`` (upper-snake of the
  field name), resolved declaratively by ``pydantic-settings`` — there are no
  dual keys. The role-based names (``draft_*`` / ``repair_*``, matching
  :class:`ModelRouter` parameters and CLI flags) are the only names.
- Distinct credentials never share an env name:
  ``UBT_LLM_API_KEY`` (outbound LLM key) falls back to the OpenAI-standard
  ``OPENAI_API_KEY``; ``UBT_API_KEY`` is the inbound HTTP gate only; and
  ``UBT_OCR_API_KEY`` is the OCR/vision key only. Sharing a name across these
  three silently couples the inbound gate to the outbound provider secret
  (a security defect — see ``_check_invariants``). ``UBT_BASE_URL`` falls back
  to ``OPENAI_BASE_URL``. Declared via :class:`AliasChoices`, no manual
  ``os.getenv`` chains.
- Every alias is prefixed or a deliberate third-party name: a bare, prefix-less
  field name (``API_KEY``, ``BASE_URL``, ``PAGES``, ...) is NEVER an env name,
  so an unrelated variable in the process environment cannot hijack the
  outbound credential, the endpoint, or the inbound gate.
- No credential is ever scraped from another program: an outbound key resolves
  only from ``UBT_LLM_API_KEY`` or the explicit third-party names below, and
  ``"mock-key"`` stands in for dry-run / tests. Explicit env always wins — the
  settings layer resolves env before the factory ever runs, so factories never
  peek at other fields' env vars.
- Closed value sets are :data:`Literal` types (fail fast with a clear error
  instead of silently accepting typos); cross-field invariants live in one
  ``model_validator``.
"""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, Literal
from urllib.parse import urlsplit

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

ApiMode = Literal["chat", "responses", "anthropic"]
# Deliberately open, unlike its Literal neighbours: the valid set is
# _PDF_ENGINE_REGISTRY (plus "auto"), which in-process registrations
# extend at runtime. Enumerating it here would be a second source of truth to drift.
# Membership is enforced where the truth lives, by
# ubt.adapters.factory._resolve_pdf_adapter.
PdfEngine = str
# "heuristic" emits a discrete defect-class proxy (12 fixed bands),
# NOT a calibrated quality score — see QE_DEFECT_CLASS_LEGEND in
# ubt/core/qe/comet_runner.py. Calibrated numbers require a neural runner
# ("comet"/"subprocess") or the tiered mix.
QeEngine = Literal["heuristic", "comet", "cometkiwi", "neural", "subprocess", "tiered"]
DualMode = Literal["inline", "alternating", "monolingual", "facing", "auto"]
PeExportFormat = Literal["csv", "xliff", "none"]
RenderEngine = Literal["rigid", "reflow", "auto", "publication"]
# One canonical name per render route: `rigid` keeps each block on its source
# page geometry and paints the translated text back into the original bounding
# boxes, so every non-text element stays pixel-intact; `reflow` (a.k.a.
# `publication`) re-typesets a fresh document from the extracted IR. 'auto'
# selects per document.
RIGID_ENGINES: frozenset[str] = frozenset({"rigid"})
PUBLICATION_ENGINES: frozenset[str] = frozenset({"publication", "reflow"})
# Engine alias mapping to fold input variants onto canonical engine names.
# Shared by ``canonical_render_engine`` and the ``render_engine`` field validator.
RENDER_ENGINE_LEGACY_ALIASES: dict[str, str] = {
    "inplace": "rigid",
    "hybrid": "auto",
}


def canonical_render_engine(value: str | None) -> str:
    """Fold a render-engine alias onto its canonical name.

    Folds aliases like ``inplace``/``hybrid`` to prevent unvalidated metadata
    from defaulting incorrectly. 'auto' passes through.
    """
    engine = (value or "").strip().lower()
    engine = RENDER_ENGINE_LEGACY_ALIASES.get(engine, engine)
    if engine in RIGID_ENGINES:
        return "rigid"
    if engine in PUBLICATION_ENGINES or not engine:
        return "publication"
    return engine


def resolve_repair_model(
    draft_model: str, repair_model: str, *, repair_is_independent: bool
) -> str:
    """Return the repair model, following the draft unless repair was chosen alone.

    "Alone" means configured independently of the draft: an explicit repair
    value, a provider profile whose repair differs from its draft, or a base
    config whose repair already differs from its draft. When repair only ever
    inherited the draft, moving the draft must move repair with it. This is the
    single owner of the draft->repair default; the env constructor, the request
    override path and the model validator all call it.
    """
    return repair_model if repair_is_independent else draft_model


def profile_repair_is_independent(profile: Mapping[str, Any]) -> bool:
    """Whether a provider profile chose a repair model distinct from its draft."""
    return "repair_model" in profile and profile.get("repair_model") != profile.get("draft_model")


def merge_provider_profile(
    explicit: Mapping[str, Any],
    profile_name: str,
    *,
    env_supplied: frozenset[str] | set[str] | None = None,
) -> dict[str, Any]:
    """Layer a provider profile under explicit field values; one profile semantics.

    Explicit values win over the profile. ``repair_model`` follows the effective
    draft unless the profile chose a repair distinct from its own draft (then it
    is independent and kept); when neither side pins a repair, the model
    validator syncs it to the draft. Shared by ``from_env`` and the request
    override path so the same request yields the same models on both.

    ``env_supplied`` names fields the operator set via the real environment or
    ``.env``. Those outrank the profile: without this, a profile's default
    ``draft_model`` in the project ubt.toml silently overwrote
    ``UBT_DRAFT_MODEL``, contradicting ``from_env``'s "explicit env wins"
    contract.
    """
    from ubt.core.profiles import load_provider_profile

    profile = dict(load_provider_profile(profile_name))
    if env_supplied:
        for key in list(profile):
            if key in env_supplied:
                profile.pop(key, None)
    merged = {**profile, **explicit}
    if (
        "draft_model" in explicit
        and "repair_model" not in explicit
        and not profile_repair_is_independent(profile)
    ):
        # Drop the profile's inherited repair so the validator re-syncs it to
        # the caller's draft instead of leaving a stale profile draft value.
        merged.pop("repair_model", None)
    return merged


CoverMode = Literal["auto", "always", "never"]
PromptStrategyName = Literal["auto", "minimal", "hybrid", "rich"]
OcrMode = Literal["auto", "sidecar", "cloud", "vlm", "rapidocr", "off"]
ExecMode = Literal["auto", "short", "long"]
FormulaMode = Literal["strict", "readable"]
FormulaEnrichment = Literal["auto", "on", "off"]
FormulaRender = Literal["native", "image", "witness"]
MathBackend = Literal["typst", "mathjax", "image"]
GranularityMode = Literal["micro", "macro"]

_ZEN_BASE_URL = "https://opencode.ai/zen/go/v1"
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
    from a name this module declares; see the ``api_key`` aliases.
    """
    return SecretStr(MOCK_API_KEY)


def _url_hostname(url: str) -> str:
    """Lowercased hostname of a URL, tolerating a missing scheme.

    Endpoint-family detection must be host-based: a bare
    ``"api.anthropic.com" in base_url`` also matched a host like
    ``api.anthropic.com.evil.example``.
    """
    candidate = (url or "").strip()
    if "://" not in candidate:
        candidate = "https://" + candidate
    try:
        return (urlsplit(candidate).hostname or "").lower()
    except ValueError:
        return ""


def _default_base_url() -> str:
    # Runs only when no base_url alias (UBT_BASE_URL / OPENAI_BASE_URL /
    # ANTHROPIC_BASE_URL / OPENCODE_BASE_URL) is set.
    import os

    opencode_url = os.getenv("OPENCODE_BASE_URL")
    if opencode_url:
        return opencode_url

    # Provider-specific endpoints MUST follow the same order the ``api_key``
    # aliases use (UBT_LLM > OPENAI > OPENCODE > DEEPSEEK > ANTHROPIC > GEMINI).
    # Deriving the URL in a different order sent a credential to a host it did
    # not belong to: with DEEPSEEK_API_KEY and GEMINI_API_KEY both set the key
    # resolved to DeepSeek while the URL resolved to Google. The generic
    # UBT_LLM/OPENAI keys name no endpoint, so they fall through to the OpenAI
    # default rather than guessing a provider.
    if os.getenv("UBT_LLM_API_KEY") or os.getenv("OPENAI_API_KEY"):
        return _OPENAI_BASE_URL
    if os.getenv("OPENCODE_API_KEY"):
        return _ZEN_BASE_URL
    if os.getenv("DEEPSEEK_API_KEY"):
        return "https://api.deepseek.com/v1"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "https://api.anthropic.com"
    if os.getenv("GEMINI_API_KEY"):
        return "https://generativelanguage.googleapis.com/v1beta/openai"
    return _OPENAI_BASE_URL


def _default_opencode_session_id() -> str:
    # Runs only when neither UBT_OPENCODE_SESSION_ID nor OPENCODE_SESSION_ID is set.
    import os

    sid = os.getenv("UBT_OPENCODE_SESSION_ID") or os.getenv("OPENCODE_SESSION_ID")
    if sid:
        return sid
    return f"ses_{uuid.uuid4().hex[:16]}" if os.getenv("OPENCODE_API_KEY") else ""


def packaged_comet_script() -> Path:
    """Absolute path to the packaged CometKiwi scorer, independent of cwd.

    The scorer ships inside ``ubt/`` so ``packages=["ubt"]`` carries it into
    the wheel. The historical probe was ``Path("scripts/comet_score_ipc.py")``
    — relative to the *current working directory* — which resolves to ``None``
    for every installed user (a wheel has no ``scripts/``), silently downgrading
    neural QE to the heuristic scorer.
    """
    return Path(__file__).resolve().parent / "qe" / "comet_score_ipc.py"


def _default_comet_script() -> Path | None:
    candidate = packaged_comet_script()
    return candidate if candidate.exists() else None


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
    """Custom environment settings source for UBTConfig.

    Ensures:
    1. Outbound api_key only reads UBT_LLM_API_KEY / OPENAI_API_KEY from environment,
       never the inbound service key UBT_API_KEY or bare API_KEY.
    2. Bare prefix-less environment names (API_KEY, BASE_URL, PAGES, etc.) are never
       read from environment variables.
    """

    def _extract_field_info(self, field: FieldInfo, field_name: str) -> list[tuple[str, str, bool]]:
        info = super()._extract_field_info(field, field_name)
        if field_name == "api_key":
            return [
                (k, env, is_c)
                for k, env, is_c in info
                if env.upper()
                in (
                    "UBT_LLM_API_KEY",
                    "OPENAI_API_KEY",
                    "OPENCODE_API_KEY",
                    "DEEPSEEK_API_KEY",
                    "ANTHROPIC_API_KEY",
                    "GEMINI_API_KEY",
                )
            ]
        if field_name == "service_api_key":
            return [(k, env, is_c) for k, env, is_c in info if env.upper() == "UBT_API_KEY"]
        return [
            (k, env, is_c)
            for k, env, is_c in info
            if env.upper().startswith("UBT_")
            or env.upper().startswith("OPENAI_")
            or env.upper().startswith("ANTHROPIC_")
            or env.upper().startswith("OPENCODE_")
            or env.upper() == "OPENCODE_SESSION_ID"
        ]


class UBTDotEnvSettingsSource(DotEnvSettingsSource):
    """Read only ``UBT_``-prefixed names out of ``.env``.

    ``DotEnvSettingsSource`` resolves fields through the same ``AliasChoices``
    the real environment does, so a bare ``OPENAI_API_KEY=`` line in ``.env``
    would quietly become an *outbound* credential — sending an unpublished
    manuscript to whichever endpoint the file also named, from a file users
    expect to hold no credentials. Credentials belong in the real environment,
    where :class:`UBTEnvSettingsSource` enforces the inbound/outbound split; a
    repository- or cwd-local ``.env`` stays a plain config file.
    """

    def _extract_field_info(self, field: FieldInfo, field_name: str) -> list[tuple[str, str, bool]]:
        info = super()._extract_field_info(field, field_name)
        return [(k, env, is_c) for k, env, is_c in info if env.upper().startswith("UBT_")]


class UBTConfig(BaseSettings):
    """Configuration settings governing translation pipeline, models, and thresholds."""

    model_config = SettingsConfigDict(
        env_prefix="UBT_",
        # The dotenv source is wired in ``settings_customise_sources`` but is
        # inert without an ``env_file``, so this line is what activates
        # file-based config. Precedence is explicit init > real environment >
        # .env, so an exported variable still wins over the file. Only
        # ``UBT_``-prefixed names are read from the file -- bare provider keys
        # such as OPENAI_API_KEY work as real environment variables, not from
        # .env.
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
        populate_by_name=True,
        validate_assignment=True,
    )

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
            UBTDotEnvSettingsSource(settings_cls, env_prefix="UBT_"),
            file_secret_settings,
        )

    # -- API credentials and endpoints -------------------------------------
    api_key: SecretStr = Field(
        default_factory=_default_api_key,
        validation_alias=AliasChoices(
            "UBT_LLM_API_KEY",
            "OPENAI_API_KEY",
            "OPENCODE_API_KEY",
            "DEEPSEEK_API_KEY",
            "ANTHROPIC_API_KEY",
            "GEMINI_API_KEY",
        ),
    )
    base_url: str = Field(
        default_factory=_default_base_url,
        validation_alias=AliasChoices(
            "UBT_BASE_URL",
            "OPENAI_BASE_URL",
            "ANTHROPIC_BASE_URL",
            "OPENCODE_BASE_URL",
        ),
    )
    api_mode: ApiMode = "chat"
    provider_profile: str | None = None
    # Chat-template flags forwarded verbatim into the chat request body for
    # llama.cpp / vLLM style servers (e.g. {"enable_thinking": false}). Populated
    # from a provider profile's ``chat_template_kwargs`` table.
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
    # UBT_LLM_API_KEY / OPENAI_API_KEY / the other declared provider names for
    # LLM calls). The service gate
    # must ONLY honor an explicitly configured UBT_API_KEY — it must never
    # read the outbound LLM credential or un-prefixed keys. Empty = open (development mode).
    service_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias="UBT_API_KEY",
    )

    # -- Model routing tiers (role-based canonical names) -------------------
    draft_model: str = "muse-spark-1.3-contributor"
    repair_model: str = "muse-spark-1.3-contributor"
    # Ordered model-level fallback chain. When the draft/repair
    # model fails (fail-fast client errors or exhausted retries), the router
    # retries the request on the next entry instead of failing every block.
    fallback_models: list[str] = Field(default_factory=list)
    draft_reasoning_effort: str = "low"
    repair_reasoning_effort: str = "high"

    # -- PDF engine selection ------------------------------------------------
    # 'auto' routes via the first-page heuristic: born-digital single-column
    # PDFs take the pypdfium2 fast path, scans and multi-column layouts stay
    # on the Docling mainline. Explicit 'docling' / 'pdfium' forces one engine.
    pdf_engine: PdfEngine = "auto"

    # -- Pluggable OCR Configuration -------------------------------------------
    # 'auto' (default): Probes local Docker sidecar -> local rapidocr -> Cloud/VLM -> off
    # 'sidecar': Local or remote Docker OCR sidecar container (HTTP)
    # 'cloud': Public cloud OCR REST endpoint (Baidu, Tencent, Aliyun, Azure)
    # 'vlm': Vision LLM (OpenAI-compatible /chat/completions with image_url)
    # 'rapidocr': In-process local rapidocr-onnxruntime
    # 'off': Explicitly disable OCR (scanned pages preserved as source)
    ocr_mode: OcrMode = "auto"
    ocr_endpoint: str = ""
    # Vision model for the OCR channel (env UBT_OCR_MODEL). Declared here so the
    # spend pre-flight and the assessor can price the channel the driver
    # actually bills against; the driver reads the same env var
    # (ubt/adapters/pdf/vlm/drivers/cloud_driver.py, whose DEFAULT_VISION_MODEL
    # this default is pinned to by test_config_ocr_model_default_matches_the_driver).
    ocr_model: str = "gpt-4o-mini"
    # Plain declarative field: env_prefix maps this to UBT_OCR_API_KEY only, so
    # neither the bare ``OCR_API_KEY`` nor a shared LLM key can reach it.
    ocr_api_key: SecretStr = SecretStr("")

    # -- SQLite ledger storage directory ------------------------------------
    db_dir: Path = Path(".ubt/ledgers")
    # Opt-in KDP / human-review Markdown companion beside *_quality_report.json.
    # Off by default: it is a review artifact, not part of the machine contract,
    # and no stale-report sweep knows about the extra file. Set
    # UBT_KDP_AUDIT_MARKDOWN=1 to emit it.
    kdp_audit_markdown: bool = False

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
    bottom_percentile: float = Field(default=0.15, ge=0.0, le=1.0)
    # Best-of-n repair candidates (MBR-lite). 1 = single candidate (default);
    # >1 samples k repairs per repair candidate and keeps the best by a neural
    # QE score. Only meaningful with a neural qe_engine (COMET-Kiwi/tiered);
    # the heuristic runner cannot rank.
    rerank_k: int = Field(default=1, ge=1, le=5)
    # Document-level terminology consistency enforcement, run after repair.
    # "off"    = detect + report only (default; no extra LLM spend)
    # "report" = plan the targeted re-translations and log them, still no spend
    # "repair" = run the constrained re-translations (bounded by the cap below)
    consistency_enforce: Literal["off", "report", "repair"] = "off"
    consistency_max_repairs: int = Field(default=50, ge=0)
    qe_engine: QeEngine = "heuristic"
    comet_model: str = "Unbabel/wmt22-cometkiwi-da"
    comet_script_path: Path | None = Field(default_factory=_default_comet_script)

    # -- L3 LLM-as-Judge (suspect-only gray-zone rescoring, off by default) --
    qe_judge_enabled: bool = False
    qe_judge_model: str | None = None
    # Gray zone is the band sent to the (paid) LLM judge. The lower bound is
    # Kept high so the judge only reviews scores *near the
    # pass line* — it must never re-score clearly-failed blocks (e.g. a
    # numeric-fidelity 0.55 washed up to 0.9), which would silently whitewash
    # real defects. The judge may only LOWER a score (see TieredQERunner).
    qe_judge_gray_low: float = Field(default=0.7, ge=0.0, le=1.0)
    qe_judge_gray_high: float = Field(default=0.8, ge=0.0, le=1.0)
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
    qe_judge_allow_upgrade: bool = Field(
        default=False,
        description="Allow LLM Judge to upgrade gray-zone scores above threshold when justified (default: False, conservative lower-only).",
    )

    # -- Post-render visual gate (T0/T1 deterministic always safe; T2 VLM sampled) --
    visual_gate_enabled: bool = True
    # -- Spend-the-tokens gate (pre-flight render before Stage 3 bills) --
    # Compiles a source-text sample through the real PDF renderer so a broken
    # toolchain fails with zero tokens spent instead of at Stage 6 after the
    # whole book is translated. Covers toolchain-shaped failures (missing or
    # wrong-version Typst, unrenderable generated markup); translation-content
    # syntax defects that only appear once target text exists stay behind the
    # Stage 6 healer.
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
    # and the docs-completeness gate can see them. Because the helper reads the
    # environment, constructing UBTConfig(bill_local_endpoint=True) in code does
    # NOT change pricing: set the variable.
    #
    # Extra hosts declared self-hosted (a LAN inference box), comma/`os.pathsep`
    # separated. Loopback and the container-host aliases need no declaration.
    local_endpoints: str = ""
    # Treat self-hosted endpoints as billable: for a PAID gateway behind
    # 127.0.0.1 (LiteLLM, an OpenCode/Qoder proxy). Default false = local
    # endpoints are free, which is what makes --budget-usd usable with
    # Ollama/llama.cpp, whose model names are absent from the price table.
    bill_local_endpoint: bool = False

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
        ocr_can_bill_cloud = self.ocr_mode in ("cloud", "vlm") or (
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
    # Render-fidelity probe (source-vs-artifact non-text residual + painted
    # coverage). The rigid route measures it on every run — that is where the
    # pixel-preservation promise lives — and contributes advisory ``info``
    # findings plus the ``fidelity_*`` KPIs (so it never blocks delivery). This
    # flag is the explicit opt-in for any other engine, whose reflowed masks
    # would be meaningless, so it is normally left off.
    render_fidelity_enabled: bool = False
    # -- visual blocking gate (opt-in for long docs; short docs fail closed) ---
    # Short docs (<= QA_FULL_GATE_MAX_PAGES) enforce visual blocking fail-closed
    # by default. For longer docs, setting this to true refuses export when
    # CRITICAL visual findings are detected; default false keeps the visual report
    # + NEEDS_HUMAN quarantine without blocking long doc delivery.
    visual_blocking_gate_enabled: bool = False
    # -- export coverage gate (fail-closed) --------------------------------
    # Share of blocks that must carry a target before export may render.
    # Without it, a run in which every block failed still produced a finished
    # book — the renderer falls back to ``source_text`` on an empty target —
    # and the job finalized as "completed". Set to 0 to disable the gate.
    export_min_completion_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    # -- Typst syntax-fallback gate (fail-closed) ---------------------------
    # Max translated lines the Typst self-healer may comment out before export
    # refuses delivery. The healer guarantees a PDF by degrading failing lines,
    # so without this the job finalizes "completed" with content silently gone
    # (only visible in logs + quality_report.syntax_fallbacks). 0 = fail on any
    # removal.
    export_max_syntax_fallbacks: int = Field(default=5, ge=0)

    # -- Concurrency / pagination ----------------------------------------------
    max_concurrency: int = Field(default=10, gt=0)
    batch_limit: int = Field(default=30, gt=0)
    macro_chunk_size: int = Field(
        default=1,
        ge=1,
        le=30,
        description="Number of consecutive micro-blocks packed into a single LLM draft request (1 = single block)",
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

    # -- Draft retry (transient provider failures) ---------------------------
    draft_max_retries: int = Field(default=2, ge=0)
    draft_retry_base_delay: float = Field(default=1.0, ge=0.0)

    # -- Hierarchical memory step snapshots --------------------------------------
    step_chars: int = Field(default=3500, gt=0)

    # -- Cross-chapter rolling summary (one bulk call per chapter transition) --
    enable_rolling_summary: bool = True

    # -- Batch API (async draft workloads at ~50% discount, off by default) ----
    # Only enable against endpoints implementing /v1/files + /v1/batches;
    # the draft stage falls back to the interactive path automatically.
    batch_enabled: bool = False
    offline_batch_enabled: bool = Field(
        default=False,
        description="Draft entire book asynchronously using cloud Batch API (OpenAI/Anthropic 50% discount)",
    )
    batch_poll_interval: float = Field(default=30.0, gt=0)

    # -- Decoupled External Glossary & Domain ----------------------------------
    glossary_path: Path | None = Field(
        default=None,
        description="Path to custom external glossary file (.csv, .tsv, or .json) for domain terms",
    )
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
    batch_poll_timeout: float = Field(default=3600.0, gt=0)
    batch_min_blocks: int = Field(default=5, gt=0)
    # Privacy: reap the Files-API objects a batch run leaves on the provider —
    # the input file holds the whole batch's source text and the output file its
    # translations. Delete is best-effort (a gateway without the endpoint just
    # logs); turn this off only to keep those handles for manual re-download.
    batch_delete_files: bool = True

    # -- Bilingual render-mode advisory -----------------------------------------
    # 'auto' lets the advisor decide; explicit modes only warn.
    # Default 'inline' produces standard single-document interleaved output.
    dual_mode: DualMode = "inline"
    facing_spread: bool = False
    # Also render the complementary artifact (mono when primary is dual and
    # vice versa), BabelDOC no-dual/no-mono style.
    emit_both: bool = False
    # Explicitly deliver a zero-cost '*_rigid.pdf' companion alongside reflow.
    emit_companion_rigid: bool = False

    # -- PDF render engine --------------------------------------------------------
    # 'auto' (default): density dispatch — formula/table/figure-dense documents
    #   take the rigid engine (source page as canvas: geometry, figures and
    #   equations cannot be corrupted by re-extraction), plain prose takes the
    #   reflow route's better typography. This is the smart recommendation for
    #   documents whose structure extraction quality is unknown up front.
    # 'reflow' (alias 'publication'): full Typst reflow with academic
    #   typography and bilingual modes; page count may change. Best on prose
    #   books; risky whenever the parser's table/figure extraction is lossy.
    # 'rigid' (alias 'inplace'; 'hybrid' folds to 'auto'): region-locked
    #   typesetting —
    #   each block owns a rectangle on the original page, source text inside is
    #   replaced by the translation typeset to fit (bounded shrink);
    #   figures/equations stay untouched. Monolingual output only.
    # Honored by PDF adapters via manifest.metadata; other formats ignore it.
    render_engine: RenderEngine = "auto"
    font_family: str | None = Field(
        default=None,
        description="Override body font family for typesetter (e.g. 'Noto Serif CJK SC').",
    )

    # -- Running head / footer translation (opt-in) -----------------------------
    # Off by default: chrome stays source-visible (chrome opt-in,
    # page numbers never translated) because a
    # stale or truncated chrome target would strip a legible running head and
    # repaint garbage. When on, ingest lets HEADER/FOOTER roles (never
    # PAGE_NUMBER) reach the translator and the rigid engine paints them in
    # their own band, still fail-closed on fit.
    translate_chrome: bool = False

    # -- Cover-page policy (deterministic, never heuristic) ---------------------
    # 'auto' (default): page 1 renders as a cover only when it carries no
    # body text (headings/images/meta only). 'always' forces a cover page,
    # 'never' renders page 1 as interior content. Single-page documents and
    # chapter pages therefore always keep their body as content.
    cover_mode: CoverMode = "auto"

    # -- Prompt-strategy override (local test vs production) --------------------
    # 'auto' (default): the capability registry decides per model.
    # Force 'minimal' / 'hybrid' / 'rich' to run every model under one
    # strategy — e.g. production runs against large models with 'rich',
    # while local smoke tests keep 'minimal'. Extraction and
    # output-format profiles are NOT overridden, only prompt assembly.
    prompt_strategy: PromptStrategyName = "auto"

    # -- Execution-mode router (unified entry, adaptive execution) --------------
    # 'auto' (default): decide() probes the PDF (pages/chars/scan) and routes
    #   <= short_max_pages born-digital pages to the short chain (whole-chapter
    #   rewrite + reflow + full visual gate), everything else to the 6-stage
    #   long chain. 'short' / 'long' force one chain (short on a 300-page book
    #   raises a clear error instead of burning context).
    exec_mode: ExecMode = "auto"
    short_max_pages: int = Field(default=30, gt=0)
    # Short-chain formula policy: 'readable' (default) renders formulas as
    # Unicode近似 + 编号对应 (Codex-style, 可读优先); 'strict' keeps the
    # byte-identical four-gate pipeline (严谨优先).
    formula_mode: FormulaMode = "readable"
    # Docling math formula enrichment (CodeFormulaV2 VLM): 'auto' (GPU probe with graceful
    # degradation), 'on' (force VLM), or 'off' (bypass VLM for fast ingest).
    formula_enrichment: FormulaEnrichment = "auto"
    # Display-formula fidelity mode. 'witness' (recommended):
    # keep converted Typst math, but verify every display formula against its
    # source pixels and fall back to the source graphic on a structural
    # mismatch. 'image': render every display formula from its source graphic
    # (zero conversion risk). 'native': keep the converted math as-is.
    formula_render: FormulaRender = "witness"
    # Display-formula rendering backend. 'mathjax' (default):
    # the OCR LaTeX is typeset by MathJax into an SVG vector and embedded,
    # with the formula witness verifying it against the source crop and the
    # source graphic as fallback; degrades to the 'typst' behaviour when
    # Node or the pinned scripts/mathjax packages are absent. 'image':
    # every display formula is the source crop (zero rendering risk).
    # 'typst': home-grown LaTeX->Typst converter with formula_render.
    math_backend: MathBackend = "mathjax"

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
    allowed_dir: str = ""
    model_profiles_json: str = ""
    model_profiles_file: str = ""
    opencode_session_id: str = Field(
        default_factory=_default_opencode_session_id,
        validation_alias=AliasChoices("UBT_OPENCODE_SESSION_ID", "OPENCODE_SESSION_ID"),
    )

    # -- Human PE (HITL) queue: MQM severity triage + post-editing ---------------
    # MQM severity triage (Critical -> escalated repair / BLOCKED_HUMAN;
    # Major -> NEEDS_HUMAN) ALWAYS runs in the pipeline so the
    # "Critical escape rate 0" guarantee holds regardless of this flag. This
    # flag gates ONLY the human post-editing export (CSV default / XLIFF
    # 2.1); re-imported revisions flow back into the TM as ``human_pe``.
    pe_queue_enabled: bool = False
    pe_export_format: PeExportFormat = "csv"

    # -- Chunking granularity --------------------------------------------------
    # 'micro' (default): atomic blocks / paragraphs with frozen-math protection.
    # 'macro' (whole-section / chapter synthesis) is unsupported and rejected by
    # the validator below rather than accepted-and-ignored:
    # resolve_adaptive_policy always returns MICRO.
    granularity: GranularityMode = "micro"

    # -- Normalization ------------------------------------------------------------
    # Canonical QE engines are heuristic | subprocess | tiered.
    # Alias names comet / cometkiwi / neural all map to "external subprocess scorer"
    # and normalize to subprocess.
    _QE_LEGACY_ALIASES: ClassVar[dict[str, str]] = {
        "comet": "subprocess",
        "cometkiwi": "subprocess",
        "neural": "subprocess",
    }

    @field_validator(
        "api_mode",
        "pdf_engine",
        "ocr_mode",
        "qe_engine",
        "dual_mode",
        "pe_export_format",
        "render_engine",
        "cover_mode",
        "prompt_strategy",
        "exec_mode",
        "formula_mode",
        "formula_enrichment",
        "formula_render",
        "math_backend",
        "granularity",
        "env",
        mode="before",
    )
    @classmethod
    def _lower_choice(cls, value: object) -> object:
        if isinstance(value, str):
            lowered = value.strip().lower()
            return lowered
        return value

    @field_validator("granularity", mode="after")
    @classmethod
    def _reject_retired_macro(cls, value: str) -> str:
        """Reject the unsupported 'macro' granularity instead of silently ignoring it.

        ``resolve_adaptive_policy`` always returns MICRO, so accepting 'macro'
        would let a user believe whole-section synthesis is running when it is
        not. Failing fast with the reason is the honest option — reviving macro
        would mean reimplementing it.
        """
        if value == "macro":
            raise ValueError(
                "granularity='macro' (legacy whole-section synthesis) has been retired in "
                "favour of the canonical frozen-math micro-block architecture, so the setting "
                "had no effect. Use 'micro' (the default) or remove the setting entirely."
            )
        return value

    @field_validator("qe_engine", mode="before")
    @classmethod
    def _normalize_qe_engine(cls, value: object) -> object:
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in cls._QE_LEGACY_ALIASES:
                logger.warning(
                    "qe_engine=%r is deprecated; normalized to 'subprocess' "
                    "(same external scorer, no behavior change)",
                    value,
                )
                return cls._QE_LEGACY_ALIASES[lowered]
            return lowered
        return value

    _RENDER_ENGINE_LEGACY_ALIASES: ClassVar[dict[str, str]] = RENDER_ENGINE_LEGACY_ALIASES

    @field_validator("render_engine", mode="before")
    @classmethod
    def _normalize_render_engine(cls, value: object) -> object:
        if isinstance(value, str):
            lowered = value.strip().lower()
            alias = cls._RENDER_ENGINE_LEGACY_ALIASES.get(lowered)
            if alias is not None:
                logger.warning(
                    "render_engine=%r is retired; normalized to %r "
                    "(inplace/hybrid were removed in the rigid convergence)",
                    value,
                    alias,
                )
                return alias
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
        if self.offline_batch_enabled and not self.batch_enabled:
            object.__setattr__(self, "batch_enabled", True)
        if not self.qe_judge_gray_low < self.qe_judge_gray_high:
            raise ValueError(
                "UBT_QE_JUDGE_GRAY_LOW must be < UBT_QE_JUDGE_GRAY_HIGH "
                f"(got {self.qe_judge_gray_low} >= {self.qe_judge_gray_high})"
            )
        # Repair mode with a zero budget plans drift repairs and then discards
        # every one (plan_consistency_tasks caps to max_repairs): the stage
        # returns before re-translating a block, so the setting reads "on"
        # while being a guaranteed no-op. "report" is the detect-only mode.
        if self.consistency_enforce == "repair" and self.consistency_max_repairs == 0:
            raise ValueError(
                "consistency_enforce='repair' requires consistency_max_repairs > 0; "
                "use 'report' to detect drift without re-translating."
            )
        # Best-of-n rerank ranks candidates by a neural QE score. The heuristic
        # runner emits discrete defect-class bands with nothing to rank, and
        # the tiered runner scores through the same heuristic leg, so
        # RepairLoop disables reranking for both; surface the no-op instead of
        # silently spending a draft per extra candidate.
        if self.rerank_k > 1 and self.qe_engine in ("heuristic", "tiered"):
            logger.warning(
                "rerank_k=%d has no effect with qe_engine='%s' (no calibrated "
                "score to rank candidates); set a neural qe_engine or rerank_k=1.",
                self.rerank_k,
                self.qe_engine,
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
                "LLM key (UBT_LLM_API_KEY / OPENAI_API_KEY); refusing to reuse one "
                "secret for both."
            )

        # Provider URL normalization & protocol auto-detection
        raw_base = self.base_url.strip()
        if raw_base.lower() == "gemini":
            self.base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
        elif _url_hostname(raw_base) == "generativelanguage.googleapis.com":
            base_clean = raw_base.rstrip("/")
            if not base_clean.endswith("/openai"):
                if base_clean.endswith("/v1beta"):
                    self.base_url = f"{base_clean}/openai"
                elif base_clean in (
                    "https://generativelanguage.googleapis.com",
                    "http://generativelanguage.googleapis.com",
                ):
                    self.base_url = f"{base_clean}/v1beta/openai"
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
        # api_mode is derived from the endpoint and the model family, but only
        # when the caller did not choose it. The derivation must not be sticky:
        # ``apply_config_overrides`` assigns one field at a time, so a derived
        # value recorded in ``model_fields_set`` would block re-derivation when
        # a later override moves off a muse- model or onto the Anthropic
        # endpoint (the api_mode latch — a chat-only model then went out on the
        # Responses wire). Assign, then drop the derived marker so the next
        # pass recomputes from the new inputs; an explicit api_mode is never in
        # play here because the guard skips it.
        if "api_mode" not in self.model_fields_set:
            derived_api_mode: ApiMode = "chat"
            # Only the draft model decides the wire. A muse *repair* model is
            # still routed to Responses per call by
            # ``Provider._select_transport`` (it upgrades muse->responses), but
            # letting it force the whole provider onto Responses sent a
            # non-muse draft model there too, where the endpoint 404s.
            if self.draft_model.startswith("muse-"):
                derived_api_mode = "responses"
            # Anthropic wins when both signals hold, ensuring the endpoint
            # check takes priority over the model prefix.
            if _url_hostname(self.base_url) == "api.anthropic.com":
                derived_api_mode = "anthropic"
            if derived_api_mode != self.api_mode:
                self.api_mode = derived_api_mode
                self.model_fields_set.discard("api_mode")

        return self

    def is_strict_auth(self) -> bool:
        """Whether the API must enforce authentication (strict/production)."""
        return bool(self.strict_auth) or self.env == "production"

    def allowed_base_dirs(self) -> list[Path]:
        """Parse allowed_dirs/allowed_dir into resolved base directories.

        Supports ``os.pathsep`` (';' on Windows, ':' on POSIX) as well as comma/semicolon,
        avoiding accidental splitting of Windows drive letters (e.g. C:\\path).
        """
        import os
        import re

        bases: list[Path] = []
        raw = self.allowed_dirs.strip()
        if raw:
            parts = re.split(r"[,;]", raw) if os.pathsep != ":" else re.split(r"[:,;]", raw)
            for part in parts:
                if part.strip():
                    bases.append(Path(part.strip()).resolve())
        elif self.allowed_dir.strip():
            bases.append(Path(self.allowed_dir.strip()).resolve())
        return bases

    def get_selected_pages(self) -> set[int] | None:
        """Parse the configured page range into a set of 1-based integers (or None)."""
        return parse_page_ranges(self.pages)

    @classmethod
    def from_env(cls, **overrides: Any) -> UBTConfig:
        """Canonical constructor: environment + fallbacks, profile, then validated overrides.

        None values are skipped, so optional CLI flags can be passed straight
        through without erasing an environment-provided value.
        """
        clean = {k: v for k, v in overrides.items() if v is not None}
        profile_name = clean.get("provider_profile") or os.getenv("UBT_PROVIDER_PROFILE")
        if not profile_name:
            env_file = cls.model_config.get("env_file")
            if env_file and Path(str(env_file)).exists():
                from dotenv import dotenv_values

                profile_name = dotenv_values(str(env_file)).get("UBT_PROVIDER_PROFILE")
        if profile_name:
            clean = merge_provider_profile(
                clean, profile_name, env_supplied=_env_supplied_field_names()
            )
        return cls(**clean)


def _env_supplied_field_names() -> set[str]:
    """Field names the operator set via the real environment or the dotenv file.

    Feeds ``merge_provider_profile`` so a profile's default cannot override an
    explicit ``UBT_*`` setting — the precedence ``from_env`` documents.
    """
    supplied: set[str] = set()
    env_file = UBTConfig.model_config.get("env_file")
    file_values: dict[str, Any] = {}
    if env_file and Path(str(env_file)).exists():
        from dotenv import dotenv_values

        file_values = dict(dotenv_values(str(env_file)))
    for name in UBTConfig.model_fields:
        if any(var in os.environ or var in file_values for var in env_var_names(name)):
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


def require_api_key(config: UBTConfig | None = None) -> str:
    """Return the configured API key, or fail fast with an actionable message.

    Single source of truth for scripts and one-off tools: honors the canonical
    precedence (``UBT_LLM_API_KEY`` > ``OPENAI_API_KEY`` > the other declared
    names) and refuses to silently run live workloads against the ``mock-key``
    placeholder. Raises :class:`SystemExit` (exit code 2) when unconfigured.
    """
    key = (config or UBTConfig()).api_key.get_secret_value()
    if not key or key == MOCK_API_KEY:
        raise SystemExit(
            "error: no API key configured — set UBT_LLM_API_KEY (or one of the "
            "declared provider names, or a profile in ubt.toml) in the environment"
        )
    return key
