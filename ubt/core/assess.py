"""`ubt assess` — the pre-flight quote: route, cost and risk before any spend.

Product contract: users of a paid engine confirm *quotes*, not
parameters. This module turns the intelligence already scattered across the
router-mode probe, the pdfium-gated PDF witnesses and the cost-estimate
subsystem into one side-effect-free report: what the document is, which route
the pipeline will take, what it is expected to cost, and which known damage
signals were detected — so a bad ``--preset`` choice is caught before the
first billable request instead of after a whole book of bad output.

Two limits are contractual, not sloppiness:

* **No quality prediction.** MTQE only exists on drafted text; emitting a
  made-up pre-run score would launder a guess into a promise. The report gives
  route *confidence* and factual damage signals instead.
* **Money is expected, never guaranteed.** Repair/QE/VLM fan-out follows the
  config knobs' ceilings and average defect fractions; actual spend varies
  with real defect rates. Unpriced models surface as ``None``, never as $0.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ubt.core.archetype import DocCategory, MathDensity, analyze_archetype
from ubt.core.config import RIGID_ENGINES
from ubt.core.engine.cost_estimate import (
    estimate_draft_cost_from_totals,
    measure_prefix_tokens,
)
from ubt.core.policy.adaptive_policy import resolve_render_engine_from_signals
from ubt.core.policy.layout_policy import PROBE_MIN_CHARS
from ubt.core.ports import (
    inspect_font_encoding_damage,
    inspect_pdf_route_plan,
    page_kind_enum,
    probe_pdf_pages,
    profile_pdf_pages,
    resolve_adapter,
    summarize_font_encoding_damage,
    supported_suffixes,
)
from ubt.core.router.pricing import (
    BATCH_API_DISCOUNT,
    price_is_known,
    resolve_model_prices,
)
from ubt.core.router_mode import decide as decide_route

if TYPE_CHECKING:
    from ubt.core.config import UBTConfig

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# Fast-mode block estimate: a translatable block averages ~350 source chars
# (calibrated from real document IR paragraph, heading, and caption splits).
# Deliberately flagged: --deep replaces it with exact ledger-bound counts.
APPROX_BLOCK_CHARS = 350
# Expected characters per scanned page when OCR is active (~300 tokens/page).
EXPECTED_SCANNED_PAGE_CHARS = 1200
# Repair fan-out: the gate repairs the bottom 15% of MTQE scores, up to
# max_repair_rounds passes (mirrors the ledger status page's "Bottom 15%" line).
REPAIR_DEFECT_FRACTION = 0.15
# QE-judge fan-out: only gray-band blocks reach the LLM judge (subprocess/tiered
# engines); a quarter of blocks is the gray-band width the thresholds allow.
QE_GRAY_FRACTION = 0.25
# Vision-call heuristic: a page image is ~1024 tokens (router's own
# reservation constant) plus the judge scaffolding; answers are short.
_VLM_PROMPT_TOKENS = 1500
_VLM_COMPLETION_TOKENS = 300
# Pages whose drawing-row count is this high make textgeom's line merge
# quadratic-slow (documented defect: textgeom's row merge is quadratic on
# rect-heavy pages): deep mode
# on such a document can take minutes.
PATHOLOGICAL_RECT_ROWS = 2000

_CENTS_EXCLUDED_NOTE = (
    "报价为「预期非保证」：修复/QE/视觉环节按配置上限与平均缺陷率外推，实际花费随真实缺陷率浮动；"
    "不含术语回填。费用未知表示模型无价格表条目，绝非 $0。"
)


class AssessmentError(Exception):
    """The document cannot be assessed at all (missing, unreadable, unsupported)."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code


@dataclass(frozen=True)
class AssessmentWarning:
    """A stable machine code plus Chinese human copy — consumers switch on code."""

    code: str
    level: str  # "warn" | "info"
    detail_zh: str


@dataclass(frozen=True)
class DocumentFacts:
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
    # PDF structural facts (None for other formats):
    primary_engine: str | None = None
    has_vector_diagrams: bool | None = None
    has_multicolumn: bool | None = None
    has_formulas: bool | None = None
    text_layer_coverage: float | None = None
    scan_page_share: float | None = None


@dataclass(frozen=True)
class RouteRecommendation:
    mode: str  # short | long | auto (probe unavailable)
    reason: str
    recommended_preset: str
    recommended_render_engine: str
    recommended_dual_mode: str
    recommended_profile: str
    confidence: float
    confidence_basis: str


@dataclass(frozen=True)
class CostQuote:
    draft_model: str
    repair_model: str
    prefix_tokens_per_call: int | None
    billable_blocks: int
    billable_blocks_is_exact: bool
    prompt_tokens: int
    completion_tokens: int
    draft_cost_usd_cached: float | None
    draft_cost_usd_uncached: float | None
    repair_blocks: int
    repair_cost_usd: float | None
    qe_calls: int
    qe_cost_usd: float | None
    vlm_page_calls: int
    ocr_page_calls: int
    vision_cost_usd: float | None
    rollup_calls: int
    total_cost_usd: float | None
    money_is_unknown: bool
    rollup_cost_usd: float | None = None
    is_estimate_only: bool = True
    excluded_note: str = _CENTS_EXCLUDED_NOTE
    #: Effective call counts (post batch/macro-chunk and rerank fan-out) for the
    #: runtime estimate — ``billable_blocks``/``repair_blocks`` are pre-fan-out.
    draft_calls: int = 0
    repair_calls: int = 0


@dataclass(frozen=True)
class RuntimeEstimate:
    # Always True: no per-stage timing is persisted anywhere, so every
    # number here is derived from the rate-limiter knobs, not from history.
    heuristic: bool
    est_seconds_low: float
    est_seconds_high: float
    basis: str


@dataclass(frozen=True)
class AssessmentReport:
    schema_version: int
    status: str  # "ok" — failures raise AssessmentError instead
    path: str
    deep: bool
    document: DocumentFacts
    route: RouteRecommendation
    cost: CostQuote
    runtime: RuntimeEstimate
    quality_signals: list[str]
    warnings: list[AssessmentWarning]
    next_step_command: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _price_component(
    model: str, prompt_tokens: int, completion_tokens: int, *, base_url: str | None = None
) -> float | None:
    """USD for one fan-out component, or None when the price is not knowable.

    A self-hosted endpoint (``base_url`` is loopback/declared) prices at $0.0 —
    known, not unknown — so an Ollama/llama.cpp quote states the money instead
    of refusing to state anything.
    """
    if not price_is_known(model, base_url=base_url):
        return None
    input_price, output_price = resolve_model_prices(model)
    return (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000


def _build_offline_router(config: UBTConfig) -> Any:
    """A router used only for pure prompt assembly — provider is never called.

    ``measure_prefix_tokens`` walks ``build_draft_prompt`` (registry + string
    formatting), which touches no provider/network/key, so a mock provider
    keeps the whole quote credential-free.
    """
    from ubt.core.router.provider import MockModelProvider
    from ubt.core.router.router import ModelRouter

    return ModelRouter(
        MockModelProvider(),
        draft_model=config.draft_model,
        repair_model=config.repair_model,
        prompt_strategy_override=config.prompt_strategy,
    )


def _measure_prefix_tokens_or_warn(
    config: UBTConfig, warnings: list[AssessmentWarning], *, target_lang: str, source_lang: str
) -> int:
    prefix: int | None = None
    try:
        prefix = measure_prefix_tokens(
            _build_offline_router(config), target_lang=target_lang, source_lang=source_lang
        )
    except Exception as exc:  # a quote must survive any probe failure
        logger.debug("assess: prefix measurement failed (%s)", exc)
    if prefix is None:
        warnings.append(
            AssessmentWarning(
                "PREFIX_UNMEASURED",
                "info",
                "无法实测 prompt 前缀 token，已回退历史校准常量 1000（前缀未测量）。",
            )
        )
        return 1000
    return prefix


def _safe(
    fn: Callable[[], Any], warnings: list[AssessmentWarning], code: str, detail_zh: str
) -> Any:
    """Run one probe; a failure degrades the report, it never aborts it."""
    try:
        return fn()
    except Exception as exc:  # every primitive is optional evidence
        logger.debug("assess: probe %s unavailable (%s)", code, exc)
        warnings.append(AssessmentWarning(code, "info", detail_zh))
        return None


def _pdf_facts(
    path: Path,
    warnings: list[AssessmentWarning],
    known_pages_and_chars: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Aggregate the PDF-only witnesses into plain facts (all fail-soft)."""
    # Guard against password-protected encrypted PDFs. An encrypted file is a
    # hard refusal, not a probe that may degrade the report, so the
    # ``AssessmentError`` is raised from *inside* a handler: a sibling
    # ``except Exception`` cannot swallow it the way the old nested form's
    # outer ``except (ImportError, Exception)`` did (it caught the very
    # exception the inner handler raised, making the guard a no-op).
    try:
        import pikepdf
    except ImportError:
        pass
    else:
        try:
            with pikepdf.open(str(path)):
                pass
        except pikepdf.PasswordError as exc:
            raise AssessmentError(
                "ENCRYPTED", f"PDF 文件受密码保护或已加密，请先解密后再评估: {path}"
            ) from exc
        except Exception as exc:  # any other open failure degrades
            logger.debug("assess: pikepdf could not open %s (%s)", path, exc)

    page_kind = page_kind_enum()
    min_chars = PROBE_MIN_CHARS

    out: dict[str, Any] = {}

    plan = _safe(
        lambda: inspect_pdf_route_plan(path),
        warnings,
        "PDF_PLAN_UNAVAILABLE",
        "PDF 路由计划探测失败，主引擎推荐缺位。",
    )
    if plan is not None:
        out["primary_engine"] = plan.primary_engine
        out["has_vector_diagrams"] = plan.has_vector_diagrams
        out["has_formulas"] = plan.has_formulas
        out["has_multicolumn"] = plan.has_multicolumn

    pages = _safe(
        lambda: profile_pdf_pages(path),
        warnings,
        "PAGE_PROFILE_UNAVAILABLE",
        "逐页分类不可用，扫描占比按前 5 页启发式估计。",
    )
    if pages:
        n = len(pages)
        # Coverage = pages whose text layer yields real characters. Not the
        # EDITABLE_TEXT kind share: two-column academic pages classify as
        # MIXED_COMPLEX/VECTOR_HEAVY yet extract fine.
        with_text = sum(1 for p in pages if p.facts.n_chars >= min_chars)
        scanned = sum(1 for p in pages if p.kind == page_kind.SCAN_IMAGE)
        out["page_count"] = n
        out["text_layer_coverage"] = round(with_text / n, 4)
        out["scan_page_share"] = round(scanned / n, 4)
        out["max_rect_rows"] = max(p.facts.n_rect_rows for p in pages)
        out["poster_pages"] = sum(1 for p in pages if p.kind == page_kind.POSTER_FIXED)
        out["resume_pages"] = sum(1 for p in pages if p.kind == page_kind.RESUME_DENSE)

    wit = _safe(
        lambda: summarize_font_encoding_damage(inspect_font_encoding_damage(path)),
        warnings,
        "FONT_WITNESS_UNAVAILABLE",
        "字体编码损伤检测不可用。",
    )
    if wit is not None:
        out["witness"] = wit

    if known_pages_and_chars is not None and known_pages_and_chars[0] > 0:
        out["probed_pages"], out["probed_chars"] = known_pages_and_chars
    else:
        probe = _safe(
            lambda: probe_pdf_pages(path),
            warnings,
            "PDF_PROBE_UNAVAILABLE",
            "PDF 全量字符统计失败，规模按采样估计。",
        )
        if probe is not None:
            out["probed_pages"], out["probed_chars"] = probe
    return out


async def _deep_blocks(path: Path, config: UBTConfig) -> tuple[int, int]:
    """Exact (billable_blocks, source_chars) from the real adapter ingest path."""
    from ubt.core.cleaners.skip_rules import classify_skip
    from ubt.core.ir.models import BlockType

    adapter = resolve_adapter(path, pdf_engine=config.pdf_engine)
    await adapter.extract_manifest(path)
    blocks: list[Any] = []
    async for chapter in adapter.parse_stream(path):
        blocks.extend(chapter.blocks)
    billable = [
        b
        for b in blocks
        if not getattr(b, "skip_translate", False)
        and getattr(b, "block_type", None) not in (BlockType.FORMULA, BlockType.IMAGE)
        and not classify_skip(getattr(b, "source_text", "") or "")
    ]
    return len(billable), sum(len(getattr(b, "source_text", "") or "") for b in billable)


def _recommend_route(
    arch: Any, route: Any, pdf: dict[str, Any], config: UBTConfig
) -> RouteRecommendation:
    """Correctness-dimension synthesis (aligned with adaptive_policy density dispatch)."""
    math_heavy = arch.math_density == MathDensity.HIGH
    academic = arch.category in (DocCategory.ACADEMIC_PAPER, DocCategory.TECHNICAL_BOOK)
    pdf_format = arch.format_ext == "pdf"
    has_formula_signal = bool(pdf.get("has_formulas")) or math_heavy
    has_vector_signal = bool(pdf.get("has_vector_diagrams"))

    # Scanned-page truth: prefer the per-page census over the 5-page heuristic.
    if pdf.get("scan_page_share") is not None:
        scanned = pdf["scan_page_share"] > 0.5
        heuristic_agrees = scanned == arch.is_scanned
        confidence = 0.9 if heuristic_agrees else 0.55
        basis = (
            "逐页文本层普查 + 前5页启发式一致"
            if heuristic_agrees
            else "前5页启发式与逐页分类不一致，判型置信度降低"
        )
    elif route is not None:
        scanned = arch.is_scanned
        confidence, basis = 0.7, "路由探测可用，但未做逐页普查"
    else:
        scanned = arch.is_scanned
        confidence, basis = 0.5, "仅前5页采样启发式"

    # Route render_engine recommendation: predict the runtime ``auto`` dispatch
    # through the same canonical resolver the renderer uses
    # (``adaptive_policy.resolve_render_engine_from_signals``), so the route a
    # user is quoted is the route that runs. The probe cannot see the block mix,
    # so it estimates the structure signal from its diagram/scan facts: a
    # document that carries vector figures or scanned pages yields IMAGE blocks
    # and is structure-dense enough for rigid. A purely academic classification
    # is NOT a routing signal on its own — academic prose with no math or
    # figures reflows better and the runtime dispatcher would say so.
    if pdf_format:
        canonical = resolve_render_engine_from_signals(
            "auto",
            has_math=has_formula_signal,
            struct_share=1.0 if (has_vector_signal or scanned) else 0.0,
            has_geometry=True,
        )
        render_engine = "rigid" if canonical == "rigid" else "reflow"
    else:
        render_engine = "reflow"

    dual_mode = "monolingual" if render_engine in RIGID_ENGINES else "inline"
    preset = "publication" if (math_heavy or academic) else "standard"
    profile = "textbook" if (math_heavy or academic) else "general"

    if route is not None:
        mode, reason = route.mode, route.reason
    else:
        mode, reason = "auto", "路由探测不可用，按默认策略执行"
    return RouteRecommendation(
        mode=mode,
        reason=reason,
        recommended_preset=preset,
        recommended_render_engine=render_engine,
        recommended_dual_mode=dual_mode,
        recommended_profile=profile,
        confidence=confidence,
        confidence_basis=basis,
    )


def _build_cost(
    config: UBTConfig,
    *,
    billable_blocks: int,
    source_chars: int,
    prefix: int,
    is_exact: bool,
    pages: int,
    scan_pages: int,
    chapters: int,
    route_mode: str,
    profile_name: str,
    warnings: list[AssessmentWarning],
    source_lang: str = "en",
    target_lang: str = "zh",
    estimated_tokens: int | None = None,
) -> CostQuote:
    """Draft (measured tooling) + config-driven fan-out, all labeled expected."""
    # Shared with the engine so the rollup gate cannot drift from the
    # policy that actually decides whether summaries are written.
    from ubt.core.engine.stages.draft import ACADEMIC_PROFILES

    # ``--pages``/UBT_PAGES restricts what ingest parses, so a sliced job
    # drafts a fraction of what this quote covers. Say so on the same screen
    # instead of letting the figure read as the cost of the slice.
    if config.get_selected_pages():
        warnings.append(
            AssessmentWarning(
                "PAGES_SLICE_QUOTED_WHOLE",
                "warn",
                "已设置 --pages 页码范围，但本报价按整本文档估算（未压缩到所选页）；"
                "实际草稿量只会更低。",
            )
        )

    batch_requested = bool(
        getattr(config, "offline_batch_enabled", False) or getattr(config, "batch_enabled", False)
    )
    # The discount is only real if the run can actually be batched. The engine
    # gates it on ``router.supports_batch_api`` — true only for the
    # chat/completions wire — and falls back to *interactive full price* when a
    # batch submission fails, so quoting it for a Responses-API profile cut the
    # draft line in half against a spend that never got the discount.
    batch_discount = (
        BATCH_API_DISCOUNT
        if batch_requested and str(getattr(config, "api_mode", "openai-chat")) == "openai-chat"
        else 1.0
    )
    if batch_requested and batch_discount >= 1.0:
        warnings.append(
            AssessmentWarning(
                "BATCH_DISCOUNT_UNAVAILABLE",
                "warn",
                "已请求批量 API，但当前 api_mode 不是 openai-chat（批处理仅支持 "
                "chat/completions 线路）；草稿按交互全价估算。",
            )
        )
    if batch_discount < 1.0:
        warnings.append(
            AssessmentWarning(
                "BATCH_DISCOUNT_APPLIED",
                "info",
                "已启用云端批量 API 模式（--offline-batch），草稿 Token 成本按 50% 离线折扣计算。",
            )
        )

    macro_chunk_size = max(1, getattr(config, "macro_chunk_size", 1))
    if batch_discount < 1.0 and macro_chunk_size > 1:
        # ``run_whole_book_batch`` builds one request *per block* — macro
        # chunking only applies to the interactive retry path — and every one of
        # those requests carries the full prompt prefix. Dividing the call count
        # by the chunk size here under-quoted the prefix by that factor.
        macro_chunk_size = 1
    draft = estimate_draft_cost_from_totals(
        billable_blocks=billable_blocks,
        source_chars=source_chars,
        draft_model=config.draft_model,
        prefix_tokens=prefix,
        macro_chunk_size=macro_chunk_size,
        batch_discount=batch_discount,
        base_url=config.base_url,
        source_lang=source_lang,
        target_lang=target_lang,
        source_tokens_override=estimated_tokens,
    )
    draft_calls = math.ceil(billable_blocks / macro_chunk_size) if billable_blocks else 0

    avg_block_chars = source_chars // max(billable_blocks, 1)
    if config.max_repair_rounds > 0 and billable_blocks:
        # Fan-out fraction derived from ``config.bottom_percentile`` (UBT_BOTTOM_PERCENTILE).
        # None defaults to REPAIR_DEFECT_FRACTION; explicit 0.0 disables repair passes.
        _bottom = getattr(config, "bottom_percentile", None)
        defect_fraction = REPAIR_DEFECT_FRACTION if _bottom is None else float(_bottom)
        # The short chain caps the rounds at 1 regardless of the configured
        # ceiling (stages/repair.py), so quoting ``max_repair_rounds`` there
        # priced repair passes that can never run.
        rounds = 1 if route_mode == "short" else config.max_repair_rounds
        repair_blocks = math.ceil(defect_fraction * billable_blocks) * rounds
        # Best-of-n repair only multiplies calls when the QE runner can
        # actually rank: the heuristic and tiered runners report
        # ``is_calibrated() == False``, so ``RepairLoop._rerank_enabled`` is
        # off for them and a ``rerank_k`` multiplier would over-quote the
        # repair line. When rerank *is* on, each of the k candidates carries
        # its own source span and emits its own completion, so both prompt and
        # completion sides scale with k.
        rerank_on = config.rerank_k > 1 and config.qe_engine in (
            "comet",
            "cometkiwi",
            "neural",
            "subprocess",
        )
        rerank_mult = config.rerank_k if rerank_on else 1
        repair_calls = repair_blocks * rerank_mult
        repair_tokens = (
            math.ceil((estimated_tokens * repair_blocks) / billable_blocks)
            if (estimated_tokens is not None and billable_blocks)
            else None
        )
        repair = estimate_draft_cost_from_totals(
            billable_blocks=repair_blocks * rerank_mult,
            source_chars=avg_block_chars * repair_blocks * rerank_mult,
            draft_model=config.repair_model,
            prefix_tokens=prefix,
            macro_chunk_size=1,
            base_url=config.base_url,
            source_lang=source_lang,
            target_lang=target_lang,
            source_tokens_override=repair_tokens,
        )

    else:
        repair_blocks = 0
        repair_calls = 0
        repair = estimate_draft_cost_from_totals(
            billable_blocks=0,
            source_chars=0,
            draft_model=config.repair_model,
            prefix_tokens=prefix,
            base_url=config.base_url,
            source_lang=source_lang,
            target_lang=target_lang,
        )

    qe_calls = 0
    qe_cost: float | None = None
    qe_model = getattr(config, "qe_judge_model", None) or config.repair_model
    # The LLM judge only ever runs when the pipeline can wrap it, and
    # ``PipelineOrchestrator`` wraps the *heuristic* runner alone
    # (``isinstance(self.qe_runner, HeuristicQERunner)``). On the subprocess/
    # COMET engine the flag is inert and scoring costs no tokens at all, so
    # quoting a judge there added a line the run can never spend.
    judge_can_run = config.qe_engine in ("heuristic", "tiered") and getattr(
        config, "qe_judge_enabled", False
    )
    if judge_can_run and billable_blocks:
        qe_calls = math.ceil(QE_GRAY_FRACTION * billable_blocks)
        avg_tokens = avg_block_chars * 2 // 4  # judge sees source + target
        # QE judge prompt is concise (~250 tokens instruction scaffolding), not the whole draft prompt prefix
        qe_cost = _price_component(
            qe_model,
            qe_calls * (250 + avg_tokens),
            qe_calls * 120,
            base_url=config.base_url,
        )

    # The visual judge ships rendered pages to a vision model; with the page
    # egress gate closed (the default) every call fails before the request and
    # is recorded as an info finding, so the quoted pages can never be billed.
    vlm_page_calls = (
        min(pages, config.visual_max_vlm_pages)
        if (config.visual_judge_enabled and config.allow_page_upload)
        else 0
    )
    ocr_page_calls = scan_pages if (scan_pages and config.ocr_mode in ("cloud", "vlm")) else 0
    # The two channels bill different models: the visual judge runs on
    # ``visual_judge_model`` while the OCR driver bills ``UBT_OCR_MODEL``. UBT
    # ships no OCR model default, so an unset one leaves the vision figure
    # unknown (warned below) rather than priced at a vendor's shipped model.
    vlm_model = getattr(config, "visual_judge_model", None) or config.repair_model
    if ocr_page_calls and not config.ocr_model.strip():
        warnings.append(
            AssessmentWarning(
                "OCR_MODEL_NOT_CONFIGURED",
                "warn",
                "OCR 已启用但未配置 ocr 模型：该通道记不到计费模型，费用呈现为「未知」。"
                "请设置 UBT_OCR_MODEL；vision 模式在未设置时会直接报错。",
            )
        )
    # The OCR channel bills its own endpoint (empty map value = the OpenAI
    # default, i.e. remote); the visual judge bills the router's ``base_url``.
    ocr_endpoint = config.remote_billing_models().get(config.ocr_model, config.base_url)
    vision_cost: float | None = None
    for calls, model, endpoint in (
        (vlm_page_calls, vlm_model, config.base_url),
        (ocr_page_calls, config.ocr_model, ocr_endpoint),
    ):
        if not calls:
            continue
        component = _price_component(
            model,
            calls * _VLM_PROMPT_TOKENS,
            calls * _VLM_COMPLETION_TOKENS,
            base_url=endpoint,
        )
        if component is None:
            # Unpriced channel: the whole quote's vision figure is unknown.
            vision_cost = None
            break
        vision_cost = (vision_cost or 0.0) + component

    # ``resolve_draft_policy`` disables rolling summaries for a single chapter,
    # for more than 40 of them, for academic/textbook profiles and for
    # page-sliced EPUBs; quoting a rollup call for those documents priced work
    # the engine provably never does.
    rollup_chapters_ok = 1 < chapters <= 40 and profile_name.lower() not in ACADEMIC_PROFILES
    rollup_calls = (
        chapters
        if (route_mode == "long" and config.enable_rolling_summary and rollup_chapters_ok)
        else 0
    )
    rollup_cost: float | None = None
    if rollup_calls:
        rollup_cost = _price_component(
            config.draft_model,
            rollup_calls * (prefix + 4000),
            rollup_calls * 400,
            base_url=config.base_url,
        )

    unpriced = [
        model
        for model, billed, endpoint in (
            (config.draft_model, billable_blocks > 0, config.base_url),
            (config.repair_model, repair_blocks > 0, config.base_url),
            (qe_model, qe_calls > 0, config.base_url),
            (vlm_model, vlm_page_calls > 0, config.base_url),
            # OCR bills its own model (UBT_OCR_MODEL) through its own endpoint.
            (config.ocr_model, ocr_page_calls > 0, ocr_endpoint),
            (config.draft_model, rollup_calls > 0, config.base_url),
        )
        if billed and model and not price_is_known(model, base_url=endpoint)
    ]
    if unpriced:
        warnings.append(
            AssessmentWarning(
                "MODEL_UNPRICED",
                "warn",
                f"模型 {'、'.join(sorted(set(unpriced)))} 无价格表条目、且不在自托管端点上，"
                "费用呈现为「未知」而非 $0；如需对外报价请在 ubt/core/router/pricing.py 增补单价。",
            )
        )

    components = [
        draft.cost_usd_uncached if billable_blocks else 0.0,
        repair.cost_usd_uncached if repair_blocks else 0.0,
        qe_cost if qe_calls else 0.0,
        vision_cost if (vlm_page_calls or ocr_page_calls) else 0.0,
        rollup_cost if rollup_calls else 0.0,
    ]
    total: float | None = None
    if all(c is not None for c in components):
        total = round(sum(c for c in components if c is not None), 5)

    return CostQuote(
        draft_model=config.draft_model,
        repair_model=config.repair_model,
        prefix_tokens_per_call=prefix,
        billable_blocks=billable_blocks,
        billable_blocks_is_exact=is_exact,
        prompt_tokens=draft.prompt_tokens,
        completion_tokens=draft.completion_tokens,
        draft_cost_usd_cached=draft.cost_usd_cached,
        draft_cost_usd_uncached=draft.cost_usd_uncached,
        repair_blocks=repair_blocks,
        repair_cost_usd=repair.cost_usd_uncached if repair_blocks else None,
        qe_calls=qe_calls,
        qe_cost_usd=qe_cost,
        vlm_page_calls=vlm_page_calls,
        ocr_page_calls=ocr_page_calls,
        vision_cost_usd=vision_cost,
        rollup_calls=rollup_calls,
        total_cost_usd=total,
        money_is_unknown=total is None,
        rollup_cost_usd=rollup_cost,
        draft_calls=draft_calls,
        repair_calls=repair_calls,
    )


def _runtime_estimate(total_calls: int, total_tokens: int, config: UBTConfig) -> RuntimeEstimate:
    """Wall-clock from the rate-limiter knobs only — nothing timing is persisted."""
    rpm_floor = total_calls / max(config.rate_limit_rpm / 60.0, 1e-9)
    tpm_floor = total_tokens / max(config.rate_limit_tpm / 60.0, 1e-9)
    low = max(rpm_floor, tpm_floor, total_calls / max(config.max_concurrency, 1) * 3.0)
    return RuntimeEstimate(
        heuristic=True,
        est_seconds_low=round(low, 1),
        est_seconds_high=round(low * 3, 1),
        basis=(
            "启发式：由 rate_limit_rpm/tpm 与 max_concurrency 推得（系统无历史耗时持久化）；"
            "上界按局部延迟波动 ×3。"
        ),
    )


def _synthesize_warnings(
    arch: Any,
    pdf: dict[str, Any],
    config: UBTConfig,
    route: RouteRecommendation,
    quality_signals: list[str],
) -> list[AssessmentWarning]:
    warnings: list[AssessmentWarning] = []
    if not config.draft_model.strip():
        warnings.append(
            AssessmentWarning(
                "MODEL_NOT_CONFIGURED",
                "warn",
                "未配置 draft 模型：报价按模型名未知处理，费用呈现为「未知」。"
                "请设置 UBT_DRAFT_MODEL、选择 provider（UBT_PROVIDER=...），"
                "或声明 [providers.<name>]。",
            )
        )
    share = pdf.get("scan_page_share")
    if (share is not None and share > 0.5) or (share is None and arch.is_scanned):
        warnings.append(
            AssessmentWarning(
                "SCANNED_PAGES_DOMINANT",
                "warn",
                "文档以扫描/图像页为主，译文依赖视觉转写，成本与时延显著上升；建议先做 OCR 预检或选 rigid 路线。",
            )
        )
    wit = pdf.get("witness")
    if wit and (wit["confirmed_pages"] or wit["at_risk_pages"]):
        warnings.append(
            AssessmentWarning(
                "FONT_RESIDUE_RISK",
                "warn",
                f"字体编码损伤：{wit['confirmed_pages']} 页确认残留、{wit['at_risk_pages']} 页高危"
                f"（共 {wit['residue_chars']} 个残字符），部分文字可能不可恢复，建议 rigid 或重新扫描。",
            )
        )
        quality_signals.append(
            f"字体 witness：confirmed {wit['confirmed_pages']} 页 / at-risk {wit['at_risk_pages']} 页"
        )
    effective_engine = (
        config.render_engine if config.render_engine != "auto" else route.recommended_render_engine
    )
    if effective_engine in RIGID_ENGINES and config.dual_mode != "monolingual":
        warnings.append(
            AssessmentWarning(
                "OVERLAY_CONFLICT",
                "warn",
                f"Rigid/Overlay 引擎仅支持 monolingual 输出，当前 --dual-mode '{config.dual_mode}' 将被降级；"
                "需要中英对照请改用 reflow (注意公式/图表排版风险)。",
            )
        )
    if arch.math_density == MathDensity.HIGH and config.formula_enrichment == "off":
        warnings.append(
            AssessmentWarning(
                "FORMULA_HEAVY_NEEDS_ENRICHMENT",
                "warn",
                "文档公式密集但视觉公式增强已关闭（--preset preview 的常见取舍），公式转写质量可能受限。",
            )
        )
    if pdf.get("max_rect_rows", 0) >= PATHOLOGICAL_RECT_ROWS:
        warnings.append(
            AssessmentWarning(
                "PATHOLOGICAL_PAGE_RISK",
                "warn",
                f"存在矩形行数达 {pdf['max_rect_rows']} 的异常页面，几何分析可能极慢（已知 O(n²) 缺陷），"
                "--deep 或完整翻译会明显耗时。",
            )
        )
    if pdf.get("text_layer_coverage") is not None:
        quality_signals.append(f"文本层覆盖率 {pdf['text_layer_coverage']:.0%}")
    if pdf.get("poster_pages"):
        quality_signals.append(f"海报/固定版式页 {pdf['poster_pages']} 页（建议 rigid）")
    if arch.detected_domain != "general":
        quality_signals.append(
            f"领域判定 {arch.detected_domain}（置信度 {arch.domain_confidence}），可挂载领域术语表"
        )
    return warnings


async def assess_document_async(
    path: Path | str,
    config: UBTConfig,
    *,
    deep: bool = False,
    target_lang: str = "zh",
    source_lang: str = "en",
) -> AssessmentReport:
    """Profile a cold document into a quote — async-native, safe for FastAPI / MCP event loops.

    Raises :class:`AssessmentError` only for missing/unsupported input; every
    individual probe failure degrades into an info warning so a report always
    comes out. ``deep`` runs the real adapter ingest (exact block counts) and
    can take minutes on large PDFs (docling model load).
    """
    started = time.monotonic()
    p = Path(path)
    if not p.is_file():
        raise AssessmentError("UNREADABLE", f"文件不存在或不是普通文件: {p}")
    if p.stat().st_size == 0:
        raise AssessmentError("EMPTY_FILE", f"文档大小为 0 字节，无法评估: {p}")

    known = {s.lstrip(".").lower() for s in supported_suffixes()}
    ext = p.suffix.lower().lstrip(".")
    if ext not in known:
        raise AssessmentError(
            "UNSUPPORTED", f"格式 .{ext} 不在适配器注册表 (支持: {', '.join(sorted(known))})"
        )

    warnings: list[AssessmentWarning] = []
    quality_signals: list[str] = []

    arch = await asyncio.to_thread(analyze_archetype, p)
    if ext == "pdf":
        route = await asyncio.to_thread(
            _safe,
            lambda: decide_route(
                p, short_max_pages=config.short_max_pages, exec_mode=config.exec_mode
            ),
            warnings,
            "ROUTE_PROBE_UNAVAILABLE",
            "路由探测失败，路线按默认呈现。",
        )
    else:
        route = await asyncio.to_thread(
            _safe,
            lambda: decide_route(
                doc_path=p, short_max_pages=config.short_max_pages, exec_mode=config.exec_mode
            ),
            warnings,
            "ROUTE_PROBE_UNAVAILABLE",
            "路由探测失败，路线按默认呈现。",
        )

    pdf: dict[str, Any] = {}
    if ext == "pdf":
        known_dims = (route.pages, route.chars) if route is not None else None
        pdf = await asyncio.to_thread(_pdf_facts, p, warnings, known_dims)

    pages = (
        int(pdf.get("probed_pages") or pdf.get("page_count") or (route.pages if route else 0))
        or arch.page_or_ch_count
    )
    if route is not None:
        chapters = route.chapters
        fallback_chars = route.chars
    else:
        chapters = (
            arch.page_or_ch_count
            if ext in ("epub", "md", "markdown", "txt", "html", "htm", "docx")
            else 1
        )
        fallback_chars = arch.sample_chars
    source_chars = int(pdf.get("probed_chars") or fallback_chars)

    # Scanned PDF with missing text layer: calibrate estimated chars if OCR is active.
    # (is_scanned is only ever true for PDFs — sample_document returns False for
    # every other format — so the share is simply absent outside the PDF route.)
    scan_share = pdf.get("scan_page_share") if ext == "pdf" else 0.0
    is_scanned_dominant = (scan_share is not None and scan_share > 0.5) or (
        scan_share is None and arch.is_scanned
    )
    if (
        is_scanned_dominant
        and pages > 0
        and source_chars < pages * 100
        and config.ocr_mode != "off"
    ):
        estimated_ocr_chars = pages * EXPECTED_SCANNED_PAGE_CHARS
        source_chars = max(source_chars, estimated_ocr_chars)
        warnings.append(
            AssessmentWarning(
                "SCANNED_PAGE_OCR_ESTIMATED",
                "info",
                f"检测到文档无有效文本层（纯扫描件或图像化排版），已按预估 OCR 转写规模 (~{EXPECTED_SCANNED_PAGE_CHARS}字/页，共 ~{source_chars} 字)"
                "估算草稿翻译成本；实际费用将取决于 OCR 识别出的文本量。",
            )
        )

    estimated_tokens = (
        route.estimated_tokens
        if (route is not None and not is_scanned_dominant)
        else source_chars // 4
    )

    if deep:
        try:
            billable, deep_chars = await _deep_blocks(p, config)
            billable_blocks, source_chars = billable, deep_chars or source_chars
            is_exact = True
        except Exception as exc:  # deep is best-effort precision
            logger.debug("assess deep ingest failed (%s); falling back to estimate", exc)
            warnings.append(
                AssessmentWarning(
                    "DEEP_INGEST_FAILED",
                    "info",
                    f"深度解析失败（{exc.__class__.__name__}），已回退快速估算分块数。",
                )
            )
            billable_blocks, is_exact = (
                max(chapters, math.ceil(source_chars / APPROX_BLOCK_CHARS)),
                False,
            )
    else:
        billable_blocks = max(chapters, math.ceil(source_chars / APPROX_BLOCK_CHARS))
        is_exact = False

    prefix = _measure_prefix_tokens_or_warn(
        config, warnings, target_lang=target_lang, source_lang=source_lang
    )
    rec = _recommend_route(arch, route, pdf, config)
    scan_pages = math.ceil((pdf.get("scan_page_share") or 0.0) * pages)
    cost = _build_cost(
        config,
        billable_blocks=billable_blocks,
        source_chars=source_chars,
        prefix=prefix,
        is_exact=is_exact,
        pages=pages,
        scan_pages=scan_pages,
        chapters=chapters,
        route_mode=rec.mode,
        # The profile the recommendation would run the job under, and which the
        # emitted next-step command now carries: ``resolve_draft_policy``
        # decides rolling summaries from it, so quoting rollups under any other
        # profile would price work the recommended run never does.
        profile_name=rec.recommended_profile,
        warnings=warnings,
        source_lang=source_lang,
        target_lang=target_lang,
        estimated_tokens=estimated_tokens,
    )

    draft_calls = cost.draft_calls
    total_calls = (
        draft_calls
        + cost.repair_calls
        + cost.qe_calls
        + cost.vlm_page_calls
        + cost.ocr_page_calls
        + cost.rollup_calls
    )
    total_tokens = cost.prompt_tokens + cost.completion_tokens
    runtime = _runtime_estimate(total_calls, total_tokens, config)
    warnings.extend(_synthesize_warnings(arch, pdf, config, rec, quality_signals))

    doc = DocumentFacts(
        file_name=p.name,
        file_size_bytes=p.stat().st_size,
        format_ext=ext,
        pages=pages,
        chapters=chapters,
        source_chars=source_chars,
        estimated_tokens=estimated_tokens,
        category=arch.category.value,
        detected_domain=arch.detected_domain,
        domain_confidence=arch.domain_confidence,
        math_density=arch.math_density.value,
        is_scanned=arch.is_scanned,
        primary_engine=pdf.get("primary_engine"),
        has_vector_diagrams=pdf.get("has_vector_diagrams"),
        has_multicolumn=pdf.get("has_multicolumn"),
        has_formulas=pdf.get("has_formulas"),
        text_layer_coverage=pdf.get("text_layer_coverage"),
        scan_page_share=pdf.get("scan_page_share"),
    )
    report = AssessmentReport(
        schema_version=SCHEMA_VERSION,
        status="ok",
        path=str(p),
        deep=deep,
        document=doc,
        route=rec,
        cost=cost,
        runtime=runtime,
        quality_signals=quality_signals,
        warnings=warnings,
        next_step_command="",  # filled by the CLI, which owns the user's argv
        meta={
            "duration_ms": int((time.monotonic() - started) * 1000),
            "exec_mode": config.exec_mode,
            "prompt_strategy": config.prompt_strategy,
            "ocr_mode": config.ocr_mode,
            "formula_enrichment": config.formula_enrichment,
            "provider": config.provider,
        },
    )
    return report


def assess_document(
    path: Path | str,
    config: UBTConfig,
    *,
    deep: bool = False,
    target_lang: str = "zh",
    source_lang: str = "en",
) -> AssessmentReport:
    """Profile a cold document into a quote — sync convenience wrapper.

    For callers running inside an active asyncio event loop (e.g. FastAPI / MCP),
    prefer calling :func:`assess_document_async` directly.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    coro = assess_document_async(
        path, config, deep=deep, target_lang=target_lang, source_lang=source_lang
    )
    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)
