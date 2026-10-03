"""Document diagnostic and configuration advisor (presentation-free).

The archetype detection itself lives in ``ubt.core.archetype`` (shared with the
``ubt assess`` quote engine); this module adds only the recommendation layer.
It renders nothing and imports no presentation library, so every front end
(CLI pre-flight panel today, a web or desktop client later) can consume the
same advice over the same engine seams. The re-exports below keep
``from ubt.core.advisor import DocCategory`` working.
"""

from __future__ import annotations

import importlib.util
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from ubt.core.archetype import (
    DocCategory,
    MathDensity,
    analyze_archetype,
)

__all__ = [
    "DocCategory",
    "MathDensity",
    "analyze_archetype",
    "AdvisoryReport",
    "DocumentAdvisor",
]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ubt.core.assess import AssessmentReport
from ubt.core.config import (
    MOCK_API_KEY,
    RIGID_ENGINES,
    DualMode,
    ExecMode,
    FormulaEnrichment,
    FormulaMode,
    RenderEngine,
    UBTConfig,
)
from ubt.core.env import has_accelerator as _has_accelerator
from ubt.core.policy.adaptive_policy import resolve_render_engine_from_signals
from ubt.core.presets import Preset
from ubt.core.router_mode import decide as decide_route


@dataclass(frozen=True)
class AdvisoryReport:
    """Comprehensive diagnostic analysis and intelligent recommendations for a document."""

    file_path: Path
    file_name: str
    file_size_mb: float
    page_or_ch_count: int
    format_ext: str
    is_scanned: bool
    math_density: MathDensity
    detected_domain: str
    domain_confidence: float
    category: DocCategory

    # Hardware & environment capabilities
    has_gpu: bool
    has_typst: bool
    has_docling: bool
    api_ready: bool

    # Recommended execution parameters
    recommended_render_engine: RenderEngine
    recommended_dual_mode: DualMode
    recommended_formula_enrichment: FormulaEnrichment
    recommended_formula_mode: FormulaMode
    recommended_glossary: Path | None

    # Probing facts: route the engine will take and the tier we
    # suggest; both are read-only facts for the probe card, never questions.
    route_mode: str
    route_reason: str
    recommended_preset: Preset
    recommended_profile: str

    # Explanations and warnings
    reasons: list[str] = field(default_factory=list)
    conflict_warnings: list[str] = field(default_factory=list)
    assessment: AssessmentReport | None = None

    def check_conflict(self, chosen_render_engine: str, chosen_dual_mode: str) -> list[str]:
        """Check for physical layout or architectural conflicts if user overrides recommendations."""
        warnings: list[str] = []
        if chosen_render_engine in RIGID_ENGINES and chosen_dual_mode != "monolingual":
            warnings.append(
                f"【版式说明】Overlay 引擎以源页面为画布、只覆盖正文区域，当前仅支持纯目标语言 (monolingual) 输出；"
                f"请求的 '{chosen_dual_mode}' 将被降级为 monolingual。需要中英对照请改用 reflow 引擎。"
            )
        if (
            self.format_ext == "pdf"
            and self.recommended_render_engine == "rigid"
            and chosen_render_engine in ("reflow", "publication")
        ):
            warnings.append(
                f"【高密度学术排版风险】当前文档包含密集数学公式或二维结构 (math_density={self.math_density})，"
                f"系统推荐 'rigid' 原位引擎以 100% 保持交换图、公式与多栏坐标；强制使用 '{chosen_render_engine}' + "
                f"'{chosen_dual_mode}' 可能改变原页面几何布局并在公式密集段产生中英穿插顿挫。"
            )
        return warnings


class DocumentAdvisor:
    """Probes documents and environment to generate optimal translation strategies."""

    @classmethod
    def analyze(
        cls,
        file_path: Path | str,
        *,
        short_max_pages: int | None = None,
        exec_mode: ExecMode | None = None,
    ) -> AdvisoryReport:
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Target document not found: {path}")

        file_size_mb = round(path.stat().st_size / (1024 * 1024), 2)

        # Sample, classify math density, domain and category in one pass
        arch = analyze_archetype(path)
        format_ext = arch.format_ext
        page_or_ch_count = arch.page_or_ch_count
        is_scanned = arch.is_scanned
        math_density = arch.math_density
        detected_domain = arch.detected_domain
        domain_conf = arch.domain_confidence
        category = arch.category

        # Hardware and environment capabilities probe
        has_gpu = _has_accelerator()
        has_typst = shutil.which("typst") is not None
        has_docling = importlib.util.find_spec("docling") is not None

        config = UBTConfig.from_env()
        api_key_val = config.api_key.get_secret_value()
        api_ready = bool(api_key_val and api_key_val != MOCK_API_KEY)
        # Route probe uses the operator's thresholds (same call the pipeline
        # makes) instead of decide()'s import-time defaults.
        route_short_max = short_max_pages if short_max_pages is not None else config.short_max_pages
        route_exec_mode = exec_mode if exec_mode is not None else config.exec_mode

        # Recommendation synthesis
        reasons: list[str] = []
        conflict_warnings: list[str] = []

        # Route probe (also feeds the engine signal). Runs before the engine
        # recommendation so the page-level multi-column share can steer it.
        route_mode = "auto"
        route_reason = "结构化文档，无需 PDF 路由探测"
        route_multicolumn_share = 0.0
        route_structural_page_share = 0.0
        if format_ext == "pdf":
            try:
                route = decide_route(
                    path,
                    short_max_pages=route_short_max,
                    exec_mode=route_exec_mode,
                )
                route_mode = route.mode
                route_reason = route.reason
                route_multicolumn_share = float(
                    getattr(route, "multicolumn_page_share", 0.0) or 0.0
                )
                route_structural_page_share = float(
                    getattr(route, "structural_page_share", 0.0) or 0.0
                )
            except Exception:  # probing must never block the wizard
                route_mode = "auto"
                route_reason = "路由探测不可用，按默认策略执行"

        # 1. Render engine recommendation — predicted through the same canonical
        # resolver the runtime ``auto`` dispatch uses
        # (``adaptive_policy.resolve_render_engine_from_signals``), so the advice
        # matches the route that actually runs.
        if format_ext == "pdf":
            canonical = resolve_render_engine_from_signals(
                "auto",
                has_math=math_density == MathDensity.HIGH,
                struct_share=1.0 if is_scanned else 0.0,
                has_geometry=True,
                multicolumn_share=route_multicolumn_share,
                structural_page_share=route_structural_page_share,
                profile="paper" if category == DocCategory.ACADEMIC_PAPER else detected_domain,
                category=category,
            )
            if canonical == "rigid":
                recommended_render_engine: RenderEngine = "rigid"
                if is_scanned and math_density != MathDensity.HIGH:
                    reasons.append(
                        "检测到扫描版/无矢量文本 PDF：推荐原版位高保真引擎 (rigid)——保留原始页面扫描底图与图像题注位置。"
                    )
                else:
                    reasons.append(
                        "检测到密集数学公式/科技学术专著：推荐原版位高保真引擎 (rigid)——以原页面为底版，"
                        "公式与矢量图保持原位物理保真，彻底避免流式抽取导致的图表丢失与公式碎裂 (对标 arXiv 2609.20519 保真策略)。"
                    )
            else:
                recommended_render_engine = "reflow"
                reasons.append("推荐出版级流式重排引擎 (reflow)，输出高保真排版文档。")
                if not has_typst:
                    conflict_warnings.append(
                        "系统未检测到 typst 编译器，流水线将在导出阶段自动回退或建议安装 `curl -fsSL https://typst.community | sh`。"
                    )
        else:
            # Markdown / EPUB / DOCX
            recommended_render_engine = "reflow"
            reasons.append(f"对于 .{format_ext} 文档，采用原生结构化流式排版与重构。")

        # 2. Dual mode recommendation
        if format_ext == "pdf" and recommended_render_engine == "rigid":
            recommended_dual_mode: DualMode = "monolingual"
            reasons.append(
                "rigid（原位覆盖）引擎只产出纯目标语言译文（不做双语流式重排）；需要中英对照请改用 reflow 引擎。"
            )
        else:
            recommended_dual_mode = "inline"
            reasons.append(
                "双语版式推荐：段落级中英对照 (inline)，最适合连续自然语言叙事与双语精读对照。"
            )

        # 3. Formula enrichment recommendation
        if has_gpu:
            recommended_formula_enrichment: FormulaEnrichment = "auto"
            reasons.append(
                "硬件加速：检测到 CUDA / MPS 算力设备，自动激活 CodeFormulaV2 深度视觉公式识别。"
            )
        else:
            recommended_formula_enrichment = "off"
            reasons.append(
                "算力策略：未检测到 GPU 加速器，自动平滑降级视觉公式模型 (off)，采用轻量级 AST 规则提取，保障 CPU 流水线零阻塞流畅运行。"
            )

        recommended_formula_mode: FormulaMode = "readable"

        # 4. Route + tier + profile suggestions (route probed above)

        if math_density == MathDensity.HIGH or category in (
            DocCategory.ACADEMIC_PAPER,
            DocCategory.TECHNICAL_BOOK,
        ):
            recommended_preset = Preset.PUBLICATION
        else:
            recommended_preset = Preset.STANDARD

        recommended_profile = (
            "textbook"
            if math_density == MathDensity.HIGH
            or category in (DocCategory.ACADEMIC_PAPER, DocCategory.TECHNICAL_BOOK)
            else "general"
        )

        # 5. Domain glossary auto-binding
        recommended_glossary: Path | None = None
        if detected_domain != "general":
            glossary_candidate = (
                Path(__file__).resolve().parent.parent
                / "resources"
                / "glossaries"
                / detected_domain
                / "en-zh.json"
            )
            if glossary_candidate.exists():
                recommended_glossary = glossary_candidate
                reasons.append(
                    f"领域术语自动匹配：侦测到【{detected_domain}】专业领域，推荐挂载外部权威术语表 (`{glossary_candidate.name}`) 进行首译锁定。"
                )

        assessment: AssessmentReport | None = None
        try:
            from ubt.core.assess import assess_document

            assessment = assess_document(path, config)
        except Exception:
            assessment = None

        return AdvisoryReport(
            file_path=path,
            file_name=path.name,
            file_size_mb=file_size_mb,
            page_or_ch_count=page_or_ch_count,
            format_ext=format_ext,
            is_scanned=is_scanned,
            math_density=math_density,
            detected_domain=detected_domain,
            domain_confidence=domain_conf,
            category=category,
            has_gpu=has_gpu,
            has_typst=has_typst,
            has_docling=has_docling,
            api_ready=api_ready,
            recommended_render_engine=recommended_render_engine,
            recommended_dual_mode=recommended_dual_mode,
            recommended_formula_enrichment=recommended_formula_enrichment,
            recommended_formula_mode=recommended_formula_mode,
            recommended_glossary=recommended_glossary,
            route_mode=route_mode,
            route_reason=route_reason,
            recommended_preset=recommended_preset,
            recommended_profile=recommended_profile,
            reasons=reasons,
            conflict_warnings=conflict_warnings,
            assessment=assessment,
        )
