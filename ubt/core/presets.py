"""Quality presets: the single source of truth for user-facing policy bundles.

A preset is a deterministic bundle over *quality* parameters — prompt depth,
formula handling, exec mode, and which PDF **parser** to use. It never disables
a correctness gate and never picks a **model**: "how much care to spend
translating" is a policy decision, while the model is the operator's.

``pdf_engine`` is in the bundle because parser choice *is* a quality decision
(pdfium is fast and geometry-only, docling is lossless and slow), and leaving it
out made ``--preset fast`` still pay for a docling conversion. What stays out is
the **render route**: every PDF is rebuilt from the semantic IR by the Typst
publication renderer, and nothing selects otherwise.

**Formula enrichment is not a quality gradient.** Only ``on`` runs the VLM math
transcription; both ``auto`` and ``off`` end up cropping the source region
(``resolve_formula_enrichment`` returns ``False`` for ``auto``, deliberately:
a faithful crop is exact and instant, while the VLM's LaTeX is a model's
transcription that can misread symbols). So ``publication`` and ``standard``
render identical math — they differ only in prompt depth — and advertising
"vector formula re-typesetting" for them was a promise no preset kept.
:attr:`Preset.PUBLICATION_MATH` is the preset that actually pays for it.

Shared by the advisor and the CLI ``--preset`` layer: the CLI
resolves explicit flags over the preset bundle over the engine defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ubt.core.config import (
    ExecMode,
    FormulaEnrichment,
    PromptStrategyName,
    QeEngine,
    TableRenderMode,
)


class Preset(StrEnum):
    """User-facing quality tiers."""

    #: Rich prompts on the lossless engine. ``formula_enrichment`` stays ``auto``,
    #: which resolves to the crop path — see the module docstring for why that is
    #: a property of ``auto`` and not a downgrade.
    PUBLICATION = "publication"
    #: Same ingest and formula policy as PUBLICATION; only the prompt depth
    #: differs (model-adaptive rather than forced rich).
    STANDARD = "standard"
    #: Rich prompts *and* ``formula_enrichment="on"``: the VLM transcribes each
    #: formula to vector LaTeX instead of the source being cropped. Much slower
    #: (roughly 15-60s per formula on CPU) and only worth it when the delivered
    #: PDF must scale its math.
    PUBLICATION_MATH = "publication_math"
    PREVIEW = "preview"
    FAST = "fast"


@dataclass(frozen=True)
class PresetPolicy:
    """Engine parameters a preset controls, with user-facing copy."""

    key: Preset
    label: str
    tagline: str
    description: str
    exec_mode: ExecMode
    prompt_strategy: PromptStrategyName
    formula_enrichment: FormulaEnrichment
    qe_engine: QeEngine
    table_render_mode: TableRenderMode
    emit_both: bool
    #: PDF parser the preset selects. ``auto`` is the lossless docling engine,
    #: unconditionally (the probe reports what the document contains; it no
    #: longer votes on the engine). ``pdfium`` is an explicit fast-path opt-in:
    #: it still escapes to docling for a scanned document, because pdfium cannot
    #: read one at all. An explicit ``--pdf-engine`` always beats this.
    pdf_engine: str

    def engine_overrides(self) -> dict[str, Any]:
        """Engine parameter overrides this preset contributes."""
        return {key: getattr(self, key) for key in PRESET_ENGINE_FIELDS}


# User-facing copy is not an engine parameter. Deriving the field list from the
# dataclass (rather than re-spelling it per caller) means a new preset knob
# cannot silently miss the preset layer in job_options/CLI.
_COPY_FIELDS = frozenset({"key", "label", "tagline", "description"})
PRESET_ENGINE_FIELDS: tuple[str, ...] = tuple(
    name for name in PresetPolicy.__dataclass_fields__ if name not in _COPY_FIELDS
)


PRESETS: dict[Preset, PresetPolicy] = {
    Preset.PUBLICATION: PresetPolicy(
        key=Preset.PUBLICATION,
        label="出版级",
        tagline="最佳质量：高保真术语 ＋ 全上下文提示，对标印刷交付",
        description="智能路由（长/短链自适应） · 强制富提示词 · 公式裁切源图保真 · 双语/单语全覆盖",
        exec_mode="auto",
        prompt_strategy="rich",
        formula_enrichment="auto",
        qe_engine="heuristic",
        table_render_mode="auto",
        emit_both=False,
        pdf_engine="auto",
    ),
    Preset.STANDARD: PresetPolicy(
        key=Preset.STANDARD,
        label="标准",
        tagline="质量与成本平衡：路由交给文档探测",
        description="自动短/长链 · 模型自适应提示 · 公式裁切源图保真（与出版级同一公式策略）",
        exec_mode="auto",
        prompt_strategy="auto",
        formula_enrichment="auto",
        qe_engine="heuristic",
        table_render_mode="auto",
        emit_both=False,
        pdf_engine="auto",
    ),
    Preset.PUBLICATION_MATH: PresetPolicy(
        key=Preset.PUBLICATION_MATH,
        label="出版级（公式矢量）",
        tagline="公式转为矢量 LaTeX，可无级缩放；以速度换保真",
        description="富提示词 ＋ 视觉模型逐条转写公式（每条约 15-60 秒 CPU）· 其余同出版级",
        exec_mode="auto",
        prompt_strategy="rich",
        formula_enrichment="on",
        qe_engine="heuristic",
        table_render_mode="auto",
        emit_both=False,
        pdf_engine="auto",
    ),
    Preset.PREVIEW: PresetPolicy(
        key=Preset.PREVIEW,
        label="快速预览",
        tagline="省时省费：跳过公式视觉识别，公式直接用源图",
        description="快速路径（pdfium，扫描件自动转 Docling）· 精简提示 · 公式源图保真（不重排）· 完整质量报告",
        exec_mode="auto",
        prompt_strategy="minimal",
        formula_enrichment="off",
        qe_engine="heuristic",
        table_render_mode="auto",
        emit_both=False,
        pdf_engine="pdfium",
    ),
    Preset.FAST: PresetPolicy(
        key=Preset.FAST,
        label="极速",
        tagline="极速开箱即用：零重型依赖，轻量几何抽取，短链快速翻译",
        description="快速路径（pdfium；扫描件无文本层，自动转 Docling）· 最小提示词 · 公式源图保真 · 零重型环境门槛",
        exec_mode="short",
        prompt_strategy="minimal",
        formula_enrichment="off",
        qe_engine="heuristic",
        table_render_mode="auto",
        emit_both=False,
        pdf_engine="pdfium",
    ),
}


def resolve_engine_params(
    preset: Preset | None,
    explicit: dict[str, Any],
) -> dict[str, Any]:
    """Resolve engine parameters: explicit user flags beat the preset bundle.

    ``explicit`` maps engine parameter names to the CLI values; a ``None``
    value means the flag was not passed, so the preset (or, without a preset,
    the engine default applied by the config layer) wins. Keys absent from
    the returned mapping are left to the engine default, which keeps an
    un-preset CLI invocation byte-identical to the pre-preset behaviour.
    """
    resolved = {key: value for key, value in explicit.items() if value is not None}
    if preset is not None:
        for key, value in PRESETS[preset].engine_overrides().items():
            resolved.setdefault(key, value)
    return resolved
