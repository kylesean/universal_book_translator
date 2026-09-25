"""Quality presets: the single source of truth for user-facing policy bundles.

A preset is a deterministic bundle over *quality* parameters — prompt depth,
formula handling, exec mode. It never disables a correctness gate, and it
never picks the render route: "how much care to spend translating" and "how
to typeset the result" are orthogonal decisions, and bundling them made
``--preset publication`` a silent footgun on figure/table-heavy documents,
where the reflow route it forced shattered multi-row headers and dropped
vector figures (arXiv 2609.20519). The render engine stays with
``--render-engine`` and the ``auto`` dispatch in
:func:`ubt.core.policy.adaptive_policy.resolve_pdf_engine`.

Shared by the TUI wizard and the CLI ``--preset`` layer: the CLI
resolves explicit flags over the preset bundle over the engine defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ubt.core.config import (
    ExecMode,
    FormulaEnrichment,
    FormulaRender,
    MathBackend,
    PromptStrategyName,
    UBTConfig,
)


class Preset(StrEnum):
    """User-facing quality tiers."""

    PUBLICATION = "publication"
    STANDARD = "standard"
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
    math_backend: MathBackend
    formula_render: FormulaRender
    emit_both: bool

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
        description="智能路由（长/短链自适应） · 数学公式矢量高保真 ＋ 逐条视觉见证 · 双语/单语全覆盖",
        exec_mode="auto",
        prompt_strategy="rich",
        formula_enrichment="auto",
        math_backend="mathjax",
        formula_render="witness",
        emit_both=False,
    ),
    Preset.STANDARD: PresetPolicy(
        key=Preset.STANDARD,
        label="标准",
        tagline="质量与成本平衡：路由交给文档探测",
        description="自动短/长链 · 模型自适应提示 · 公式引擎渲染＋逐条见证回退",
        exec_mode="auto",
        prompt_strategy="auto",
        formula_enrichment="auto",
        math_backend="mathjax",
        formula_render="witness",
        emit_both=False,
    ),
    Preset.PREVIEW: PresetPolicy(
        key=Preset.PREVIEW,
        label="快速预览",
        tagline="省时省费：跳过公式视觉识别，公式直接用源图",
        description="自动路由 · 精简提示 · 公式源图保真（不重排）· 完整质量报告",
        exec_mode="auto",
        prompt_strategy="minimal",
        formula_enrichment="off",
        math_backend="image",
        formula_render="witness",
        emit_both=False,
    ),
    Preset.FAST: PresetPolicy(
        key=Preset.FAST,
        label="极速论文",
        tagline="极速开箱即用：零重型依赖，轻量几何抽取，短链快速翻译",
        description="短链极速执行 · 最小提示词 · 公式源图保真 · 零重型环境门槛",
        exec_mode="short",
        prompt_strategy="minimal",
        formula_enrichment="off",
        math_backend="image",
        formula_render="image",
        emit_both=False,
    ),
}


def preset_options() -> list[tuple[str, str, str]]:
    """Interactive-select options for the preset menu (key, label, description)."""
    return [
        (policy.key.value, policy.label, f"{policy.tagline}（{policy.description}）")
        for policy in PRESETS.values()
    ]


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


def apply_preset(config: UBTConfig, preset: Preset) -> UBTConfig:
    """Apply preset engine overrides onto a UBTConfig instance."""
    overrides = PRESETS[preset].engine_overrides()
    return config.model_copy(update=overrides)
