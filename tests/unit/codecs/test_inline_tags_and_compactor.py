"""Unit tests for inline anchor validation and overflow compaction loop."""

from __future__ import annotations

import pytest

from ubt.codecs.protocol import InlineToken, OverflowItem, TranslationUnit, UnitRole
from ubt.core.engine.overflow_compactor import compact_overflowing_units
from ubt.core.validators.inline_tags import strip_inline_tags, validate_inline_tags


@pytest.mark.fast
def test_validate_inline_tags_accepts_reordered_and_normalizes_case() -> None:
    src = "In <b1>Figure 2</b1>, <f1/> converges to <f2/> fast<n1/>."
    tgt = "在<b1>图 2</b1>中，<F2 /> 快速收敛于 <f1> <n1 />。"
    res = validate_inline_tags(src, tgt)
    assert res.valid is True
    assert "<f2/>" in res.normalized_text
    assert "<f1/>" in res.normalized_text
    assert "<n1/>" in res.normalized_text
    assert strip_inline_tags(res.normalized_text) == "在图 2中， 快速收敛于  。"


@pytest.mark.fast
def test_validate_inline_tags_catches_missing_rigid_and_bad_nesting() -> None:
    src = "Here <b1>bold <i2>both</i2></b1> with <f1/>."
    bad = "这里<b1>粗体<i2>交叉</b1></i2>丢了公式。"
    res = validate_inline_tags(src, bad)
    assert res.valid is False
    assert "f1" in res.missing_rigid
    assert any("mismatched nesting" in e for e in res.errors)


@pytest.mark.fast
def test_validate_inline_tags_rejects_hallucinated_tags() -> None:
    src = "Plain sentence with <f1/>."
    tgt = "普通句子带 <f1/> 和 <b9>多余标签</b9>。"
    res = validate_inline_tags(src, tgt)
    assert res.valid is False
    assert "b9" in res.extra_tags


@pytest.mark.fast
@pytest.mark.asyncio
async def test_overflow_compactor_accepts_shorter_valid_and_rejects_broken_tags() -> None:
    u1 = TranslationUnit(
        unit_id="u1",
        chapter_id="p1",
        role=UnitRole.PARAGRAPH,
        source_marked="Value <f1/> increases.",
        source_plain="Value x increases.",
        tokens=(InlineToken(tag_id="f1", kind="rigid"),),
    )
    u2 = TranslationUnit(
        unit_id="u2",
        chapter_id="p1",
        role=UnitRole.PARAGRAPH,
        source_marked="See <b1>Table 1</b1> for <f1/>.",
        source_plain="See Table 1 for y.",
        tokens=(
            InlineToken(tag_id="b1", kind="pair"),
            InlineToken(tag_id="f1", kind="rigid"),
        ),
    )
    overflows = [
        OverflowItem("u1", 1, 1.25, 10, 0.72, True),
        OverflowItem("u2", 1, 1.18, 10, 0.75, True),
    ]
    translations = {
        "u1": "数值 <f1/> 在实验过程中显著持续增加并且保持稳定趋势。",
        "u2": "请详细参阅 <b1>表 1</b1> 中关于 <f1/> 的完整统计数据说明。",
    }

    async def fake_llm(_sys: str, user: str) -> str:
        if "Value <f1/>" in user:
            return "数值 <f1/> 持续增加。"
        return "详见表 1 统计。"  # Dropped <b1> and <f1/> -> must be rejected!

    summary = await compact_overflowing_units(
        overflows=overflows,
        units_by_id={"u1": u1, "u2": u2},
        translations=translations,
        complete_fn=fake_llm,
        target_lang="zh",
    )
    assert summary.attempted == 2
    assert summary.accepted == 1
    assert summary.rejected_tag_violation == 1
    assert summary.compacted == {"u1": "数值 <f1/> 持续增加。"}


@pytest.mark.fast
def test_typesetter_handles_uppercase_rigid_and_ignores_stray_closing_formula_tag() -> None:
    from ubt.codecs.pdf.font_engine import FontEngine
    from ubt.codecs.pdf.formula_box import InlineFormulaBox
    from ubt.codecs.pdf.layout_fusion import ParagraphGeometry
    from ubt.codecs.pdf.paragraph_typesetter import typeset_paragraph

    geom = ParagraphGeometry(
        bbox=(72.0, 680.0, 400.0, 720.0),
        dominant_font_size=10.5,
        line_height=13.0,
        alignment="left",
        first_line_indent=0.0,
        is_serif=False,
        is_bold=False,
        color_rgb=(0.0, 0.0, 0.0),
        max_borrow_down_pt=0.0,
    )
    fbox = InlineFormulaBox(
        tag_id="f1",
        resource_name="/UBT_FX_0",
        width_pt=12.0,
        height_pt=10.0,
        ascent_pt=8.0,
        descent_pt=2.0,
        baseline_offset_pt=2.0,
        op_indices=(),
        rule_op_indices=(),
        bbox=(0.0, 0.0, 12.0, 10.0),
        xobject=None,
    )
    para = typeset_paragraph(
        unit_id="u1",
        page_no=1,
        role=UnitRole.PARAGRAPH,
        marked_text="公式 <F1 /> 与</f1>文本。",
        tokens=(InlineToken(tag_id="f1", kind="rigid"),),
        geom=geom,
        formulas={"f1": fbox},
        font_engine=FontEngine(target_lang="zh"),
        target_lang="zh",
    )
    formula_runs = [r for r in para.runs if r.formula is not None]
    assert len(formula_runs) == 1
    assert all("<F1" not in r.text and "</f1>" not in r.text for r in para.runs)


@pytest.mark.fast
def test_validate_and_typeset_unwrap_latex_wrapped_rigid_anchors() -> None:
    from ubt.codecs.pdf.font_engine import FontEngine
    from ubt.codecs.pdf.formula_box import InlineFormulaBox
    from ubt.codecs.pdf.layout_fusion import ParagraphGeometry
    from ubt.codecs.pdf.paragraph_typesetter import typeset_paragraph
    from ubt.core.validators.inline_tags import resolve_inline_tags_for_reflow

    src = "Where <f1/> is code at level <f2/> with <c1>link</c1> and <f3/> = 0 (<f4/> = <f5/>)."
    weak_model_out = (
        r"其中 $\text{<f1/>}$ 是层级 $<f2/>$ 的代码，见 <c1>链接</c1>，"
        r"且 $\text{<f3/>} = 0$（$\text{<f4/>} = \text{<f5/>}$）。"
    )
    res = validate_inline_tags(src, weak_model_out)
    assert res.valid is True
    assert (
        res.normalized_text
        == "其中 <f1/> 是层级 <f2/> 的代码，见 <c1>链接</c1>，且 <f3/> = 0（<f4/> = <f5/>）。"
    )
    assert (
        resolve_inline_tags_for_reflow(
            weak_model_out, {"f1": "r_k^i", "f2": "k", "f3": "α", "f4": "N", "f5": "L"}
        )
        == "其中 r_k^i 是层级 k 的代码，见 链接，且 α = 0（N = L）。"
    )

    # Even if un-normalized weak-model text reaches the typesetter directly, no `$\text{` leaks.
    geom = ParagraphGeometry(
        bbox=(72.0, 680.0, 400.0, 720.0),
        dominant_font_size=10.0,
        line_height=12.5,
        alignment="left",
        first_line_indent=0.0,
        is_serif=True,
        is_bold=False,
        color_rgb=(0.0, 0.0, 0.0),
        max_borrow_down_pt=0.0,
    )
    f1 = InlineFormulaBox(
        tag_id="f1",
        resource_name="/UBT_FX_0",
        width_pt=10.0,
        height_pt=9.0,
        ascent_pt=7.0,
        descent_pt=2.0,
        baseline_offset_pt=2.0,
        op_indices=(),
        rule_op_indices=(),
        bbox=(0.0, 0.0, 10.0, 9.0),
        xobject=None,
    )
    para = typeset_paragraph(
        unit_id="u2",
        page_no=1,
        role=UnitRole.PARAGRAPH,
        marked_text=r"其中 $\text{<f1/>}$ 成立。",
        tokens=(InlineToken(tag_id="f1", kind="rigid"),),
        geom=geom,
        formulas={"f1": f1},
        font_engine=FontEngine(target_lang="zh"),
        target_lang="zh",
    )
    rendered_txt = "".join(r.text for r in para.runs)
    assert r"\text" not in rendered_txt
    assert "$" not in rendered_txt
    assert sum(1 for r in para.runs if r.formula is not None) == 1
