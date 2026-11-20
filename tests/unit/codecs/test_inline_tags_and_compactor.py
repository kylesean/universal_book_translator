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
