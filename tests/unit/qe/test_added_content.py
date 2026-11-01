from __future__ import annotations

import pytest

from ubt.core.qe.added_content import (
    AddedContentDecision,
    AddedContentGate,
    _reference_token_sets,
    markdown_headings,
    reference_tokens,
)

pytestmark = pytest.mark.fast


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("See [25]", {"25"}),
        ("[27,28]", {"27", "28"}),
        ("[25-28]", {"25", "26", "27", "28"}),
        ("Fig. 3.9", {"3.9"}),
        ("Figs. 3.14 and 3.15", {"3.14", "3.15"}),
        ("\u56fe 3.9", {"3.9"}),
        ("\u7b2c 3.2 \u8282", {"3.2"}),
        ("(3.11)", {"3.11"}),
        ("no references here", set()),
    ],
)
def test_reference_tokens(text: str, expected: set[str]) -> None:
    assert reference_tokens(text) == frozenset(expected)


def test_a_long_citation_range_is_not_expanded() -> None:
    assert reference_tokens("[25-100]") == frozenset({"25", "100"})


def test_docling_spaced_version_numbers_are_collapsed() -> None:
    assert reference_tokens("E q s . \\, ( 3 . 7 )") == frozenset({"3.7"})


def test_anchored_tokens_exclude_bare_parenthesised_numbers() -> None:
    all_tokens, anchored = _reference_token_sets("(3.5)")
    assert all_tokens == frozenset({"3.5"})
    assert anchored == frozenset()


@pytest.mark.parametrize("text", ["Fig. 3.5", "\u7b2c 3.2 \u8282", "[25]"])
def test_keyword_and_citation_tokens_are_anchored(text: str) -> None:
    all_tokens, anchored = _reference_token_sets(text)
    assert all_tokens == anchored


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("# Title", ("# Title",)),
        ("## Sub", ("## Sub",)),
        ("   ### Indented", ("### Indented",)),
        ("    #### four spaces", ()),
        ("####### seven hashes", ()),
        ("plain text", ()),
        ("Title\n=====", ()),
    ],
)
def test_markdown_headings(text: str, expected: tuple[str, ...]) -> None:
    assert markdown_headings(text) == expected


def test_headings_preserve_order() -> None:
    assert markdown_headings("# A\nbody\n## B") == ("# A", "## B")


def test_a_clean_translation_passes() -> None:
    decision = AddedContentGate().evaluate("The value is 3.5", "\u503c\u662f 3.5")
    assert decision.passed is True
    assert decision.reason == "No added content detected"
    assert decision.fabricated_refs == ()
    assert decision.leaked_headings == ()


def test_empty_texts_pass() -> None:
    assert AddedContentGate().evaluate("", "").passed is True


def test_an_anchored_callout_the_source_lacks_is_fabricated() -> None:
    decision = AddedContentGate().evaluate("The value is 3.5", "\u89c1\u56fe 3.5")
    assert decision.passed is False
    assert decision.fabricated_refs == ("3.5",)
    assert "Added reference(s)" in decision.reason


def test_a_plain_parenthesised_quantity_is_exempt() -> None:
    decision = AddedContentGate().evaluate("The value is 3.5", "\uff083.5\uff09\u500d")
    assert decision.passed is True


def test_a_fabricated_citation_is_flagged() -> None:
    decision = AddedContentGate().evaluate("no refs", "see [25]")
    assert decision.passed is False
    assert decision.fabricated_refs == ("25",)


def test_a_target_only_heading_is_a_scaffold_leak() -> None:
    decision = AddedContentGate().evaluate(
        "plain text", "### \u6e90\u6bb5\u843d\u7ffb\u8bd1\ncontent"
    )
    assert decision.passed is False
    assert decision.leaked_headings == ("### \u6e90\u6bb5\u843d\u7ffb\u8bd1",)
    assert "Prompt scaffold" in decision.reason


def test_a_heading_is_allowed_when_the_source_has_one() -> None:
    decision = AddedContentGate().evaluate("# Title\ncontent", "## \u6807\u9898\n\u5185\u5bb9")
    assert decision.passed is True


def test_a_chinese_section_ref_for_a_source_quantity_is_exempt() -> None:
    decision = AddedContentGate().evaluate("There are 1 items", "\u7b2c 1 \u7ae0")
    assert decision.passed is True


def test_the_decision_reports_both_reference_sets() -> None:
    decision = AddedContentGate().evaluate("See Fig. 3.9", "\u89c1\u56fe 3.9")
    assert decision.source_refs == frozenset({"3.9"})
    assert decision.target_refs == frozenset({"3.9"})
    assert decision.passed is True


def test_the_decision_defaults_are_empty() -> None:
    decision = AddedContentDecision(passed=True, reason="ok")
    assert decision.fabricated_refs == ()
    assert decision.leaked_headings == ()
    assert decision.source_refs == frozenset()
    assert decision.target_refs == frozenset()


def test_section_symbol_reference_is_not_fabricated() -> None:
    src = "In our evaluation in §8.4, DAMON with balloon free-page reporting reduces memory consumption by 21.2%."
    tgt = "在第 8.4 节的评估中，DAMON 结合 balloon 空闲页报告在未引入显著 CPU 开销的情况下，将内存消耗降低了 21.2%。"
    decision = AddedContentGate().evaluate(src, tgt)
    assert decision.passed is True
    assert "8.4" in decision.source_refs
    assert "8.4" in decision.target_refs


def test_a_translated_listing_callout_is_not_fabricated() -> None:
    decision = AddedContentGate().evaluate(
        "List. 1 shows a minimal container session through libdsec.",
        "列表 1 展示了通过 libdsec 实现的最小容器会话。",
    )
    assert decision.passed is True
    assert "1" in decision.source_refs
    assert "1" in decision.target_refs


def test_a_listing_keyword_without_a_number_is_not_a_reference() -> None:
    decision = AddedContentGate().evaluate("A list of items follows.", "接下来是项目列表。")
    assert decision.passed is True
    assert decision.source_refs == frozenset()


@pytest.mark.parametrize(
    "text",
    [
        "将模式3设置好",
        "模式3很常见",
        "格式2已定义",
        "方式1不同",
        "公式3成立",
        "样式4很好看",
        "形式2存在",
        "表达式2求值",
        "方程式1无解",
        "他代表3个部门",
        "地图3标注了边界",
        "手表2走时精准",
        "图表2说明了趋势",
    ],
)
def test_a_cjk_compound_tail_is_not_a_bare_keyword(text: str) -> None:
    assert reference_tokens(text) == frozenset()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("见图 3.5", {"3.5"}),
        ("见表 2", {"2"}),
        ("如表 3.1 所示", {"3.1"}),
        ("如式 (3.1) 所示", {"3.1"}),
        ("参见图 4", {"4"}),
        ("下图 2 展示了流程", {"2"}),
        ("图 3.14 和图 3.15", {"3.14", "3.15"}),
    ],
)
def test_a_genuine_cjk_callout_still_matches(text: str, expected: set[str]) -> None:
    assert reference_tokens(text) == frozenset(expected)


def test_a_mode_number_is_not_a_fabricated_reference() -> None:
    decision = AddedContentGate().evaluate("Type 3 is common", "模式3很常见")
    assert decision.passed is True
    assert decision.fabricated_refs == ()
