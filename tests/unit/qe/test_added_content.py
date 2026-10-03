"""The 0-token added-content gate — the complement of the omission gate.

Two deterministic signals catch content a translation *gained*:

* a reference (citation / equation / figure / table number) the target cites but
  the source never mentions — an anchored callout the source lacked is direct
  evidence of fabrication, while a plain parenthesised quantity the source also
  carries is a legitimate rendering choice;
* a markdown heading in the target while the source carries none — the model
  reproducing prompt scaffolding.

Pinned here: the reference tokenizer (citations, ranges, keyword callouts, bare
parenthesised refs, Chinese section refs, docling spacing), the anchored-vs-all
distinction, heading extraction (CommonMark 0-3 space indent, setext excluded),
and the gate's fabrication/exemption/heading-leak verdicts.
"""

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


# --------------------------------------------------------------------------- #
# reference_tokens
# --------------------------------------------------------------------------- #


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
    # 100 - 25 = 75 > 50, so only the endpoints survive.
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


# --------------------------------------------------------------------------- #
# markdown_headings
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# AddedContentGate
# --------------------------------------------------------------------------- #


def test_a_clean_translation_passes() -> None:
    decision = AddedContentGate().evaluate("The value is 3.5", "\u503c\u662f 3.5")
    assert decision.passed is True
    assert decision.reason == "No added content detected"
    assert decision.fabricated_refs == ()
    assert decision.leaked_headings == ()


def test_empty_texts_pass() -> None:
    assert AddedContentGate().evaluate("", "").passed is True


def test_an_anchored_callout_the_source_lacks_is_fabricated() -> None:
    # The source carries "3.5" only as a bare quantity; turning it into a
    # figure callout is fabrication.
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
    # A source text with "in §8.4" translated as "在第 8.4 节" must not be flagged
    src = "In our evaluation in §8.4, DAMON with balloon free-page reporting reduces memory consumption by 21.2%."
    tgt = "在第 8.4 节的评估中，DAMON 结合 balloon 空闲页报告在未引入显著 CPU 开销的情况下，将内存消耗降低了 21.2%。"
    decision = AddedContentGate().evaluate(src, tgt)
    assert decision.passed is True
    assert "8.4" in decision.source_refs
    assert "8.4" in decision.target_refs
