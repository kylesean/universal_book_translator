"""Unit tests for the Minimal AST Tagging & Self-Healing Decoder."""

from __future__ import annotations

import pytest

from ubt.core.ir.ast_protector import (
    DominantStyleTagger,
    SlotIntegrityValidator,
    TagSelfHealingDecoder,
)
from ubt.core.ir.models import InlineRun

pytestmark = pytest.mark.fast


def test_plain_text_has_zero_tags() -> None:
    text = "This is a clean sentence without special styling."
    payload = DominantStyleTagger.encode(text)
    assert payload.tagged_text == text
    assert payload.span_style_map == {}
    assert payload.slot_ids == ()

    decoded = TagSelfHealingDecoder.decode(payload.tagged_text, payload.span_style_map)
    assert decoded.clean_text == text
    assert decoded.inline_runs == ()


def test_dominant_style_zero_tag() -> None:
    # 90% plain, one bold word
    text = "This is an important parameter in the system."
    runs = [InlineRun(text="important", bold=True)]
    payload = DominantStyleTagger.encode(text, runs)

    assert "<s" in payload.tagged_text
    assert "<s1>important</s1>" in payload.tagged_text
    assert payload.base_style.bold is False

    # Simulate translation preserving the tag
    translated = "这是系统中的一个 <s1>重要</s1> 参数。"
    decoded = TagSelfHealingDecoder.decode(translated, payload.span_style_map, payload.base_style)
    assert decoded.clean_text == "这是系统中的一个 重要 参数。"
    assert len(decoded.inline_runs) == 1
    assert decoded.inline_runs[0].text == "重要"
    assert decoded.inline_runs[0].bold is True


def test_self_closing_slot_masking_for_math_and_citations() -> None:
    text = "Complexity is $\\mathcal{O}(N)$ as shown in [12]."
    payload = DominantStyleTagger.encode(
        text,
        math_slots=["$\\mathcal{O}(N)$"],
        citation_slots=["[12]"],
    )
    assert "<s1/>" in payload.tagged_text
    assert "<s2/>" in payload.tagged_text
    assert len(payload.slot_ids) == 2

    # Translated with slots moved or reordered
    translated = "正如 <s2/> 所示，时间复杂度为 <s1/>。"
    decoded = TagSelfHealingDecoder.decode(translated, payload.span_style_map, payload.base_style)
    assert decoded.clean_text == "正如 [12] 所示，时间复杂度为 $\\mathcal{O}(N)$。"
    assert decoded.slot_occurrences.get("s1") == 1
    assert decoded.slot_occurrences.get("s2") == 1

    report = SlotIntegrityValidator.validate(payload.slot_ids, decoded.slot_occurrences)
    assert report.is_valid is True
    assert report.missing_slots == ()
    assert report.hallucinated_slots == ()


def test_slot_integrity_detects_missing_and_hallucinated() -> None:
    expected = ("s1", "s2")
    # Translation dropped s2 and invented s3
    observed = {"s1": 1, "s3": 1}
    report = SlotIntegrityValidator.validate(expected, observed)
    assert report.is_valid is False
    assert report.missing_slots == ("s2",)
    assert report.hallucinated_slots == ("s3",)


def test_self_healing_decoder_recovers_unclosed_tag() -> None:
    span_map = {"s1": InlineRun(text="关键", bold=True)}
    base_style = InlineRun(text="", bold=False)

    # Model failed to close <s1>
    translated = "这是一个 <s1>关键特性 以及后续内容。"
    decoded = TagSelfHealingDecoder.decode(translated, span_map, base_style)
    assert decoded.clean_text == "这是一个 关键特性 以及后续内容。"
    assert len(decoded.inline_runs) == 1
    assert decoded.inline_runs[0].text == "关键特性 以及后续内容。"
    assert decoded.inline_runs[0].bold is True
    assert "s1" in decoded.unclosed_tags


def test_self_healing_decoder_ignores_unmatched_closing_tag() -> None:
    span_map = {"s1": InlineRun(text="重点", bold=True)}
    base_style = InlineRun(text="", bold=False)

    # Model emitted a stray </s99>
    translated = "前言 </s99> 随后 <s1>重点</s1> 完毕。"
    decoded = TagSelfHealingDecoder.decode(translated, span_map, base_style)
    assert decoded.clean_text == "前言  随后 重点 完毕。"
    assert len(decoded.inline_runs) == 1
    assert decoded.inline_runs[0].text == "重点"
    assert "s99" in decoded.unmatched_closes
