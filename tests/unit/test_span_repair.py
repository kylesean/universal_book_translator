"""SpanRepairSplicer: infill a rejected span without rewriting the block.

Originally part of a pre-split phase-2 architecture module. The annotator half
of ubt/core/validators/span_repair.py is covered in test_mqm_triage.py; this is
the splice half, which previously had no canonical file.
"""

from __future__ import annotations

import pytest

from ubt.core.validators.span_repair import MQMErrorSpan, SpanRepairSplicer


def test_span_repair_splicer_targeted_infilling() -> None:
    """Verify that SpanRepairSplicer splices correction tags in-place with zero over-editing."""
    splicer = SpanRepairSplicer()

    original_draft = (
        "在这个体系中，所有的数据节点都依托于分散记账簿进行状态同步，从而保证网络高可用。"
    )
    spans = [
        MQMErrorSpan(
            id="1",
            error_type="terminology",
            reason="Use approved term",
            expected="分布式账本",
            start_pos=original_draft.find("分散记账簿"),
            end_pos=original_draft.find("分散记账簿") + len("分散记账簿"),
            erroneous_text="分散记账簿",
        )
    ]

    # Model returned minimal in-place correction tag
    model_output = '<correction id="1">分布式账本</correction>'

    repaired, was_spliced = splicer.splice_repairs(
        original_draft=original_draft,
        spans=spans,
        model_output=model_output,
    )

    assert was_spliced is True
    assert (
        repaired
        == "在这个体系中，所有的数据节点都依托于分布式账本进行状态同步，从而保证网络高可用。"
    )
    # Zero over-editing assertion: prefix and suffix must match 100%
    assert repaired.startswith("在这个体系中，所有的数据节点都依托于")
    assert repaired.endswith("进行状态同步，从而保证网络高可用。")


def test_span_repair_splicer_final_translation_fallback() -> None:
    """Verify that SpanRepairSplicer cleanly handles full-text fallback if model provides final_translation."""
    splicer = SpanRepairSplicer()

    original = "这是一篇存在缺陷的草稿。"
    spans = [
        MQMErrorSpan(
            id="1",
            error_type="fluency",
            reason="Unnatural",
            expected="",
            start_pos=0,
            end_pos=len(original),
            erroneous_text=original,
        )
    ]

    model_output = "<final_translation>这是经过深度润色修正后的标准译文。</final_translation>"

    repaired, was_spliced = splicer.splice_repairs(
        original_draft=original,
        spans=spans,
        model_output=model_output,
    )

    assert was_spliced is False
    assert repaired == "这是经过深度润色修正后的标准译文。"


def test_span_repair_splicer_missing_number_prefers_final_translation() -> None:
    """Verify that missing numbers anchored at the end do not blindly concatenate to sentence tail."""
    splicer = SpanRepairSplicer()
    original = "他有苹果和橙子。"
    spans = [
        MQMErrorSpan(
            id="1",
            error_type="numeric",
            reason="Source number '5' missing from translation",
            expected="5",
            start_pos=len(original),
            end_pos=len(original),
            erroneous_text="5",
        )
    ]

    # Model returned correction AND final_translation: must use final_translation, not "他有苹果和橙子。5"
    model_output = '<correction id="1">5</correction>\n<final_translation>他有5个苹果和橙子。</final_translation>'
    repaired, was_spliced = splicer.splice_repairs(
        original_draft=original,
        spans=spans,
        model_output=model_output,
    )
    assert was_spliced is False
    assert repaired == "他有5个苹果和橙子。"
    assert not repaired.endswith("。5")


def test_span_repair_splicer_unspliced_correction_never_ships_protocol() -> None:
    """An end-of-text insertion with no final_translation must not ship raw <correction> XML.

    The splice is deliberately skipped for end-of-text insertions (it would
    blindly append the number), so with no <final_translation> the only safe
    output is the unmodified draft — never the repair protocol itself.
    """
    splicer = SpanRepairSplicer()
    original = "他有苹果和橙子。"
    spans = [
        MQMErrorSpan(
            id="2",
            error_type="numeric",
            reason="Source number '42' missing from translation",
            expected="42",
            start_pos=len(original),
            end_pos=len(original),
            erroneous_text="42",
        )
    ]

    model_output = '<correction id="2">42</correction>'
    repaired, was_spliced = splicer.splice_repairs(
        original_draft=original,
        spans=spans,
        model_output=model_output,
    )

    assert was_spliced is False
    assert "<correction" not in repaired
    assert repaired == original


@pytest.mark.fast
def test_span_repair_unescapes_html_entities() -> None:
    splicer = SpanRepairSplicer()
    original_draft = "Using List<T> and stdio."
    span = MQMErrorSpan(
        id="e1",
        start_pos=6,
        end_pos=13,
        error_type="untranslated",
        reason="Needs translation",
        erroneous_text="List<T>",
        expected="List<T> & Vector<T>",
    )
    # LLM returned XML-escaped entities inside <correction>
    model_output = '<correction id="e1">List&lt;T&gt; &amp; Vector&lt;T&gt;</correction>'
    result, spliced = splicer.splice_repairs(original_draft, [span], model_output)
    assert spliced is True
    assert "&lt;" not in result, f"Escaped entity &lt; leaked: {result}"
    assert "&amp;" not in result, f"Escaped entity &amp; leaked: {result}"
    assert "Using List<T> & Vector<T> and stdio." in result
