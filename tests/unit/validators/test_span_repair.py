"""Contract tests for span-level error annotation and deterministic splicing."""

from __future__ import annotations

import pytest

from ubt.core.validators.span_repair import (
    SEVERITY_ORDER,
    MQMErrorSpan,
    MQMSpanAnnotator,
    SpanRepairSplicer,
    _is_cjk_expansion_blocked,
    max_severity,
    severity_for_error_type,
    span_to_dict,
)

pytestmark = pytest.mark.fast


# --------------------------------------------------------------------------- #
# Severity ladder
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("error_type", "expected"),
    [
        ("numeric", "critical"),
        ("terminology", "major"),
        ("terminology_leak", "major"),
        ("numeric_drift", "critical"),  # substring fallback
        ("Terminology Issue", "major"),  # case-insensitive substring
        ("other", "minor"),
        ("", "minor"),
    ],
)
def test_severity_for_error_type(error_type: str, expected: str) -> None:
    assert severity_for_error_type(error_type) == expected


def test_max_severity_picks_the_highest_tier() -> None:
    assert max_severity([]) == "minor"
    assert max_severity(["minor"]) == "minor"
    assert max_severity(["minor", "major"]) == "major"
    assert max_severity(["major", "critical"]) == "critical"
    assert max_severity(["critical", "minor"]) == "critical"
    assert max_severity(["bogus", "major"]) == "major"  # unknown treated as minor


def test_severity_order_is_a_total_ladder() -> None:
    assert SEVERITY_ORDER == {"minor": 0, "major": 1, "critical": 2}


def test_span_to_dict_round_trips_every_field() -> None:
    span = MQMErrorSpan("1", "numeric", "r", "5", 0, 1, "5", "critical")
    assert span_to_dict(span) == {
        "id": "1",
        "error_type": "numeric",
        "reason": "r",
        "expected": "5",
        "start_pos": 0,
        "end_pos": 1,
        "erroneous_text": "5",
        "severity": "critical",
    }


# --------------------------------------------------------------------------- #
# CJK compound guard
# --------------------------------------------------------------------------- #


def test_cjk_expansion_blocked_when_surface_is_cjk_flanked() -> None:
    # surface 网络 expands into 神经网络 and sits inside a CJK run.
    assert _is_cjk_expansion_blocked("网络层", 0, 2, "网络", "神经网络") is True
    assert _is_cjk_expansion_blocked("网网络", 1, 3, "网络", "神经网络") is True


def test_cjk_expansion_not_blocked_when_not_flanked() -> None:
    # Latin neighbour on the only side that exists.
    assert _is_cjk_expansion_blocked("网络x", 0, 2, "网络", "神经网络") is False


def test_cjk_expansion_requires_a_strict_expansion() -> None:
    # Same length / not a superset: never blocked.
    assert _is_cjk_expansion_blocked("abc", 0, 2, "ab", "abc") is False
    assert _is_cjk_expansion_blocked("网络", 0, 2, "网络", "网络") is False


# --------------------------------------------------------------------------- #
# MQMSpanAnnotator
# --------------------------------------------------------------------------- #


def test_annotate_empty_draft_is_a_noop() -> None:
    assert MQMSpanAnnotator().annotate_draft("x", "", []) == ("", [])


def test_annotate_without_defects_returns_the_draft() -> None:
    assert MQMSpanAnnotator().annotate_draft("x", "hello", []) == ("hello", [])


def test_annotate_alias_violation_wraps_the_surface() -> None:
    glossary = [{"source": "FinFET", "translation": "鳍式场效应晶体管", "aliases": ["finfet"]}]
    annotated, spans = MQMSpanAnnotator().annotate_draft("x", "the finfet here", [], glossary)
    assert spans == [
        MQMErrorSpan(
            id="1",
            error_type="terminology",
            severity="major",
            reason="Use standard terminology '鳍式场效应晶体管' instead of 'finfet'",
            expected="鳍式场效应晶体管",
            start_pos=4,
            end_pos=10,
            erroneous_text="finfet",
        )
    ]
    assert annotated == (
        'the <error_span id="1" type="terminology" '
        'reason="Use standard terminology &#x27;鳍式场效应晶体管&#x27; instead of &#x27;finfet&#x27;" '
        'expected="鳍式场效应晶体管">finfet</error_span> here'
    )


def test_annotate_source_leak_is_terminology_leak() -> None:
    glossary = [{"source": "FinFET", "translation": "鳍式场效应晶体管"}]
    _, spans = MQMSpanAnnotator().annotate_draft(
        "The FinFET is fast", "The FinFET is fast", [], glossary
    )
    assert len(spans) == 1
    assert spans[0].error_type == "terminology_leak"
    assert spans[0].severity == "major"
    assert spans[0].expected == "鳍式场效应晶体管"


def test_annotate_escapes_entities_in_attributes_and_content() -> None:
    glossary = [{"source": "R&D", "translation": "研发"}]
    annotated, _ = MQMSpanAnnotator().annotate_draft("the R&D team", "the R&D team", [], glossary)
    assert "R&amp;D" in annotated
    assert "&#x27;研发&#x27;" in annotated
    assert "R&D" not in annotated


def test_cjk_expansion_alias_is_not_annotated() -> None:
    glossary = [{"source": "NeuralNet", "translation": "神经网络", "aliases": ["网络"]}]
    annotated, spans = MQMSpanAnnotator().annotate_draft("x", "网络层", [], glossary)
    assert annotated == "网络层"
    assert spans == []


def test_numeric_mismatch_span_is_critical() -> None:
    annotated, spans = MQMSpanAnnotator().annotate_draft("value is 42", "value is 99", ["numeric"])
    assert spans == [
        MQMErrorSpan(
            id="1",
            error_type="numeric",
            severity="critical",
            reason="Mismatched number '99', expected '42'",
            expected="42",
            start_pos=9,
            end_pos=11,
            erroneous_text="99",
        )
    ]
    assert annotated == (
        'value is <error_span id="1" type="numeric" '
        'reason="Mismatched number &#x27;99&#x27;, expected &#x27;42&#x27;" expected="42">99</error_span>'
    )


def test_missing_source_number_is_anchored_at_end_of_text() -> None:
    annotated, spans = MQMSpanAnnotator().annotate_draft("value is 42", "the value", ["numeric"])
    assert len(spans) == 1
    assert spans[0].start_pos == spans[0].end_pos == len("the value")
    assert spans[0].reason == "Source number '42' missing from translation"
    assert annotated.endswith('expected="42">42</error_span>')


def test_numeric_span_ids_are_stable_across_multiple_defects() -> None:
    _, spans = MQMSpanAnnotator().annotate_draft("value is 42 and 7", "value is 99", ["numeric"])
    assert [(s.id, s.erroneous_text, s.start_pos, s.end_pos) for s in spans] == [
        ("1", "99", 9, 11),
        ("2", "42", 11, 11),
        ("3", "7", 11, 11),
    ]


def test_a_numeric_mismatch_is_not_flagged_without_the_flag() -> None:
    annotated, spans = MQMSpanAnnotator().annotate_draft("value is 42", "value is 99", [])
    assert annotated == "value is 99"
    assert spans == []


def test_the_numeric_flag_is_matched_case_insensitively() -> None:
    _, spans = MQMSpanAnnotator().annotate_draft("value is 42", "value is 99", ["NUMERIC"])
    assert [s.error_type for s in spans] == ["numeric"]


def test_a_correct_number_is_never_flagged() -> None:
    _, spans = MQMSpanAnnotator().annotate_draft("value is 42", "value is 42", ["numeric"])
    assert spans == []


def test_overlapping_alias_spans_are_collapsed_to_the_first() -> None:
    # '网络' matches at (0, 2); the longer '网络层' at (0, 3) overlaps and is dropped.
    glossary = [{"source": "NetworkTerm", "translation": "Y", "aliases": ["网络", "网络层"]}]
    annotated, spans = MQMSpanAnnotator().annotate_draft("x", "网络层", [], glossary)
    assert [(s.erroneous_text, s.start_pos, s.end_pos) for s in spans] == [("网络", 0, 2)]
    assert annotated.startswith('<error_span id="1"')
    assert annotated.endswith("层")


# --------------------------------------------------------------------------- #
# SpanRepairSplicer
# --------------------------------------------------------------------------- #


_ONE_SPAN = [MQMErrorSpan("1", "numeric", "r", "42", 9, 11, "99", "critical")]
_END_SPAN = [MQMErrorSpan("1", "numeric", "r", "42", 9, 9, "42", "critical")]


def test_splicer_without_spans_extracts_final_translation() -> None:
    assert SpanRepairSplicer().splice_repairs(
        "draft", [], "<final_translation>  hi  </final_translation>"
    ) == ("hi", False)


def test_splicer_without_spans_falls_back_to_raw_output() -> None:
    assert SpanRepairSplicer().splice_repairs("draft", [], "  raw output  ") == (
        "raw output",
        False,
    )


def test_splicer_in_place_replaces_by_offset() -> None:
    assert SpanRepairSplicer().splice_repairs(
        "value is 99", _ONE_SPAN, '<correction id="1">42</correction>'
    ) == ("value is 42", True)


def test_splicer_accepts_an_unquoted_correction_id() -> None:
    assert SpanRepairSplicer().splice_repairs(
        "value is 99", _ONE_SPAN, "<correction id=1>42</correction>"
    ) == ("value is 42", True)


def test_splicer_unescapes_correction_content() -> None:
    result, spliced = SpanRepairSplicer().splice_repairs(
        "a & b 99", _ONE_SPAN, '<correction id="1">x &amp; y</correction>'
    )
    assert (result, spliced) == ("a & b 99x & y", True)


def test_splicer_strips_nested_error_span_tags_from_a_correction() -> None:
    assert SpanRepairSplicer().splice_repairs(
        "value is 99", _ONE_SPAN, '<correction id="1"><error_span>42</error_span></correction>'
    ) == ("value is 42", True)


def test_splicer_prefers_final_translation_for_end_insertions() -> None:
    assert SpanRepairSplicer().splice_repairs(
        "the value", _END_SPAN, "<final_translation>the value 42</final_translation>"
    ) == ("the value 42", False)


def test_splicer_prefers_final_translation_over_a_large_end_replacement() -> None:
    big = "z" * 20
    output = f'<correction id="1">{big}</correction><final_translation>FINAL</final_translation>'
    assert SpanRepairSplicer().splice_repairs("the value", _END_SPAN, output) == ("FINAL", False)


def test_splicer_replaces_multiple_spans_right_to_left() -> None:
    spans = [
        MQMErrorSpan("1", "numeric", "r", "11", 2, 4, "11", "critical"),
        MQMErrorSpan("2", "numeric", "r", "22", 7, 9, "22", "critical"),
    ]
    output = '<correction id="1">X</correction><correction id="2">Y</correction>'
    assert SpanRepairSplicer().splice_repairs("a 11 b 22", spans, output) == ("a X b Y", True)


def test_splicer_never_blindly_appends_an_end_insertion() -> None:
    assert SpanRepairSplicer().splice_repairs(
        "the value", _END_SPAN, '<correction id="1">42</correction>'
    ) == ("the value", False)


def test_splicer_returns_a_large_end_replacement_wholesale() -> None:
    big = "z" * 20  # > 50% of the draft length
    assert SpanRepairSplicer().splice_repairs(
        "the value", _END_SPAN, f'<correction id="1">{big}</correction>'
    ) == (big, False)


def test_splicer_prefers_the_original_draft_over_unspliced_protocol() -> None:
    # A correction for an id that is not among the spans cannot be spliced; the
    # raw <correction> protocol must never ship as the translation.
    assert SpanRepairSplicer().splice_repairs(
        "draft", _END_SPAN, '<correction id="2">x</correction>'
    ) == ("draft", False)


def test_splicer_falls_back_to_final_translation_when_no_correction_splices() -> None:
    output = '<correction id="2">x</correction><final_translation>FINAL</final_translation>'
    assert SpanRepairSplicer().splice_repairs("value is 99", _ONE_SPAN, output) == ("FINAL", False)


def test_splicer_strips_residual_tags_and_unescapes() -> None:
    output = "<error_span>hi</error_span> <correction>y</correction>"
    assert SpanRepairSplicer().splice_repairs("the value", _END_SPAN, output) == ("hi y", False)
    assert SpanRepairSplicer().splice_repairs("the value", _END_SPAN, "raw &amp; out") == (
        "raw & out",
        False,
    )


def test_splicer_unescapes_entities_on_the_final_translation_path() -> None:
    assert SpanRepairSplicer().splice_repairs(
        "draft", [], "<final_translation>AT&amp;T</final_translation>"
    ) == ("AT&T", False)


def test_correction_pattern_tolerates_extra_attributes() -> None:
    output = '<correction id="1" confidence="0.9">42</correction>'
    assert SpanRepairSplicer().splice_repairs("value is 99", _ONE_SPAN, output) == (
        "value is 42",
        True,
    )
