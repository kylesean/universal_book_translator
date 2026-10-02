"""The 0-token omission gate — dropped sentences, terms and truncated residue.

Three deterministic signals, each with a "must strictly exceed" floor and each
armed only when the source has enough signal to be meaningful:

* sentence-count ratio (armed at >= ``min_source_sentences``);
* proper-noun (identifier-shaped) recall (armed at >= ``min_identifier_terms``);
* char n-gram recall over verbatim residue (armed at >= ``min_chrf_ngrams``).

Pinned here: the acceptance ladder for identifier surfaces (exact -> plural/
singular -> LaTeX variant -> source-attested digit gluing -> source-attested
merge -> camel decomposition), the identifier/hyphen-ghost extraction rule, the
table exemption, and the gate's per-signal verdicts.
"""

from __future__ import annotations

from collections import Counter

import pytest

from ubt.core.qe.omission import (
    OmissionDecision,
    OmissionGate,
    OmissionMetrics,
    _char_ngrams,
    _digit_glued,
    _is_hyphen_ghost,
    _is_table_content,
    _match_term_surfaces,
    _math_term_variants,
    _merge_candidates,
    identifier_terms,
    singular_variant,
)

pytestmark = pytest.mark.fast


# --------------------------------------------------------------------------- #
# singular_variant
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("term", "expected"),
    [
        ("FinFETs", "FinFET"),
        ("cats", "cat"),
        ("class", None),  # ends with 'ss'
        ("bus", None),  # ends with 'us'
        ("axis", None),  # ends with 'is'
        ("s", None),  # too short
        ("as", None),  # too short
        ("cat", None),  # not plural
    ],
)
def test_singular_variant(term: str, expected: str | None) -> None:
    assert singular_variant(term) == expected


# --------------------------------------------------------------------------- #
# digit gluing / merge candidates
# --------------------------------------------------------------------------- #


def test_digit_glued_detects_a_trailing_digit() -> None:
    assert _digit_glued("Vtm", "x5 Vtm7") == (False, True)


def test_digit_glued_detects_a_leading_digit() -> None:
    assert _digit_glued("Vtm", "5Vtm x") == (True, False)


def test_digit_glued_is_false_without_gluing() -> None:
    assert _digit_glued("Vtm", "the Vtm here") == (False, False)


def test_merge_candidates_attest_a_left_merge() -> None:
    assert _merge_candidates("DS", "V DS") == ["VDS"]


def test_merge_candidates_without_a_neighbour_are_empty() -> None:
    assert _merge_candidates("DS", "a b") == []


# --------------------------------------------------------------------------- #
# hyphen ghost / identifier_terms
# --------------------------------------------------------------------------- #


def test_a_single_unhyphenated_run_is_a_ghost_of_a_repeated_hyphenated_run() -> None:
    freq = Counter({"NewtonRaphson": 1, "Newton-Raphson": 2})
    assert _is_hyphen_ghost("NewtonRaphson", freq) is True


def test_a_hyphenated_term_is_never_a_ghost() -> None:
    assert _is_hyphen_ghost("Newton-Raphson", Counter({"Newton-Raphson": 2})) is False


def test_a_one_off_run_is_not_a_ghost() -> None:
    assert _is_hyphen_ghost("NewtonRaphson", Counter({"Newton-Raphson": 1})) is False


def test_identifier_terms_keep_only_identifier_shaped_runs() -> None:
    assert identifier_terms("The FinFET device whereQ0") == {"FinFET", "whereQ0"}


def test_identifier_terms_drop_a_hyphen_ghost() -> None:
    text = "NewtonRaphson Newton-Raphson Newton-Raphson"
    assert identifier_terms(text) == set()


# --------------------------------------------------------------------------- #
# math variants
# --------------------------------------------------------------------------- #


def test_math_variants_for_an_underscore_term() -> None:
    assert _math_term_variants("V_ch") == ["V_{ch}", "V_{ch}", "V_ch", "v_{ch}"]


def test_math_variants_for_a_letter_digit_term() -> None:
    assert _math_term_variants("Q0") == ["Q_0", "Q_{0}"]


def test_math_variants_for_a_letter_word_term() -> None:
    assert _math_term_variants("Vch") == ["V_{ch}", "V_{CH}", "V_ch"]


# --------------------------------------------------------------------------- #
# _match_term_surfaces acceptance ladder
# --------------------------------------------------------------------------- #


def test_match_exact() -> None:
    assert _match_term_surfaces("FinFET", "the FinFET here") == ["FinFET"]


def test_match_singular_of_a_plural_source() -> None:
    assert _match_term_surfaces("FinFETs", "the FinFET here") == ["FinFET"]


def test_match_plural_of_a_singular_source() -> None:
    assert _match_term_surfaces("FinFET", "the FinFETs here") == ["FinFETs"]


def test_match_a_latex_variant() -> None:
    assert _match_term_surfaces("Vch", "see V_{ch} ok") == ["V_{ch}"]


def test_match_source_attested_digit_gluing() -> None:
    assert _match_term_surfaces("Vtm", "x5Vtm7", "x5Vtm7") == ["Vtm"]


def test_match_source_attested_merge() -> None:
    assert _match_term_surfaces("DS", "VDS", "V DS") == ["VDS"]


def test_match_camel_decomposition_parts() -> None:
    assert _match_term_surfaces("whereQ0", "where Q0", "whereQ0 text") == ["Q0"]


def test_a_fully_dropped_camel_term_fails() -> None:
    assert _match_term_surfaces("DataLoader", "nothing here", "DataLoader text") is None


def test_an_unmatched_term_is_none() -> None:
    assert _match_term_surfaces("FinFET", "无") is None


# --------------------------------------------------------------------------- #
# _char_ngrams / _is_table_content
# --------------------------------------------------------------------------- #


def test_char_ngrams_counts_each_size() -> None:
    assert _char_ngrams("abc", (2, 3)) == Counter({"ab": 1, "bc": 1, "abc": 1})


def test_char_ngrams_skips_sizes_longer_than_the_text() -> None:
    assert _char_ngrams("ab", (3,)) == Counter()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", False),
        ("plain text", False),
        ("| a | b |\n| c | d |", True),
        ("| a | b |\nplain", True),
        ("| a | b |\nx\ny\nz", False),
    ],
)
def test_is_table_content(text: str, expected: bool) -> None:
    assert _is_table_content(text) is expected


# --------------------------------------------------------------------------- #
# OmissionGate
# --------------------------------------------------------------------------- #


def test_the_metrics_dataclass_carries_all_signals() -> None:
    metrics = OmissionMetrics(4, 1, 0.25, 0.0, 1.0, 0.4)
    assert metrics.source_sentences == 4
    assert metrics.verbatim_chrf_recall == 0.4


def test_a_dropped_sentence_run_is_flagged() -> None:
    decision = OmissionGate().evaluate("One. Two. Three. Four.", "一句。")
    assert decision.passed is False
    assert decision.metrics.source_sentences == 4
    assert decision.metrics.target_sentences == 1
    assert decision.metrics.sentence_ratio == 0.25
    assert "Omission suspected" in decision.reason


def test_a_short_source_is_not_sentence_gated() -> None:
    # 2 sentences < min_source_sentences (3); the ratio (0.5) would otherwise
    # trip the floor, so this pins that the arming gate is what exempts it.
    decision = OmissionGate().evaluate("One. Two.", "一句。")
    assert decision.metrics.sentence_ratio == 0.5
    assert decision.passed is True


def test_a_table_is_exempt_from_the_sentence_signal() -> None:
    assert OmissionGate().evaluate("One. Two. Three. Four.", "一句。", is_table=True).passed


def test_a_table_block_type_is_detected() -> None:
    decision = OmissionGate().evaluate("One. Two. Three. Four.", "一句。", block_type="table")
    assert decision.passed is True


def test_table_content_is_auto_detected() -> None:
    source = "| a | b |\n| c | d |\n| e | f |"
    assert OmissionGate().evaluate(source, "| x | y |", is_table=None).passed is True


def test_missing_identifier_terms_are_flagged() -> None:
    decision = OmissionGate().evaluate("The FinFET and GPU run fast.", "它运行很快。")
    assert decision.passed is False
    assert decision.metrics.proper_noun_recall == 0.0
    assert "identifier term" in decision.reason


def test_a_single_identifier_term_does_not_gate() -> None:
    decision = OmissionGate().evaluate("The FinFET runs.", "它运行。")
    assert decision.passed is True


def test_present_identifier_terms_pass() -> None:
    decision = OmissionGate().evaluate("The FinFET and GPU run fast.", "FinFET 与 GPU 运行很快。")
    assert decision.passed is True
    assert decision.metrics.proper_noun_recall == 1.0


def test_truncated_verbatim_residue_is_flagged() -> None:
    decision = OmissionGate().evaluate(
        "Values 1234567 and 9876543 appear.", "数值 1234 与 9876 出现。"
    )
    assert decision.passed is False
    assert decision.metrics.verbatim_chrf_recall <= 0.7
    assert "verbatim residue" in decision.reason


def test_a_clean_block_passes_with_no_signals() -> None:
    decision = OmissionGate().evaluate("The FinFET runs fast.", "FinFET 运行很快。")
    assert decision.passed is True
    assert decision.reason == "No omission signals"


def test_decision_defaults() -> None:
    metrics = OmissionMetrics(0, 0, 0.0, 1.0, 1.0, 1.0)
    decision = OmissionDecision(passed=True, reason="ok", metrics=metrics)
    assert decision.metrics is metrics
