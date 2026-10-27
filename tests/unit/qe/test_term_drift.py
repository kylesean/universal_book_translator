"""The single glossary-drift detection primitive.

Five consumers used to decide independently whether a glossary term is drifted;
this module is now the one owner, and every hit comes from the rewriter's own
``find_term_occurrences`` so detection and enforcement cannot disagree.

The load-bearing facts pinned here:

* the source caliber is the strict one — ``[source] + aliases``, boundary-aware,
  case-folded, outside protected spans;
* the target is scanned only for entries the source carries, and asks whether
  the canonical rendering occurs under the same matcher (so ``drifted`` is
  exactly "source carries it, target does not render it");
* inflected variants satisfy the rendering check only when the canonical form is
  absent;
* target violations distinguish an ``alias`` from a ``leak`` (the untranslated
  source term), match case-**sensitively** (mirroring the rewriter), and skip
  surfaces that are approved globally as another entry's rendering.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

import pytest

from ubt.core.qe.term_drift import (
    GlossaryTerm,
    TermDriftFinding,
    detect_target_term_violations,
    detect_term_drift,
    normalize_glossary,
)

pytestmark = pytest.mark.fast


# --------------------------------------------------------------------------- #
# normalize_glossary
# --------------------------------------------------------------------------- #


def test_normalize_empty_glossary() -> None:
    assert normalize_glossary([]) == ()


def test_normalize_skips_entries_without_a_source_or_rendering() -> None:
    terms = normalize_glossary(
        [
            {"source": "", "translation": "x"},
            {"source": "x", "translation": ""},
            {"source": "  ", "translation": "  "},
            {"source": "kept", "translation": "留"},
        ]
    )
    assert [t.source for t in terms] == ["kept"]


def test_normalize_strips_and_deduplicates_aliases() -> None:
    (term,) = normalize_glossary(
        [{"source": "FinFET", "translation": "鳍", "aliases": [" finfet ", "finfet", "", None]}]
    )
    assert term.aliases == ("finfet",)


def test_normalize_coerces_non_string_aliases() -> None:
    (term,) = normalize_glossary([{"source": "X", "translation": "Y", "aliases": [123]}])
    assert term.aliases == ("123",)


def test_normalize_strips_and_filters_inflected_variants() -> None:
    (term,) = normalize_glossary(
        [
            {
                "source": "color",
                "translation": "颜色",
                "inflected_variants": [" colour ", "colour", "颜色", "", None],
            }
        ]
    )
    assert term.inflected_variants == ("colour",)


# --------------------------------------------------------------------------- #
# GlossaryTerm.source_surfaces
# --------------------------------------------------------------------------- #


def test_source_surfaces_are_source_plus_aliases_deduplicated() -> None:
    term = GlossaryTerm("FinFET", "鳍", aliases=("FinFET", "finfet", ""))
    assert term.source_surfaces == ("FinFET", "finfet")


def test_source_surfaces_without_aliases_is_just_the_source() -> None:
    assert GlossaryTerm("Solo", "独").source_surfaces == ("Solo",)


# --------------------------------------------------------------------------- #
# TermDriftFinding properties
# --------------------------------------------------------------------------- #


def _finding(
    source_hits: tuple[tuple[int, int], ...], expected_hits: tuple[tuple[int, int], ...]
) -> TermDriftFinding:
    return TermDriftFinding(
        source="s",
        expected="e",
        aliases=(),
        matched_surface="s" if source_hits else "",
        source_hits=source_hits,
        expected_hits=expected_hits,
    )


def test_finding_properties() -> None:
    rendered = _finding(((0, 1),), ((0, 1),))
    assert rendered.occurs_in_source is True
    assert rendered.rendered is True
    assert rendered.drifted is False

    drifted = _finding(((0, 1),), ())
    assert drifted.occurs_in_source is True
    assert drifted.rendered is False
    assert drifted.drifted is True

    absent = _finding((), ())
    assert absent.occurs_in_source is False
    assert absent.drifted is False


# --------------------------------------------------------------------------- #
# detect_term_drift
# --------------------------------------------------------------------------- #


_GLOSSARY = [{"source": "FinFET", "translation": "鳍式场效应晶体管", "aliases": ["finfet"]}]


def test_source_carrying_the_term_and_target_rendering_it_is_not_drifted() -> None:
    (finding,) = detect_term_drift("The FinFET device", "鳍式场效应晶体管器件", _GLOSSARY)
    assert finding.occurs_in_source is True
    assert finding.rendered is True
    assert finding.drifted is False
    assert finding.matched_surface == "FinFET"


def test_an_alias_in_the_source_still_carries_the_term() -> None:
    glossary = [{"source": "FinFET", "translation": "鳍式场效应晶体管", "aliases": ["FET"]}]
    (finding,) = detect_term_drift("the FET here", "no rendering", glossary)
    assert finding.occurs_in_source is True
    assert finding.matched_surface == "FET"
    assert finding.drifted is True


def test_the_source_surface_wins_over_an_alias() -> None:
    # The source surface is scanned first; case folding means the lowercase
    # alias spelling still resolves to the source surface.
    (finding,) = detect_term_drift("a finfet here", "no rendering", _GLOSSARY)
    assert finding.matched_surface == "FinFET"


def test_an_absent_term_is_never_scanned_in_the_target() -> None:
    (finding,) = detect_term_drift("nothing relevant", "鳍式场效应晶体管", _GLOSSARY)
    assert finding.occurs_in_source is False
    assert finding.expected_hits == ()
    assert finding.drifted is False


def test_detection_folds_case_on_both_sides() -> None:
    (finding,) = detect_term_drift("the finfet", "鳍式场效应晶体管", _GLOSSARY)
    assert finding.occurs_in_source is True
    assert finding.rendered is True


def test_protected_spans_hide_the_source_term() -> None:
    (finding,) = detect_term_drift("code `FinFET` here", "鳍式场效应晶体管", _GLOSSARY)
    assert finding.occurs_in_source is False


def test_protected_spans_hide_the_rendering() -> None:
    """A quoted rendering counts as rendered; a code span does not.

    The old rule asked only "is the rendering outside every protected span",
    and answered "not rendered" for anything the rewriter must not touch. That
    conflates two different protections. Measured on arXiv 2606.23344: the
    block "基于 “检测与分割可通过统一查询联合优化” 这一洞察" renders 检测
    correctly inside a *quoted phrase*, and was reported as terminology drift,
    sent to repair, and left for a human on a correct translation.

    A code span or a math region is different: the target there is untouched
    source by definition, so the rendering is not the translator's output.
    """
    # Quoted phrase — the translator's own wording, deliberately quoted.
    (quoted,) = detect_term_drift("The FinFET device", "基于 “鳍式场效应晶体管” 这一", _GLOSSARY)
    assert quoted.occurs_in_source is True
    assert quoted.rendered is True
    assert quoted.drifted is False
    assert quoted.rendered_protected_at is not None
    assert quoted.expected_hits == ()

    # Code span — verbatim source, so the rendering is not established.
    (coded,) = detect_term_drift("The FinFET device", "code `鳍式场效应晶体管` here", _GLOSSARY)
    assert coded.occurs_in_source is True
    assert coded.rendered is False
    assert coded.drifted is True


def test_an_inflected_variant_satisfies_the_rendering() -> None:
    glossary = [{"source": "color", "translation": "颜色", "inflected_variants": ["colour"]}]
    (finding,) = detect_term_drift("color space", "the colour space", glossary)
    assert finding.rendered is True
    assert finding.expected_hits == ((4, 10),)


def test_the_canonical_rendering_wins_over_an_inflected_variant() -> None:
    glossary = [{"source": "color", "translation": "颜色", "inflected_variants": ["colour"]}]
    (finding,) = detect_term_drift("color space", "颜色 and colour", glossary)
    assert finding.rendered is True
    assert len(finding.expected_hits) == 1


def test_pre_normalized_entries_are_accepted() -> None:
    entries = normalize_glossary(_GLOSSARY)
    (finding,) = detect_term_drift("The FinFET device", "鳍式场效应晶体管", entries)
    assert finding.occurs_in_source is True


def test_explicit_protected_spans_are_used() -> None:
    (finding,) = detect_term_drift(
        "The FinFET device",
        "鳍式场效应晶体管",
        _GLOSSARY,
        source_protected=[(4, 10)],
    )
    assert finding.occurs_in_source is False


def test_one_finding_per_entry_in_glossary_order() -> None:
    glossary = [
        {"source": "Alpha", "translation": "甲"},
        {"source": "Beta", "translation": "乙"},
    ]
    findings = detect_term_drift("Alpha and Beta", "甲 and 乙", glossary)
    assert [f.source for f in findings] == ["Alpha", "Beta"]


# --------------------------------------------------------------------------- #
# detect_target_term_violations
# --------------------------------------------------------------------------- #


def test_an_alias_in_the_target_is_a_violation() -> None:
    glossary = [{"source": "FinFET", "translation": "鳍式场效应晶体管", "aliases": ["finfet"]}]
    (violation,) = detect_target_term_violations("the finfet here", glossary)
    assert violation.surface == "finfet"
    assert violation.kind == "alias"


def test_the_untranslated_source_in_the_target_is_a_leak() -> None:
    glossary = [{"source": "FinFET", "translation": "鳍式场效应晶体管"}]
    (violation,) = detect_target_term_violations("the FinFET here", glossary)
    assert violation.surface == "FinFET"
    assert violation.kind == "leak"


def test_alias_and_leak_are_reported_in_order() -> None:
    glossary = [{"source": "FinFET", "translation": "鳍式场效应晶体管", "aliases": ["finfet"]}]
    violations = detect_target_term_violations("the finfet and FinFET here", glossary)
    assert [(v.surface, v.kind) for v in violations] == [("finfet", "alias"), ("FinFET", "leak")]


def test_a_short_non_uppercase_source_is_not_a_leak() -> None:
    glossary = [{"source": "ab", "translation": "甲乙"}]
    assert detect_target_term_violations("the ab here", glossary) == ()


def test_a_short_uppercase_source_is_a_leak() -> None:
    glossary = [{"source": "AI", "translation": "人工智能"}]
    (violation,) = detect_target_term_violations("the AI here", glossary)
    assert violation.kind == "leak"


def test_an_alias_equal_to_the_rendering_is_approved() -> None:
    glossary = [{"source": "X", "translation": "Y", "aliases": ["Y"]}]
    assert detect_target_term_violations("Y", glossary) == ()


def test_an_alias_that_is_another_entrys_rendering_is_approved_globally() -> None:
    glossary: list[dict[str, Any]] = [
        {"source": "Alpha", "translation": "甲", "aliases": ["乙"]},
        {"source": "Beta", "translation": "乙"},
    ]
    assert detect_target_term_violations("乙", glossary) == ()


def test_matching_is_case_sensitive() -> None:
    glossary = [{"source": "X", "translation": "Y", "aliases": ["foo"]}]
    assert detect_target_term_violations("FOO", glossary) == ()


def test_protected_spans_hide_target_violations() -> None:
    glossary = [{"source": "FinFET", "translation": "鳍式场效应晶体管"}]
    assert detect_target_term_violations("`FinFET`", glossary) == ()


def test_acronym_alias_matches_case_sensitively() -> None:
    # An acronym alias like "BE" for "best-effort" must not match the common English verb "be"
    glossary = [{"source": "best-effort", "translation": "BE尽力而为", "aliases": ["BE"]}]
    findings = detect_term_drift(
        "This model can be used for translation.", "该模型可用于翻译。", glossary
    )
    assert not findings[0].occurs_in_source
    assert not findings[0].drifted

    # But it must match uppercase "BE"
    findings_be = detect_term_drift(
        "For BE tasks, we need...", "对于普通任务，我们需要...", glossary
    )
    assert findings_be[0].occurs_in_source
    assert findings_be[0].drifted


# --------------------------------------------------------------------------- #
# replace_term_surface — the workbench's "replace with recommended term" action
# --------------------------------------------------------------------------- #


def test_replace_term_surface_rewrites_every_alias_occurrence() -> None:
    from ubt.core.qe.term_drift import replace_term_surface

    glossary = [{"source": "FinFET", "translation": "鳍式场效应晶体管", "aliases": ["finfet"]}]
    rewritten, count = replace_term_surface(
        "a finfet and another finfet", glossary, surface="finfet", expected="鳍式场效应晶体管"
    )
    assert count == 2
    assert rewritten == "a 鳍式场效应晶体管 and another 鳍式场效应晶体管"


def test_replace_term_surface_rewrites_a_leaked_source_term() -> None:
    from ubt.core.qe.term_drift import replace_term_surface

    glossary = [{"source": "FinFET", "translation": "鳍式场效应晶体管"}]
    rewritten, count = replace_term_surface(
        "the FinFET device", glossary, surface="FinFET", expected="鳍式场效应晶体管"
    )
    assert count == 1
    assert rewritten == "the 鳍式场效应晶体管 device"


def test_replace_term_surface_leaves_protected_spans_alone() -> None:
    from ubt.core.qe.term_drift import replace_term_surface

    glossary = [{"source": "FinFET", "translation": "鳍式场效应晶体管"}]
    # The detector skips inline code, so the rewriter must too — otherwise a
    # human "fix" would corrupt a code span the exporter deliberately left.
    rewritten, count = replace_term_surface(
        "`FinFET` and FinFET", glossary, surface="FinFET", expected="鳍式场效应晶体管"
    )
    assert count == 1
    assert rewritten == "`FinFET` and 鳍式场效应晶体管"


def test_replace_term_surface_is_a_noop_without_the_surface() -> None:
    from ubt.core.qe.term_drift import replace_term_surface

    glossary = [{"source": "FinFET", "translation": "鳍式场效应晶体管"}]
    assert replace_term_surface(
        "clean text", glossary, surface="FinFET", expected="鳍式场效应晶体管"
    ) == (
        "clean text",
        0,
    )
    # surface == expected is not a rewrite.
    assert replace_term_surface("FinFET", glossary, surface="FinFET", expected="FinFET") == (
        "FinFET",
        0,
    )


# --------------------------------------------------------------------------- #
# Two false positives measured on a real delivery (arXiv 2606.23344)
# --------------------------------------------------------------------------- #


def test_a_correct_synonym_is_not_a_violation() -> None:
    """A term rendered naturally is not drift, even under a listed alias.

    Measured: the glossary asked for 公开可用, the translator wrote 公开获取 —
    correct Chinese for "publicly available", and no listed alias matched, so the
    block was reported as terminologically drifted on a correct translation. The
    detection primitive cannot see the block's *meaning*; the minimax rule in
    :func:`ubt.core.qe.term_drift.detect_target_term_violations` is what keeps such
    a block out of the human queue. This test pins that it is a *violation* scan
    question, not a drift-scan one: the drift finding still reports the rendering
    as non-canonical, because it answers a different question.
    """
    glossary = [{"source": "Publicly Available", "translation": "公开可用"}]
    (finding,) = detect_term_drift(
        "The weights are publicly available.",
        "模型权重可公开获取。",
        normalize_glossary(glossary),
    )
    assert finding.occurs_in_source is True
    assert finding.rendered is False  # not the canonical surface — true, not a verdict
    # ...but it is not something the rewriter should touch either, and the block
    # is not accused of carrying a forbidden surface.
    assert detect_target_term_violations("模型权重可公开获取。", glossary) == ()


def test_a_rendering_inside_a_quoted_phrase_counts_as_rendered() -> None:
    """The translator quoted the phrase it translated; that is not drift.

    Measured: "基于 “检测与分割可通过统一查询联合优化” 这一洞察" renders 检测
    correctly, inside a quoted phrase. The matcher skips quoted spans (the
    rewriter must not touch them), so the term read as missing.
    """
    glossary = [{"source": "Detection", "translation": "检测"}]
    (finding,) = detect_term_drift(
        "detection and segmentation can be jointly optimized",
        "基于 “检测与分割可通过统一查询联合优化” 这一洞察",
        normalize_glossary(glossary),
    )
    assert finding.occurs_in_source is True
    assert finding.rendered is True
    assert finding.drifted is False
    assert finding.rendered_protected_at is not None


def test_a_rendering_inside_a_code_span_does_not_count() -> None:
    """A code span is verbatim source; a rendering there proves nothing."""
    glossary = [{"source": "Detection", "translation": "检测"}]
    (finding,) = detect_term_drift(
        "detection here",
        "run `检测` to see",
        normalize_glossary(glossary),
    )
    assert finding.rendered is False
    assert finding.drifted is True


def test_an_alias_beside_the_canonical_rendering_is_not_a_violation() -> None:
    """Using the canonical term and also naming a synonym is not drifting from it."""
    glossary = [{"source": "edge computing", "translation": "边缘计算", "aliases": ["边缘运算"]}]
    text = "边缘计算与边缘运算同义。"
    assert detect_target_term_violations(text, glossary) == ()


def test_a_leak_is_reported_even_when_the_rendering_is_present() -> None:
    """The untranslated source term surviving is a defect whatever else is there."""
    glossary = [{"source": "edge computing", "translation": "边缘计算"}]
    violations = detect_target_term_violations("边缘计算（edge computing）", glossary)
    assert [v.kind for v in violations] == ["leak"]


# --------------------------------------------------------------------------- #
# Two more false positives from the same delivery
# --------------------------------------------------------------------------- #


def test_a_triangle_is_not_a_number() -> None:
    """上三角 (upper triangular) must not become 上3角 under the numeric view.

    Measured: the block "其中 𝑃 表示有效上三角元素对（不含对角线）" was rejected
    with "Unauthorized numeric tokens: ['3']". 角 was listed as a measure word,
    so the numeral normaliser rewrote the 三 of 三角 (triangle) into 3 and the
    invention gate saw a fabricated number in a correct translation. 分/度/面/步
    were already excluded for the same reason; 角 joins them.
    """
    from ubt.core.validators.consistency import normalize_for_numeric_matching

    assert normalize_for_numeric_matching("有效上三角元素对") == "有效上三角元素对"
    assert "3" not in normalize_for_numeric_matching("三角矩阵与上三角部分")


def test_real_quantities_still_fold() -> None:
    """The fix must not blunt the normaliser it lives in."""
    from ubt.core.validators.consistency import normalize_for_numeric_matching

    assert normalize_for_numeric_matching("五项") == "5项"
    assert normalize_for_numeric_matching("第七章") == "第7章"
    assert normalize_for_numeric_matching("三次迭代") == "3次迭代"
    assert normalize_for_numeric_matching("两倍") == "2倍"


def test_a_heading_of_pure_identifiers_is_not_an_echo() -> None:
    """There was nothing to translate, so keeping the source is not a failure.

    Measured: "4.2.1. OmniDocBench v1.5" and "4.2.2. Real5-OmniDocBench" were
    both rejected as untranslated echoes, repaired, and left for a human on
    every run. A section number plus a model name has no prose in it.
    """
    from ubt.core.qe.fast_pass import is_verbatim_echo

    assert not is_verbatim_echo("4.2.1. OmniDocBench v1.5", "4.2.1. OmniDocBench v1.5")
    assert not is_verbatim_echo("4.2.2. Real5-OmniDocBench", "4.2.2. Real5-OmniDocBench")
    assert not is_verbatim_echo("PP-DocLayoutV3", "PP-DocLayoutV3")


def test_an_untranslated_heading_is_still_an_echo() -> None:
    """The near miss: a heading that DOES contain prose must still be caught."""
    from ubt.core.qe.fast_pass import is_verbatim_echo

    assert is_verbatim_echo("3. Proposed Approach", "3. Proposed Approach")
    assert is_verbatim_echo(
        "This paper proposes a new method for layout analysis.",
        "This paper proposes a new method for layout analysis.",
    )


# --------------------------------------------------------------------------- #
# Numeric fidelity: two correct translations accused of dropping numbers
# --------------------------------------------------------------------------- #


def test_a_count_suffix_may_be_written_out_in_full() -> None:
    """ "over 38k samples" == "3.8 万份样本" — the same number, spelled differently.

    Measured on arXiv 2606.23344: the check is textual, so it looked for '38'
    and found '3.8'. The suffix states the magnitude, so the value is 38000 on
    both sides.
    """
    from ubt.core.validators.consistency import NumericConsistencyValidator

    validator = NumericConsistencyValidator()
    assert validator.validate(
        "we scaled our training corpus to over 38k high-quality document samples.",
        "我们将训练语料库扩展至超过 3.8 万份高质量文档样本。",
    ).is_valid
    # The suffix itself surviving is accepted too.
    assert validator.validate("over 38k samples", "超过 38k 个样本").is_valid


def test_a_count_suffix_is_not_an_identifier_or_a_unit() -> None:
    """The near miss: only a real count gets the magnitude reading."""
    from ubt.core.validators.consistency import _count_suffix_values

    assert _count_suffix_values("PP-DocLayout_plus-L") == {}
    assert _count_suffix_values("a 38kb file") == {}
    assert _count_suffix_values("over 38k samples") == {"38": {"38000"}}


def test_dropping_a_count_is_still_a_failure() -> None:
    """Authorizing 38000 must not authorize dropping the quantity."""
    from ubt.core.validators.consistency import NumericConsistencyValidator

    assert (
        not NumericConsistencyValidator()
        .validate(
            "we scaled our corpus to over 38k samples.",
            "我们扩大了训练语料库。",
        )
        .is_valid
    )


def test_a_scientific_exponent_is_not_a_subtraction() -> None:
    """2 × 10 -4 == 2 × 10⁻⁴; the superscript form has no minus to preserve.

    Measured on arXiv 2606.23344: the flat exponent in the source made the
    "dropped minus sign changes the fact" rule reject a correct translation
    that rendered the same number as a superscript.
    """
    from ubt.core.validators.consistency import NumericConsistencyValidator, _is_scientific_exponent

    assert _is_scientific_exponent("2 × 10 -4 to", 8) is True
    assert _is_scientific_exponent("2 × 10^-4", 8) is True
    # A real subtraction stays a subtraction.
    assert _is_scientific_exponent("a range of -4 to 5", 12) is False
    assert _is_scientific_exponent("impact: -4%", 9) is False

    validator = NumericConsistencyValidator()
    assert validator.validate(
        "The learning rate is set to a constant 2 × 10 -4 to ensure stable convergence.",
        "学习率固定为 2 × 10⁻⁴，以确保稳定收敛。",
    ).is_valid
    # A genuinely dropped minus is still caught.
    assert not validator.validate("the temperature is -4 degrees", "温度是 4 度").is_valid


# --------------------------------------------------------------------------- #
# A term that only ever appears as a label
# --------------------------------------------------------------------------- #


def test_a_term_kept_only_as_a_table_label_is_not_drift() -> None:
    """``bbox+mask`` is a label the source wrote; translating it breaks the row.

    Measured on arXiv 2606.23344 (b0099): the glossary asks Mask -> 掩码 and the
    term occurs four times in the source, every time as the cell ``bbox+mask``.
    The translation renders every prose cell around it (Scan -> 扫描, Warp ->
    形变) and keeps the four labels verbatim, and the gate counted that as four
    missed renderings.
    """
    glossary = [{"source": "Mask", "translation": "掩码"}]
    source = "| bbox+mask | XYCut | 93.81 |\n| bbox+mask | Decoupled | 93.87 |"
    target = "| bbox+mask | XYCut | 93.81 |\n| bbox+mask | 解耦 | 93.87 |"
    (finding,) = detect_term_drift(source, target, normalize_glossary(glossary))
    assert finding.occurs_in_source is True
    assert finding.kept_as_label is True
    assert finding.drifted is False


def test_a_term_in_a_formula_is_not_drift() -> None:
    """``\\lambda_{\\text{mask}}`` is reproduced verbatim; it is not a miss.

    Measured on b0049, where every occurrence of mask sat inside a math span and
    the block's only prose was translated around it.
    """
    glossary = [{"source": "Mask", "translation": "掩码"}]
    source = "weights are set as $\\lambda_{mask} = 5$ and $K = \\{cls, bbox, mask\\}$"
    target = "权重设定为 $\\lambda_{\\text{mask}} = 5$ 且 $K = \\{\\text{cls, bbox, mask}\\}$"
    (finding,) = detect_term_drift(source, target, normalize_glossary(glossary))
    assert finding.drifted is False


def test_an_untranslated_prose_term_is_still_drift() -> None:
    """The near miss that must keep working: prose occurrences are not labels."""
    glossary = [{"source": "Mask", "translation": "掩码"}]
    assert (
        detect_term_drift(
            "The mask head predicts masks for every query.",
            "头部为每个查询预测结果。",
            normalize_glossary(glossary),
        )[0].drifted
        is True
    )
    # ...and a bare occurrence in running prose is not excused as a label either.
    assert (
        detect_term_drift(
            "See the mask section for details.",
            "详情参见 mask 一节。",
            normalize_glossary(glossary),
        )[0].drifted
        is True
    )


# --------------------------------------------------------------------------- #
# the literal candidate index (Aho fast path) matches the enforcer exactly
# --------------------------------------------------------------------------- #


def _reference_source_scan(
    source_text: str,
    terms: Sequence[GlossaryTerm],
    protected: list[tuple[int, int]],
) -> list[tuple[str, tuple[tuple[int, int], ...]]]:
    """The retired per-surface scan, kept as the equivalence oracle.

    This is detect_term_drift's source-side loop exactly as it read before the
    Aho candidate index: one find_term_occurrences call per surface, first
    surface with hits wins. The index must reproduce it surface for surface —
    a discrepancy is either a missed candidate (a false "drifted") or an extra
    one (a false "carried"), and both flip real repair decisions.
    """
    from ubt.core.validators.glossary_enforcer import find_term_occurrences

    results: list[tuple[str, tuple[tuple[int, int], ...]]] = []
    for term in terms:
        matched = ""
        source_hits: tuple[tuple[int, int], ...] = ()
        for surface in term.source_surfaces:
            is_acronym = surface.isupper() and len(surface) <= 5
            hits = find_term_occurrences(
                source_text, surface, protected, case_insensitive=not is_acronym
            )
            if hits:
                matched = surface
                source_hits = tuple(hits)
                break
        results.append((matched, source_hits))
    return results


_UNICODE_EXOTICS = ("İstanbul", "Kelvin unit K stands", "ſtring", "µmeter μm", "ß sharp")

_ASCII_WORDS = ("transformer", "Scaling", "Law", "BERT", "MIG", "mask", "head", "GPU")
_CJK_WORDS = ("缩放", "掩码", "深度学习", "调度器")


def _random_case(source: str, rng: random.Random) -> str:
    return "".join(ch.upper() if rng.random() < 0.5 else ch.lower() for ch in source)


class TestLiteralIndexMatchesTheEnforcer:
    """Randomized equivalence: index.hits == find_term_occurrences, both flags.

    The corpus deliberately includes the characters where CPython's simple case
    fold diverges from ``str.lower()`` — ``İ`` folds to ``i`` in sre but expands
    to two characters in ``str.lower()``, and the Kelvin sign ``K`` folds to
    ``k`` — because a fold mismatch here silently drops candidates and flips
    real blocks into repair.
    """

    def test_random_glossaries_and_texts(self) -> None:
        from ubt.core.qe.term_drift import _LiteralSourceIndex
        from ubt.core.validators.glossary_enforcer import (
            extract_protected_spans,
            find_term_occurrences,
        )

        rng = random.Random(20261028)
        for round_no in range(120):
            surfaces: list[str] = []
            for _ in range(rng.randint(1, 6)):
                kind = rng.random()
                if kind < 0.35:
                    surfaces.append(_random_case(rng.choice(_ASCII_WORDS), rng))
                elif kind < 0.55:
                    surfaces.append(rng.choice(_CJK_WORDS))
                elif kind < 0.7:
                    # Mixed-script term: the enforcer's own case handling is
                    # the oracle, the index must not care.
                    surfaces.append(rng.choice(_ASCII_WORDS) + rng.choice(_CJK_WORDS))
                elif kind < 0.85:
                    surfaces.append(_random_case("BE", rng) if rng.random() < 0.5 else "RL")
                else:
                    surfaces.append(rng.choice(_UNICODE_EXOTICS)[: rng.randint(2, 6)])
            pieces: list[str] = []
            for _ in range(rng.randint(0, 12)):
                pick = rng.random()
                if pick < 0.45 and surfaces:
                    pieces.append(_random_case(rng.choice(surfaces), rng))
                elif pick < 0.6:
                    pieces.append(rng.choice(_UNICODE_EXOTICS))
                elif pick < 0.8:
                    pieces.append(rng.choice(_ASCII_WORDS + _CJK_WORDS))
                elif pick < 0.9:
                    pieces.append(rng.choice(("_", "-", "0", "7", " ", "\t", "\n")))
                else:
                    pieces.append(rng.choice(("<code>x</code>", "`span`", "http://u.io/a")))
            text = rng.choice(("", " ")).join(pieces)
            protected = extract_protected_spans(text)
            terms = tuple(GlossaryTerm(source=s, expected=s + "译") for s in surfaces if s)
            index = _LiteralSourceIndex.build(text, [s for t in terms for s in t.source_surfaces])
            if index is None:  # ahocorasick wheel absent: the slow path is the only path
                return
            for term in terms:
                for surface in term.source_surfaces:
                    is_acronym = surface.isupper() and len(surface) <= 5
                    for case_insensitive in (not is_acronym, True, False):
                        assert index.hits(
                            surface, case_insensitive=case_insensitive, protected=protected
                        ) == find_term_occurrences(
                            text, surface, protected, case_insensitive=case_insensitive
                        ), (
                            f"round {round_no}: {surface!r} "
                            f"(case_insensitive={case_insensitive}) in {text!r}"
                        )

    def test_detect_source_scan_equals_the_reference(self) -> None:
        """End to end: the production entry point's source scan is unchanged."""
        from ubt.core.qe.term_drift import _coerce_terms
        from ubt.core.validators.glossary_enforcer import extract_protected_spans

        rng = random.Random(42)
        for _ in range(40):
            surfaces = [
                _random_case(rng.choice(_ASCII_WORDS), rng) for _ in range(rng.randint(1, 5))
            ]
            surfaces += [rng.choice(_CJK_WORDS) for _ in range(rng.randint(0, 3))]
            rng.shuffle(surfaces)
            terms = tuple(
                GlossaryTerm(
                    source=s,
                    expected=s + "译",
                    aliases=(_random_case(s, rng),) if rng.random() < 0.3 else (),
                )
                for s in surfaces
                if s
            )
            text = " ".join(
                _random_case(rng.choice(surfaces + list(_UNICODE_EXOTICS)), rng)
                for _ in range(rng.randint(0, 10))
            )
            protected = extract_protected_spans(text)
            actual = [
                (f.matched_surface, f.source_hits)
                for f in detect_term_drift(" ".join([text]), " ", terms, source_protected=protected)
            ]
            expected = _reference_source_scan(text, _coerce_terms(terms), protected)
            assert actual == expected
