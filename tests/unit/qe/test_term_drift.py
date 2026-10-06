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
    (finding,) = detect_term_drift("The FinFET device", "code `鳍式场效应晶体管` here", _GLOSSARY)
    assert finding.occurs_in_source is True
    assert finding.rendered is False
    assert finding.drifted is True


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
