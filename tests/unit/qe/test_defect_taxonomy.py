"""Single source of truth for defect-flag classification.

The same ``error_flags`` strings are read by the quality gate, the repair loop,
triage and the resume path. The load-bearing facts pinned here:

* the three "not translated" phrasings (exact echo, near-verbatim, same-script
  residue) are all structural *and* critical — matching only one lets a
  never-translated paragraph fall through to "other structural rejection" and be
  auto-passed;
* all four masked-token corrupt markers are structural and critical;
* terminology violation and the provenance-only "Visual witness discrepancy" are
  structural but deliberately **not** critical (Major, not quarantine);
* format-only classes (HTML delta, length) are structural but not critical and
  can bypass expensive reasoning only when *every* flag is format-only;
* lifecycle failures re-queue on resume, but a non-retryable drafting failure
  never does, and a repair-only transient failure must go back to
  REPAIR_PENDING rather than PENDING.
"""

from __future__ import annotations

import pytest

from ubt.core.qe.defect_taxonomy import (
    CRITICAL_DEFECT_MARKERS,
    ECHO_MARKERS,
    STRUCTURAL_DEFECT_MARKERS,
    has_critical_defect,
    has_structural_defect,
    has_triage_verdict,
    is_format_only,
    is_repair_only_transient_failure,
    is_transient_failure,
    is_transient_lifecycle_only,
)

pytestmark = pytest.mark.fast


# --------------------------------------------------------------------------- #
# has_structural_defect
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("marker", ECHO_MARKERS)
def test_every_echo_phrasing_is_structural(marker: str) -> None:
    assert has_structural_defect([marker]) is True


@pytest.mark.parametrize(
    "marker",
    [
        "Empty target text",
        "math_token_corrupt",
        "soup_token_corrupt",
        "code_token_corrupt",
        "cite_token_corrupt",
        "Omission suspected",
        "Numeric fidelity",
        "Visual witness discrepancy",
        "Added reference",
        "Glossary term violation",
        "Table grid mismatch",
        "Table dropped",
    ],
)
def test_representative_markers_are_structural(marker: str) -> None:
    assert has_structural_defect([marker]) is True


def test_structural_match_is_substring() -> None:
    assert has_structural_defect(["prefix Empty target text suffix"]) is True


def test_unrelated_flags_are_not_structural() -> None:
    assert has_structural_defect([]) is False
    assert has_structural_defect(["some unrelated note"]) is False


# --------------------------------------------------------------------------- #
# has_critical_defect
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "marker",
    [
        "Empty target text",
        "Numeric fidelity",
        "Math span mismatch",
        "Undelimited math",
        "Hallucinated LaTeX",
        "Omission suspected",
        "Table dropped",
        "Added reference",
        "Prompt scaffold",
        "Prompt template XML artifacts",
        "Repetitive loop hallucination",
        "repair_structural_failure",
    ],
)
def test_untrustworthy_markers_are_critical(marker: str) -> None:
    assert has_critical_defect([marker]) is True


@pytest.mark.parametrize("marker", ECHO_MARKERS)
def test_echoes_are_critical(marker: str) -> None:
    assert has_critical_defect([marker]) is True


@pytest.mark.parametrize(
    "marker",
    [
        "Glossary term violation",  # Major, not quarantine
        "Visual witness discrepancy",  # provenance-only
        "HTML delta failure",  # format-only
        "html_tag_mismatch",
        "Target text suspiciously truncated",
        "Target text suspiciously inflated",
    ],
)
def test_major_and_format_markers_are_not_critical(marker: str) -> None:
    assert has_structural_defect([marker]) is True
    assert has_critical_defect([marker]) is False


def test_unrelated_flags_are_not_critical() -> None:
    assert has_critical_defect([]) is False
    assert has_critical_defect(["some unrelated note"]) is False


# --------------------------------------------------------------------------- #
# is_format_only
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "flag",
    [
        "HTML delta failure",
        "html_tag_mismatch",
        "Target text suspiciously truncated",
        "Target text suspiciously inflated",
    ],
)
def test_a_single_format_flag_is_format_only(flag: str) -> None:
    assert is_format_only([flag]) is True


def test_empty_flags_are_not_format_only() -> None:
    assert is_format_only([]) is False
    assert is_format_only(["", ""]) is False


def test_a_mixed_bag_is_not_format_only() -> None:
    assert is_format_only(["HTML delta failure", "Numeric fidelity"]) is False


# --------------------------------------------------------------------------- #
# lifecycle failures
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "flag",
    ["Drafting error: timeout", "Repair error: boom", "untranslated: stale"],
)
def test_transient_prefixes_are_retryable(flag: str) -> None:
    assert is_transient_failure([flag]) is True


def test_a_non_retryable_draft_is_not_retryable() -> None:
    assert is_transient_failure(["Drafting unrecoverable: 401"]) is False
    assert is_transient_failure([]) is False


def test_repair_only_requires_every_transient_to_be_a_repair() -> None:
    assert is_repair_only_transient_failure(["Repair error: x"]) is True
    assert is_repair_only_transient_failure(["Repair error: x", "Numeric fidelity"]) is True
    assert is_repair_only_transient_failure(["Drafting error: x"]) is False
    assert is_repair_only_transient_failure(["untranslated: x"]) is False
    assert is_repair_only_transient_failure(["Repair error: x", "Drafting error: y"]) is False
    assert is_repair_only_transient_failure([]) is False


def test_lifecycle_only_excludes_quality_defects() -> None:
    assert is_transient_lifecycle_only(["Drafting error: timeout"]) is True
    assert is_transient_lifecycle_only(["Repair error: boom"]) is True
    assert is_transient_lifecycle_only(["untranslated: stale"]) is True
    assert is_transient_lifecycle_only(["Drafting error: x", "Numeric fidelity"]) is False


def test_lifecycle_only_excludes_a_pure_non_retryable_draft() -> None:
    assert is_transient_lifecycle_only(["Drafting unrecoverable: 401"]) is False
    assert is_transient_lifecycle_only([]) is False


def test_lifecycle_only_ignores_blank_flags() -> None:
    assert is_transient_lifecycle_only(["", "Drafting error: x"]) is True


# --------------------------------------------------------------------------- #
# has_triage_verdict
# --------------------------------------------------------------------------- #


def test_triage_verdicts_are_matched_exactly() -> None:
    assert has_triage_verdict(["needs_human_review"]) is True
    assert has_triage_verdict(["mqm_critical_blocked"]) is True
    assert has_triage_verdict(["prefix needs_human_review"]) is False
    assert has_triage_verdict([]) is False


# --------------------------------------------------------------------------- #
# Table coherence
# --------------------------------------------------------------------------- #


def test_the_critical_table_is_a_subset_of_the_structural_table() -> None:
    structural = set(STRUCTURAL_DEFECT_MARKERS)
    assert set(CRITICAL_DEFECT_MARKERS) <= structural
