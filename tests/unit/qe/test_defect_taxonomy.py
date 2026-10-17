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

from typing import Any

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


# --------------------------------------------------------------------------- #
# The single routing decision
#
# Before this, "repair automatically?" was answered twice: the quality gate
# routed every FastPass failure to REPAIR_PENDING, and the repair loop re-decided
# with ``error_flags or score < threshold * 0.8``. Every block reaching the
# second check already carried a flag, so the score clause was dead and the
# ``* 0.8`` constant was a second pass line no documented knob controlled.
# --------------------------------------------------------------------------- #


def test_a_clean_block_passes() -> None:
    from ubt.core.qe.defect_taxonomy import classify_defect

    assert classify_defect([], score=0.92, threshold=0.75) == "PASS"
    assert classify_defect([], score=None, threshold=0.75) == "PASS"


def test_a_named_invariant_failure_is_repairable() -> None:
    """Repair is verifiable: the same check that rejected it will judge the fix."""
    from ubt.core.qe.defect_taxonomy import classify_defect

    assert (
        classify_defect(["Numeric fidelity failure: missing ['33']"], score=0.55, threshold=0.75)
        == "DETERMINISTIC_DEFECT"
    )
    assert (
        classify_defect(
            ["Omission suspected: identifier term(s) missing"], score=0.35, threshold=0.75
        )
        == "DETERMINISTIC_DEFECT"
    )


def test_a_judge_opinion_alone_is_not_repairable() -> None:
    """A rewrite cannot be verified against the judgement that asked for it."""
    from ubt.core.qe.defect_taxonomy import classify_defect

    assert classify_defect(["judge: semantic loss"], score=0.55, threshold=0.75) == "SEMANTIC_DOUBT"
    assert classify_defect([], score=0.55, threshold=0.75) == "SEMANTIC_DOUBT"


def test_a_deterministic_marker_outranks_a_semantic_one() -> None:
    """A dropped number is a deterministic defect even if the judge also objected."""
    from ubt.core.qe.defect_taxonomy import classify_defect

    outcome = classify_defect(
        ["judge: awkward phrasing", "Numeric fidelity failure: missing ['7']"],
        score=0.55,
        threshold=0.75,
    )
    assert outcome == "DETERMINISTIC_DEFECT"


def test_a_flagged_block_is_never_released_by_a_high_score() -> None:
    """The score cannot whitewash a named defect."""
    from ubt.core.qe.defect_taxonomy import classify_defect

    assert (
        classify_defect(["Numeric fidelity failure: ..."], score=0.99, threshold=0.75)
        == "DETERMINISTIC_DEFECT"
    )


def test_the_repair_selector_uses_the_shared_decision() -> None:
    """The repair loop must not carry its own second pass line."""
    from unittest.mock import MagicMock

    from ubt.core.engine.repair_loop import RepairLoop
    from ubt.core.ir.models import BlockStatus
    from ubt.core.qe.heuristic import HeuristicQERunner

    loop = RepairLoop(qe_runner=HeuristicQERunner(), router=MagicMock(), qe_threshold=0.75)

    def block(bid: str, flags: list[str], score: float | None) -> Any:
        b = MagicMock()
        b.id = bid
        b.error_flags = flags
        b.mtqe_score = score
        b.skip_translate = False
        b.is_finalized = False
        b.repair_rounds = 0
        b.status = BlockStatus.REPAIR_PENDING
        return b

    deterministic = block("det", ["Numeric fidelity failure"], 0.55)
    semantic = block("sem", ["judge: semantic loss"], 0.55)
    clean = block("ok", [], 0.92)

    candidates = [deterministic, semantic, clean]
    selected = loop.select_repair_candidates(candidates)
    assert [b.id for b in selected] == ["det"]


# --------------------------------------------------------------------------- #
# One "shippable" rule
#
# ``max(0.5, qe_threshold)`` appeared four times — triage's auto-pass, the
# repair stage's standalone finalizer, and two circuit-breaker exits. Four
# copies of the most consequential threshold in the pipeline.
# --------------------------------------------------------------------------- #


def test_the_shippable_bar_tracks_the_threshold_but_never_drops_below_the_floor() -> None:
    from ubt.core.qe.score_policy import SHIPPABLE_FLOOR, shippable_threshold

    # Above the floor: the operator's setting is honoured.
    assert shippable_threshold(0.9) == 0.9
    assert shippable_threshold(0.75) == 0.75
    # At or below it: the floor holds, so a low setting cannot promote a draft
    # the deterministic gates rejected (their classes sit at or below 0.70).
    assert shippable_threshold(0.2) == SHIPPABLE_FLOOR
    assert shippable_threshold(0.5) == SHIPPABLE_FLOOR


def test_a_block_with_a_defect_flag_is_never_shippable() -> None:
    """A favourable score is not evidence the defect was fixed."""
    from ubt.core.qe.score_policy import is_shippable

    assert is_shippable(0.99, 0.75, ["Numeric fidelity failure"]) is False
    assert is_shippable(0.99, 0.75, []) is True
    assert is_shippable(None, 0.75, []) is False
    assert is_shippable(0.55, 0.75, []) is False


def test_the_judge_cannot_raise_a_score_over_the_pass_line() -> None:
    """There is no upgrade switch: a judge opinion may only lower.

    ``qe_judge_allow_upgrade`` used to let a favourable verdict lift a real
    STRUCTURAL_OTHER failure (0.70) over the line, releasing a draft on an
    opinion alone. The knob is gone.
    """
    import asyncio

    from ubt.core.qe.heuristic import QE_SCORE_STRUCTURAL_OTHER, HeuristicQERunner
    from ubt.core.qe.llm_judge import LLMJudgeQERunner, TieredQERunner

    class _FixedHeuristic(HeuristicQERunner):
        async def score_pairs(self, pairs: list[dict[str, str]]) -> list[float]:
            return [QE_SCORE_STRUCTURAL_OTHER for _ in pairs]

    async def judge_fn(**_kwargs: Any) -> str:
        return "score: 95"

    tiered = TieredQERunner(
        heuristic=_FixedHeuristic(),
        judge=LLMJudgeQERunner(judge_fn=judge_fn),
    )
    scores = asyncio.run(tiered.score_pairs([{"src": "a", "mt": "b"}]))
    assert scores == [QE_SCORE_STRUCTURAL_OTHER]
