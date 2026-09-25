"""Typed lifecycle failure taxonomy over ``error_flags``."""

from __future__ import annotations

from ubt.core.qe.defect_taxonomy import (
    DRAFTING_ERROR_PREFIX,
    FLAG_MQM_CRITICAL_BLOCKED,
    FLAG_NEEDS_HUMAN_REVIEW,
    NON_RETRYABLE_DRAFT_PREFIX,
    REPAIR_ERROR_PREFIX,
    TRANSIENT_FAILURE_PREFIXES,
    TRIAGE_VERDICT_FLAGS,
    UNTRANSLATED_PREFIX,
    has_critical_defect,
    has_structural_defect,
    has_triage_verdict,
    is_repair_only_transient_failure,
    is_transient_failure,
)


def test_writer_prefixes_are_the_transient_set() -> None:
    assert TRANSIENT_FAILURE_PREFIXES == (
        DRAFTING_ERROR_PREFIX,
        REPAIR_ERROR_PREFIX,
        UNTRANSLATED_PREFIX,
    )


def test_transient_failures_are_reset() -> None:
    assert is_transient_failure([f"{DRAFTING_ERROR_PREFIX} ConnectionError"])
    assert is_transient_failure([f"{REPAIR_ERROR_PREFIX} 429 rate limit"])
    assert is_transient_failure([f"{UNTRANSLATED_PREFIX} block was still non-terminal"])
    assert is_transient_failure(["some_other_flag", f"{DRAFTING_ERROR_PREFIX} boom"])


def test_quality_defects_are_not_transient() -> None:
    assert not is_transient_failure([])
    assert not is_transient_failure(["mqm_critical_blocked"])
    assert not is_transient_failure(["needs_human_review"])
    assert not is_transient_failure(["Numeric fidelity", "Omission suspected"])
    # A bare colon-less lookalike must not match a prefixed marker.
    assert not is_transient_failure(["drafting error without prefix"])


def test_triage_verdicts_are_recognized() -> None:
    assert TRIAGE_VERDICT_FLAGS == (FLAG_NEEDS_HUMAN_REVIEW, FLAG_MQM_CRITICAL_BLOCKED)
    assert has_triage_verdict([FLAG_NEEDS_HUMAN_REVIEW])
    assert has_triage_verdict([FLAG_MQM_CRITICAL_BLOCKED])
    assert has_triage_verdict(["other", FLAG_MQM_CRITICAL_BLOCKED])
    assert not has_triage_verdict([])
    assert not has_triage_verdict([f"{DRAFTING_ERROR_PREFIX} boom"])


def test_non_retryable_draft_failure_is_not_transient() -> None:
    """A 401/402 recurs on every resume, so re-queueing it only re-bills.

    The prefix must also not *start with* the transient drafting prefix, or the
    ``startswith`` match in ``is_transient_failure`` would reclassify it.
    """
    assert NON_RETRYABLE_DRAFT_PREFIX not in TRANSIENT_FAILURE_PREFIXES
    assert not NON_RETRYABLE_DRAFT_PREFIX.startswith(DRAFTING_ERROR_PREFIX)
    assert not is_transient_failure([f"{NON_RETRYABLE_DRAFT_PREFIX} HTTP 401"])
    assert is_transient_failure([f"{DRAFTING_ERROR_PREFIX} ConnectionError"])


def test_dropped_table_is_a_structural_and_critical_defect() -> None:
    """FastPass emits "Table dropped" with no classifier branch (0.70).

    Without the marker it is one dual-witness +0.05 boost from the default 0.75
    threshold, where the gate marks it MTQE_PASSED and wipes the flag.
    """
    flags = ["Table dropped: the source held a table that was flattened"]
    assert has_structural_defect(flags)
    assert has_critical_defect(flags)
    assert not is_transient_failure(flags)


def test_empty_target_text_is_a_structural_and_critical_defect() -> None:
    """FastPass emits "Empty target text" when target is whitespace or empty.

    Must be classified as both structural and critical so it is quarantined
    at triage rather than treated as a minor defect.
    """
    flags = ["Empty target text"]
    assert has_structural_defect(flags)
    assert has_critical_defect(flags)
    assert not is_transient_failure(flags)


def test_repair_only_marker_is_the_all_transient_repair_case() -> None:
    """What the resume path may re-queue *without* discarding the paid draft.

    ``stages/repair.py`` persists ``target_text`` on a transient repair failure,
    so a block carrying only ``Repair error:`` markers still has an already
    billed draft; a block that also has a drafting/untranslated marker never
    produced usable text and must keep the full PENDING reset.
    """
    assert is_repair_only_transient_failure([f"{REPAIR_ERROR_PREFIX} timeout"])
    assert not is_repair_only_transient_failure([f"{DRAFTING_ERROR_PREFIX} ConnectionReset"])
    assert not is_repair_only_transient_failure([f"{UNTRANSLATED_PREFIX} silent model"])
    # Mixed: the draft leg failed too, so nothing is worth preserving.
    assert not is_repair_only_transient_failure(
        [f"{REPAIR_ERROR_PREFIX} timeout", f"{DRAFTING_ERROR_PREFIX} 503"]
    )
    # Non-transient flags say nothing about the draft: not a repair-only requeue.
    assert not is_repair_only_transient_failure(["needs_human_review"])
    assert not is_repair_only_transient_failure([])
