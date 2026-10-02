"""Differential shadow accounting: legacy verdicts vs the modern Proof.

Phase 0 must be a pure refactor: the legacy check and the new verifier run on
identical inputs, and their three-valued outcomes must agree. ``ShadowRun`` is
the executable form of that acceptance criterion -- it maps each legacy verdict
word to the :class:`ProofOutcome` it *means* and tallies agreement, so a future
edit to either side cannot silently change a verdict.

The contract pinned here is the accounting, not the corpus run: the verdict
vocabulary, what counts as compared vs skipped, what a mismatch records, and the
pass/agreement/summary arithmetic.
"""

from __future__ import annotations

import pytest

from ubt.model.fidelity import Proof, ProofKind, ProofOutcome
from ubt.verify.shadow import AgreementReport, Mismatch, ShadowRun, expected_outcome

pytestmark = pytest.mark.fast

_VERIFIED_TOKENS = ("pass", "verified", "ok")
_FAILED_TOKENS = ("fail", "failed", "corrupt")
_UNVERIFIABLE_TOKENS = ("skip", "skipped", "unwitnessable", "unverifiable")


# --------------------------------------------------------------------------- #
# The one translation table: legacy words -> the outcome they mean.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("token", _VERIFIED_TOKENS)
def test_verified_words_map_to_verified(token: str) -> None:
    assert expected_outcome(token) is ProofOutcome.VERIFIED


@pytest.mark.parametrize("token", _FAILED_TOKENS)
def test_failed_words_map_to_failed(token: str) -> None:
    assert expected_outcome(token) is ProofOutcome.FAILED


@pytest.mark.parametrize("token", _UNVERIFIABLE_TOKENS)
def test_unverifiable_words_map_to_unverifiable(token: str) -> None:
    assert expected_outcome(token) is ProofOutcome.UNVERIFIABLE


@pytest.mark.parametrize("token", ["  PASS ", "Verified", "OK", "Corrupt", "UNWITNESSABLE"])
def test_legacy_words_are_matched_case_and_whitespace_insensitively(token: str) -> None:
    assert expected_outcome(token) is not None


def test_an_unknown_legacy_word_is_not_translated() -> None:
    assert expected_outcome("not-a-verdict") is None


# --------------------------------------------------------------------------- #
# AgreementReport: the tally and its arithmetic.
# --------------------------------------------------------------------------- #


def test_empty_report_agrees_by_convention() -> None:
    report = AgreementReport()
    assert report.total == 0
    assert report.agreement == 1.0
    assert report.passed


def test_agreement_is_the_agreed_fraction() -> None:
    report = AgreementReport(total=4, agreed=3)
    assert report.agreement == 0.75


def test_report_passes_only_without_mismatches() -> None:
    mismatch = Mismatch("l", "pass", "failed", "verified")
    report = AgreementReport(total=1, agreed=0, mismatches=[mismatch])
    assert not report.passed


def test_summary_states_agreement_skips_and_mismatches() -> None:
    report = AgreementReport(total=4, agreed=3, skipped=2)
    assert report.summary() == "3/4 agree (75.0%), 2 skipped, 0 mismatch(es)"


# --------------------------------------------------------------------------- #
# ShadowRun.record: compared vs skipped, agree vs mismatch.
# --------------------------------------------------------------------------- #


def test_a_recognized_verdict_that_agrees_is_counted() -> None:
    run = ShadowRun()
    run.record("l1", "pass", Proof.ok(ProofKind.PREDICATE))
    report = run.report
    assert report.total == 1
    assert report.agreed == 1
    assert report.by_outcome == {"verified": 1}
    assert report.passed


def test_a_recognized_verdict_that_disagrees_records_a_mismatch() -> None:
    run = ShadowRun()
    run.record("l5", "pass", Proof.fail(ProofKind.PREDICATE, "ratio"))
    report = run.report
    assert report.total == 1
    assert report.agreed == 0
    assert not report.passed
    (mismatch,) = report.mismatches
    # label, the legacy word, the modern outcome, and what the legacy word meant.
    assert (mismatch.label, mismatch.legacy, mismatch.modern, mismatch.expected) == (
        "l5",
        "pass",
        "failed",
        "verified",
    )


def test_an_unrecognized_verdict_is_skipped_not_compared() -> None:
    run = ShadowRun()
    run.record("l4", "not-a-verdict", Proof.ok(ProofKind.PREDICATE))
    report = run.report
    assert report.total == 0
    assert report.agreed == 0
    assert report.skipped == 1
    assert report.by_outcome == {}


def test_every_outcome_is_tallied_in_by_outcome() -> None:
    run = ShadowRun()
    run.record("a", "pass", Proof.ok(ProofKind.PREDICATE))
    run.record("b", "corrupt", Proof.fail(ProofKind.STRUCTURAL, "x"))
    run.record("c", "unwitnessable", Proof.unknown(ProofKind.PIXEL, "y"))
    assert run.report.by_outcome == {"verified": 1, "failed": 1, "unverifiable": 1}


def test_skip_counts_a_case_the_harness_could_not_compare() -> None:
    run = ShadowRun()
    run.skip("no source crop")
    assert run.report.skipped == 1
    assert run.report.total == 0
    assert run.report.passed


def test_agreement_is_computed_over_compared_cases_only() -> None:
    run = ShadowRun()
    run.record("a", "pass", Proof.ok(ProofKind.PREDICATE))
    run.record("b", "pass", Proof.fail(ProofKind.PREDICATE, "x"))
    run.skip("no crop")
    report = run.report
    assert report.total == 2
    assert report.agreed == 1
    assert report.agreement == 0.5
    assert report.skipped == 1
