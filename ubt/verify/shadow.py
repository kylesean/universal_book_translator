"""Differential shadow check: prove the verifier seam changes no behaviour.

The seam migration must be a pure refactor. This module runs the *legacy* check and the
*new* verifier on identical inputs and asserts their three-valued outcomes
agree, so a future edit to either side cannot silently drift. It is the
executable form of the regression differential acceptance criterion ("100% agreement over the
corpus").

Agreement is judged on *semantics*, not on the legacy spelling: each legacy
token is mapped once, here, to a :class:`ProofOutcome`. A mismatch therefore
means the wrapper does not faithfully represent the old check -- the one failure
mode that would turn "unify the vocabulary" into "quietly change the verdict".
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ubt.model.fidelity import Proof, ProofOutcome

#: Legacy verdict words -> the outcome they mean. The single translation table;
#: adding a check means adding its words here, nowhere else.
_LEGACY_OUTCOME: dict[str, ProofOutcome] = {
    "pass": ProofOutcome.VERIFIED,
    "verified": ProofOutcome.VERIFIED,
    "ok": ProofOutcome.VERIFIED,
    "fail": ProofOutcome.FAILED,
    "failed": ProofOutcome.FAILED,
    "corrupt": ProofOutcome.FAILED,
    "skip": ProofOutcome.UNVERIFIABLE,
    "skipped": ProofOutcome.UNVERIFIABLE,
    "unwitnessable": ProofOutcome.UNVERIFIABLE,
    "unverifiable": ProofOutcome.UNVERIFIABLE,
}


def expected_outcome(legacy_token: str) -> ProofOutcome | None:
    """The outcome a legacy verdict word means, or None if unrecognized."""
    return _LEGACY_OUTCOME.get(legacy_token.strip().lower())


@dataclass(frozen=True, slots=True)
class Mismatch:
    """One legacy verdict whose Proof came back different."""

    label: str
    legacy: str
    modern: str
    expected: str


@dataclass(slots=True)
class AgreementReport:
    """Tally of a shadow run."""

    total: int = 0
    agreed: int = 0
    skipped: int = 0
    by_outcome: dict[str, int] = field(default_factory=dict)
    mismatches: list[Mismatch] = field(default_factory=list)

    @property
    def agreement(self) -> float:
        """Fraction of compared cases that agreed (1.0 when none compared)."""
        return 1.0 if self.total == 0 else self.agreed / self.total

    @property
    def passed(self) -> bool:
        return not self.mismatches

    def summary(self) -> str:
        return (
            f"{self.agreed}/{self.total} agree ({self.agreement:.1%}), "
            f"{self.skipped} skipped, {len(self.mismatches)} mismatch(es)"
        )


@dataclass(slots=True)
class ShadowRun:
    """Accumulates legacy-vs-modern comparisons for one acceptance run."""

    report: AgreementReport = field(default_factory=AgreementReport)

    def record(self, label: str, legacy_token: str, proof: Proof) -> None:
        """Compare one legacy verdict word against the modern Proof."""
        expected = expected_outcome(legacy_token)
        if expected is None:
            self.report.skipped += 1
            return
        self.report.total += 1
        self.report.by_outcome[proof.outcome.value] = (
            self.report.by_outcome.get(proof.outcome.value, 0) + 1
        )
        if proof.outcome is expected:
            self.report.agreed += 1
        else:
            self.report.mismatches.append(
                Mismatch(label, legacy_token, proof.outcome.value, expected.value)
            )

    def skip(self, reason: str = "") -> None:
        """Count one case the harness could not compare (e.g. no source crop)."""
        _ = reason
        self.report.skipped += 1


__all__ = ["AgreementReport", "Mismatch", "ShadowRun", "expected_outcome"]
