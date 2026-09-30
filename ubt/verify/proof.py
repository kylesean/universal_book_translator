"""The one vocabulary every verifier in UBT speaks.

Before this module, a verification outcome was spelled three incompatible ways:
``asset_verify.VerifyResult`` used ``pass/fail/skip``, the pixel witnesses used
``pass/fail/unwitnessable``, and ``FastPassFilter`` returned a ``passed`` bool.
Every new gate had to learn each dialect, and a cross-cutting rule ("only a
verified reconstruction may be reported as reconstructed") had to be
re-implemented per call site.

``Proof`` collapses them into one three-valued outcome plus the *kind* of
evidence behind it. The three values are deliberately not a boolean: the system
must tell "checked and good" apart from "could not check", because only the
former may back a reconstructed verdict -- the latter is the fail-open path that
keeps the original. That distinction is exactly Axiom A:

- ``VERIFIED``     -- checked; a reconstruction may be reported as such.
- ``FAILED``       -- checked and wrong; corruption.
- ``UNVERIFIABLE`` -- not checked; no evidence either way (keep what you had).

Nothing here performs IO, imports a verifier, or knows about PDFs. It is the
pure seed of the future ``ubt.model`` fidelity layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ProofOutcome(StrEnum):
    """Three-valued result of one verification: good, bad, or unknown."""

    VERIFIED = "verified"
    FAILED = "failed"
    UNVERIFIABLE = "unverifiable"


class ProofKind(StrEnum):
    """What kind of evidence backs a proof."""

    STRUCTURAL = "structural"  # cheap grammar/shape check on delivered text
    PIXEL = "pixel"  # raster comparison against the source region
    PREDICATE = "predicate"  # invariant check on a (source, target) pair
    PRESERVED = "preserved"  # lossless by construction (opaque source slice)
    NONE = "none"  # no check ran


@dataclass(frozen=True, slots=True)
class Proof:
    """Outcome of verifying one element, plus the evidence behind it."""

    outcome: ProofOutcome
    kind: ProofKind = ProofKind.NONE
    detail: str = ""
    findings: tuple[str, ...] = ()

    @property
    def verified(self) -> bool:
        """Checked and good. Only this may back a reconstructed verdict."""
        return self.outcome is ProofOutcome.VERIFIED

    @property
    def failed(self) -> bool:
        """Checked and wrong (corruption)."""
        return self.outcome is ProofOutcome.FAILED

    @property
    def unverifiable(self) -> bool:
        """No evidence either way; callers keep what they had (fail-open)."""
        return self.outcome is ProofOutcome.UNVERIFIABLE

    @classmethod
    def ok(cls, kind: ProofKind, detail: str = "") -> Proof:
        return cls(ProofOutcome.VERIFIED, kind, detail)

    @classmethod
    def fail(cls, kind: ProofKind, detail: str = "", *, findings: tuple[str, ...] = ()) -> Proof:
        return cls(ProofOutcome.FAILED, kind, detail, findings)

    @classmethod
    def unknown(cls, kind: ProofKind, detail: str = "", *, findings: tuple[str, ...] = ()) -> Proof:
        return cls(ProofOutcome.UNVERIFIABLE, kind, detail, findings)

    @classmethod
    def preserved(cls, detail: str = "") -> Proof:
        """Lossless by construction: an opaque source slice, never re-created."""
        return cls(ProofOutcome.VERIFIED, ProofKind.PRESERVED, detail)


__all__ = ["Proof", "ProofKind", "ProofOutcome"]
