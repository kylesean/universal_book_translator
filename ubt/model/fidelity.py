"""Fidelity lattice, proofs, and attestations (core document model).

The lattice is the whole safety story in five lines: every element is realized
at some fidelity, and it may only ever move *down* this ordered set. The bottom
rung is :attr:`Fidelity.PRESERVED_OPAQUE` -- the original bytes -- so no
realization can be worse than the source. ``DROPPED`` sits below it only for
elements explicitly marked decorative; anything else that would land there is a
construction-time error, not a warning.

This module also owns :class:`Proof` (the one verification vocabulary) and
:class:`Attestation` (a proof pinned to an element at a given rung). ``FIDELITY_DESCENT`` states, per element
class, the order a backend tries before giving up.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum

from ubt.model.ast import (
    Caption,
    CodeBlock,
    Dialogue,
    Element,
    Figure,
    Formula,
    Heading,
    ListItem,
    Paragraph,
    Table,
)


class Fidelity(IntEnum):
    """How an element was realized. Ordered: a realization may only descend.

    ``PRESERVED_OPAQUE`` is the floor and is lossless by construction (an
    opaque source slice), so the ordering is also a *quality* ordering with a
    guaranteed lower bound.
    """

    DROPPED = 0  # removed on purpose (decorative only); recorded, never silent
    PRESERVED_OPAQUE = 1  # original bytes placed whole -- lossless
    RECONSTRUCTED_ADAPTED = 2  # re-created and verified; layout may adapt
    RECONSTRUCTED_VERIFIED = 3  # re-created and verified; geometry also matches


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


@dataclass(frozen=True, slots=True)
class Attestation:
    """A proof pinned to one element at the fidelity it was realized at."""

    element_id: str
    fidelity: Fidelity
    proof: Proof
    note: str = ""

    @property
    def delivered(self) -> bool:
        """False only for a dropped element (decorative by construction)."""
        return self.fidelity is not Fidelity.DROPPED


#: The order a backend tries per element class before the element is lost.
#: Assets that are only ever placed keep a single rung; text may be re-created
#: or, failing that, kept verbatim as an opaque slice (Axiom B: translate or
#: keep, never drop).
FIDELITY_DESCENT: dict[type[Element], tuple[Fidelity, ...]] = {
    Heading: (Fidelity.RECONSTRUCTED_ADAPTED, Fidelity.PRESERVED_OPAQUE),
    Paragraph: (Fidelity.RECONSTRUCTED_ADAPTED, Fidelity.PRESERVED_OPAQUE),
    Dialogue: (Fidelity.RECONSTRUCTED_ADAPTED, Fidelity.PRESERVED_OPAQUE),
    ListItem: (Fidelity.RECONSTRUCTED_ADAPTED, Fidelity.PRESERVED_OPAQUE),
    Caption: (Fidelity.RECONSTRUCTED_ADAPTED, Fidelity.PRESERVED_OPAQUE),
    CodeBlock: (Fidelity.PRESERVED_OPAQUE,),
    Formula: (Fidelity.RECONSTRUCTED_VERIFIED, Fidelity.PRESERVED_OPAQUE),
    Table: (Fidelity.RECONSTRUCTED_VERIFIED, Fidelity.PRESERVED_OPAQUE),
    Figure: (Fidelity.PRESERVED_OPAQUE,),
}


__all__ = [
    "FIDELITY_DESCENT",
    "Attestation",
    "Fidelity",
    "Proof",
    "ProofKind",
    "ProofOutcome",
]
