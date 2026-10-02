"""``realize()``: the execution core, with the two axioms as construction-time errors.

A fake backend and a fake verifier isolate the *control flow* of the fidelity
descent -- which rung is tried, when a rung is skipped, when the lossless floor
is reached -- from the real verifier logic. ``realize`` is pure: given a
capability set and a produced payload, the attestation is deterministic.

What is pinned here:

- Axiom A/B: an element with no lossless realization raises ``IntegrityViolation``
  (never a silent drop), unless it is explicitly decorative.
- ``PRESERVED_OPAQUE`` is attested *without* running a reconstruction predicate.
- A rung that is unsupported, or whose ``produce`` returns ``None``, or whose
  proof does not verify, is skipped rather than accepted.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import ClassVar

import pytest

from ubt.model.ast import ELEMENT_CLASSES, Element, Formula, Paragraph
from ubt.model.fidelity import Fidelity, Proof, ProofKind
from ubt.model.span import CanonicalSource, Span
from ubt.pipeline.steps import IntegrityViolation, realize
from ubt.render.capability import Capabilities, Produced
from ubt.verify.verifier import Verifiers

pytestmark = pytest.mark.fast

_SOURCE = CanonicalSource(doc_id="doc", path="doc.pdf")


# --------------------------------------------------------------------------- #
# Doubles: a backend that answers from a table, verifiers that answer from a
# fixed Proof (or refuse to run at all).
# --------------------------------------------------------------------------- #


class _FakeBackend:
    """A backend whose capabilities and productions are declared per test."""

    name: ClassVar[str] = "fake"

    def __init__(
        self,
        supported: Iterable[tuple[type[Element], Fidelity]] = (),
        produced: dict[tuple[type[Element], Fidelity], Produced | None] | None = None,
    ) -> None:
        self._capabilities = Capabilities(frozenset(supported))
        self._produced = produced or {}
        self.produce_calls: list[tuple[type[Element], Fidelity]] = []

    def capabilities(self) -> Capabilities:
        return self._capabilities

    def produce(
        self, element: Element, fidelity: Fidelity, source: CanonicalSource
    ) -> Produced | None:
        self.produce_calls.append((type(element), fidelity))
        return self._produced.get((type(element), fidelity))


class _RecordingVerifier:
    """Returns a fixed Proof and records every subject it was handed."""

    kind: ClassVar[ProofKind] = ProofKind.STRUCTURAL

    def __init__(self, result: Proof) -> None:
        self._result = result
        self.subjects: list[object] = []

    def verify(self, subject: object) -> Proof:
        self.subjects.append(subject)
        return self._result


class _PoisonVerifier:
    """Must never run; proves a code path skips verification entirely."""

    kind: ClassVar[ProofKind] = ProofKind.STRUCTURAL

    def verify(self, subject: object) -> Proof:
        raise AssertionError(f"verification must not run, but got {subject!r}")


def _stub_verifiers(result: Proof) -> tuple[Verifiers, _RecordingVerifier]:
    stub = _RecordingVerifier(result)
    verifiers = Verifiers(structural=stub, formula=stub, table=stub, text=stub)  # type: ignore[arg-type]
    return verifiers, stub


def _poison_verifiers() -> Verifiers:
    poison = _PoisonVerifier()
    return Verifiers(structural=poison, formula=poison, table=poison, text=poison)  # type: ignore[arg-type]


def _paragraph(**overrides: object) -> Paragraph:
    return Paragraph(id="p1", spine_index=0, span=Span(page=1), **overrides)  # type: ignore[arg-type]


def _element(element_cls: type[Element]) -> Element:
    return element_cls(id="e1", spine_index=0, span=Span(page=1))


# --------------------------------------------------------------------------- #
# Axioms A and B: no element is ever silently dropped.
# --------------------------------------------------------------------------- #


def test_no_supported_rung_raises_integrity_violation() -> None:
    with pytest.raises(IntegrityViolation, match="no lossless realization"):
        realize(_paragraph(), _FakeBackend(), _poison_verifiers(), _SOURCE)


def test_decorative_element_with_no_realization_is_dropped_not_raised() -> None:
    attestation = realize(_paragraph(decorative=True), _FakeBackend(), _poison_verifiers(), _SOURCE)
    assert attestation.fidelity is Fidelity.DROPPED
    assert not attestation.delivered


@pytest.mark.parametrize("element_cls", ELEMENT_CLASSES)
def test_every_element_class_is_deliverable_at_the_floor(element_cls: type[Element]) -> None:
    # A backend that can only place the opaque source slice still delivers every
    # class in the closed union -- the descent always bottoms out at the floor.
    backend = _FakeBackend(
        supported={(element_cls, Fidelity.PRESERVED_OPAQUE)},
        produced={(element_cls, Fidelity.PRESERVED_OPAQUE): Produced("source slice")},
    )
    attestation = realize(_element(element_cls), backend, _poison_verifiers(), _SOURCE)
    assert attestation.fidelity is Fidelity.PRESERVED_OPAQUE
    assert attestation.delivered


# --------------------------------------------------------------------------- #
# The floor is lossless by construction: attested without a predicate.
# --------------------------------------------------------------------------- #


def test_preserved_opaque_is_attested_without_running_the_verifier() -> None:
    backend = _FakeBackend(
        supported={(Paragraph, Fidelity.PRESERVED_OPAQUE)},
        produced={(Paragraph, Fidelity.PRESERVED_OPAQUE): Produced("原文", note="opaque slice")},
    )
    attestation = realize(_paragraph(), backend, _poison_verifiers(), _SOURCE)
    assert attestation.fidelity is Fidelity.PRESERVED_OPAQUE
    assert attestation.proof.kind is ProofKind.PRESERVED
    assert attestation.proof.verified
    assert attestation.note == "opaque slice"
    assert attestation.delivered


# --------------------------------------------------------------------------- #
# Rung selection: highest supported rung that produces and verifies.
# --------------------------------------------------------------------------- #


def _two_rung_backend(element_cls: type[Element], high: Fidelity) -> _FakeBackend:
    return _FakeBackend(
        supported={(element_cls, high), (element_cls, Fidelity.PRESERVED_OPAQUE)},
        produced={
            (element_cls, high): Produced("reconstructed"),
            (element_cls, Fidelity.PRESERVED_OPAQUE): Produced("source slice"),
        },
    )


def test_highest_supported_rung_that_verifies_wins() -> None:
    verifiers, stub = _stub_verifiers(Proof.ok(ProofKind.PREDICATE))
    attestation = realize(
        _paragraph(),
        _two_rung_backend(Paragraph, Fidelity.RECONSTRUCTED_ADAPTED),
        verifiers,
        _SOURCE,
    )
    assert attestation.fidelity is Fidelity.RECONSTRUCTED_ADAPTED
    assert len(stub.subjects) == 1


def test_formula_prefers_verified_reconstruction_over_opaque() -> None:
    verifiers, _ = _stub_verifiers(Proof.ok(ProofKind.STRUCTURAL))
    attestation = realize(
        Formula(id="f1", spine_index=1, span=Span(page=1), source="a^2+b^2"),
        _two_rung_backend(Formula, Fidelity.RECONSTRUCTED_VERIFIED),
        verifiers,
        _SOURCE,
    )
    assert attestation.fidelity is Fidelity.RECONSTRUCTED_VERIFIED


def test_failed_verification_falls_through_to_the_floor() -> None:
    # Axiom B: a failed reconstruction must never drop a non-decorative element;
    # it lands on the opaque floor and stays delivered.
    verifiers, _ = _stub_verifiers(Proof.fail(ProofKind.PREDICATE, "length ratio"))
    attestation = realize(
        _paragraph(),
        _two_rung_backend(Paragraph, Fidelity.RECONSTRUCTED_ADAPTED),
        verifiers,
        _SOURCE,
    )
    assert attestation.fidelity is Fidelity.PRESERVED_OPAQUE
    assert attestation.proof.kind is ProofKind.PRESERVED
    assert attestation.delivered


def test_produce_returning_none_is_a_skip_not_a_loss() -> None:
    backend = _FakeBackend(
        supported={
            (Paragraph, Fidelity.RECONSTRUCTED_ADAPTED),
            (Paragraph, Fidelity.PRESERVED_OPAQUE),
        },
        # The reconstructed rung is supported but produces nothing -> skip it.
        produced={(Paragraph, Fidelity.PRESERVED_OPAQUE): Produced("source slice")},
    )
    verifiers, _ = _stub_verifiers(Proof.ok(ProofKind.PREDICATE))
    attestation = realize(_paragraph(), backend, verifiers, _SOURCE)
    assert attestation.fidelity is Fidelity.PRESERVED_OPAQUE
    assert backend.produce_calls == [
        (Paragraph, Fidelity.RECONSTRUCTED_ADAPTED),
        (Paragraph, Fidelity.PRESERVED_OPAQUE),
    ]


def test_unsupported_rung_is_skipped_without_producing_or_verifying() -> None:
    backend = _FakeBackend(
        supported={(Paragraph, Fidelity.PRESERVED_OPAQUE)},
        produced={(Paragraph, Fidelity.PRESERVED_OPAQUE): Produced("source slice")},
    )
    verifiers, stub = _stub_verifiers(Proof.ok(ProofKind.PREDICATE))
    realize(_paragraph(), backend, verifiers, _SOURCE)
    assert backend.produce_calls == [(Paragraph, Fidelity.PRESERVED_OPAQUE)]
    assert stub.subjects == []
