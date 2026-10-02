"""The fidelity lattice: the ordering, the descent, and the proof vocabulary.

These are the project's core invariants -- every element is realized at some
fidelity and may only ever move *down* the ordered set, with ``PRESERVED_OPAQUE``
as the lossless floor. Pure data, no IO, no optional dependencies.
"""

from __future__ import annotations

import pytest

from ubt.model.ast import ELEMENT_CLASSES, Figure
from ubt.model.fidelity import (
    FIDELITY_DESCENT,
    Attestation,
    Fidelity,
    Proof,
    ProofKind,
    ProofOutcome,
)

pytestmark = pytest.mark.fast


def test_fidelity_is_a_total_order() -> None:
    assert Fidelity.DROPPED < Fidelity.PRESERVED_OPAQUE
    assert Fidelity.PRESERVED_OPAQUE < Fidelity.RECONSTRUCTED_ADAPTED
    assert Fidelity.RECONSTRUCTED_ADAPTED < Fidelity.RECONSTRUCTED_VERIFIED


def test_every_element_class_has_a_descent() -> None:
    assert set(FIDELITY_DESCENT) == set(ELEMENT_CLASSES)


@pytest.mark.parametrize("element_cls", ELEMENT_CLASSES)
def test_descent_is_strictly_descending(element_cls: type) -> None:
    rungs = FIDELITY_DESCENT[element_cls]
    assert rungs, f"{element_cls.__name__} has no realization path"
    values = [int(fidelity) for fidelity in rungs]
    assert values == sorted(values, reverse=True)
    assert len(set(values)) == len(values)


@pytest.mark.parametrize("element_cls", ELEMENT_CLASSES)
def test_descent_floors_at_preserved_opaque(element_cls: type) -> None:
    # The bottom rung is the opaque source slice -- lossless by construction,
    # so no realization can be worse than the source.
    assert FIDELITY_DESCENT[element_cls][-1] is Fidelity.PRESERVED_OPAQUE


def test_asset_that_is_only_ever_placed_has_a_single_rung() -> None:
    assert FIDELITY_DESCENT[Figure] == (Fidelity.PRESERVED_OPAQUE,)


def test_proof_constructors_map_to_the_three_valued_outcome() -> None:
    assert Proof.ok(ProofKind.STRUCTURAL).verified
    assert Proof.fail(ProofKind.STRUCTURAL).failed
    assert Proof.unknown(ProofKind.STRUCTURAL).unverifiable


def test_preserved_proof_is_verified_and_lossless() -> None:
    proof = Proof.preserved("opaque slice")
    assert proof.verified
    assert proof.kind is ProofKind.PRESERVED
    assert proof.outcome is ProofOutcome.VERIFIED


def test_attestation_delivered_is_false_only_for_dropped() -> None:
    proof = Proof.ok(ProofKind.STRUCTURAL)
    assert Attestation("e1", Fidelity.PRESERVED_OPAQUE, proof).delivered
    assert Attestation("e1", Fidelity.RECONSTRUCTED_ADAPTED, proof).delivered
    dropped = Attestation("e1", Fidelity.DROPPED, Proof.fail(ProofKind.NONE, "decorative"))
    assert not dropped.delivered
