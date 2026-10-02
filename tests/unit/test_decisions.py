"""The realization decision plan and its shadow comparison.

``plan_realization`` is ``attest_document`` (pinned in ``test_attest.py``); what
is pinned *here* is how the plan is consumed:

- ``plan_fidelities`` is the renderer's input -- the committed rung per element;
- ``divergences`` is the shadow check -- a lowering may keep the source
  (descend), but it must never silently fall below what the plan committed to.

The rule is one-directional: falling *below* the plan is a divergence to record;
matching, exceeding, or not being covered at all is not.
"""

from __future__ import annotations

import pytest

from ubt.model.fidelity import Attestation, Fidelity, Proof, ProofKind
from ubt.pipeline.attest import AttestationReport
from ubt.pipeline.decisions import divergences, plan_fidelities

pytestmark = pytest.mark.fast


def _attestation(element_id: str, fidelity: Fidelity) -> Attestation:
    proof = (
        Proof.preserved("x")
        if fidelity is Fidelity.PRESERVED_OPAQUE
        else Proof.ok(ProofKind.PREDICATE)
    )
    return Attestation(element_id, fidelity, proof, "note")


def _plan(*attestations: Attestation) -> AttestationReport:
    return AttestationReport(
        total=len(attestations), text=(), assets=(), attestations=tuple(attestations)
    )


# --------------------------------------------------------------------------- #
# plan_fidelities: the renderer's lookup table.
# --------------------------------------------------------------------------- #


def test_plan_fidelities_maps_each_element_to_its_committed_rung() -> None:
    plan = _plan(
        _attestation("e1", Fidelity.RECONSTRUCTED_ADAPTED),
        _attestation("e2", Fidelity.PRESERVED_OPAQUE),
    )
    assert plan_fidelities(plan) == {
        "e1": Fidelity.RECONSTRUCTED_ADAPTED,
        "e2": Fidelity.PRESERVED_OPAQUE,
    }


def test_plan_fidelities_of_an_empty_plan_is_empty() -> None:
    assert plan_fidelities(_plan()) == {}


# --------------------------------------------------------------------------- #
# divergences: only a lowering *below* the plan counts.
# --------------------------------------------------------------------------- #


def test_a_lowering_below_the_plan_is_a_divergence() -> None:
    plan = _plan(_attestation("e1", Fidelity.RECONSTRUCTED_ADAPTED))
    found = divergences(plan, {"e1": Fidelity.PRESERVED_OPAQUE})
    assert found == (("e1", Fidelity.RECONSTRUCTED_ADAPTED, Fidelity.PRESERVED_OPAQUE),)


def test_a_lowering_that_matches_the_plan_is_not_a_divergence() -> None:
    plan = _plan(_attestation("e1", Fidelity.PRESERVED_OPAQUE))
    assert divergences(plan, {"e1": Fidelity.PRESERVED_OPAQUE}) == ()


def test_a_lowering_above_the_plan_is_not_a_divergence() -> None:
    plan = _plan(_attestation("e1", Fidelity.PRESERVED_OPAQUE))
    assert divergences(plan, {"e1": Fidelity.RECONSTRUCTED_VERIFIED}) == ()


def test_an_element_absent_from_the_realized_map_is_not_a_divergence() -> None:
    # A missing id means the plan does not cover it; the renderer falls back to
    # its own judgment, which is not a divergence from the plan.
    plan = _plan(_attestation("e1", Fidelity.RECONSTRUCTED_ADAPTED))
    assert divergences(plan, {}) == ()


def test_ids_the_plan_does_not_know_are_ignored() -> None:
    plan = _plan(_attestation("e1", Fidelity.RECONSTRUCTED_ADAPTED))
    assert (
        divergences(plan, {"e1": Fidelity.RECONSTRUCTED_ADAPTED, "extra": Fidelity.DROPPED}) == ()
    )


def test_divergences_follow_the_plan_order() -> None:
    plan = _plan(
        _attestation("e1", Fidelity.RECONSTRUCTED_VERIFIED),
        _attestation("e2", Fidelity.RECONSTRUCTED_VERIFIED),
    )
    found = divergences(plan, {"e1": Fidelity.PRESERVED_OPAQUE, "e2": Fidelity.PRESERVED_OPAQUE})
    assert [element_id for element_id, _, _ in found] == ["e1", "e2"]


def test_an_empty_plan_has_no_divergences() -> None:
    assert divergences(_plan(), {"e1": Fidelity.PRESERVED_OPAQUE}) == ()
