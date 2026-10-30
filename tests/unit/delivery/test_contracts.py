"""Contract vocabulary and compatibility-map tests (Phase 0 scaffold)."""

from __future__ import annotations

from typing import get_args

import pytest

from ubt.core.engine.delivery_contract import ContractName
from ubt.delivery.contracts import (
    DeliveryContract,
    DeliveryMode,
    DeliverySignals,
    DocumentClass,
    legacy_delivery_contract,
)
from ubt.delivery.router import route_delivery

pytestmark = pytest.mark.fast


@pytest.mark.parametrize(
    ("legacy", "expected"),
    [
        ("reflow", DeliveryContract.READABLE),
        ("facsimile", DeliveryContract.FAITHFUL),
        ("page_locked", DeliveryContract.FAITHFUL),
    ],
)
def test_legacy_modes_map_to_target_contracts(
    legacy: str,
    expected: DeliveryContract,
) -> None:
    assert legacy_delivery_contract(legacy) is expected


def test_unknown_legacy_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown legacy delivery mode"):
        legacy_delivery_contract("mystery")


def test_delivery_contract_values_match_the_artifact_contract() -> None:
    """Routing vocabulary and artifact verdict vocabulary must not drift.

    ``ubt.core.engine.delivery_contract`` already owns the artifact-side
    ``ContractName`` literal. The router's enum is the same two contracts;
    this guard makes a rename fail here instead of producing a report with a
    contract the router cannot parse.
    """
    assert set(get_args(ContractName)) == {contract.value for contract in DeliveryContract}


def test_plan_serializes_with_stable_schema_version() -> None:
    plan = route_delivery(
        DeliverySignals(
            document_class=DocumentClass.ACADEMIC,
            text_trust=0.95,
        ),
        requested=DeliveryMode.AUTO,
    )
    payload = plan.to_manifest()
    assert payload["schema_version"] == 1
    assert payload["contract"] == "readable"
    assert plan.signals.document_class is DocumentClass.ACADEMIC
