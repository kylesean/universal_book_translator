from __future__ import annotations

from typing import get_args

import pytest

from ubt.core.engine.delivery_contract import ContractName
from ubt.delivery.contracts import (
    DeliveryContract,
    DeliveryMode,
    DeliverySignals,
    DocumentClass,
    LayoutComplexity,
)
from ubt.delivery.router import route_delivery

pytestmark = pytest.mark.fast


def test_delivery_contract_values_match_the_artifact_contract() -> None:
    assert set(get_args(ContractName)) == {contract.value for contract in DeliveryContract}


def test_plan_serializes_with_stable_schema_version() -> None:
    plan = route_delivery(
        DeliverySignals(
            document_class=DocumentClass.ACADEMIC,
            text_trust=0.95,
            layout_complexity=LayoutComplexity.LOW,
        ),
        requested=DeliveryMode.AUTO,
    )
    payload = plan.to_manifest()
    assert payload["schema_version"] == 1
    assert payload["contract"] == "faithful"
    assert plan.signals.document_class is DocumentClass.ACADEMIC
