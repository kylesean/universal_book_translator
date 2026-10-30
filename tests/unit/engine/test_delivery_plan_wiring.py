"""Pipeline preflight delivery-plan wiring tests (non-PDF path, no model)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ubt.core.config import UBTConfig
from ubt.core.engine.pipeline import resolve_delivery_for_input
from ubt.delivery.contracts import DeliveryContract

pytestmark = pytest.mark.fast


def test_auto_on_non_pdf_routes_to_readable() -> None:
    decision = resolve_delivery_for_input(
        UBTConfig(delivery_mode="auto", document_mode="paper"),
        Path("notes.md"),
    )
    assert decision.effective_mode == "reflow"
    assert decision.plan.contract is DeliveryContract.READABLE
    assert decision.plan.feasible


def test_new_faithful_on_non_pdf_fails_before_rendering() -> None:
    decision = resolve_delivery_for_input(
        UBTConfig(delivery_mode="faithful"),
        Path("notes.md"),
    )
    assert decision.effective_mode is None
    assert decision.plan.feasible is False
    assert decision.plan.contract is None


def test_allow_degrade_on_non_pdf_routes_to_readable() -> None:
    decision = resolve_delivery_for_input(
        UBTConfig(delivery_mode="faithful", allow_degrade=True),
        Path("notes.md"),
    )
    assert decision.effective_mode == "reflow"
    assert decision.plan.contract is DeliveryContract.READABLE


def test_legacy_facsimile_is_grandfathered_on_non_pdf() -> None:
    decision = resolve_delivery_for_input(
        UBTConfig(delivery_mode="facsimile"),
        Path("notes.md"),
    )
    assert decision.effective_mode == "facsimile"
    assert decision.plan.contract is DeliveryContract.FAITHFUL
    assert decision.plan.feasible
