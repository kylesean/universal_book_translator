from __future__ import annotations

import pytest

from ubt.delivery.contracts import (
    DegradePolicy,
    DeliveryContract,
    DocumentClass,
    PageKind,
    PageStrategy,
)
from ubt.delivery.planning import (
    DeliveryDecision,
    build_delivery_signals,
    document_class_from_mode,
    engine_supports_contract,
    plan_for_document,
    resolve_pdf_engine,
)

pytestmark = pytest.mark.fast


def _plan(
    requested: str,
    *,
    document_mode: str = "paper",
    is_pdf: bool = True,
    has_scan: bool = False,
    font_damage_pages: int = 0,
    page_count: int = 10,
    multicolumn_page_share: float = 0.0,
    structural_page_share: float = 0.0,
    formula_heavy: bool = False,
    allow_degrade: bool = False,
) -> DeliveryDecision:
    return plan_for_document(
        requested,
        document_mode=document_mode,
        is_pdf=is_pdf,
        has_scan=has_scan,
        font_damage_pages=font_damage_pages,
        page_count=page_count,
        multicolumn_page_share=multicolumn_page_share,
        structural_page_share=structural_page_share,
        formula_heavy=formula_heavy,
        allow_degrade=allow_degrade,
    )


def test_document_class_mapping() -> None:
    assert document_class_from_mode("paper") is DocumentClass.ACADEMIC
    assert document_class_from_mode("book") is DocumentClass.LONG_BOOK
    assert document_class_from_mode("auto") is DocumentClass.OTHER


def test_non_pdf_is_never_page_lock_feasible() -> None:
    signals = build_delivery_signals(document_mode="book", is_pdf=False)
    assert signals.page_kind is PageKind.NATIVE
    assert signals.page_lock_feasible is False


def test_clean_pdf_faithful_is_provisional_but_feasible() -> None:
    decision = _plan("faithful")
    assert decision.effective_mode == "faithful"
    assert decision.plan.contract is DeliveryContract.FAITHFUL
    assert decision.plan.feasible
    assert decision.plan.eligibility.provisional is True
    assert decision.plan.default_page_strategy is PageStrategy.CANVAS
    assert any("provisional" in reason for reason in decision.plan.reasons)


def test_scanned_pdf_new_faithful_is_infeasible() -> None:
    decision = _plan("faithful", has_scan=True)
    assert decision.effective_mode is None
    assert decision.plan.feasible is False
    assert decision.plan.contract is None
    assert any("page_kind" in reason for reason in decision.plan.reasons)


def test_font_damaged_pdf_new_faithful_is_infeasible() -> None:
    decision = _plan("faithful", font_damage_pages=3, page_count=10)
    assert decision.effective_mode is None
    assert decision.plan.feasible is False
    assert any("text_trust" in reason for reason in decision.plan.reasons)


def test_non_pdf_new_faithful_is_infeasible() -> None:
    decision = _plan("faithful", is_pdf=False, document_mode="book")
    assert decision.effective_mode is None
    assert decision.plan.feasible is False
    assert any("page_lock_feasible" in reason for reason in decision.plan.reasons)


def test_new_faithful_degrades_to_readable_with_allow_degrade() -> None:
    decision = _plan("faithful", has_scan=True, allow_degrade=True)
    assert decision.effective_mode == "readable"
    assert decision.plan.contract is DeliveryContract.READABLE
    assert decision.plan.degrade_policy is DegradePolicy.ALLOW_DEGRADE


def test_auto_paper_routes_to_native_faithful() -> None:
    decision = _plan("auto", document_mode="paper")
    assert decision.effective_mode == "faithful"
    assert decision.plan.contract is DeliveryContract.FAITHFUL
    assert decision.plan.default_page_strategy is PageStrategy.CANVAS


def test_auto_structured_paper_upgrades_to_faithful() -> None:
    decision = _plan(
        "auto",
        document_mode="paper",
        multicolumn_page_share=0.8,
        formula_heavy=True,
    )
    assert decision.effective_mode == "faithful"
    assert decision.plan.contract is DeliveryContract.FAITHFUL
    assert decision.plan.degrade_policy is DegradePolicy.ALLOW_DEGRADE
    assert decision.plan.default_page_strategy is PageStrategy.CANVAS


def test_auto_book_routes_to_native_faithful() -> None:
    decision = _plan("auto", document_mode="book")
    assert decision.effective_mode == "faithful"
    assert decision.plan.contract is DeliveryContract.FAITHFUL


def test_auto_non_pdf_routes_to_readable() -> None:
    decision = _plan("auto", is_pdf=False, document_mode="book")
    assert decision.effective_mode == "readable"
    assert decision.plan.contract is DeliveryContract.READABLE


def test_explicit_readable_always_maps_to_reflow() -> None:
    decision = _plan("readable", document_mode="book")
    assert decision.effective_mode == "readable"
    assert decision.plan.contract is DeliveryContract.READABLE


def test_high_complexity_does_not_veto_a_clean_native_faithful() -> None:
    decision = _plan(
        "faithful",
        multicolumn_page_share=0.8,
        structural_page_share=0.7,
        formula_heavy=True,
    )
    assert decision.plan.contract is DeliveryContract.FAITHFUL
    assert decision.plan.eligibility.eligible is True


def test_unknown_delivery_mode_raises() -> None:
    with pytest.raises(ValueError, match="unknown delivery mode"):
        _plan("mystery")


def test_auto_engine_follows_the_delivery_contract() -> None:
    """The plan owns the promise; ``auto`` picks an engine that can keep it.

    A readable plan must not be handed to the in-place codec (which would make
    the artifact faithful while the plan says readable), and a faithful plan
    must not be handed to Typst reflow (which cannot promise page identity).
    """
    assert resolve_pdf_engine("auto", is_pdf=True, contract=DeliveryContract.FAITHFUL) == "native"
    assert resolve_pdf_engine("auto", is_pdf=True, contract=DeliveryContract.READABLE) == "docling"
    assert resolve_pdf_engine("auto", is_pdf=False, contract=DeliveryContract.READABLE) == "auto"
    assert (
        resolve_pdf_engine("docling", is_pdf=True, contract=DeliveryContract.READABLE) == "docling"
    )


def test_engine_capability_pins_contracts() -> None:
    assert engine_supports_contract("native", DeliveryContract.READABLE)
    assert engine_supports_contract("native", DeliveryContract.FAITHFUL)
    assert engine_supports_contract("docling", DeliveryContract.READABLE)
    assert not engine_supports_contract("docling", DeliveryContract.FAITHFUL)
    assert engine_supports_contract("docling", None)
