"""Planning adapter tests: facts -> signals -> plan -> effective renderer mode."""

from __future__ import annotations

import logging

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
    _warn_legacy_mode,
    build_delivery_signals,
    document_class_from_mode,
    plan_for_document,
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
    assert decision.effective_mode == "facsimile"
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
    assert decision.effective_mode == "reflow"
    assert decision.plan.contract is DeliveryContract.READABLE
    assert decision.plan.degrade_policy is DegradePolicy.ALLOW_DEGRADE


def test_auto_paper_routes_to_readable() -> None:
    decision = _plan("auto", document_mode="paper")
    assert decision.effective_mode == "reflow"
    assert decision.plan.contract is DeliveryContract.READABLE
    assert decision.plan.default_page_strategy is PageStrategy.DOCUMENT_REFLOW


def test_auto_book_routes_to_readable() -> None:
    decision = _plan("auto", document_mode="book")
    assert decision.effective_mode == "reflow"
    assert decision.plan.contract is DeliveryContract.READABLE


def test_auto_non_pdf_routes_to_readable() -> None:
    decision = _plan("auto", is_pdf=False, document_mode="book")
    assert decision.effective_mode == "reflow"
    assert decision.plan.contract is DeliveryContract.READABLE


def test_explicit_readable_always_maps_to_reflow() -> None:
    decision = _plan("readable", document_mode="book")
    assert decision.effective_mode == "reflow"
    assert decision.plan.contract is DeliveryContract.READABLE


def test_legacy_facsimile_is_grandfathered_on_a_scanned_source() -> None:
    decision = _plan("facsimile", has_scan=True)
    assert decision.legacy is True
    assert decision.effective_mode == "facsimile"
    assert decision.plan.feasible is True
    assert decision.plan.contract is DeliveryContract.FAITHFUL
    assert decision.plan.eligibility.provisional is True


def test_legacy_page_locked_keeps_its_renderer() -> None:
    decision = _plan("page_locked")
    assert decision.legacy is True
    assert decision.effective_mode == "page_locked"
    assert decision.plan.contract is DeliveryContract.FAITHFUL


def test_legacy_reflow_keeps_its_renderer() -> None:
    decision = _plan("reflow")
    assert decision.legacy is True
    assert decision.effective_mode == "reflow"
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


def test_legacy_mode_emits_one_deprecation_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    _warn_legacy_mode.cache_clear()
    with caplog.at_level(logging.WARNING, logger="ubt.delivery.planning"):
        _plan("facsimile")
        _plan("facsimile")
    assert caplog.text.count("legacy compatibility value") == 1
    assert "2027-03-31" in caplog.text


def test_target_mode_does_not_emit_legacy_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    _warn_legacy_mode.cache_clear()
    with caplog.at_level(logging.WARNING, logger="ubt.delivery.planning"):
        _plan("auto")
    assert "legacy compatibility value" not in caplog.text
