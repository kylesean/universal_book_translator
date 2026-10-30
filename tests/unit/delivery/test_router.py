"""Delivery router tests: routing matrix, eligibility, and fail-closed plans."""

from __future__ import annotations

import pytest

from ubt.delivery.contracts import (
    DegradePolicy,
    DeliveryContract,
    DeliveryMode,
    DeliverySignals,
    DocumentClass,
    ExpansionRisk,
    LayoutComplexity,
    PageKind,
    PageStrategy,
)
from ubt.delivery.router import evaluate_faithful_eligibility, route_delivery

pytestmark = pytest.mark.fast


def _signals(
    *,
    document_class: DocumentClass = DocumentClass.ACADEMIC,
    text_trust: float = 1.0,
    page_kind: PageKind = PageKind.NATIVE,
    layout_complexity: LayoutComplexity = LayoutComplexity.MEDIUM,
    source_overlap: bool = False,
    expansion_risk: ExpansionRisk = ExpansionRisk.LOW,
    page_lock_feasible: bool = True,
) -> DeliverySignals:
    return DeliverySignals(
        document_class=document_class,
        text_trust=text_trust,
        page_kind=page_kind,
        layout_complexity=layout_complexity,
        source_overlap=source_overlap,
        expansion_risk=expansion_risk,
        page_lock_feasible=page_lock_feasible,
    )


def test_academic_defaults_to_readable() -> None:
    plan = route_delivery(_signals(document_class=DocumentClass.ACADEMIC))
    assert plan.feasible
    assert plan.contract is DeliveryContract.READABLE
    assert plan.default_page_strategy is PageStrategy.DOCUMENT_REFLOW
    assert plan.degrade_policy is DegradePolicy.FAIL


def test_technical_long_book_and_mobile_default_to_readable() -> None:
    for document_class in (
        DocumentClass.TECHNICAL,
        DocumentClass.LONG_BOOK,
        DocumentClass.MOBILE,
    ):
        plan = route_delivery(_signals(document_class=document_class))
        assert plan.contract is DeliveryContract.READABLE


def test_contract_document_defaults_to_faithful_when_eligible() -> None:
    plan = route_delivery(_signals(document_class=DocumentClass.CONTRACT))
    assert plan.feasible
    assert plan.contract is DeliveryContract.FAITHFUL
    assert plan.default_page_strategy is PageStrategy.CANVAS
    assert plan.eligibility.eligible


def test_unknown_document_defaults_to_readable() -> None:
    plan = route_delivery(_signals(document_class=DocumentClass.OTHER))
    assert plan.contract is DeliveryContract.READABLE


def test_scanned_academic_uses_ocr_reflow() -> None:
    plan = route_delivery(
        _signals(document_class=DocumentClass.ACADEMIC, page_kind=PageKind.SCANNED)
    )
    assert plan.contract is DeliveryContract.READABLE
    assert plan.default_page_strategy is PageStrategy.OCR_REFLOW


def test_low_text_trust_disqualifies_faithful() -> None:
    eligibility = evaluate_faithful_eligibility(
        _signals(document_class=DocumentClass.CONTRACT, text_trust=0.5)
    )
    assert not eligibility.eligible
    assert any("text_trust" in reason for reason in eligibility.reasons)


def test_scanned_page_disqualifies_faithful() -> None:
    eligibility = evaluate_faithful_eligibility(
        _signals(document_class=DocumentClass.CONTRACT, page_kind=PageKind.SCANNED)
    )
    assert not eligibility.eligible
    assert any("page_kind" in reason for reason in eligibility.reasons)


def test_source_overlap_disqualifies_faithful() -> None:
    eligibility = evaluate_faithful_eligibility(
        _signals(document_class=DocumentClass.CONTRACT, source_overlap=True)
    )
    assert not eligibility.eligible
    assert any("source_overlap" in reason for reason in eligibility.reasons)


def test_infeasible_page_lock_disqualifies_faithful() -> None:
    eligibility = evaluate_faithful_eligibility(
        _signals(document_class=DocumentClass.CONTRACT, page_lock_feasible=False)
    )
    assert not eligibility.eligible
    assert any("page_lock_feasible" in reason for reason in eligibility.reasons)


def test_high_layout_complexity_does_not_disqualify_faithful() -> None:
    plan = route_delivery(
        _signals(
            document_class=DocumentClass.CONTRACT,
            layout_complexity=LayoutComplexity.HIGH,
        )
    )
    assert plan.contract is DeliveryContract.FAITHFUL


def test_high_expansion_risk_does_not_disqualify_faithful_when_page_lock_is_feasible() -> None:
    plan = route_delivery(
        _signals(
            document_class=DocumentClass.CONTRACT,
            expansion_risk=ExpansionRisk.HIGH,
        )
    )
    assert plan.contract is DeliveryContract.FAITHFUL


def test_explicit_readable_wins_for_contract_document() -> None:
    plan = route_delivery(
        _signals(document_class=DocumentClass.CONTRACT),
        requested=DeliveryMode.READABLE,
    )
    assert plan.contract is DeliveryContract.READABLE
    assert plan.reasons == ("explicit readable request",)


def test_explicit_faithful_wins_on_academic_document_when_eligible() -> None:
    plan = route_delivery(
        _signals(document_class=DocumentClass.ACADEMIC),
        requested=DeliveryMode.FAITHFUL,
    )
    assert plan.contract is DeliveryContract.FAITHFUL


def test_explicit_faithful_is_infeasible_without_allow_degrade() -> None:
    plan = route_delivery(
        _signals(document_class=DocumentClass.CONTRACT, text_trust=0.5),
        requested=DeliveryMode.FAITHFUL,
    )
    assert not plan.feasible
    assert plan.contract is None
    assert plan.degrade_policy is DegradePolicy.FAIL
    assert any("allow_degrade" in reason for reason in plan.reasons)


def test_explicit_faithful_degrades_to_readable_with_allow_degrade() -> None:
    plan = route_delivery(
        _signals(document_class=DocumentClass.CONTRACT, text_trust=0.5),
        requested=DeliveryMode.FAITHFUL,
        allow_degrade=True,
    )
    assert plan.feasible
    assert plan.contract is DeliveryContract.READABLE
    assert plan.degrade_policy is DegradePolicy.ALLOW_DEGRADE
    assert any("degraded to readable" in reason for reason in plan.reasons)


def test_auto_contract_infeasible_still_fails_closed() -> None:
    plan = route_delivery(_signals(document_class=DocumentClass.CONTRACT, source_overlap=True))
    assert not plan.feasible
    assert plan.contract is None


def test_no_silent_downgrade_of_explicit_faithful() -> None:
    plan = route_delivery(
        _signals(document_class=DocumentClass.CONTRACT, page_kind=PageKind.SCANNED),
        requested=DeliveryMode.FAITHFUL,
    )
    assert plan.contract is None
    assert not plan.feasible
