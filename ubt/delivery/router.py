"""Deterministic delivery routing (the Phase 0 ``auto`` contract).

The router answers one question: *which contract can this document honestly
promise?* It does not render, and it does not silently downgrade a user's
explicit ``faithful`` request. ``feasible=False`` is a first-class result that
the export stage must turn into an explicit failure or an authorized
``allow_degrade`` re-route.

The rules mirror the PRD's §5.2 matrix and §5.3 eligibility formula. They are
intentionally pure functions over :class:`DeliverySignals` so routing can be
unit-tested without a PDF, Typst, or a model call.
"""

from __future__ import annotations

from ubt.delivery.contracts import (
    DegradePolicy,
    DeliveryContract,
    DeliveryMode,
    DeliveryPlan,
    DeliverySignals,
    DocumentClass,
    FaithfulEligibility,
    PageKind,
    PageStrategy,
)

#: Text trust below this value forbids ``faithful``; the page must be OCR/VLM
#: repaired and re-flowed instead.
MIN_FAITHFUL_TEXT_TRUST = 0.80

#: Document classes whose reading value beats page identity. Academic and
#: technical documents are the PRD's default ``readable`` users even when the
#: source happens to be page-lock feasible: reflow can expand pages for Chinese
#: and keeps formulas/figures in reading order.
_READABLE_FIRST_CLASSES: frozenset[DocumentClass] = frozenset(
    {
        DocumentClass.ACADEMIC,
        DocumentClass.TECHNICAL,
        DocumentClass.LONG_BOOK,
        DocumentClass.MOBILE,
    }
)

#: Classes where page identity is part of the content. A damaged source in
#: these classes must fail or be explicitly degraded, never silently reflowed.
_PAGE_IDENTITY_CLASSES: frozenset[DocumentClass] = frozenset(
    {
        DocumentClass.CONTRACT,
        DocumentClass.FORM,
        DocumentClass.DESIGN,
    }
)


def evaluate_faithful_eligibility(signals: DeliverySignals) -> FaithfulEligibility:
    """Apply the PRD's ``faithful`` eligibility formula.

    A page-lock promise requires a trustworthy native text layer, no source
    overlap, and a measured feasible page-lock solution at the body floor.
    Layout complexity and expansion risk do *not* disqualify a document: CELS
    exists to handle those, and ``page_lock_feasible`` is the measured veto.
    """
    reasons: list[str] = []
    if signals.text_trust < MIN_FAITHFUL_TEXT_TRUST:
        reasons.append(f"text_trust={signals.text_trust:.2f} < {MIN_FAITHFUL_TEXT_TRUST:.2f}")
    if signals.page_kind is not PageKind.NATIVE:
        reasons.append(f"page_kind={signals.page_kind.value!r} is not native")
    if signals.source_overlap:
        reasons.append("source_overlap=true")
    if not signals.page_lock_feasible:
        reasons.append("page_lock_feasible=false")
    return FaithfulEligibility(eligible=not reasons, reasons=tuple(reasons))


def route_delivery(
    signals: DeliverySignals,
    *,
    requested: DeliveryMode = DeliveryMode.AUTO,
    allow_degrade: bool = False,
) -> DeliveryPlan:
    """Choose the honest contract for ``signals``.

    ``AUTO`` is deliberately conservative:

    - academic / technical / long-book / mobile -> ``readable``;
    - contract / form / design -> ``faithful`` only if eligible, otherwise
      infeasible unless ``allow_degrade`` is set;
    - unknown class -> ``readable`` (the safe default, never an accidental
      page-lock promise).

    An explicit request still wins when it is feasible. An explicit
    ``faithful`` request that is not eligible produces ``feasible=False``
    unless the caller explicitly authorized degradation.
    """
    eligibility = evaluate_faithful_eligibility(signals)
    policy = DegradePolicy.ALLOW_DEGRADE if allow_degrade else DegradePolicy.FAIL

    if requested is DeliveryMode.READABLE:
        return _readable_plan(
            signals,
            eligibility,
            requested,
            policy,
            reasons=("explicit readable request",),
        )

    if requested is DeliveryMode.FAITHFUL:
        return _faithful_request_plan(
            signals,
            eligibility,
            requested,
            policy,
            allow_degrade=allow_degrade,
        )

    # requested is AUTO.
    if signals.document_class in _READABLE_FIRST_CLASSES:
        return _readable_plan(
            signals,
            eligibility,
            requested,
            policy,
            reasons=(f"document_class={signals.document_class.value!r} defaults to readable",),
        )
    if signals.document_class in _PAGE_IDENTITY_CLASSES:
        if eligibility.eligible:
            return _faithful_plan(
                signals,
                eligibility,
                requested,
                policy,
                reasons=("layout-sensitive document is faithful-eligible",),
            )
        if allow_degrade:
            return _degraded_plan(signals, eligibility, requested, policy)
        return _infeasible_plan(signals, eligibility, requested)
    return _readable_plan(
        signals,
        eligibility,
        requested,
        policy,
        reasons=("no layout-sensitive contract requested; readable is the safe default",),
    )


def _faithful_request_plan(
    signals: DeliverySignals,
    eligibility: FaithfulEligibility,
    requested: DeliveryMode,
    policy: DegradePolicy,
    *,
    allow_degrade: bool,
) -> DeliveryPlan:
    if eligibility.eligible:
        return _faithful_plan(
            signals,
            eligibility,
            requested,
            policy,
            reasons=("explicit faithful request and eligibility passed",),
        )
    if allow_degrade:
        return _degraded_plan(signals, eligibility, requested, policy)
    return _infeasible_plan(signals, eligibility, requested)


def _readable_plan(
    signals: DeliverySignals,
    eligibility: FaithfulEligibility,
    requested: DeliveryMode,
    policy: DegradePolicy,
    *,
    reasons: tuple[str, ...],
) -> DeliveryPlan:
    strategy = (
        PageStrategy.OCR_REFLOW
        if signals.page_kind in (PageKind.SCANNED, PageKind.MIXED)
        else PageStrategy.DOCUMENT_REFLOW
    )
    return DeliveryPlan(
        requested_mode=requested,
        contract=DeliveryContract.READABLE,
        feasible=True,
        eligibility=eligibility,
        default_page_strategy=strategy,
        degrade_policy=policy,
        reasons=reasons,
        signals=signals,
    )


def _faithful_plan(
    signals: DeliverySignals,
    eligibility: FaithfulEligibility,
    requested: DeliveryMode,
    policy: DegradePolicy,
    *,
    reasons: tuple[str, ...],
) -> DeliveryPlan:
    return DeliveryPlan(
        requested_mode=requested,
        contract=DeliveryContract.FAITHFUL,
        feasible=True,
        eligibility=eligibility,
        default_page_strategy=PageStrategy.CANVAS,
        degrade_policy=policy,
        companion_reflow=True,
        reasons=reasons,
        signals=signals,
    )


def _degraded_plan(
    signals: DeliverySignals,
    eligibility: FaithfulEligibility,
    requested: DeliveryMode,
    policy: DegradePolicy,
) -> DeliveryPlan:
    return _readable_plan(
        signals,
        eligibility,
        requested,
        policy,
        reasons=(
            *eligibility.reasons,
            "faithful is not eligible; degraded to readable because allow_degrade is set",
        ),
    )


def _infeasible_plan(
    signals: DeliverySignals,
    eligibility: FaithfulEligibility,
    requested: DeliveryMode,
) -> DeliveryPlan:
    return DeliveryPlan(
        requested_mode=requested,
        contract=None,
        feasible=False,
        eligibility=eligibility,
        default_page_strategy=None,
        degrade_policy=DegradePolicy.FAIL,
        reasons=(
            *eligibility.reasons,
            "faithful is not eligible; export must fail or be re-run with allow_degrade",
        ),
        signals=signals,
    )
