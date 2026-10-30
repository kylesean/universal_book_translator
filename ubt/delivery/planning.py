"""Document facts -> DeliverySignals -> DeliveryPlan (Phase 0 wiring).

The router is deliberately pure; this module is the narrow adapter between the
PDF structure census and the router. It keeps the compatibility rule in one
place:

- legacy ``facsimile`` / ``page_locked`` keep their old renderer and are
  grandfathered as ``faithful`` for one release (eligibility is marked
  provisional, not silently re-evaluated);
- new ``auto`` / ``readable`` / ``faithful`` use the router;
- non-PDF targets can never be ``faithful`` (there is no source page to lock).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cache

from ubt.delivery.contracts import (
    DegradePolicy,
    DeliveryContract,
    DeliveryMode,
    DeliveryPlan,
    DeliverySignals,
    DocumentClass,
    ExpansionRisk,
    FaithfulEligibility,
    LayoutComplexity,
    PageKind,
    PageStrategy,
    legacy_delivery_contract,
)
from ubt.delivery.router import route_delivery

logger = logging.getLogger(__name__)

#: Legacy delivery values remain accepted for one release after the G1 default
#: switch. This date is the documented sunset window; removing the values is a
#: separate compatibility PR, not an implicit cleanup.
LEGACY_DELIVERY_SUNSET = "2027-03-31"

_LEGACY_MODES: frozenset[str] = frozenset({"reflow", "facsimile", "page_locked"})
_TARGET_MODES: frozenset[str] = frozenset({mode.value for mode in DeliveryMode}) - _LEGACY_MODES

#: Provisional text trust for a clean native PDF with no confirmed font damage:
#: the extraction witness has not produced a calibrated score yet, but the page
#: is not known to be damaged. G2 replaces this with the measured value.
_CLEAN_NATIVE_TEXT_TRUST = 0.95


@dataclass(frozen=True)
class DeliveryDecision:
    """The resolved plan plus the renderer mode it maps to.

    ``effective_mode=None`` means no contract is feasible; the caller must fail
    before rendering. ``legacy=True`` records that a current public mode was
    preserved rather than re-routed through the new eligibility rules.
    """

    plan: DeliveryPlan
    effective_mode: str | None
    legacy: bool = False


def document_class_from_mode(document_mode: str) -> DocumentClass:
    """Map the existing paper/book/auto vocabulary to the routing matrix.

    Contract/form/design have no automatic signal in the current config; a user
    who needs those must request ``faithful`` explicitly. Guessing them from
    file content would be worse than the safe ``other -> readable`` default.
    """
    if document_mode == "paper":
        return DocumentClass.ACADEMIC
    if document_mode == "book":
        return DocumentClass.LONG_BOOK
    return DocumentClass.OTHER


def build_delivery_signals(
    *,
    document_mode: str,
    is_pdf: bool,
    has_scan: bool = False,
    font_damage_pages: int = 0,
    page_count: int = 0,
    multicolumn_page_share: float = 0.0,
    structural_page_share: float = 0.0,
    formula_heavy: bool = False,
) -> DeliverySignals:
    """Derive the router's inputs from the Phase 0 structure census.

    The census is a *prefilter*, not a page-lock feasibility proof. A clean
    native PDF is marked ``page_lock_feasible=True`` provisionally; G2's
    MeasurementService replaces that with the real per-page feasibility result.
    A scanned or font-damaged PDF is vetoed here, because no later stage can
    make its text layer trustworthy without OCR/VLM.
    """
    document_class = document_class_from_mode(document_mode)
    if not is_pdf:
        return DeliverySignals(
            text_trust=1.0,
            page_kind=PageKind.NATIVE,
            layout_complexity=LayoutComplexity.LOW,
            source_overlap=False,
            expansion_risk=ExpansionRisk.LOW,
            page_lock_feasible=False,
            document_class=document_class,
        )

    if has_scan:
        text_trust = 0.0
    elif font_damage_pages > 0:
        text_trust = (
            max(0.0, 1.0 - float(font_damage_pages) / float(page_count)) if page_count > 0 else 0.0
        )
    else:
        text_trust = _CLEAN_NATIVE_TEXT_TRUST

    high_complexity = formula_heavy or structural_page_share >= 0.5 or multicolumn_page_share >= 0.5
    medium_complexity = formula_heavy or structural_page_share > 0.0 or multicolumn_page_share > 0.0
    return DeliverySignals(
        text_trust=text_trust,
        page_kind=PageKind.SCANNED if has_scan else PageKind.NATIVE,
        layout_complexity=(
            LayoutComplexity.HIGH
            if high_complexity
            else (LayoutComplexity.MEDIUM if medium_complexity else LayoutComplexity.LOW)
        ),
        source_overlap=False,
        expansion_risk=ExpansionRisk.LOW,
        # No per-page solve in Phase 0. A clean native page is provisionally
        # feasible; a scan/font-damaged page is vetoed outright.
        page_lock_feasible=not has_scan and font_damage_pages == 0,
        document_class=document_class,
    )


def plan_for_document(
    requested: str,
    *,
    document_mode: str,
    is_pdf: bool,
    has_scan: bool = False,
    font_damage_pages: int = 0,
    page_count: int = 0,
    multicolumn_page_share: float = 0.0,
    structural_page_share: float = 0.0,
    formula_heavy: bool = False,
    allow_degrade: bool = False,
) -> DeliveryDecision:
    """Resolve the requested mode into a plan and an effective renderer mode.

    Raises ``ValueError`` for a mode outside both vocabularies. An infeasible
    new ``faithful`` request returns ``effective_mode=None``; the pipeline
    raises before spending a model call. Legacy modes remain feasible by
    compatibility decision, with provisional eligibility recorded in the plan.
    """
    normalized = requested.strip().lower()
    if normalized not in _LEGACY_MODES and normalized not in _TARGET_MODES:
        allowed = ", ".join(sorted(_LEGACY_MODES | _TARGET_MODES))
        raise ValueError(f"unknown delivery mode {requested!r}; expected one of: {allowed}")

    signals = build_delivery_signals(
        document_mode=document_mode,
        is_pdf=is_pdf,
        has_scan=has_scan,
        font_damage_pages=font_damage_pages,
        page_count=page_count,
        multicolumn_page_share=multicolumn_page_share,
        structural_page_share=structural_page_share,
        formula_heavy=formula_heavy,
    )
    policy = DegradePolicy.ALLOW_DEGRADE if allow_degrade else DegradePolicy.FAIL

    if normalized in _LEGACY_MODES:
        _warn_legacy_mode(normalized)
        return _legacy_decision(normalized, signals, policy)

    requested_mode = DeliveryMode(normalized)
    plan = route_delivery(signals, requested=requested_mode, allow_degrade=allow_degrade)
    if plan.contract is None:
        return DeliveryDecision(plan=plan, effective_mode=None)
    if plan.contract is DeliveryContract.FAITHFUL:
        return DeliveryDecision(
            plan=_mark_faithful_provisional(plan),
            effective_mode="facsimile",
        )
    return DeliveryDecision(plan=plan, effective_mode="reflow")


@cache
def _warn_legacy_mode(mode: str) -> None:
    """Emit one deprecation warning per legacy value per process."""
    logger.warning(
        "delivery_mode=%r is a legacy compatibility value; it maps to %r and "
        "will be removed after %s. Use auto/readable/faithful instead.",
        mode,
        legacy_delivery_contract(mode).value,
        LEGACY_DELIVERY_SUNSET,
    )


def _mark_faithful_provisional(plan: DeliveryPlan) -> DeliveryPlan:
    """Record that Phase 0's feasibility signal is a census, not a solve."""
    return plan.model_copy(
        update={
            "eligibility": plan.eligibility.model_copy(update={"provisional": True}),
            "reasons": (
                *plan.reasons,
                "page_lock_feasible is provisional (structure census; G2 MeasurementService pending)",
            ),
        }
    )


def _legacy_decision(
    legacy_mode: str,
    signals: DeliverySignals,
    policy: DegradePolicy,
) -> DeliveryDecision:
    contract = legacy_delivery_contract(legacy_mode)
    if contract is DeliveryContract.READABLE:
        plan = DeliveryPlan(
            requested_mode=DeliveryMode.READABLE,
            contract=DeliveryContract.READABLE,
            feasible=True,
            eligibility=FaithfulEligibility(eligible=False),
            default_page_strategy=PageStrategy.DOCUMENT_REFLOW,
            degrade_policy=policy,
            reasons=(f"legacy mode {legacy_mode!r} preserved",),
            signals=signals,
        )
        return DeliveryDecision(plan=plan, effective_mode="reflow", legacy=True)

    plan = DeliveryPlan(
        requested_mode=DeliveryMode.FAITHFUL,
        contract=DeliveryContract.FAITHFUL,
        feasible=True,
        # Legacy compatibility is not the new eligibility proof. Mark it
        # provisional so a report cannot mistake the grandfather clause for a
        # measured page-lock guarantee.
        eligibility=FaithfulEligibility(
            eligible=True,
            provisional=True,
            reasons=("legacy mode compatibility; eligibility not re-evaluated in Phase 0",),
        ),
        default_page_strategy=PageStrategy.CANVAS,
        degrade_policy=policy,
        reasons=(f"legacy mode {legacy_mode!r} preserved",),
        signals=signals,
    )
    return DeliveryDecision(plan=plan, effective_mode=legacy_mode, legacy=True)
