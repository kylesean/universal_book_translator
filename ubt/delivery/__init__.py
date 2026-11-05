"""Adaptive delivery vocabulary: contracts, routing, planning.

This package carries the adaptive-delivery vocabulary: the request/response
contract, the pure-function router and the structure-census planner.
``plan_for_document`` is wired into the pipeline preflight (CLI, API and MCP
requests all funnel through it), with legacy modes mapped through
:func:`ubt.delivery.contracts.legacy_delivery_contract`.

The CELS layout and font solvers live in ``ubt.core.layout`` and
``ubt.adapters.pdf.cels_*``. The artifact-side verdict lives in
``ubt.core.engine.delivery_contract`` and the PDF evidence reader in
``ubt.adapters.pdf.delivery_evidence``; this package does not duplicate those
checks. It owns the routing contract and the ``DeliveryPlan``.
"""

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
from ubt.delivery.errors import DeliveryInfeasibleError
from ubt.delivery.planning import (
    DeliveryDecision,
    build_delivery_signals,
    document_class_from_mode,
    plan_for_document,
)
from ubt.delivery.router import evaluate_faithful_eligibility, route_delivery

__all__ = [
    "DegradePolicy",
    "DeliveryContract",
    "DeliveryDecision",
    "DeliveryInfeasibleError",
    "DeliveryMode",
    "DeliveryPlan",
    "DeliverySignals",
    "DocumentClass",
    "ExpansionRisk",
    "FaithfulEligibility",
    "LayoutComplexity",
    "PageKind",
    "PageStrategy",
    "build_delivery_signals",
    "document_class_from_mode",
    "evaluate_faithful_eligibility",
    "legacy_delivery_contract",
    "plan_for_document",
    "route_delivery",
]
