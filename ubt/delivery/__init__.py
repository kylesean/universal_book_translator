"""Adaptive delivery vocabulary: contracts, routing, planning, measurement.

This package carries the target PRD's delivery vocabulary
(``docs/design/adaptive_delivery_prd.md``): the request/response contract,
the pure-function router, the structure-census planner, and the measurement
protocol/cache. ``plan_for_document`` is wired into the pipeline preflight
(CLI, API and MCP requests all funnel through it), with legacy modes mapped
through :func:`ubt.delivery.contracts.legacy_delivery_contract`.

Not wired yet: the MeasurementService has no production caller (fit decisions
still come from the render-side estimators), and the Stage 3A/5 CELS solvers
live in ``ubt.core.layout`` and ``ubt.adapters.pdf.cels_*``.

The artifact-side verdict lives in ``ubt.core.engine.delivery_contract`` and
the PDF evidence reader in ``ubt.adapters.pdf.delivery_evidence``; this package
does not duplicate those checks. It owns the routing contract, the
``DeliveryPlan``, and the measurement protocol/cache.
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
from ubt.delivery.measurement import (
    CachedMeasurementService,
    LineBox,
    MeasuredBlock,
    MeasurementError,
    MeasurementService,
    MeasurementStyle,
    measurement_cache_key,
)
from ubt.delivery.planning import (
    DeliveryDecision,
    build_delivery_signals,
    document_class_from_mode,
    plan_for_document,
)
from ubt.delivery.router import evaluate_faithful_eligibility, route_delivery

__all__ = [
    "CachedMeasurementService",
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
    "LineBox",
    "MeasuredBlock",
    "MeasurementError",
    "MeasurementService",
    "MeasurementStyle",
    "PageKind",
    "PageStrategy",
    "build_delivery_signals",
    "document_class_from_mode",
    "evaluate_faithful_eligibility",
    "legacy_delivery_contract",
    "measurement_cache_key",
    "plan_for_document",
    "route_delivery",
]
