"""Versioned KPI layer over the quality and visual reports.

Public surface: the KPI registry (:mod:`schema`), the artifact collector
(:mod:`collect`) and the regression/threshold gates (:mod:`compare`).
"""

from __future__ import annotations

from ubt.core.metrics.collect import collect_kpis, load_kpis, save_metrics_report
from ubt.core.metrics.compare import (
    KpiDelta,
    RegressionReport,
    check_thresholds,
    compare_kpi_sets,
    compare_kpis,
)
from ubt.core.metrics.schema import (
    KPI_BY_NAME,
    KPI_DEFINITIONS,
    SCHEMA_VERSION,
    Direction,
    KpiDefinition,
    KpiSet,
)

__all__ = [
    "KPI_BY_NAME",
    "KPI_DEFINITIONS",
    "SCHEMA_VERSION",
    "Direction",
    "KpiDefinition",
    "KpiDelta",
    "KpiSet",
    "RegressionReport",
    "check_thresholds",
    "collect_kpis",
    "compare_kpi_sets",
    "compare_kpis",
    "load_kpis",
    "save_metrics_report",
]
