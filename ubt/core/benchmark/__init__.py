"""Industrial corpus automated benchmark runner, multi-dimensional metrics, and drift dashboard."""

from __future__ import annotations

from ubt.core.benchmark.drift import (
    DEFAULT_DRIFT_TOLERANCES,
    compute_benchmark_drift,
)
from ubt.core.benchmark.models import (
    BenchmarkDriftReport,
    BenchmarkMode,
    CaseBenchmarkResult,
    CaseDrift,
    ContractBenchmarkMetrics,
    CorpusBenchmarkReport,
    CostBenchmarkMetrics,
    LayoutBenchmarkMetrics,
    QualityBenchmarkMetrics,
)
from ubt.core.benchmark.reporters import (
    render_html_dashboard,
    render_json_report,
    render_markdown_report,
    render_terminal_dashboard,
)
from ubt.core.benchmark.runner import (
    BenchmarkRunner,
    extract_cost_metrics,
    extract_quality_metrics,
    select_layout_spec_for_case,
)

__all__ = [
    "DEFAULT_DRIFT_TOLERANCES",
    "BenchmarkDriftReport",
    "BenchmarkMode",
    "BenchmarkRunner",
    "CaseBenchmarkResult",
    "CaseDrift",
    "ContractBenchmarkMetrics",
    "CorpusBenchmarkReport",
    "CostBenchmarkMetrics",
    "LayoutBenchmarkMetrics",
    "QualityBenchmarkMetrics",
    "compute_benchmark_drift",
    "extract_cost_metrics",
    "extract_quality_metrics",
    "render_html_dashboard",
    "render_json_report",
    "render_markdown_report",
    "render_terminal_dashboard",
    "select_layout_spec_for_case",
]
