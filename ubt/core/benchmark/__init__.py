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
from ubt.core.benchmark.pareto import (
    ModelPoint,
    ParetoAnalysisReport,
    ParetoFrontier,
    RecommendationCategory,
    build_pareto_from_benchmark_reports,
    compute_pareto_frontier,
    dominates,
    simulate_model_pareto,
)
from ubt.core.benchmark.pareto_reporters import (
    render_pareto_html_dashboard,
    render_pareto_json,
    render_pareto_markdown,
    render_pareto_terminal,
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
    "ModelPoint",
    "ParetoAnalysisReport",
    "ParetoFrontier",
    "QualityBenchmarkMetrics",
    "RecommendationCategory",
    "build_pareto_from_benchmark_reports",
    "compute_benchmark_drift",
    "compute_pareto_frontier",
    "dominates",
    "extract_cost_metrics",
    "extract_quality_metrics",
    "render_html_dashboard",
    "render_json_report",
    "render_markdown_report",
    "render_pareto_html_dashboard",
    "render_pareto_json",
    "render_pareto_markdown",
    "render_pareto_terminal",
    "render_terminal_dashboard",
    "select_layout_spec_for_case",
    "simulate_model_pareto",
]
