"""Unit tests for multi-model quality vs. cost Pareto frontier evaluation engine."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ubt.core.benchmark.models import (
    CaseBenchmarkResult,
    CorpusBenchmarkReport,
    CostBenchmarkMetrics,
    QualityBenchmarkMetrics,
)
from ubt.core.benchmark.pareto import (
    ModelPoint,
    ParetoAnalysisReport,
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

pytestmark = pytest.mark.fast


def test_dominates_logic() -> None:
    # A dominates B if A is cheaper and higher quality
    a = ModelPoint(
        model_name="model_a", quality_score=0.95, cost_per_10k_words=0.05, total_cost_usd=0.05
    )
    b = ModelPoint(
        model_name="model_b", quality_score=0.90, cost_per_10k_words=0.10, total_cost_usd=0.10
    )
    assert dominates(a, b) is True
    assert dominates(b, a) is False

    # Trade-off: C is cheaper but lower quality -> neither dominates
    c = ModelPoint(
        model_name="model_c", quality_score=0.88, cost_per_10k_words=0.02, total_cost_usd=0.02
    )
    assert dominates(a, c) is False
    assert dominates(c, a) is False

    # Equal points: identical points do not dominate each other
    d = ModelPoint(
        model_name="model_d", quality_score=0.95, cost_per_10k_words=0.05, total_cost_usd=0.05
    )
    assert dominates(a, d) is False


def test_compute_pareto_frontier_basic() -> None:
    points = [
        ModelPoint(
            model_name="free_good",
            quality_score=0.80,
            cost_per_10k_words=0.0,
            total_cost_usd=0.0,
            throughput_cps=50,
        ),
        ModelPoint(
            model_name="cheap_bad",
            quality_score=0.70,
            cost_per_10k_words=0.01,
            total_cost_usd=0.01,
            throughput_cps=40,
        ),  # dominated by free_good
        ModelPoint(
            model_name="mid_sweet",
            quality_score=0.92,
            cost_per_10k_words=0.03,
            total_cost_usd=0.03,
            throughput_cps=120,
        ),
        ModelPoint(
            model_name="expensive_flagship",
            quality_score=0.98,
            cost_per_10k_words=0.15,
            total_cost_usd=0.15,
            throughput_cps=100,
        ),
        ModelPoint(
            model_name="expensive_worse",
            quality_score=0.91,
            cost_per_10k_words=0.18,
            total_cost_usd=0.18,
            throughput_cps=80,
        ),  # dominated
    ]

    frontier = compute_pareto_frontier(points, min_qe_threshold=0.75)

    pareto_names = [p.model_name for p in frontier.frontier_points]
    assert "free_good" in pareto_names
    assert "mid_sweet" in pareto_names
    assert "expensive_flagship" in pareto_names
    assert "cheap_bad" not in pareto_names
    assert "expensive_worse" not in pareto_names

    # Check recommendations
    assert frontier.budget_pick is not None
    assert frontier.budget_pick.model_name == "free_good"
    assert frontier.flagship_pick is not None
    assert frontier.flagship_pick.model_name == "expensive_flagship"
    assert frontier.sweet_spot is not None
    assert frontier.sweet_spot.model_name in {"free_good", "mid_sweet"}


def test_compute_pareto_frontier_empty() -> None:
    frontier = compute_pareto_frontier([])
    assert frontier.frontier_points == []
    assert frontier.all_points == []
    assert frontier.sweet_spot is None
    assert frontier.budget_pick is None
    assert frontier.flagship_pick is None
    assert frontier.speed_pick is None


def test_simulate_model_pareto() -> None:
    report = simulate_model_pareto(sample_words=10000, min_qe_threshold=0.80)
    assert isinstance(report, ParetoAnalysisReport)
    assert report.corpus_id == "contemporary_corpus_eval"
    assert len(report.frontier.all_points) > 0
    assert len(report.frontier.frontier_points) > 0

    # Ensure sweet spot exists and has marginal return calculated
    assert report.frontier.sweet_spot is not None
    assert report.frontier.flagship_pick is not None
    assert report.frontier.budget_pick is not None


def test_simulate_model_pareto_custom_candidates() -> None:
    custom_models = ["deepseek-chat", "gpt-4o", "claude-3-5-sonnet"]
    report = simulate_model_pareto(candidate_models=custom_models, sample_words=5000)
    names = {p.model_name for p in report.frontier.all_points}
    assert names == set(custom_models)


def test_build_pareto_from_benchmark_reports() -> None:
    report1 = CorpusBenchmarkReport(
        run_id="run_01",
        timestamp="2026-10-28T00:00:00Z",
        corpus_dir="corpus",
        mode="eval-only",
        total_cases=1,
        passed_cases=1,
        overall_passed=True,
        avg_delivered_ratio=1.0,
        total_tokens=10000,
        total_cost_usd=0.03,
        total_runtime_sec=20.0,
        cases=[
            CaseBenchmarkResult(
                id="case_01",
                quality=QualityBenchmarkMetrics(qe_measured=True, avg_qe=0.92, term_precision=0.95),
                cost=CostBenchmarkMetrics(estimated_cost_usd=0.03),
            )
        ],
    )
    report2 = CorpusBenchmarkReport(
        run_id="run_02",
        timestamp="2026-10-28T00:00:00Z",
        corpus_dir="corpus",
        mode="eval-only",
        total_cases=1,
        passed_cases=1,
        overall_passed=True,
        avg_delivered_ratio=1.0,
        total_tokens=10000,
        total_cost_usd=0.15,
        total_runtime_sec=30.0,
        cases=[
            CaseBenchmarkResult(
                id="case_01",
                quality=QualityBenchmarkMetrics(qe_measured=True, avg_qe=0.97, term_precision=0.99),
                cost=CostBenchmarkMetrics(estimated_cost_usd=0.15),
            )
        ],
    )

    reports_map = {
        "deepseek-chat": report1,
        "claude-3-5-sonnet": report2,
    }

    report = build_pareto_from_benchmark_reports(reports_map)
    assert len(report.frontier.all_points) == 2
    assert len(report.frontier.frontier_points) == 2
    assert report.frontier.flagship_pick is not None
    assert report.frontier.flagship_pick.model_name == "claude-3-5-sonnet"


def test_pareto_reporters(tmp_path: Path) -> None:
    report = simulate_model_pareto(sample_words=10000)

    # 1. Terminal (should not crash)
    render_pareto_terminal(report)

    # 2. Markdown
    md_content = render_pareto_markdown(report)
    assert "# Multi-Model Quality vs. Cost Pareto Frontier Analysis" in md_content
    assert "Sweet Spot" in md_content
    assert "Evaluated Model Options Matrix" in md_content

    # 3. HTML Dashboard
    html_content = render_pareto_html_dashboard(report)
    assert "<!DOCTYPE html>" in html_content
    assert "<svg" in html_content  # embedded SVG chart
    assert "cdn." not in html_content.lower()  # zero external CDN
    assert "<script" not in html_content.lower()  # zero external scripts
    assert '<link rel="stylesheet"' not in html_content.lower()  # zero external CSS

    # 4. JSON export
    json_content = render_pareto_json(report)
    parsed = json.loads(json_content)
    assert "frontier" in parsed
    assert "all_points" in parsed["frontier"]
    assert len(parsed["frontier"]["all_points"]) > 0
