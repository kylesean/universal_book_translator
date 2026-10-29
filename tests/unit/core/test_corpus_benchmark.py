"""Unit tests for corpus automated benchmark runner, models, drift detection, and reporters."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rich.console import Console

from ubt.core.benchmark.drift import (
    compute_benchmark_drift,
)
from ubt.core.benchmark.models import (
    BenchmarkMode,
    CaseBenchmarkResult,
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
from ubt.core.content.contract import (
    ReconciliationReport,
)
from ubt.core.content.verify import contract_path_for_artifact
from ubt.core.job_options import sidecar_path

pytestmark = pytest.mark.fast


def _make_dummy_pdf(path: Path) -> None:
    """Create a minimal valid 1-page PDF binary for layout diagnostics."""
    pdf_content = (
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 595.28 841.89] /Contents 4 0 R >> endobj\n"
        b"4 0 obj << /Length 50 >> stream\n"
        b"BT /F1 12 Tf 100 700 Td (Hello Benchmark) Tj ET\n"
        b"endstream endobj\n"
        b"xref\n0 5\n0000000000 65535 f \n"
        b"0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000214 00000 n \n"
        b"trailer << /Size 5 /Root 1 0 R >>\nstartxref\n314\n%%EOF\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pdf_content)


def test_benchmark_models_and_serialization(tmp_path: Path) -> None:
    """Verify data model construction, validation, and JSON roundtrip."""
    c_metrics = ContractBenchmarkMetrics(
        total_text=100,
        delivered_text=80,
        verbatim_text=10,
        delivered_ratio=0.8,
        accounted_ratio=0.9,
        contract_passed=True,
    )
    l_metrics = LayoutBenchmarkMetrics(
        total_pages=5,
        spill_pages_count=0,
        spill_rate=0.0,
        max_overflow_pt=0.0,
        layout_passed=True,
    )
    q_metrics = QualityBenchmarkMetrics(
        avg_qe=0.92,
        qe_measured=True,
        pass_rate=1.0,
    )
    cost_metrics = CostBenchmarkMetrics(
        prompt_tokens=1500,
        completion_tokens=800,
        cached_tokens=500,
        total_tokens=2300,
        cache_hit_rate=0.3333,
        estimated_cost_usd=0.0042,
    )

    case_res = CaseBenchmarkResult(
        id="case-test-1",
        description="Test case 1",
        status="pass",
        runtime_sec=1.23,
        contract=c_metrics,
        layout=l_metrics,
        quality=q_metrics,
        cost=cost_metrics,
    )
    assert case_res.passed is True

    report = CorpusBenchmarkReport(
        run_id="bench_test_run",
        timestamp="2026-10-28T12:00:00Z",
        corpus_dir=str(tmp_path),
        mode="dry-run",
        total_cases=1,
        passed_cases=1,
        failed_cases=0,
        skipped_cases=0,
        overall_passed=True,
        total_runtime_sec=1.23,
        avg_delivered_ratio=0.8,
        avg_accounted_ratio=0.9,
        cases=[case_res],
    )

    json_str = report.to_json()
    deserialized = CorpusBenchmarkReport.from_json(json_str)
    assert deserialized.run_id == "bench_test_run"
    assert len(deserialized.cases) == 1
    assert deserialized.cases[0].id == "case-test-1"

    out_file = tmp_path / "report.json"
    report.save(out_file)
    assert out_file.exists()


def test_select_layout_spec_for_case() -> None:
    """Verify correct layout specification mapping based on case name."""
    s1 = select_layout_spec_for_case("twocol-paper-2609")
    assert s1.columns == 2
    assert s1.name == "academic_dual"

    s2 = select_layout_spec_for_case("paper-2608")
    assert s2.columns == 1
    assert s2.name == "academic_single"

    s3 = select_layout_spec_for_case("book-chapter-3")
    assert s3.columns == 1
    assert s3.name == "book"


def test_extract_quality_and_cost_metrics(tmp_path: Path) -> None:
    """Verify parsing of quality report and usage stats."""
    # When report does not exist
    q_empty = extract_quality_metrics(tmp_path / "nonexistent_quality_report.json")
    assert q_empty.qe_measured is False
    assert q_empty.avg_qe == 0.0

    # With valid report
    rep_file = tmp_path / "test_quality_report.json"
    payload = {
        "summary": {
            "pass_rate": 0.98,
            "estimated_cost_usd": 0.015,
            "cache_hit_rate": 0.45,
        },
        "scores": {
            "avg_qe": 0.94,
            "min_qe": 0.88,
            "p50_qe": 0.95,
            "p90_qe": 0.97,
            "scored_blocks": 25,
            "qe_measured": True,
        },
        "retention": {
            "retention_rate": 0.99,
        },
        "terminology": {
            "term_precision": 0.91,
            "term_recall": 0.89,
        },
    }
    rep_file.write_text(json.dumps(payload), encoding="utf-8")

    q_parsed = extract_quality_metrics(rep_file)
    assert q_parsed.qe_measured is True
    assert q_parsed.avg_qe == 0.94
    assert q_parsed.pass_rate == 0.98
    assert q_parsed.term_precision == 0.91

    # Extract cost from usage dict
    usage_dict = {
        "deepseek-chat": {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "cached_tokens": 400,
            "total_tokens": 1500,
        }
    }
    cost_parsed = extract_cost_metrics(report_path=rep_file, usage_dict=usage_dict)
    assert cost_parsed.total_tokens == 1500
    assert cost_parsed.cached_tokens == 400
    assert cost_parsed.cache_hit_rate == 0.4
    assert cost_parsed.estimated_cost_usd == 0.015


def test_drift_detection_regression_and_improvement() -> None:
    """Verify drift detection logic accurately flags regressions and improvements."""
    base_case_1 = CaseBenchmarkResult(
        id="case-1",
        status="pass",
        contract=ContractBenchmarkMetrics(
            delivered_ratio=0.75, accounted_ratio=0.90, errors_count=0
        ),
        layout=LayoutBenchmarkMetrics(spill_rate=0.01, max_overflow_pt=0.0, layout_passed=True),
    )
    base_case_2 = CaseBenchmarkResult(
        id="case-2",
        status="pass",
        contract=ContractBenchmarkMetrics(
            delivered_ratio=0.80, accounted_ratio=0.95, errors_count=0
        ),
        layout=LayoutBenchmarkMetrics(spill_rate=0.0, max_overflow_pt=0.0, layout_passed=True),
    )
    baseline = CorpusBenchmarkReport(
        run_id="base_run",
        timestamp="2026-10-20T00:00:00Z",
        corpus_dir="corpus",
        mode="dry-run",
        total_cases=2,
        passed_cases=2,
        avg_delivered_ratio=0.775,
        avg_accounted_ratio=0.925,
        avg_spill_rate=0.005,
        cases=[base_case_1, base_case_2],
    )

    # Candidate: case-1 regressed (delivered ratio dropped by 5%, errors introduced)
    # Candidate: case-2 improved (delivered ratio up)
    cand_case_1 = CaseBenchmarkResult(
        id="case-1",
        status="fail",
        failure_reason="Contract expectation failure",
        contract=ContractBenchmarkMetrics(
            delivered_ratio=0.69, accounted_ratio=0.88, errors_count=1, contract_passed=False
        ),
        layout=LayoutBenchmarkMetrics(spill_rate=0.05, max_overflow_pt=8.0, layout_passed=False),
    )
    cand_case_2 = CaseBenchmarkResult(
        id="case-2",
        status="pass",
        contract=ContractBenchmarkMetrics(
            delivered_ratio=0.85, accounted_ratio=0.98, errors_count=0
        ),
        layout=LayoutBenchmarkMetrics(spill_rate=0.0, max_overflow_pt=0.0, layout_passed=True),
    )
    candidate = CorpusBenchmarkReport(
        run_id="cand_run",
        timestamp="2026-10-28T00:00:00Z",
        corpus_dir="corpus",
        mode="dry-run",
        total_cases=2,
        passed_cases=1,
        failed_cases=1,
        overall_passed=False,
        avg_delivered_ratio=0.770,
        avg_accounted_ratio=0.930,
        avg_spill_rate=0.025,
        cases=[cand_case_1, cand_case_2],
    )

    drift_report = compute_benchmark_drift(baseline, candidate)
    assert drift_report.drift_passed is False
    assert drift_report.regressed_count == 1
    assert drift_report.improved_count == 1
    assert drift_report.stable_count == 0

    c1_drift = next(d for d in drift_report.case_drifts if d.case_id == "case-1")
    assert c1_drift.verdict == "REGRESSED"
    assert c1_drift.is_regression is True
    assert any("Status regressed" in r for r in c1_drift.reasons)
    assert any("Contract errors increased" in r for r in c1_drift.reasons)

    c2_drift = next(d for d in drift_report.case_drifts if d.case_id == "case-2")
    assert c2_drift.verdict == "IMPROVED"
    assert any("Delivered ratio improved" in r for r in c2_drift.reasons)


def test_benchmark_runner_in_process_execution(tmp_path: Path) -> None:
    """Test BenchmarkRunner end-to-end using an in-process mock runner."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    docs_dir = corpus_dir / "documents"
    docs_dir.mkdir()

    doc_file = docs_dir / "test_doc.pdf"
    _make_dummy_pdf(doc_file)

    manifest_payload = {
        "schema_version": 1,
        "cases": [
            {
                "id": "mock-case-1",
                "description": "Mock Case One",
                "document": "documents/test_doc.pdf",
                "expect": {
                    "max_errors": 0,
                    "min_delivered_ratio": 0.5,
                },
            }
        ],
    }
    (corpus_dir / "cases.json").write_text(json.dumps(manifest_payload), encoding="utf-8")

    def mock_runner(doc: Path, case_id: str, mode: str, out_pdf: Path) -> None:
        _make_dummy_pdf(out_pdf)
        contract = ReconciliationReport(
            doc_id=case_id,
            total_text=10,
            delivered_text=8,
            verbatim_text=2,
            missing_assets=0,
            # ``violations`` is a tuple (frozen model) and there is no
            # ``target_lang``/``source_lang`` field: language is not part of
            # the delivery contract, so passing them was silently dropped.
            violations=(),
        )
        c_path = contract_path_for_artifact(out_pdf)
        c_path.write_text(contract.model_dump_json(indent=2), encoding="utf-8")
        q_path = sidecar_path(out_pdf, "quality_report.json")
        q_path.write_text(
            json.dumps(
                {
                    "summary": {
                        "pass_rate": 1.0,
                        "estimated_cost_usd": 0.001,
                        "cache_hit_rate": 0.2,
                    },
                    "scores": {"avg_qe": 0.95, "qe_measured": True},
                }
            ),
            encoding="utf-8",
        )

    runner = BenchmarkRunner(
        corpus_dir=corpus_dir,
        mode=BenchmarkMode.DRY_RUN,
        tolerance_pt=5.0,
        runner_fn=mock_runner,
    )

    report = runner.run_all()
    assert report.total_cases == 1
    assert report.passed_cases == 1
    assert report.overall_passed is True
    assert report.cases[0].contract.delivered_ratio == 0.8
    assert report.cases[0].contract.accounted_ratio == 1.0
    assert report.cases[0].quality.avg_qe == 0.95
    assert report.cases[0].cost.estimated_cost_usd == 0.001


def test_benchmark_reporters(tmp_path: Path) -> None:
    """Verify all report rendering formats: Rich terminal, Markdown, HTML, and JSON."""
    case_res = CaseBenchmarkResult(
        id="twocol-paper-test",
        description="Two-column test paper",
        status="pass",
        runtime_sec=3.45,
        contract=ContractBenchmarkMetrics(
            total_text=50,
            delivered_text=40,
            verbatim_text=5,
            delivered_ratio=0.8,
            accounted_ratio=0.9,
            contract_passed=True,
        ),
        layout=LayoutBenchmarkMetrics(
            total_pages=4,
            spill_pages_count=0,
            spill_rate=0.0,
            max_overflow_pt=0.0,
            layout_passed=True,
        ),
        quality=QualityBenchmarkMetrics(avg_qe=0.93, qe_measured=True),
        cost=CostBenchmarkMetrics(total_tokens=4200, estimated_cost_usd=0.005),
    )
    report = CorpusBenchmarkReport(
        run_id="bench_rep_test",
        timestamp="2026-10-28T14:00:00Z",
        corpus_dir=str(tmp_path),
        mode="dry-run",
        total_cases=1,
        passed_cases=1,
        failed_cases=0,
        overall_passed=True,
        total_runtime_sec=3.45,
        avg_delivered_ratio=0.8,
        avg_accounted_ratio=0.9,
        total_tokens=4200,
        total_cost_usd=0.005,
        cases=[case_res],
    )

    # 1. Terminal Dashboard
    test_console = Console(record=True, width=120)
    render_terminal_dashboard(report, console=test_console)
    term_text = test_console.export_text()
    assert "UBT Corpus Automated Benchmark Dashboard" in term_text
    assert "twocol-paper-test" in term_text
    assert "80.0%" in term_text

    # 2. Markdown Report
    md_text = render_markdown_report(report)
    assert "# UBT Corpus Benchmark Report" in md_text
    assert "| `twocol-paper-test` |" in md_text
    assert "80.0%" in md_text

    # 3. HTML Dashboard
    html_text = render_html_dashboard(report)
    assert "<!DOCTYPE html>" in html_text
    assert "UBT Corpus Benchmark Dashboard" in html_text
    assert "twocol-paper-test" in html_text
    assert "meter-bar" in html_text

    # 4. JSON Report
    json_text = render_json_report(report)
    data = json.loads(json_text)
    assert "report" in data
    assert data["report"]["run_id"] == "bench_rep_test"
