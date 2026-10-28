"""Unit tests for the 'ubt benchmark' CLI command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ubt.cli.main import app
from ubt.core.benchmark.models import (
    CaseBenchmarkResult,
    ContractBenchmarkMetrics,
    CorpusBenchmarkReport,
    CostBenchmarkMetrics,
    LayoutBenchmarkMetrics,
    QualityBenchmarkMetrics,
)
from ubt.core.benchmark.runner import BenchmarkRunner

pytestmark = pytest.mark.fast

runner = CliRunner()


def _make_dummy_pdf(path: Path) -> None:
    """Create minimal valid PDF binary."""
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


def _setup_mock_corpus(corpus_dir: Path) -> None:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    docs_dir = corpus_dir / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)
    _make_dummy_pdf(docs_dir / "doc1.pdf")

    manifest = {
        "schema_version": 1,
        "cases": [
            {
                "id": "case-alpha",
                "description": "Case Alpha",
                "document": "documents/doc1.pdf",
                "expect": {"max_errors": 0, "min_delivered_ratio": 0.5},
            }
        ],
    }
    (corpus_dir / "cases.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_benchmark_command_missing_corpus() -> None:
    """Missing corpus directory exits with code 2."""
    result = runner.invoke(app, ["benchmark", "--corpus", "nonexistent_corpus_dir_123"])
    assert result.exit_code == 2
    assert "not found" in result.output.lower() or "error" in result.output.lower()


def test_benchmark_command_invalid_mode() -> None:
    """Invalid execution mode exits with code 2."""
    result = runner.invoke(app, ["benchmark", "--mode", "unsupported-mode"])
    assert result.exit_code == 2
    assert "invalid mode" in result.output.lower() or "error" in result.output.lower()


def test_benchmark_command_json_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Valid run with --json prints valid structured report."""
    corpus_dir = tmp_path / "corpus"
    _setup_mock_corpus(corpus_dir)

    def mock_run_all(
        self: BenchmarkRunner, case_ids: list[str] | None = None
    ) -> CorpusBenchmarkReport:
        case = CaseBenchmarkResult(
            id="case-alpha",
            status="pass",
            runtime_sec=0.5,
            contract=ContractBenchmarkMetrics(delivered_ratio=0.85, accounted_ratio=0.95),
            layout=LayoutBenchmarkMetrics(spill_rate=0.0, max_overflow_pt=0.0),
            quality=QualityBenchmarkMetrics(avg_qe=0.95, qe_measured=True),
            cost=CostBenchmarkMetrics(total_tokens=1200, estimated_cost_usd=0.002),
        )
        return CorpusBenchmarkReport(
            run_id="bench_mock",
            timestamp="2026-10-28T00:00:00Z",
            corpus_dir=str(corpus_dir),
            mode="dry-run",
            total_cases=1,
            passed_cases=1,
            avg_delivered_ratio=0.85,
            avg_accounted_ratio=0.95,
            cases=[case],
        )

    monkeypatch.setattr(BenchmarkRunner, "run_all", mock_run_all)

    result = runner.invoke(app, ["benchmark", "--corpus", str(corpus_dir), "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "report" in data
    assert data["report"]["passed_cases"] == 1
    assert data["report"]["cases"][0]["id"] == "case-alpha"


def test_benchmark_command_exports_and_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify HTML, Markdown, JSON file exports, baseline saving and drift comparison."""
    corpus_dir = tmp_path / "corpus"
    _setup_mock_corpus(corpus_dir)

    # 1. First run: save baseline
    def mock_run_baseline(
        self: BenchmarkRunner, case_ids: list[str] | None = None
    ) -> CorpusBenchmarkReport:
        case = CaseBenchmarkResult(
            id="case-alpha",
            status="pass",
            runtime_sec=1.0,
            contract=ContractBenchmarkMetrics(
                delivered_ratio=0.80, accounted_ratio=0.90, errors_count=0
            ),
            layout=LayoutBenchmarkMetrics(spill_rate=0.0, max_overflow_pt=0.0),
        )
        return CorpusBenchmarkReport(
            run_id="bench_base",
            timestamp="2026-10-20T00:00:00Z",
            corpus_dir=str(corpus_dir),
            mode="dry-run",
            total_cases=1,
            passed_cases=1,
            avg_delivered_ratio=0.80,
            cases=[case],
        )

    monkeypatch.setattr(BenchmarkRunner, "run_all", mock_run_baseline)

    base_json = tmp_path / "baseline.json"
    res1 = runner.invoke(
        app,
        [
            "benchmark",
            "--corpus",
            str(corpus_dir),
            "--save-baseline",
            str(base_json),
        ],
    )
    assert res1.exit_code == 0
    assert base_json.exists()

    # 2. Second run: compare against baseline and write markdown / html / json
    def mock_run_candidate(
        self: BenchmarkRunner, case_ids: list[str] | None = None
    ) -> CorpusBenchmarkReport:
        case = CaseBenchmarkResult(
            id="case-alpha",
            status="pass",
            runtime_sec=0.8,
            contract=ContractBenchmarkMetrics(
                delivered_ratio=0.85, accounted_ratio=0.95, errors_count=0
            ),
            layout=LayoutBenchmarkMetrics(spill_rate=0.0, max_overflow_pt=0.0),
        )
        return CorpusBenchmarkReport(
            run_id="bench_cand",
            timestamp="2026-10-28T00:00:00Z",
            corpus_dir=str(corpus_dir),
            mode="dry-run",
            total_cases=1,
            passed_cases=1,
            avg_delivered_ratio=0.85,
            cases=[case],
        )

    monkeypatch.setattr(BenchmarkRunner, "run_all", mock_run_candidate)

    html_out = tmp_path / "out.html"
    md_out = tmp_path / "out.md"
    json_out = tmp_path / "out.json"

    res2 = runner.invoke(
        app,
        [
            "benchmark",
            "--corpus",
            str(corpus_dir),
            "--baseline",
            str(base_json),
            "--html-out",
            str(html_out),
            "--markdown-out",
            str(md_out),
            "--json-out",
            str(json_out),
        ],
    )
    assert res2.exit_code == 0
    assert html_out.exists()
    assert md_out.exists()
    assert json_out.exists()

    html_content = html_out.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in html_content
    assert "Historical Baseline Drift Comparison" in html_content

    md_content = md_out.read_text(encoding="utf-8")
    assert "# UBT Corpus Benchmark Report" in md_content
    assert "Baseline Drift" in md_content


def test_benchmark_command_regression_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drift regression causes exit code 1."""
    corpus_dir = tmp_path / "corpus"
    _setup_mock_corpus(corpus_dir)

    base_json = tmp_path / "base.json"
    base_report = CorpusBenchmarkReport(
        run_id="base",
        timestamp="2026-10-20T00:00:00Z",
        corpus_dir=str(corpus_dir),
        mode="dry-run",
        total_cases=1,
        passed_cases=1,
        cases=[
            CaseBenchmarkResult(
                id="case-alpha",
                status="pass",
                contract=ContractBenchmarkMetrics(delivered_ratio=0.80, errors_count=0),
                layout=LayoutBenchmarkMetrics(spill_rate=0.0),
            )
        ],
    )
    base_report.save(base_json)

    # Candidate has regression (errors introduced)
    def mock_run_cand(
        self: BenchmarkRunner, case_ids: list[str] | None = None
    ) -> CorpusBenchmarkReport:
        case = CaseBenchmarkResult(
            id="case-alpha",
            status="fail",
            contract=ContractBenchmarkMetrics(
                delivered_ratio=0.60, errors_count=2, contract_passed=False
            ),
            layout=LayoutBenchmarkMetrics(spill_rate=0.10, layout_passed=False),
        )
        return CorpusBenchmarkReport(
            run_id="cand",
            timestamp="2026-10-28T00:00:00Z",
            corpus_dir=str(corpus_dir),
            mode="dry-run",
            total_cases=1,
            passed_cases=0,
            failed_cases=1,
            overall_passed=False,
            cases=[case],
        )

    monkeypatch.setattr(BenchmarkRunner, "run_all", mock_run_cand)

    res = runner.invoke(
        app,
        [
            "benchmark",
            "--corpus",
            str(corpus_dir),
            "--baseline",
            str(base_json),
        ],
    )
    assert res.exit_code == 1
    assert "REGRESSION DETECTED" in res.output or "REGRESSED" in res.output
