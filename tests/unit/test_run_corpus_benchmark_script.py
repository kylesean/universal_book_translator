"""Unit tests for the 'scripts/run_corpus_benchmark.py' standalone script."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_corpus_benchmark import main
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


def _setup_mock_corpus(corpus_dir: Path) -> None:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "cases": [
            {
                "id": "case-script-test",
                "description": "Script Test Case",
                "document": "documents/dummy.pdf",
                "expect": {"max_errors": 0},
            }
        ],
    }
    (corpus_dir / "cases.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_script_missing_corpus(capsys: pytest.CaptureFixture[str]) -> None:
    """Missing corpus path exits with code 2."""
    ret = main(["--corpus", "nonexistent_dir_xyz"])
    assert ret == 2
    captured = capsys.readouterr()
    assert "not found" in captured.err.lower() or "not found" in captured.out.lower()


def test_script_json_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--json prints machine-readable structured report."""
    corpus_dir = tmp_path / "corpus"
    _setup_mock_corpus(corpus_dir)

    def mock_run_all(
        self: BenchmarkRunner, case_ids: list[str] | None = None
    ) -> CorpusBenchmarkReport:
        case = CaseBenchmarkResult(
            id="case-script-test",
            status="pass",
            contract=ContractBenchmarkMetrics(delivered_ratio=0.88),
            layout=LayoutBenchmarkMetrics(spill_rate=0.0),
            quality=QualityBenchmarkMetrics(avg_qe=0.96, qe_measured=True),
            cost=CostBenchmarkMetrics(total_tokens=900, estimated_cost_usd=0.0015),
        )
        return CorpusBenchmarkReport(
            run_id="bench_script",
            timestamp="2026-10-28T00:00:00Z",
            corpus_dir=str(corpus_dir),
            mode="dry-run",
            total_cases=1,
            passed_cases=1,
            cases=[case],
        )

    monkeypatch.setattr(BenchmarkRunner, "run_all", mock_run_all)

    ret = main(["--corpus", str(corpus_dir), "--json"])
    assert ret == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert "report" in data
    assert data["report"]["run_id"] == "bench_script"
    assert data["report"]["cases"][0]["id"] == "case-script-test"


def test_script_exports_and_drift_detection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify script file exports (--html-out, --markdown-out, --json-out, --save-baseline, --baseline)."""
    corpus_dir = tmp_path / "corpus"
    _setup_mock_corpus(corpus_dir)

    base_case = CaseBenchmarkResult(
        id="case-script-test",
        status="pass",
        contract=ContractBenchmarkMetrics(delivered_ratio=0.80, errors_count=0),
        layout=LayoutBenchmarkMetrics(spill_rate=0.0),
    )
    base_report = CorpusBenchmarkReport(
        run_id="base",
        timestamp="2026-10-20T00:00:00Z",
        corpus_dir=str(corpus_dir),
        mode="dry-run",
        total_cases=1,
        passed_cases=1,
        cases=[base_case],
    )
    base_file = tmp_path / "baseline.json"
    base_report.save(base_file)

    def mock_run_cand(
        self: BenchmarkRunner, case_ids: list[str] | None = None
    ) -> CorpusBenchmarkReport:
        case = CaseBenchmarkResult(
            id="case-script-test",
            status="pass",
            contract=ContractBenchmarkMetrics(delivered_ratio=0.85, errors_count=0),
            layout=LayoutBenchmarkMetrics(spill_rate=0.0),
        )
        return CorpusBenchmarkReport(
            run_id="cand",
            timestamp="2026-10-28T00:00:00Z",
            corpus_dir=str(corpus_dir),
            mode="dry-run",
            total_cases=1,
            passed_cases=1,
            cases=[case],
        )

    monkeypatch.setattr(BenchmarkRunner, "run_all", mock_run_cand)

    html_file = tmp_path / "dash.html"
    md_file = tmp_path / "dash.md"
    json_file = tmp_path / "dash.json"
    new_base_file = tmp_path / "new_base.json"

    ret = main(
        [
            "--corpus",
            str(corpus_dir),
            "--baseline",
            str(base_file),
            "--save-baseline",
            str(new_base_file),
            "--html-out",
            str(html_file),
            "--markdown-out",
            str(md_file),
            "--json-out",
            str(json_file),
        ]
    )
    assert ret == 0
    assert html_file.exists()
    assert md_file.exists()
    assert json_file.exists()
    assert new_base_file.exists()
