"""Unit tests for the 'ubt table' CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ubt.cli.main import app

pytestmark = pytest.mark.fast

runner = CliRunner()


def test_table_inspect_command(tmp_path: Path) -> None:
    """Verify ubt table inspect detects tables and outputs terminal panel and JSON."""
    doc_file = tmp_path / "tables.md"
    doc_file.write_text(
        "# Benchmark Results\n\n"
        "| Architecture | Top-1 | Latency |\n"
        "| :--- | :---: | ---: |\n"
        "| ResNet-18 | 69.8% | 8ms |\n"
        "| ResNet-50 | 76.1% | 15ms |\n\n"
        "| Table 1 (continued) | | |\n"
        "| :--- | :---: | ---: |\n"
        "| ResNet-101 | 77.4% | 24ms |\n\n"
        "## Summary Notes\n\n"
        "Testing complete.\n",
        encoding="utf-8",
    )

    # 1. Terminal inspect output
    res_term = runner.invoke(app, ["table", "inspect", str(doc_file)])
    assert res_term.exit_code == 0
    assert "Multi-Page Table Continuity Inspection" in res_term.stdout
    assert "Total Tables Found:" in res_term.stdout

    # 2. JSON inspect output
    res_json = runner.invoke(app, ["table", "inspect", str(doc_file), "--json"])
    assert res_json.exit_code == 0
    data = json.loads(res_json.stdout)
    assert data["total_tables"] >= 1
    assert "tables" in data


def test_table_stitch_command(tmp_path: Path) -> None:
    """Verify ubt table stitch fuses split tables and can export output."""
    doc_file = tmp_path / "split_doc.md"
    doc_file.write_text(
        "| Architecture | Top-1 | Latency |\n"
        "| :--- | :---: | ---: |\n"
        "| ResNet-18 | 69.8% | 8ms |\n\n"
        "| Table 1 (continued) | | |\n"
        "| :--- | :---: | ---: |\n"
        "| ResNet-101 | 77.4% | 24ms |\n",
        encoding="utf-8",
    )

    out_file = tmp_path / "fused_out.md"
    res = runner.invoke(app, ["table", "stitch", str(doc_file), "-o", str(out_file)])
    assert res.exit_code == 0
    assert out_file.exists()

    # Verify JSON mode
    res_json = runner.invoke(app, ["table", "stitch", str(doc_file), "--json"])
    assert res_json.exit_code == 0
    data = json.loads(res_json.stdout)
    assert "tables_stitched" in data


def test_table_command_file_not_found() -> None:
    """Verify graceful error reporting when input does not exist."""
    res = runner.invoke(app, ["table", "inspect", "nonexistent_file_123.pdf"])
    assert res.exit_code != 0
    assert "Error loading blocks" in res.stdout


def test_table_style_command(tmp_path: Path) -> None:
    """Verify ubt table style renders repeated headers and continuation notices."""
    doc_file = tmp_path / "long_table.md"
    doc_file.write_text(
        "# Benchmark\n\n"
        "| Model | Score | FPS |\n"
        "| :--- | :---: | ---: |\n"
        "| A | 85.2 | 120 |\n"
        "| B | 88.4 | 110 |\n"
        "| C | 91.0 | 95 |\n"
        "| D | 93.5 | 80 |\n"
        "| E | 95.1 | 65 |\n",
        encoding="utf-8",
    )

    # Typst format
    res_typ = runner.invoke(app, ["table", "style", str(doc_file), "--format", "typst"])
    assert res_typ.exit_code == 0
    assert "table.header(" in res_typ.output
    assert "（续表）" in res_typ.output
    assert "（续下页）" in res_typ.output

    # HTML format to file
    html_out = tmp_path / "styled.html"
    res_html = runner.invoke(
        app,
        ["table", "style", str(doc_file), "--format", "html", "-o", str(html_out)],
    )
    assert res_html.exit_code == 0
    assert html_out.exists()
    content = html_out.read_text(encoding="utf-8")
    assert '<table class="ubt-table">' in content
    assert "ubt-table-continued-notice" in content


def test_table_audit_command(tmp_path: Path) -> None:
    """Verify ubt table audit reports continuation eligibility in terminal and JSON."""
    doc_file = tmp_path / "audit_table.md"
    doc_file.write_text(
        "# Data Summary\n\n"
        "| Col1 | Col2 |\n"
        "| --- | --- |\n"
        "| A | B |\n\n"
        "| Model | Acc | F1 | Latency |\n"
        "| :--- | :---: | :---: | ---: |\n"
        "| M1 | 0.9 | 0.9 | 10ms |\n"
        "| M2 | 0.92 | 0.91 | 12ms |\n"
        "| M3 | 0.95 | 0.94 | 15ms |\n"
        "| M4 | 0.96 | 0.95 | 18ms |\n",
        encoding="utf-8",
    )

    # Terminal output
    res = runner.invoke(app, ["table", "audit", str(doc_file)])
    assert res.exit_code == 0
    assert "Publication Table Continuity Audit" in res.output
    assert "Total Tables Found: 2" in res.output
    assert "Continuation Styling Eligible: 1" in res.output

    # JSON output
    res_json = runner.invoke(app, ["table", "audit", str(doc_file), "--json"])
    assert res_json.exit_code == 0
    data = json.loads(res_json.output)
    assert data["total_tables"] == 2
    assert data["continuation_eligible"] == 1
    assert len(data["tables"]) == 2
    assert data["tables"][0]["continuation_eligible"] is False
    assert data["tables"][1]["continuation_eligible"] is True
