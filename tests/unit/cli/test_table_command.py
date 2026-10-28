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
