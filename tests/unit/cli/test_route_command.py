"""Unit tests for the 'ubt route' CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ubt.cli.main import app

pytestmark = pytest.mark.fast

runner = CliRunner()


def test_route_simulate_command_table_and_json(tmp_path: Path) -> None:
    """Verify ubt route simulate outputs rich table and clean JSON."""
    doc_path = tmp_path / "book.md"
    doc_path.write_text(
        "# Chapter 1: Foundations\n\n"
        "Machine learning models discover statistical associations in data.\n\n"
        "$$\\min_{\\theta} \\frac{1}{N} \\sum_{i=1}^N \\mathcal{L}(f_\\theta(x_i), y_i)$$\n\n"
        "```python\n"
        "def predict(x):\n"
        "    return model(x)\n"
        "```\n\n"
        "Conclusion of the section.\n",
        encoding="utf-8",
    )

    # 1. Terminal table output
    res_table = runner.invoke(app, ["route", "simulate", str(doc_path)])
    assert res_table.exit_code == 0
    assert "Dynamic Pareto Routing Simulation" in res_table.stdout
    assert "Estimated Savings" in res_table.stdout
    assert "deepseek-chat" in res_table.stdout

    # 2. JSON output
    res_json = runner.invoke(app, ["route", "simulate", str(doc_path), "--json"])
    assert res_json.exit_code == 0
    data = json.loads(res_json.stdout)
    assert data["total_blocks"] == 5
    assert data["savings_percentage"] > 0.0

    # 3. Markdown output
    res_md = runner.invoke(app, ["route", "simulate", str(doc_path), "--markdown"])
    assert res_md.exit_code == 0
    assert "# Dynamic Pareto Routing Simulation Report" in res_md.stdout


def test_route_inspect_command(tmp_path: Path) -> None:
    """Verify ubt route inspect shows block-by-block diagnostics."""
    doc_path = tmp_path / "doc.md"
    doc_path.write_text(
        "# Intro\n\nNormal text.\n\n$$E = mc^2$$\n",
        encoding="utf-8",
    )

    # Human table
    res = runner.invoke(app, ["route", "inspect", str(doc_path), "--limit", "10"])
    assert res.exit_code == 0
    assert "Block Complexity Inspection" in res.stdout
    assert "blk-0001" in res.stdout
    assert "TRIVIAL" in res.stdout
    assert "EXTREME" in res.stdout

    # Filter by tier
    res_filtered = runner.invoke(app, ["route", "inspect", str(doc_path), "--tier", "extreme"])
    assert res_filtered.exit_code == 0
    assert "EXTREME" in res_filtered.stdout
    assert "TRIVIAL" not in res_filtered.stdout

    # JSON output
    res_json = runner.invoke(app, ["route", "inspect", str(doc_path), "--json"])
    assert res_json.exit_code == 0
    items = json.loads(res_json.stdout)
    assert len(items) == 3


def test_route_invalid_policy(tmp_path: Path) -> None:
    """Verify ubt route simulate rejects invalid policy name."""
    doc_path = tmp_path / "simple.md"
    doc_path.write_text("Simple text.", encoding="utf-8")

    res = runner.invoke(app, ["route", "simulate", str(doc_path), "--policy", "unknown_policy"])
    assert res.exit_code == 1
    assert "Invalid policy" in res.stdout
