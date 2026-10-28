"""Unit tests for 'ubt ref' CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ubt.cli.main import app

pytestmark = pytest.mark.fast

runner = CliRunner()


def test_ref_inspect_command(tmp_path: Path) -> None:
    """Verify ubt ref inspect analyzes internal anchors and outputs terminal/JSON reports."""
    doc_file = tmp_path / "paper.md"
    doc_file.write_text(
        "# 1. Introduction\n\n"
        "We build upon the architecture proposed in [1]. See Section 1 for details.\n\n"
        "## 2. Methodology\n\n"
        "As illustrated in Figure 1, the pipeline executes sequentially.\n\n"
        "Figure 1: Pipeline Overview\n\n"
        "# References\n\n"
        "[1] Vaswani et al. Attention Is All You Need. 2017.\n",
        encoding="utf-8",
    )

    # 1. Terminal inspect output
    res_term = runner.invoke(app, ["ref", "inspect", str(doc_file)])
    assert res_term.exit_code == 0
    assert "Document Citation & Internal Link Topology" in res_term.stdout
    assert "Total Anchors Discovered:" in res_term.stdout
    assert "Resolved Hyperlinks:" in res_term.stdout

    # 2. JSON inspect output
    res_json = runner.invoke(app, ["ref", "inspect", str(doc_file), "--json"])
    assert res_json.exit_code == 0
    data = json.loads(res_json.stdout)
    assert data["total_anchors"] >= 2
    assert data["resolved_citations_count"] >= 1
    assert "anchors" in data
    assert "resolved_links" in data


def test_ref_check_command(tmp_path: Path) -> None:
    """Verify ubt ref check gates citation resolution rate."""
    doc_file = tmp_path / "test_doc.md"
    doc_file.write_text(
        "# 1. Introduction\n\nCiting [1] and [2].\n\n# References\n\n[1] Reference One.\n",
        encoding="utf-8",
    )

    # Pass case with lower threshold (50%)
    res_pass = runner.invoke(app, ["ref", "check", str(doc_file), "--min-rate", "40.0"])
    assert res_pass.exit_code == 0
    assert "PASS" in res_pass.stdout

    # Fail case with strict threshold (90%)
    res_fail = runner.invoke(app, ["ref", "check", str(doc_file), "--min-rate", "90.0"])
    assert res_fail.exit_code != 0
    assert "FAIL" in res_fail.stdout

    # JSON mode
    res_json = runner.invoke(app, ["ref", "check", str(doc_file), "--min-rate", "40.0", "--json"])
    assert res_json.exit_code == 0
    data = json.loads(res_json.stdout)
    assert data["passed"] is True
    assert "resolution_rate_percent" in data


def test_ref_command_file_not_found() -> None:
    """Verify graceful error reporting when file does not exist."""
    res = runner.invoke(app, ["ref", "inspect", "nonexistent_ref_123.md"])
    assert res.exit_code != 0
    assert "Error loading blocks" in res.stdout
