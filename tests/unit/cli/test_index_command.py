"""Unit tests for the 'ubt index' CLI command suite."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ubt.cli.main import app

runner = CliRunner()
pytestmark = pytest.mark.fast


@pytest.fixture
def sample_doc_with_terms(tmp_path: Path) -> Path:
    doc = tmp_path / "distributed_systems.md"
    content = (
        "# 分布式系统导论\n\n"
        "在本章中，我们将学习共识算法（Consensus Algorithm）和状态机复制（State Machine Replication）。\n\n"
        "## 容错设计\n\n"
        "拜占庭容错（Byzantine Fault Tolerance）是保证系统在恶意节点下仍能达成共识算法（Consensus Algorithm）的核心协议。\n"
    )
    doc.write_text(content, encoding="utf-8")
    return doc


def test_index_inspect_command(sample_doc_with_terms: Path) -> None:
    # Terminal text output
    result = runner.invoke(app, ["index", "inspect", str(sample_doc_with_terms)])
    assert result.exit_code == 0
    assert "Consensus Algorithm" in result.output
    assert "共识算法" in result.output

    # JSON output
    json_res = runner.invoke(app, ["index", "inspect", str(sample_doc_with_terms), "--json"])
    assert json_res.exit_code == 0
    data = json.loads(json_res.output)
    assert data["total_terms"] >= 2
    assert any(e["source_term"] == "Consensus Algorithm" for e in data["entries"])


def test_index_generate_command(sample_doc_with_terms: Path, tmp_path: Path) -> None:
    out_typst = tmp_path / "index.typ"
    result = runner.invoke(
        app,
        [
            "index",
            "generate",
            str(sample_doc_with_terms),
            "-o",
            str(out_typst),
            "--format",
            "typst",
            "--columns",
            "3",
        ],
    )
    assert result.exit_code == 0
    assert out_typst.exists()
    content = out_typst.read_text(encoding="utf-8")
    assert "#heading(level: 1, numbering: none)[索引 (Index)]" in content
    assert "#columns(3" in content
    assert "Consensus Algorithm" in content

    # HTML format to stdout
    html_res = runner.invoke(
        app,
        ["index", "generate", str(sample_doc_with_terms), "--format", "html"],
    )
    assert html_res.exit_code == 0
    assert '<section class="ubt-concordance-index">' in html_res.output

    # Markdown format to stdout
    md_res = runner.invoke(
        app,
        ["index", "generate", str(sample_doc_with_terms), "--format", "markdown"],
    )
    assert md_res.exit_code == 0
    assert "## 📑 术语索引 (Index)" in md_res.output


def test_index_stats_command(sample_doc_with_terms: Path) -> None:
    # Text output
    res = runner.invoke(app, ["index", "stats", str(sample_doc_with_terms)])
    assert res.exit_code == 0
    assert "Total Indexed Terms:" in res.output

    # JSON output
    json_res = runner.invoke(app, ["index", "stats", str(sample_doc_with_terms), "--json"])
    assert json_res.exit_code == 0
    data = json.loads(json_res.output)
    assert data["total_terms"] >= 2
    assert "english_letter_groups" in data
    assert "chinese_letter_groups" in data
