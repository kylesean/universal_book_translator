"""Unit tests for the 'ubt epigraph' CLI command suite."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ubt.cli.main import app

runner = CliRunner()
pytestmark = pytest.mark.fast


@pytest.fixture
def sample_epigraph_doc(tmp_path: Path) -> Path:
    doc = tmp_path / "epigraph_sample.md"
    content = (
        "# 第 1 章：计算之美\n\n"
        '> "Algorithms + Data Structures = Programs"\n'
        "> —— Niklaus Wirth\n\n"
        "**本章导读**：本章探讨图论基础与最短路径搜索算法。\n\n"
        "计算机科学的核心在于寻找最优解。\n"
    )
    doc.write_text(content, encoding="utf-8")
    return doc


def test_epigraph_inspect_command(sample_epigraph_doc: Path) -> None:
    # Text mode
    result = runner.invoke(app, ["epigraph", "inspect", str(sample_epigraph_doc)])
    assert result.exit_code == 0
    assert "Total Epigraphs Found: 1" in result.output
    assert "Total Abstracts Found: 1" in result.output
    assert "Niklaus Wirth" in result.output

    # JSON mode
    json_res = runner.invoke(app, ["epigraph", "inspect", str(sample_epigraph_doc), "--json"])
    assert json_res.exit_code == 0
    data = json.loads(json_res.output)
    assert data["total_epigraphs"] == 1
    assert data["total_abstracts"] == 1
    assert data["epigraphs"][0]["source"] == "Niklaus Wirth"
    assert data["abstracts"][0]["title"] == "本章导读"


def test_epigraph_generate_command_typst() -> None:
    res = runner.invoke(
        app,
        [
            "epigraph",
            "generate",
            "Simplicity is the soul of efficiency.",
            "-s",
            "Austin Freeman",
            "--style",
            "classic_right",
            "--format",
            "typst",
        ],
    )
    assert res.exit_code == 0
    assert "#align(right)[" in res.output
    assert "Simplicity is the soul of efficiency." in res.output
    assert "Austin Freeman" in res.output


def test_epigraph_generate_abstract_html(tmp_path: Path) -> None:
    out_file = tmp_path / "abstract.html"
    res = runner.invoke(
        app,
        [
            "epigraph",
            "generate",
            "This chapter introduces concurrent programming models.",
            "--abstract",
            "--title",
            "Chapter Overview",
            "--format",
            "html",
            "-o",
            str(out_file),
        ],
    )
    assert res.exit_code == 0
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert '<aside class="ubt-chapter-abstract">' in content
    assert "Chapter Overview" in content
    assert "concurrent programming models" in content
