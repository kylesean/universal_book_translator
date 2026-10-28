"""Unit tests for the 'ubt decor' CLI command suite."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ubt.cli.main import app

runner = CliRunner()
pytestmark = pytest.mark.fast


def test_decor_preview_typst() -> None:
    result = runner.invoke(
        app,
        [
            "decor",
            "preview",
            "第 1 章：分布式系统导论",
            "-b",
            "在大规模分布式系统中，高可用性与强一致性始终是系统设计的核心目标。",
            "--style",
            "classic",
            "--format",
            "typst",
        ],
    )
    assert result.exit_code == 0
    assert "第 1 章" in result.output
    assert "分布式系统导论" in result.output
    assert "#align(center)" in result.output
    assert '#ubt-dropcap("在"' in result.output


def test_decor_preview_html() -> None:
    result = runner.invoke(
        app,
        [
            "decor",
            "preview",
            "Chapter 2: Consensus Protocols",
            "-b",
            "Consensus is the core abstraction in distributed computing.",
            "--style",
            "modern",
            "--format",
            "html",
        ],
    )
    assert result.exit_code == 0
    assert '<header class="ubt-chapter-header ubt-header-modern">' in result.output
    assert '<h1 class="ubt-chapter-title">Consensus Protocols</h1>' in result.output
    assert 'class="ubt-dropcap-letter">C</span>' in result.output


def test_decor_preview_json() -> None:
    result = runner.invoke(
        app,
        [
            "decor",
            "preview",
            "第 3 篇：数据存储 -- LSM-Tree",
            "-b",
            "“数据写入性能优先的模型。”",
            "--style",
            "academic",
            "--json",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["parsed"]["label"] == "第 3 篇"
    assert data["parsed"]["title"] == "数据存储"
    assert data["parsed"]["subtitle"] == "LSM-Tree"
    assert data["dropcap"]["prefix"] == "“"
    assert data["dropcap"]["initial_char"] == "数"
    assert data["style"] == "academic"
    assert "markup" in data


def test_decor_preview_invalid_format() -> None:
    result = runner.invoke(
        app,
        [
            "decor",
            "preview",
            "Chapter 1",
            "--format",
            "pdf",
        ],
    )
    assert result.exit_code == 1
    assert "Unsupported format" in result.output


def test_decor_inspect_command(tmp_path: Path) -> None:
    doc = tmp_path / "book.md"
    content = (
        "# 第 1 章：引言\n\n"
        "本书主要介绍现代软件工程理论与方法。\n\n"
        "## 背景与动机\n\n"
        "软件系统的复杂度不断上升。\n\n"
        "# 第 2 章：架构模式\n\n"
        "“微服务架构与单体架构各有取舍。”\n"
    )
    doc.write_text(content, encoding="utf-8")

    # Text mode
    result = runner.invoke(app, ["decor", "inspect", str(doc)])
    assert result.exit_code == 0
    assert "第 1 章" in result.output
    assert "引言" in result.output
    assert "第 2 章" in result.output
    assert "架构模式" in result.output

    # JSON mode
    json_res = runner.invoke(app, ["decor", "inspect", str(doc), "--json"])
    assert json_res.exit_code == 0
    data = json.loads(json_res.output)
    assert data["total_chapters"] == 2
    assert data["chapters"][0]["label"] == "第 1 章"
    assert data["chapters"][0]["title"] == "引言"
    assert data["chapters"][0]["dropcap_letter"] == "本"
    assert data["chapters"][1]["dropcap_letter"] == "“微"


def test_decor_inspect_nonexistent_file() -> None:
    result = runner.invoke(app, ["decor", "inspect", "/nonexistent/path/doc.md"])
    assert result.exit_code == 1
    assert "Error loading blocks" in result.output
