"""Unit tests for the 'ubt styler' CLI commands."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ubt.cli.main import app

pytestmark = pytest.mark.fast

runner = CliRunner()


def test_styler_inspect_command(tmp_path: Path) -> None:
    doc_path = tmp_path / "sample.md"
    doc_path.write_text(
        "# 第一章 架构设计\n\n"
        "本书深入剖析了分布式系统的核心原理与实现细节。\n\n"
        "| 列1 | 列2 |\n| :--- | :--- |\n| 数据A | 数据B |\n\n"
        "```python\ndef test(): pass\n```\n",
        encoding="utf-8",
    )

    # Terminal output
    result = runner.invoke(app, ["styler", "inspect", str(doc_path)])
    assert result.exit_code == 0
    assert "Responsive EPUB / HTML Layout Inspector" in result.output
    assert "table-responsive" in result.output

    # JSON output
    json_res = runner.invoke(app, ["styler", "inspect", str(doc_path), "--json"])
    assert json_res.exit_code == 0
    data = json.loads(json_res.output)
    assert data["total_blocks"] >= 3
    assert data["responsive_features"]["mobile_table_wrapper"] is True
    assert data["responsive_features"]["code_copy_button"] is True


def test_styler_render_html_command(tmp_path: Path) -> None:
    doc_path = tmp_path / "sample.md"
    doc_path.write_text(
        "# 概述\n\n这是测试段落内容。\n",
        encoding="utf-8",
    )
    out_html = tmp_path / "output.html"

    result = runner.invoke(
        app,
        ["styler", "render-html", str(doc_path), "-o", str(out_html), "--bilingual", "--json"],
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert out_html.exists()
    assert data["bilingual"] is True
    assert data["size_bytes"] > 0
    html_content = out_html.read_text(encoding="utf-8")
    assert "ubt-toolbar" in html_content
    assert "mode-target" in html_content


def test_styler_render_epub_command(tmp_path: Path) -> None:
    doc_path = tmp_path / "sample.md"
    doc_path.write_text(
        "# 电子书章节\n\n这是供墨水屏电纸书阅读的高清排版段落。\n",
        encoding="utf-8",
    )
    out_epub = tmp_path / "output.epub"

    result = runner.invoke(
        app,
        ["styler", "render-epub", str(doc_path), "-o", str(out_epub), "--bilingual", "--json"],
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert out_epub.exists()
    assert data["size_bytes"] > 0

    with zipfile.ZipFile(out_epub) as archive:
        names = archive.namelist()
        assert "mimetype" in names
        assert "OEBPS/content.opf" in names
        assert "OEBPS/style.css" in names
        css = archive.read("OEBPS/style.css").decode("utf-8")
        assert "text-justify: inter-ideograph;" in css
        assert ".bilingual-pair" in css
