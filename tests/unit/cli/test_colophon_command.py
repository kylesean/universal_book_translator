"""Unit tests for the 'ubt colophon' CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ubt.cli.main import app

pytestmark = pytest.mark.fast

runner = CliRunner()


def test_colophon_inspect_command(tmp_path: Path) -> None:
    doc_path = tmp_path / "sample.md"
    doc_path.write_text(
        "# 分布式系统设计与实践\n\n"
        "作者：Leslie Lamport。\n\n"
        "本书详细阐述了 Paxos 算法与分布式时钟的核心数学原理。\n",
        encoding="utf-8",
    )

    # Terminal output
    result = runner.invoke(
        app, ["colophon", "inspect", str(doc_path), "--isbn", "978-7-121-39655-7"]
    )
    assert result.exit_code == 0
    assert "Publication Colophon & CIP Data Inspector" in result.output
    assert "分布式系统设计与实践" in result.output
    assert "Valid Checksum" in result.output

    # JSON output
    json_res = runner.invoke(
        app, ["colophon", "inspect", str(doc_path), "--isbn", "978-7-121-39655-7", "--json"]
    )
    assert json_res.exit_code == 0
    data = json.loads(json_res.output)
    assert data["title"] == "分布式系统设计与实践"
    assert data["is_valid_isbn"] is True
    assert data["estimated_words"] > 0


def test_colophon_generate_command(tmp_path: Path) -> None:
    doc_path = tmp_path / "sample.md"
    doc_path.write_text("# 编译原理研习\n\n测试正文内容。\n", encoding="utf-8")
    out_typst = tmp_path / "colophon.typ"

    # Generate Typst
    res_typ = runner.invoke(
        app,
        [
            "colophon",
            "generate",
            str(doc_path),
            "-o",
            str(out_typst),
            "--format",
            "typst",
            "--isbn",
            "978-7-121-39655-7",
        ],
    )
    assert res_typ.exit_code == 0
    assert out_typst.exists()
    typ_content = out_typst.read_text(encoding="utf-8")
    assert "图书在版编目（CIP）数据" in typ_content
    assert "编译原理研习" in typ_content

    # Generate HTML stdout
    res_html = runner.invoke(
        app,
        ["colophon", "generate", str(doc_path), "--format", "html"],
    )
    assert res_html.exit_code == 0
    assert '<section class="ubt-colophon"' in res_html.output

    # Generate JSON stdout
    res_json = runner.invoke(
        app,
        ["colophon", "generate", str(doc_path), "--format", "json"],
    )
    assert res_json.exit_code == 0
    data = json.loads(res_json.output)
    assert "title" in data
    assert data["title"] == "编译原理研习"


def test_colophon_validate_command() -> None:
    # Valid ISBN-13
    res_valid = runner.invoke(app, ["colophon", "validate", "978-7-121-39655-7"])
    assert res_valid.exit_code == 0
    assert "Valid ISBN" in res_valid.output

    # Invalid ISBN-13
    res_invalid = runner.invoke(app, ["colophon", "validate", "978-7-121-39655-8"])
    assert res_invalid.exit_code == 1

    # Valid ISBN-10 with JSON output
    res_json = runner.invoke(app, ["colophon", "validate", "0-8044-2957-X", "--json"])
    assert res_json.exit_code == 0
    data = json.loads(res_json.output)
    assert data["is_valid"] is True
    assert data["formatted"] == "0-804-42957-X"
