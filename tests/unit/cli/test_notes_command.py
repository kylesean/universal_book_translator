"""Unit tests for the 'ubt notes' CLI command suite."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ubt.cli.main import app

runner = CliRunner()
pytestmark = pytest.mark.fast


@pytest.fixture
def sample_notes_doc(tmp_path: Path) -> Path:
    doc = tmp_path / "book_notes.md"
    content = (
        "# 第 1 章：分布式架构\n\n"
        "拜占庭容错算法[^1]是分布式共识的基础。\n\n"
        "[^1]: 参见 Castro & Liskov 1999 年经典论文。\n\n"
        "# 第 2 章：存储引擎\n\n"
        "LSM 树具有优秀的写入吞吐量[^1]。\n\n"
        "[^1]: O'Neil 等人于 1996 年提出。\n"
    )
    doc.write_text(content, encoding="utf-8")
    return doc


def test_notes_inspect_command(sample_notes_doc: Path) -> None:
    # Terminal text table
    result = runner.invoke(app, ["notes", "inspect", str(sample_notes_doc)])
    assert result.exit_code == 0
    assert "Total Notes Found: 2" in result.output
    assert "100.0%" in result.output
    assert "Castro & Liskov" in result.output

    # JSON output
    json_res = runner.invoke(app, ["notes", "inspect", str(sample_notes_doc), "--json"])
    assert json_res.exit_code == 0
    data = json.loads(json_res.output)
    assert data["total_notes"] == 2
    assert data["resolution_rate"] == 1.0
    assert len(data["notes"]) == 2


def test_notes_generate_command(sample_notes_doc: Path, tmp_path: Path) -> None:
    # Typst format
    res_typst = runner.invoke(
        app,
        ["notes", "generate", str(sample_notes_doc), "--format", "typst"],
    )
    assert res_typst.exit_code == 0
    assert "#heading(level: 2)[注释]" in res_typst.output
    assert "Castro & Liskov" in res_typst.output

    # HTML format
    res_html = runner.invoke(
        app,
        ["notes", "generate", str(sample_notes_doc), "--format", "html"],
    )
    assert res_html.exit_code == 0
    assert '<section class="ubt-endnotes"' in res_html.output
    assert '<ol class="ubt-endnotes-list">' in res_html.output

    # Write to file
    out_file = tmp_path / "notes_out.md"
    res_file = runner.invoke(
        app,
        [
            "notes",
            "generate",
            str(sample_notes_doc),
            "-o",
            str(out_file),
            "--format",
            "markdown",
        ],
    )
    assert res_file.exit_code == 0
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "[^1]:" in content


def test_notes_audit_command(sample_notes_doc: Path, tmp_path: Path) -> None:
    # Audit pass
    res_pass = runner.invoke(
        app, ["notes", "audit", str(sample_notes_doc), "--min-resolution", "0.9"]
    )
    assert res_pass.exit_code == 0
    assert "PASS" in res_pass.output

    # Create document with missing definition
    bad_doc = tmp_path / "orphan.md"
    bad_doc.write_text("正文内容出现孤立注释引用[^missing]。", encoding="utf-8")

    # Audit fail
    res_fail = runner.invoke(app, ["notes", "audit", str(bad_doc), "--min-resolution", "0.8"])
    assert res_fail.exit_code == 1
    assert "FAIL" in res_fail.output


def test_notes_nonexistent_file() -> None:
    res = runner.invoke(app, ["notes", "inspect", "/nonexistent/doc.md"])
    assert res.exit_code == 1
    assert "Error loading blocks" in res.output
