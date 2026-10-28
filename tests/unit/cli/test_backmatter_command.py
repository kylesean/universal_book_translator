"""Unit tests for the 'ubt backmatter' CLI command suite."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ubt.cli.main import app

runner = CliRunner()
pytestmark = pytest.mark.fast


@pytest.fixture
def sample_backmatter_doc(tmp_path: Path) -> Path:
    doc = tmp_path / "book_backmatter.md"
    content = (
        "# 第 1 章：系统概述\n\n"
        "系统架构设计原则。\n\n"
        "# 附录 A：形式化证明\n\n"
        "定理 1.1 的完整推导。\n\n"
        "# 附录 B：基准评测原始数据\n\n"
        "测试环境指标与吞吐量表格。\n\n"
        "# 参考文献\n\n"
        "[1] Dean & Ghemawat 2004.\n"
    )
    doc.write_text(content, encoding="utf-8")
    return doc


def test_backmatter_inspect_command(sample_backmatter_doc: Path) -> None:
    # Text mode
    result = runner.invoke(app, ["backmatter", "inspect", str(sample_backmatter_doc)])
    assert result.exit_code == 0
    assert "Total Backmatter Sections: 3" in result.output
    assert "Appendices Found: 2" in result.output
    assert "APPENDIX" in result.output
    assert "BIBLIOGRAPHY" in result.output

    # JSON mode
    json_res = runner.invoke(app, ["backmatter", "inspect", str(sample_backmatter_doc), "--json"])
    assert json_res.exit_code == 0
    data = json.loads(json_res.output)
    assert data["total_sections"] == 3
    assert data["appendix_count"] == 2
    assert data["has_bibliography"] is True
    assert data["sections"][0]["letter_index"] == "A"
    assert data["sections"][1]["letter_index"] == "B"


def test_backmatter_generate_command(sample_backmatter_doc: Path, tmp_path: Path) -> None:
    # Typst format
    res_typst = runner.invoke(
        app,
        ["backmatter", "generate", str(sample_backmatter_doc), "--format", "typst"],
    )
    assert res_typst.exit_code == 0
    assert "#counter(heading).update(0)" in res_typst.output
    assert '#set heading(numbering: "A.1")' in res_typst.output
    assert "附录 A" in res_typst.output
    assert "numbering: none" in res_typst.output

    # HTML format
    res_html = runner.invoke(
        app,
        ["backmatter", "generate", str(sample_backmatter_doc), "--format", "html"],
    )
    assert res_html.exit_code == 0
    assert '<section class="ubt-backmatter ubt-appendix"' in res_html.output
    assert 'role="doc-appendix"' in res_html.output

    # Write to file
    out_file = tmp_path / "backmatter_skel.md"
    res_file = runner.invoke(
        app,
        [
            "backmatter",
            "generate",
            str(sample_backmatter_doc),
            "-o",
            str(out_file),
            "--format",
            "markdown",
        ],
    )
    assert res_file.exit_code == 0
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "# 附录 A" in content
    assert "# 参考文献" in content


def test_backmatter_validate_command(sample_backmatter_doc: Path, tmp_path: Path) -> None:
    # Passing validation
    res_pass = runner.invoke(app, ["backmatter", "validate", str(sample_backmatter_doc)])
    assert res_pass.exit_code == 0
    assert "PASS" in res_pass.output

    # Failing validation (out of sequence letters, e.g. Appendix C directly without A/B)
    bad_doc = tmp_path / "bad_backmatter.md"
    bad_doc.write_text("# 附录 C：跳跃编号的附录\n\n正文。\n", encoding="utf-8")

    res_fail = runner.invoke(app, ["backmatter", "validate", str(bad_doc)])
    assert res_fail.exit_code == 1
    assert "FAIL" in res_fail.output
    assert "expected 'A'" in res_fail.output


def test_backmatter_nonexistent_file() -> None:
    res = runner.invoke(app, ["backmatter", "inspect", "/nonexistent/doc.md"])
    assert res.exit_code == 1
    assert "Error loading blocks" in res.output
