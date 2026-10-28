"""Unit tests for the 'ubt kinsoku' CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ubt.cli.main import app

pytestmark = pytest.mark.fast

runner = CliRunner()


def test_kinsoku_inspect_command_terminal_and_json(tmp_path: Path) -> None:
    doc_path = tmp_path / "sample.md"
    doc_path.write_text(
        "# 第一章 架构设计\n\n，行首错误标点。\n\n行尾错误括号（\n\n正常段落说明内容。\n",
        encoding="utf-8",
    )

    # Terminal output
    result = runner.invoke(app, ["kinsoku", "inspect", str(doc_path)])
    assert result.exit_code == 0
    assert "CJK Typography & Kinsoku Audit" in result.output
    assert "line_start_prohibited" in result.output

    # JSON output
    json_res = runner.invoke(app, ["kinsoku", "inspect", str(doc_path), "--json"])
    assert json_res.exit_code == 0
    data = json.loads(json_res.output)
    assert data["total_lines"] >= 4
    assert len(data["violations"]) >= 2
    assert data["is_clean"] is False


def test_kinsoku_audit_command(tmp_path: Path) -> None:
    good_doc = tmp_path / "good.md"
    good_doc.write_text(
        "# 第一章 架构设计\n\n"
        "本书深入剖析了分布式系统的核心原理与实现细节。\n"
        "涵盖了高并发处理、故障自愈与共识算法等关键领域。\n",
        encoding="utf-8",
    )

    # Clean doc should pass
    res_pass = runner.invoke(app, ["kinsoku", "audit", str(good_doc), "--strict"])
    assert res_pass.exit_code == 0
    assert "Typography Gate Passed" in res_pass.output

    bad_doc = tmp_path / "bad.md"
    bad_doc.write_text(
        "，行首错误\n（行尾错误\n孤\n",
        encoding="utf-8",
    )
    # Bad doc should fail under strict mode
    res_fail = runner.invoke(app, ["kinsoku", "audit", str(bad_doc), "--strict"])
    assert res_fail.exit_code == 1


def test_kinsoku_fix_command(tmp_path: Path) -> None:
    doc_path = tmp_path / "messy.md"
    doc_path.write_text(
        "这是重复标点，，需要处理。。同时包含《深度学习》和特性——高性能。\n完成。\n",
        encoding="utf-8",
    )
    out_path = tmp_path / "repaired.md"

    result = runner.invoke(app, ["kinsoku", "fix", str(doc_path), "-o", str(out_path), "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["joiners_injected"] > 0
    assert out_path.exists()
    content = out_path.read_text(encoding="utf-8")
    assert "，，" not in content
    assert "。。" not in content
    assert "\u2060" in content
