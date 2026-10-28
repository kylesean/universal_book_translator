"""Unit tests for the 'ubt coref' CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ubt.cli.main import app

pytestmark = pytest.mark.fast

runner = CliRunner()


def test_coref_inspect_command(tmp_path: Path):
    """Verify ubt coref inspect outputs table and clean JSON."""
    doc_path = tmp_path / "chapter.md"
    doc_path.write_text(
        "We introduce the Raft Consensus Protocol.\n\n"
        "It achieves consensus through replicated logs. "
        "This approach maintains safety under server crashes.\n",
        encoding="utf-8",
    )

    # 1. Terminal table output
    res_table = runner.invoke(app, ["coref", "inspect", str(doc_path)])
    assert res_table.exit_code == 0
    assert "Coreference Mention Disambiguation" in res_table.stdout
    assert "It" in res_table.stdout
    assert "Raft" in res_table.stdout
    assert "Protocol" in res_table.stdout

    # 2. JSON output
    res_json = runner.invoke(app, ["coref", "inspect", str(doc_path), "--json"])
    assert res_json.exit_code == 0
    items = json.loads(res_json.stdout)
    assert len(items) >= 2
    spans = [item["span"] for item in items]
    assert "It" in spans


def test_coref_audit_command(tmp_path: Path):
    """Verify ubt coref audit detects excessive '它-ism'."""
    clean_path = tmp_path / "clean_zh.txt"
    clean_path.write_text(
        "深度学习模型具有强大的泛化能力。该方法通过反向传播自动学习特征。",
        encoding="utf-8",
    )

    res_clean = runner.invoke(app, ["coref", "audit", str(clean_path), "--strict"])
    assert res_clean.exit_code == 0
    assert "PASSED (NATURAL FLOW)" in res_clean.stdout

    # Overused '它-ism' file
    overuse_path = tmp_path / "overuse_zh.txt"
    overuse_path.write_text(
        "它是很好的系统。它的结构很复杂，它的性能很高，它的参数很多，它能够处理各种数据。",
        encoding="utf-8",
    )

    res_overuse = runner.invoke(app, ["coref", "audit", str(overuse_path)])
    assert res_overuse.exit_code == 0
    assert "ATTENTION (PRONOUN OVERUSE)" in res_overuse.stdout

    # Strict mode exits 1
    res_strict = runner.invoke(app, ["coref", "audit", str(overuse_path), "--strict"])
    assert res_strict.exit_code == 1

    # JSON output
    res_json = runner.invoke(app, ["coref", "audit", str(clean_path), "--json"])
    assert res_json.exit_code == 0
    data = json.loads(res_json.stdout)
    assert data["is_acceptable"] is True
