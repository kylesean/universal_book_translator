"""Unit tests for ``ubt pareto`` and ``ubt benchmark --pareto`` CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ubt.cli.main import app

pytestmark = pytest.mark.fast

runner = CliRunner()


def test_pareto_cli_default() -> None:
    result = runner.invoke(app, ["pareto"], env={"COLUMNS": "180"})
    assert result.exit_code == 0
    assert "UBT Multi-Model Pareto Frontier Analysis" in result.stdout
    assert "SWEET SPOT" in result.stdout
    assert "Model Evaluation & Frontier Status Matrix" in result.stdout


def test_pareto_cli_custom_models() -> None:
    result = runner.invoke(
        app,
        ["pareto", "--models", "deepseek-chat,gpt-4o,claude-3-5-sonnet"],
        env={"COLUMNS": "180"},
    )
    assert result.exit_code == 0
    assert "deepseek-chat" in result.stdout
    assert "gpt-4o" in result.stdout
    assert "claude-3-5-sonnet" in result.stdout


def test_pareto_cli_file_exports(tmp_path: Path) -> None:
    md_file = tmp_path / "pareto.md"
    html_file = tmp_path / "pareto.html"
    json_file = tmp_path / "pareto.json"

    result = runner.invoke(
        app,
        [
            "pareto",
            "--quiet",
            "--markdown-out",
            str(md_file),
            "--html-out",
            str(html_file),
            "--json-out",
            str(json_file),
        ],
    )
    assert result.exit_code == 0
    assert md_file.exists()
    assert html_file.exists()
    assert json_file.exists()

    md_text = md_file.read_text(encoding="utf-8")
    assert "# Multi-Model Quality vs. Cost Pareto Frontier Analysis" in md_text

    html_text = html_file.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in html_text
    assert "<svg" in html_text

    data = json.loads(json_file.read_text(encoding="utf-8"))
    assert "frontier" in data


def test_pareto_cli_quiet_mode() -> None:
    result = runner.invoke(app, ["pareto", "--quiet"])
    assert result.exit_code == 0
    assert "Model Evaluation & Frontier Status Matrix" not in result.stdout
