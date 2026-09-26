"""CLI contract tests for `ubt assess` (exit codes, one-object --json, presets)."""

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from ubt.cli.main import app

runner = CliRunner()


@pytest.fixture
def md_book(tmp_path: Path) -> Path:
    f = tmp_path / "cli_quote.md"
    f.write_text("# Chapter 1\n\n" + ("A calm river runs past the mill. " * 80), encoding="utf-8")
    return f


def test_assess_human_mode_renders_quote(md_book: Path) -> None:
    result = runner.invoke(app, ["assess", str(md_book)])
    assert result.exit_code == 0
    assert "译前报价" in result.stdout
    assert "推荐路线" in result.stdout
    assert "ubt translate" in result.stdout  # the copy-paste next step


def test_assess_json_is_one_object(md_book: Path) -> None:
    result = runner.invoke(app, ["assess", str(md_book), "--json"])
    assert result.exit_code == 0
    payload = result.stdout.strip()
    d = json.loads(payload)  # parses as exactly ONE object
    assert d["status"] == "ok"
    assert d["schema_version"] == 1
    assert d["next_step_command"].startswith("ubt translate")
    assert "--job-id" in d["next_step_command"]


def test_assess_missing_file_exit_1(tmp_path: Path) -> None:
    result = runner.invoke(app, ["assess", str(tmp_path / "gone.pdf")])
    assert result.exit_code == 1
    assert "无法评估" in result.stdout


def test_assess_unsupported_json_error_contract(tmp_path: Path) -> None:
    f = tmp_path / "conf.toml"
    f.write_text("[x]\n", encoding="utf-8")
    result = runner.invoke(app, ["assess", str(f), "--json"])
    assert result.exit_code == 1
    d = json.loads(result.stdout.strip())
    assert d["status"] == "error"
    assert d["code"] == "UNSUPPORTED"


def test_assess_invalid_config_exits_2(md_book: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UBT_RATE_LIMIT_RPM", "0")  # violates gt=0
    result = runner.invoke(app, ["assess", str(md_book), "--json"])
    assert result.exit_code == 2
    d = json.loads(result.stdout.strip())
    assert d["code"] == "INVALID_CONFIG"


def test_assess_bad_provider_profile_is_json_not_traceback(
    md_book: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A missing profile raises ProfileNotFoundError (a UBTError), not a
    ValidationError; it must still honour the one-object --json contract."""
    empty_toml = tmp_path / "ubt.toml"
    empty_toml.write_text("", encoding="utf-8")
    monkeypatch.setattr("ubt.core.profiles.DEFAULT_CONFIG_LOCATIONS", (empty_toml,))
    result = runner.invoke(
        app, ["assess", str(md_book), "--provider-profile", "does-not-exist", "--json"]
    )
    assert result.exit_code == 2
    d = json.loads(result.stdout.strip())  # exactly one object, no traceback
    assert d["code"] == "INVALID_CONFIG"


def test_assess_preset_changes_quoted_policy(md_book: Path) -> None:
    base = runner.invoke(app, ["assess", str(md_book), "--json"])
    preview = runner.invoke(app, ["assess", str(md_book), "--preset", "preview", "--json"])
    assert base.exit_code == 0 and preview.exit_code == 0
    d_base = json.loads(base.stdout.strip())
    d_prev = json.loads(preview.stdout.strip())
    assert d_prev["meta"]["formula_enrichment"] == "off"
    assert d_prev["cost"]["billable_blocks"] == d_base["cost"]["billable_blocks"]
    # The next-step command echoes the preset the quote was priced under.
    assert "--preset preview" in d_prev["next_step_command"]


def test_assess_deep_flag_reports_exact_counts(md_book: Path) -> None:
    result = runner.invoke(app, ["assess", str(md_book), "--deep", "--json"])
    assert result.exit_code == 0
    d = json.loads(result.stdout.strip())
    assert d["deep"] is True
    assert d["cost"]["billable_blocks_is_exact"] is True


def test_assess_command_carries_the_recommended_profile(
    md_book: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The quote is taken under the recommended profile, so the command must run
    under it too -- profile decides prompt assembly, seed glossary and whether
    chapter rolling summaries happen at all."""
    import ubt.core.assess as assess

    original = assess._recommend_route

    def _textbook(arch: Any, route: Any, pdf: Any, config: Any) -> assess.RouteRecommendation:
        rec = original(arch, route, pdf, config)
        return dataclasses.replace(rec, recommended_profile="textbook")

    monkeypatch.setattr(assess, "_recommend_route", _textbook)
    result = runner.invoke(app, ["assess", str(md_book), "--json"])
    assert result.exit_code == 0
    command = json.loads(result.stdout.strip())["next_step_command"]
    assert "--profile textbook" in command
