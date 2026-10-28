"""Unit tests for the `ubt config` CLI command with sub-domain filtering and validation."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from ubt.cli.main import app

pytestmark = pytest.mark.fast

runner = CliRunner()


def _output(result: object) -> str:
    """stdout+stderr: typer.echo(err=True) lands on stderr under click >= 8.2."""
    text = getattr(result, "output", "") or ""
    return text + (getattr(result, "stderr", "") or "")


def test_config_command_default_lists_all_fields() -> None:
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0
    out = _output(result)
    assert "UBTConfig" in out
    assert "draft_model" in out
    assert "dual_mode" in out
    assert "qe_threshold" in out
    assert "db_dir" in out


def test_config_command_subdomain_router() -> None:
    result = runner.invoke(app, ["config", "--domain", "router"])
    assert result.exit_code == 0
    out = _output(result)
    assert "router" in out.lower()
    assert "draft_model" in out
    assert "rate_limit_rpm" in out
    # Typst / engine fields should not be in router sub-domain view
    assert "font_family" not in out
    assert "db_dir" not in out


def test_config_command_subdomain_typst() -> None:
    result = runner.invoke(app, ["config", "-d", "typst"])
    assert result.exit_code == 0
    out = _output(result)
    assert "typst" in out.lower()
    assert "dual_mode" in out
    assert "emit_xliff_companion" in out
    # Router fields should not be in typst sub-domain view
    assert "rate_limit_rpm" not in out


def test_config_command_subdomain_quality() -> None:
    result = runner.invoke(app, ["config", "--domain", "quality"])
    assert result.exit_code == 0
    out = _output(result)
    assert "quality" in out.lower()
    assert "qe_threshold" in out
    assert "max_repair_rounds" in out
    assert "dual_mode" not in out


def test_config_command_subdomain_engine() -> None:
    result = runner.invoke(app, ["config", "-d", "engine"])
    assert result.exit_code == 0
    out = _output(result)
    assert "engine" in out.lower()
    assert "db_dir" in out
    assert "max_concurrency" in out
    assert "draft_model" not in out


def test_config_command_unknown_domain_rejected() -> None:
    result = runner.invoke(app, ["config", "--domain", "invalid_domain"])
    assert result.exit_code == 2
    out = _output(result)
    assert "Unknown domain 'invalid_domain'" in out
    assert "'router'" in out
    assert "'typst'" in out


def test_config_command_json_output() -> None:
    result = runner.invoke(app, ["config", "--domain", "router", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert isinstance(payload, list)
    fields = {item["field"] for item in payload}
    assert "draft_model" in fields
    assert "rate_limit_rpm" in fields
    assert "font_family" not in fields


def test_config_command_set_only_flag() -> None:
    result = runner.invoke(app, ["config", "--set-only"])
    assert result.exit_code == 0
    out = _output(result)
    assert "UBTConfig" in out


def test_config_validate_clean_default_exits_zero() -> None:
    result = runner.invoke(app, ["config", "--validate"])
    assert result.exit_code == 0
    out = _output(result)
    assert "UBT Configuration Validation" in out
    assert "Configuration passed sanity checks" in out


def test_config_validate_json_mode() -> None:
    result = runner.invoke(app, ["config", "-v", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["valid"] is True
    assert isinstance(payload["findings"], list)
    domains = {f["domain"] for f in payload["findings"]}
    assert "router" in domains
    assert "quality" in domains
    assert "typst" in domains
    assert "engine" in domains


def test_config_validate_detects_invalid_values(monkeypatch: pytest.MonkeyPatch) -> None:
    # Set invalid URL scheme (e.g. ftp:// instead of http/https)
    monkeypatch.setenv("UBT_BASE_URL", "ftp://invalid-endpoint/v1")
    result = runner.invoke(app, ["config", "--validate", "--domain", "router"])
    assert result.exit_code == 1
    out = _output(result)
    assert "FAIL" in out
    assert "base_url must be an absolute" in out
    assert "ftp://invalid-endpoint/v1" in out
