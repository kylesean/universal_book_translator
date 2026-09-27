"""The config surface must be discoverable and documented, not just greppable.

`ubt config` derives its list from the model schema, and the docs-completeness
test fails the moment a field's env var is missing from the user guide — so the
env surface cannot silently drift away from its documentation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from ubt.cli.commands.config_cmd import config_command
from ubt.core.config import UBTConfig, UBTEnvSettingsSource, env_var_names

pytestmark = pytest.mark.fast

_REPO_ROOT = Path(__file__).resolve().parents[2]
_USER_GUIDE = _REPO_ROOT / "docs" / "guides" / "USER_GUIDE.md"


def test_env_var_names_match_pydantic_settings() -> None:
    """The documented names must be the ones the settings layer actually reads."""
    source = UBTEnvSettingsSource(UBTConfig)
    for name in UBTConfig.model_fields:
        field = UBTConfig.model_fields[name]
        authoritative = {
            env.upper() for (_key, env, _is_complex) in source._extract_field_info(field, name)
        }
        assert set(env_var_names(name)) == authoritative, name


def test_every_field_env_var_is_documented() -> None:
    """A field whose env var is absent from the guide is undiscoverable.

    Whole-token match, not substring: ``UBT_OCR_MODE`` is a prefix of
    ``UBT_OCR_MODEL``, so a substring check passed while the shorter name was
    in fact undocumented.
    """
    if not _USER_GUIDE.exists():
        pytest.skip("docs/ is unversioned; the guide lives only in the working tree")
    guide = _USER_GUIDE.read_text(encoding="utf-8")
    missing = [
        (name, env_var_names(name))
        for name in UBTConfig.model_fields
        if not any(re.search(rf"\b{re.escape(env)}\b", guide) for env in env_var_names(name))
    ]
    assert not missing, f"undocumented env vars (add them to docs/guides/USER_GUIDE.md): {missing}"


def test_config_command_lists_every_field(capsys: pytest.CaptureFixture[str]) -> None:
    config_command(json_output=True, set_only=False)
    rows = json.loads(capsys.readouterr().out)
    by_field = {row["field"]: row for row in rows}
    assert set(by_field) == set(UBTConfig.model_fields)
    assert by_field["draft_model"]["env"] == ["UBT_DRAFT_MODEL"]
    assert by_field["api_key"]["env"][0] == "UBT_LLM_API_KEY"


def test_config_command_defaults_are_never_pydantic_undefined(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A ``default_factory`` field used to print ``PydanticUndefined`` — and a
    secret factory printed a fake ``<set: 17 chars>`` (the sentinel's length)."""
    config_command(json_output=True, set_only=False)
    rows = json.loads(capsys.readouterr().out)
    by_field = {row["field"]: row for row in rows}
    assert all("PydanticUndefined" not in row["default"] for row in rows)
    # A static default renders its value; only a factory field shows "<dynamic>".
    assert by_field["base_url"]["default"] == "https://api.openai.com/v1"
    assert by_field["api_key"]["default"] == "<dynamic>"
    # Core carries no vendor model default: the model comes from a provider
    # preset, the environment, or an explicit argument.
    assert by_field["draft_model"]["default"] == ""


def test_config_command_redacts_secrets(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    secret = "sk-super-secret-value-1234567890"
    monkeypatch.setenv("UBT_LLM_API_KEY", secret)
    config_command(json_output=True, set_only=True)
    out = capsys.readouterr().out
    assert secret not in out
    rows: list[dict[str, Any]] = json.loads(out)
    api_key = next(row for row in rows if row["field"] == "api_key")
    assert api_key["value"].startswith("<set:")


_r0918_ubt_root = Path(__file__).resolve().parents[2]


def test_render_engine_is_canonically_rigid_and_old_spellings_are_gone() -> None:
    from ubt.core.config import canonical_render_engine

    # `rigid` is the single canonical name for the source-geometry route (the
    # old anchored/overlay duality was collapsed with zero backward compat).
    assert canonical_render_engine("rigid") == "rigid"
    assert canonical_render_engine("reflow") == "publication"
    owner = "ubt/core/config.py"
    offenders = [
        path.relative_to(_r0918_ubt_root).as_posix()
        for path in sorted((_r0918_ubt_root / "ubt").rglob("*.py"))
        if path.relative_to(_r0918_ubt_root).as_posix() != owner
        and (
            '"anchored"' in path.read_text(encoding="utf-8")
            or '"overlay"' in path.read_text(encoding="utf-8")
        )
    ]
    assert offenders == [], f"retired render-engine spellings re-spelled in {offenders}"


def test_job_id_pattern_and_default_output_path_have_one_owner() -> None:
    import importlib

    api_app = importlib.import_module("ubt.api.app")
    mcp_server = importlib.import_module("ubt.mcp.server")
    from ubt.core.job_options import JOB_ID_RE, default_output_dir, default_output_path

    assert api_app.JOB_ID_RE is JOB_ID_RE
    assert mcp_server.JOB_ID_RE is JOB_ID_RE
    assert default_output_path("tests/fixtures/synthetic-duo.pdf") == (
        default_output_dir() / "synthetic-duo_bilingual.pdf"
    )
    assert default_output_path("/x/book.epub") == (default_output_dir() / "book_bilingual.epub")
    # Every surface derives the same name; nothing may hardcode the directory.
    owner = "ubt/core/job_options.py"
    spellers = [
        path.relative_to(_r0918_ubt_root).as_posix()
        for path in sorted((_r0918_ubt_root / "ubt").rglob("*.py"))
        if path.relative_to(_r0918_ubt_root).as_posix() != owner
        and 'Path("tmp/output")' in path.read_text(encoding="utf-8")
    ]
    assert spellers == [], f"tmp/output re-derived in {spellers}"


def test_extra_headers_values_are_redacted() -> None:
    """``extra_headers`` carries bearer tokens; only names may be printed."""
    from ubt.cli.commands.config_cmd import _render_value

    rendered = _render_value("extra_headers", {"Authorization": "Bearer supersecret"})
    assert "supersecret" not in rendered
    assert "Authorization" in rendered
