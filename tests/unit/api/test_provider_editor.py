"""The provider write path: what it may change, and what it may never touch.

The console can now edit a ``[providers.<name>]`` block. That is a write to the
file the engine trusts, reached over an HTTP surface that this project routinely
starts on loopback with no API key, so the boundary needs to be tested as a
boundary rather than as a happy path with a few rejections bolted on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.fast

_BASE_BLOCK = """\
[providers.demo]
base_url = "http://127.0.0.1:9090/v1"
api_mode = "openai-chat"
api_key = "${DEMO_KEY}"
draft_model = "demo-model"
# an operator comment that must survive an edit
is_free = true
"""


@pytest.fixture()
def config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway working directory holding one ``ubt.toml``.

    ``ubt.core.providers`` resolves the config file relative to the process CWD,
    so the test chdirs rather than threading a path through the API — the same
    way the real server finds the operator's file.
    """
    (tmp_path / "ubt.toml").write_text(_BASE_BLOCK, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    # The loader caches the *shipped* registry only; the user file is re-read on
    # every call, which is what makes an edit visible to the next job.
    return tmp_path


def test_an_edit_writes_the_block_and_preserves_everything_else(config_dir: Path) -> None:
    from ubt.api.provider_editor import upsert_provider

    result = upsert_provider("demo", {"supports_vision": True, "draft_model": "demo-model-2"})
    assert result["supports_vision"] is True
    assert result["draft_model"] == "demo-model-2"

    text = (config_dir / "ubt.toml").read_text(encoding="utf-8")
    # The credential reference is untouched: this path never reads or writes one.
    assert 'api_key = "${DEMO_KEY}"' in text
    # A comment the operator wrote survives, because tomlkit round-trips.
    assert "# an operator comment that must survive an edit" in text
    # Unrelated keys are not dropped by a partial edit.
    assert "is_free = true" in text
    assert 'base_url = "http://127.0.0.1:9090/v1"' in text


def test_a_credential_is_refused_unless_the_caller_opts_in(config_dir: Path) -> None:
    """The default is still "no", and the gate is a parameter, not a guess.

    ``validate_edit`` defaults to ``allow_credential=False`` so a new caller cannot
    write a secret by forgetting an argument. The route passes the deployment's
    actual answer (loopback + non-strict), which only it knows.
    """
    from ubt.api.provider_editor import ProviderEditError, upsert_provider, validate_edit

    with pytest.raises(ProviderEditError) as excinfo:
        validate_edit("demo", {"api_key": "sk-injected"})
    assert "credential cannot be written" in str(excinfo.value)

    with pytest.raises(ProviderEditError):
        upsert_provider("demo", {"api_key": "sk-injected"})
    assert "sk-injected" not in (config_dir / "ubt.toml").read_text(encoding="utf-8")


def test_a_credential_is_written_when_the_deployment_allows_it(config_dir: Path) -> None:
    """On loopback the write is allowed, and the value never comes back out."""
    from ubt.api.provider_editor import upsert_provider

    result = upsert_provider("demo", {"api_key": "sk-local"}, allow_credential=True)
    assert result["has_credential"] is True
    # Reported as a fixed mask, never the value.
    assert result["credential_hint"] == "********"
    assert "sk-local" not in str(result)
    # And it reached the file, which is tightened because it may now hold one.
    text = (config_dir / "ubt.toml").read_text(encoding="utf-8")
    assert 'api_key = "sk-local"' in text
    assert (config_dir / "ubt.toml").stat().st_mode & 0o777 == 0o600


def test_the_other_credentials_are_still_unwritable(config_dir: Path) -> None:
    """``ocr_api_key`` and the inbound gate secret have no console path at all.

    ``ocr_api_key`` belongs to the OCR driver's channel, not the LLM path, and
    ``service_api_key`` is the inbound gate — writing it over HTTP would let a
    caller holding the current key mint a new gate for everyone.
    """
    from ubt.api.provider_editor import ProviderEditError, validate_edit

    for field in ("ocr_api_key", "service_api_key"):
        with pytest.raises(ProviderEditError) as excinfo:
            validate_edit("demo", {field: "x"}, allow_credential=True)
        assert field in str(excinfo.value)


def test_credential_write_is_gated_on_bind_and_auth_posture() -> None:
    """Both conditions are required, and each alone is enough to refuse."""
    from ubt.api.provider_editor import credential_write_allowed

    assert credential_write_allowed(is_loopback=True, is_strict=False) is True
    # Strict auth promises the server authenticates; a bearer token must not mint
    # another one.
    assert credential_write_allowed(is_loopback=True, is_strict=True) is False
    # A non-loopback bind is a shared service.
    assert credential_write_allowed(is_loopback=False, is_strict=False) is False
    assert credential_write_allowed(is_loopback=False, is_strict=True) is False


def test_the_rest_request_model_still_forbids_unknown_fields() -> None:
    """``extra="forbid"`` is what turns a typo into a named 422.

    ``api_key`` is a declared field now, so this checks the mechanism with a key
    that is genuinely not accepted — a silently ignored field reads to a client as
    "the save succeeded", the worst of both outcomes.
    """
    from pydantic import ValidationError

    from ubt.api.models import ProviderEditRequest

    with pytest.raises(ValidationError) as excinfo:
        ProviderEditRequest(wibble="x")  # type: ignore[call-arg]
    assert "wibble" in str(excinfo.value)


def test_only_whitelisted_fields_are_writable(config_dir: Path) -> None:
    """A field outside the whitelist is rejected, including global run knobs."""
    from ubt.api.provider_editor import WRITABLE_FIELDS, ProviderEditError, validate_edit

    for field in ("db_dir", "prices_file", "service_api_key", "wibble"):
        with pytest.raises(ProviderEditError):
            validate_edit("demo", {field: "x"})

    # The whitelist itself must stay a subset of what the loader will read back,
    # or an edit would produce a config that fails on the next job.
    from ubt.core.providers import PROVIDER_ALLOWED_KEYS

    assert set(WRITABLE_FIELDS) <= PROVIDER_ALLOWED_KEYS
    # A credential is writable only through the separate, deployment-aware gate.
    assert "api_key" in WRITABLE_FIELDS
    assert "ocr_api_key" not in WRITABLE_FIELDS


def test_invalid_values_are_rejected_before_they_reach_the_file(config_dir: Path) -> None:
    from ubt.api.provider_editor import ProviderEditError, validate_edit

    cases: list[dict[str, Any]] = [
        {"api_mode": "nonsense"},
        {"supports_temperature": "yes"},
        {"base_url": "file:///etc/passwd"},
        {"extraction_strategy": "guess"},
        {"reasoning_dialect": "sideways"},
        {"api_timeout": -1},
        {"fallback_models": "not-a-list"},
    ]
    for changes in cases:
        with pytest.raises(ProviderEditError):
            validate_edit("demo", changes)


def test_a_provider_name_cannot_escape_its_table(config_dir: Path) -> None:
    """The name becomes a TOML key path, so it may not carry structure."""
    from ubt.api.provider_editor import ProviderEditError, validate_edit

    for name in ("../evil", "a.b", "x]", "", "name with space"):
        with pytest.raises(ProviderEditError):
            validate_edit(name, {"is_free": True})


def test_the_edit_is_atomic_and_tightens_permissions(config_dir: Path) -> None:
    """A half-written config would fail every subsequent job at startup."""
    from ubt.api.provider_editor import upsert_provider

    (config_dir / "ubt.toml").chmod(0o644)
    upsert_provider("demo", {"is_free": False})

    mode = (config_dir / "ubt.toml").stat().st_mode & 0o777
    assert mode == 0o600, f"config left at {oct(mode)}; it may hold a literal credential"
    # No temp file survives the write.
    assert not [p for p in config_dir.iterdir() if p.name.startswith(".ubt.toml")]


def test_reading_a_provider_reports_presence_not_the_secret(config_dir: Path) -> None:
    from ubt.api.provider_editor import read_provider

    detail = read_provider("demo")
    assert detail["has_credential"] is True
    assert detail["credential_in_file"] is True
    # The value itself is never in the payload — only the boolean.
    assert "${DEMO_KEY}" not in str(detail)
    assert "DEMO_KEY" not in str(detail)


def test_a_new_provider_can_be_created(config_dir: Path) -> None:
    from ubt.api.provider_editor import read_provider, upsert_provider

    result = upsert_provider(
        "fresh",
        {
            "base_url": "https://api.example.com/v1",
            "api_mode": "openai-responses",
            "draft_model": "some-model",
            "supports_reasoning_effort": True,
        },
    )
    assert result["declared"] is True
    assert result["supports_reasoning_effort"] is True
    # A block created without a credential must say so rather than imply one.
    assert read_provider("fresh")["has_credential"] is False


def test_the_saved_block_is_what_the_loader_reads_back(config_dir: Path) -> None:
    """The write path and the read path must agree, or a save is a lie."""
    from ubt.api.provider_editor import upsert_provider
    from ubt.core.providers import load_provider_block

    upsert_provider(
        "demo",
        {
            "supports_temperature": False,
            "supports_vision": True,
            "deployment_backend": "llama-swap",
            "extraction_strategy": "xml_tag",
        },
    )
    block = load_provider_block("demo")
    assert block["supports_temperature"] is False
    assert block["supports_vision"] is True
    assert block["deployment_backend"] == "llama-swap"
    assert block["extraction_strategy"] == "xml_tag"


def test_an_edit_applies_to_the_next_job_without_a_restart(config_dir: Path) -> None:
    """The whole point of writing the file: no server restart to change a model.

    ``UBTConfig.from_env`` reads ``load_layer`` on every call, so a new job sees
    the new block. This is the behaviour the console's save button promises.
    """
    from ubt.api.provider_editor import upsert_provider
    from ubt.core.config import UBTConfig

    assert UBTConfig.from_env(provider="demo").draft_model == "demo-model"
    upsert_provider("demo", {"draft_model": "demo-model-3", "supports_reasoning_effort": True})
    reloaded = UBTConfig.from_env(provider="demo")
    assert reloaded.draft_model == "demo-model-3"
    assert reloaded.supports_reasoning_effort is True


def test_a_declared_flag_can_be_cleared_back_to_undeclared(config_dir: Path) -> None:
    """``false`` and "absent" are different states, and both must be reachable.

    ``supports_temperature = false`` claims the endpoint REJECTS the parameter
    (the gpt-5 / o1 family does); an absent key means "not declared, use the safe
    default". A form that could set but never clear would turn one into the other
    permanently, and a two-state checkbox cannot express either transition.
    """
    from ubt.api.provider_editor import read_provider, upsert_provider

    upsert_provider("demo", {"supports_temperature": False})
    assert read_provider("demo")["supports_temperature"] is False

    # Submitting null removes the key rather than storing a third value.
    upsert_provider("demo", {"supports_temperature": None})
    detail = read_provider("demo")
    assert detail.get("supports_temperature") is None
    text = (config_dir / "ubt.toml").read_text(encoding="utf-8")
    assert "supports_temperature" not in text


def test_a_required_field_cannot_be_cleared(config_dir: Path) -> None:
    """Clearing an endpoint or model id would leave a block that cannot run."""
    from ubt.api.provider_editor import ProviderEditError, upsert_provider

    for field in ("base_url", "draft_model", "api_mode"):
        with pytest.raises(ProviderEditError):
            upsert_provider("demo", {field: None})
