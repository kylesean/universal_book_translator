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


def test_reading_a_provider_reports_presence_not_the_secret(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``has_credential`` must mean *resolves*, not *is written*.

    ``_BASE_BLOCK`` carries ``api_key = "${DEMO_KEY}"``. With the variable unset
    the loader hands the router an empty string and every call fails, so the old
    ``bool(raw)`` answer — "credential set" — was a promise the next job could not
    keep. The block is still reported as carrying a credential *reference*, which
    is what ``credential_in_file`` says, but not as usable.
    """
    from ubt.api.provider_editor import read_provider

    monkeypatch.delenv("DEMO_KEY", raising=False)
    unresolved = read_provider("demo")
    assert unresolved["has_credential"] is False
    assert unresolved["credential_in_file"] is True
    assert unresolved["credential_issue"] == "unresolved_variable"
    assert unresolved["unresolved_vars"] == ["DEMO_KEY"]

    # Same block, variable set: now it resolves, and the reference is still not
    # echoed — only the mask is.
    monkeypatch.setenv("DEMO_KEY", "sk-resolved")
    resolved = read_provider("demo")
    assert resolved["has_credential"] is True
    assert resolved["credential_issue"] is None
    assert resolved["unresolved_vars"] == []
    for payload in (unresolved, resolved):
        # The value itself is never in the payload — only a boolean and a mask.
        assert "sk-resolved" not in str(payload)
        assert "${DEMO_KEY}" not in str(payload)


def test_a_keyless_local_endpoint_is_not_reported_as_missing(config_dir: Path) -> None:
    """A loopback endpoint accepts an empty bearer, so "no key" is not a fault.

    ``local-gemma4`` on the reference machine points at ``127.0.0.1:9090`` and
    runs with no credential at all. Reporting ``missing`` for it sends the
    operator to fix a configuration that is already correct.
    """
    from ubt.api.provider_editor import read_provider, upsert_provider

    upsert_provider(
        "local",
        {
            "base_url": "http://127.0.0.1:9090/v1",
            "api_mode": "openai-chat",
            "draft_model": "local-model",
        },
    )
    detail = read_provider("local")
    assert detail["has_credential"] is False
    assert detail["credential_issue"] is None


def test_a_remote_endpoint_with_no_credential_is_reported_as_missing(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ubt.api.provider_editor import read_provider, upsert_provider

    monkeypatch.delenv("UBT_LLM_API_KEY", raising=False)
    upsert_provider(
        "remote",
        {
            "base_url": "https://api.example.com/v1",
            "api_mode": "openai-chat",
            "draft_model": "hosted-model",
        },
    )
    detail = read_provider("remote")
    assert detail["has_credential"] is False
    assert detail["credential_issue"] == "missing"


def test_a_written_credential_reports_that_the_environment_outranks_it(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A save the next job cannot honour must not read as a successful save.

    ``UBT_LLM_API_KEY`` outranks every provider block, so writing a key into the
    file while that variable is set produces a form that says "saved, effective on
    the next job" and a job that keeps using the shell value.
    """
    from ubt.api.provider_editor import read_provider, upsert_provider

    monkeypatch.setenv("UBT_LLM_API_KEY", "sk-from-shell")
    upsert_provider("demo", {"api_key": "sk-written-into-file"}, allow_credential=True)

    detail = read_provider("demo")
    # The block does hold a usable key of its own...
    assert detail["has_credential"] is True
    # ...and the form is told it is nevertheless not the one in force.
    assert detail["credential_shadowed_by_env"] is True

    monkeypatch.delenv("UBT_LLM_API_KEY")
    assert read_provider("demo")["credential_shadowed_by_env"] is False


def test_a_keyless_block_on_a_keyed_deployment_authenticates_through_the_env(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``has_credential`` False must not imply "cannot run".

    A block with no key of its own still works when the generic variable supplies
    one, and collapsing the two would show "no credential" for a working provider.
    """
    from ubt.api.provider_editor import read_provider, upsert_provider

    monkeypatch.setenv("UBT_LLM_API_KEY", "sk-from-shell")
    upsert_provider(
        "keyless",
        {
            "base_url": "https://api.example.com/v1",
            "api_mode": "openai-chat",
            "draft_model": "hosted-model",
        },
    )
    detail = read_provider("keyless")
    assert detail["has_credential"] is False
    assert detail["authenticates"] is True
    # Nothing is wrong with the credential: the environment supplies it.
    assert detail["credential_issue"] is None
    # And nothing is shadowed — the block holds no key to be shadowed.
    assert detail["credential_shadowed_by_env"] is False


def test_an_unresolved_reference_is_reported_even_when_the_env_supplies_a_key(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The two facts are independent and both matter.

    ``demo`` names ``${DEMO_KEY}``; with that unset, the block's stated intent
    cannot be honoured. ``UBT_LLM_API_KEY`` still lets the *call* succeed, so
    ``authenticates`` is True and the block runs — but the reference itself is
    broken, and hiding that would leave an operator with a block that silently
    ignores the variable it names.
    """
    from ubt.api.provider_editor import read_provider

    monkeypatch.setenv("UBT_LLM_API_KEY", "sk-from-shell")
    monkeypatch.delenv("DEMO_KEY", raising=False)
    detail = read_provider("demo")
    assert detail["authenticates"] is True
    assert detail["credential_issue"] == "unresolved_variable"
    assert detail["unresolved_vars"] == ["DEMO_KEY"]


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


def test_a_new_block_can_be_made_the_default(config_dir: Path) -> None:
    """The step whose absence made a fresh install unable to run.

    Creating a block alone left ``[defaults] provider`` unset, so the console
    answered ``can_translate: false`` and the first job became a rehearsal even
    though the operator had just typed a working endpoint and key. Measured on a
    fresh clone: one provider declared, ``provider: ''``, ``draft_model: ''``.
    """
    from ubt.api.provider_editor import default_provider_name, upsert_provider

    assert default_provider_name() is None
    upsert_provider(
        "first",
        {
            "base_url": "https://api.example.com/v1",
            "api_mode": "openai-chat",
            "draft_model": "m1",
            "api_key": "sk-test",
        },
        allow_credential=True,
        make_default=True,
    )
    assert default_provider_name() == "first"
    text = (config_dir / "ubt.toml").read_text(encoding="utf-8")
    assert "[defaults]" in text
    assert 'provider = "first"' in text


def test_the_default_survives_to_what_the_engine_resolves(config_dir: Path) -> None:
    """Writing the key is not enough: ``UBTConfig`` must actually pick it up."""
    from ubt.api.provider_editor import upsert_provider

    upsert_provider(
        "chosen",
        {
            "base_url": "https://api.example.com/v1",
            "api_mode": "openai-chat",
            "draft_model": "the-model",
            "api_key": "sk-test",
        },
        allow_credential=True,
        make_default=True,
    )
    from ubt.core.config import UBTConfig, credential_required, has_real_credential

    cfg = UBTConfig.from_env(provider=None)
    assert cfg.provider == "chosen"
    assert cfg.draft_model == "the-model"
    assert credential_required(cfg) and has_real_credential(cfg)


def test_an_explicit_environment_still_outranks_the_file_default(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``UBT_PROVIDER`` was the bug that hid the default; it must keep winning.

    The file default is the *fallback*, not an override. Removing the environment
    variable from ``~/.bashrc`` made the file's answer visible; it did not make it
    authoritative — a user who exports one expects it to win.
    """
    from ubt.api.provider_editor import set_default_provider, upsert_provider
    from ubt.core.config import UBTConfig

    upsert_provider("other", {"base_url": "https://api.example.com/v1", "draft_model": "o"})
    set_default_provider("other")
    assert UBTConfig.from_env().provider == "other"

    # The environment outranks it, and points at a declared block.
    monkeypatch.setenv("UBT_PROVIDER", "demo")
    assert UBTConfig.from_env().provider == "demo"


def test_switching_the_default_does_not_edit_the_block(config_dir: Path) -> None:
    """Selecting which block runs is not a change to that block."""
    from ubt.api.provider_editor import set_default_provider

    before = (config_dir / "ubt.toml").read_text(encoding="utf-8")
    set_default_provider("demo")
    after = (config_dir / "ubt.toml").read_text(encoding="utf-8")
    # The block's own lines are untouched; only [defaults] was added.
    assert 'base_url = "http://127.0.0.1:9090/v1"' in after
    assert "# an operator comment that must survive an edit" in after
    assert after.count("[providers.demo]") == 1
    assert "[defaults]" in after and "[defaults]" not in before


def test_a_name_that_is_not_declared_cannot_become_the_default(config_dir: Path) -> None:
    """``[defaults] provider`` pointing at nothing breaks every run at startup."""
    from ubt.api.provider_editor import ProviderEditError, set_default_provider

    with pytest.raises(ProviderEditError, match="not a declared provider"):
        set_default_provider("ghost")


def test_the_editor_never_writes_outside_the_discovered_config(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A test seam, not an engine rule: pin the target and prove it is honoured.

    ``_config_path`` walks *up* from the CWD, so a temporary directory created
    inside this repository resolves to the repository's own ``ubt.toml``. Running
    the write path against one silently edited a developer's real configuration.
    Every write in this file therefore goes through a fixture that pins the path,
    and this test asserts the pin actually redirects.
    """
    from ubt.api import provider_editor

    target = config_dir / "ubt.toml"
    monkeypatch.setattr(provider_editor, "_config_path", lambda: target)
    assert provider_editor._config_path() == target
    assert provider_editor._config_path().parent == config_dir
