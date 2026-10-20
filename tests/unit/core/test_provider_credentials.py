"""How a provider block's credential resolves — and the three ways it lies.

``read_provider`` used to answer ``bool(raw)`` on the merged block, which counts
the literal text ``"${OPENCODE_API_KEY}"`` as a credential. The console then
reported "credential set" for a block whose loader output was ``""``, and every
call failed at the first request. These tests pin the corrected contract, which
the router depends on: a configured key is not a usable key.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

_BLOCK = """\
[providers.literal]
base_url = "https://api.example.com/v1"
api_key = "sk-literal-value"

[providers.variable]
base_url = "https://api.example.com/v1"
api_key = "${TEST_CRED_VAR}"

[providers.with_default]
base_url = "https://api.example.com/v1"
api_key = "${TEST_CRED_ABSENT:-sk-default-value}"

[providers.keyless]
base_url = "https://api.example.com/v1"
"""


@pytest.fixture()
def config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "ubt.toml").write_text(_BLOCK, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("UBT_LLM_API_KEY", raising=False)
    monkeypatch.delenv("TEST_CRED_VAR", raising=False)
    monkeypatch.delenv("TEST_CRED_ABSENT", raising=False)
    return tmp_path


def test_a_literal_credential_is_usable(config_dir: Path) -> None:
    from ubt.core.providers import credential_resolution

    state = credential_resolution("literal")
    assert state["kind"] == "literal"
    assert state["usable"] is True
    assert state["unresolved_vars"] == []
    assert "sk-literal-value" not in str(state)


def test_an_unset_variable_is_present_but_not_usable(config_dir: Path) -> None:
    """The bug the console shipped: ``${VAR}`` counted as a credential."""
    from ubt.core.providers import credential_resolution

    state = credential_resolution("variable")
    assert state["kind"] == "variable"
    assert state["usable"] is False
    assert state["unresolved_vars"] == ["TEST_CRED_VAR"]
    # The variable's *value* is never in the payload, and neither is the name of
    # a variable that did resolve.
    assert "TEST_CRED_VAR" in state["unresolved_vars"]


def test_a_set_variable_is_usable_and_no_longer_unresolved(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ubt.core.providers import credential_resolution

    monkeypatch.setenv("TEST_CRED_VAR", "sk-from-env")
    state = credential_resolution("variable")
    assert state["usable"] is True
    assert state["unresolved_vars"] == []
    assert "sk-from-env" not in str(state)


def test_a_default_makes_an_absent_variable_resolve(config_dir: Path) -> None:
    """``${VAR:-default}`` resolves without the variable, so it is not an error.

    Reporting it as unresolved would send the operator to export a variable the
    block deliberately does not require.
    """
    from ubt.core.providers import credential_resolution

    state = credential_resolution("with_default")
    assert state["usable"] is True
    assert state["unresolved_vars"] == []
    # The default is a secret and must not be echoed.
    assert "sk-default-value" not in str(state)


def test_a_keyless_block_uses_the_generic_variable(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ubt.core.providers import credential_resolution

    assert credential_resolution("keyless")["authenticates"] is False
    monkeypatch.setenv("UBT_LLM_API_KEY", "sk-generic")
    state = credential_resolution("keyless")
    assert state["authenticates"] is True
    # The block holds nothing to shadow: the variable is supplying, not hiding.
    assert state["shadowed_by_env"] is False


def test_a_block_credential_is_reported_as_shadowed_when_the_env_is_set(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The save that could not take effect.

    ``UBT_LLM_API_KEY`` outranks the block, so a credential written into the file
    is inert while it is exported. The console promised the next job would use it.
    """
    from ubt.core.providers import credential_resolution

    assert credential_resolution("literal")["shadowed_by_env"] is False
    monkeypatch.setenv("UBT_LLM_API_KEY", "sk-generic")
    state = credential_resolution("literal")
    assert state["usable"] is True
    assert state["shadowed_by_env"] is True


def test_the_loader_and_the_report_agree(config_dir: Path) -> None:
    """``usable`` must match what the engine would actually send.

    This is the invariant that makes the report worth showing: if it says usable,
    ``load_provider_block`` must hand the router a non-empty credential.
    """
    from ubt.core.providers import credential_resolution, load_provider_block

    for name in ("literal", "variable", "with_default", "keyless"):
        state = credential_resolution(name)
        resolved = str(load_provider_block(name).get("api_key") or "").strip()
        assert state["usable"] is bool(resolved), (
            f"{name}: report says usable={state['usable']} but the loader "
            f"resolves to {'a value' if resolved else 'an empty string'}"
        )
