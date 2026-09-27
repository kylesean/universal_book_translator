"""Credential-field isolation guard (security regression).

Historical defect: ``api_key``, ``service_api_key`` and ``ocr_api_key`` shared
env aliases (``UBT_API_KEY`` / ``OPENAI_API_KEY``). Setting the outbound LLM
key silently switched the inbound ``X-API-Key`` gate on *with the provider
secret as the expected value*, so every authenticated API client necessarily
held the LLM credential. These tests pin one env name per credential.

Second historical defect: every ``AliasChoices`` listed the bare, prefix-less
field name first (``api_key`` / ``base_url`` / ``pages`` / ...), so with
pydantic-settings' default ``case_sensitive=False`` an unrelated ``API_KEY``,
``BASE_URL`` or ``PAGES`` variable in the process environment hijacked the
outbound credential, the endpoint, or the page filter. The settings source now
accepts only ``UBT_*`` names (plus the ambient ``OPENCODE_SESSION_ID``).

Third: the third-party vendor names (``OPENAI_API_KEY`` / ``DEEPSEEK_API_KEY`` /
...) used to be ``api_key`` / ``base_url`` aliases, and the endpoint was guessed
from whichever of them happened to be set. They are no longer aliases — a
provider block names the variable holding its credential via ``api_key_env``,
so the key and the endpoint always come from the *same* provider.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr

from ubt.core import config as config_mod
from ubt.core.config import MOCK_API_KEY, UBTConfig

_CRED_ENV = (
    "UBT_LLM_API_KEY",
    "UBT_PROVIDER",
    "OPENAI_API_KEY",
    "OPENCODE_API_KEY",
    "DEEPSEEK_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "UBT_API_KEY",
    "UBT_SERVICE_API_KEY",
    "UBT_OCR_API_KEY",
    "UBT_BASE_URL",
    "OPENAI_BASE_URL",
    "OPENCODE_BASE_URL",
    "ANTHROPIC_BASE_URL",
)

# Every env name a field USED to accept without the UBT_ prefix. None may resolve.
_BARE_ALIAS_ENV = (
    "API_KEY",
    "BASE_URL",
    "PAGES",
    "API_TIMEOUT",
    "SERVICE_API_KEY",
    "OCR_API_KEY",
)
_AMBIENT_COMPAT_ENV = ("OPENCODE_SESSION_ID",)

_STRAY = "stray-injected-value"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in (*_CRED_ENV, *_BARE_ALIAS_ENV, *_AMBIENT_COMPAT_ENV):
        monkeypatch.delenv(name, raising=False)
    for name in (
        "UBT_PAGES",
        "UBT_PAGE_RANGE",
        "UBT_API_TIMEOUT",
        "UBT_TIMEOUT",
        "UBT_OPENCODE_SESSION_ID",
        "UBT_MAX_CONCURRENCY",
    ):
        monkeypatch.delenv(name, raising=False)
    # Hermetic: the repository's own ubt.toml must not leak a provider block
    # into a ``from_env`` assertion here.
    empty = tmp_path / "config.toml"
    empty.write_text("", encoding="utf-8")
    monkeypatch.setattr("ubt.core.providers.DEFAULT_CONFIG_LOCATIONS", (empty,))


def _cfg() -> UBTConfig:
    return UBTConfig()


def test_foreign_credential_stores_are_never_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UBT must not translate a book with another program's account.

    The credential chain used to end by parsing opencode's ``auth.json`` and
    reading ``~/.config/deepseek_key``, and the matching endpoint sniff then sent
    the whole manuscript to that third party -- on a machine with no UBT
    configuration at all, with ``doctor`` reporting a bare "API key OK". All
    three stores are planted here; a single one of them being consulted flips
    the key away from the mock placeholder.
    """
    (tmp_path / ".local/share/opencode").mkdir(parents=True)
    (tmp_path / ".local/share/opencode/auth.json").write_text(
        '{"opencode-go": {"key": "sk-from-other-app"}}', encoding="utf-8"
    )
    (tmp_path / ".config").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".config/deepseek_key").write_text("sk-deepseek-file", encoding="utf-8")
    (tmp_path / ".deepseek_key").write_text("sk-deepseek-home", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    assert UBTConfig().api_key.get_secret_value() == MOCK_API_KEY
    # The stores those helpers used to reach for; pinning their absence keeps a
    # future "convenient" fallback from being re-added under the old names.
    assert not hasattr(config_mod, "_read_opencode_key")
    assert not hasattr(config_mod, "_read_deepseek_file_key")
    assert not hasattr(config_mod, "_OPENCODE_AUTH_PATH")


def test_bare_prefixless_env_names_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """Risk: a stray ``API_KEY`` / ``BASE_URL`` / ``PAGES`` in the environment
    silently became the outbound LLM credential, the API endpoint, and the page
    filter (and with an opencode auth.json present the foreign key was then
    sent to the Zen host the user never chose)."""
    for name in _BARE_ALIAS_ENV:
        monkeypatch.setenv(name, _STRAY)
    monkeypatch.setenv("API_TIMEOUT", "5")
    cfg = _cfg()
    assert cfg.api_key.get_secret_value() == MOCK_API_KEY
    assert _STRAY not in cfg.base_url
    assert cfg.pages is None
    assert cfg.api_timeout == 180.0
    assert cfg.service_api_key.get_secret_value() == ""
    assert cfg.ocr_api_key.get_secret_value() == ""


def test_canonical_ubt_names_still_resolve(monkeypatch: pytest.MonkeyPatch) -> None:
    """Risk: removing the bare aliases must not break the one documented
    canonical ``UBT_<FIELD>`` name per field (operators configure only those)."""
    monkeypatch.setenv("UBT_LLM_API_KEY", "sk-llm")
    monkeypatch.setenv("UBT_BASE_URL", "https://canonical.example/v1")
    monkeypatch.setenv("UBT_PAGES", "2-4")
    monkeypatch.setenv("UBT_API_TIMEOUT", "42")
    monkeypatch.setenv("UBT_OCR_API_KEY", "ocr-k")
    monkeypatch.setenv("UBT_OPENCODE_SESSION_ID", "ubt-session")
    cfg = _cfg()
    assert cfg.api_key.get_secret_value() == "sk-llm"
    assert cfg.base_url == "https://canonical.example/v1"
    assert cfg.pages == "2-4"
    assert cfg.api_timeout == 42.0
    assert cfg.ocr_api_key.get_secret_value() == "ocr-k"
    assert cfg.opencode_session_id == "ubt-session"


def test_secondary_prefixed_aliases_still_resolve(monkeypatch: pytest.MonkeyPatch) -> None:
    """Risk: the prefixed secondary names (``UBT_PAGE_RANGE`` / ``UBT_TIMEOUT``)
    are documented aliases; dropping them with the bare ones would silently
    ignore existing operator configuration."""
    monkeypatch.setenv("UBT_PAGE_RANGE", "7")
    monkeypatch.setenv("UBT_TIMEOUT", "11")
    cfg = _cfg()
    assert cfg.pages == "7"
    assert cfg.api_timeout == 11.0


def test_openai_provider_reads_the_standard_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A selected provider resolves its credential from its ``api_key_env``."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    cfg = UBTConfig.from_env(provider="openai")
    assert cfg.api_key.get_secret_value() == "sk-openai"
    assert cfg.base_url == "https://api.openai.com/v1"
    # ...but the vendor name stays out of the inbound gate and the OCR key.
    assert cfg.service_api_key.get_secret_value() == ""
    assert cfg.ocr_api_key.get_secret_value() == ""


def test_opencode_provider_reads_the_standard_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENCODE_API_KEY", "sk-opencode")
    cfg = UBTConfig.from_env(provider="opencode")
    assert cfg.api_key.get_secret_value() == "sk-opencode"
    assert cfg.base_url == "https://opencode.ai/zen/go/v1"
    assert cfg.api_mode == "responses"
    assert cfg.service_api_key.get_secret_value() == ""
    assert cfg.ocr_api_key.get_secret_value() == ""


def test_credential_and_endpoint_come_from_the_same_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The key and the URL must never come from different providers.

    Both are supplied by the selected provider block, so with DEEPSEEK_API_KEY
    and GEMINI_API_KEY both set, selecting deepseek sends the DeepSeek secret to
    the DeepSeek endpoint — the old implicit sniffing paired the key from one
    vendor with the host of another.
    """
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
    monkeypatch.setenv("GEMINI_API_KEY", "gm-key")
    cfg = UBTConfig.from_env(provider="deepseek")
    assert cfg.api_key.get_secret_value() == "sk-deepseek"
    assert "deepseek" in cfg.base_url


def test_third_party_names_are_inert_without_a_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only ``UBT_LLM_API_KEY`` / ``UBT_BASE_URL`` configure the wire by default.

    A vendor variable is meaningful only once its provider is selected; on its
    own it must not redirect the endpoint or supply a credential.
    """
    for name in ("DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(name, "sk-ignored")
    cfg = _cfg()
    assert cfg.api_key.get_secret_value() == MOCK_API_KEY
    assert cfg.base_url == config_mod._OPENAI_BASE_URL


def test_ambient_opencode_session_id_still_resolves(monkeypatch: pytest.MonkeyPatch) -> None:
    """Risk: ``OPENCODE_SESSION_ID`` is the session variable the opencode CLI
    exports to child processes — it is a deliberate third-party fallback (never
    the bare field name) and must keep working behind ``UBT_OPENCODE_SESSION_ID``."""
    monkeypatch.setenv("OPENCODE_SESSION_ID", "ambient-session")
    assert _cfg().opencode_session_id == "ambient-session"
    monkeypatch.setenv("UBT_OPENCODE_SESSION_ID", "explicit-session")
    assert _cfg().opencode_session_id == "explicit-session"


def test_ubt_llm_key_sets_only_outbound(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UBT_LLM_API_KEY", "sk-llm")
    cfg = _cfg()
    assert cfg.api_key.get_secret_value() == "sk-llm"
    assert cfg.service_api_key.get_secret_value() == ""


def test_inbound_gate_key_is_isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UBT_API_KEY", "gate-secret")
    cfg = _cfg()
    assert cfg.service_api_key.get_secret_value() == "gate-secret"
    # The inbound gate must never become the outbound credential.
    assert cfg.api_key.get_secret_value() == MOCK_API_KEY


def test_ocr_key_is_isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UBT_OCR_API_KEY", "ocr-secret")
    cfg = _cfg()
    assert cfg.ocr_api_key.get_secret_value() == "ocr-secret"
    assert cfg.api_key.get_secret_value() == MOCK_API_KEY
    assert cfg.service_api_key.get_secret_value() == ""


def test_legacy_service_alias_is_gone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UBT_SERVICE_API_KEY", "legacy")
    cfg = _cfg()
    assert cfg.service_api_key.get_secret_value() == ""
    assert cfg.api_key.get_secret_value() == MOCK_API_KEY


def test_reusing_one_secret_for_both_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UBT_LLM_API_KEY", "same-secret")
    monkeypatch.setenv("UBT_API_KEY", "same-secret")
    with pytest.raises(ValueError, match="must differ"):
        _cfg()


def test_programmatic_overrides_go_through_from_env_and_are_validated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Overrides must be applied AND validated, not silently dropped.

    ``from_env(**overrides)`` is the supported programmatic path for the
    scripts. Like the CLI/API paths, every override re-runs validation, so a bad
    value fails fast
    instead of silently misconfiguring a billed run.
    """
    monkeypatch.delenv("UBT_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    cfg = UBTConfig.from_env(
        api_key=SecretStr("override-key"),
        base_url="https://override.example/v1",
        pages="7-9",
        max_concurrency=3,
    )
    assert cfg.api_key.get_secret_value() == "override-key"
    assert cfg.base_url == "https://override.example/v1"
    assert cfg.pages == "7-9"
    assert cfg.max_concurrency == 3

    # None means "leave whatever the environment provided" (CLI flags are optional).
    monkeypatch.setenv("UBT_PAGES", "1-2")
    assert UBTConfig.from_env(pages=None).pages == "1-2"

    # Validation still runs on the override (the --concurrency 0 hang class).
    with pytest.raises(ValueError, match="greater than 0"):
        UBTConfig.from_env(max_concurrency=0)


def test_constructor_kwargs_for_aliased_fields_are_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Constructor kwargs like UBTConfig(api_key=...) and from_env are both supported."""
    monkeypatch.delenv("UBT_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert UBTConfig(api_key=SecretStr("custom")).api_key.get_secret_value() == "custom"
    assert UBTConfig.from_env(api_key=SecretStr("custom")).api_key.get_secret_value() == "custom"


def test_repair_model_follows_draft_only_when_never_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invariant owner: when draft_model is configured but repair_model
    never is, repair follows draft (single source: UBTConfig)."""
    monkeypatch.setenv("UBT_LLM_API_KEY", "sk-test")
    monkeypatch.delenv("UBT_REPAIR_MODEL", raising=False)
    cfg = UBTConfig.from_env(draft_model="solo-draft")
    assert cfg.repair_model == "solo-draft"


def test_explicit_repair_model_not_overridden_by_draft(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A request-level draft_model override (what `--draft-model` becomes)
    must NOT silently override an explicitly-configured UBT_REPAIR_MODEL. The
    CLI used to pre-mirror draft→repair, making CLI disagree with API/MCP on the
    same request; the mirror is gone and config owns the sync."""
    from ubt.core.job_options import apply_config_overrides

    monkeypatch.setenv("UBT_LLM_API_KEY", "sk-test")
    monkeypatch.setenv("UBT_DB_DIR", str(tmp_path))
    monkeypatch.setenv("UBT_DRAFT_MODEL", "env-draft")
    monkeypatch.setenv("UBT_REPAIR_MODEL", "env-repair")
    base = UBTConfig.from_env()
    assert (base.draft_model, base.repair_model) == ("env-draft", "env-repair")
    merged = apply_config_overrides(base, {"draft_model": "req-draft"})
    assert merged.draft_model == "req-draft"
    assert merged.repair_model == "env-repair"  # not dragged along by draft


def test_apply_config_overrides_syncs_repair_model_when_unset() -> None:
    """When repair_model was not distinct from draft_model, draft_model overrides sync repair_model."""
    from ubt.core.job_options import apply_config_overrides

    base = UBTConfig(draft_model="modelA")
    updated = apply_config_overrides(base, {"draft_model": "deepseek-chat"})
    assert updated.draft_model == "deepseek-chat"
    assert updated.repair_model == "deepseek-chat"


@pytest.mark.fast
def test_service_api_key_is_secret_str_and_masked_in_repr() -> None:
    """UBTConfig.service_api_key must be SecretStr so repr() and str() never leak it."""
    cfg = UBTConfig(service_api_key=SecretStr("top-secret-inbound-gate-key"))
    assert isinstance(cfg.service_api_key, SecretStr)
    assert cfg.service_api_key.get_secret_value() == "top-secret-inbound-gate-key"
    assert "top-secret-inbound-gate-key" not in repr(cfg)
    assert "top-secret-inbound-gate-key" not in str(cfg)
