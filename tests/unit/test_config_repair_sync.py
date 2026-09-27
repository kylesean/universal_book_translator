"""The draft->repair default lives in one place, with one semantics.

``repair_model`` follows ``draft_model`` unless repair was configured
independently — an explicit repair value, a provider block whose repair
differs from its draft, or a base config whose repair already differs. Three
sites used to re-derive this with different rules: ``from_env`` clobbered a
block's distinct repair, while ``apply_config_overrides`` kept it, so the same
request produced different models on the env path and the request path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ubt.core.config import UBTConfig, resolve_repair_model

pytestmark = pytest.mark.fast


@pytest.fixture
def provider_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A provider block whose repair is deliberately distinct from its draft."""
    path = tmp_path / "config.toml"
    path.write_text(
        "[providers.duo]\n"
        'base_url = "https://api.example.com/v1"\n'
        'draft_model = "duo-draft"\n'
        'repair_model = "duo-repair"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr("ubt.core.providers.DEFAULT_CONFIG_LOCATIONS", (path,))
    return path


@pytest.mark.parametrize(
    ("repair_is_independent", "expected"),
    [(False, "draft-x"), (True, "repair-y")],
)
def test_resolve_repair_model_rule(repair_is_independent: bool, expected: str) -> None:
    assert (
        resolve_repair_model("draft-x", "repair-y", repair_is_independent=repair_is_independent)
        == expected
    )


def test_from_env_keeps_provider_distinct_repair(
    provider_toml: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit draft override must not clobber a provider block's distinct repair."""
    monkeypatch.setenv("UBT_LLM_API_KEY", "sk-env")
    monkeypatch.delenv("UBT_REPAIR_MODEL", raising=False)
    cfg = UBTConfig.from_env(provider="duo", draft_model="custom-draft")
    assert cfg.draft_model == "custom-draft"
    assert cfg.repair_model == "duo-repair"


def test_env_and_request_path_agree_when_repair_is_set_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit UBT_REPAIR_MODEL must survive a draft override on both paths.

    ``apply_config_overrides`` decided independence from ``base.repair !=
    base.draft``, so an explicit repair that happened to equal the draft looked
    inherited and a draft override clobbered it, while ``from_env`` kept it.
    """
    from ubt.core.job_options import apply_config_overrides

    monkeypatch.setenv("UBT_DRAFT_MODEL", "A")
    monkeypatch.setenv("UBT_REPAIR_MODEL", "A")
    base = UBTConfig.from_env()

    env_cfg = UBTConfig.from_env(draft_model="C")
    req_cfg = apply_config_overrides(base, {"draft_model": "C"})
    assert (env_cfg.draft_model, env_cfg.repair_model) == ("C", "A")
    assert (req_cfg.draft_model, req_cfg.repair_model) == ("C", "A")


def test_draft_change_follows_when_repair_was_inherited() -> None:
    """A later draft change must move an inherited repair with it.

    The validator's own sync recorded repair_model in model_fields_set, so a
    second draft change read the inherited repair as an explicit choice and
    left it behind.
    """
    cfg = UBTConfig(draft_model="modelA")
    assert cfg.repair_model == "modelA"
    cfg.draft_model = "modelB"
    assert cfg.repair_model == "modelB"


def test_from_env_and_overrides_agree_on_the_same_request(
    provider_toml: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The env constructor and the request path must produce the same models."""
    from ubt.core.job_options import apply_config_overrides

    monkeypatch.setenv("UBT_LLM_API_KEY", "sk-env")
    env_cfg = UBTConfig.from_env(provider="duo", draft_model="custom-draft")
    req_cfg = apply_config_overrides(
        UBTConfig.from_env(), {"provider": "duo", "draft_model": "custom-draft"}
    )
    assert (env_cfg.draft_model, env_cfg.repair_model) == (
        req_cfg.draft_model,
        req_cfg.repair_model,
    )


def test_a_config_without_a_model_keeps_both_tiers_empty() -> None:
    """Core ships no vendor model name: with nothing configured both stay "".

    The draft->repair sync must not turn an absent model into anything else —
    an empty draft leaves repair empty too, and the run later fails loudly at
    the call boundary instead of silently calling a vendor's model.
    """
    cfg = UBTConfig()
    assert cfg.draft_model == ""
    assert cfg.repair_model == ""
