"""Router ownership and per-run usage accounting."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr

from ubt.core.config import UBTConfig
from ubt.core.engine.pipeline import PipelineOrchestrator
from ubt.core.exceptions import UBTError
from ubt.core.router.provider import MockModelProvider
from ubt.core.router.router import ModelRouter


def _config(tmp_path: Path) -> UBTConfig:
    return UBTConfig.from_env(
        api_key=SecretStr("test-key"), db_dir=tmp_path, draft_model="mock-draft"
    )


def test_injected_router_is_not_owned(tmp_path: Path) -> None:
    router = ModelRouter(provider=MockModelProvider(default_response="x"), draft_model="m")
    orch = PipelineOrchestrator(config=_config(tmp_path), router=router)
    assert orch._owns_router is False


def test_default_router_is_owned(tmp_path: Path) -> None:
    orch = PipelineOrchestrator(config=_config(tmp_path))
    assert orch._owns_router is True


def test_building_its_own_router_requires_a_draft_model(tmp_path: Path) -> None:
    """No router injected and no model configured is a configuration error."""
    config = UBTConfig.from_env(api_key=SecretStr("test-key"), db_dir=tmp_path)
    assert config.draft_model == ""
    with pytest.raises(UBTError, match="No draft model configured"):
        PipelineOrchestrator(config=config)


def test_an_injected_router_needs_no_configured_model(tmp_path: Path) -> None:
    """An injected router owns the model choice, so a blank config is fine."""
    config = UBTConfig.from_env(api_key=SecretStr("test-key"), db_dir=tmp_path)
    router = ModelRouter(provider=MockModelProvider(default_response="x"), draft_model="m")
    orch = PipelineOrchestrator(config=config, router=router)
    assert orch.router is router


def test_usage_delta_subtracts_baseline() -> None:
    assert PipelineOrchestrator._usage_delta({}, {"m": {"calls": 2}}) == {"m": {"calls": 2}}
    delta = PipelineOrchestrator._usage_delta(
        {"m": {"calls": 1, "prompt_tokens": 10}},
        {"m": {"calls": 3, "prompt_tokens": 25}},
    )
    assert delta == {"m": {"calls": 2, "prompt_tokens": 15}}


def test_usage_delta_drops_unchanged_models() -> None:
    delta = PipelineOrchestrator._usage_delta(
        {"m": {"calls": 1}}, {"m": {"calls": 1}, "n": {"calls": 1}}
    )
    assert delta == {"n": {"calls": 1}}
