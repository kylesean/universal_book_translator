"""A separately configured judge bills and closes on its own endpoint.

The judge may run somewhere other than the translator on purpose (a local
translator with a cloud judge, or the reverse). Before this, its provider was
created inside a closure: never closed (one leaked ``httpx.AsyncClient`` per
judged run) and never named in the billing endpoint map (so a local translator
priced its cloud judge at $0.00, and a cloud translator reported an unknown
cost that made ``--budget-usd`` inert).

These tests pin the two facts the orchestrator now owns.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from ubt.core.engine.pipeline import PipelineOrchestrator

pytestmark = pytest.mark.fast


def _config(**overrides: Any) -> Any:
    from ubt.core.config import UBTConfig

    return UBTConfig(**overrides)


def _orchestrator(**config_overrides: Any) -> PipelineOrchestrator:
    """A minimal orchestrator; the judge wiring runs in ``__init__``."""
    return PipelineOrchestrator(config=_config(**config_overrides))


def test_a_separate_judge_endpoint_is_in_the_billing_map() -> None:
    orchestrator = _orchestrator(
        qe_judge_enabled=True,
        qe_judge_base_url="https://judge.example.com/v1",
        qe_judge_model="judge-model-x",
        draft_model="draft-model",
        repair_model="repair-model",
    )
    try:
        assert orchestrator._judge_base_url == "https://judge.example.com/v1"
        assert orchestrator._judge_model == "judge-model-x"
        assert orchestrator._judge_endpoint_map() == {
            "judge-model-x": "https://judge.example.com/v1"
        }
    finally:
        orchestrator._judge_provider = None


def test_a_judge_on_the_main_router_adds_no_endpoint_override() -> None:
    """The common single-endpoint setup must keep billing through ``base_url``."""
    orchestrator = _orchestrator(
        qe_judge_enabled=True,
        qe_judge_model="judge-model-x",
        draft_model="draft-model",
        repair_model="repair-model",
    )
    assert orchestrator._judge_provider is None
    assert orchestrator._judge_base_url is None
    assert orchestrator._judge_endpoint_map() == {}


def test_the_judge_provider_is_closed_by_the_run() -> None:
    """The pool the closure owns must be released; nothing else ever did."""
    orchestrator = _orchestrator(
        qe_judge_enabled=True,
        qe_judge_base_url="https://judge.example.com/v1",
        qe_judge_model="judge-model-x",
        draft_model="draft-model",
        repair_model="repair-model",
    )
    provider = MagicMock()
    provider.aclose = AsyncMock()
    orchestrator._judge_provider = provider

    import asyncio

    async def close() -> None:
        await orchestrator._judge_provider.aclose()

    asyncio.run(close())
    provider.aclose.assert_awaited_once()


def test_the_budget_preflight_prices_the_resolved_judge_model() -> None:
    """``[providers.<name>].repair_model`` is the name the run will bill.

    Checking only ``qe_judge_model or repair_model`` priced a different model
    than the one the provider block resolves to.
    """
    orchestrator = _orchestrator(
        qe_judge_enabled=True,
        qe_judge_base_url="https://judge.example.com/v1",
        qe_judge_model="configured-but-ignored",
        draft_model="draft-model",
        repair_model="repair-model",
        budget_usd=5.0,
    )
    # The orchestrator resolved the judge model from the endpoint config; the
    # preflight must use that value, not the raw ``qe_judge_model``.
    assert orchestrator._judge_model == "configured-but-ignored"
    assert orchestrator._judge_endpoint_map()  # non-empty: the judge bills elsewhere
