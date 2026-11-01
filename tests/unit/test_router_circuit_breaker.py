from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from ubt.core.exceptions import ModelProviderError
from ubt.core.router.provider import MockModelProvider
from ubt.core.router.router import ModelRouter

pytestmark = pytest.mark.fast


@pytest.mark.asyncio
async def test_router_circuit_breaker_tripping_and_skip() -> None:
    provider = MockModelProvider()
    router = ModelRouter(
        provider=provider,
        draft_model="failing-model",
        fallback_models=["healthy-model"],
    )

    call_counts = {"failing-model": 0, "healthy-model": 0}

    async def fake_execute_single(
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float,
        reasoning_effort: str | None = None,
        max_tokens: int | None = None,
        **kwargs: object,
    ) -> str:
        call_counts[model] += 1
        if model == "failing-model":
            raise ModelProviderError("503 Service Unavailable")
        return f"translated by {model}"

    with patch.object(router, "_execute_single_model", side_effect=fake_execute_single):
        res1 = await router._execute_with_retry("sys", "user", "failing-model", 0.3)
        assert res1 == "translated by healthy-model"
        assert call_counts["failing-model"] == 1
        assert router._model_circuit["failing-model"][0] == 1

        res2 = await router._execute_with_retry("sys", "user", "failing-model", 0.3)
        assert res2 == "translated by healthy-model"
        assert call_counts["failing-model"] == 2
        assert router._model_circuit["failing-model"][0] == 2

        res3 = await router._execute_with_retry("sys", "user", "failing-model", 0.3)
        assert res3 == "translated by healthy-model"
        assert call_counts["failing-model"] == 3
        fails, cooldown = router._model_circuit["failing-model"]
        assert fails == 3
        assert cooldown > time.monotonic()

        res4 = await router._execute_with_retry("sys", "user", "failing-model", 0.3)
        assert res4 == "translated by healthy-model"
        assert call_counts["failing-model"] == 3
        assert call_counts["healthy-model"] == 4


@pytest.mark.asyncio
async def test_circuit_cooldown_uses_failure_time_not_entry_time() -> None:
    provider = MockModelProvider()
    router = ModelRouter(provider=provider, draft_model="failing-model", fallback_models=[])
    clock = {"t": 1000.0}

    def fake_monotonic() -> float:
        return clock["t"]

    async def slow_failure(*args: object, **kwargs: object) -> str:
        clock["t"] += 30.0
        raise ModelProviderError("429 Too Many Requests")

    with (
        patch.object(router, "_execute_single_model", side_effect=slow_failure),
        patch("ubt.core.router.router.time.monotonic", fake_monotonic),
    ):
        for _ in range(3):
            with pytest.raises(ModelProviderError):
                await router._execute_with_retry("sys", "user", "failing-model", 0.3)

    fails, cooldown = router._model_circuit["failing-model"]
    assert fails == 3
    assert cooldown == pytest.approx(1150.0)


@pytest.mark.asyncio
async def test_router_unstructured_auth_error_fails_fast() -> None:
    provider = MockModelProvider()
    router = ModelRouter(
        provider=provider,
        draft_model="auth-fail-model",
        fallback_models=["fallback-1", "fallback-2"],
    )
    attempted: list[str] = []

    async def fake_execute_single(
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float,
        reasoning_effort: str | None = None,
        max_tokens: int | None = None,
        **kwargs: object,
    ) -> str:
        attempted.append(model)
        raise ModelProviderError("Model API error (401): Unauthorized")

    with (
        patch.object(router, "_execute_single_model", side_effect=fake_execute_single),
        pytest.raises(ModelProviderError, match="401"),
    ):
        await router._execute_with_retry("sys", "user", "auth-fail-model", 0.3)

    assert attempted == ["auth-fail-model"], "an auth failure must not walk the fallback chain"


@pytest.mark.asyncio
async def test_router_repair_fallback_uses_repair_provider() -> None:
    primary_provider = MockModelProvider()
    repair_provider = MockModelProvider()
    router = ModelRouter(
        provider=primary_provider,
        repair_provider=repair_provider,
        repair_model="primary-repair",
        fallback_models=["fallback-repair"],
    )

    assert router._provider_for("fallback-repair", is_repair=True) is repair_provider
    assert router._provider_for("fallback-repair", is_repair=False) is primary_provider
