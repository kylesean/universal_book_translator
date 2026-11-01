from __future__ import annotations

import asyncio
from contextlib import suppress

import pytest

from ubt.core.exceptions import JobInterruptedError, ModelProviderError
from ubt.core.router.provider import MockModelProvider
from ubt.core.router.router import (
    _RETRY_AFTER_MAX_SECONDS,
    ModelRouter,
    _cancel_token_ctx,
    set_run_cancel_token,
)

pytestmark = pytest.mark.fast


class _RateLimitedProvider(MockModelProvider):
    def __init__(self, retry_after: str, failures: int = 1) -> None:
        super().__init__()
        self._retry_after = retry_after
        self._failures_left = failures
        self.calls = 0

    async def generate_with_finish_reason(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = 0.3,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> tuple[str, str | None]:
        self.calls += 1
        if self._failures_left > 0:
            self._failures_left -= 1
            raise ModelProviderError(
                "429 Too Many Requests",
                details={"status_code": 429, "retry_after": self._retry_after},
            )
        return "[TRANSLATED]", None


@pytest.mark.asyncio
async def test_an_absurd_retry_after_is_clamped_before_the_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(ModelRouter, "_interruptible_sleep", lambda self, s: fake_sleep(s))
    provider = _RateLimitedProvider("86400")
    router = ModelRouter(provider=provider, draft_model="m", max_retries=1)

    await router._execute_single_model("sys", "user", "m", 0.0)

    assert len(sleeps) == 1
    assert sleeps[0] <= _RETRY_AFTER_MAX_SECONDS * 1.25
    assert sleeps[0] < 86400


@pytest.mark.asyncio
async def test_a_modest_retry_after_is_still_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(ModelRouter, "_interruptible_sleep", lambda self, s: fake_sleep(s))
    provider = _RateLimitedProvider("7")
    router = ModelRouter(provider=provider, draft_model="m", max_retries=1)

    await router._execute_single_model("sys", "user", "m", 0.0)

    assert len(sleeps) == 1
    assert sleeps[0] >= 7


@pytest.mark.asyncio
async def test_a_backoff_sleep_aborts_on_cancel() -> None:
    token = asyncio.Event()
    provider = _RateLimitedProvider("3600")
    router = ModelRouter(provider=provider, draft_model="m", max_retries=3, cancel_token=token)

    async def cancel_soon() -> None:
        await asyncio.sleep(0.02)
        token.set()

    canceller = asyncio.ensure_future(cancel_soon())
    try:
        with pytest.raises(JobInterruptedError):
            await asyncio.wait_for(router._interruptible_sleep(3600), timeout=2.0)
    finally:
        await canceller


@pytest.mark.asyncio
async def test_a_backoff_sleep_without_a_token_still_sleeps() -> None:
    router = ModelRouter(provider=MockModelProvider(), draft_model="m")
    assert router.cancel_token is None
    await router._interruptible_sleep(0)


@pytest.mark.asyncio
async def test_two_jobs_on_one_shared_router_cancel_independently() -> None:
    router = ModelRouter(provider=MockModelProvider(), draft_model="m")

    token_a, token_b = asyncio.Event(), asyncio.Event()

    async def job_a() -> str:
        set_run_cancel_token(token_a)
        await router._interruptible_sleep(60)
        return "a-finished"

    async def job_b() -> str:
        set_run_cancel_token(token_b)
        await router._interruptible_sleep(60)
        return "b-finished"

    async def cancel_b_soon() -> None:
        await asyncio.sleep(0.02)
        token_b.set()

    task_a = asyncio.ensure_future(job_a())
    task_b = asyncio.ensure_future(job_b())
    canceller = asyncio.ensure_future(cancel_b_soon())
    try:
        with pytest.raises(JobInterruptedError):
            await asyncio.wait_for(task_b, timeout=2.0)
        await asyncio.sleep(0.05)
        assert not task_a.done()
    finally:
        for task in (task_a, task_b, canceller):
            task.cancel()
        await asyncio.gather(task_a, task_b, canceller, return_exceptions=True)
    token_c = asyncio.Event()
    router_c = ModelRouter(provider=MockModelProvider(), draft_model="m", cancel_token=token_c)
    token_c.set()
    with pytest.raises(JobInterruptedError):
        await asyncio.wait_for(router_c._interruptible_sleep(60), timeout=2.0)


@pytest.mark.asyncio
async def test_two_jobs_sharing_a_router_cancel_independently() -> None:
    router = ModelRouter(provider=MockModelProvider(), draft_model="m")
    token_a, token_b = asyncio.Event(), asyncio.Event()
    results: dict[str, str] = {}

    async def job(name: str, token: asyncio.Event, cancel_after: float | None) -> None:
        set_run_cancel_token(token)
        if cancel_after is not None:
            await asyncio.sleep(cancel_after)
            token.set()
        try:
            await router._interruptible_sleep(3600)
            results[name] = "slept-out"
        except JobInterruptedError:
            results[name] = "interrupted"

    gather_task = asyncio.ensure_future(
        asyncio.gather(job("a", token_a, None), job("b", token_b, 0.02))
    )
    await asyncio.sleep(0.1)
    assert results.get("b") == "interrupted"
    assert "a" not in results, "job B's cancel must not abort job A's backoff"
    gather_task.cancel()
    with suppress(asyncio.CancelledError):
        await gather_task


@pytest.mark.asyncio
async def test_construction_cancel_token_still_works_as_default() -> None:
    token = asyncio.Event()
    router = ModelRouter(provider=MockModelProvider(), draft_model="m", cancel_token=token)
    token.set()
    with pytest.raises(JobInterruptedError):
        await router._interruptible_sleep(3600)


@pytest.mark.asyncio
async def test_shared_router_cancel_of_one_job_does_not_abort_the_other() -> None:
    router = ModelRouter(provider=MockModelProvider(), draft_model="m")
    token_a, token_b = asyncio.Event(), asyncio.Event()
    outcomes: dict[str, str] = {}

    async def job(name: str, token: asyncio.Event) -> None:
        set_run_cancel_token(token)
        try:
            await asyncio.wait_for(router._interruptible_sleep(30), timeout=2.0)
            outcomes[name] = "slept_out"
        except JobInterruptedError:
            outcomes[name] = "interrupted"

    task_a = asyncio.ensure_future(job("a", token_a))
    await asyncio.sleep(0.01)
    task_b = asyncio.ensure_future(job("b", token_b))
    await asyncio.sleep(0.01)
    token_b.set()
    await asyncio.wait_for(task_b, timeout=2.0)
    assert outcomes["b"] == "interrupted"
    await asyncio.sleep(0.05)
    assert "a" not in outcomes or outcomes["a"] != "interrupted"
    token_a.set()
    await asyncio.wait_for(task_a, timeout=2.0)
    assert outcomes["a"] == "interrupted"
    assert _cancel_token_ctx.get() is None


@pytest.mark.asyncio
async def test_a_cancelled_job_stops_the_retry_loop() -> None:
    token = asyncio.Event()
    provider = _RateLimitedProvider("3600", failures=10)
    router = ModelRouter(provider=provider, draft_model="m", max_retries=10, cancel_token=token)

    async def cancel_soon() -> None:
        await asyncio.sleep(0.02)
        token.set()

    canceller = asyncio.ensure_future(cancel_soon())
    try:
        with pytest.raises(JobInterruptedError):
            await asyncio.wait_for(
                router._execute_single_model("sys", "user", "m", 0.0), timeout=2.0
            )
    finally:
        await canceller


async def test_the_fallback_chain_has_a_total_attempt_budget() -> None:
    from typing import Any, cast

    from ubt.core.router.router import ModelRouter

    class _AlwaysFails:
        is_mock = False

        def __init__(self) -> None:
            self.calls = 0

        async def generate_with_finish_reason(self, **kwargs: Any) -> tuple[str, str | None]:
            self.calls += 1
            raise ModelProviderError("boom", details={"status_code": 503})

    provider = _AlwaysFails()
    router = ModelRouter(
        provider=cast(Any, provider),
        draft_model="m1",
        fallback_models=["m2", "m3"],
        max_retries=2,
    )
    with pytest.raises(ModelProviderError):
        await router.complete_raw("sys", "translate this text")

    assert provider.calls == 3
