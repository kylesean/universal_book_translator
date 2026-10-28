"""Unit tests for lease-loss and cancellation propagation in ModelRouter."""

from __future__ import annotations

import asyncio

import pytest

from ubt.core.exceptions import (
    JobInterruptedError,
    LedgerWriterLeaseLostError,
    ModelProviderError,
)
from ubt.core.router.provider import MockModelProvider
from ubt.core.router.router import (
    ModelRouter,
    _cancel_token_ctx,
    _lease_lost_ctx,
    _raise_if_cancelled,
    set_run_cancel_token,
)

pytestmark = pytest.mark.fast


class _FailingRetryProvider(MockModelProvider):
    """Mock provider that always raises 429."""

    def __init__(self) -> None:
        super().__init__()
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
        raise ModelProviderError(
            "429 Rate Limit",
            details={"status_code": 429, "retry_after": "100"},
        )


@pytest.mark.asyncio
async def test_raise_if_cancelled_distinguishes_events() -> None:
    set_run_cancel_token(None, None)
    # Neither set: no error
    _raise_if_cancelled()

    # Cancel token set: JobInterruptedError
    cancel_token = asyncio.Event()
    cancel_token.set()
    set_run_cancel_token(cancel_token, None)
    with pytest.raises(JobInterruptedError, match="Job cancelled"):
        _raise_if_cancelled()

    # Lease lost set: LedgerWriterLeaseLostError
    lease_lost = asyncio.Event()
    lease_lost.set()
    set_run_cancel_token(None, lease_lost)
    with pytest.raises(LedgerWriterLeaseLostError, match="Writer lease was reclaimed"):
        _raise_if_cancelled()

    # Both set: Lease lost takes priority
    set_run_cancel_token(cancel_token, lease_lost)
    with pytest.raises(LedgerWriterLeaseLostError, match="Writer lease was reclaimed"):
        _raise_if_cancelled()

    # Reset
    set_run_cancel_token(None, None)
    _raise_if_cancelled()


@pytest.mark.asyncio
async def test_interruptible_sleep_preset_lease_lost() -> None:
    router = ModelRouter(provider=MockModelProvider(), draft_model="m")
    lease_lost = asyncio.Event()
    lease_lost.set()
    set_run_cancel_token(None, lease_lost)
    try:
        with pytest.raises(LedgerWriterLeaseLostError, match="Writer lease was reclaimed"):
            await router._interruptible_sleep(100.0)
    finally:
        set_run_cancel_token(None, None)


@pytest.mark.asyncio
async def test_interruptible_sleep_preset_cancel_token() -> None:
    router = ModelRouter(provider=MockModelProvider(), draft_model="m")
    cancel_token = asyncio.Event()
    cancel_token.set()
    set_run_cancel_token(cancel_token, None)
    try:
        with pytest.raises(JobInterruptedError, match="Job cancelled"):
            await router._interruptible_sleep(100.0)
    finally:
        set_run_cancel_token(None, None)


@pytest.mark.asyncio
async def test_interruptible_sleep_race_lease_lost() -> None:
    router = ModelRouter(provider=MockModelProvider(), draft_model="m")
    lease_lost = asyncio.Event()
    set_run_cancel_token(None, lease_lost)

    async def trigger_lease_lost() -> None:
        await asyncio.sleep(0.02)
        lease_lost.set()

    asyncio.create_task(trigger_lease_lost())
    try:
        with pytest.raises(LedgerWriterLeaseLostError, match="Writer lease was reclaimed"):
            await asyncio.wait_for(router._interruptible_sleep(100.0), timeout=2.0)
    finally:
        set_run_cancel_token(None, None)


@pytest.mark.asyncio
async def test_interruptible_sleep_race_cancel_token() -> None:
    router = ModelRouter(provider=MockModelProvider(), draft_model="m")
    cancel_token = asyncio.Event()
    set_run_cancel_token(cancel_token, None)

    async def trigger_cancel() -> None:
        await asyncio.sleep(0.02)
        cancel_token.set()

    asyncio.create_task(trigger_cancel())
    try:
        with pytest.raises(JobInterruptedError, match="Job cancelled"):
            await asyncio.wait_for(router._interruptible_sleep(100.0), timeout=2.0)
    finally:
        set_run_cancel_token(None, None)


@pytest.mark.asyncio
async def test_execute_single_model_aborts_immediately_on_lease_lost() -> None:
    provider = _FailingRetryProvider()
    router = ModelRouter(provider=provider, draft_model="m", max_retries=5)
    lease_lost = asyncio.Event()
    set_run_cancel_token(None, lease_lost)

    async def trigger_lease_lost() -> None:
        await asyncio.sleep(0.02)
        lease_lost.set()

    asyncio.create_task(trigger_lease_lost())
    try:
        with pytest.raises(LedgerWriterLeaseLostError):
            await asyncio.wait_for(
                router._execute_single_model("sys", "user", "m", 0.0),
                timeout=2.0,
            )
        # Should have failed on the first retry sleep without exhausting all 5 retries
        assert provider.calls == 1
    finally:
        set_run_cancel_token(None, None)


@pytest.mark.asyncio
async def test_task_isolation_between_concurrent_jobs() -> None:
    router = ModelRouter(provider=MockModelProvider(), draft_model="m")
    lease_lost_a = asyncio.Event()
    cancel_b = asyncio.Event()
    results: dict[str, str] = {}

    async def job_a() -> None:
        set_run_cancel_token(None, lease_lost_a)
        try:
            await router._interruptible_sleep(100.0)
            results["a"] = "completed"
        except LedgerWriterLeaseLostError:
            results["a"] = "lease_lost"
        except JobInterruptedError:
            results["a"] = "cancelled"

    async def job_b() -> None:
        set_run_cancel_token(cancel_b, None)
        try:
            await router._interruptible_sleep(100.0)
            results["b"] = "completed"
        except LedgerWriterLeaseLostError:
            results["b"] = "lease_lost"
        except JobInterruptedError:
            results["b"] = "cancelled"

    task_a = asyncio.create_task(job_a())
    task_b = asyncio.create_task(job_b())

    await asyncio.sleep(0.01)
    # Trigger lease loss for job A
    lease_lost_a.set()
    await asyncio.wait_for(task_a, timeout=2.0)
    assert results.get("a") == "lease_lost"
    assert "b" not in results

    # Now trigger cancellation for job B
    cancel_b.set()
    await asyncio.wait_for(task_b, timeout=2.0)
    assert results.get("b") == "cancelled"

    # ContextVars in caller remain clean
    assert _cancel_token_ctx.get() is None
    assert _lease_lost_ctx.get() is None
