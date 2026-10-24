"""P2-8: a rate-limit backoff is bounded, logged, and cancellable.

Two defects lived in the 429 retry path of :meth:`ModelRouter._execute_single_model`:

* the ``Retry-After`` header was honoured verbatim, so a server answering
  ``Retry-After: 86400`` parked every concurrent worker for 24 hours with no log
  line — a longer outage is better surfaced than silently ridden out;
* the resulting ``asyncio.sleep`` had no cancel point, so a job cancelled during
  the pause kept a worker pinned until the full header elapsed.

The fix clamps the pause to :data:`_RETRY_AFTER_MAX_SECONDS` and races the sleep
against the router's cooperative ``cancel_token``, raising
:class:`JobInterruptedError` (the same stop the stages raise) when it is set.
"""

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
    """Mock provider whose first call answers 429 with a Retry-After header."""

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
    """A 24h Retry-After must not be slept verbatim; it is capped."""
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(ModelRouter, "_interruptible_sleep", lambda self, s: fake_sleep(s))
    provider = _RateLimitedProvider("86400")
    router = ModelRouter(provider=provider, draft_model="m", max_retries=1)

    await router._execute_single_model("sys", "user", "m", 0.0)

    assert len(sleeps) == 1
    # Clamped to the ceiling (plus at most 25% jitter on the ceiling itself).
    assert sleeps[0] <= _RETRY_AFTER_MAX_SECONDS * 1.25
    assert sleeps[0] < 86400


@pytest.mark.asyncio
async def test_a_modest_retry_after_is_still_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    """The clamp must not shrink a reasonable pause below the header."""
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
    """A set cancel token cuts the backoff short instead of sleeping it out."""
    token = asyncio.Event()
    provider = _RateLimitedProvider("3600")
    router = ModelRouter(provider=provider, draft_model="m", max_retries=3, cancel_token=token)

    async def cancel_soon() -> None:
        await asyncio.sleep(0.02)
        token.set()

    canceller = asyncio.ensure_future(cancel_soon())
    try:
        # Without the cancel point this would sleep an hour; the token fires in
        # ~20ms and the wait must abort with the cooperative-stop exception.
        with pytest.raises(JobInterruptedError):
            await asyncio.wait_for(router._interruptible_sleep(3600), timeout=2.0)
    finally:
        await canceller


@pytest.mark.asyncio
async def test_a_backoff_sleep_without_a_token_still_sleeps() -> None:
    """No token configured: the plain sleep path is unchanged."""
    router = ModelRouter(provider=MockModelProvider(), draft_model="m")
    assert router.cancel_token is None
    await router._interruptible_sleep(0)  # returns immediately, no token to race


@pytest.mark.asyncio
async def test_two_jobs_on_one_shared_router_cancel_independently() -> None:
    """The cancel token is per-task (ContextVar), not per-router.

    A shared router (JobWorker with concurrency>1) serves several jobs at
    once; the old field write let job B's run() overwrite job A's token, so
    cancelling B aborted A's backoff while cancelling A could not interrupt
    A's own sleep. The ContextVar is inherited per asyncio task, so each
    run's context carries its own token.
    """
    router = ModelRouter(provider=MockModelProvider(), draft_model="m")

    token_a, token_b = asyncio.Event(), asyncio.Event()

    async def job_a() -> str:
        set_run_cancel_token(token_a)
        # A sleeps "forever"; only A's own token can cut it.
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
        # B is cancelled; A must still be sleeping (not aborted by B's token).
        await asyncio.sleep(0.05)
        assert not task_a.done()
    finally:
        for task in (task_a, task_b, canceller):
            task.cancel()
        await asyncio.gather(task_a, task_b, canceller, return_exceptions=True)
    # The construction-time field remains the fallback when no ContextVar is
    # set (library callers that never go through the pipeline).
    token_c = asyncio.Event()
    router_c = ModelRouter(provider=MockModelProvider(), draft_model="m", cancel_token=token_c)
    token_c.set()
    with pytest.raises(JobInterruptedError):
        await asyncio.wait_for(router_c._interruptible_sleep(60), timeout=2.0)


@pytest.mark.asyncio
async def test_two_jobs_sharing_a_router_cancel_independently() -> None:
    """A per-run ContextVar token, not a shared field.

    JobWorker(concurrency>1) drives several jobs over ONE router instance.
    The old field write meant job B's run() overwrote job A's token: a
    cancel for B aborted A's backoff (A died with JobInterruptedError while
    B kept running), and a cancel for A could not interrupt A's own sleep.
    The ContextVar is inherited per asyncio task, so each job's context
    carries its own token.
    """
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

    # A never cancels: it must sleep on untouched. B starts later (in the old
    # field-write world its set_run_cancel_token overwrote A's) and is
    # cancelled quickly — B must be the one interrupted, and only B.
    gather_task = asyncio.ensure_future(
        asyncio.gather(job("a", token_a, None), job("b", token_b, 0.02))
    )
    await asyncio.sleep(0.1)
    assert results.get("b") == "interrupted"
    assert "a" not in results, "job B's cancel must not abort job A's backoff"
    # A is still sleeping; tear the gather down without waiting 3600s.
    gather_task.cancel()
    with suppress(asyncio.CancelledError):
        await gather_task


@pytest.mark.asyncio
async def test_construction_cancel_token_still_works_as_default() -> None:
    """The __init__ token remains a valid construction default (library use)."""
    token = asyncio.Event()
    router = ModelRouter(provider=MockModelProvider(), draft_model="m", cancel_token=token)
    token.set()
    with pytest.raises(JobInterruptedError):
        await router._interruptible_sleep(3600)


@pytest.mark.asyncio
async def test_shared_router_cancel_of_one_job_does_not_abort_the_other() -> None:
    """Two jobs over one router: the ContextVar token stays per-task.

    The old field write (`router.cancel_token = token` in every run()) let job
    B's start overwrite job A's token: cancelling B aborted A's backoff, and
    cancelling A could not interrupt A's own sleep. The token now rides a
    ContextVar each run sets in its own task context.
    """
    router = ModelRouter(provider=MockModelProvider(), draft_model="m")
    token_a, token_b = asyncio.Event(), asyncio.Event()
    outcomes: dict[str, str] = {}

    async def job(name: str, token: asyncio.Event) -> None:
        set_run_cancel_token(token)
        # Wait long enough that the other job can start and cancel first.
        try:
            await asyncio.wait_for(router._interruptible_sleep(30), timeout=2.0)
            outcomes[name] = "slept_out"
        except JobInterruptedError:
            outcomes[name] = "interrupted"

    task_a = asyncio.ensure_future(job("a", token_a))
    await asyncio.sleep(0.01)
    task_b = asyncio.ensure_future(job("b", token_b))
    await asyncio.sleep(0.01)
    # Cancel B only. A's token must be untouched.
    token_b.set()
    await asyncio.wait_for(task_b, timeout=2.0)
    assert outcomes["b"] == "interrupted"
    await asyncio.sleep(0.05)
    assert "a" not in outcomes or outcomes["a"] != "interrupted"
    # Now cancel A in its own context: it interrupts too.
    token_a.set()
    await asyncio.wait_for(task_a, timeout=2.0)
    assert outcomes["a"] == "interrupted"
    # And the tasks ran in their own contexts (the ContextVar set inside one
    # task cannot leak into the caller or the sibling).
    assert _cancel_token_ctx.get() is None


@pytest.mark.asyncio
async def test_a_cancelled_job_stops_the_retry_loop() -> None:
    """The retry loop surfaces JobInterruptedError, not a fresh draft."""
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
    """``max_retries`` is per model, so N models meant N*(max_retries+1) calls.

    Each retry can sleep up to a clamped 375 s, so a 3-model chain with the
    default 3 retries could spend 12 paid calls and over an hour on one block
    that is going to fail. The chain now shares one attempt budget, so a bad
    chain fails in bounded time instead of multiplying per-model retries.
    """
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

    # 3 models x (2 retries + 1) would be 9; the chain budget is 3.
    assert provider.calls == 3
