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

import pytest

from ubt.core.exceptions import JobInterruptedError, ModelProviderError
from ubt.core.router.provider import MockModelProvider
from ubt.core.router.router import _RETRY_AFTER_MAX_SECONDS, ModelRouter

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
