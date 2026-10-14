"""Router funnel coverage: the retry/continuation/vision paths behind the wire.

The transports' wire contract is pinned in ``test_transports_wire.py``; this
module covers the *router* logic that wraps every provider call — the parts a
wrong answer bills real money or stalls a whole book:

- **Truncation continuation**: a ``finish_reason == "length"`` generation is
  continued tail-anchored, the echoed overlap is merged away, and a failed
  continuation keeps the partial (half a block beats no block) — but a
  budget/cancel hard stop propagates instead of continuing a still-billing
  generation.
- **Retry classification**: a 429 reports to the AIMD limiter and retries;
  an unparseable 200 fails the model fast (retrying the same request only
  burns quota); a hard stop (budget/cancel) is never retried.
- **Vision egress**: page images ship only when the run allows it, and the
  request is charged at the vision tile count.
- **Repair tier**: repair calls reach the repair provider, not the draft one.

Every test drives :class:`ModelRouter` over a scripted fake provider — no
network, no limiter wall-clock waits (the limiter is a pass-through stub).
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from ubt.core.exceptions import BudgetExceededError, ModelProviderError
from ubt.core.router.rate_limiter import AdaptiveTokenBucket
from ubt.core.router.router import ModelRouter

pytestmark = pytest.mark.fast


class _ScriptedProvider:
    """A provider whose generate_with_finish_reason plays a script of outcomes."""

    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    async def generate_with_finish_reason(self, **kwargs: Any) -> tuple[str, str | None]:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        text, finish = outcome
        return (str(text), finish)

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        text, _ = await self.generate_with_finish_reason(
            prompt=prompt, system_prompt=kwargs.get("system_prompt")
        )
        return text


class _NullLimiter:
    """A limiter that admits everything and records the reservations."""

    def __init__(self) -> None:
        self.reserved: list[int] = []
        self.reported_429 = 0

    async def acquire(self, estimated_tokens: int = 0) -> None:
        self.reserved.append(estimated_tokens)

    async def report_success_async(self) -> None:
        return None

    async def report_429_async(self) -> None:
        self.reported_429 += 1

    def report_success(self) -> None:
        return None

    def report_429(self) -> None:
        self.reported_429 += 1


def _router(provider: Any, **kwargs: Any) -> ModelRouter:
    kwargs.setdefault("draft_model", "m")
    kwargs.setdefault("rate_limiter", _NullLimiter())
    return ModelRouter(provider=provider, **kwargs)


# --------------------------------------------------------------------------- #
# Truncation continuation
# --------------------------------------------------------------------------- #


async def test_a_truncated_generation_is_continued_and_merged() -> None:
    provider = _ScriptedProvider(
        [
            ("partial half of a sente", "length"),
            ("nce that completes it", "stop"),
        ]
    )
    router = _router(provider)
    text = await router.complete_raw("sys", "user prompt")
    assert "partial half of a sente" in text
    assert "nce that completes it" in text
    # The continuation call re-sends the user prompt with the tail anchor.
    assert len(provider.calls) == 2
    continuation_prompt = provider.calls[1]["prompt"]
    assert "user prompt" in continuation_prompt
    assert "Continue it from exactly where it stopped" in continuation_prompt


async def test_a_failed_continuation_keeps_the_partial_output() -> None:
    provider = _ScriptedProvider(
        [
            ("partial output stays", "length"),
            ModelProviderError("503 continuation died"),
        ]
    )
    router = _router(provider)
    text = await router.complete_raw("sys", "user prompt")
    assert text == "partial output stays"


async def test_a_budget_stop_during_continuation_propagates() -> None:
    """A spend cap must not be swallowed: the generation is still billing."""
    provider = _ScriptedProvider(
        [
            ("partial output", "length"),
            BudgetExceededError("budget cap reached"),
        ]
    )
    router = _router(provider)
    with pytest.raises(BudgetExceededError):
        await router.complete_raw("sys", "user prompt")


# --------------------------------------------------------------------------- #
# Retry classification
# --------------------------------------------------------------------------- #


async def test_a_429_reports_to_the_limiter_and_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _ScriptedProvider(
        [
            ModelProviderError("429 Too Many Requests", details={"status_code": 429}),
            ("recovered", "stop"),
        ]
    )
    limiter = _NullLimiter()
    router = _router(provider, rate_limiter=limiter)
    # Collapse the backoff waits: the classification, not the wait, is the test.
    monkeypatch.setattr(router, "_interruptible_sleep", _async_noop)
    text = await router.complete_raw("sys", "user")
    assert text == "recovered"
    assert limiter.reported_429 == 1


async def _async_noop(_seconds: float) -> None:
    return None


async def test_a_hard_stop_is_never_retried() -> None:
    provider = _ScriptedProvider([BudgetExceededError("cap")])
    router = _router(provider)
    with pytest.raises(BudgetExceededError):
        await router.complete_raw("sys", "user")
    assert len(provider.calls) == 1, "a budget stop was retried"


async def test_an_unparseable_200_fails_the_model_fast() -> None:
    """A 200 whose body is not the declared JSON is deterministic: no retry."""
    provider = _ScriptedProvider([ValueError("not the declared JSON"), ("never reached", "stop")])
    router = _router(provider)
    with pytest.raises(ModelProviderError, match="could not be parsed"):
        await router.complete_raw("sys", "user")
    assert len(provider.calls) == 1


async def test_an_unexpected_error_is_wrapped_not_silently_dropped() -> None:
    provider = _ScriptedProvider([KeyError("internal bug")])
    router = _router(provider)
    with pytest.raises(ModelProviderError, match="Unexpected provider error"):
        await router.complete_raw("sys", "user")


# --------------------------------------------------------------------------- #
# Vision egress
# --------------------------------------------------------------------------- #


async def test_vision_is_refused_while_page_upload_is_disabled() -> None:
    provider = _ScriptedProvider([])
    router = _router(provider, allow_page_upload=False)
    with pytest.raises(ModelProviderError, match="page-image egress disabled"):
        await router.complete_with_images("describe", ["QUJD"])
    assert provider.calls == [], "a refused vision call reached the provider"


class _VisionProvider(_ScriptedProvider):
    async def generate_with_images(self, prompt: str, images: list[str], **kwargs: Any) -> str:
        self.calls.append({"prompt": prompt, "images": images, **kwargs})
        return "seen"

    async def generate_with_finish_reason(self, **kwargs: Any) -> tuple[str, str | None]:
        raise ModelProviderError("vision provider has no text channel")


async def test_vision_charges_the_tile_count_to_the_limiter() -> None:
    provider = _VisionProvider([])
    limiter = _NullLimiter()
    router = _router(provider, rate_limiter=limiter, allow_page_upload=True)
    text = await router.complete_with_images("describe this page", ["QUJD"] * 3)
    assert text == "seen"
    # The reservation must exceed the text-only estimate by the per-image
    # tile charge (one page image's vision tokens are far more than text).
    assert limiter.reserved, "the vision call bypassed the limiter"
    assert limiter.reserved[0] > 3000, limiter.reserved


# --------------------------------------------------------------------------- #
# Repair tier routing
# --------------------------------------------------------------------------- #


def _block(text: str) -> Any:
    """A minimal IRBlock for the repair call (only source_text/block_type read)."""
    from ubt.core.ir.models import BlockType, IRBlock, make_element

    return IRBlock(
        element=make_element(
            id="b1", spine_index=0, block_type=BlockType.NARRATIVE, source_text=text
        )
    )


async def test_repair_calls_reach_the_repair_provider() -> None:
    draft_provider = _ScriptedProvider([("from draft", "stop")])
    repair_provider = _ScriptedProvider([("translated repair", "stop")])
    router = _router(draft_provider, repair_provider=repair_provider)
    text = await router.repair(
        _block("source sentence"), draft_text="draft sentence", error_flags=["numeric"]
    )
    assert text == "translated repair"
    assert draft_provider.calls == []
    assert len(repair_provider.calls) == 1


# --------------------------------------------------------------------------- #
# The AIMD-built limiter still exists when none is injected
# --------------------------------------------------------------------------- #


def test_a_router_without_an_injected_limiter_builds_one_for_a_real_provider() -> None:
    """Mock traffic gets the never-gating limiter; a real provider gets AIMD.

    The duck-typed ``is_mock`` flag decides: a mock provider cannot exhaust a
    credential, so the bucket would only add latency to tests; anything else
    (the scripted stand-in has no ``is_mock``) must get a real limiter, with
    the oscillation guard at the config default rather than the bucket's 0.0.
    """

    class _RealProvider(_ScriptedProvider):
        is_mock = False

    router = ModelRouter(provider=cast(Any, _RealProvider([])), draft_model="m")
    assert isinstance(router.rate_limiter, AdaptiveTokenBucket)
    assert router.rate_limiter.backoff_cooldown_sec > 0

    from ubt.core.router.provider import MockModelProvider

    mock_router = ModelRouter(provider=MockModelProvider(), draft_model="m")
    # NullRateLimiter subclasses AdaptiveTokenBucket, so the exact type decides.
    assert type(mock_router.rate_limiter) is not AdaptiveTokenBucket
