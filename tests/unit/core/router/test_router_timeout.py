"""Tests for ModelRouter request_timeout enforcement."""

from __future__ import annotations

import asyncio

import pytest

from ubt.core.exceptions import ModelProviderError
from ubt.core.ir.models import BlockType, IRBlock, make_element
from ubt.core.router.provider import MockModelProvider
from ubt.core.router.router import ModelRouter

pytestmark = pytest.mark.fast


class _SlowProvider(MockModelProvider):
    """Provider that delays its answer by a configurable time."""

    def __init__(self, delay_sec: float, fail_times: int = 1) -> None:
        super().__init__()
        self.delay_sec = delay_sec
        self.fail_times = fail_times
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
        if self.calls <= self.fail_times:
            await asyncio.sleep(self.delay_sec)
        return "<translation>hello</translation>", None


@pytest.mark.asyncio
async def test_router_enforces_request_timeout_and_retries() -> None:
    # First call hangs (0.3s > 0.05s timeout); second call returns immediately.
    provider = _SlowProvider(delay_sec=0.3, fail_times=1)
    router = ModelRouter(
        provider=provider,
        draft_model="mock-fast",
        max_retries=2,
        request_timeout=0.05,
    )
    el1 = make_element(
        id="b1",
        spine_index=0,
        block_type=BlockType.NARRATIVE,
        source_text="hello world",
    )
    block = IRBlock(element=el1)
    result = await router.draft(block)
    assert result == "hello"
    assert provider.calls == 2


@pytest.mark.asyncio
async def test_router_raises_after_exhausting_timeout_retries() -> None:
    # All calls hang for 0.3s; with timeout 0.05s and max_retries=1, it raises 408 error.
    provider = _SlowProvider(delay_sec=0.3, fail_times=5)
    router = ModelRouter(
        provider=provider,
        draft_model="mock-fast",
        max_retries=1,
        request_timeout=0.05,
    )
    el2 = make_element(
        id="b2",
        spine_index=0,
        block_type=BlockType.NARRATIVE,
        source_text="hello world",
    )
    block = IRBlock(element=el2)
    with pytest.raises(ModelProviderError) as exc_info:
        await router.draft(block)

    assert "timed out after 0.05s" in str(exc_info.value)
    assert exc_info.value.details.get("status_code") == 408
    assert provider.calls == 2
