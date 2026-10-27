"""A reasoning model must not fail every block at the same output budget."""

from __future__ import annotations

import pytest

from ubt.core.exceptions import ModelProviderError
from ubt.core.router.provider import MockModelProvider
from ubt.core.router.router import (
    _REASONING_MIN_OUTPUT_TOKENS,
    _REASONING_OUTPUT_HEADROOM,
    ModelRouter,
)

pytestmark = pytest.mark.fast


class _ReasoningBudgetProvider(MockModelProvider):
    def __init__(self) -> None:
        super().__init__()
        self.budgets: list[int | None] = []
        self.prompts: list[str] = []

    async def generate_with_finish_reason(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = 0.3,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> tuple[str, str | None]:
        self.budgets.append(max_tokens)
        self.prompts.append(prompt)
        if len(self.budgets) == 1:
            raise ModelProviderError(
                "reasoning budget exhausted",
                details={"reasoning_budget_exhausted": True, "reasoning_tokens": max_tokens},
            )
        return "译文", "stop"


@pytest.mark.asyncio
async def test_reasoning_budget_is_widened_once() -> None:
    provider = _ReasoningBudgetProvider()
    router = ModelRouter(provider=provider, draft_model="reasoning-model")

    text = await router._execute_single_model(
        "sys", "user", "reasoning-model", 0.3, max_tokens=1024
    )

    assert text == "译文"
    assert provider.budgets == [1024, _REASONING_MIN_OUTPUT_TOKENS]


@pytest.mark.asyncio
async def test_reasoning_budget_widening_is_not_repeated() -> None:
    provider = _ReasoningBudgetProvider()
    router = ModelRouter(provider=provider, draft_model="reasoning-model", max_retries=1)

    async def always_exhausted(*args: object, **kwargs: object) -> tuple[str, str | None]:
        provider.budgets.append(kwargs.get("max_tokens"))  # type: ignore[arg-type]
        raise ModelProviderError(
            "reasoning budget exhausted",
            details={"reasoning_budget_exhausted": True},
        )

    provider.generate_with_finish_reason = always_exhausted  # type: ignore[method-assign]
    with pytest.raises(ModelProviderError):
        await router._execute_single_model("sys", "user", "reasoning-model", 0.3, max_tokens=1024)
    assert provider.budgets == [1024, _REASONING_MIN_OUTPUT_TOKENS]


@pytest.mark.asyncio
async def test_reasoning_budget_is_widened_when_the_request_left_it_unset() -> None:
    """A reasoning profile may omit max_tokens and inherit a gateway default."""
    provider = _ReasoningBudgetProvider()
    router = ModelRouter(provider=provider, draft_model="reasoning-model")

    text = await router._execute_single_model(
        "sys", "user", "reasoning-model", 0.3, max_tokens=None
    )

    assert text == "译文"
    assert provider.budgets == [None, _REASONING_MIN_OUTPUT_TOKENS]


@pytest.mark.asyncio
async def test_reasoning_draft_sets_a_scaled_budget_before_the_first_call() -> None:
    """A gateway default must not be the first budget a reasoning draft sees."""
    from ubt.core.ir.models import BlockStatus, IRBlock
    from ubt.model.ast import Paragraph
    from ubt.model.span import Span

    provider = _ReasoningBudgetProvider()
    router = ModelRouter(provider=provider, draft_model="reasoning-model")
    block = IRBlock(
        element=Paragraph(
            id="p1",
            spine_index=0,
            span=Span(page=1),
            text="word " * 10_000,
        ),
        status=BlockStatus.PENDING,
    )

    await router.draft(block=block, target_lang="zh", source_lang="en")

    assert provider.budgets[0] is not None
    assert provider.budgets[0] > _REASONING_MIN_OUTPUT_TOKENS
    # The prompt template itself is part of the source-token estimate, so the
    # exact value is above the bare 10k-token payload.
    assert provider.budgets[0] >= 10_000 // 4 * _REASONING_OUTPUT_HEADROOM


@pytest.mark.asyncio
async def test_reasoning_macro_chunk_budget_covers_all_blocks() -> None:
    """One eight-block macro call must budget for the whole chunk, not one block."""
    from ubt.core.ir.models import BlockStatus, IRBlock
    from ubt.model.ast import Paragraph
    from ubt.model.span import Span

    provider = _ReasoningBudgetProvider()
    router = ModelRouter(provider=provider, draft_model="reasoning-model")
    blocks = [
        IRBlock(
            element=Paragraph(
                id=f"p{i}",
                spine_index=i,
                span=Span(page=1),
                text="word " * 1_000,
            ),
            status=BlockStatus.PENDING,
        )
        for i in range(8)
    ]

    await router.draft_macro_chunk(blocks=blocks, target_lang="zh", source_lang="en")

    assert provider.budgets[0] is not None
    assert provider.budgets[0] > _REASONING_MIN_OUTPUT_TOKENS
    # Eight ~1000-token blocks must scale beyond the one-block floor.
    assert provider.budgets[0] > 20_000


@pytest.mark.asyncio
async def test_reasoning_repair_uses_a_scaled_budget_too() -> None:
    """Repair is the same reasoning model and must not fall back to 1024."""
    from ubt.core.ir.models import BlockStatus, IRBlock
    from ubt.model.ast import Paragraph
    from ubt.model.span import Span

    provider = _ReasoningBudgetProvider()
    router = ModelRouter(provider=provider, draft_model="reasoning-model")
    block = IRBlock(
        element=Paragraph(
            id="p1",
            spine_index=0,
            span=Span(page=1),
            text="word " * 10_000,
        ),
        status=BlockStatus.DRAFTED,
    )

    await router.repair(
        block=block,
        draft_text="译文",
        error_flags=["missing term"],
        target_lang="zh",
        source_lang="en",
    )

    assert provider.budgets[0] is not None
    assert provider.budgets[0] >= 10_000 // 4 * _REASONING_OUTPUT_HEADROOM


def test_anthropic_owns_its_thinking_budget_instead_of_double_counting() -> None:
    """Messages API raises max_tokens over budget_tokens; do not pre-inflate it."""
    router = ModelRouter(provider=MockModelProvider(), draft_model="reasoning-model")

    class _AnthropicProvider(MockModelProvider):
        _api_mode = "anthropic-messages"

    ordinary = router._reasoning_output_tokens(1_000, provider=_AnthropicProvider())
    chat = router._reasoning_output_tokens(1_000, provider=MockModelProvider())

    assert ordinary == 4_096
    assert chat == _REASONING_MIN_OUTPUT_TOKENS
