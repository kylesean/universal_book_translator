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
    from ubt.core.ir.models import IRBlock
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
        status="pending",
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
    from ubt.core.ir.models import IRBlock
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
            status="pending",
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
    from ubt.core.ir.models import IRBlock
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
        status="drafted",
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


@pytest.mark.asyncio
async def test_reasoning_batch_requests_carry_an_explicit_budget() -> None:
    """The Batch path has no continuation loop, so it cannot inherit 1024."""
    from ubt.core.router.registry import ModelCapabilityRegistry
    from ubt.core.router.router_errors import BatchDraftRequest

    captured: list[list[dict[str, object]]] = []

    class _BatchProvider(MockModelProvider):
        @property
        def supports_batch_api(self) -> bool:
            return True

        async def create_batch_job(self, requests: list[dict[str, object]]) -> str:
            captured.extend(requests)
            return "batch-1"

        async def get_batch_job(self, batch_id: str) -> dict[str, object]:
            return {"status": "completed", "output_file_id": "file-1"}

        async def fetch_batch_results(self, batch_id: str) -> dict[str, dict[str, object]]:
            return {}

        async def cleanup_batch_files(self, batch_id: str) -> None:
            return None

    provider = _BatchProvider()
    router = ModelRouter(
        provider=provider,
        draft_model="reasoning-model",
        registry=ModelCapabilityRegistry(),
    )
    # Declare reasoning on the exact model so the request body takes that path.
    router.registry = ModelCapabilityRegistry(
        [
            router._get_profile("reasoning-model").model_copy(
                update={"model_pattern": "reasoning-model", "supports_reasoning_effort": True}
            )
        ]
    )

    await router.draft_batch([BatchDraftRequest(custom_id="c1", source_text="word " * 10_000)])

    body = captured[0]["body"]  # type: ignore[index]
    assert isinstance(body, dict)
    assert body["max_tokens"] >= _REASONING_MIN_OUTPUT_TOKENS
