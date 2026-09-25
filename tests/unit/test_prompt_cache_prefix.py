"""Prompt prefix-cache topology tests.

Guarantees:
- the static book-level prefix (system prompt, global glossary) is
  byte-identical across blocks of the same book;
- per-chunk glossaries are never silently dropped;
- TM few-shot references stay in the dynamic tail, immediately
  before the source paragraph, never in the static prefix.
"""

import pytest

from ubt.core.ir.models import FlowID, IRBlock
from ubt.core.router.capabilities import (
    ExtractionStrategy,
    ModelProfile,
    PromptStrategy,
)
from ubt.core.router.provider import BaseModelProvider, MockModelProvider
from ubt.core.router.registry import ModelCapabilityRegistry
from ubt.core.router.router import ModelRouter

GLOBAL_GLOSSARY = "| 原文 | 别名 | 译文 |\n| :--- | :--- | :--- |\n| Darcy | Mr. Darcy | 达西 |"
CHUNK_A = "| 原文 | 别名 | 译文 |\n| :--- | :--- | :--- |\n| Pemberley | | 彭伯里 |"
CHUNK_B = "| 原文 | 别名 | 译文 |\n| :--- | :--- | :--- |\n| Longbourn | | 朗博恩 |"


def _router_for(strategy: PromptStrategy) -> ModelRouter:
    registry = ModelCapabilityRegistry()
    registry.register(
        ModelProfile(
            model_pattern="cache-test-model",
            prompt_strategy=strategy,
            extraction_strategy=ExtractionStrategy.AUTO,
        )
    )
    return ModelRouter(
        provider=MockModelProvider(),
        draft_model="cache-test-model",
        registry=registry,
    )


def _block(text: str) -> IRBlock:
    return IRBlock(
        id="ch01#b001",
        flow_id=FlowID.MAIN_STORY,
        spine_index=1,
        source_text=text,
    )


@pytest.mark.parametrize(
    "strategy",
    [PromptStrategy.RICH, PromptStrategy.HYBRID, PromptStrategy.MINIMAL],
)
def test_global_and_chunk_glossaries_are_merged(strategy: PromptStrategy) -> None:
    """Regression: global and chunk glossaries must coexist in every strategy."""
    router = _router_for(strategy)
    system_prompt, user_prompt = router.build_draft_prompt(
        source_text="Elizabeth was reading.",
        glossary_table=CHUNK_A,
        target_lang="zh",
        source_lang="en",
        global_glossary=GLOBAL_GLOSSARY,
    )
    combined = f"{system_prompt}\n{user_prompt}"
    assert "Darcy" in combined  # global term preserved
    assert "Pemberley" in combined  # chunk term no longer silently dropped


@pytest.mark.parametrize("strategy", [PromptStrategy.RICH, PromptStrategy.HYBRID])
def test_static_prefix_is_byte_stable_across_blocks(strategy: PromptStrategy) -> None:
    """Same book, different blocks: the static prefix must be identical."""
    router = _router_for(strategy)
    args: dict[str, str] = {
        "target_lang": "zh",
        "source_lang": "en",
        "global_glossary": GLOBAL_GLOSSARY,
    }
    sys_a, _ = router.build_draft_prompt(source_text="Text A.", glossary_table=CHUNK_A, **args)
    sys_b, _ = router.build_draft_prompt(source_text="Text B.", glossary_table=CHUNK_B, **args)
    assert sys_a == sys_b


def test_rich_global_glossary_lives_in_static_system_tier() -> None:
    router = _router_for(PromptStrategy.RICH)
    system_prompt, user_prompt = router.build_draft_prompt(
        source_text="Text.",
        glossary_table=CHUNK_A,
        target_lang="zh",
        source_lang="en",
        global_glossary=GLOBAL_GLOSSARY,
    )
    assert "Darcy" in system_prompt
    assert "Darcy" not in user_prompt
    assert "Pemberley" in user_prompt


def test_hybrid_global_glossary_precedes_rolling_summary() -> None:
    """Book-static content must precede chapter-level context for prefix caching."""
    router = _router_for(PromptStrategy.HYBRID)
    _, user_prompt = router.build_draft_prompt(
        source_text="Text.",
        glossary_table=CHUNK_A,
        target_lang="zh",
        source_lang="en",
        rolling_summary="PREVIOUS-CHAPTER-SUMMARY",
        global_glossary=GLOBAL_GLOSSARY,
    )
    darcy = user_prompt.find("Darcy")
    summary = user_prompt.find("PREVIOUS-CHAPTER-SUMMARY")
    pemberley = user_prompt.find("Pemberley")
    assert 0 <= darcy < summary < pemberley


def test_few_shot_reference_stays_in_dynamic_tail() -> None:
    """TM fuzzy few-shot sits after neighbors, immediately before the source."""
    router = _router_for(PromptStrategy.RICH)
    few_shot = (
        "### Reference Translation (fuzzy match from translation memory, 91% similar\n"
        "— reuse its terminology and phrasing where applicable)\n"
        "Source: She was reading.\nTranslation: 她在读书。"
    )
    _, user_prompt = router.build_draft_prompt(
        source_text="Elizabeth was reading.",
        neighbor_context="CONTEXT-EXCERPT",
        target_lang="zh",
        source_lang="en",
        few_shot_reference=few_shot,
    )
    assert few_shot in user_prompt
    assert (
        user_prompt.find("CONTEXT-EXCERPT")
        < user_prompt.find(few_shot)
        < user_prompt.find("Elizabeth was reading.")
    )


def test_minimal_strategy_keeps_single_message_contract() -> None:
    router = _router_for(PromptStrategy.MINIMAL)
    system_prompt, user_prompt = router.build_draft_prompt(
        source_text="Text.",
        glossary_table=CHUNK_A,
        target_lang="zh",
        source_lang="en",
        global_glossary=GLOBAL_GLOSSARY,
    )
    assert system_prompt == ""
    assert "Darcy" in user_prompt
    assert "Pemberley" in user_prompt


def test_draft_method_passes_few_shot_reference_to_prompt() -> None:
    """The draft() entry point threads few_shot_reference into prompt building."""
    registry = ModelCapabilityRegistry()
    registry.register(
        ModelProfile(
            model_pattern="cache-capture-model",
            prompt_strategy=PromptStrategy.RICH,
            extraction_strategy=ExtractionStrategy.AUTO,
        )
    )
    provider = MockModelProvider()
    router = ModelRouter(provider=provider, draft_model="cache-capture-model", registry=registry)
    import asyncio

    asyncio.run(
        router.draft(
            block=_block("Elizabeth was reading."),
            glossary_table=CHUNK_A,
            target_lang="zh",
            source_lang="en",
            few_shot_reference="FEW-SHOT-BLOCK",
        )
    )
    assert len(provider.call_history) == 1
    assert "FEW-SHOT-BLOCK" in provider.call_history[0]["prompt"]


class DummyProvider(BaseModelProvider):
    """Minimal dummy provider for testing."""

    @property
    def provider_name(self) -> str:
        return "dummy"

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = 0.0,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        return "<final_translation>Dummy output</final_translation>"


def test_context_caching_prefix_invariance() -> None:
    """Verify Tier 1 System Prompt is 100% byte-for-byte identical across varying paragraphs for prefix cache hits."""
    router = ModelRouter(provider=DummyProvider())

    global_glossary = (
        "| Source | Target |\n|---|---|\n| Transformer | 变换器架构 |\n| Attention | 注意力机制 |"
    )

    # Chunk 1
    sys1, user1 = router.build_draft_prompt(
        source_text="This is paragraph one.",
        neighbor_context="[READ-ONLY PRECEDING CONTEXT]\n```text\nPrev text\n```",
        rolling_summary="Macro summary of chapter 1.",
        global_glossary=global_glossary,
    )

    # Chunk 2 (completely different source and neighbor context, but same book glossary and macro summary)
    sys2, user2 = router.build_draft_prompt(
        source_text="This is an entirely different paragraph with new ideas.",
        neighbor_context="[READ-ONLY PRECEDING CONTEXT]\n```text\nCompletely different prev\n```",
        rolling_summary="Macro summary of chapter 1.",
        global_glossary=global_glossary,
    )

    # 1. Tier 1 (System Prompt) MUST BE 100% identical byte-for-byte -> 100% Prefix Cache hit!
    assert sys1 == sys2
    assert "### Translation Bible" in sys1
    assert "Transformer" in sys1

    # 2. Tier 2 in User Prompt (Macro snapshot prefix) must be identical
    assert user1.startswith(
        "### Document Continuation Context (Macro Snapshot)\nMacro summary of chapter 1."
    )
    assert user2.startswith(
        "### Document Continuation Context (Macro Snapshot)\nMacro summary of chapter 1."
    )

    # 3. Tier 3 is dynamically varying tail
    assert "This is paragraph one." in user1
    assert "This is an entirely different paragraph" in user2
