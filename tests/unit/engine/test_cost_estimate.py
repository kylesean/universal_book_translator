"""Unit tests for pre-flight draft cost estimation."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ubt.core.engine.cost_estimate import (
    count_text_tokens,
    estimate_draft_cost,
    estimate_draft_cost_from_totals,
    measure_prefix_tokens,
    resolve_output_token_ratio,
)
from ubt.core.ir.models import BlockStatus, BlockType, BoundingBox, IRBlock, make_element

pytestmark = pytest.mark.fast


def test_count_text_tokens() -> None:
    assert count_text_tokens("") == 0
    # Pure ASCII: ~4 chars per token
    assert count_text_tokens("abcd") == 1
    assert count_text_tokens("abcdefgh") == 2

    # CJK: ~0.85 tokens per char
    cjk_text = " universal book translator 深度学习测试"
    cjk_tokens = count_text_tokens(cjk_text)
    assert cjk_tokens > 0


def test_resolve_output_token_ratio() -> None:
    # None fallback
    assert resolve_output_token_ratio(None, None) == 1.5

    # Same language
    assert resolve_output_token_ratio("en", "en") == 1.0

    # Non-CJK to CJK
    assert resolve_output_token_ratio("en", "zh") == 1.25

    # CJK to CJK
    assert resolve_output_token_ratio("zh", "ja") == 1.05

    # High expansion targets
    assert resolve_output_token_ratio("en", "de") == 1.40
    assert resolve_output_token_ratio("zh", "ru") == 1.05


def test_estimate_draft_cost_from_totals_known_model() -> None:
    quote = estimate_draft_cost_from_totals(
        billable_blocks=100,
        source_chars=4000,
        draft_model="gpt-4o",
        prefix_tokens=500,
        source_lang="en",
        target_lang="zh",
    )
    assert quote.billable_blocks == 100
    assert not quote.money_is_unknown
    assert quote.cost_usd_uncached is not None
    assert quote.cost_usd_cached is not None
    assert quote.cost_usd_cached <= quote.cost_usd_uncached
    desc = quote.describe()
    assert "Pre-flight estimate" in desc
    assert "gpt-4o" in desc


def test_estimate_draft_cost_from_totals_unpriced_model() -> None:
    quote = estimate_draft_cost_from_totals(
        billable_blocks=10,
        source_chars=1000,
        draft_model="non-existent-unpriced-model-xyz",
        prefix_tokens=200,
        base_url="https://api.openai.com/v1",
    )
    assert quote.money_is_unknown
    assert quote.cost_usd_uncached is None
    assert "unknown" in quote.describe()


def test_estimate_draft_cost_from_ir_blocks() -> None:
    b1 = IRBlock(
        element=make_element(
            id="b1",
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            source_text="First block to translate.",
            bbox=BoundingBox(x0=0, y0=0, x1=10, y1=10, page=1),
        ),
        target_text="",
        status=BlockStatus.PENDING,
    )
    b_skip = IRBlock(
        element=make_element(
            id="b2",
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            source_text="Skip this block.",
            bbox=BoundingBox(x0=0, y0=0, x1=10, y1=10, page=1),
            skip_translate=True,
        ),
        target_text="",
    )
    b_done = IRBlock(
        element=make_element(
            id="b3",
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            source_text="Already translated.",
            bbox=BoundingBox(x0=0, y0=0, x1=10, y1=10, page=1),
        ),
        target_text="已翻译。",
        status=BlockStatus.MTQE_PASSED,
    )

    quote = estimate_draft_cost(
        [b1, b_skip, b_done],
        draft_model="gpt-4o",
        prefix_tokens=100,
    )
    # Only b1 is billable
    assert quote.billable_blocks == 1


def test_measure_prefix_tokens() -> None:
    router = MagicMock()
    router.build_draft_prompt.return_value = (
        "System prompt instructions here.",
        "User scaffolding here.",
    )

    tokens = measure_prefix_tokens(router, target_lang="zh", source_lang="en")
    assert tokens is not None
    assert tokens > 0

    # Failed router prompt build
    router.build_draft_prompt.side_effect = ValueError("prompt build error")
    assert measure_prefix_tokens(router, target_lang="zh", source_lang="en") is None
