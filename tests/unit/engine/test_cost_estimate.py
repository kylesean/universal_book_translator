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


# --- the script-aware heuristic's per-script constants ----------------------
#
# ``count_text_tokens`` is a character heuristic, not a tokenizer (it must run
# without loading a tokenizer). Its constants therefore need pinning to
# measurements, or they drift: the non-ASCII constant sat at 0.55 while the
# XLM-RoBERTa tokenizer measured ~0.31, over-quoting Cyrillic and Arabic by
# 50-67%. These bounds are loose enough to survive a tokenizer refresh and
# tight enough to catch a constant that is simply wrong.

#: (text, expected tokens under XLM-RoBERTa, allowed relative error)
_SCRIPT_SAMPLES: tuple[tuple[str, str, int], ...] = (
    (
        "en",
        "The transistor was fabricated using a standard CMOS process with a "
        "gate oxide thickness of two nanometers.",
        29,
    ),
    ("zh", "该晶体管采用标准CMOS工艺制备，栅氧化层厚度为两纳米，沟道长度为五十纳米。", 34),
    (
        "ja",
        "このトランジスタは標準的なCMOSプロセスで作製され、ゲート酸化膜の厚さは2ナノメートルである。",
        33,
    ),
    (
        "ko",
        "이 트랜지스터는 표준 CMOS 공정으로 제조되었으며 게이트 산화막 두께는 2나노미터이다.",
        30,
    ),
    (
        "ru",
        "Этот транзистор был изготовлен по стандартному КМОП-процессу с толщиной "
        "подзатворного окисла два нанометра.",
        33,
    ),
    (
        "ar",
        "تم تصنيع هذا الترانزستور باستخدام عملية CMOS القياسية بسماكة أكسيد البوابة نانومترين.",
        28,
    ),
    (
        "de",
        "Der Transistor wurde in einem Standard-CMOS-Prozess mit einer "
        "Gateoxiddicke von zwei Nanometern hergestellt.",
        30,
    ),
)


@pytest.mark.parametrize(("lang", "text", "xlmr_tokens"), _SCRIPT_SAMPLES)
def test_the_script_heuristic_stays_within_25_percent_of_a_real_tokenizer(
    lang: str, text: str, xlmr_tokens: int
) -> None:
    """Every script must land within 25% of XLM-RoBERTa's own count.

    The bound is 25% because no character heuristic can do better across model
    families — the same English sentence is 29 tokens to XLM-R and 19 to
    DeepSeek — but a constant that is off by 50%+ is a bug, not a tolerance.
    """
    estimated = count_text_tokens(text)
    error = (estimated - xlmr_tokens) / xlmr_tokens
    assert abs(error) <= 0.25, (
        f"{lang}: heuristic {estimated} vs XLM-R {xlmr_tokens} ({error:+.0%}); "
        "the per-script constant needs re-measuring"
    )


def test_non_ascii_scripts_are_not_over_quoted_by_half() -> None:
    """The specific regression: 0.55 tokens/char for Cyrillic/Arabic.

    A ru/ar source quoted at 0.55 per character came out 50-67% above the real
    count, which inflated every budget check and could refuse an affordable
    run. Pinned as a ratio rather than a constant so the fix can be tuned
    without rewriting the test.
    """
    from ubt.core.engine import cost_estimate

    assert 0.25 <= cost_estimate._NON_ASCII_TOKENS_PER_CHAR <= 0.40, (
        "non-ASCII tokens/char drifted away from the measured ~0.31"
    )
    # CJK was already correct and must not have moved with it.
    assert pytest.approx(0.85) == cost_estimate._CJK_TOKENS_PER_CHAR


def test_the_heuristic_documents_that_it_is_not_a_tokenizer() -> None:
    """A caller that reads this as exact will quote an invoice it cannot meet."""
    doc = count_text_tokens.__doc__ or ""
    assert "not a tokenizer" in doc.lower() or "heuristic" in doc.lower()
    assert "estimate" in doc.lower()
