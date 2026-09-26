"""Unit tests for the MT suitability gate and draft model override."""

import asyncio
from typing import Any

import pytest

from ubt.core.ir.models import BlockType, FlowID, IRBlock
from ubt.core.policy.layout_policy import NON_TEXT_BLOCK_TYPES
from ubt.core.policy.verdict import judge_block
from ubt.core.qe.defect_taxonomy import STRUCTURAL_DEFECT_MARKERS
from ubt.core.qe.fast_pass import FastPassFilter
from ubt.core.qe.mt_gate import count_sentences, is_mt_suitable
from ubt.core.router.provider import MockModelProvider
from ubt.core.router.router import ModelRouter

# ---------------------------------------------------------------------------
# Gate predicate matrix (precision-first: every condition is a hard AND)
# ---------------------------------------------------------------------------


def test_gate_admits_single_short_narrative() -> None:
    assert is_mt_suitable(
        "The lighthouse keeper climbed the tower.",
        BlockType.NARRATIVE,
        has_terms=False,
        has_few_shot=False,
        has_masked_spans=False,
    )


def test_gate_rejects_structured_block_types() -> None:
    for block_type in (
        BlockType.CODE,
        BlockType.FORMULA,
        BlockType.IMAGE,
        BlockType.TABLE,
    ):
        assert not is_mt_suitable(
            "Simple text.",
            block_type,
            has_terms=False,
            has_few_shot=False,
            has_masked_spans=False,
        ), block_type


def test_gate_rejects_overlong_text() -> None:
    long_text = "A single sentence that keeps going far beyond the admission window. " * 5
    assert not is_mt_suitable(
        long_text,
        BlockType.NARRATIVE,
        has_terms=False,
        has_few_shot=False,
        has_masked_spans=False,
    )


def test_gate_rejects_multi_sentence_text() -> None:
    assert not is_mt_suitable(
        "First sentence ends here. Second sentence follows it.",
        BlockType.NARRATIVE,
        has_terms=False,
        has_few_shot=False,
        has_masked_spans=False,
    )


def test_gate_sentence_counter_ignores_decimal_points() -> None:
    assert count_sentences("Pi is roughly 3.14 in normal usage.") == 1
    assert count_sentences("Hello world.") == 1
    # The shared counter counts non-empty fragments (omission gate semantics):
    # a terminator-less heading is one fragment, not zero. The gate verdict is
    # unchanged either way — it only asks ``count_sentences(...) > 1``.
    assert count_sentences("Chapter Overview") == 1
    assert count_sentences("Chinese ends with 。fullwidth marks！too？") == 3


def test_count_sentences_is_the_single_shared_implementation() -> None:
    """mt_gate and omission must share ONE counter (the abbreviation-masking one).

    mt_gate used to keep a second, abbreviation-blind copy that counted
    'Eq.'/'Fig.' periods as sentence breaks and therefore disagreed with the
    omission gate about the very same text.
    """
    from ubt.core.qe import mt_gate, omission
    from ubt.core.qe.term_shape import count_sentences as shared_count

    assert mt_gate.count_sentences is shared_count
    assert omission.count_sentences is shared_count

    samples = [
        "Hello world.",
        "Chapter Overview",
        "Pi is roughly 3.14 in normal usage.",
        "Chinese ends with 。fullwidth marks！too？",
        "Eq. (3.11) is implicit here. Fig. 3.5 shows the result.",
        "First sentence ends here. Second sentence follows it.",
    ]
    for sample in samples:
        assert mt_gate.count_sentences(sample) == omission.count_sentences(sample), sample
    # The disagreement the two copies used to have, now gone:
    formula_dense = "Eq. (3.11) is implicit here. Fig. 3.5 shows the result."
    assert shared_count(formula_dense) == 2


def test_gate_rejects_terms_few_shot_and_masked_spans() -> None:
    text = "One clean sentence."
    kwargs: dict[str, Any] = {"has_terms": False, "has_few_shot": False, "has_masked_spans": False}
    assert is_mt_suitable(text, BlockType.NARRATIVE, **kwargs)

    assert not is_mt_suitable(text, BlockType.NARRATIVE, **{**kwargs, "has_terms": True})
    assert not is_mt_suitable(text, BlockType.NARRATIVE, **{**kwargs, "has_few_shot": True})
    assert not is_mt_suitable(text, BlockType.NARRATIVE, **{**kwargs, "has_masked_spans": True})


def test_gate_rejects_empty_text() -> None:
    assert not is_mt_suitable(
        "",
        BlockType.NARRATIVE,
        has_terms=False,
        has_few_shot=False,
        has_masked_spans=False,
    )


# ---------------------------------------------------------------------------
# Router draft model override
# ---------------------------------------------------------------------------


def test_router_draft_model_override_uses_minimal_profile() -> None:
    provider = MockModelProvider(default_response="[TRANSLATED]")
    router = ModelRouter(provider=provider, draft_model="gpt-4")
    from ubt.core.router.capabilities import (
        ExtractionStrategy,
        ModelProfile,
        PromptStrategy,
    )

    router.registry.register(
        ModelProfile(
            model_pattern="translategemma-test",
            prompt_strategy=PromptStrategy.MINIMAL,
            extraction_strategy=ExtractionStrategy.RAW,
            supports_system_prompt=False,
        )
    )
    block = IRBlock(
        id="b1",
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        flow_id=FlowID.MAIN_STORY,
        source_text="A short sentence.",
    )

    result = asyncio.run(router.draft(block, model="translategemma-test"))

    assert result == "[TRANSLATED]"
    assert len(provider.call_history) == 1
    call = provider.call_history[0]
    assert call["model"] == "translategemma-test"
    # MINIMAL profile: no system turn, direct translation instruction only.
    assert call["system_prompt"] is None
    assert "Output ONLY the translation" in call["prompt"]
    # RICH gpt-4 machinery (XML schema, genre, static bible tiers) absent.
    assert "<translation>" not in call["prompt"]


def test_router_draft_without_override_keeps_default_tier() -> None:
    provider = MockModelProvider(default_response="<translation>你好</translation>")
    router = ModelRouter(provider=provider, draft_model="gpt-4")
    block = IRBlock(
        id="b1",
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        flow_id=FlowID.MAIN_STORY,
        source_text="A short sentence.",
    )

    result = asyncio.run(router.draft(block))

    assert result == "你好"
    assert provider.call_history[0]["model"] == "gpt-4"


@pytest.mark.asyncio
async def test_draft_prompt_no_token_leak() -> None:
    """Override path must not leak the default tier's model name into prompts."""
    provider = MockModelProvider(default_response="[TRANSLATED]")
    router = ModelRouter(provider=provider, draft_model="gpt-4")
    block = IRBlock(
        id="b1",
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        flow_id=FlowID.MAIN_STORY,
        source_text="A short sentence.",
    )
    await router.draft(block, model="translategemma-test")
    assert "gpt-4" not in provider.call_history[0]["prompt"]


_r0918_TABLE_OK = "| 单元格 | 阿尔法 | Beta 值 |\n| --- | --- | --- |\n| 12 | 34 | 56 |"

_r0918_TABLE_SRC = "| Cell | Alpha | Beta Value |\n| --- | --- | --- |\n| 12 | 34 | 56 |"


def test_untranslated_table_no_longer_passes_the_script_density_gate() -> None:
    assert (
        not FastPassFilter(source_lang="en", target_lang="zh")
        .evaluate(_r0918_TABLE_SRC, _r0918_TABLE_SRC)
        .passed
    )


def test_translated_table_with_identifier_carryovers_still_passes() -> None:
    assert (
        FastPassFilter(source_lang="en", target_lang="zh")
        .evaluate(_r0918_TABLE_SRC, _r0918_TABLE_OK)
        .passed
    )


_r0918b_GRID_REASON = "Table grid mismatch"


def _r0918b_fp(source_lang: str, target_lang: str) -> FastPassFilter:
    return FastPassFilter(source_lang=source_lang, target_lang=target_lang)


def _r0918b_irblock(**kwargs: object) -> IRBlock:
    base: dict[str, object] = {
        "id": "t1",
        "spine_index": 0,
        "source_text": "| Cell | Alpha | Beta |\n| --- | --- | --- |\n| 1 | 2 | 3 |",
    }
    base.update(kwargs)
    return IRBlock(**base)  # type: ignore[arg-type]


def test_tables_are_translation_content_not_verbatim_residue() -> None:
    """The parser marks tables ``skip=False`` and the QE layer requires a
    translated table, but the verdict kept them out of the model entirely while
    ingest stamped them MTQE_PASSED/1.0."""
    assert BlockType.TABLE not in NON_TEXT_BLOCK_TYPES
    verdict = judge_block(_r0918b_irblock(block_type=BlockType.TABLE))
    assert verdict.translate is True
    for keep in (BlockType.FORMULA, BlockType.CODE, BlockType.IMAGE):
        assert judge_block(_r0918b_irblock(block_type=keep)).translate is False


def test_translated_table_grid_passes_and_a_mangled_one_does_not() -> None:
    src = "| Cell | Alpha | Beta Value |\n| --- | --- | --- |\n| 12 | 34 | 56 |"
    translated = "| 单元格 | 阿尔法 | Beta 值 |\n| --- | --- | --- |\n| 12 | 34 | 56 |"
    dropped_column = "| 单元格 | 阿尔法 |\n| --- | --- |\n| 12 | 34 |"
    fp = _r0918b_fp("en", "zh")
    assert fp.evaluate(src, translated, block_type=BlockType.TABLE).passed
    decision = fp.evaluate(src, dropped_column, block_type=BlockType.TABLE)
    assert not decision.passed and _r0918b_GRID_REASON in decision.reason
    assert any(m in decision.reason for m in STRUCTURAL_DEFECT_MARKERS)
