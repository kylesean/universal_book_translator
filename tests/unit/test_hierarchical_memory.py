"""HierarchicalMemoryManager: step/L3 epoch summarisation.

Originally part of a pre-split phase-2 architecture module, filed under the
review round that produced it. Nothing else in the suite exercised this class.
"""

from __future__ import annotations

import pytest

from ubt.core.ir.models import BlockType, IRBlock
from ubt.core.memory.hierarchical_memory import (
    EpochSnapshot,
    HierarchicalMemoryManager,
    StepSnapshot,
)


async def test_hierarchical_memory_step_trigger() -> None:
    """Verify that unsegmented documents trigger macro snapshot when step_chars threshold is met."""
    mgr = HierarchicalMemoryManager(step_chars=500)

    # 1. Initial state
    assert not mgr.should_trigger_snapshot()
    assert mgr.get_latest_macro_summary() == ""

    # 2. Add blocks below threshold
    b1 = IRBlock(
        id="block_001",
        spine_index=0,
        block_type=BlockType.NARRATIVE,
        source_text="First paragraph with around twenty words of text for testing.",
        target_text="第一段，包含大约二十个用于测试的词汇。",
    )
    mgr.record_drafted_block(b1)
    assert not mgr.should_trigger_snapshot()

    # 3. Add large blocks: b1+b2 stay under the 500-char threshold, b3 crosses it.
    long_target = "这是一个非常长的段落文本。" * 25  # 325 chars
    b2 = IRBlock(
        id="block_002",
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        source_text="Long paragraph...",
        target_text=long_target,
    )
    mgr.record_drafted_block(b2)
    b3 = IRBlock(
        id="block_003",
        spine_index=2,
        block_type=BlockType.NARRATIVE,
        source_text="Another long paragraph...",
        target_text=long_target,
    )
    mgr.record_drafted_block(b3)

    assert mgr.should_trigger_snapshot()

    # 4. Generate snapshot with mock complete function
    async def mock_complete(sys_prompt: str, user_prompt: str) -> str:
        assert "story-bible summarizer" in sys_prompt
        return "前文主要讨论了分布式系统的存储与状态同步方案。"

    summary = await mgr.generate_step_snapshot(mock_complete, target_lang="zh")
    assert summary == "前文主要讨论了分布式系统的存储与状态同步方案。"
    assert mgr.get_latest_macro_summary() == summary
    assert len(mgr.snapshots) == 1
    assert not mgr.should_trigger_snapshot()  # Buffer cleared


async def test_hierarchical_overlong_summary_is_truncated_not_discarded() -> None:
    """A >cap summary is truncated, not replaced by a short deterministic excerpt."""
    mgr = HierarchicalMemoryManager(step_chars=100)

    mgr.record_drafted_block(
        IRBlock(
            id="doc_p1",
            spine_index=0,
            block_type=BlockType.NARRATIVE,
            source_text="SOURCE-EXCERPT",
            target_text="SOURCE-EXCERPT",
        )
    )

    async def long_complete(sys_prompt: str, user_prompt: str) -> str:
        return "MODEL-SUMMARY " + ("情节推进。" * 200)

    summary = await mgr.generate_step_snapshot(long_complete, target_lang="zh")
    assert summary.startswith("MODEL-SUMMARY")
    assert len(summary) <= 800
    assert "SOURCE-EXCERPT" not in summary


async def test_hierarchical_memory_fallback_to_deterministic() -> None:
    """Verify graceful degradation to deterministic excerpt when LLM call fails."""
    mgr = HierarchicalMemoryManager(step_chars=100)

    b = IRBlock(
        id="doc_p1",
        spine_index=0,
        block_type=BlockType.NARRATIVE,
        source_text="The quick brown fox jumps over the lazy dog. Scientific study continues.",
        target_text="敏捷的棕色狐狸跃过了懒狗。科学研究仍在继续。",
    )
    mgr.record_drafted_block(b)

    async def failing_complete(sys_prompt: str, user_prompt: str) -> str:
        raise RuntimeError("API timeout")

    summary = await mgr.generate_step_snapshot(failing_complete, target_lang="zh")
    assert "敏捷的棕色狐狸" in summary
    assert mgr.get_latest_macro_summary() == summary


def test_hierarchical_memory_chapter_transition() -> None:
    """Verify that chapter transitions trigger a snapshot even before step_chars is reached."""
    mgr = HierarchicalMemoryManager(step_chars=5000)

    b1 = IRBlock(
        id="ch_001#b001",
        spine_index=0,
        block_type=BlockType.NARRATIVE,
        source_text="Chapter one content with substantial words to make it meaningful for continuity summary.",
        target_text="第一章内容，包含足够多的词汇以生成有意义的连贯性摘要。",
    )
    b2 = IRBlock(
        id="ch_001#b002",
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        source_text="More chapter one content.",
        target_text="更多第一章内容。",
    )
    mgr.record_drafted_block(b1)
    mgr.record_drafted_block(b2)

    next_block_ch2 = IRBlock(
        id="ch_002#b001",
        spine_index=2,
        block_type=BlockType.NARRATIVE,
        source_text="Chapter two begins here.",
    )

    assert mgr.should_trigger_snapshot(next_block=next_block_ch2)


async def test_l3_epoch_compression_every_n_steps() -> None:
    """Every N closed L2 steps compress into one stable L3 entry."""
    mgr = HierarchicalMemoryManager(step_chars=100, l3_epoch_steps=3)

    async def fake_complete(sys_prompt: str, user_prompt: str) -> str:
        return "这是一个足够长的宏观摘要内容，用于触发快照生成的最小长度要求。"

    for step in range(4):
        mgr.record_drafted_block(
            IRBlock(
                id=f"doc_p{step}",
                spine_index=step,
                block_type=BlockType.NARRATIVE,
                source_text=f"Content chunk number {step} with plenty of text for the buffer.",
                target_text="第" * 60,
            )
        )
        await mgr.generate_step_snapshot(fake_complete, target_lang="zh")

    assert len(mgr.snapshots) == 4
    # After 3 L2 steps the first epoch closed; the 4th step has not yet.
    assert len(mgr.epochs) == 1
    l3 = mgr.get_l3_summary()
    assert l3
    assert mgr.format_l3_prompt_block(l3).startswith("### Book Continuity")
    # Epoch spans the first three L2 snapshots' spines.
    assert mgr.epochs[0].start_spine == 0
    assert mgr.epochs[0].end_spine == 2


async def test_l3_epoch_uses_the_full_l3_char_budget() -> None:
    """The epoch text is capped at the L3 budget (1200), not the 400 default.

    ``deterministic_summary`` defaults to 400 chars, so slicing its result to
    ``_MAX_L3_CHARS`` (1200) was dead code: every L3 epoch rode the static
    prompt prefix at a third of the reserved budget.
    """
    mgr = HierarchicalMemoryManager(step_chars=100, l3_epoch_steps=2)

    async def long_complete(sys_prompt: str, user_prompt: str) -> str:
        return "情节推进。" * 120  # 600 chars, under the 800-char per-step cap

    for step in range(2):
        mgr.record_drafted_block(
            IRBlock(
                id=f"doc_p{step}",
                spine_index=step,
                block_type=BlockType.NARRATIVE,
                source_text=f"Chunk {step}.",
                target_text="内" * 60,
            )
        )
        await mgr.generate_step_snapshot(long_complete, target_lang="zh")

    assert len(mgr.epochs) == 1
    l3_len = len(mgr.epochs[0].summary_text)
    assert 400 < l3_len <= 1200, l3_len


def test_l3_summary_empty_before_first_epoch() -> None:
    """No L3 text until the first epoch of steps closes."""
    mgr = HierarchicalMemoryManager(step_chars=100, l3_epoch_steps=4)
    assert mgr.get_l3_summary() == ""
    assert mgr.format_l3_prompt_block("") == ""


async def test_l3_epoch_stable_across_later_steps() -> None:
    """The L3 text of a closed epoch never mutates afterwards."""
    mgr = HierarchicalMemoryManager(step_chars=100, l3_epoch_steps=2)

    async def fake_complete(sys_prompt: str, user_prompt: str) -> str:
        return "这一段宏观摘要满足最小长度阈值，便于断言其内容保持稳定不变。"

    closed_epoch_text: str | None = None

    for step in range(5):
        mgr.record_drafted_block(
            IRBlock(
                id=f"doc_p{step}",
                spine_index=step,
                block_type=BlockType.NARRATIVE,
                source_text=f"Chunk {step}: enough prose to cross the step threshold.",
                target_text="内" * 60,
            )
        )
        await mgr.generate_step_snapshot(fake_complete, target_lang="zh")
        if closed_epoch_text is None and len(mgr.epochs) == 2:
            closed_epoch_text = mgr.epochs[0].summary_text

    assert closed_epoch_text is not None, "step 4 should have closed epoch 0"
    assert len(mgr.epochs) == 2
    # Captured while epoch 1 was still open: later steps must not rewrite it.
    assert mgr.epochs[0].summary_text == closed_epoch_text
    # get_l3_summary is cumulative across closed epochs (2026-09-24 HIGH-T3-3:
    # closing a new epoch must not discard the earlier ones), so the closed
    # epoch's text has to survive in it rather than be replaced by the latest
    # epoch alone.
    cumulative = mgr.get_l3_summary()
    assert closed_epoch_text in cumulative
    assert mgr.epochs[-1].summary_text in cumulative


@pytest.mark.fast
def test_hierarchical_memory_macro_context_spine_boundary() -> None:
    """get_macro_context_for_block enforces snap.end_spine <= block.spine_index."""
    mgr = HierarchicalMemoryManager()

    # Pre-populate two snapshots
    snap1 = StepSnapshot(
        step_index=0,
        start_spine=0,
        end_spine=5,
        char_count=100,
        summary_text="Chapter 1 part 1 summary",
        chapter_id="ch01",
    )
    snap2 = StepSnapshot(
        step_index=1,
        start_spine=6,
        end_spine=10,
        char_count=100,
        summary_text="Chapter 1 part 2 summary",
        chapter_id="ch01",
    )
    mgr._snapshots.extend([snap1, snap2])

    # Block at spine_index 3 (in chapter 1) should NOT see snap2 (end_spine=10)
    # It shouldn't even see snap1 if snap1.end_spine > 3
    early_block = IRBlock(
        id="ch01#b002",
        spine_index=3,
        block_type=BlockType.NARRATIVE,
        source_text="Early paragraph",
    )
    ctx_early = mgr.get_macro_context_for_block(early_block)
    assert ctx_early == "", f"Early block leaked future snapshot: {ctx_early}"

    # Block at spine_index 7 should see snap1 (end_spine=5 <= 7), but NOT snap2 (end_spine=10 > 7)
    mid_block = IRBlock(
        id="ch01#b007",
        spine_index=7,
        block_type=BlockType.NARRATIVE,
        source_text="Mid paragraph",
    )
    ctx_mid = mgr.get_macro_context_for_block(mid_block)
    assert ctx_mid == "Chapter 1 part 1 summary"


@pytest.mark.fast
def test_hierarchical_memory_get_l3_summary_retains_multi_epoch_history() -> None:
    """[HIGH-T3-3] get_l3_summary must include earlier epochs (clamped to _MAX_L3_CHARS),
    not overwrite/discard earlier epochs when a new epoch closes."""
    mem = HierarchicalMemoryManager()
    mem._epochs = [
        EpochSnapshot(
            epoch_index=0,
            start_spine=0,
            end_spine=9,
            summary_text="Epoch 0: Alice enters the rabbit hole.",
        ),
        EpochSnapshot(
            epoch_index=1, start_spine=10, end_spine=19, summary_text="Epoch 1: The Mad Tea-Party."
        ),
    ]
    l3 = mem.get_l3_summary()
    assert "Epoch 0: Alice enters the rabbit hole." in l3
    assert "Epoch 1: The Mad Tea-Party." in l3
