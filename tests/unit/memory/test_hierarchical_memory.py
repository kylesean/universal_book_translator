from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from ubt.core.ir.models import BlockType, IRBlock, make_element
from ubt.core.memory.hierarchical_memory import (
    DEFAULT_L3_EPOCH_STEPS,
    DEFAULT_STEP_CHARS,
    EpochSnapshot,
    HierarchicalMemoryManager,
    StepSnapshot,
    refresh_chapter_memory,
)

pytestmark = pytest.mark.fast


def _block(
    block_id: str,
    spine: int,
    *,
    target: str | None = None,
    draft: str | None = None,
    source: str = "source text",
) -> IRBlock:
    element = make_element(
        id=block_id,
        spine_index=spine,
        block_type=BlockType.NARRATIVE,
        source_text=source,
        skip_translate=False,
    )
    return IRBlock(element=element, target_text=target, draft_text=draft)


async def _fixed_complete(response: str) -> Callable[[str, str], Awaitable[str]]:
    async def complete(system: str, user: str) -> str:
        return response

    return complete


async def _snapshot(
    manager: HierarchicalMemoryManager, blocks: list[IRBlock], response: str = ""
) -> str:
    for b in blocks:
        manager.record_drafted_block(b)
    complete = await _fixed_complete(response)
    return await manager.generate_step_snapshot(complete)


def test_defaults() -> None:
    manager = HierarchicalMemoryManager()
    assert manager.step_chars == DEFAULT_STEP_CHARS == 3500
    assert manager.l3_epoch_steps == DEFAULT_L3_EPOCH_STEPS == 4
    assert manager.enabled is True
    assert manager.snapshots == []
    assert manager.epochs == []
    assert manager.get_latest_macro_summary() == ""
    assert manager.get_l3_summary() == ""


def test_disabled_manager_buffers_nothing_and_triggers_no_snapshot() -> None:
    manager = HierarchicalMemoryManager(enabled=False, step_chars=1)
    manager.record_drafted_block(_block("c1#b1", 0, target="x" * 50))
    assert manager.should_trigger_snapshot() is False
    assert manager.get_l3_summary() == ""
    assert manager.snapshots == []


async def test_disabled_manager_generate_step_snapshot_is_a_no_op() -> None:
    manager = HierarchicalMemoryManager(enabled=False, step_chars=1)
    manager.record_drafted_block(_block("c1#b1", 0, target="x" * 50))
    complete = await _fixed_complete("should never be called")
    assert await manager.generate_step_snapshot(complete) == ""
    assert manager.snapshots == []
    assert manager.get_l3_summary() == ""


def test_l3_epoch_steps_is_floored_at_one() -> None:
    assert HierarchicalMemoryManager(l3_epoch_steps=0).l3_epoch_steps == 1
    assert HierarchicalMemoryManager(l3_epoch_steps=-5).l3_epoch_steps == 1


def test_snapshot_properties_return_copies() -> None:
    manager = HierarchicalMemoryManager()
    manager.snapshots.append(StepSnapshot(0, 0, 0, "", "x", 0))
    manager.epochs.append(EpochSnapshot(0, 0, 0, "y"))
    assert manager.snapshots == []
    assert manager.epochs == []


def test_record_prefers_target_then_draft_then_source() -> None:
    manager = HierarchicalMemoryManager()
    manager.record_drafted_block(_block("a", 0, target="TT", draft="DD", source="SS"))
    manager.record_drafted_block(_block("b", 1, draft="DDDD", source="SSSS"))
    manager.record_drafted_block(_block("c", 2, source="SSSSSS"))
    assert manager._current_buffer_chars == len("TT") + len("DDDD") + len("SSSSSS")


def test_no_buffer_never_triggers() -> None:
    manager = HierarchicalMemoryManager()
    assert manager.should_trigger_snapshot() is False


def test_char_threshold_triggers() -> None:
    manager = HierarchicalMemoryManager(step_chars=5)
    manager.record_drafted_block(_block("a", 0, target="12345"))
    assert manager.should_trigger_snapshot() is True


def test_chapter_transition_triggers_with_enough_content() -> None:
    manager = HierarchicalMemoryManager(step_chars=10_000)
    manager.record_drafted_block(_block("ch1#b1", 0, target="x" * 300))
    assert manager.should_trigger_snapshot(_block("ch2#b1", 1)) is True


def test_chapter_transition_needs_two_blocks_when_short() -> None:
    manager = HierarchicalMemoryManager(step_chars=10_000)
    manager.record_drafted_block(_block("ch1#b1", 0, target="short"))
    assert manager.should_trigger_snapshot(_block("ch2#b1", 1)) is False
    manager.record_drafted_block(_block("ch1#b2", 1, target="short"))
    assert manager.should_trigger_snapshot(_block("ch2#b1", 2)) is True


def test_same_chapter_does_not_trigger() -> None:
    manager = HierarchicalMemoryManager(step_chars=10_000)
    manager.record_drafted_block(_block("ch1#b1", 0, target="x" * 300))
    assert manager.should_trigger_snapshot(_block("ch1#b2", 1)) is False


def test_missing_chapter_namespace_does_not_trigger() -> None:
    manager = HierarchicalMemoryManager(step_chars=10_000)
    manager.record_drafted_block(_block("plain1", 0, target="x" * 300))
    assert manager.should_trigger_snapshot(_block("plain2", 1)) is False
    assert manager.should_trigger_snapshot(_block("ch2#b1", 1)) is False


def test_no_next_block_uses_char_threshold_only() -> None:
    manager = HierarchicalMemoryManager(step_chars=10_000)
    manager.record_drafted_block(_block("ch1#b1", 0, target="x" * 300))
    assert manager.should_trigger_snapshot() is False


async def test_empty_buffer_returns_latest_summary() -> None:
    manager = HierarchicalMemoryManager()
    await _snapshot(manager, [_block("ch1#b1", 0, target="alpha content")], "Summary Alpha")
    assert await manager.generate_step_snapshot(await _fixed_complete("ignored")) == "Summary Alpha"


async def test_snapshot_records_fields_and_increments_step() -> None:
    manager = HierarchicalMemoryManager(l3_epoch_steps=100)
    await _snapshot(
        manager,
        [_block("ch1#b2", 5, target="hello world"), _block("ch1#b1", 1, target="more text")],
        "Summary Alpha",
    )
    snap = manager.snapshots[0]
    assert snap.step_index == 0
    assert snap.start_spine == 1
    assert snap.end_spine == 5
    assert snap.chapter_id == "ch1"
    assert snap.summary_text == "Summary Alpha"
    assert snap.char_count == len("hello world") + len("more text")


async def test_snapshot_clears_the_buffer() -> None:
    manager = HierarchicalMemoryManager()
    await _snapshot(manager, [_block("ch1#b1", 0, target="alpha content")], "Summary Alpha")
    assert manager._unsummarized_blocks == []
    assert manager._current_buffer_chars == 0
    assert manager.should_trigger_snapshot() is False


async def test_llm_summary_is_unquoted_and_whitespace_normalized() -> None:
    manager = HierarchicalMemoryManager()
    summary = await _snapshot(
        manager, [_block("ch1#b1", 0, target="alpha content")], "  “Summary   Alpha”  "
    )
    assert summary == "Summary Alpha"


async def test_short_llm_summary_falls_back_to_deterministic() -> None:
    manager = HierarchicalMemoryManager()
    summary = await _snapshot(manager, [_block("ch1#b1", 0, target="alpha content here")], "tiny")
    assert summary == "alpha content here"


async def test_min_summary_boundary_is_exactly_ten_chars() -> None:
    manager = HierarchicalMemoryManager()
    summary = await _snapshot(manager, [_block("ch1#b1", 0, target="x" * 40)], "0123456789")
    assert summary == "0123456789"


async def test_overlong_llm_summary_is_truncated_to_800() -> None:
    manager = HierarchicalMemoryManager()
    summary = await _snapshot(manager, [_block("ch1#b1", 0, target="x" * 40)], "word " * 300)
    assert 100 < len(summary) <= 800


async def test_deterministic_fallback_uses_uncapped_content() -> None:
    manager = HierarchicalMemoryManager()
    summary = await _snapshot(manager, [_block("ch1#b1", 0, target="A" * 100)], "tiny")
    assert summary == "A" * 100


async def test_llm_exception_degrades_to_deterministic() -> None:
    manager = HierarchicalMemoryManager()

    async def boom(system: str, user: str) -> str:
        raise RuntimeError("provider down")

    manager.record_drafted_block(_block("ch1#b1", 0, target="alpha content here"))
    summary = await manager.generate_step_snapshot(boom)
    assert summary == "alpha content here"


async def test_blank_blocks_produce_empty_summary() -> None:
    manager = HierarchicalMemoryManager()
    summary = await _snapshot(
        manager, [_block("ch1#b1", 0, target="   "), _block("ch1#b2", 1, source="")], "unused"
    )
    assert summary == ""
    assert manager.snapshots[0].summary_text == ""


async def test_epoch_generated_every_l3_epoch_steps() -> None:
    manager = HierarchicalMemoryManager(l3_epoch_steps=2)
    await _snapshot(manager, [_block("ch1#b1", 0, target="a" * 40)], "Summary Alpha")
    assert manager.epochs == []
    await _snapshot(manager, [_block("ch2#b1", 5, target="b" * 40)], "Summary Beta")
    assert len(manager.epochs) == 1
    epoch = manager.epochs[0]
    assert epoch.epoch_index == 0
    assert epoch.start_spine == 0
    assert epoch.end_spine == 5
    assert "Summary Alpha" in epoch.summary_text
    assert "Summary Beta" in epoch.summary_text


async def test_epoch_skipped_when_all_summaries_blank() -> None:
    manager = HierarchicalMemoryManager(l3_epoch_steps=1)
    await _snapshot(manager, [_block("ch1#b1", 0, target="   ")], "unused")
    assert manager.epochs == []


async def test_epoch_index_increments_per_epoch() -> None:
    manager = HierarchicalMemoryManager(l3_epoch_steps=1)
    await _snapshot(manager, [_block("ch1#b1", 0, target="a" * 40)], "Summary One")
    await _snapshot(manager, [_block("ch2#b1", 5, target="b" * 40)], "Summary Two")
    assert [e.epoch_index for e in manager.epochs] == [0, 1]


async def test_second_epoch_uses_the_latest_window() -> None:
    manager = HierarchicalMemoryManager(l3_epoch_steps=2)
    await _snapshot(manager, [_block("ch1#b1", 0, target="a" * 40)], "Summary One")
    await _snapshot(manager, [_block("ch1#b2", 1, target="b" * 40)], "Summary Two")
    await _snapshot(manager, [_block("ch1#b3", 2, target="c" * 40)], "Summary Three")
    await _snapshot(manager, [_block("ch1#b4", 3, target="d" * 40)], "Summary Four")
    assert len(manager.epochs) == 2
    second = manager.epochs[1].summary_text
    assert "Summary Three" in second
    assert "Summary Four" in second
    assert "Summary One" not in second


async def test_epoch_summary_uses_the_l3_cap_not_the_default_400() -> None:
    manager = HierarchicalMemoryManager(l3_epoch_steps=4)
    for i in range(4):
        await _snapshot(manager, [_block(f"ch1#b{i}", i, target="x" * 40)], "A" * 300)
    assert len(manager.epochs) == 1
    assert len(manager.epochs[0].summary_text) > 400


def test_get_l3_summary_keeps_earliest_whole_epochs() -> None:
    manager = HierarchicalMemoryManager()
    manager._epochs = [
        EpochSnapshot(0, 0, 1, "A" * 590),
        EpochSnapshot(1, 2, 3, "B" * 590),
        EpochSnapshot(2, 4, 5, "C" * 590),
    ]
    summary = manager.get_l3_summary()
    assert summary.startswith("A" * 590)
    assert "B" * 590 in summary
    assert "C" not in summary
    assert len(summary) <= 1200


def test_get_l3_summary_truncates_a_single_over_budget_epoch() -> None:
    manager = HierarchicalMemoryManager()
    manager._epochs = [EpochSnapshot(0, 0, 1, "Z" * 2000)]
    assert len(manager.get_l3_summary()) == 1200


def test_get_l3_summary_skips_blank_epochs() -> None:
    manager = HierarchicalMemoryManager()
    manager._epochs = [EpochSnapshot(0, 0, 0, "  "), EpochSnapshot(1, 1, 1, "real")]
    assert manager.get_l3_summary() == "real"


def test_get_l3_summary_sliding_window() -> None:
    manager = HierarchicalMemoryManager(sliding_window=True)
    manager._epochs = [
        EpochSnapshot(0, 0, 1, "Old" * 200),
        EpochSnapshot(1, 2, 3, "Middle" * 50),
        EpochSnapshot(2, 4, 5, "Recent" * 50),
    ]
    summary = manager.get_l3_summary()
    assert "Recent" in summary
    assert "Middle" in summary
    assert "Old" not in summary
    assert len(summary) <= 1200


async def test_max_epochs_capping_in_ultra_long_books() -> None:
    manager = HierarchicalMemoryManager(l3_epoch_steps=1, max_epochs=3)
    for i in range(10):
        await _snapshot(manager, [_block(f"ch1#b{i}", i, target="x" * 40)], f"summary_{i}")
    assert len(manager.epochs) == 3
    assert [e.epoch_index for e in manager.epochs] == [7, 8, 9]


def test_restore_state_caps_max_epochs() -> None:
    from ubt.core.memory.hierarchical_memory import DEFAULT_MAX_EPOCHS

    manager = HierarchicalMemoryManager(max_epochs=3)
    state = {
        "snapshots": [],
        "epochs": [
            {"epoch_index": i, "start_spine": i, "end_spine": i, "summary_text": f"s_{i}"}
            for i in range(10)
        ],
        "step_counter": 10,
    }
    manager.restore_state(state)
    assert len(manager.epochs) == 3
    assert [e.epoch_index for e in manager.epochs] == [7, 8, 9]

    default_mgr = HierarchicalMemoryManager()
    assert default_mgr.max_epochs == DEFAULT_MAX_EPOCHS == 50


def test_replace_chapter_snapshot_drops_stale_recaps_and_rebuilds_epochs() -> None:
    manager = HierarchicalMemoryManager(l3_epoch_steps=2)
    manager._snapshots = [
        StepSnapshot(0, 0, 1, "ch1", "old chapter one", 10),
        StepSnapshot(1, 2, 3, "ch1", "older chapter one", 10),
        StepSnapshot(2, 4, 5, "ch2", "chapter two", 10),
    ]
    manager.replace_chapter_snapshot(
        "ch1", start_spine=0, end_spine=3, summary_text="repaired chapter one", char_count=10
    )
    ch1 = [s for s in manager.snapshots if s.chapter_id == "ch1"]
    assert len(ch1) == 1
    assert ch1[0].summary_text == "repaired chapter one"
    combined = " ".join(e.summary_text for e in manager.epochs)
    assert "repaired chapter one" in combined
    assert "older chapter one" not in combined


async def test_refresh_chapter_memory_replaces_the_post_repair_recap() -> None:
    class _Ledger:
        def __init__(self) -> None:
            self.meta: dict[str, Any] = {}

        def get_job_metadata_value(self, _job: str, key: str) -> Any:
            return self.meta.get(key)

        def set_job_metadata_value(self, _job: str, key: str, value: Any) -> None:
            self.meta[key] = value

    ledger = _Ledger()
    blocks = [
        _block("ch1#b1", 1, target="修复后的第一段"),
        _block("ch1#b2", 2, target="修复后的第二段"),
    ]

    async def complete(_system: str, _user: str) -> str:
        return "修复后的章节摘要内容如下：第一段、第二段。"

    assert await refresh_chapter_memory(ledger, "job", "ch1", blocks, complete, "zh") is True
    state = ledger.meta["memory_state"]
    assert state["snapshots"][0]["summary_text"] == "修复后的章节摘要内容如下：第一段、第二段。"
    assert state["snapshots"][0]["chapter_id"] == "ch1"


async def test_get_latest_macro_summary() -> None:
    manager = HierarchicalMemoryManager()
    assert manager.get_latest_macro_summary() == ""
    await _snapshot(manager, [_block("ch1#b1", 0, target="a" * 40)], "Summary First")
    await _snapshot(manager, [_block("ch2#b1", 5, target="b" * 40)], "Summary Second")
    assert manager.get_latest_macro_summary() == "Summary Second"


async def test_export_restore_round_trip() -> None:
    manager = HierarchicalMemoryManager(l3_epoch_steps=1)
    await _snapshot(manager, [_block("ch1#b1", 0, target="a" * 40)], "Summary Alpha")
    state = manager.export_state()
    assert set(state) == {"snapshots", "epochs", "step_counter"}

    restored = HierarchicalMemoryManager()
    restored.restore_state(state)
    assert restored.export_state() == state


def test_restore_filters_non_dicts_and_falls_back_on_bad_counter() -> None:
    manager = HierarchicalMemoryManager()
    manager.restore_state(
        {
            "snapshots": [
                {
                    "step_index": 0,
                    "start_spine": 1,
                    "end_spine": 2,
                    "chapter_id": "c",
                    "summary_text": "s",
                    "char_count": 3,
                },
                "junk",
            ],
            "epochs": [
                {"epoch_index": 0, "start_spine": 1, "end_spine": 2, "summary_text": "e"},
                None,
            ],
            "step_counter": "not-an-int",
        }
    )
    assert len(manager.snapshots) == 1
    assert len(manager.epochs) == 1
    assert manager._step_counter == 1


def test_restore_defaults_counter_to_snapshot_count() -> None:
    manager = HierarchicalMemoryManager()
    manager.restore_state(
        {
            "snapshots": [
                {
                    "step_index": 0,
                    "start_spine": 1,
                    "end_spine": 2,
                    "chapter_id": "c",
                    "summary_text": "s",
                    "char_count": 3,
                }
            ]
        }
    )
    assert manager._step_counter == 1


async def _chaptered_manager() -> HierarchicalMemoryManager:
    manager = HierarchicalMemoryManager(l3_epoch_steps=100)
    await _snapshot(manager, [_block("ch1#b1", 1, target="a" * 40)], "Summary Alpha")
    await _snapshot(manager, [_block("ch2#b1", 5, target="b" * 40)], "Summary Beta")
    return manager


async def test_macro_context_prefers_exact_chapter_match() -> None:
    manager = await _chaptered_manager()
    assert manager.get_macro_context_for_block(_block("ch1#b9", 3)) == "Summary Alpha"


async def test_macro_context_chapter_match_beats_a_more_recent_snapshot() -> None:
    manager = await _chaptered_manager()
    assert manager.get_macro_context_for_block(_block("ch1#b9", 6)) == "Summary Alpha"


async def test_macro_context_falls_back_to_latest_prior_snapshot() -> None:
    manager = await _chaptered_manager()
    assert manager.get_macro_context_for_block(_block("ch9#b1", 9)) == "Summary Beta"


async def test_macro_context_empty_when_no_prior_snapshot() -> None:
    manager = await _chaptered_manager()
    assert manager.get_macro_context_for_block(_block("ch1#b1", 0)) == ""


async def test_macro_context_without_chapter_uses_spine_fallback() -> None:
    manager = await _chaptered_manager()
    assert manager.get_macro_context_for_block(_block("plain", 9)) == "Summary Beta"


def test_macro_context_skips_blank_summaries() -> None:
    manager = HierarchicalMemoryManager()
    manager._snapshots = [
        StepSnapshot(0, 0, 1, "ch1", "real", 5),
        StepSnapshot(1, 0, 2, "ch1", "", 5),
    ]
    assert manager.get_macro_context_for_block(_block("plain", 3)) == "real"


def test_l1_context_delegates_to_neighbor_builder() -> None:
    manager = HierarchicalMemoryManager(neighbor_chars=50)
    target = _block("b2", 2, target="target")
    surrounding = [_block("b1", 1, target="previous text"), target]
    assert "previous text" in manager.get_l1_context(target, surrounding)


def test_l1_context_uses_fallbacks() -> None:
    manager = HierarchicalMemoryManager()
    target = _block("b2", 2, target="target")
    rendered = manager.get_l1_context(
        target, [target], fallback_prev_text="PB", fallback_next_text="NB"
    )
    assert "PB" in rendered
    assert "NB" in rendered


def test_export_state_is_json_serializable_shape() -> None:
    manager = HierarchicalMemoryManager()
    manager._snapshots = [StepSnapshot(0, 1, 2, "c", "s", 3)]
    state: dict[str, Any] = manager.export_state()
    assert state["snapshots"][0]["chapter_id"] == "c"


def test_pending_block_count_reports_the_buffer() -> None:
    manager = HierarchicalMemoryManager(step_chars=10_000)
    assert manager.pending_block_count == 0
    manager.record_drafted_block(_block("ch1#b1", 1, target="第一章的正文内容。"))
    assert manager.pending_block_count == 1
    manager.discard_pending_blocks()
    assert manager.pending_block_count == 0


@pytest.mark.asyncio
async def test_a_short_chapter_can_be_summarized_before_the_buffer_is_dropped() -> None:
    manager = HierarchicalMemoryManager(step_chars=10_000)
    blocks = [_block("ch1#b1", 1, target="第一章的正文内容。" * 100)]
    for block in blocks:
        manager.record_drafted_block(block)

    assert manager.should_trigger_snapshot() is False
    assert manager.pending_block_count == 1

    summary = await manager.generate_step_snapshot(
        await _fixed_complete("本章讨论了算术逻辑单元的用途以及它在流水线中的位置。")
    )
    assert summary, "a chapter-end flush produced no summary at all"
    assert manager.pending_block_count == 0
    assert len(manager.snapshots) == 1

    state = manager.export_state()
    restored = HierarchicalMemoryManager(step_chars=10_000)
    restored.restore_state(state)
    assert len(restored.snapshots) == 1
    assert restored.snapshots[0].summary_text == summary


def test_a_chapter_transition_triggers_once_the_buffer_has_content() -> None:
    manager = HierarchicalMemoryManager(step_chars=10_000)
    manager.record_drafted_block(_block("ch1#b1", 1, target="第一章的正文内容。" * 100))

    assert manager.should_trigger_snapshot() is False
    assert manager.should_trigger_snapshot(_block("ch2#b1", 2, target="第二章。")) is True
    assert manager.should_trigger_snapshot(_block("ch1#b2", 2, target="同章。")) is False
