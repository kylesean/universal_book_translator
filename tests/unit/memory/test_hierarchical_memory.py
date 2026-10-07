"""Contract tests for the hierarchical (L1/L2/L3) memory manager."""

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


# --------------------------------------------------------------------------- #
# Construction / defaults
# --------------------------------------------------------------------------- #


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
    # resolve_draft_policy turns hierarchical L2/L3 off for a single-chapter
    # document. Before the flag, that route still paid the L2 llm_summary calls
    # while refusing to inject the L2 macro slot -- L3 alone rode the prompt,
    # at L2's price.
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
    assert manager.snapshots == []  # internal list untouched
    assert manager.epochs == []


# --------------------------------------------------------------------------- #
# record_drafted_block
# --------------------------------------------------------------------------- #


def test_record_prefers_target_then_draft_then_source() -> None:
    manager = HierarchicalMemoryManager()
    manager.record_drafted_block(_block("a", 0, target="TT", draft="DD", source="SS"))
    manager.record_drafted_block(_block("b", 1, draft="DDDD", source="SSSS"))
    manager.record_drafted_block(_block("c", 2, source="SSSSSS"))
    assert manager._current_buffer_chars == len("TT") + len("DDDD") + len("SSSSSS")


# --------------------------------------------------------------------------- #
# should_trigger_snapshot
# --------------------------------------------------------------------------- #


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
    # One short block, chapter differs: not enough evidence yet.
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
    # And a delimited next block against an undelimited buffer is also silent.
    assert manager.should_trigger_snapshot(_block("ch2#b1", 1)) is False


def test_no_next_block_uses_char_threshold_only() -> None:
    manager = HierarchicalMemoryManager(step_chars=10_000)
    manager.record_drafted_block(_block("ch1#b1", 0, target="x" * 300))
    assert manager.should_trigger_snapshot() is False


# --------------------------------------------------------------------------- #
# generate_step_snapshot
# --------------------------------------------------------------------------- #


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
    assert snap.start_spine == 1  # sorted by spine
    assert snap.end_spine == 5
    assert snap.chapter_id == "ch1"  # from the first (spine-ordered) block
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
    # A short LLM reply forces the deterministic fallback over the block text.
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


# --------------------------------------------------------------------------- #
# L3 epochs
# --------------------------------------------------------------------------- #


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
    assert "C" not in summary  # would exceed the 1200-char clamp
    assert len(summary) <= 1200


def test_get_l3_summary_truncates_a_single_over_budget_epoch() -> None:
    manager = HierarchicalMemoryManager()
    manager._epochs = [EpochSnapshot(0, 0, 1, "Z" * 2000)]
    assert len(manager.get_l3_summary()) == 1200


def test_get_l3_summary_skips_blank_epochs() -> None:
    manager = HierarchicalMemoryManager()
    manager._epochs = [EpochSnapshot(0, 0, 0, "  "), EpochSnapshot(1, 1, 1, "real")]
    assert manager.get_l3_summary() == "real"


def test_format_l3_prompt_block() -> None:
    manager = HierarchicalMemoryManager()
    assert manager.format_l3_prompt_block("   ") == ""
    rendered = manager.format_l3_prompt_block("  history  ")
    assert rendered == "### Book Continuity (Compressed History)\nhistory"


# --------------------------------------------------------------------------- #
# Latest summary / export / restore
# --------------------------------------------------------------------------- #


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
    assert manager._step_counter == 1  # falls back to len(snapshots)


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


# --------------------------------------------------------------------------- #
# get_macro_context_for_block
# --------------------------------------------------------------------------- #


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
    # Spine 6 is past both snapshots, but the ch1 snapshot is the chapter match.
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
    # No chapter id -> spine fallback walks back past the newest (blank) snapshot.
    assert manager.get_macro_context_for_block(_block("plain", 3)) == "real"


# --------------------------------------------------------------------------- #
# L1 delegation
# --------------------------------------------------------------------------- #


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
