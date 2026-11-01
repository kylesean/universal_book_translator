from __future__ import annotations

import pytest

from ubt.api.sse_replay import SseReplayBuffer

pytestmark = pytest.mark.fast


def _frame(text: str) -> str:
    return f"event: progress\ndata: {text}\n\n"


def test_emit_stamps_increasing_ids_and_prefixes_the_frame() -> None:
    buffer = SseReplayBuffer()
    first = buffer.emit("job1", _frame("a"))
    second = buffer.emit("job1", _frame("b"))
    assert first.startswith("id: 1\n")
    assert second.startswith("id: 2\n")
    assert first.endswith(_frame("a"))


def test_replay_without_a_cursor_is_empty() -> None:
    buffer = SseReplayBuffer()
    buffer.emit("job1", _frame("a"))
    assert buffer.replay("job1", None) == []


def test_replay_returns_only_frames_after_the_cursor() -> None:
    buffer = SseReplayBuffer()
    buffer.emit("job1", _frame("a"))
    buffer.emit("job1", _frame("b"))
    third = buffer.emit("job1", _frame("c"))

    replayed = buffer.replay("job1", 1)
    assert len(replayed) == 2
    assert replayed[-1] == third
    assert buffer.replay("job1", 3) == []


def test_replay_is_scoped_per_job() -> None:
    buffer = SseReplayBuffer()
    buffer.emit("job1", _frame("a"))
    buffer.emit("job2", _frame("b"))
    assert len(buffer.replay("job1", 0)) == 1
    assert len(buffer.replay("job2", 0)) == 1


def test_frames_are_bounded_per_job() -> None:
    buffer = SseReplayBuffer(max_frames=3)
    for index in range(5):
        buffer.emit("job1", _frame(str(index)))
    replayed = buffer.replay("job1", 0)
    assert len(replayed) == 3
    assert replayed[0].startswith("id: 3\n")


def test_jobs_are_bounded_and_evict_the_oldest() -> None:
    buffer = SseReplayBuffer(max_jobs=2)
    buffer.emit("job1", _frame("a"))
    buffer.emit("job2", _frame("b"))
    buffer.emit("job3", _frame("c"))
    assert buffer.replay("job1", 0) == []
    assert len(buffer.replay("job3", 0)) == 1


def test_forget_drops_a_jobs_history() -> None:
    buffer = SseReplayBuffer()
    buffer.emit("job1", _frame("a"))
    buffer.forget("job1")
    assert buffer.replay("job1", 0) == []


def test_a_repeated_frame_does_not_consume_a_second_slot() -> None:
    buffer = SseReplayBuffer()
    first = buffer.emit("job1", _frame("same"))
    again = buffer.emit("job1", _frame("same"))
    assert again == first, "a repeat must reuse the retained id, not advance it"
    assert len(buffer.replay("job1", 0)) == 1


def test_a_terminal_job_reconnecting_repeatedly_cannot_flush_its_history() -> None:
    buffer = SseReplayBuffer(max_frames=64)
    snapshot = _frame('{"stage": "package", "status": "completed"}')
    terminal = _frame('{"status": "completed"}')
    for _ in range(30):
        buffer.emit("job1", snapshot)
        buffer.emit("job1", terminal)

    retained = buffer.replay("job1", 0)
    assert len(retained) == 2, f"duplicates flushed the history: {len(retained)} frames"
    assert retained[-1].endswith(terminal)


def test_distinct_progress_frames_still_accumulate() -> None:
    buffer = SseReplayBuffer(max_frames=64)
    for index in range(40):
        buffer.emit("job1", _frame(str(index)))
    assert len(buffer.replay("job1", 0)) == 40


def test_a_refreshed_frame_keeps_its_cursor_valid() -> None:
    buffer = SseReplayBuffer(max_frames=4)
    for index in range(3):
        buffer.emit("job1", _frame(str(index)))
    first_id = buffer.emit("job1", _frame("7")).split("\n", 1)[0]
    assert first_id == "id: 4"
    assert buffer.emit("job1", _frame("0")).split("\n", 1)[0] == "id: 1"
    assert [f.split("id: ")[1].split("\n")[0] for f in buffer.replay("job1", 3)] == ["4"]
