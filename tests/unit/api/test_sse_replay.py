"""The bounded SSE frame history behind ``Last-Event-ID`` replay (PRD §9.2)."""

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
    # A fresh subscriber converges on the current snapshot; it gets no history.
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
    assert replayed[0].startswith("id: 3\n")  # ids 1 and 2 aged out


def test_jobs_are_bounded_and_evict_the_oldest() -> None:
    buffer = SseReplayBuffer(max_jobs=2)
    buffer.emit("job1", _frame("a"))
    buffer.emit("job2", _frame("b"))
    buffer.emit("job3", _frame("c"))  # evicts job1
    assert buffer.replay("job1", 0) == []
    assert len(buffer.replay("job3", 0)) == 1


def test_forget_drops_a_jobs_history() -> None:
    buffer = SseReplayBuffer()
    buffer.emit("job1", _frame("a"))
    buffer.forget("job1")
    assert buffer.replay("job1", 0) == []
