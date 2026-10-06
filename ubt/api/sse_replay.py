"""Bounded per-job SSE frame history for ``Last-Event-ID`` replay.

The console's progress stream is long-lived; a dropped connection (sleep,
proxy timeout) used to lose whatever frames were emitted while the client was
away. Each frame is stamped with a monotonic ``id:`` and kept in a small ring
buffer per job, so a reconnecting client that sends ``Last-Event-ID`` gets the
missed frames before the live stream resumes (PRD §9.2).

The history is deliberately small and per-process: a frame is a full state
snapshot, so a client that reconnects *without* a cursor still converges on the
current state — the buffer only spares it the intermediate frames.
"""

from __future__ import annotations

import threading
from collections import OrderedDict, deque

#: Frames retained per job. A snapshot stream does not need deep history.
DEFAULT_MAX_FRAMES = 64

#: Jobs whose history is retained. Oldest-inserted jobs are evicted first, so a
#: long-lived server cannot grow this without bound.
DEFAULT_MAX_JOBS = 256


class SseReplayBuffer:
    """Thread-safe, bounded ``job_id -> [(seq, frame)]`` ring buffer."""

    def __init__(
        self, *, max_frames: int = DEFAULT_MAX_FRAMES, max_jobs: int = DEFAULT_MAX_JOBS
    ) -> None:
        self._max_frames = max_frames
        self._max_jobs = max_jobs
        self._lock = threading.Lock()
        self._seq: dict[str, int] = {}
        self._buffers: OrderedDict[str, deque[tuple[int, str]]] = OrderedDict()

    def emit(self, job_id: str, frame: str) -> str:
        """Stamp ``frame`` with the next sequence id and retain it.

        Returns the frame prefixed with its ``id:`` line, ready to yield.
        """
        with self._lock:
            seq = self._seq.get(job_id, 0) + 1
            self._seq[job_id] = seq
            buffer = self._buffers.get(job_id)
            if buffer is None:
                buffer = deque(maxlen=self._max_frames)
                self._buffers[job_id] = buffer
                if len(self._buffers) > self._max_jobs:
                    evicted, _ = self._buffers.popitem(last=False)
                    self._seq.pop(evicted, None)
            else:
                self._buffers.move_to_end(job_id)
            stamped = f"id: {seq}\n{frame}"
            buffer.append((seq, stamped))
            return stamped

    def replay(self, job_id: str, last_event_id: int | None) -> list[str]:
        """Frames with a sequence greater than ``last_event_id``.

        A client that sends no cursor gets an empty replay: it is answered with
        the current snapshot instead of the retained history.
        """
        if last_event_id is None:
            return []
        with self._lock:
            buffer = self._buffers.get(job_id)
            if not buffer:
                return []
            return [frame for seq, frame in buffer if seq > last_event_id]

    def forget(self, job_id: str) -> None:
        """Drop a job's history (e.g. once it reaches a terminal state)."""
        with self._lock:
            self._buffers.pop(job_id, None)
            self._seq.pop(job_id, None)
