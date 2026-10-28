"""Data models for the interactive console workbench."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class WorkbenchMode(StrEnum):
    """Operation mode for the interactive workbench."""

    SINGLE_JOB = "single_job"
    QUEUE_WORKER = "queue_worker"
    QUEUE_MONITOR = "queue_monitor"


class JobControlStatus(StrEnum):
    """Execution state of the active job controller."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    FAILED = "failed"


class KeyAction(StrEnum):
    """Recognized user keyboard interactions in the workbench."""

    TOGGLE_PAUSE = "toggle_pause"
    INCREASE_CONCURRENCY = "increase_concurrency"
    DECREASE_CONCURRENCY = "decrease_concurrency"
    NEXT_TAB = "next_tab"
    PREV_TAB = "prev_tab"
    NEXT_JOB = "next_job"
    PREV_JOB = "prev_job"
    RETRY_FAILED = "retry_failed"
    TOGGLE_HELP = "toggle_help"
    QUIT = "quit"


@dataclass
class ResourceSnapshot:
    """System and pipeline resource metrics."""

    rss_mb: float = 0.0
    cpu_time_s: float = 0.0
    concurrency_limit: int = 1
    active_workers: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0
    cache_hit_rate: float = 0.0
    throughput_wps: float = 0.0


@dataclass
class JobViewItem:
    """Summary of a single translation job in the queue or pipeline."""

    job_id: str
    title: str = ""
    status: str = "queued"
    stage: str = "init"
    progress_pct: float = 0.0
    blocks_done: int = 0
    blocks_total: int = 0
    quarantined_count: int = 0
    repaired_count: int = 0
    error: str | None = None
    started_at: float | None = None
    elapsed_s: float = 0.0
    lease_expires_in: float | None = None


@dataclass
class WorkbenchState:
    """Complete mutable state of the active workbench session."""

    mode: WorkbenchMode = WorkbenchMode.SINGLE_JOB
    status: JobControlStatus = JobControlStatus.IDLE
    selected_tab: int = 0  # 0: Pipeline, 1: Queue, 2: Resources, 3: Diagnostics
    active_job_id: str = ""
    active_job_title: str = ""
    target_lang: str = "zh"
    source_lang: str = "en"
    start_time: float = field(default_factory=time.time)
    current_stage: str = "Ready"
    progress_pct: float = 0.0
    blocks_done: int = 0
    blocks_total: int = 0
    quarantined_count: int = 0
    repaired_count: int = 0
    concurrency: int = 1
    show_help: bool = False
    status_message: str = "Press [?] for help, [P] to pause/resume"
    jobs: list[JobViewItem] = field(default_factory=list)
    resource: ResourceSnapshot = field(default_factory=ResourceSnapshot)
    recent_events: list[str] = field(default_factory=list)
    quarantined_details: list[dict[str, Any]] = field(default_factory=list)

    @property
    def elapsed_s(self) -> float:
        return max(0.0, time.time() - self.start_time)

    def add_event(self, msg: str, max_events: int = 8) -> None:
        self.recent_events.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        if len(self.recent_events) > max_events:
            self.recent_events.pop(0)
