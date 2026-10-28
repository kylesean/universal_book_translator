"""Controller for the interactive console workbench."""

from __future__ import annotations

import asyncio
import logging
import resource
import time
from collections.abc import Callable
from typing import Any

from ubt.cli.workbench.models import (
    JobControlStatus,
    KeyAction,
    ResourceSnapshot,
    WorkbenchMode,
    WorkbenchState,
)

logger = logging.getLogger(__name__)


class WorkbenchController:
    """Manages async job state, keyboard control dispatch, and resource monitoring."""

    def __init__(
        self,
        mode: WorkbenchMode = WorkbenchMode.SINGLE_JOB,
        concurrency: int = 1,
        cancel_token: asyncio.Event | None = None,
    ) -> None:
        self.state = WorkbenchState(mode=mode, concurrency=max(1, concurrency))
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # Running by default
        self._cancel_token = cancel_token or asyncio.Event()
        self._subscribers: list[Callable[[WorkbenchState], None]] = []
        self._last_cpu_time = time.process_time()
        self._last_sample_time = time.time()

    @property
    def is_paused(self) -> bool:
        return not self._pause_event.is_set()

    @property
    def is_cancelled(self) -> bool:
        return self._cancel_token.is_set()

    @property
    def cancel_token(self) -> asyncio.Event:
        return self._cancel_token

    def subscribe(self, callback: Callable[[WorkbenchState], None]) -> None:
        """Register a state change listener."""
        self._subscribers.append(callback)

    def notify(self) -> None:
        """Trigger update on all registered subscribers."""
        for sub in self._subscribers:
            try:
                sub(self.state)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Workbench subscriber failed: %s", exc)

    def pause(self) -> None:
        """Pause execution at next checkpoint."""
        self._pause_event.clear()
        self.state.status = JobControlStatus.PAUSED
        self.state.status_message = "PAUSED. Press [P] or [Space] to resume."
        self.state.add_event("Pipeline paused by operator")
        self.notify()

    def resume(self) -> None:
        """Resume execution."""
        self._pause_event.set()
        self.state.status = JobControlStatus.RUNNING
        self.state.status_message = "RUNNING"
        self.state.add_event("Pipeline resumed by operator")
        self.notify()

    def toggle_pause(self) -> None:
        """Toggle between pause and resume."""
        if self.is_paused:
            self.resume()
        else:
            self.pause()

    async def wait_if_paused(self) -> None:
        """Cooperative checkpoint: pause coroutine if operator pressed Pause."""
        if not self._pause_event.is_set():
            await self._pause_event.wait()

    def request_cancel(self) -> None:
        """Request graceful cancellation and checkpoint persistence."""
        self._cancel_token.set()
        self._pause_event.set()  # Unblock if paused so cancel proceeds
        self.state.status = JobControlStatus.CANCELLING
        self.state.status_message = "CANCELLING... Flushing checkpoints to ledger."
        self.state.add_event("Graceful shutdown requested by operator")
        self.notify()

    def increase_concurrency(self, max_limit: int = 16) -> int:
        """Dynamically increase concurrency throttle."""
        if self.state.concurrency < max_limit:
            self.state.concurrency += 1
            self.state.status_message = f"Concurrency increased to {self.state.concurrency}"
            self.state.add_event(f"Concurrency scaled up -> {self.state.concurrency}")
            self.notify()
        return self.state.concurrency

    def decrease_concurrency(self, min_limit: int = 1) -> int:
        """Dynamically decrease concurrency throttle."""
        if self.state.concurrency > min_limit:
            self.state.concurrency -= 1
            self.state.status_message = f"Concurrency decreased to {self.state.concurrency}"
            self.state.add_event(f"Concurrency scaled down -> {self.state.concurrency}")
            self.notify()
        return self.state.concurrency

    def next_tab(self) -> None:
        """Cycle to next view tab."""
        self.state.selected_tab = (self.state.selected_tab + 1) % 4
        self.notify()

    def prev_tab(self) -> None:
        """Cycle to previous view tab."""
        self.state.selected_tab = (self.state.selected_tab - 1) % 4
        self.notify()

    def toggle_help(self) -> None:
        """Toggle help keybinding overlay."""
        self.state.show_help = not self.state.show_help
        self.notify()

    def handle_key(self, key: str) -> KeyAction | None:
        """Translate raw key input to high-level action and execute state change."""
        if not key:
            return None

        # Space or P: Pause/Resume
        if key in ("p", "P", " "):
            self.toggle_pause()
            return KeyAction.TOGGLE_PAUSE

        # + or =: Increase concurrency
        if key in ("+", "="):
            self.increase_concurrency()
            return KeyAction.INCREASE_CONCURRENCY

        # - or _: Decrease concurrency
        if key in ("-", "_"):
            self.decrease_concurrency()
            return KeyAction.DECREASE_CONCURRENCY

        # Tab or l: Next tab
        if key in ("\t", "l", "L"):
            self.next_tab()
            return KeyAction.NEXT_TAB

        # h: Prev tab
        if key in ("h", "H"):
            self.prev_tab()
            return KeyAction.PREV_TAB

        # j / Down: Next Job
        if key in ("j", "J"):
            return KeyAction.NEXT_JOB

        # k / Up: Prev Job
        if key in ("k", "K"):
            return KeyAction.PREV_JOB

        # r: Retry Failed
        if key in ("r", "R"):
            self.state.add_event("Retrying quarantined blocks requested")
            self.notify()
            return KeyAction.RETRY_FAILED

        # ?: Help
        if key == "?":
            self.toggle_help()
            return KeyAction.TOGGLE_HELP

        # q / Q / Ctrl-C (\x03)
        if key in ("q", "Q", "\x03"):
            self.request_cancel()
            return KeyAction.QUIT

        return None

    def sample_resources(self) -> ResourceSnapshot:
        """Sample local process resource utilization (zero external dependencies)."""
        now = time.time()
        cpu_now = time.process_time()

        # On Linux, ru_maxrss is in KiB
        try:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            rss_mb = usage.ru_maxrss / 1024.0
        except Exception:  # noqa: BLE001
            rss_mb = 0.0

        res = self.state.resource
        res.rss_mb = round(rss_mb, 1)
        res.cpu_time_s = round(cpu_now, 2)
        res.concurrency_limit = self.state.concurrency

        self._last_cpu_time = cpu_now
        self._last_sample_time = now
        return res

    def update_from_event(self, event_type: str, message: str, meta: dict[str, Any]) -> None:
        """Update workbench state from pipeline/worker progress events."""
        self.state.current_stage = event_type
        self.state.add_event(f"[{event_type}] {message}")

        if "pct" in meta or "percent" in meta:
            self.state.progress_pct = float(meta.get("pct", meta.get("percent", 0.0)))

        if "blocks_done" in meta:
            self.state.blocks_done = int(meta["blocks_done"])
        if "blocks_total" in meta:
            self.state.blocks_total = int(meta["blocks_total"])
        if "quarantined" in meta:
            self.state.quarantined_count = int(meta["quarantined"])
        if "repaired" in meta:
            self.state.repaired_count = int(meta["repaired"])

        # Token & cost stats
        res = self.state.resource
        if "prompt_tokens" in meta:
            res.prompt_tokens += int(meta["prompt_tokens"])
        if "completion_tokens" in meta:
            res.completion_tokens += int(meta["completion_tokens"])
        if "cached_tokens" in meta:
            res.cached_tokens += int(meta["cached_tokens"])
        if "cost_usd" in meta:
            res.cost_usd += float(meta["cost_usd"])

        tot = res.prompt_tokens + res.cached_tokens
        if tot > 0:
            res.cache_hit_rate = round(res.cached_tokens / tot * 100.0, 1)

        self.sample_resources()
        self.notify()
