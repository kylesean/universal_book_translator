"""Asynchronous interactive runtime loop and terminal IO handler for the workbench."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sys
from collections.abc import Callable, Coroutine
from typing import Any

from rich.live import Live

from ubt.cli.workbench.controller import WorkbenchController
from ubt.cli.workbench.dashboard import render_dashboard
from ubt.cli.workbench.models import JobControlStatus, JobViewItem

logger = logging.getLogger(__name__)


class InteractiveKeyReader:
    """Non-blocking keyboard event reader for POSIX terminals."""

    def __init__(self, on_key: Callable[[str], object]) -> None:
        # ``object`` rather than ``None``: the workbench passes
        # ``controller.handle_key``, which returns a ``KeyAction | None``
        # describing what it did. The reader deliberately ignores that value --
        # the action has already been applied by the time it returns -- so the
        # parameter accepts any return type instead of forcing call sites to
        # wrap a callback in a discarding lambda.
        self.on_key = on_key
        self._fd = sys.stdin.fileno() if sys.stdin.isatty() else None
        self._old_settings: Any = None
        self._registered = False

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._fd is None:
            return
        # Bind once so the closures below capture a narrowed ``int``; mypy does
        # not track the ``None`` check into an attribute read inside a callback.
        fd = self._fd

        try:
            import termios
            import tty

            self._old_settings = termios.tcgetattr(fd)
            tty.setcbreak(fd)

            def _read_char() -> None:
                try:
                    ch = os.read(fd, 32).decode("utf-8", errors="ignore")
                    for c in ch:
                        self.on_key(c)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Key read error: %s", exc)

            loop.add_reader(fd, _read_char)
            self._registered = True
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to initialize cbreak mode on stdin: %s", exc)

    def stop(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._fd is None:
            return

        if self._registered:
            with contextlib.suppress(Exception):
                loop.remove_reader(self._fd)
            self._registered = False

        if self._old_settings is not None:
            with contextlib.suppress(Exception):
                import termios

                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
            self._old_settings = None


async def run_workbench(
    controller: WorkbenchController,
    task_fn: Callable[[WorkbenchController], Coroutine[Any, Any, None]] | None = None,
    queue: Any | None = None,
    refresh_rate: float = 4.0,
) -> None:
    """Run interactive full-screen workbench with concurrent task execution."""
    loop = asyncio.get_running_loop()
    key_reader = InteractiveKeyReader(on_key=controller.handle_key)

    key_reader.start(loop)
    controller.state.status = JobControlStatus.RUNNING

    worker_task: asyncio.Task[None] | None = None
    if task_fn is not None:
        worker_task = asyncio.create_task(task_fn(controller))

    try:
        with Live(
            render_dashboard(controller.state),
            refresh_per_second=refresh_rate,
            screen=True,
        ) as live:
            while not controller.is_cancelled:
                # 1. Update queue items if queue is attached
                if queue is not None:
                    try:
                        raw_jobs = queue.list_jobs(limit=10)
                        controller.state.jobs = [
                            JobViewItem(
                                job_id=j.job_id,
                                title=str(j.payload.get("input_path", j.job_id)),
                                status=j.status.value,
                                progress_pct=float(j.progress.get("percent", 0.0)),
                                blocks_done=int(j.progress.get("blocks_done", 0)),
                                blocks_total=int(j.progress.get("blocks_total", 0)),
                                error=j.error,
                            )
                            for j in raw_jobs
                        ]
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("Queue poll error in workbench: %s", exc)

                # 2. Sample process resources
                controller.sample_resources()

                # 3. Check if worker completed
                if worker_task is not None and worker_task.done():
                    if worker_task.exception() is not None:
                        controller.state.status = JobControlStatus.FAILED
                        controller.state.status_message = f"FAILED: {worker_task.exception()}"
                    else:
                        controller.state.status = JobControlStatus.COMPLETED
                        controller.state.status_message = "COMPLETED! Press [Q] to return to shell."
                    live.update(render_dashboard(controller.state))
                    break

                # 4. Render frame
                live.update(render_dashboard(controller.state))
                await asyncio.sleep(1.0 / refresh_rate)

            # Final render frame
            live.update(render_dashboard(controller.state))

            # If cancelled, cancel worker
            if controller.is_cancelled and worker_task is not None and not worker_task.done():
                worker_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await worker_task

    finally:
        key_reader.stop(loop)
