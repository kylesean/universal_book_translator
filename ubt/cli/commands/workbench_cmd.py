"""CLI command for launching the interactive operator console workbench."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer

from ubt.cli.commands import resolve_db_dir


def workbench_command(
    db_dir: Annotated[
        Path | None,
        typer.Option(
            "--db-dir",
            help="Directory holding the job queue and per-job ledgers (default: config / UBT_DB_DIR)",
            show_default=False,
        ),
    ] = None,
    queue_db: Annotated[
        Path | None,
        typer.Option(
            "--queue-db", help="Override the queue SQLite path (<db-dir>/job_queue.sqlite)"
        ),
    ] = None,
    concurrency: Annotated[
        int, typer.Option("--concurrency", "-c", help="Concurrent worker execution slots")
    ] = 2,
    job_id: Annotated[
        str | None, typer.Option("--job-id", help="Focus workbench on a specific active job ID")
    ] = None,
    refresh_rate: Annotated[
        float, typer.Option("--refresh-rate", help="Screen refresh frequency in Hz")
    ] = 4.0,
) -> None:
    """Launch the full-screen interactive operator console workbench.

    Provides real-time pipeline telemetry, dynamic pause/resume ([P]),
    runtime concurrency throttling ([+]/[-]), and multi-job queue switching ([Tab]).
    """
    from ubt.cli.workbench.controller import WorkbenchController
    from ubt.cli.workbench.models import WorkbenchMode
    from ubt.cli.workbench.runner import run_workbench
    from ubt.core.config import UBTConfig
    from ubt.core.engine.job_queue import JobQueue
    from ubt.core.job_options import apply_config_overrides

    resolve_fn = resolve_db_dir()
    db_dir_resolved = resolve_fn(db_dir)
    worker_config = apply_config_overrides(UBTConfig.from_env(), {"db_dir": db_dir_resolved})

    queue_path = queue_db or worker_config.job_queue_path or (db_dir_resolved / "job_queue.sqlite")
    queue = JobQueue(
        queue_path,
        global_max_running=worker_config.job_max_running,
        default_tenant_max_running=worker_config.job_tenant_max_running,
        max_queued=worker_config.job_max_queued,
    )

    controller = WorkbenchController(
        mode=WorkbenchMode.QUEUE_MONITOR if not job_id else WorkbenchMode.SINGLE_JOB,
        concurrency=concurrency,
    )
    if job_id:
        controller.state.active_job_id = job_id
        controller.state.active_job_title = f"Inspecting Job {job_id}"

    # Asynchronous monitor task
    async def _monitor_loop(ctrl: WorkbenchController) -> None:
        while not ctrl.is_cancelled:
            await ctrl.wait_if_paused()
            await asyncio.sleep(0.5)

    try:
        asyncio.run(
            run_workbench(
                controller=controller,
                task_fn=_monitor_loop,
                queue=queue,
                refresh_rate=refresh_rate,
            )
        )
    except KeyboardInterrupt:
        pass
    finally:
        queue.close()
