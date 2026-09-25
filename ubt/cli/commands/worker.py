"""CLI command for running persistent background job workers."""

import asyncio
import os
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console

console = Console()


def _get_resolve_db_dir() -> Any:
    from ubt.cli.main import _resolve_db_dir

    return _resolve_db_dir


def worker_command(
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
        int, typer.Option("--concurrency", "-c", help="Concurrent jobs this worker process runs")
    ] = 1,
    once: Annotated[
        bool,
        typer.Option("--once", help="Drain queued jobs and exit (batch/CI); default polls forever"),
    ] = False,
    poll_interval: Annotated[
        float, typer.Option("--poll-interval", help="Idle poll interval in seconds")
    ] = 2.0,
    lease_seconds: Annotated[
        float, typer.Option("--lease-seconds", help="Job lease length, renewed by heartbeat")
    ] = 60.0,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Verbose DEBUG logging")] = False,
) -> None:
    """Run a worker that drains the durable service job queue.

    Enqueue with the REST API (or an SDK/MCP client); a worker survives restarts
    and reclaims jobs whose lease expired. ``--once`` drains and exits.
    """
    from ubt.core.config import UBTConfig
    from ubt.core.engine.job_queue import JobQueue
    from ubt.core.engine.job_worker import JobWorker
    from ubt.core.job_options import apply_config_overrides
    from ubt.core.log_config import setup_logging
    from ubt.core.router.rate_limiter import build_rate_limiter

    resolve_fn = _get_resolve_db_dir()
    db_dir = resolve_fn(db_dir)
    worker_config = apply_config_overrides(UBTConfig.from_env(), {"db_dir": db_dir})

    if verbose:
        setup_logging(verbose=True)
    queue_path = queue_db or worker_config.job_queue_path or (db_dir / "job_queue.sqlite")
    # Same caps the API applies: a worker started with `ubt worker` must not
    # silently run 8 jobs when the operator set UBT_JOB_MAX_RUNNING=1.
    queue = JobQueue(
        queue_path,
        global_max_running=worker_config.job_max_running,
        default_tenant_max_running=worker_config.job_tenant_max_running,
        max_queued=worker_config.job_max_queued,
    )
    worker = JobWorker(
        queue,
        worker_config,
        worker_id=f"worker-{os.getpid()}",
        concurrency=concurrency,
        poll_interval=poll_interval,
        lease_seconds=lease_seconds,
        rate_limiter=build_rate_limiter(
            worker_config,
            shared_path=queue_path.with_name("rate_limiter.sqlite"),
        ),
    )
    try:
        if once:
            processed = asyncio.run(worker.run_until_idle())
            if worker.failed_jobs > 0:
                console.print(
                    f"[red]Drained {processed} job(s) from {queue_path} "
                    f"({worker.failed_jobs} failed).[/]"
                )
                raise typer.Exit(code=1)
            console.print(f"[green]Drained {processed} job(s) from {queue_path}.[/]")
        else:
            console.print(
                f"[cyan]Worker {worker.worker_id} polling {queue_path} "
                f"(concurrency={concurrency}). Ctrl-C to stop.[/]"
            )
            asyncio.run(worker.run_forever())
    except KeyboardInterrupt:
        worker.stop()
    finally:
        queue.close()
        limiter = worker.rate_limiter
        closer = getattr(limiter, "close", None)
        if callable(closer):
            closer()
