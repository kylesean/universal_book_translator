"""Industrial Universal Book Translator CLI application powered by Typer and Rich."""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from ubt import __version__
from ubt.cli.commands.assess import _assess_money, assess_cmd
from ubt.cli.commands.config_cmd import config_command
from ubt.cli.commands.doctor import doctor_command
from ubt.cli.commands.status import inspect_book, job_status, pe_import, recheck_gates_cmd
from ubt.cli.commands.translate import console as console
from ubt.cli.commands.translate import translate
from ubt.cli.commands.verify import verify_command
from ubt.cli.commands.worker import worker_command
from ubt.core.config import (
    MOCK_API_KEY,
    RIGID_ENGINES,
    UBTConfig,
)
from ubt.core.engine.dry_run import create_dry_run_orchestrator
from ubt.core.engine.events import EventType, TranslationProgressEvent
from ubt.core.engine.pipeline import PipelineOrchestrator
from ubt.core.exceptions import UBTError
from ubt.core.job_options import (
    JOB_ID_MAX_LEN,
    apply_config_overrides,
    job_id_is_valid,
    overrides_from_request,
    run_kwargs_from_request,
)
from ubt.core.job_options import (
    LANG_CODE_RE as LANG_CODE_RE,
)
from ubt.core.language_profile import is_supported_lang, supported_lang_codes
from ubt.core.log_config import setup_logging
from ubt.core.metrics import (
    KPI_DEFINITIONS,
    SCHEMA_VERSION,
    compare_kpi_sets,
    load_kpis,
)

UserRenderEngine = Literal["rigid", "reflow", "auto"]

app = typer.Typer(
    name="ubt",
    help="Industrial-grade Universal Bilingual Book Translation Engine.",
    no_args_is_help=True,
)

# Advisories and errors that would otherwise prepend non-JSON text to a
# ``--json`` command's stdout. ``console`` stays the stdout channel for human
# output; this one carries the machine-readable mode's diagnostics instead.
_err_console = Console(stderr=True)


@app.callback()
def main_callback(
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable verbose DEBUG logging to stderr.",
        ),
    ] = False,
) -> None:
    """Industrial-grade Universal Bilingual Book Translation Engine."""
    # Diagnostics go to stderr by default, so stdout stays clean for the
    # machine-readable `--json` output of every subcommand. Commands that draw a
    # Rich Live progress bar (translate) re-route records onto the shared stdout
    # console just before the bar starts, which is the only way a log line and
    # the bar can share one terminal region without tearing.
    setup_logging(verbose=verbose)


def _build_config(overrides: dict[str, Any]) -> UBTConfig:
    """Load the env config and apply CLI overrides through pydantic validation.

    Uses the shared request→config mapping so the CLI, REST API and MCP server
    agree on which option names are config fields.
    """
    return apply_config_overrides(UBTConfig.from_env(), overrides)


def _resolve_db_dir(db_dir: Path | None) -> Path:
    """An unset --db-dir follows the config chain (ubt.toml / UBT_DB_DIR).

    Injecting a typer literal default like ``Path(".ubt/ledgers")`` here would
    act as an explicit override, silently hiding ledgers kept elsewhere and
    making commands report "no such job" against a healthy ledger.
    """
    if db_dir is not None:
        return db_dir
    return UBTConfig.from_env().db_dir


def resolve_cli_adaptive_dual_mode(
    explicit_dual_mode: str | None,
    profile: str | None,
    render_engine: str | None,
) -> str | None:
    """Resolve smart profile-aware and engine-aware dual_mode default at the CLI layer.

    When --dual-mode is explicitly passed by the user, that choice is preserved.
    When unset (None), derives the smart default:
    - Academic papers ('paper') default to 'monolingual' (standard academic publication format);
    - Novels / fiction ('fiction', 'novel') default to 'monolingual' (continuous prose reading);
    - Rigid engine ('rigid', 'inplace') defaults to 'monolingual' (rigid is strictly monolingual-only);
    - Other combinations (e.g. 'general', 'textbook', 'humanities' on reflow) leave it
      unset to follow config / UBT_DUAL_MODE / 'inline'.
    """
    # Single-sourced in ``ubt.core.job_options`` so API/MCP resolve the same
    # default (see ``overrides_from_request``).
    from ubt.core.job_options import adaptive_dual_mode

    return adaptive_dual_mode(explicit_dual_mode, profile, render_engine)


async def _run_translation(
    input_path: Path,
    output_path: Path | None,
    dry_run: bool = False,
    quiet: bool = False,
    **request: Any,
) -> Path:
    """Async execution wrapper with rich interactive dual-dashboard.

    The keyword surface is the shared job-request mapping
    (:func:`ubt.core.job_options.overrides_from_request`) that API and
    MCP already submit through: request keys are UBTConfig field names
    (plus ``glossary`` -> ``glossary_path``), unset flags arrive as ``None``
    and are skipped so ``UBT_*`` env keeps precedence. A hand-mirrored
    48-parameter copy used to live here; the request keys drifted exactly
    once per entry point per quarter, so no more lists to keep in sync.
    """
    job_id = request.get("job_id")
    if job_id is not None and not job_id_is_valid(job_id):
        raise UBTError(
            f"Invalid --job-id {job_id!r}: use letters, digits, '-' or '_' "
            f"(up to {JOB_ID_MAX_LEN} characters; the shared rule in "
            "ubt.core.job_options, same as API/MCP)"
        )
    # Language codes reach the ledger file name (derive_job_id) and, after
    # sanitizing, Typst markup; the API and MCP surfaces already reject
    # anything non-ISO-ish up front, the CLI was the lone pass-through.
    for key in ("source_lang", "target_lang"):
        value = request.get(key)
        if value is not None and LANG_CODE_RE.fullmatch(str(value)) is None:
            raise UBTError(
                f"Invalid {key.replace('_', '-')} {value!r}: use an ISO-ish "
                "language code such as 'en', 'zh' or 'zh-CN' (the shared "
                "rule in ubt.core.job_options, same as API/MCP)"
            )
    target_lang_value = request.get("target_lang")
    if target_lang_value is not None and not is_supported_lang(str(target_lang_value)):
        raise UBTError(
            f"Unsupported target-lang {target_lang_value!r}: supported base languages "
            f"are {', '.join(supported_lang_codes())} (region tags such as 'zh-CN' are accepted)"
        )
    overrides: dict[str, Any] = overrides_from_request(request)

    # ``overrides_from_request`` already applied the shared adaptive default, so
    # the CLI no longer derives dual_mode here (that was the only place it did).
    explicit_dual = request.get("dual_mode")
    effective_dual = overrides.get("dual_mode") or explicit_dual
    # Do NOT mirror draft_model→repair_model here. UBTConfig already syncs
    # repair_model←draft_model *only when repair was not explicitly set*
    # (config.py `_check_invariants`); pre-setting it here would push a value
    # into the overrides and let `--draft-model` silently override an explicit
    # `UBT_REPAIR_MODEL`, diverging the CLI from the API/MCP path.
    if request.get("facing_spread") is None and effective_dual == "facing":
        overrides["facing_spread"] = True

    # Call the module-global directly: a `monkeypatch.setattr(ubt.cli.main,
    # "_build_config", …)` rebinds this same global at call time, so the old
    # `sys.modules.get(...)._build_config` lookup was pure indirection.
    config = _build_config(overrides)

    if (
        explicit_dual is not None
        and config.render_engine in RIGID_ENGINES
        and config.dual_mode != "monolingual"
    ):
        warning_console = _err_console if quiet else console
        warning_console.print(
            f"[yellow]Warning:[/] --render-engine rigid is monolingual-only: the requested "
            f"--dual-mode '{config.dual_mode}' will be downgraded to 'monolingual'. "
            "Use --render-engine reflow for a bilingual artifact."
        )

    if not dry_run:
        api_key_missing = not config.api_key.get_secret_value() or (
            config.api_key.get_secret_value() == MOCK_API_KEY
        )
    else:
        api_key_missing = False
    if api_key_missing:
        raise RuntimeError(
            "No LLM credential is configured (empty or mock-key). Set "
            "UBT_LLM_API_KEY, configure api_key in ubt.toml, or pass --api-key; "
            "run with --dry-run for zero-token validation."
        )

    if dry_run:
        orchestrator = create_dry_run_orchestrator(config)
    else:
        orchestrator = PipelineOrchestrator(config=config)

    async def _drain() -> Path | None:
        final_output: Path | None = None
        async for event in orchestrator.run(
            input_path=input_path,
            output_path=output_path,
            **run_kwargs_from_request(request),
        ):
            _note_progress(progress, task_id, event)
            if event.event_type == EventType.EXPORT_COMPLETED:
                final_output = Path(event.artifact_path or event.message)
        return final_output

    if quiet:
        progress: Progress | None = None
        task_id: object = None
        final_output = await _drain()
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task(
                "[yellow]Simulating Pipeline (Dry-Run)..."
                if dry_run
                else "[cyan]Initializing Translation Pipeline...",
                total=100,
            )
            final_output = await _drain()

    if final_output is None:
        raise UBTError("Pipeline failed to complete document rendering.")

    return final_output


def _note_progress(
    progress: Progress | None, task_id: Any, event: TranslationProgressEvent
) -> None:
    """Mirror one pipeline event onto the Rich progress bar (no-op in quiet mode)."""
    if progress is None:
        return
    # Terminal-state numerator (matches ProgressSnapshot.processed_blocks): counting
    # only terminal blocks ensures the progress bar strictly advances monotonically.
    done = min(
        event.total_blocks,
        event.completed_blocks
        + event.failed_blocks
        + event.needs_human_blocks
        + event.blocked_human_blocks,
    )
    if event.total_blocks > 0:
        # The numerator counts blocks that reached *a terminal status*, which
        # happens the moment drafting ends — while repair, render and export
        # are still running. Showing 100% then tells the operator to stop
        # watching a job that is still doing work, so park one short of the
        # total until EXPORT_COMPLETED proves the artifact actually exists.
        bar_done = (
            done
            if event.event_type is EventType.EXPORT_COMPLETED
            else min(done, event.total_blocks - 1)
        )
        progress.update(
            task_id,
            total=event.total_blocks,
            completed=bar_done,
            description=f"[bold green]{event.event_type.value.upper()}[/] [dim]({done}/{event.total_blocks})[/]",
        )

    now_str = datetime.now().strftime("%H:%M:%S")
    progress.console.print(
        f"[dim]{now_str}[/] [bold green]{event.event_type.value.upper()}[/] "
        f"[{done}/{event.total_blocks if event.total_blocks > 0 else '?'}] "
        f"{event.message}"
    )


def _strict_failures(report_path: Path) -> list[str]:
    """Human-readable delivery-integrity failures from a quality report.

    An unreadable or malformed report is itself a failure. ``--strict`` exists
    to refuse shipping when integrity cannot be verified, and returning ``[]``
    here made the gate pass in exactly the case it must not: a crash after
    export, a report path mismatch, or a truncated write — all of which leave
    the run green while nothing about the artifact was actually checked.
    """
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except OSError as exc:
        return [f"quality report unreadable ({report_path.name}): {exc}"]
    except ValueError as exc:
        return [f"quality report is not valid JSON ({report_path.name}): {exc}"]
    if not isinstance(report, dict):
        return [f"quality report has an unexpected shape ({report_path.name})"]
    summary = report.get("summary", {})
    coverage = report.get("render_coverage", {})
    if not isinstance(summary, dict) or not isinstance(coverage, dict):
        return [f"quality report is missing summary/render_coverage ({report_path.name})"]

    failures: list[str] = []
    for key, label in (
        ("failed_blocks", "block(s) failed"),
        ("needs_human_blocks", "block(s) waiting for human review"),
        ("blocked_human_blocks", "block(s) blocked from shipping"),
    ):
        if int(summary.get(key, 0) or 0) > 0:
            failures.append(f"{int(summary.get(key, 0) or 0)} {label}")
    skipped = int(coverage.get("skipped_blocks", 0) or 0)
    if skipped > 0:
        failures.append(f"{skipped} fail-closed render skip(s) left source text on the page")
    return failures


@app.command(name="version")
def version() -> None:
    """Print Universal Book Translator engine version and runtime environment."""
    # Imported here, not at module scope: `ubt --help` / `ubt version` would
    # otherwise pay the whole adapter import graph (docling, pdf_oxide, pikepdf
    # — ~260 ms) before Typer parses a single argument.
    from ubt.adapters import supported_suffixes

    console.print(
        f"[bold cyan]Universal Book Translator (UBT)[/] version [bold green]{__version__}[/]"
    )
    console.print(f"Python: {sys.version.split()[0]} on {sys.platform}")
    console.print(f"Input formats: {', '.join(supported_suffixes())}")


metrics_app = typer.Typer(
    name="metrics",
    help="Versioned KPI artifacts: definitions, per-run inspection, regression gate.",
    no_args_is_help=True,
)
app.add_typer(metrics_app, name="metrics")


@metrics_app.command(name="definitions")
def metrics_definitions() -> None:
    """List the KPI registry (definition, unit, direction, regression tolerance)."""
    table = Table(title=f"KPI definitions (schema v{SCHEMA_VERSION})")
    table.add_column("KPI", style="bold cyan")
    table.add_column("Unit")
    table.add_column("Direction")
    table.add_column("Tol", justify="right")
    table.add_column("Definition")
    for definition in KPI_DEFINITIONS:
        table.add_row(
            definition.name,
            definition.unit,
            definition.direction.value,
            f"{definition.tolerance:g}",
            definition.definition,
        )
    console.print(table)


@metrics_app.command(name="show")
def metrics_show(
    metrics_path: Annotated[Path, typer.Argument(help="Path to a *_metrics.json artifact")],
    json_output: Annotated[bool, typer.Option("--json", help="Emit the KPI set as JSON")] = False,
) -> None:
    """Show the KPIs of one finished run."""
    if not metrics_path.exists():
        _err_console.print(f"[bold red]Error:[/] metrics artifact not found: {metrics_path}")
        raise typer.Exit(code=1)
    kpis = load_kpis(metrics_path)
    if json_output:
        print(kpis.model_dump_json())
        return
    title = f"KPIs: {kpis.job.get('job_id', metrics_path.stem)} (schema v{kpis.schema_version})"
    table = Table(title=title)
    table.add_column("KPI", style="bold cyan")
    table.add_column("Value", justify="right")
    table.add_column("Direction")
    for definition in KPI_DEFINITIONS:
        if definition.name in kpis.kpis:
            table.add_row(
                definition.name, f"{kpis.kpis[definition.name]:.4f}", definition.direction.value
            )
    console.print(table)


@metrics_app.command(name="compare")
def metrics_compare(
    baseline: Annotated[Path, typer.Argument(help="Baseline *_metrics.json (golden)")],
    candidate: Annotated[Path, typer.Argument(help="Candidate *_metrics.json")],
    fail_on_regression: Annotated[
        bool,
        typer.Option("--fail-on-regression", help="Exit non-zero on any KPI regression"),
    ] = False,
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit the regression report as JSON")
    ] = False,
    strict_names: Annotated[
        bool,
        typer.Option(
            "--strict-names",
            help="Fail on a KPI the baseline does not carry, instead of skipping it",
        ),
    ] = False,
) -> None:
    """Compare a candidate run against a baseline, direction- and tolerance-aware."""
    for path in (baseline, candidate):
        if not path.exists():
            _err_console.print(f"[bold red]Error:[/] metrics artifact not found: {path}")
            raise typer.Exit(code=1)
    report = compare_kpi_sets(load_kpis(baseline), load_kpis(candidate), strict_names=strict_names)
    if json_output:
        print(report.model_dump_json())
    else:
        table = Table(title="KPI regression compare")
        table.add_column("KPI", style="bold cyan")
        table.add_column("Baseline", justify="right")
        table.add_column("Candidate", justify="right")
        table.add_column("Delta", justify="right")
        table.add_column("Status")
        for delta in report.deltas:
            status = "[bold red]REGRESSED[/]" if delta.regressed else "[green]ok[/]"
            table.add_row(
                delta.name,
                f"{delta.baseline:.4f}",
                f"{delta.candidate:.4f}",
                f"{delta.delta:+.4f}",
                status,
            )
        console.print(table)
        for violation in report.violations:
            console.print(f"[bold red]regression:[/] {violation}")
        if report.passed:
            console.print("[bold green]✓ No KPI regression.[/]")
    if fail_on_regression and not report.passed:
        raise typer.Exit(code=1)


tm_app = typer.Typer(
    name="tm",
    help="Translation-memory audit: list reusable entries and evict poisoned ones.",
    no_args_is_help=True,
)
app.add_typer(tm_app, name="tm")


@tm_app.command(name="scan")
def tm_scan(
    db_dir: Annotated[
        Path | None,
        typer.Option("--db-dir", help="Directory holding tm.sqlite (default: config/UBT_DB_DIR)"),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit entries as JSON")] = False,
    limit: Annotated[int, typer.Option("--limit", help="Max rows to show (0 = all)")] = 50,
) -> None:
    """List reusable translation-memory entries (id, pair, provenance, reuse)."""
    from ubt.core.memory.tm import TranslationMemory

    tm_path = _resolve_db_dir(db_dir) / "tm.sqlite"
    if not tm_path.exists():
        _err_console.print(f"[bold red]Error:[/] translation memory not found: {tm_path}")
        raise typer.Exit(code=1)
    tm = TranslationMemory(tm_path)
    try:
        entries = tm.scan()
    finally:
        tm.close()
    shown = entries if limit <= 0 else entries[:limit]
    if json_output:
        print(
            json.dumps(
                [
                    {
                        "id": e.id,
                        "src_lang": e.src_lang,
                        "tgt_lang": e.tgt_lang,
                        "provenance": e.provenance,
                        "domain": e.domain,
                        "use_count": e.use_count,
                        "source_text": e.source_text,
                        "target_text": e.target_text,
                    }
                    for e in shown
                ],
                ensure_ascii=False,
            )
        )
        return
    table = Table(title=f"Translation memory: {len(entries)} entries")
    table.add_column("id", justify="right", style="bold cyan")
    table.add_column("pair")
    table.add_column("provenance")
    table.add_column("reuse", justify="right")
    table.add_column("source", overflow="fold")
    for e in shown:
        preview = e.source_text.replace("\n", " ")[:60]
        table.add_row(
            str(e.id), f"{e.src_lang}->{e.tgt_lang}", e.provenance, str(e.use_count), preview
        )
    console.print(table)
    if limit > 0 and len(entries) > limit:
        console.print(f"[dim]… {len(entries) - limit} more; use --limit 0 for all.[/]")


@tm_app.command(name="evict")
def tm_evict(
    ids: Annotated[list[int], typer.Argument(help="Translation-memory entry ids to delete")],
    db_dir: Annotated[
        Path | None,
        typer.Option("--db-dir", help="Directory holding tm.sqlite (default: config/UBT_DB_DIR)"),
    ] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the confirmation prompt")] = False,
) -> None:
    """Delete poisoned/stale translation-memory entries by id.

    A poisoned entry is worse than a missing one: ``lookup_exact`` serves it
    verbatim on every later run, so one bad generation becomes the permanent,
    authoritative translation.
    """
    from ubt.core.memory.tm import TranslationMemory

    if not ids:
        _err_console.print("[bold red]Error:[/] no entry ids given")
        raise typer.Exit(code=1)
    tm_path = _resolve_db_dir(db_dir) / "tm.sqlite"
    if not tm_path.exists():
        _err_console.print(f"[bold red]Error:[/] translation memory not found: {tm_path}")
        raise typer.Exit(code=1)
    if not yes:
        typer.confirm(f"Delete {len(set(ids))} translation-memory entr(y/ies)?", abort=True)
    tm = TranslationMemory(tm_path)
    try:
        removed = tm.evict_ids(ids)
    finally:
        tm.close()
    console.print(f"[bold green]✓ Evicted {removed} entr(y/ies).[/]")


# Register modular subcommands
@app.command(name="api")
def api_command(
    host: Annotated[
        str | None, typer.Option("--host", help="Bind host (default 127.0.0.1)")
    ] = None,
    port: Annotated[int | None, typer.Option("--port", help="Bind port (default 8000)")] = None,
) -> None:
    """Start the REST API server (same as the ``ubt-api`` entry point)."""
    from ubt.api.app import run_server

    run_server(host=host, port=port)


app.command(name="translate")(translate)
app.command(name="inspect")(inspect_book)
app.command(name="assess")(assess_cmd)
app.command(name="status")(job_status)
app.command(name="recheck-gates")(recheck_gates_cmd)
app.command(name="pe-import")(pe_import)
app.command(name="doctor")(doctor_command)
app.command(name="worker")(worker_command)
app.command(name="config")(config_command)
app.command(name="verify")(verify_command)

__all__ = [
    "app",
    "translate",
    "inspect_book",
    "assess_cmd",
    "_assess_money",
    "job_status",
    "pe_import",
    "doctor_command",
    "worker_command",
    "config_command",
    "verify_command",
    "version",
    "_build_config",
    "_resolve_db_dir",
    "_strict_failures",
    "_note_progress",
    "_run_translation",
    "resolve_cli_adaptive_dual_mode",
]

if __name__ == "__main__":
    app()
