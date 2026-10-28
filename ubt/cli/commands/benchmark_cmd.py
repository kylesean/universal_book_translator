"""``ubt benchmark`` -- automated corpus benchmark evaluation and drift dashboard.

Executes and evaluates cases defined in ``corpus/cases.json`` across delivery contract,
physical layout spills, translation quality, and financial/token costs.
Provides multi-format reporting (Terminal, Markdown, HTML, JSON) and baseline drift comparison.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from ubt.core.benchmark.drift import compute_benchmark_drift
from ubt.core.benchmark.models import BenchmarkDriftReport, BenchmarkMode, CorpusBenchmarkReport
from ubt.core.benchmark.pareto import simulate_model_pareto
from ubt.core.benchmark.pareto_reporters import render_pareto_terminal
from ubt.core.benchmark.reporters import (
    render_html_dashboard,
    render_json_report,
    render_markdown_report,
    render_terminal_dashboard,
)
from ubt.core.benchmark.runner import BenchmarkRunner

console = Console()


def benchmark_command(
    corpus: Annotated[
        Path,
        typer.Option(
            "--corpus",
            help="Directory containing cases.json manifest",
        ),
    ] = Path("corpus"),
    cases: Annotated[
        list[str] | None,
        typer.Option(
            "--case",
            "-c",
            help="Specific case ID(s) to evaluate (repeatable or omit for all)",
        ),
    ] = None,
    mode: Annotated[
        str,
        typer.Option(
            "--mode",
            "-m",
            help="Execution mode: 'dry-run' (mock tokens), 'live' (real model), or 'eval-only'",
        ),
    ] = "dry-run",
    artifacts_dir: Annotated[
        Path | None,
        typer.Option(
            "--artifacts-dir",
            "-a",
            help="Directory containing pre-built delivered artifacts (for eval-only or output)",
        ),
    ] = None,
    baseline: Annotated[
        Path | None,
        typer.Option(
            "--baseline",
            "-b",
            help="Path to baseline JSON report to calculate metric drift against",
        ),
    ] = None,
    save_baseline: Annotated[
        Path | None,
        typer.Option(
            "--save-baseline",
            help="Save current benchmark report as new baseline to this JSON path",
        ),
    ] = None,
    tolerance_pt: Annotated[
        float,
        typer.Option(
            "--tolerance-pt",
            help="Physical layout spill tolerance margin in points",
        ),
    ] = 5.0,
    html_out: Annotated[
        Path | None,
        typer.Option(
            "--html-out",
            help="Path to write standalone responsive HTML visual dashboard",
        ),
    ] = None,
    markdown_out: Annotated[
        Path | None,
        typer.Option(
            "--markdown-out",
            help="Path to write GitHub Flavored Markdown benchmark report",
        ),
    ] = None,
    json_out: Annotated[
        Path | None,
        typer.Option(
            "--json-out",
            help="Path to write structured JSON report",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output machine-readable JSON to stdout",
        ),
    ] = False,
    pareto: Annotated[
        bool,
        typer.Option(
            "--pareto",
            help="Also compute and display multi-model Pareto frontier analysis",
        ),
    ] = False,
) -> None:
    """Run automated multi-scenario corpus benchmark and generate quality/cost/spill dashboards."""
    if not corpus.exists():
        msg = f"Corpus directory not found: {corpus}"
        if json_output:
            print(json.dumps({"status": "error", "error": msg}))
        else:
            console.print(f"[bold red]Error:[/] {msg}")
        raise typer.Exit(code=2)

    valid_modes = {m.value for m in BenchmarkMode}
    if mode not in valid_modes:
        msg = f"Invalid mode '{mode}'. Choose from: {', '.join(sorted(valid_modes))}"
        if json_output:
            print(json.dumps({"status": "error", "error": msg}))
        else:
            console.print(f"[bold red]Error:[/] {msg}")
        raise typer.Exit(code=2)

    runner = BenchmarkRunner(
        corpus_dir=corpus,
        mode=mode,
        artifacts_dir=artifacts_dir,
        tolerance_pt=tolerance_pt,
    )

    try:
        report = runner.run_all(case_ids=cases)
    except Exception as exc:
        msg = f"Benchmark execution failed: {exc}"
        if json_output:
            print(json.dumps({"status": "error", "error": msg}))
        else:
            console.print(f"[bold red]Error:[/] {msg}")
        raise typer.Exit(code=2) from exc

    # Drift analysis if baseline provided
    drift: BenchmarkDriftReport | None = None
    if baseline:
        if not baseline.exists():
            msg = f"Baseline file not found: {baseline}"
            if json_output:
                print(json.dumps({"status": "error", "error": msg}))
            else:
                console.print(f"[bold red]Error:[/] {msg}")
            raise typer.Exit(code=2)
        try:
            base_report = CorpusBenchmarkReport.from_json(baseline.read_text(encoding="utf-8"))
            drift = compute_benchmark_drift(baseline=base_report, candidate=report)
        except Exception as exc:
            msg = f"Failed to compute baseline drift: {exc}"
            if json_output:
                print(json.dumps({"status": "error", "error": msg}))
            else:
                console.print(f"[bold red]Error:[/] {msg}")
            raise typer.Exit(code=2) from exc

    # Save baseline if requested
    if save_baseline:
        try:
            report.save(save_baseline)
            if not json_output:
                console.print(f"[bold green]Baseline saved to:[/] {save_baseline}")
        except Exception as exc:
            console.print(f"[bold red]Failed to save baseline:[/] {exc}")

    # Output exports
    if html_out:
        html_content = render_html_dashboard(report, drift=drift)
        html_out.parent.mkdir(parents=True, exist_ok=True)
        html_out.write_text(html_content, encoding="utf-8")
        if not json_output:
            console.print(f"[bold green]HTML dashboard written to:[/] {html_out}")

    if markdown_out:
        md_content = render_markdown_report(report, drift=drift)
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(md_content, encoding="utf-8")
        if not json_output:
            console.print(f"[bold green]Markdown report written to:[/] {markdown_out}")

    if json_out:
        json_content = render_json_report(report, drift=drift)
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json_content, encoding="utf-8")
        if not json_output:
            console.print(f"[bold green]JSON report written to:[/] {json_out}")

    # Terminal or JSON output
    if json_output:
        print(render_json_report(report, drift=drift))
    else:
        render_terminal_dashboard(report, drift=drift, console=console)
        if pareto:
            pareto_report = simulate_model_pareto()
            render_pareto_terminal(pareto_report, console=console)

    # Determine exit code
    success = report.overall_passed
    if drift and not drift.drift_passed:
        success = False

    raise typer.Exit(code=0 if success else 1)
