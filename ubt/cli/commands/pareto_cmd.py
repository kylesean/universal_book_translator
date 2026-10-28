"""``ubt pareto`` -- Automated multi-model quality vs. cost Pareto frontier evaluation and recommendation.

Evaluates contemporary and custom LLM configurations across quality (MTQE, terminology fidelity),
unit cost (USD per 10,000 words based on prices.toml), and throughput speed.
Identifies the non-dominated Pareto frontier and outputs tailored recommendations:
- Sweet Spot (Best Value / Knee Point)
- Ultra-Budget (Lowest cost clearing quality floor)
- Flagship (Uncompromised highest quality)
- Speed Champion (Fastest throughput)
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from ubt.core.benchmark.pareto import simulate_model_pareto
from ubt.core.benchmark.pareto_reporters import (
    render_pareto_html_dashboard,
    render_pareto_json,
    render_pareto_markdown,
    render_pareto_terminal,
)

console = Console()


def pareto_command(
    models: Annotated[
        str | None,
        typer.Option(
            "--models",
            "-m",
            help="Comma-separated model names to evaluate (e.g. 'deepseek-chat,claude-3-5-sonnet,gpt-4o,gemini-2.5-flash'). Unset uses default contemporary roster.",
        ),
    ] = None,
    min_qe: Annotated[
        float,
        typer.Option(
            "--min-qe",
            help="Minimum acceptable quality threshold (MTQE score 0.0 ~ 1.0)",
        ),
    ] = 0.75,
    budget: Annotated[
        float | None,
        typer.Option(
            "--budget",
            "-b",
            help="Optional target budget limit (USD) to bound selection",
        ),
    ] = None,
    sample_words: Annotated[
        int,
        typer.Option(
            "--sample-words",
            "-w",
            help="Reference document word count to scale cost estimates",
        ),
    ] = 10000,
    html_out: Annotated[
        Path | None,
        typer.Option(
            "--html-out",
            help="Path to write standalone responsive HTML visual dashboard with SVG curve",
        ),
    ] = None,
    markdown_out: Annotated[
        Path | None,
        typer.Option(
            "--markdown-out",
            help="Path to write GitHub Flavored Markdown Pareto report",
        ),
    ] = None,
    json_out: Annotated[
        Path | None,
        typer.Option(
            "--json-out",
            help="Path to write raw JSON report data",
        ),
    ] = None,
    quiet: Annotated[
        bool,
        typer.Option(
            "--quiet",
            "-q",
            help="Suppress terminal dashboard and write output files only",
        ),
    ] = False,
) -> None:
    """Execute multi-model Pareto frontier analysis and output recommendations."""
    candidate_list = [m.strip() for m in models.split(",") if m.strip()] if models else None

    report = simulate_model_pareto(
        candidate_models=candidate_list,
        sample_words=sample_words,
        min_qe_threshold=min_qe,
        target_budget_usd=budget,
    )

    if not quiet:
        render_pareto_terminal(report)

    if markdown_out:
        md_text = render_pareto_markdown(report)
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(md_text, encoding="utf-8")
        console.print(f"[dim]Markdown report saved to: {markdown_out}[/]")

    if html_out:
        html_text = render_pareto_html_dashboard(report)
        html_out.parent.mkdir(parents=True, exist_ok=True)
        html_out.write_text(html_text, encoding="utf-8")
        console.print(f"[dim]HTML visual dashboard saved to: {html_out}[/]")

    if json_out:
        json_text = render_pareto_json(report)
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json_text, encoding="utf-8")
        console.print(f"[dim]JSON report saved to: {json_out}[/]")


__all__ = ["pareto_command"]
