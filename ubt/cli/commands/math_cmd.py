"""CLI commands for adaptive mathematical formula breaking, operator optimization, and layout audits."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ubt.core.math.breaker import audit_math_blocks, break_equation
from ubt.core.math.models import MathSyntax
from ubt.core.patch.engine import load_document_blocks

console = Console()
math_app = typer.Typer(
    name="math",
    help="Adaptive mathematical formula breaking, operator precedence optimization, and layout health auditing.",
    no_args_is_help=True,
)


@math_app.command(name="inspect")
def math_inspect_command(
    file_or_job: Annotated[
        str,
        typer.Argument(
            help="Document file path or SQLite ledger job ID to inspect.",
        ),
    ],
    max_width: Annotated[
        int,
        typer.Option(
            "--max-width",
            "-w",
            help="Maximum formula character width threshold before flagging (e.g. 40 for double-column, 65 for single-column).",
        ),
    ] = 50,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output structured JSON results.",
        ),
    ] = False,
) -> None:
    """Inspect all display and inline mathematical equations in a document for width and nesting."""
    try:
        resolved_src, blocks, _ = asyncio.run(load_document_blocks(file_or_job))
    except Exception as exc:
        console.print(f"[bold red]Error loading blocks from '{file_or_job}': {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    report = audit_math_blocks(blocks, target_width=max_width)

    if json_output:
        res_data = report.to_dict()
        res_data["source"] = resolved_src
        console.print(json.dumps(res_data, indent=2, ensure_ascii=False), soft_wrap=True)
        return

    table = Table(
        title=f"Mathematical Equation Layout Health ({report.total_equations} equations, Safe Score: {report.compliance_score}%)",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("ID", style="cyan", width=12)
    table.add_column("Length", justify="right", style="white", width=8)
    table.add_column("Threshold", justify="right", style="dim", width=10)
    table.add_column("Status", width=14)
    table.add_column("Suggested Breaks", justify="right", style="yellow", width=16)
    table.add_column("Preview", style="dim", overflow="fold")

    for audit in report.audits[:30]:
        if audit.exceeds_threshold:
            status = "[bold red]⚠ OVERSIZED[/bold red]"
        elif audit.is_multiline:
            status = "[green]✓ Multiline[/green]"
        else:
            status = "[blue]✓ Safe[/blue]"

        table.add_row(
            audit.equation_id,
            str(audit.char_length),
            f"≤ {audit.threshold}",
            status,
            str(audit.recommended_breaks_count),
            audit.raw_text[:45] + ("..." if len(audit.raw_text) > 45 else ""),
        )

    console.print(table)


@math_app.command(name="break")
def math_break_command(
    expr_or_file: Annotated[
        str,
        typer.Argument(
            help="Mathematical expression string or path to a file containing formula.",
        ),
    ],
    max_width: Annotated[
        int,
        typer.Option(
            "--max-width",
            "-w",
            help="Maximum target line character width before breaking.",
        ),
    ] = 45,
    syntax_format: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Mathematical syntax: 'typst' or 'latex'.",
        ),
    ] = "typst",
    align: Annotated[
        bool,
        typer.Option(
            "--align/--no-align",
            help="Inject alignment anchors ('&') for relations and operators.",
        ),
    ] = True,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Optional file path to write broken formula.",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output structured JSON results.",
        ),
    ] = False,
) -> None:
    """Break a long mathematical equation across lines using operator precedence."""
    # Check if input is an existing file
    p = Path(expr_or_file)
    formula_text = p.read_text(encoding="utf-8") if p.exists() and p.is_file() else expr_or_file

    syn = MathSyntax.LATEX if syntax_format.lower().strip() == "latex" else MathSyntax.TYPST
    broken = break_equation(formula_text, target_width=max_width, syntax=syn, align=align)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(broken, encoding="utf-8")
        if json_output:
            console.print(
                json.dumps(
                    {
                        "original_length": len(formula_text),
                        "output_file": str(output),
                        "syntax": str(syn),
                        "align": align,
                    },
                    indent=2,
                )
            )
        else:
            console.print(f"[bold green]Successfully saved broken formula to:[/] {output}")
        return

    if json_output:
        console.print(
            json.dumps(
                {
                    "original": formula_text,
                    "original_length": len(formula_text),
                    "broken": broken,
                    "lines_count": len(broken.splitlines()),
                    "syntax": str(syn),
                    "align": align,
                },
                indent=2,
                ensure_ascii=False,
            ),
            soft_wrap=True,
        )
    else:
        console.print(broken)


@math_app.command(name="audit")
def math_audit_command(
    file_or_job: Annotated[
        str,
        typer.Argument(
            help="Document file path or SQLite ledger job ID to audit.",
        ),
    ],
    max_width: Annotated[
        int,
        typer.Option(
            "--max-width",
            "-w",
            help="Maximum formula character width threshold before flagging.",
        ),
    ] = 50,
    min_score: Annotated[
        float,
        typer.Option(
            "--min-score",
            "-s",
            help="Minimum compliance score percentage required to pass gate.",
        ),
    ] = 90.0,
    strict: Annotated[
        bool,
        typer.Option(
            "--strict",
            help="Fail closed if any equation exceeds max-width.",
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output structured JSON results.",
        ),
    ] = False,
) -> None:
    """Run publication CI/CD layout gate audit for mathematical equations."""
    try:
        resolved_src, blocks, _ = asyncio.run(load_document_blocks(file_or_job))
    except Exception as exc:
        console.print(f"[bold red]Error loading blocks from '{file_or_job}': {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    report = audit_math_blocks(blocks, target_width=max_width)

    score_ok = report.compliance_score >= min_score
    strict_ok = not (strict and report.oversized_equations > 0)
    passed = score_ok and strict_ok

    if json_output:
        res_data = report.to_dict()
        res_data["source"] = resolved_src
        res_data["passed"] = passed
        res_data["strict"] = strict
        console.print(json.dumps(res_data, indent=2, ensure_ascii=False), soft_wrap=True)
        if not passed:
            raise typer.Exit(code=1)
        return

    score_color = "green" if passed else "red"
    summary_panel = Panel(
        f"[bold cyan]Document/Job:[/] {resolved_src}\n"
        f"[bold cyan]Total Equations:[/] {report.total_equations}\n"
        f"[bold cyan]Oversized Formulas:[/] [{score_color}]{report.oversized_equations}[/{score_color}]\n"
        f"[bold cyan]Multiline Formulas:[/] {report.multiline_equations}\n"
        f"[bold cyan]Compliance Score:[/] [{score_color}]{report.compliance_score}%[/{score_color}] "
        f"(Target: ≥ {min_score}%)\n"
        f"[bold cyan]Strict Mode:[/] {'ON' if strict else 'OFF'}\n\n"
        f"[bold {score_color}]VERDICT: {'PASSED' if passed else 'FAILED'}[/bold {score_color}]",
        title="[bold blue]Math Layout Quality Gate Audit[/bold blue]",
    )
    console.print(summary_panel)

    if not passed:
        raise typer.Exit(code=1)
