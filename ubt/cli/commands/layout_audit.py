"""CLI command for Typst physical layout spill and overflow diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from ubt.render.typst.layout_diagnostics import (
    DEFAULT_TOLERANCE_PT,
    PageLayoutSpec,
    diagnose_pdf_layout,
    print_rich_layout_report,
)

console = Console()


def _parse_page_range(raw: str | None) -> list[int] | None:
    """Parse page specification strings like '1-5', '1,3,7', '4-' into a sorted list of 1-based page numbers."""
    if not raw or not raw.strip():
        return None
    pages: set[int] = set()
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    for part in parts:
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str.strip()) if start_str.strip() else 1
            if end_str.strip():
                end = int(end_str.strip())
                pages.update(range(start, end + 1))
            else:
                pages.add(start)
        elif part.isdigit():
            pages.add(int(part))
    return sorted(pages) if pages else None


def layout_audit_command(
    pdf_path: Annotated[
        Path,
        typer.Argument(
            help="Path to rendered PDF artifact to audit for layout spills and margin overflows.",
            exists=False,
            dir_okay=False,
        ),
    ],
    template: Annotated[
        str,
        typer.Option(
            "--template",
            "-t",
            help="Document layout template spec ('auto', 'book', 'academic', 'academic-dual').",
        ),
    ] = "auto",
    tolerance: Annotated[
        float,
        typer.Option(
            "--tolerance",
            help="Physical margin overflow tolerance in points (default: 5.0 pt).",
        ),
    ] = DEFAULT_TOLERANCE_PT,
    pages: Annotated[
        str | None,
        typer.Option(
            "--pages",
            "-p",
            help="Subset of 1-based pages to inspect (e.g. '1-10', '1,3,7'). Default: all pages.",
        ),
    ] = None,
    heatmap: Annotated[
        bool,
        typer.Option(
            "--heatmap/--no-heatmap",
            help="Render visual ASCII/Rich 2D spatial layout heatmaps for anomalous pages.",
        ),
    ] = True,
    strict: Annotated[
        bool,
        typer.Option(
            "--strict",
            help="Strict mode: exit with code 1 on layout warnings as well as critical spills.",
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Emit machine-readable structured JSON report to stdout.",
        ),
    ] = False,
) -> None:
    """Audit a PDF document for Typst layout spills, margin leaks, and column overruns."""
    if not pdf_path.exists():
        if json_output:
            typer.echo(
                '{"passed": false, "error": "PDF file does not exist", "pdf_path": "'
                + str(pdf_path)
                + '"}'
            )
        else:
            console.print(f"[bold red]Error:[/] PDF file not found at {pdf_path}")
        raise typer.Exit(code=2)

    if pdf_path.suffix.lower() != ".pdf":
        if json_output:
            typer.echo(
                '{"passed": false, "error": "Not a PDF document", "pdf_path": "'
                + str(pdf_path)
                + '"}'
            )
        else:
            console.print(f"[bold red]Error:[/] File is not a PDF: {pdf_path.name}")
        raise typer.Exit(code=2)

    target_pages = _parse_page_range(pages)

    # Resolve layout specification
    spec: PageLayoutSpec | None = None
    if template.lower() != "auto":
        spec = PageLayoutSpec.from_name(template, tolerance_pt=tolerance)

    try:
        report = diagnose_pdf_layout(
            pdf_path=pdf_path,
            spec=spec,
            pages=target_pages,
            tolerance_pt=tolerance,
        )
    except Exception as exc:
        if json_output:
            typer.echo(
                f'{{"passed": false, "error": "Diagnosis failed: {exc}", "pdf_path": "{pdf_path}"}}'
            )
        else:
            console.print(f"[bold red]Layout diagnosis error:[/] {exc}")
        raise typer.Exit(code=1) from None

    if json_output:
        typer.echo(report.to_json(indent=2))
    else:
        active_spec = spec or PageLayoutSpec.book(tolerance_pt=tolerance)
        print_rich_layout_report(
            report=report,
            spec=active_spec,
            console=console,
            show_heatmap=heatmap,
        )

    # Determine exit code
    if report.error_count > 0:
        raise typer.Exit(code=1)
    if strict and report.warning_count > 0:
        raise typer.Exit(code=1)
    raise typer.Exit(code=0)
