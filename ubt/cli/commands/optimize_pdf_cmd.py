"""CLI command for PDF size miniaturization and lossless stream compression."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ubt.adapters.pdf.pdf_optimizer import (
    OptimizationEngine,
    OptimizationProfile,
    PdfOptimizeOptions,
    optimize_pdf,
)
from ubt.cli.ui import emit_json

console = Console()


def _format_bytes(size: int) -> str:
    """Format bytes into human-readable KiB/MiB string."""
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KiB"
    return f"{size / (1024 * 1024):.2f} MiB"


def optimize_pdf_command(
    pdf_path: Annotated[
        Path,
        typer.Argument(
            help="Path to the PDF file to miniaturize and optimize.",
            exists=True,
            dir_okay=False,
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Target path for optimized PDF (defaults to atomic in-place replacement).",
            dir_okay=False,
        ),
    ] = None,
    profile: Annotated[
        str,
        typer.Option(
            "--profile",
            "-p",
            help="Optimization target profile: 'lossless' (default), 'ebook', 'screen'.",
        ),
    ] = "lossless",
    engine: Annotated[
        str,
        typer.Option(
            "--engine",
            "-e",
            help="Compression engine: 'auto' (default), 'pikepdf', 'ghostscript'.",
        ),
    ] = "auto",
    dedup: Annotated[
        bool,
        typer.Option(
            "--dedup/--no-dedup",
            help="Deduplicate identical resource streams across pages.",
        ),
    ] = True,
    linearize: Annotated[
        bool,
        typer.Option(
            "--linearize/--no-linearize",
            help="Generate linearized PDF for Fast Web View on mobile/browsers.",
        ),
    ] = True,
    subset: Annotated[
        bool,
        typer.Option(
            "--subset/--no-subset",
            help="Attempt font glyph subsetting pass.",
        ),
    ] = True,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output machine-readable JSON metrics.",
        ),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option(
            "--quiet",
            "-q",
            help="Suppress human-readable terminal output.",
        ),
    ] = False,
) -> None:
    """Miniaturize, deduplicate, and compress a PDF deliverable."""
    try:
        prof = OptimizationProfile(profile.lower().strip())
    except ValueError:
        console.print(
            f"[bold red]Error:[/] Unknown profile '{profile}'. "
            f"Choose from: {', '.join(p.value for p in OptimizationProfile)}"
        )
        raise typer.Exit(code=2) from None

    try:
        eng = OptimizationEngine(engine.lower().strip())
    except ValueError:
        console.print(
            f"[bold red]Error:[/] Unknown engine '{engine}'. "
            f"Choose from: {', '.join(e.value for e in OptimizationEngine)}"
        )
        raise typer.Exit(code=2) from None

    opts = PdfOptimizeOptions(
        profile=prof,
        engine=eng,
        deduplicate_streams=dedup,
        linearize=linearize,
        subset_fonts=subset,
    )

    result = optimize_pdf(pdf_path, output_path=output, options=opts)

    if json_output:
        emit_json(result.model_dump())
        if result.status == "failed":
            raise typer.Exit(code=1)
        return

    if quiet:
        if result.status == "failed":
            raise typer.Exit(code=1)
        return

    # Rich Terminal Presentation
    status_color = (
        "green"
        if result.status == "optimized"
        else ("yellow" if result.status == "skipped_already_optimal" else "red")
    )
    status_label = {
        "optimized": "OPTIMIZED",
        "skipped_already_optimal": "ALREADY OPTIMAL",
        "failed": "FAILED",
    }.get(result.status, result.status.upper())

    table = Table(box=None, padding=(0, 2), show_header=False)
    table.add_column("Property", style="bold cyan")
    table.add_column("Value")

    table.add_row("Document Path", str(result.path))
    table.add_row("Pages Processed", str(result.pages_processed))
    table.add_row("Profile", f"[bold]{result.profile}[/]")
    table.add_row("Engine", result.engine_used)
    table.add_row(
        "Original Size",
        f"{_format_bytes(result.original_size_bytes)} ({result.original_size_bytes:,} bytes)",
    )
    table.add_row(
        "Optimized Size",
        f"{_format_bytes(result.optimized_size_bytes)} ({result.optimized_size_bytes:,} bytes)",
    )

    if result.status == "optimized":
        saved_str = (
            f"[bold green]-{_format_bytes(result.reduction_bytes)} (-{result.percentage_saved}%)[/]"
        )
        table.add_row("Reduction", saved_str)
    elif result.status == "skipped_already_optimal":
        table.add_row("Reduction", "[yellow]0 B (Original file was already compact)[/]")

    table.add_row("Streams Deduplicated", str(result.streams_deduplicated))
    table.add_row("Images Downsampled", str(result.images_optimized))
    table.add_row(
        "Fonts Subsetting", "[green]Yes[/]" if result.fonts_subsetted else "Preserved / Skipped"
    )
    table.add_row("Execution Duration", f"{result.duration_ms:.1f} ms")

    if result.error:
        table.add_row("Error", f"[red]{result.error}[/]")

    panel = Panel(
        table,
        title=f"[bold]PDF Miniaturization Report: [{status_color}]{status_label}[/{status_color}][/bold]",
        subtitle="[dim]Universal Book Translator PDF Engine[/dim]",
        border_style=status_color,
    )
    console.print(panel)

    if result.status == "failed":
        raise typer.Exit(code=1)


__all__ = ["optimize_pdf_command"]
