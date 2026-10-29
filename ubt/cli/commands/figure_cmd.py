"""CLI commands for publication multi-panel figure inspection, alignment, and generation."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ubt.cli.ui import emit_json
from ubt.core.figures.detector import (
    calculate_grid_columns,
    detect_figure_panels,
    extract_main_and_subcaptions,
)
from ubt.core.figures.generator import (
    render_figure_grid_html,
    render_figure_grid_typst,
)
from ubt.core.figures.models import (
    FigureLayoutMode,
    FigurePanel,
    SubFigure,
)
from ubt.core.patch.engine import load_document_blocks

console = Console()
figure_app = typer.Typer(
    name="figure",
    help="Publication Figure & Subfigure Aligner: multi-panel grids, subcaption parsing, and Typst/HTML layout.",
    no_args_is_help=True,
)


@figure_app.command(name="inspect")
def figure_inspect_command(
    file_or_job: Annotated[
        str,
        typer.Argument(
            help="Document file path or SQLite ledger job ID to inspect.",
        ),
    ],
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output structured JSON results.",
        ),
    ] = False,
) -> None:
    """Inspect all detected figures, subfigures, composite captions, and grid panels."""
    try:
        resolved_src, blocks, _ = asyncio.run(load_document_blocks(file_or_job))
    except Exception as exc:
        console.print(f"[bold red]Error loading blocks from '{file_or_job}': {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    panels = detect_figure_panels(blocks)
    composite_panels = [p for p in panels if p.is_composite]

    if json_output:
        res = {
            "source": resolved_src,
            "total_panels": len(panels),
            "composite_panels": len(composite_panels),
            "panels": [
                {
                    "panel_id": p.panel_id,
                    "columns": p.columns,
                    "gutter_pt": p.gutter_pt,
                    "span_full_width": p.span_full_width,
                    "is_composite": p.is_composite,
                    "main_caption": p.main_caption,
                    "subfigures": [
                        {
                            "asset": sf.asset,
                            "label": sf.label,
                            "caption": sf.caption,
                            "width_pt": sf.width_pt,
                            "block_id": sf.block_id,
                        }
                        for sf in p.subfigures
                    ],
                }
                for p in panels
            ],
        }
        emit_json(res)
        return

    console.print(
        Panel(
            f"[bold cyan]Source Document:[/] {resolved_src}\n"
            f"[bold cyan]Total Figure Panels:[/] {len(panels)}\n"
            f"[bold cyan]Composite/Grid Panels:[/] {len(composite_panels)}\n"
            f"[bold cyan]Single Image Panels:[/] {len(panels) - len(composite_panels)}",
            title="[bold blue]Publication Multi-Panel Figure Audit[/bold blue]",
        )
    )

    if panels:
        table = Table(
            title=f"Detected Figure Panels ({len(panels)} panels)",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Panel ID", style="bold cyan", width=14)
        table.add_column("Type", justify="center", style="bold yellow", width=12)
        table.add_column("Cols", justify="center", style="white", width=6)
        table.add_column("Subfigures", justify="center", style="cyan", width=10)
        table.add_column("Full Width", justify="center", style="dim", width=10)
        table.add_column("Main Caption", style="white")

        for p in panels:
            kind_label = "GRID" if p.is_composite else "SINGLE"
            table.add_row(
                p.panel_id,
                kind_label,
                str(p.columns),
                str(len(p.subfigures)),
                "Yes" if p.span_full_width else "No",
                (p.main_caption[:50] + "...")
                if len(p.main_caption) > 50
                else (p.main_caption or "-"),
            )
        console.print(table)


@figure_app.command(name="generate")
def figure_generate_command(
    assets: Annotated[
        list[str],
        typer.Argument(
            help="Image asset file paths or names to arrange into a grid panel.",
        ),
    ],
    caption: Annotated[
        str,
        typer.Option(
            "--caption",
            "-c",
            help="Main figure caption or composite caption string.",
        ),
    ] = "",
    subcaptions: Annotated[
        str | None,
        typer.Option(
            "--subcaptions",
            help="Comma-separated or semicolon-separated subcaptions (e.g. 'A: Model, B: Loss').",
        ),
    ] = None,
    columns: Annotated[
        int | None,
        typer.Option(
            "--columns",
            help="Number of grid columns (defaults to auto calculation: 1x2, 2x2, 1x3, etc.).",
        ),
    ] = None,
    gutter_pt: Annotated[
        float,
        typer.Option(
            "--gutter-pt",
            help="Gutter spacing between subfigures in PDF points.",
        ),
    ] = 10.0,
    span_full_width: Annotated[
        bool,
        typer.Option(
            "--span-full-width",
            help="Force the figure panel to span the full text width across multi-column layout.",
        ),
    ] = False,
    output_format: Annotated[
        str,
        typer.Option(
            "--format",
            help="Output markup format: 'typst' or 'html'.",
        ),
    ] = "typst",
    output_path: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="File path to write the generated markup.",
        ),
    ] = None,
) -> None:
    """Generate publication-ready Typst or HTML figure grid markup from assets."""
    if not assets:
        console.print("[bold red]No image assets provided.[/bold red]")
        raise typer.Exit(code=1)

    main_cap = caption
    parsed_subs: list[tuple[str, str]] = []

    if subcaptions:
        # User provided explicit subcaptions string
        _, parsed_subs = extract_main_and_subcaptions(subcaptions)
        if not parsed_subs:
            # Fallback to splitting by comma or semicolon
            raw_parts = [s.strip() for s in re.split(r"[,;]", subcaptions) if s.strip()]
            for idx, part in enumerate(raw_parts):
                char = chr(ord("a") + (idx % 26))
                parsed_subs.append((f"({char})", part))
    elif caption:
        # Parse subcaptions from composite caption if present
        extracted_main, extracted_subs = extract_main_and_subcaptions(caption)
        if extracted_subs:
            main_cap = extracted_main
            parsed_subs = extracted_subs

    subs: list[SubFigure] = []
    for idx, asset in enumerate(assets):
        lbl = ""
        sub_cap = ""
        if idx < len(parsed_subs):
            lbl, sub_cap = parsed_subs[idx]
        elif len(assets) > 1:
            char = chr(ord("a") + (idx % 26))
            lbl = f"({char})"

        subs.append(
            SubFigure(
                asset=asset,
                label=lbl,
                caption=sub_cap,
            )
        )

    cols = columns if columns is not None else calculate_grid_columns(len(subs))

    panel = FigurePanel(
        panel_id="fig-1",
        main_caption=main_cap,
        subfigures=subs,
        columns=cols,
        gutter_pt=gutter_pt,
        span_full_width=span_full_width,
        layout_mode=FigureLayoutMode.GRID if len(subs) > 1 else FigureLayoutMode.AUTO,
    )

    fmt = output_format.lower().strip()
    if fmt == "typst":
        markup = render_figure_grid_typst(panel)
    elif fmt == "html":
        markup = render_figure_grid_html(panel)
    else:
        console.print(
            f"[bold red]Unsupported format '{output_format}': use 'typst' or 'html'.[/bold red]"
        )
        raise typer.Exit(code=1)

    if output_path is not None:
        try:
            output_path.write_text(markup, encoding="utf-8")
            console.print(
                f"[bold green]Generated {fmt.upper()} figure written to {output_path}[/bold green]"
            )
        except Exception as exc:
            console.print(f"[bold red]Failed to write to '{output_path}': {exc}[/bold red]")
            raise typer.Exit(code=1) from exc
    else:
        console.print(markup, soft_wrap=True, markup=False)
