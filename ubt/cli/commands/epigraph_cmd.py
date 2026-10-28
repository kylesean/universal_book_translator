"""CLI commands for chapter epigraph and abstract inspection and generation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ubt.core.epigraph.detector import detect_chapter_epigraphs_and_abstracts
from ubt.core.epigraph.generator import (
    render_abstract_html,
    render_abstract_typst,
    render_epigraph_html,
    render_epigraph_typst,
)
from ubt.core.epigraph.models import (
    ChapterAbstract,
    Epigraph,
    EpigraphConfig,
    EpigraphStyle,
)
from ubt.core.patch.engine import load_document_blocks

console = Console()
epigraph_app = typer.Typer(
    name="epigraph",
    help="Chapter Epigraph & Abstract Formatter: quotes, motto attributions, and introductory callout boxes.",
    no_args_is_help=True,
)


@epigraph_app.command(name="inspect")
def epigraph_inspect_command(
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
    """Inspect chapter-opening epigraph quotations and introductory abstracts."""
    try:
        resolved_src, blocks, _ = asyncio.run(load_document_blocks(file_or_job))
    except Exception as exc:
        console.print(f"[bold red]Error loading blocks from '{file_or_job}': {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    epigraphs_by_id, abstracts_by_id = detect_chapter_epigraphs_and_abstracts(blocks)

    if json_output:
        res = {
            "source": resolved_src,
            "total_epigraphs": len(epigraphs_by_id),
            "total_abstracts": len(abstracts_by_id),
            "epigraphs": [
                {
                    "block_id": epi.block_id,
                    "chapter": epi.chapter_index,
                    "text": epi.text,
                    "source": epi.source,
                }
                for epi in epigraphs_by_id.values()
            ],
            "abstracts": [
                {
                    "block_id": ab.block_id,
                    "chapter": ab.chapter_index,
                    "title": ab.title,
                    "text": ab.text,
                }
                for ab in abstracts_by_id.values()
            ],
        }
        console.print(json.dumps(res, indent=2, ensure_ascii=False), soft_wrap=True, markup=False)
        return

    console.print(
        Panel(
            f"[bold cyan]Source Document:[/] {resolved_src}\n"
            f"[bold cyan]Total Epigraphs Found:[/] {len(epigraphs_by_id)}\n"
            f"[bold cyan]Total Abstracts Found:[/] {len(abstracts_by_id)}",
            title="[bold blue]Chapter Epigraph & Abstract Audit[/bold blue]",
        )
    )

    if epigraphs_by_id:
        table = Table(
            title=f"Detected Chapter Epigraphs ({len(epigraphs_by_id)} items)",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Chapter", justify="center", style="bold yellow", width=10)
        table.add_column("Block ID", style="dim", width=14)
        table.add_column("Quotation Text", style="white")
        table.add_column("Author / Source", style="cyan", width=24)

        for epi in epigraphs_by_id.values():
            disp_text = (epi.text[:60] + "...") if len(epi.text) > 60 else epi.text
            table.add_row(
                f"Chapter {epi.chapter_index}",
                epi.block_id,
                disp_text,
                epi.source or "-",
            )
        console.print(table)

    if abstracts_by_id:
        table_abs = Table(
            title=f"Detected Chapter Abstracts ({len(abstracts_by_id)} items)",
            show_header=True,
            header_style="bold green",
        )
        table_abs.add_column("Chapter", justify="center", style="bold yellow", width=10)
        table_abs.add_column("Block ID", style="dim", width=14)
        table_abs.add_column("Title", style="bold white", width=16)
        table_abs.add_column("Abstract Preview", style="dim")

        for ab in abstracts_by_id.values():
            disp_ab = (ab.text[:60] + "...") if len(ab.text) > 60 else ab.text
            table_abs.add_row(
                f"Chapter {ab.chapter_index}",
                ab.block_id,
                ab.title,
                disp_ab,
            )
        console.print(table_abs)


@epigraph_app.command(name="generate")
def epigraph_generate_command(
    text: Annotated[
        str,
        typer.Argument(
            help="Quotation or abstract text to format.",
        ),
    ],
    source: Annotated[
        str | None,
        typer.Option(
            "--source",
            "-s",
            help="Author or book title attribution (e.g. 'Donald E. Knuth, TAOCP').",
        ),
    ] = None,
    is_abstract: Annotated[
        bool,
        typer.Option(
            "--abstract",
            help="Generate as a chapter abstract container rather than an epigraph quotation.",
        ),
    ] = False,
    abstract_title: Annotated[
        str,
        typer.Option(
            "--title",
            help="Title for the abstract box (e.g. '本章导读' or 'Chapter Summary').",
        ),
    ] = "本章导读",
    style: Annotated[
        str,
        typer.Option(
            "--style",
            help="Epigraph style: 'classic_right', 'border_left', 'callout_box', 'minimal'.",
        ),
    ] = "classic_right",
    output_format: Annotated[
        str,
        typer.Option(
            "--format",
            help="Output markup format: 'typst' or 'html'.",
        ),
    ] = "typst",
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Optional file path to write generated markup.",
        ),
    ] = None,
) -> None:
    """Generate publication-ready Typst or HTML epigraph quotation or abstract box."""
    style_enum = EpigraphStyle.CLASSIC_RIGHT
    try:
        style_enum = EpigraphStyle(style.lower().strip())
    except ValueError:
        console.print(
            f"[bold yellow]Unknown style '{style}', defaulting to '{EpigraphStyle.CLASSIC_RIGHT.value}'[/bold yellow]"
        )

    cfg = EpigraphConfig(style=style_enum)
    fmt = output_format.lower().strip()

    if is_abstract:
        ab = ChapterAbstract(title=abstract_title, text=text)
        if fmt == "typst":
            markup = render_abstract_typst(ab, config=cfg)
        elif fmt == "html":
            markup = render_abstract_html(ab, config=cfg)
        else:
            console.print(
                f"[bold red]Unsupported format '{output_format}': use 'typst' or 'html'.[/bold red]"
            )
            raise typer.Exit(code=1)
    else:
        epi = Epigraph(text=text, source=source or "")
        if fmt == "typst":
            markup = render_epigraph_typst(epi, config=cfg)
        elif fmt == "html":
            markup = render_epigraph_html(epi, config=cfg)
        else:
            console.print(
                f"[bold red]Unsupported format '{output_format}': use 'typst' or 'html'.[/bold red]"
            )
            raise typer.Exit(code=1)

    if output is not None:
        try:
            output.write_text(markup, encoding="utf-8")
            console.print(
                f"[bold green]Generated {fmt.upper()} markup written to {output}[/bold green]"
            )
        except Exception as exc:
            console.print(f"[bold red]Failed to write to '{output}': {exc}[/bold red]")
            raise typer.Exit(code=1) from exc
    else:
        console.print(markup, soft_wrap=True, markup=False)
