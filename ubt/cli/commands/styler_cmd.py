"""CLI commands for responsive EPUB and HTML layout inspection and rendering."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ubt.cli.ui import emit_json
from ubt.core.patch.engine import load_document_blocks
from ubt.model.ast import Caption, CodeBlock, Figure, Formula, Heading, ListItem, Paragraph
from ubt.model.ast import Table as ASTTable
from ubt.render.epub_view import compose_epub
from ubt.render.html_view import compose_html

console = Console()
styler_app = typer.Typer(
    name="styler",
    help="Responsive EPUB 3 & HTML layout inspection, mobile breakpoints, and styling.",
    no_args_is_help=True,
)


def _collect_element_stats(blocks: Any) -> dict[str, int]:
    """Tally AST element kinds for responsive inspection."""
    stats = {
        "headings": 0,
        "paragraphs": 0,
        "tables": 0,
        "code_blocks": 0,
        "formulas": 0,
        "figures": 0,
        "list_items": 0,
        "captions": 0,
        "other": 0,
    }
    for b in blocks:
        el = getattr(b, "element", None)
        if isinstance(el, Heading):
            stats["headings"] += 1
        elif isinstance(el, Paragraph):
            stats["paragraphs"] += 1
        elif isinstance(el, ASTTable):
            stats["tables"] += 1
        elif isinstance(el, CodeBlock):
            stats["code_blocks"] += 1
        elif isinstance(el, Formula):
            stats["formulas"] += 1
        elif isinstance(el, Figure):
            stats["figures"] += 1
        elif isinstance(el, ListItem):
            stats["list_items"] += 1
        elif isinstance(el, Caption):
            stats["captions"] += 1
        else:
            stats["other"] += 1
    return stats


@styler_app.command(name="inspect")
def styler_inspect_command(
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
    """Inspect document structure and assess multi-screen responsive readiness."""
    try:
        resolved_src, blocks, _ = asyncio.run(load_document_blocks(file_or_job))
    except Exception as exc:
        console.print(f"[bold red]Error loading blocks from '{file_or_job}': {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    stats = _collect_element_stats(blocks)
    total_blocks = len(blocks)

    result_data = {
        "source": resolved_src,
        "total_blocks": total_blocks,
        "element_breakdown": stats,
        "responsive_features": {
            "mobile_table_wrapper": stats["tables"] > 0,
            "code_copy_button": stats["code_blocks"] > 0,
            "katex_math_support": stats["formulas"] > 0,
            "fluid_typography": True,
            "bilingual_split_collapse": True,
            "epub3_eink_contrast": True,
        },
    }

    if json_output:
        emit_json(result_data)
        return

    # Terminal output
    console.print(
        Panel(
            f"[bold cyan]Document/Job:[/bold cyan] {resolved_src}\n"
            f"[bold cyan]Total Render Blocks:[/bold cyan] {total_blocks}\n"
            f"[bold cyan]Content Elements:[/bold cyan] "
            f"Headings: {stats['headings']} | Paragraphs: {stats['paragraphs']} | "
            f"Tables: {stats['tables']} | Code: {stats['code_blocks']} | "
            f"Math: {stats['formulas']} | Figures: {stats['figures']}",
            title="[bold blue]Responsive EPUB / HTML Layout Inspector[/bold blue]",
            border_style="cyan",
        )
    )

    table = Table(title="[bold yellow]Component Multi-Screen Readiness[/bold yellow]")
    table.add_column("Component", style="cyan")
    table.add_column("Count", style="bold white", width=8)
    table.add_column("Responsive Adaptation", style="green")

    table.add_row(
        "Tables",
        str(stats["tables"]),
        "Wrapped in .table-responsive (smooth horizontal touch scrolling)",
    )
    table.add_row(
        "Code Blocks", str(stats["code_blocks"]), "Auto horizontal scroll with click-to-copy button"
    )
    table.add_row(
        "Math Formulas", str(stats["formulas"]), "KaTeX vector rendering with overflow protection"
    )
    table.add_row(
        "Figures & Images", str(stats["figures"]), "Max-width: 100% with aspect-ratio preservation"
    )
    table.add_row(
        "Bilingual Split View", "-", "Auto-collapses to stacked layout on mobile viewports (<768px)"
    )
    table.add_row(
        "EPUB Typography", "-", "High-contrast E-ink friendly palette with CJK justification"
    )

    console.print(table)


@styler_app.command(name="render-html")
def styler_render_html_command(
    file_or_job: Annotated[
        str,
        typer.Argument(
            help="Document file path or SQLite ledger job ID to render.",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output HTML path. Defaults to <source>.html.",
        ),
    ] = None,
    bilingual: Annotated[
        bool,
        typer.Option(
            "--bilingual/--monolingual",
            help="Enable bilingual dual-mode reading view.",
        ),
    ] = True,
    title: Annotated[
        str,
        typer.Option(
            "--title",
            "-t",
            help="Document title for the HTML header.",
        ),
    ] = "UBT Responsive Book",
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output structured JSON results.",
        ),
    ] = False,
) -> None:
    """Render document into a modern, responsive HTML reading file."""
    try:
        resolved_src, blocks, _ = asyncio.run(load_document_blocks(file_or_job))
    except Exception as exc:
        console.print(f"[bold red]Error loading blocks from '{file_or_job}': {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    out_path = output or Path(resolved_src).with_suffix(".html")
    translations = {b.id: getattr(b, "target_text", "") or "" for b in blocks}

    comp = compose_html(blocks, translations, out_path, title=title, bilingual=bilingual)

    result = {
        "source": resolved_src,
        "output_path": str(comp.output_path),
        "total_placements": len(comp.placements),
        "bilingual": bilingual,
        "size_bytes": comp.output_path.stat().st_size if comp.output_path.exists() else 0,
    }

    if json_output:
        emit_json(result)
    else:
        console.print(f"[bold green]✓ Responsive HTML rendered to:[/bold green] {comp.output_path}")


@styler_app.command(name="render-epub")
def styler_render_epub_command(
    file_or_job: Annotated[
        str,
        typer.Argument(
            help="Document file path or SQLite ledger job ID to render.",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output EPUB path. Defaults to <source>.epub.",
        ),
    ] = None,
    bilingual: Annotated[
        bool,
        typer.Option(
            "--bilingual/--monolingual",
            help="Enable bilingual dual-mode in EPUB package.",
        ),
    ] = False,
    title: Annotated[
        str,
        typer.Option(
            "--title",
            "-t",
            help="Book title for the EPUB metadata.",
        ),
    ] = "UBT Translated Book",
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output structured JSON results.",
        ),
    ] = False,
) -> None:
    """Render document into a publication-grade EPUB 3 package."""
    try:
        resolved_src, blocks, _ = asyncio.run(load_document_blocks(file_or_job))
    except Exception as exc:
        console.print(f"[bold red]Error loading blocks from '{file_or_job}': {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    out_path = output or Path(resolved_src).with_suffix(".epub")
    translations = {b.id: getattr(b, "target_text", "") or "" for b in blocks}

    comp = compose_epub(blocks, translations, out_path, title=title, bilingual=bilingual)

    result = {
        "source": resolved_src,
        "output_path": str(comp.output_path),
        "total_placements": len(comp.placements),
        "bilingual": bilingual,
        "size_bytes": comp.output_path.stat().st_size if comp.output_path.exists() else 0,
    }

    if json_output:
        emit_json(result)
    else:
        console.print(
            f"[bold green]✓ Reflowable EPUB 3 rendered to:[/bold green] {comp.output_path}"
        )
