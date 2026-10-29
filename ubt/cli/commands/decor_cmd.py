"""CLI commands for chapter header decoration, visual themes, and drop caps."""

from __future__ import annotations

import asyncio
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ubt.cli.ui import emit_json
from ubt.core.decor.dropcap import (
    HTML_DROPCAP_CSS,
    TYPST_DROPCAP_MACRO,
    apply_dropcap_html,
    apply_dropcap_typst,
    extract_dropcap_initial,
)
from ubt.core.decor.header import (
    parse_chapter_heading,
    render_chapter_header_html,
    render_chapter_header_typst,
)
from ubt.core.decor.models import ChapterHeaderConfig, DropCapConfig, HeaderStyle
from ubt.core.ir.models import BlockType
from ubt.core.patch.engine import load_document_blocks
from ubt.model.ast import Heading

console = Console()
decor_app = typer.Typer(
    name="decor",
    help="Publication chapter header decorations, visual themes, and drop cap styling.",
    no_args_is_help=True,
)


@decor_app.command(name="preview")
def decor_preview_command(
    heading_text: Annotated[
        str,
        typer.Argument(
            help="Chapter title string (e.g. '第 1 章：分布式系统导论' or 'Chapter 3: Raft Consensus').",
        ),
    ],
    sample_body: Annotated[
        str,
        typer.Option(
            "--body",
            "-b",
            help="Sample text for the opening paragraph to test drop caps.",
        ),
    ] = "在大规模分布式系统中，高可用性与强一致性始终是系统设计的核心目标。",
    style: Annotated[
        str,
        typer.Option(
            "--style",
            "-s",
            help="Visual decoration theme: 'classic', 'modern', 'academic', or 'minimal'.",
        ),
    ] = "classic",
    output_format: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Output syntax format: 'typst' or 'html'.",
        ),
    ] = "typst",
    dropcap_lines: Annotated[
        int,
        typer.Option(
            "--dropcap-lines",
            "-l",
            help="Number of lines spanned by the drop cap (typically 2 or 3).",
        ),
    ] = 2,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output structured JSON results.",
        ),
    ] = False,
) -> None:
    """Preview a chapter header decoration banner and opening paragraph drop cap."""
    h_style = HeaderStyle(style.lower().strip())
    h_cfg = ChapterHeaderConfig(style=h_style)
    d_cfg = DropCapConfig(lines=dropcap_lines)

    label, title, subtitle = parse_chapter_heading(heading_text)
    prefix, initial_char, remainder = extract_dropcap_initial(sample_body)

    fmt = output_format.lower().strip()
    if fmt == "typst":
        header_markup = render_chapter_header_typst(heading_text, h_cfg)
        dropcap_markup = apply_dropcap_typst(sample_body, d_cfg)
        combined = f"{TYPST_DROPCAP_MACRO}\n{header_markup}\n\n{dropcap_markup}\n"
    elif fmt == "html":
        header_markup = render_chapter_header_html(heading_text, h_cfg)
        dropcap_markup = f"<p>{apply_dropcap_html(sample_body, d_cfg)}</p>"
        combined = f"<style>\n{HTML_DROPCAP_CSS}\n</style>\n{header_markup}\n{dropcap_markup}\n"
    else:
        console.print(
            f"[bold red]Unsupported format: '{fmt}'. Choose 'typst' or 'html'.[/bold red]"
        )
        raise typer.Exit(code=1)

    if json_output:
        emit_json(
            {
                "heading_raw": heading_text,
                "parsed": {
                    "label": label,
                    "title": title,
                    "subtitle": subtitle,
                },
                "dropcap": {
                    "prefix": prefix,
                    "initial_char": initial_char,
                    "lines": dropcap_lines,
                },
                "style": style,
                "format": fmt,
                "markup": combined,
            }
        )
        return

    console.print(
        Panel(
            f"[bold cyan]Parsed Label:[/] {label or 'None'}\n"
            f"[bold cyan]Parsed Title:[/] {title}\n"
            f"[bold cyan]Parsed Subtitle:[/] {subtitle or 'None'}\n"
            f"[bold cyan]Drop Cap Initial:[/] [yellow]{prefix}{initial_char}[/yellow] ({dropcap_lines} lines)\n"
            f"[bold cyan]Theme Style:[/] {h_style.value.upper()}\n\n"
            f"[bold green]Emitted {fmt.upper()} Markup:[/bold green]\n{combined}",
            title="[bold blue]Chapter Header & Drop Cap Preview[/bold blue]",
        )
    )


@decor_app.command(name="inspect")
def decor_inspect_command(
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
    """Inspect all chapter headings and opening paragraph drop cap candidates in a document."""
    try:
        resolved_src, blocks, _ = asyncio.run(load_document_blocks(file_or_job))
    except Exception as exc:
        console.print(f"[bold red]Error loading blocks from '{file_or_job}': {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    chapters: list[dict[str, str]] = []
    current_chapter: dict[str, str] | None = None

    for block in blocks:
        text = block.target_text or (block.element.text if block.element else "") or ""
        el = getattr(block, "element", None)
        level = getattr(el, "level", None) or getattr(block, "level", None) or 1
        is_h1 = (isinstance(el, Heading) and level == 1) or (
            getattr(block, "block_type", None) == BlockType.HEADING and level == 1
        )

        if is_h1 and text.strip():
            label, title, subtitle = parse_chapter_heading(text)
            current_chapter = {
                "heading": text.strip(),
                "label": label,
                "title": title,
                "subtitle": subtitle,
                "first_para": "",
                "dropcap_letter": "",
            }
            chapters.append(current_chapter)
        elif current_chapter and not current_chapter["first_para"]:
            b_type = str(getattr(block.block_type, "value", block.block_type) or "")
            if b_type == "narrative" and text.strip():
                current_chapter["first_para"] = text.strip()
                prefix, letter, _ = extract_dropcap_initial(text)
                current_chapter["dropcap_letter"] = f"{prefix}{letter}"

    if json_output:
        emit_json(
            {
                "source": resolved_src,
                "total_chapters": len(chapters),
                "chapters": chapters,
            }
        )
        return

    table = Table(
        title=f"Document Chapter Header & Drop Cap Audit ({len(chapters)} chapters)",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Chapter Label", style="cyan", width=14)
    table.add_column("Title", style="white")
    table.add_column("Subtitle", style="dim")
    table.add_column("Drop Cap", justify="center", style="bold yellow", width=10)
    table.add_column("Opening Text Preview", style="dim", overflow="fold")

    for ch in chapters:
        table.add_row(
            ch["label"] or "None",
            ch["title"],
            ch["subtitle"] or "-",
            ch["dropcap_letter"] or "-",
            ch["first_para"][:40] + ("..." if len(ch["first_para"]) > 40 else ""),
        )

    console.print(table)
