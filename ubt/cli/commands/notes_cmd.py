"""CLI commands for footnote & endnote life-cycle auditing, renumbering, and generation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ubt.core.notes.extractor import extract_notes_topology
from ubt.core.notes.manager import (
    render_endnote_section_html,
    render_endnote_section_markdown,
    render_endnote_section_typst,
)
from ubt.core.notes.models import (
    NoteNumberingScope,
    NoteNumberingStyle,
    NotePlacement,
    NotesConfig,
)
from ubt.core.patch.engine import load_document_blocks

console = Console()
notes_app = typer.Typer(
    name="notes",
    help="Footnote & Endnote Flow Manager: auditing, renumbering, and publication rendering.",
    no_args_is_help=True,
)


@notes_app.command(name="inspect")
def notes_inspect_command(
    file_or_job: Annotated[
        str,
        typer.Argument(
            help="Document file path or SQLite ledger job ID to inspect.",
        ),
    ],
    placement: Annotated[
        str,
        typer.Option(
            "--placement",
            "-p",
            help="Note layout placement: 'footnote', 'chapter_endnote', or 'book_endnote'.",
        ),
    ] = "footnote",
    scope: Annotated[
        str,
        typer.Option(
            "--scope",
            "-s",
            help="Numbering scope: 'per_chapter' or 'continuous'.",
        ),
    ] = "per_chapter",
    style: Annotated[
        str,
        typer.Option(
            "--style",
            help="Marker numbering style: 'arabic', 'circled', 'roman', or 'symbol'.",
        ),
    ] = "arabic",
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output structured JSON results.",
        ),
    ] = False,
) -> None:
    """Inspect and audit footnote callout matching, resolution rate, and chapter distribution."""
    try:
        resolved_src, blocks, _ = asyncio.run(load_document_blocks(file_or_job))
    except Exception as exc:
        console.print(f"[bold red]Error loading blocks from '{file_or_job}': {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    cfg = NotesConfig(
        placement=NotePlacement(placement.lower().strip()),
        numbering_scope=NoteNumberingScope(scope.lower().strip()),
        numbering_style=NoteNumberingStyle(style.lower().strip()),
    )
    topology = extract_notes_topology(blocks, cfg)

    if json_output:
        res = topology.to_dict()
        res["source"] = resolved_src
        console.print(json.dumps(res, indent=2, ensure_ascii=False), soft_wrap=True)
        return

    rate_color = (
        "green"
        if topology.resolution_rate >= 0.9
        else "yellow"
        if topology.resolution_rate >= 0.7
        else "red"
    )
    console.print(
        Panel(
            f"[bold cyan]Source Document:[/] {resolved_src}\n"
            f"[bold cyan]Total Notes Found:[/] {topology.total_notes}\n"
            f"[bold cyan]Resolution Rate:[/] [{rate_color}]{topology.resolution_rate * 100:.1f}%[/{rate_color}]\n"
            f"[bold cyan]Orphan Callouts (Missing Definition):[/bold cyan] {len(topology.orphan_callouts)}\n"
            f"[bold cyan]Orphan Definitions (Uncalled):[/bold cyan] {len(topology.orphan_definitions)}\n"
            f"[bold cyan]Placement Policy:[/] {cfg.placement.value}\n"
            f"[bold cyan]Numbering Scope:[/] {cfg.numbering_scope.value} ({cfg.numbering_style.value})",
            title="[bold blue]Footnote & Endnote Topology Summary[/bold blue]",
        )
    )

    if topology.notes:
        table = Table(
            title=f"Extracted Notes ({topology.total_notes} items)",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Ch.", justify="center", style="cyan", width=4)
        table.add_column("Idx", justify="center", style="bold yellow", width=5)
        table.add_column("Marker", justify="center", style="dim", width=8)
        table.add_column("Anchor ID", style="dim", width=14)
        table.add_column("Content Preview", style="white", overflow="fold")

        for n in topology.notes:
            formatted_m = n.formatted_marker(cfg.numbering_style)
            table.add_row(
                str(n.chapter_index),
                str(n.index_num),
                f"{n.callout_marker} -> {formatted_m}",
                n.note_id,
                n.content[:60] + ("..." if len(n.content) > 60 else ""),
            )
        console.print(table)


@notes_app.command(name="generate")
def notes_generate_command(
    file_or_job: Annotated[
        str,
        typer.Argument(
            help="Document file path or SQLite ledger job ID to extract notes from.",
        ),
    ],
    output_path: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Target output file path to write generated notes section.",
        ),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Output syntax format: 'typst', 'html', or 'markdown'.",
        ),
    ] = "typst",
    placement: Annotated[
        str,
        typer.Option(
            "--placement",
            "-p",
            help="Note layout placement: 'chapter_endnote' or 'book_endnote'.",
        ),
    ] = "book_endnote",
    scope: Annotated[
        str,
        typer.Option(
            "--scope",
            "-s",
            help="Numbering scope: 'per_chapter' or 'continuous'.",
        ),
    ] = "per_chapter",
    style: Annotated[
        str,
        typer.Option(
            "--style",
            help="Marker numbering style: 'arabic', 'circled', 'roman', or 'symbol'.",
        ),
    ] = "arabic",
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output structured JSON results.",
        ),
    ] = False,
) -> None:
    """Generate standalone publication-grade endnotes section source (Typst, HTML, or Markdown)."""
    try:
        resolved_src, blocks, _ = asyncio.run(load_document_blocks(file_or_job))
    except Exception as exc:
        console.print(f"[bold red]Error loading blocks from '{file_or_job}': {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    cfg = NotesConfig(
        placement=NotePlacement(placement.lower().strip()),
        numbering_scope=NoteNumberingScope(scope.lower().strip()),
        numbering_style=NoteNumberingStyle(style.lower().strip()),
    )
    topology = extract_notes_topology(blocks, cfg)

    fmt = output_format.lower().strip()
    if fmt == "typst":
        markup = render_endnote_section_typst(topology.notes, cfg)
    elif fmt == "html":
        markup = render_endnote_section_html(topology.notes, cfg)
    elif fmt in ("markdown", "md"):
        markup = render_endnote_section_markdown(topology.notes, cfg)
    else:
        console.print(
            f"[bold red]Unsupported format: '{fmt}'. Choose 'typst', 'html', or 'markdown'.[/bold red]"
        )
        raise typer.Exit(code=1)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markup, encoding="utf-8")
        console.print(f"[bold green]Successfully generated notes to '{output_path}'.[/bold green]")
        return

    if json_output:
        console.print(
            json.dumps(
                {
                    "source": resolved_src,
                    "format": fmt,
                    "total_notes": topology.total_notes,
                    "markup": markup,
                },
                indent=2,
                ensure_ascii=False,
            ),
            soft_wrap=True,
        )
        return

    console.print(markup, soft_wrap=True)


@notes_app.command(name="audit")
def notes_audit_command(
    file_or_job: Annotated[
        str,
        typer.Argument(
            help="Document file path or SQLite ledger job ID to audit.",
        ),
    ],
    min_resolution: Annotated[
        float,
        typer.Option(
            "--min-resolution",
            help="Minimum allowable note resolution rate (0.0 to 1.0).",
        ),
    ] = 0.85,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output structured JSON results.",
        ),
    ] = False,
) -> None:
    """Audit document footnote and endnote health and check for orphaned callouts or lost definitions."""
    try:
        resolved_src, blocks, _ = asyncio.run(load_document_blocks(file_or_job))
    except Exception as exc:
        console.print(f"[bold red]Error loading blocks from '{file_or_job}': {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    cfg = NotesConfig()
    topology = extract_notes_topology(blocks, cfg)

    passed = topology.resolution_rate >= min_resolution

    if json_output:
        console.print(
            json.dumps(
                {
                    "source": resolved_src,
                    "passed": passed,
                    "resolution_rate": round(topology.resolution_rate, 4),
                    "min_resolution": min_resolution,
                    "total_notes": topology.total_notes,
                    "orphan_callouts": topology.orphan_callouts,
                    "orphan_definitions": topology.orphan_definitions,
                },
                indent=2,
                ensure_ascii=False,
            ),
            soft_wrap=True,
        )
        if not passed:
            raise typer.Exit(code=1)
        return

    if passed:
        console.print(
            f"[bold green]PASS:[/] Footnote resolution rate is {topology.resolution_rate * 100:.1f}% "
            f"(>= threshold {min_resolution * 100:.1f}%). Total notes: {topology.total_notes}."
        )
    else:
        console.print(
            f"[bold red]FAIL:[/] Footnote resolution rate is {topology.resolution_rate * 100:.1f}% "
            f"(< threshold {min_resolution * 100:.1f}%).\n"
            f"Orphan Callouts: {len(topology.orphan_callouts)}, Orphan Definitions: {len(topology.orphan_definitions)}."
        )
        raise typer.Exit(code=1)
