"""CLI commands for multi-page table continuity inspection and cross-page stitching."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ubt.core.cleaners.table_stitcher import (
    TableStitchSummary,
    parse_table_fingerprint,
    stitch_multipage_tables,
)
from ubt.core.patch.engine import load_document_blocks

console = Console()
table_app = typer.Typer(
    name="table",
    help="Multi-page table continuity inspection and stitching.",
    no_args_is_help=True,
)


@table_app.command(name="inspect")
def table_inspect_command(
    file_or_job: Annotated[
        str,
        typer.Argument(
            help="Document file path or SQLite ledger job ID to inspect tables in.",
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
    """Inspect tables in a document, detecting fragmented cross-page tables and stitch candidates."""
    try:
        resolved_src, blocks, _ = asyncio.run(load_document_blocks(file_or_job))
    except Exception as exc:
        console.print(f"[bold red]Error loading blocks from '{file_or_job}': {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    summary = TableStitchSummary()
    # Non-destructive simulation: drop_absorbed=False preserves all blocks
    stitch_multipage_tables(blocks, drop_absorbed=False, summary_out=summary)

    # Collect statistics
    table_fps: list[dict[str, Any]] = []
    for b in blocks:
        fp = parse_table_fingerprint(b)
        if fp is not None:
            table_fps.append(
                {
                    "id": fp.block_id,
                    "page": fp.page,
                    "cols": fp.column_count,
                    "rows": fp.row_count,
                    "continuation": fp.has_continuation_label,
                    "headless": fp.is_headless_candidate,
                    "fused_with": b.provenance.fused_block_ids if b.provenance else [],
                }
            )

    result_data = {
        "source": resolved_src,
        "total_tables": summary.total_tables_inspected,
        "tables_stitched": summary.tables_stitched_count,
        "blocks_absorbed": summary.blocks_absorbed_count,
        "pairs": summary.stitched_pairs,
        "tables": table_fps,
    }

    if json_output:
        console.print(json.dumps(result_data, indent=2, ensure_ascii=False))
        return

    # Terminal output
    console.print(
        Panel(
            f"[bold cyan]Document/Job:[/bold cyan] {resolved_src}\n"
            f"[bold cyan]Total Tables Found:[/bold cyan] {summary.total_tables_inspected}\n"
            f"[bold cyan]Stitched Cross-Page Pairs:[/bold cyan] {summary.tables_stitched_count}\n"
            f"[bold cyan]Absorbed Continuation Blocks:[/bold cyan] {summary.blocks_absorbed_count}",
            title="[bold green]Multi-Page Table Continuity Inspection[/bold green]",
            expand=False,
        )
    )

    if summary.stitched_pairs:
        pairs_tbl = Table(title="Stitched Cross-Page Table Pairs", show_header=True)
        pairs_tbl.add_column("Primary Table", style="green")
        pairs_tbl.add_column("Secondary Fragment", style="yellow")
        pairs_tbl.add_column("Page Transition", style="cyan")
        pairs_tbl.add_column("Stitch Reason", style="magenta")

        for p in summary.stitched_pairs:
            p_page = p.get("primary_page") or "?"
            s_page = p.get("secondary_page") or "?"
            pairs_tbl.add_row(
                p.get("primary_id", ""),
                p.get("secondary_id", ""),
                f"Page {p_page} -> Page {s_page}",
                p.get("reason", ""),
            )
        console.print(pairs_tbl)
    else:
        console.print(
            "[dim italic]No multi-page split tables detected requiring stitching.[/dim italic]"
        )


@table_app.command(name="stitch")
def table_stitch_command(
    file_or_job: Annotated[
        str,
        typer.Argument(
            help="Document file path or SQLite ledger job ID to stitch tables in.",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Optional file path to output stitched table Markdown or summary.",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output JSON summary of stitching actions.",
        ),
    ] = False,
) -> None:
    """Fuse multi-page fragmented tables into single continuous streams."""
    try:
        resolved_src, blocks, _ = asyncio.run(load_document_blocks(file_or_job))
    except Exception as exc:
        console.print(f"[bold red]Error loading blocks from '{file_or_job}': {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    summary = TableStitchSummary()
    stitched_blocks = stitch_multipage_tables(blocks, drop_absorbed=True, summary_out=summary)

    data = {
        "source": resolved_src,
        "original_blocks": len(blocks),
        "stitched_blocks": len(stitched_blocks),
        "tables_stitched": summary.tables_stitched_count,
        "blocks_absorbed": summary.blocks_absorbed_count,
        "pairs": summary.stitched_pairs,
    }

    if output is not None:
        if output.suffix.lower() == ".json":
            output.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            # Write markdown representation of stitched blocks
            text_parts = [b.source_text for b in stitched_blocks if b.source_text]
            output.write_text("\n\n".join(text_parts), encoding="utf-8")
        console.print(
            f"[green]Successfully exported stitched result to [bold]{output}[/bold][/green]"
        )

    if json_output:
        console.print(json.dumps(data, indent=2, ensure_ascii=False))
    elif output is None:
        console.print(
            f"[bold green]Stitching Complete:[/bold green] {summary.tables_stitched_count} table pairs "
            f"fused ({summary.blocks_absorbed_count} blocks absorbed into continuous streams)."
        )
