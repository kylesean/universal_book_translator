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


@table_app.command(name="style")
def table_style_command(
    file_or_job: Annotated[
        str,
        typer.Argument(
            help="Document file path or SQLite ledger job ID to render tables from.",
        ),
    ],
    output_format: Annotated[
        str,
        typer.Option(
            "--format",
            help="Output markup format: 'typst' or 'html'.",
        ),
    ] = "typst",
    continued_header: Annotated[
        str | None,
        typer.Option(
            "--header",
            help="Custom continuation text on repeated header (default: '（续表）').",
        ),
    ] = None,
    next_page_footer: Annotated[
        str | None,
        typer.Option(
            "--footer",
            help="Custom continuation text on table footer (default: '（续下页）').",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Optional file path to output styled tables markup.",
        ),
    ] = None,
) -> None:
    """Render document tables with publication-grade repeated headers and continuation notices."""
    try:
        resolved_src, blocks, _ = asyncio.run(load_document_blocks(file_or_job))
    except Exception as exc:
        console.print(f"[bold red]Error loading blocks from '{file_or_job}': {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    table_blocks = [
        b
        for b in blocks
        if str(
            getattr(getattr(b, "block_type", None), "value", getattr(b, "block_type", "")) or ""
        ).lower()
        == "table"
    ]

    if not table_blocks:
        console.print("[dim italic]No tables found in document.[/dim italic]")
        return

    from ubt.core.tables.models import TableContinuationConfig
    from ubt.core.tables.styler import generate_html_table_continuation_css
    from ubt.render.typst.document_renderer import render_table

    cfg = TableContinuationConfig()
    if continued_header is not None:
        cfg.continued_header_text_zh = continued_header
        cfg.continued_header_text_en = continued_header
    if next_page_footer is not None:
        cfg.next_page_footer_text_zh = next_page_footer
        cfg.next_page_footer_text_en = next_page_footer

    fmt = output_format.lower().strip()
    rendered_outputs: list[str] = []

    if fmt == "typst":
        for b in table_blocks:
            src = (b.target_text or b.source_text or "").strip()
            is_multi = bool(b.provenance and b.provenance.fused_block_ids)
            rendered = render_table(
                src,
                is_multipage=is_multi,
                enable_continuation=True,
                table_uid=f"tbl-{b.id}",
                continuation_config=cfg,
            )
            if rendered:
                rendered_outputs.append(rendered)
        combined = "\n\n".join(rendered_outputs)
    elif fmt == "html":
        from ubt.render.table_compiler import parse_markdown_table

        rendered_outputs.append(generate_html_table_continuation_css())
        for b in table_blocks:
            src = (b.target_text or b.source_text or "").strip()
            parsed = parse_markdown_table(src, escape=True)
            if not parsed:
                continue
            headers, rows = parsed
            lines = ['<table class="ubt-table">', "  <thead>"]
            if cfg.enable_continued_header:
                lines.append(
                    f'    <tr><th colspan="{len(headers)}" class="ubt-table-continued-notice">'
                    f"{cfg.continued_header_text_zh}</th></tr>"
                )
            lines.append("    <tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>")
            lines.append("  </thead>")
            if cfg.enable_next_page_footer:
                lines.append("  <tfoot>")
                lines.append(
                    f'    <tr><td colspan="{len(headers)}" class="ubt-table-continued-notice">'
                    f"{cfg.next_page_footer_text_zh}</td></tr>"
                )
                lines.append("  </tfoot>")
            lines.append("  <tbody>")
            for r in rows:
                padded = list(r) + [""] * (len(headers) - len(r))
                lines.append("    <tr>" + "".join(f"<td>{c}</td>" for c in padded) + "</tr>")
            lines.append("  </tbody>")
            lines.append("</table>")
            rendered_outputs.append("\n".join(lines))
        combined = "\n\n".join(rendered_outputs)
    else:
        console.print(
            f"[bold red]Unsupported format '{output_format}': use 'typst' or 'html'.[/bold red]"
        )
        raise typer.Exit(code=1)

    if output is not None:
        try:
            output.write_text(combined, encoding="utf-8")
            console.print(f"[bold green]Rendered styled tables written to {output}[/bold green]")
        except Exception as exc:
            console.print(f"[bold red]Failed to write to '{output}': {exc}[/bold red]")
            raise typer.Exit(code=1) from exc
    else:
        console.print(combined, soft_wrap=True, markup=False)


@table_app.command(name="audit")
def table_audit_command(
    file_or_job: Annotated[
        str,
        typer.Argument(
            help="Document file path or SQLite ledger job ID to audit tables in.",
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
    """Audit document tables for multi-page continuity, row counts, and header styling."""
    try:
        resolved_src, blocks, _ = asyncio.run(load_document_blocks(file_or_job))
    except Exception as exc:
        console.print(f"[bold red]Error loading blocks from '{file_or_job}': {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    from ubt.render.table_compiler import parse_markdown_table

    table_blocks = [
        b
        for b in blocks
        if str(
            getattr(getattr(b, "block_type", None), "value", getattr(b, "block_type", "")) or ""
        ).lower()
        == "table"
    ]

    records: list[dict[str, Any]] = []
    continuation_eligible = 0

    for b in table_blocks:
        src = (b.target_text or b.source_text or "").strip()
        parsed = parse_markdown_table(src, escape=False)
        cols = len(parsed[0]) if parsed else 0
        rows = len(parsed[1]) if parsed else 0
        is_multi = bool(b.provenance and b.provenance.fused_block_ids)
        eligible = is_multi or rows >= 4
        if eligible:
            continuation_eligible += 1

        records.append(
            {
                "id": b.id,
                "columns": cols,
                "rows": rows,
                "is_multipage": is_multi,
                "continuation_eligible": eligible,
            }
        )

    res_data = {
        "source": resolved_src,
        "total_tables": len(table_blocks),
        "continuation_eligible": continuation_eligible,
        "tables": records,
    }

    if json_output:
        console.print(
            json.dumps(res_data, indent=2, ensure_ascii=False), soft_wrap=True, markup=False
        )
        return

    console.print(
        Panel(
            f"[bold cyan]Source Document:[/] {resolved_src}\n"
            f"[bold cyan]Total Tables Found:[/] {len(table_blocks)}\n"
            f"[bold cyan]Continuation Styling Eligible:[/] {continuation_eligible}\n"
            f"[bold cyan]Short Single-Page Tables:[/] {len(table_blocks) - continuation_eligible}",
            title="[bold blue]Publication Table Continuity Audit[/bold blue]",
        )
    )

    if records:
        table = Table(
            title=f"Document Tables ({len(records)} found)",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Table ID", style="bold cyan", width=14)
        table.add_column("Cols", justify="center", style="white", width=8)
        table.add_column("Rows", justify="center", style="yellow", width=8)
        table.add_column("Multipage", justify="center", style="cyan", width=12)
        table.add_column("Continuation", justify="center", style="bold green", width=14)

        for r in records:
            table.add_row(
                r["id"],
                str(r["columns"]),
                str(r["rows"]),
                "Yes" if r["is_multipage"] else "No",
                "Yes" if r["continuation_eligible"] else "No",
            )
        console.print(table)
