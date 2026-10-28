"""CLI commands for cross-chapter internal anchors and citation topology inspection."""

from __future__ import annotations

import asyncio
import json
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ubt.core.patch.engine import load_document_blocks
from ubt.core.references.linker import build_citation_topology

console = Console()
ref_app = typer.Typer(
    name="ref",
    help="Cross-chapter citation topology, anchor mapping, and hyperlink verification.",
    no_args_is_help=True,
)


@ref_app.command(name="inspect")
def ref_inspect_command(
    file_or_job: Annotated[
        str,
        typer.Argument(
            help="Document file path or SQLite ledger job ID to inspect references in.",
        ),
    ],
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            "-n",
            help="Maximum number of anchors and links to display in terminal tables.",
        ),
    ] = 25,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output structured JSON results.",
        ),
    ] = False,
) -> None:
    """Inspect cross-chapter citation topology, anchors, and resolved hyperlinks."""
    try:
        resolved_src, blocks, _ = asyncio.run(load_document_blocks(file_or_job))
    except Exception as exc:
        console.print(f"[bold red]Error loading blocks from '{file_or_job}': {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    topology = build_citation_topology(blocks)

    anchors_data: list[dict[str, Any]] = [
        {
            "id": a.anchor_id,
            "kind": a.kind.value,
            "key": a.key,
            "page": a.page,
            "title": a.title,
        }
        for a in list(topology.anchors.values())[:limit]
    ]

    links_data: list[dict[str, Any]] = [
        {
            "source_block": lnk.source_block_id,
            "target_anchor": lnk.target_anchor_id,
            "text": lnk.matched_text,
            "kind": lnk.kind.value,
        }
        for lnk in topology.links[:limit]
    ]

    res_rate_pct = topology.resolution_rate * 100.0
    result_data = {
        "source": resolved_src,
        "total_anchors": len(topology.anchors),
        "total_citations_found": topology.total_citations_found,
        "resolved_citations_count": topology.resolved_citations_count,
        "resolution_rate_percent": round(res_rate_pct, 2),
        "anchors": anchors_data,
        "resolved_links": links_data,
        "unresolved_mentions": topology.unresolved_mentions[:limit],
    }

    if json_output:
        console.print(json.dumps(result_data, indent=2, ensure_ascii=False))
        return

    # Terminal output
    color = "green" if res_rate_pct >= 80 else "yellow"
    console.print(
        Panel(
            f"[bold cyan]Document/Job:[/bold cyan] {resolved_src}\n"
            f"[bold cyan]Total Anchors Discovered:[/bold cyan] {len(topology.anchors)}\n"
            f"[bold cyan]Prose Citation Mentions:[/bold cyan] {topology.total_citations_found}\n"
            f"[bold cyan]Resolved Hyperlinks:[/bold cyan] {topology.resolved_citations_count} "
            f"([{color}]{res_rate_pct:.1f}%[/{color}])\n"
            f"[bold cyan]Unresolved References:[/bold cyan] {len(topology.unresolved_mentions)}",
            title="[bold green]Document Citation & Internal Link Topology[/bold green]",
            expand=False,
        )
    )

    if topology.anchors:
        anchors_tbl = Table(title="Registered Jump Anchors (Sample)", show_header=True)
        anchors_tbl.add_column("Anchor ID", style="green")
        anchors_tbl.add_column("Kind", style="cyan")
        anchors_tbl.add_column("Key", style="yellow")
        anchors_tbl.add_column("Page", style="magenta")
        anchors_tbl.add_column("Snippet", style="dim")

        for a in list(topology.anchors.values())[:limit]:
            anchors_tbl.add_row(
                a.anchor_id,
                a.kind.value,
                a.key,
                str(a.page or "?"),
                a.title[:40],
            )
        console.print(anchors_tbl)

    if topology.links:
        links_tbl = Table(title="Resolved Active Prose Links (Sample)", show_header=True)
        links_tbl.add_column("Source Block", style="green")
        links_tbl.add_column("Matched Prose Text", style="yellow")
        links_tbl.add_column("Target Anchor", style="cyan")
        links_tbl.add_column("Kind", style="magenta")

        for lnk in topology.links[:limit]:
            links_tbl.add_row(
                lnk.source_block_id,
                lnk.matched_text,
                lnk.target_anchor_id,
                lnk.kind.value,
            )
        console.print(links_tbl)


@ref_app.command(name="check")
def ref_check_command(
    file_or_job: Annotated[
        str,
        typer.Argument(
            help="Document file path or SQLite ledger job ID to check reference validity in.",
        ),
    ],
    min_resolution_rate: Annotated[
        float,
        typer.Option(
            "--min-rate",
            "-r",
            help="Minimum acceptable resolution percentage (0-100) before failing CI gate.",
        ),
    ] = 70.0,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output structured JSON gate report.",
        ),
    ] = False,
) -> None:
    """Validate internal citation integrity and gate against broken references."""
    try:
        resolved_src, blocks, _ = asyncio.run(load_document_blocks(file_or_job))
    except Exception as exc:
        console.print(f"[bold red]Error loading blocks from '{file_or_job}': {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    topology = build_citation_topology(blocks)
    rate_pct = topology.resolution_rate * 100.0
    passed = rate_pct >= min_resolution_rate

    report = {
        "source": resolved_src,
        "total_citations": topology.total_citations_found,
        "resolved_citations": topology.resolved_citations_count,
        "resolution_rate_percent": round(rate_pct, 2),
        "min_required_rate_percent": min_resolution_rate,
        "passed": passed,
        "unresolved_count": len(topology.unresolved_mentions),
        "unresolved_samples": topology.unresolved_mentions[:10],
    }

    if json_output:
        console.print(json.dumps(report, indent=2, ensure_ascii=False))
    elif passed:
        console.print(
            f"[bold green]PASS:[/bold green] Citation resolution rate {rate_pct:.1f}% "
            f"meets required gate ({min_resolution_rate:.1f}%)."
        )
    else:
        console.print(
            f"[bold red]FAIL:[/bold red] Citation resolution rate {rate_pct:.1f}% "
            f"is below required gate ({min_resolution_rate:.1f}%). "
            f"Found {len(topology.unresolved_mentions)} unresolved references."
        )

    if not passed:
        raise typer.Exit(code=1)
