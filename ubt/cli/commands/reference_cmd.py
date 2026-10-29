"""CLI commands for cross-chapter internal anchors and citation topology inspection."""

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
from ubt.core.references.linker import (
    build_citation_topology,
    format_mention_text,
    inject_hyperlinks,
)
from ubt.core.references.models import (
    CitationTopology,
    CrossRefConfig,
    CrossRefPrefixScheme,
    CrossRefStyle,
    ReferenceKind,
)

console = Console()
ref_app = typer.Typer(
    name="xref",
    help="Cross-reference transformer, anchor mapping, and hyperlink verification.",
    no_args_is_help=True,
)
xref_app = ref_app


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
        emit_json(result_data)
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
        emit_json(report)
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


@ref_app.command(name="transform")
def ref_transform_command(
    file_or_text: Annotated[
        str,
        typer.Argument(
            help="Document file path, ledger job ID, or inline text to transform cross references in.",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output file path to save transformed text to.",
        ),
    ] = None,
    target_format: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Target rendering format: 'typst', 'html', or 'markdown'.",
        ),
    ] = "typst",
    style: Annotated[
        str,
        typer.Option(
            "--style",
            "-s",
            help="Cross-reference style: 'link_only', 'with_page', 'page_only', or 'native_ref'.",
        ),
    ] = "link_only",
    prefix_scheme: Annotated[
        str,
        typer.Option(
            "--prefix-scheme",
            "-p",
            help="Prefix scheme: 'original', 'chinese_standard', 'academic_formal', or 'compact'.",
        ),
    ] = "original",
    show_page_number: Annotated[
        bool,
        typer.Option(
            "--show-pages/--no-show-pages",
            help="Whether to append dynamic page numbers to references.",
        ),
    ] = False,
    lang: Annotated[
        str,
        typer.Option(
            "--lang",
            "-l",
            help="Target language for page/label formatting: 'zh' or 'en'.",
        ),
    ] = "zh",
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output structured JSON results.",
        ),
    ] = False,
) -> None:
    """Transform cross references in a document or text with hyperlinks and custom styling."""
    try:
        resolved_src, blocks, _ = asyncio.run(load_document_blocks(file_or_text))
        topology = build_citation_topology(blocks)
        full_text = "\n\n".join(b.source_text or "" for b in blocks if b.source_text)
    except Exception:
        resolved_src = "inline_text"
        topology = CitationTopology()
        full_text = file_or_text

    try:
        style_enum = CrossRefStyle(style.lower())
    except ValueError:
        style_enum = CrossRefStyle.LINK_ONLY

    try:
        prefix_enum = CrossRefPrefixScheme(prefix_scheme.lower())
    except ValueError:
        prefix_enum = CrossRefPrefixScheme.ORIGINAL

    config = CrossRefConfig(
        style=style_enum,
        prefix_scheme=prefix_enum,
        show_page_number=show_page_number,
        target_format=target_format,
        target_lang=lang,
    )

    transformed = inject_hyperlinks(full_text, topology, target_format=target_format, config=config)

    if output is not None:
        output.write_text(transformed, encoding="utf-8")
        console.print(f"[bold green]Saved transformed document to:[/bold green] {output}")

    if json_output:
        res = {
            "source": resolved_src,
            "format": target_format,
            "style": style_enum.value,
            "prefix_scheme": prefix_enum.value,
            "show_pages": show_page_number,
            "output": transformed,
        }
        emit_json(res)
    elif output is None:
        console.print(transformed, markup=False)


@ref_app.command(name="format-mention")
def ref_format_mention_command(
    text: Annotated[
        str,
        typer.Argument(
            help="Mention text to format (e.g. 'Figure 2', '第 3 节', 'Table 1').",
        ),
    ],
    kind: Annotated[
        str,
        typer.Option(
            "--kind",
            "-k",
            help="Reference kind: 'figure', 'table', 'heading', 'equation', 'page', 'listing', 'algorithm', 'bibliography'.",
        ),
    ] = "figure",
    key: Annotated[
        str,
        typer.Option(
            "--key",
            help="Anchor key (e.g. '2', '3.1'). If not provided, inferred from text.",
        ),
    ] = "",
    prefix_scheme: Annotated[
        str,
        typer.Option(
            "--prefix-scheme",
            "-p",
            help="Prefix scheme: 'original', 'chinese_standard', 'academic_formal', 'compact'.",
        ),
    ] = "chinese_standard",
    lang: Annotated[
        str,
        typer.Option(
            "--lang",
            "-l",
            help="Target language ('zh' or 'en').",
        ),
    ] = "zh",
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output structured JSON results.",
        ),
    ] = False,
) -> None:
    """Format an individual cross-reference mention according to standard citation styles."""
    import re

    lookup_key = key
    if not lookup_key:
        m = re.search(r"(\d+(?:\.\d+)*[a-z]?)", text)
        if m:
            lookup_key = m.group(1)
        else:
            m2 = re.search(r"([A-Za-z0-9_\-]+)\s*$", text)
            lookup_key = m2.group(1) if m2 else text

    try:
        kind_enum = ReferenceKind(kind.lower())
    except ValueError:
        kind_enum = ReferenceKind.FIGURE

    try:
        prefix_enum = CrossRefPrefixScheme(prefix_scheme.lower())
    except ValueError:
        prefix_enum = CrossRefPrefixScheme.CHINESE_STANDARD

    formatted = format_mention_text(
        text,
        kind_enum,
        lookup_key,
        prefix_enum,
        lang=lang,
    )

    if json_output:
        res = {
            "original_text": text,
            "kind": kind_enum.value,
            "key": lookup_key,
            "prefix_scheme": prefix_enum.value,
            "lang": lang,
            "formatted_text": formatted,
        }
        emit_json(res)
    else:
        console.print(formatted, markup=False)


__all__ = [
    "ref_app",
    "ref_check_command",
    "ref_format_mention_command",
    "ref_inspect_command",
    "ref_transform_command",
    "xref_app",
]
