"""CLI commands for publication bilingual concordance index generation and inspection."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ubt.cli.ui import emit_json
from ubt.core.index.extractor import extract_index_from_blocks
from ubt.core.index.generator import (
    render_index_html,
    render_index_markdown,
    render_index_typst,
)
from ubt.core.memory.seed_glossary import load_external_glossary
from ubt.core.patch.engine import load_document_blocks

console = Console()
index_app = typer.Typer(
    name="index",
    help="Publication bilingual concordance index generation, term sorting, and inspection.",
    no_args_is_help=True,
)


@index_app.command(name="inspect")
def index_inspect_command(
    file_or_job: Annotated[
        str,
        typer.Argument(
            help="Document file path or SQLite ledger job ID to inspect.",
        ),
    ],
    glossary_path: Annotated[
        Path | None,
        typer.Option(
            "--glossary",
            "-g",
            help="Optional external glossary file (.csv, .tsv, .json) to track terms.",
            exists=True,
            dir_okay=False,
        ),
    ] = None,
    lang: Annotated[
        str,
        typer.Option(
            "--lang",
            "-l",
            help="Primary sorting language ('en' for English-first, 'zh' for Chinese-first).",
        ),
    ] = "en",
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            "-n",
            help="Maximum number of entries to display.",
        ),
    ] = 30,
    min_count: Annotated[
        int,
        typer.Option(
            "--min-count",
            "-m",
            help="Minimum occurrence count threshold.",
        ),
    ] = 1,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output structured JSON results.",
        ),
    ] = False,
) -> None:
    """Inspect mined bilingual terms, occurrences, and document location links."""
    try:
        resolved_src, blocks, _ = asyncio.run(load_document_blocks(file_or_job))
    except Exception as exc:
        console.print(f"[bold red]Error loading blocks from '{file_or_job}': {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    glossary_entries = load_external_glossary(glossary_path) if glossary_path else None
    idx = extract_index_from_blocks(blocks, glossary=glossary_entries, min_occurrences=min_count)

    grouped = (
        idx.grouped_by_source_initial() if lang.lower() == "en" else idx.grouped_by_target_initial()
    )

    if json_output:
        raw_entries = [
            {
                "source_term": e.source_term,
                "target_term": e.target_term,
                "initial_en": e.source_initial,
                "initial_zh": e.target_initial,
                "occurrences": e.occurrences_count,
                "locations": [
                    {
                        "block_id": loc.block_id,
                        "section_title": loc.section_title,
                        "page": loc.page,
                    }
                    for loc in e.locations
                ],
            }
            for e in idx.entries[:limit]
        ]
        emit_json(
            {
                "source": resolved_src,
                "total_terms": idx.total_terms,
                "total_occurrences": idx.total_occurrences,
                "primary_lang": lang,
                "entries": raw_entries,
            }
        )
        return

    table = Table(
        title=f"Bilingual Concordance Index ({lang.upper()}-first, Total {idx.total_terms} terms)",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Initial", style="cyan", width=8)
    table.add_column("Source Term (EN)", style="white", no_wrap=True)
    table.add_column("Target Term (ZH)", style="green", no_wrap=True)
    table.add_column("Hits", justify="right", style="yellow", width=6)
    table.add_column("Sample Locations", style="dim", overflow="fold")

    displayed = 0
    for initial, entries in grouped.items():
        for entry in entries:
            if displayed >= limit:
                break
            loc_summary = ", ".join(
                [loc.section_title[:12] or f"blk-{loc.block_id[:6]}" for loc in entry.locations[:3]]
            )
            table.add_row(
                initial,
                entry.source_term,
                entry.target_term,
                str(entry.occurrences_count),
                loc_summary,
            )
            displayed += 1
        if displayed >= limit:
            break

    console.print(table)


@index_app.command(name="generate")
def index_generate_command(
    file_or_job: Annotated[
        str,
        typer.Argument(
            help="Document file path or SQLite ledger job ID to generate index from.",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Path to write the generated index file.",
        ),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Output syntax format ('typst', 'html', or 'markdown').",
        ),
    ] = "typst",
    glossary_path: Annotated[
        Path | None,
        typer.Option(
            "--glossary",
            "-g",
            help="Optional external glossary file (.csv, .tsv, .json) to track terms.",
            exists=True,
            dir_okay=False,
        ),
    ] = None,
    lang: Annotated[
        str,
        typer.Option(
            "--lang",
            "-l",
            help="Primary sorting language ('en' for English-first, 'zh' for Chinese-first).",
        ),
    ] = "en",
    columns: Annotated[
        int,
        typer.Option(
            "--columns",
            "-c",
            help="Number of columns for multi-column layout.",
        ),
    ] = 3,
    min_count: Annotated[
        int,
        typer.Option(
            "--min-count",
            "-m",
            help="Minimum occurrence count threshold.",
        ),
    ] = 1,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output structured JSON summary with the rendered content.",
        ),
    ] = False,
) -> None:
    """Generate a complete multi-column index section in Typst, HTML, or Markdown."""
    try:
        resolved_src, blocks, _ = asyncio.run(load_document_blocks(file_or_job))
    except Exception as exc:
        console.print(f"[bold red]Error loading blocks from '{file_or_job}': {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    glossary_entries = load_external_glossary(glossary_path) if glossary_path else None
    idx = extract_index_from_blocks(blocks, glossary=glossary_entries, min_occurrences=min_count)

    fmt = output_format.lower().strip()
    if fmt == "typst":
        content = render_index_typst(idx, language_first=lang, columns=columns)
    elif fmt == "html":
        content = render_index_html(idx, language_first=lang, columns=columns)
    elif fmt in {"markdown", "md"}:
        content = render_index_markdown(idx, language_first=lang)
    else:
        console.print(
            f"[bold red]Unsupported format: '{fmt}'. Choose 'typst', 'html', or 'markdown'.[/bold red]"
        )
        raise typer.Exit(code=1)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="utf-8")
        if json_output:
            emit_json(
                {
                    "source": resolved_src,
                    "output_file": str(output),
                    "format": fmt,
                    "total_terms": idx.total_terms,
                    "total_occurrences": idx.total_occurrences,
                }
            )
        else:
            console.print(
                f"[bold green]Successfully generated {fmt.upper()} index to:[/] {output} "
                f"({idx.total_terms} terms)"
            )
        return

    if json_output:
        emit_json(
            {
                "source": resolved_src,
                "format": fmt,
                "total_terms": idx.total_terms,
                "total_occurrences": idx.total_occurrences,
                "content": content,
            }
        )
    else:
        console.print(content)


@index_app.command(name="stats")
def index_stats_command(
    file_or_job: Annotated[
        str,
        typer.Argument(
            help="Document file path or SQLite ledger job ID to analyze.",
        ),
    ],
    glossary_path: Annotated[
        Path | None,
        typer.Option(
            "--glossary",
            "-g",
            help="Optional external glossary file (.csv, .tsv, .json) to track terms.",
            exists=True,
            dir_okay=False,
        ),
    ] = None,
    min_count: Annotated[
        int,
        typer.Option(
            "--min-count",
            "-m",
            help="Minimum occurrence count threshold.",
        ),
    ] = 1,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output structured JSON stats.",
        ),
    ] = False,
) -> None:
    """Analyze term distribution, alphabetical coverage, and concordance metrics."""
    try:
        resolved_src, blocks, _ = asyncio.run(load_document_blocks(file_or_job))
    except Exception as exc:
        console.print(f"[bold red]Error loading blocks from '{file_or_job}': {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    glossary_entries = load_external_glossary(glossary_path) if glossary_path else None
    idx = extract_index_from_blocks(blocks, glossary=glossary_entries, min_occurrences=min_count)

    grouped_en = idx.grouped_by_source_initial()
    grouped_zh = idx.grouped_by_target_initial()
    top_terms = idx.top_terms(5)

    #: Values are heterogeneous (counts, nested group sizes, a term list), so
    #: the element type is declared rather than inferred -- inference from a
    #: literal with mixed value types falls back to ``object``, which then makes
    #: ``stats_data["top_5_terms"]`` unindexable below.
    stats_data: dict[str, Any] = {
        "source": resolved_src,
        "total_terms": idx.total_terms,
        "total_occurrences": idx.total_occurrences,
        "english_letter_groups": {k: len(v) for k, v in grouped_en.items()},
        "chinese_letter_groups": {k: len(v) for k, v in grouped_zh.items()},
        "top_5_terms": [
            {
                "term": f"{e.source_term} / {e.target_term}",
                "count": e.occurrences_count,
            }
            for e in top_terms
        ],
    }

    if json_output:
        emit_json(stats_data)
        return

    top_str = "\n".join(
        f"  • [bold cyan]{t['term']}[/bold cyan]: {t['count']} occurrences"
        for t in stats_data["top_5_terms"]
    )
    console.print(
        Panel(
            f"[bold cyan]Document/Job:[/] {resolved_src}\n"
            f"[bold cyan]Total Indexed Terms:[/] {idx.total_terms}\n"
            f"[bold cyan]Total Occurrences Found:[/] {idx.total_occurrences}\n"
            f"[bold cyan]Active English Groups:[/] {len(grouped_en)} groups ({', '.join(grouped_en.keys())})\n"
            f"[bold cyan]Active Chinese (Pinyin) Groups:[/] {len(grouped_zh)} groups ({', '.join(grouped_zh.keys())})\n\n"
            f"[bold yellow]Top Frequent Terms:[/bold yellow]\n{top_str or '  None'}",
            title="[bold green]Bilingual Concordance Index Stats[/bold green]",
        )
    )
