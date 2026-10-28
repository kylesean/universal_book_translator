"""CLI commands for publication colophon generation, CIP cataloging, and ISBN validation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from ubt.core.colophon.generator import (
    extract_colophon_from_blocks,
    render_colophon_html,
    render_colophon_markdown,
    render_colophon_typst,
)
from ubt.core.colophon.isbn import format_isbn, validate_isbn
from ubt.core.patch.engine import load_document_blocks

console = Console()
colophon_app = typer.Typer(
    name="colophon",
    help="Publication colophon generation, CIP cataloging, and ISBN-10/13 validation.",
    no_args_is_help=True,
)


@colophon_app.command(name="inspect")
def colophon_inspect_command(
    file_or_job: Annotated[
        str,
        typer.Argument(
            help="Document file path or SQLite ledger job ID to inspect.",
        ),
    ],
    isbn: Annotated[
        str,
        typer.Option(
            "--isbn",
            "-i",
            help="Optional ISBN to validate against document.",
        ),
    ] = "",
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output structured JSON results.",
        ),
    ] = False,
) -> None:
    """Inspect and infer book publication metadata and CIP cataloging parameters."""
    try:
        resolved_src, blocks, _ = asyncio.run(load_document_blocks(file_or_job))
    except Exception as exc:
        console.print(f"[bold red]Error loading blocks from '{file_or_job}': {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    meta = extract_colophon_from_blocks(blocks, isbn=isbn)

    result_data = {
        "source": resolved_src,
        "title": meta.title,
        "authors": meta.authors,
        "translators": meta.translators,
        "publisher": meta.publisher,
        "publication_date": meta.publication_date,
        "edition": meta.edition,
        "impression": meta.impression,
        "isbn": meta.display_isbn,
        "is_valid_isbn": meta.has_valid_isbn,
        "clc_code": meta.clc_code,
        "subject_headings": meta.subject_headings,
        "estimated_words": meta.estimated_words,
        "sheets": meta.sheets,
        "price": meta.price,
    }

    if json_output:
        console.print(json.dumps(result_data, indent=2, ensure_ascii=False), soft_wrap=True)
        return

    # Terminal output
    isbn_color = "green" if meta.has_valid_isbn else "yellow"
    console.print(
        Panel(
            f"[bold cyan]Document/Job:[/bold cyan] {resolved_src}\n"
            f"[bold cyan]Title:[/bold cyan] {meta.title}\n"
            f"[bold cyan]Author(s):[/bold cyan] {meta.authors_display}  |  "
            f"[bold cyan]Translator(s):[/bold cyan] {meta.translators_display or '未指定'}\n"
            f"[bold cyan]Publisher:[/bold cyan] {meta.publisher} ({meta.publication_date})\n"
            f"[bold cyan]ISBN:[/bold cyan] [{isbn_color}]{meta.display_isbn}[/{isbn_color}] "
            f"({'✓ Valid Checksum' if meta.has_valid_isbn else '⚠ Non-standard / Unchecked'})\n"
            f"[bold cyan]Classification:[/bold cyan] {meta.clc_code} ({' · '.join(meta.subject_headings)})\n"
            f"[bold cyan]Scale:[/bold cyan] {meta.estimated_words} words ({meta.sheets} sheets)  |  "
            f"[bold cyan]Price:[/bold cyan] {meta.price}",
            title="[bold blue]Publication Colophon & CIP Data Inspector[/bold blue]",
            border_style="cyan",
        )
    )


@colophon_app.command(name="generate")
def colophon_generate_command(
    file_or_job: Annotated[
        str,
        typer.Argument(
            help="Document file path or SQLite ledger job ID.",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output file path. If omitted, prints to stdout.",
        ),
    ] = None,
    title: Annotated[
        str,
        typer.Option(
            "--title",
            "-t",
            help="Override book title.",
        ),
    ] = "",
    author: Annotated[
        str,
        typer.Option(
            "--author",
            "-a",
            help="Override author name.",
        ),
    ] = "",
    translator: Annotated[
        str,
        typer.Option(
            "--translator",
            help="Override translator name.",
        ),
    ] = "",
    publisher: Annotated[
        str,
        typer.Option(
            "--publisher",
            "-p",
            help="Override publisher name.",
        ),
    ] = "",
    isbn: Annotated[
        str,
        typer.Option(
            "--isbn",
            "-i",
            help="Override ISBN code.",
        ),
    ] = "",
    output_format: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Output format: 'typst', 'html', 'markdown', or 'json'.",
        ),
    ] = "typst",
) -> None:
    """Generate publication colophon and CIP cataloging page."""
    try:
        resolved_src, blocks, _ = asyncio.run(load_document_blocks(file_or_job))
    except Exception as exc:
        console.print(f"[bold red]Error loading blocks from '{file_or_job}': {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    meta = extract_colophon_from_blocks(
        blocks,
        default_title=title,
        default_author=author,
        default_translator=translator,
        isbn=isbn,
    )
    if publisher:
        meta.publisher = publisher

    fmt = output_format.lower()
    if fmt == "typst":
        content = render_colophon_typst(meta)
    elif fmt == "html":
        content = render_colophon_html(meta)
    elif fmt in ("markdown", "md"):
        content = render_colophon_markdown(meta)
    elif fmt == "json":
        data = {
            "title": meta.title,
            "authors": meta.authors,
            "translators": meta.translators,
            "publisher": meta.publisher,
            "isbn": meta.display_isbn,
            "clc_code": meta.clc_code,
            "words": meta.estimated_words,
            "sheets": meta.sheets,
            "price": meta.price,
        }
        content = json.dumps(data, indent=2, ensure_ascii=False)
    else:
        console.print(f"[bold red]Unsupported format: '{output_format}'[/bold red]")
        raise typer.Exit(code=1)

    if output:
        output.write_text(content, encoding="utf-8")
        console.print(f"[bold green]✓ Colophon ({fmt}) saved to:[/bold green] {output}")
    else:
        console.print(content)


@colophon_app.command(name="validate")
def colophon_validate_isbn_command(
    isbn_code: Annotated[
        str,
        typer.Argument(
            help="ISBN-10 or ISBN-13 string to validate.",
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
    """Validate an ISBN code checksum and format standard hyphenation."""
    is_valid = validate_isbn(isbn_code)
    formatted = format_isbn(isbn_code)

    result = {
        "input": isbn_code,
        "is_valid": is_valid,
        "formatted": formatted,
        "length": len(isbn_code.replace("-", "").replace(" ", "")),
    }

    if json_output:
        console.print(json.dumps(result, indent=2), soft_wrap=True)
        if not is_valid:
            raise typer.Exit(code=1)
        return

    if is_valid:
        console.print(f"[bold green]✓ Valid ISBN:[/bold green] {formatted}")
    else:
        console.print(f"[bold red]✗ Invalid ISBN Checksum:[/bold red] '{isbn_code}'")
        raise typer.Exit(code=1)
