"""CLI commands for book appendix and backmatter inspection, validation, and generation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ubt.core.backmatter.detector import detect_backmatter_manifest
from ubt.core.backmatter.generator import (
    emit_appendix_switch_typst,
    render_backmatter_heading_typst,
    render_backmatter_section_html,
    render_backmatter_skeleton_markdown,
)
from ubt.core.backmatter.models import BackmatterConfig
from ubt.core.patch.engine import load_document_blocks

console = Console()
backmatter_app = typer.Typer(
    name="backmatter",
    help="Appendix & Backmatter Layout Manager: partitions, sequential letters, and publishing switches.",
    no_args_is_help=True,
)


@backmatter_app.command(name="inspect")
def backmatter_inspect_command(
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
    """Inspect all detected backmatter partitions, appendix letters, and block coverage."""
    try:
        resolved_src, blocks, _ = asyncio.run(load_document_blocks(file_or_job))
    except Exception as exc:
        console.print(f"[bold red]Error loading blocks from '{file_or_job}': {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    manifest = detect_backmatter_manifest(blocks)

    if json_output:
        res = manifest.to_dict()
        res["source"] = resolved_src
        console.print(json.dumps(res, indent=2, ensure_ascii=False), soft_wrap=True)
        return

    console.print(
        Panel(
            f"[bold cyan]Source Document:[/] {resolved_src}\n"
            f"[bold cyan]Total Backmatter Sections:[/] {manifest.total_sections}\n"
            f"[bold cyan]Appendices Found:[/] {manifest.appendix_count}\n"
            f"[bold cyan]Bibliography Present:[/] {'Yes' if manifest.has_bibliography else 'No'}\n"
            f"[bold cyan]Index Present:[/] {'Yes' if manifest.has_index else 'No'}\n"
            f"[bold cyan]Glossary Present:[/] {'Yes' if manifest.has_glossary else 'No'}\n"
            f"[bold cyan]Afterword Present:[/] {'Yes' if manifest.has_afterword else 'No'}\n"
            f"[bold cyan]First Backmatter Block:[/] {manifest.first_backmatter_block_id or 'None'}",
            title="[bold blue]Document Backmatter Partition Audit[/bold blue]",
        )
    )

    if manifest.sections:
        table = Table(
            title=f"Backmatter Sections ({manifest.total_sections} detected)",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Kind", style="bold cyan", width=14)
        table.add_column("Letter", justify="center", style="bold yellow", width=8)
        table.add_column("Heading ID", style="dim", width=12)
        table.add_column("Raw Title", style="white")
        table.add_column("Clean Title", style="dim")
        table.add_column("Blocks", justify="right", style="cyan", width=8)

        for s in manifest.sections:
            table.add_row(
                s.kind.value.upper(),
                s.letter_index or "-",
                s.heading_block_id,
                s.raw_title,
                s.clean_title,
                str(len(s.block_ids)),
            )
        console.print(table)


@backmatter_app.command(name="generate")
def backmatter_generate_command(
    file_or_job: Annotated[
        str,
        typer.Argument(
            help="Document file path or SQLite ledger job ID to inspect.",
        ),
    ],
    output_path: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Target output file path to write generated backmatter code.",
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
    appendix_numbering: Annotated[
        str,
        typer.Option(
            "--numbering",
            help="Typst heading numbering pattern for appendices (e.g. 'A.1').",
        ),
    ] = "A.1",
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output structured JSON results.",
        ),
    ] = False,
) -> None:
    """Generate backmatter Typst preamble switch, HTML container, or markdown skeleton."""
    try:
        resolved_src, blocks, _ = asyncio.run(load_document_blocks(file_or_job))
    except Exception as exc:
        console.print(f"[bold red]Error loading blocks from '{file_or_job}': {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    cfg = BackmatterConfig(appendix_numbering=appendix_numbering)
    manifest = detect_backmatter_manifest(blocks, cfg)

    fmt = output_format.lower().strip()
    if fmt == "typst":
        chunks: list[str] = []
        if manifest.appendix_count > 0:
            chunks.append(emit_appendix_switch_typst(cfg))
        for s in manifest.sections:
            chunks.append(render_backmatter_heading_typst(s, cfg))
        markup = "\n".join(chunks) + "\n"
    elif fmt == "html":
        chunks = [render_backmatter_section_html(s, config=cfg) for s in manifest.sections]
        markup = "\n".join(chunks) + "\n"
    elif fmt in ("markdown", "md"):
        markup = render_backmatter_skeleton_markdown(manifest.sections)
    else:
        console.print(
            f"[bold red]Unsupported format: '{fmt}'. Choose 'typst', 'html', or 'markdown'.[/bold red]"
        )
        raise typer.Exit(code=1)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markup, encoding="utf-8")
        console.print(
            f"[bold green]Successfully generated backmatter code to '{output_path}'.[/bold green]"
        )
        return

    if json_output:
        console.print(
            json.dumps(
                {
                    "source": resolved_src,
                    "format": fmt,
                    "total_sections": manifest.total_sections,
                    "markup": markup,
                },
                indent=2,
                ensure_ascii=False,
            ),
            soft_wrap=True,
        )
        return

    console.print(markup, soft_wrap=True)


@backmatter_app.command(name="validate")
def backmatter_validate_command(
    file_or_job: Annotated[
        str,
        typer.Argument(
            help="Document file path or SQLite ledger job ID to validate.",
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
    """Validate that backmatter sections and appendix letter sequences follow publishing standards."""
    try:
        resolved_src, blocks, _ = asyncio.run(load_document_blocks(file_or_job))
    except Exception as exc:
        console.print(f"[bold red]Error loading blocks from '{file_or_job}': {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    manifest = detect_backmatter_manifest(blocks)

    errors: list[str] = []
    expected_letter = "A"
    for s in manifest.sections:
        if s.kind.value == "appendix":
            if s.letter_index != expected_letter:
                errors.append(
                    f"Appendix '{s.raw_title}' has letter '{s.letter_index}', but expected '{expected_letter}'."
                )
            expected_letter = chr(ord(expected_letter) + 1)

    passed = len(errors) == 0

    if json_output:
        console.print(
            json.dumps(
                {
                    "source": resolved_src,
                    "passed": passed,
                    "total_sections": manifest.total_sections,
                    "appendix_count": manifest.appendix_count,
                    "errors": errors,
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
            f"[bold green]PASS:[/] Backmatter layout validation successful. "
            f"Found {manifest.total_sections} sections ({manifest.appendix_count} appendices, all in correct letter sequence)."
        )
    else:
        console.print(f"[bold red]FAIL:[/] Found {len(errors)} backmatter sequencing issues:")
        for err in errors:
            console.print(f"  - [red]{err}[/red]")
        raise typer.Exit(code=1)
