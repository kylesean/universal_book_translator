"""CLI commands for semantic document diffing, incremental patching, and ledger seeding."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from ubt.core.patch.differ import DocumentSemanticDiffer
from ubt.core.patch.engine import BookPatchEngine

console = Console()
patch_app = typer.Typer(
    name="patch",
    help="Semantic document diffing and incremental translation patch management.",
    no_args_is_help=True,
)


@patch_app.command(name="diff")
def patch_diff_command(
    base: Annotated[
        str,
        typer.Argument(
            help="Base source file path or active SQLite ledger job ID.",
        ),
    ],
    revised: Annotated[
        Path,
        typer.Argument(
            help="Revised document file path.",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    min_similarity: Annotated[
        float,
        typer.Option(
            "--min-similarity",
            "-s",
            help="Minimum similarity threshold (0.0 to 1.0) to align modified blocks.",
        ),
    ] = 0.40,
    output_patch: Annotated[
        Path | None,
        typer.Option(
            "--output-patch",
            "-o",
            help="Optional file path to export structured patch summary JSON.",
        ),
    ] = None,
    markdown: Annotated[
        bool,
        typer.Option(
            "--markdown",
            "-m",
            help="Output report in Markdown table format.",
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            "-j",
            help="Output structured JSON payload.",
        ),
    ] = False,
) -> None:
    """Compare a base document/job with a revised document and calculate reusable delta."""
    differ = DocumentSemanticDiffer(min_similarity=min_similarity)
    engine = BookPatchEngine(differ=differ)

    try:
        summary = asyncio.run(engine.diff_documents(base, revised))
    except Exception as exc:
        console.print(f"[bold red]Failed to compute semantic diff:[/] {exc}")
        raise typer.Exit(code=1) from exc

    if output_patch is not None:
        output_patch.write_text(differ.render_diff_json(summary), encoding="utf-8")
        console.print(f"[green]Saved patch summary JSON to:[/] {output_patch}")

    if json_output:
        print(differ.render_diff_json(summary))
    elif markdown:
        print(differ.render_diff_markdown(summary))
    else:
        differ.render_diff_terminal(summary, console=console)


@patch_app.command(name="apply")
def patch_apply_command(
    base: Annotated[
        str,
        typer.Argument(
            help="Base source file path or active SQLite ledger job ID to inherit translations from.",
        ),
    ],
    revised: Annotated[
        Path,
        typer.Argument(
            help="Revised document file path to patch.",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    new_job_id: Annotated[
        str | None,
        typer.Option(
            "--new-job-id",
            "-j",
            help="Explicit target job ID for the newly seeded ledger job.",
        ),
    ] = None,
    target_lang: Annotated[
        str,
        typer.Option(
            "--target-lang",
            "-t",
            help="Target translation language code (default: zh).",
        ),
    ] = "zh",
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Simulate patch and output metrics without committing to ledger.",
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output structured JSON payload.",
        ),
    ] = False,
) -> None:
    """Apply incremental translation patch to seed a new job ledger with 100% reused unchanged blocks."""
    differ = DocumentSemanticDiffer()
    engine = BookPatchEngine(differ=differ)

    if dry_run:
        try:
            summary = asyncio.run(engine.diff_documents(base, revised))
        except Exception as exc:
            console.print(f"[bold red]Patch simulation failed:[/] {exc}")
            raise typer.Exit(code=1) from exc

        if json_output:
            print(differ.render_diff_json(summary))
        else:
            console.print("[yellow]Dry-run simulation only: no changes written to ledger.[/]")
            differ.render_diff_terminal(summary, console=console)
        return

    try:
        actual_job_id, summary = asyncio.run(
            engine.apply_patch(
                base_source=base,
                revised_path=revised,
                new_job_id=new_job_id,
                target_lang=target_lang,
            )
        )
    except Exception as exc:
        console.print(f"[bold red]Failed to apply patch:[/] {exc}")
        raise typer.Exit(code=1) from exc

    if json_output:
        res = {
            "status": "success",
            "job_id": actual_job_id,
            "summary": summary.model_dump(),
        }
        import json

        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        con_msg = (
            f"[bold green]✓ Incremental Patch Successfully Seeded![/]\n\n"
            f"[bold]New Job ID:[/] [cyan]{actual_job_id}[/]\n"
            f"[bold]Total Revised Blocks:[/] {summary.total_revised_blocks}\n"
            f"[bold green]Reused from Base:[/] {summary.unchanged_count} ({summary.reuse_ratio * 100:.1f}%)\n"
            f"[bold yellow]Delta Pending Translation:[/] {summary.modified_count + summary.added_count} blocks\n\n"
            f"[bold]Next Step:[/] Run translation on delta blocks:\n"
            f"  [cyan]ubt translate {revised} --job-id {actual_job_id}[/]"
        )
        console.print(
            Panel(
                con_msg,
                title="[bold green]UBT Patch Application Completed[/]",
                border_style="green",
            )
        )
