"""CLI commands for hierarchical scoped glossary auditing and conflict resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ubt.cli.ui import emit_json
from ubt.core.memory.scoped_glossary import (
    GlossaryConflictVerdict,
    ScopeContext,
    audit_glossary_conflicts,
    resolve_effective_glossary,
)
from ubt.core.memory.seed_glossary import load_external_glossary

console = Console()
glossary_app = typer.Typer(
    name="glossary",
    help="Audit, disambiguate, and resolve hierarchical scoped glossaries.",
    no_args_is_help=True,
)


@glossary_app.command(name="audit")
def glossary_audit_command(
    file_path: Annotated[
        Path,
        typer.Argument(
            help="Path to external glossary file (.csv, .tsv, .json) to audit.",
            exists=True,
            dir_okay=False,
        ),
    ],
    strict: Annotated[
        bool,
        typer.Option(
            "--strict",
            "-s",
            help="Exit with non-zero code if unresolved ambiguous conflicts exist.",
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output machine-readable JSON conflict report.",
        ),
    ] = False,
) -> None:
    """Audit glossary file for polysemy, homograph conflicts, and scope coverage."""
    entries = load_external_glossary(file_path)
    if not entries:
        console.print(f"[bold red]Error:[/] No valid glossary entries found in {file_path}")
        raise typer.Exit(code=1)

    summary = audit_glossary_conflicts(entries)

    if json_output:
        emit_json(summary.model_dump())
        if strict and not summary.is_clean:
            raise typer.Exit(code=1)
        return

    # Terminal Dashboard
    status_color = "green" if summary.is_clean else "yellow"
    status_label = (
        "CLEAN (ALL CONFLICTS RESOLVED)"
        if summary.is_clean
        else f"ATTENTION ({summary.unresolved_conflicts} UNRESOLVED)"
    )

    overview_table = Table(box=None, show_header=False, padding=(0, 2))
    overview_table.add_column("Key", style="bold cyan")
    overview_table.add_column("Value")

    overview_table.add_row("Glossary File", str(file_path))
    overview_table.add_row("Total Entries", str(summary.total_terms))
    overview_table.add_row("Unique Sources", str(summary.unique_sources))
    overview_table.add_row("Homograph Sources", str(summary.conflicting_sources))
    overview_table.add_row("Resolved by Scope/Context", f"[green]{summary.resolved_conflicts}[/]")
    overview_table.add_row(
        "Unresolved Ambiguities",
        "[green]0[/]"
        if summary.unresolved_conflicts == 0
        else f"[bold red]{summary.unresolved_conflicts}[/]",
    )

    console.print(
        Panel(
            overview_table,
            title=f"[bold]Glossary Scope & Conflict Audit: [{status_color}]{status_label}[/{status_color}][/bold]",
            subtitle="Universal Book Translator Terminology Arbiter",
            border_style=status_color,
        )
    )

    if summary.conflicts:
        conflicts_table = Table(
            title="Discovered Homographs & Arbitration Analysis",
            show_header=True,
            header_style="bold magenta",
        )
        conflicts_table.add_column("Source Term", style="bold")
        conflicts_table.add_column("Competing Translations")
        conflicts_table.add_column("Assigned Scopes")
        conflicts_table.add_column("Verdict")
        conflicts_table.add_column("Arbitration Reason", style="dim")

        for item in summary.conflicts:
            v_color = (
                "green"
                if item.verdict != GlossaryConflictVerdict.UNRESOLVED_AMBIGUOUS
                else "bold red"
            )
            conflicts_table.add_row(
                item.source,
                " | ".join(item.competing_targets),
                ", ".join(item.scopes),
                f"[{v_color}]{item.verdict.value}[/{v_color}]",
                item.reason,
            )

        console.print(conflicts_table)

    if strict and not summary.is_clean:
        raise typer.Exit(code=1)


@glossary_app.command(name="resolve")
def glossary_resolve_command(
    file_path: Annotated[
        Path,
        typer.Argument(
            help="Path to external glossary file (.csv, .tsv, .json) to resolve.",
            exists=True,
            dir_okay=False,
        ),
    ],
    chapter: Annotated[
        int | None,
        typer.Option(
            "--chapter",
            "-c",
            help="Active chapter index to resolve scope inheritance for.",
        ),
    ] = None,
    section: Annotated[
        str | None,
        typer.Option(
            "--section",
            "-s",
            help="Active section identifier to resolve scope inheritance for.",
        ),
    ] = None,
    domain: Annotated[
        str | None,
        typer.Option(
            "--domain",
            "-d",
            help="Active subject domain to resolve scope inheritance for.",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output resolved glossary mappings as JSON.",
        ),
    ] = False,
) -> None:
    """Resolve effective unambiguous glossary for a target chapter/section scope."""
    entries = load_external_glossary(file_path)
    if not entries:
        console.print(f"[bold red]Error:[/] No valid glossary entries found in {file_path}")
        raise typer.Exit(code=1)

    ctx = ScopeContext(
        chapter_index=chapter,
        section_index=section,
        domain=domain,
    )
    effective = resolve_effective_glossary(entries, ctx)

    if json_output:
        emit_json(effective)
        return

    table = Table(
        title=f"Resolved Effective Glossary (Scope: ch={chapter or '*'}, sec={section or '*'}, dom={domain or '*'})",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Source Term", style="bold")
    table.add_column("Target Translation", style="bold green")
    table.add_column("Active Scope", style="dim")
    table.add_column("Kind", style="dim")

    for item in effective:
        table.add_row(
            item.get("source", ""),
            item.get("translation", "") or item.get("target", ""),
            str(item.get("scope", "*")),
            str(item.get("kind", "term")),
        )

    console.print(table)


__all__ = ["glossary_app", "glossary_audit_command", "glossary_resolve_command"]
