"""CLI commands for cross-chapter coreference resolution, entity graph inspection, and pronoun audit."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ubt.cli.ui import emit_json
from ubt.core.memory.coreference import (
    CoreferencePronounType,
    CoreferenceResolver,
)
from ubt.core.memory.seed_glossary import load_external_glossary

console = Console()
coref_app = typer.Typer(
    name="coref",
    help="Cross-chapter entity coreference resolution and pronoun naturalness audit.",
    no_args_is_help=True,
)


@coref_app.command(name="inspect")
def coref_inspect_command(
    file_path: Annotated[
        Path,
        typer.Argument(
            help="Path to source text or markdown file (.md, .txt) to inspect pronouns for.",
            exists=True,
            dir_okay=False,
        ),
    ],
    glossary: Annotated[
        Path | None,
        typer.Option(
            "--glossary",
            "-g",
            help="Optional external glossary file (.csv, .json) to extract domain concepts.",
        ),
    ] = None,
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            "-n",
            help="Maximum number of mentions to display in terminal table.",
        ),
    ] = 30,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output machine-readable JSON resolution results.",
        ),
    ] = False,
) -> None:
    """Inspect pronouns and demonstratives, resolving them to candidate antecedents."""
    text = file_path.read_text(encoding="utf-8")
    if not text.strip():
        console.print(f"[bold red]Error:[/] File {file_path} is empty.")
        raise typer.Exit(code=1)

    glossary_terms: list[str] = []
    if glossary and glossary.exists():
        entries = load_external_glossary(glossary)
        glossary_terms = [e.source for e in entries if e.source]

    resolver = CoreferenceResolver()
    mentions = resolver.resolve_mentions(text, glossary_terms=glossary_terms)

    if json_output:
        data = [m.model_dump() for m in mentions]
        emit_json(data)
        return

    resolved_count = sum(1 for m in mentions if m.inferred_antecedent)
    table = Table(
        title=f"Coreference Mention Disambiguation ({min(len(mentions), limit)} of {len(mentions)} shown, {resolved_count} resolved)",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Span", style="bold cyan")
    table.add_column("Type", style="dim")
    table.add_column("Resolved Antecedent", style="bold green")
    table.add_column("Conf", justify="right")
    table.add_column("Source", style="dim")
    table.add_column("Recommended Chinese Rendering", style="bold yellow")

    type_colors = {
        CoreferencePronounType.PERSONAL: "blue",
        CoreferencePronounType.DEMONSTRATIVE: "green",
        CoreferencePronounType.RELATIONAL: "magenta",
    }

    for m in mentions[:limit]:
        t_color = type_colors.get(m.pronoun_type, "white")
        table.add_row(
            m.span,
            f"[{t_color}]{m.pronoun_type.value}[/{t_color}]",
            m.inferred_antecedent or "[dim]unresolved[/dim]",
            f"{m.confidence:.2f}" if m.confidence > 0 else "-",
            m.antecedent_source or "-",
            m.recommended_chinese_rendering or "-",
        )

    console.print(table)


@coref_app.command(name="audit")
def coref_audit_command(
    file_path: Annotated[
        Path,
        typer.Argument(
            help="Path to translated Chinese text file to audit for '它-ism' and pronoun overuse.",
            exists=True,
            dir_okay=False,
        ),
    ],
    threshold: Annotated[
        float,
        typer.Option(
            "--threshold",
            "-t",
            help="Pronoun density threshold per 1,000 Chinese characters.",
        ),
    ] = 18.0,
    strict: Annotated[
        bool,
        typer.Option(
            "--strict",
            "-s",
            help="Exit with non-zero code if pronoun audit fails.",
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output machine-readable JSON audit report.",
        ),
    ] = False,
) -> None:
    """Audit translated text for unidiomatic '它/它们' repetition and pronoun overuse."""
    text = file_path.read_text(encoding="utf-8")
    resolver = CoreferenceResolver(pronoun_density_threshold_per_kilochar=threshold)
    report = resolver.audit_pronoun_usage(text)

    if json_output:
        emit_json(report.model_dump())
        if strict and not report.is_acceptable:
            raise typer.Exit(code=1)
        return

    status_color = "green" if report.is_acceptable else "bold red"
    status_label = (
        "PASSED (NATURAL FLOW)" if report.is_acceptable else "ATTENTION (PRONOUN OVERUSE)"
    )

    overview_table = Table(box=None, show_header=False, padding=(0, 2))
    overview_table.add_column("Metric", style="bold cyan")
    overview_table.add_column("Value")

    overview_table.add_row("Total Chinese Characters", str(report.total_chinese_chars))
    overview_table.add_row("3rd-Person Neuter Pronouns ('它/它们')", str(report.pronoun_count))
    overview_table.add_row(
        "Pronoun Density (per 1k chars)",
        f"{report.pronoun_density_per_kilochar:.2f} (Threshold: {threshold:.1f})",
    )
    overview_table.add_row("Sentences with Repetition", str(len(report.issues)))

    console.print(
        Panel(
            overview_table,
            title=f"[bold]Pronoun Idiomaticity & '它-ism' Audit: [{status_color}]{status_label}[/{status_color}][/bold]",
            subtitle="Universal Book Translator Coreference Engine",
            border_style="green" if report.is_acceptable else "red",
        )
    )

    if report.issues:
        issues_table = Table(title="Pronoun Repetition Hotspots", show_header=True)
        issues_table.add_column("Tokens", style="bold red")
        issues_table.add_column("Sentence Context")
        issues_table.add_column("Suggestion", style="dim")

        for issue in report.issues[:15]:
            issues_table.add_row(
                issue.pronoun_span,
                issue.sentence[:90] + ("..." if len(issue.sentence) > 90 else ""),
                issue.suggestion,
            )
        console.print(issues_table)

    if strict and not report.is_acceptable:
        raise typer.Exit(code=1)


__all__ = ["coref_app", "coref_audit_command", "coref_inspect_command"]
