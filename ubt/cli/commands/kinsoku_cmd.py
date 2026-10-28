"""CLI commands for CJK typesetting line-breaking (Kinsoku Shori) and punctuation compression."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ubt.core.cleaners.kinsoku import (
    WORD_JOINER,
    KinsokuAuditReport,
    audit_kinsoku_typography,
    fix_kinsoku_prose,
)
from ubt.core.patch.engine import load_document_blocks

console = Console()
kinsoku_app = typer.Typer(
    name="kinsoku",
    help="CJK line-breaking rules (Kinsoku Shori), punctuation compression, and typography audit.",
    no_args_is_help=True,
)


def _extract_full_text(blocks: Any) -> str:
    """Concatenate target_text or source element text across all blocks."""
    lines: list[str] = []
    for b in blocks:
        text = getattr(b, "target_text", None)
        if not text and hasattr(b, "element") and b.element is not None:
            text = getattr(b.element, "text", "")
        if text and text.strip():
            lines.append(text)
    return "\n\n".join(lines)


@kinsoku_app.command(name="inspect")
def kinsoku_inspect_command(
    file_or_job: Annotated[
        str,
        typer.Argument(
            help="Document file path or SQLite ledger job ID to inspect typography in.",
        ),
    ],
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            "-n",
            help="Maximum number of violations to display in terminal table.",
        ),
    ] = 25,
    target_lang: Annotated[
        str,
        typer.Option(
            "--lang",
            "-l",
            help="Target language code (e.g. 'zh' or 'ja').",
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
    """Inspect CJK typesetting line-breaking and punctuation compression violations."""
    try:
        resolved_src, blocks, _ = asyncio.run(load_document_blocks(file_or_job))
    except Exception as exc:
        console.print(f"[bold red]Error loading blocks from '{file_or_job}': {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    full_text = _extract_full_text(blocks)
    report: KinsokuAuditReport = audit_kinsoku_typography(full_text, target_lang=target_lang)

    violations_data: list[dict[str, Any]] = [
        {
            "line": v.line_number,
            "column": v.column,
            "rule": v.rule.value,
            "char": v.character,
            "snippet": v.snippet,
            "message": v.message,
            "suggested_fix": v.suggested_fix,
        }
        for v in report.violations[:limit]
    ]

    result_data = {
        "source": resolved_src,
        "total_lines": report.total_lines,
        "total_cjk_chars": report.total_cjk_chars,
        "total_violations": len(report.violations),
        "consecutive_punct_count": report.consecutive_punct_count,
        "orphan_tail_count": report.orphan_tail_count,
        "compliance_score": report.compliance_score,
        "is_clean": report.is_clean,
        "violations": violations_data,
    }

    if json_output:
        console.print(json.dumps(result_data, indent=2, ensure_ascii=False), soft_wrap=True)
        return

    # Terminal output
    score_color = (
        "green"
        if report.compliance_score >= 95.0
        else ("yellow" if report.compliance_score >= 80.0 else "red")
    )
    console.print(
        Panel(
            f"[bold cyan]Document/Job:[/bold cyan] {resolved_src}\n"
            f"[bold cyan]Total Lines:[/bold cyan] {report.total_lines}  |  "
            f"[bold cyan]CJK Characters:[/bold cyan] {report.total_cjk_chars}\n"
            f"[bold cyan]Compliance Score:[/bold cyan] [{score_color}]{report.compliance_score}%[/{score_color}]\n"
            f"[bold cyan]Violations:[/bold cyan] {len(report.violations)}  |  "
            f"[bold cyan]Consecutive Punct Gaps:[/bold cyan] {report.consecutive_punct_count}  |  "
            f"[bold cyan]Orphan Tails:[/bold cyan] {report.orphan_tail_count}",
            title="[bold blue]CJK Typography & Kinsoku Audit[/bold blue]",
            border_style=score_color,
        )
    )

    if report.violations:
        table = Table(title="[bold yellow]Typography & Line-Breaking Violations[/bold yellow]")
        table.add_column("Line", style="cyan", width=6)
        table.add_column("Rule", style="magenta", width=24)
        table.add_column("Char", style="bold red", width=6)
        table.add_column("Snippet", style="white")
        table.add_column("Suggested Fix", style="green")

        for v in report.violations[:limit]:
            table.add_row(
                str(v.line_number),
                v.rule.value,
                v.character,
                v.snippet,
                v.suggested_fix,
            )
        console.print(table)
        if len(report.violations) > limit:
            console.print(
                f"[dim](Showing {limit} of {len(report.violations)} violations. Use --limit to expand)[/dim]"
            )


@kinsoku_app.command(name="audit")
def kinsoku_audit_command(
    file_or_job: Annotated[
        str,
        typer.Argument(
            help="Document file path or SQLite ledger job ID to audit.",
        ),
    ],
    min_score: Annotated[
        float,
        typer.Option(
            "--min-score",
            "-s",
            help="Minimum acceptable typography compliance score percentage (0-100).",
        ),
    ] = 90.0,
    strict: Annotated[
        bool,
        typer.Option(
            "--strict",
            help="Fail if any typography violations exist (0 tolerance).",
        ),
    ] = False,
    target_lang: Annotated[
        str,
        typer.Option(
            "--lang",
            "-l",
            help="Target language code (e.g. 'zh' or 'ja').",
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
    """Gate check for CI/CD publishing: verify CJK typography quality compliance."""
    try:
        resolved_src, blocks, _ = asyncio.run(load_document_blocks(file_or_job))
    except Exception as exc:
        console.print(f"[bold red]Error loading blocks from '{file_or_job}': {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    full_text = _extract_full_text(blocks)
    report = audit_kinsoku_typography(full_text, target_lang=target_lang)

    passed = (report.compliance_score >= min_score) and (not strict or report.is_clean)
    result = {
        "source": resolved_src,
        "compliance_score": report.compliance_score,
        "min_score": min_score,
        "strict": strict,
        "violations_count": len(report.violations),
        "passed": passed,
    }

    if json_output:
        console.print(json.dumps(result, indent=2), soft_wrap=True)
        if not passed:
            raise typer.Exit(code=1)
        return

    if passed:
        console.print(
            f"[bold green]✓ Typography Gate Passed:[/bold green] score={report.compliance_score}% (>= {min_score}%), "
            f"violations={len(report.violations)}"
        )
    else:
        console.print(
            f"[bold red]✗ Typography Gate FAILED:[/bold red] score={report.compliance_score}% (required >= {min_score}%), "
            f"violations={len(report.violations)}"
        )
        raise typer.Exit(code=1)


@kinsoku_app.command(name="fix")
def kinsoku_fix_command(
    file_or_job: Annotated[
        str,
        typer.Argument(
            help="Document file path to format and repair typography in.",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output file path to save repaired text. If omitted, prints to stdout.",
        ),
    ] = None,
    target_lang: Annotated[
        str,
        typer.Option(
            "--lang",
            "-l",
            help="Target language code (e.g. 'zh' or 'ja').",
        ),
    ] = "zh",
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output structured repair statistics JSON.",
        ),
    ] = False,
) -> None:
    """Repair CJK line-breaking issues, compress punctuation, and inject Word Joiners."""
    try:
        resolved_src, blocks, _ = asyncio.run(load_document_blocks(file_or_job))
    except Exception as exc:
        console.print(f"[bold red]Error loading blocks from '{file_or_job}': {exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    original_text = _extract_full_text(blocks)
    fixed_text = fix_kinsoku_prose(original_text, target_lang=target_lang)

    joiners_injected = fixed_text.count(WORD_JOINER)
    chars_diff = len(fixed_text) - len(original_text)

    if output:
        output.write_text(fixed_text, encoding="utf-8")
        if not json_output:
            console.print(f"[bold green]✓ Repaired document saved to {output}[/bold green]")

    stats = {
        "source": resolved_src,
        "joiners_injected": joiners_injected,
        "characters_delta": chars_diff,
        "saved_path": str(output) if output else None,
    }

    if json_output:
        console.print(json.dumps(stats, indent=2), soft_wrap=True)
    elif not output:
        console.print(fixed_text)
