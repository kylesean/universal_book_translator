"""CLI commands for Pareto dynamic routing inspection and document workload simulation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from ubt.core.ir.models import BlockType, IRBlock, make_element
from ubt.core.memory.seed_glossary import load_external_glossary
from ubt.core.router.dynamic_scheduler import (
    ComplexityTier,
    DynamicParetoScheduler,
    DynamicRoutingPolicy,
)

console = Console()
route_app = typer.Typer(
    name="route",
    help="Inspect block complexity and simulate Pareto multi-model dynamic routing.",
    no_args_is_help=True,
)


def _extract_blocks_from_text(file_path: Path) -> list[IRBlock]:
    """Extract synthetic or raw blocks from a markdown or text file for routing simulation."""
    text = file_path.read_text(encoding="utf-8")
    raw_blocks: list[str] = [b.strip() for b in text.split("\n\n") if b.strip()]

    ir_blocks: list[IRBlock] = []
    for idx, raw in enumerate(raw_blocks, start=1):
        b_id = f"blk-{idx:04d}"
        if raw.startswith("#"):
            b_type = BlockType.HEADING
        elif raw.startswith("```"):
            b_type = BlockType.CODE
        elif raw.startswith("$$") or raw.startswith("\\["):
            b_type = BlockType.FORMULA
        elif raw.startswith("|") and raw.endswith("|"):
            b_type = BlockType.TABLE
        else:
            b_type = BlockType.NARRATIVE

        element = make_element(
            id=b_id,
            spine_index=idx,
            block_type=b_type,
            source_text=raw,
        )
        ir_blocks.append(IRBlock(element=element))
    return ir_blocks


@route_app.command(name="simulate")
def route_simulate_command(
    file_path: Annotated[
        Path,
        typer.Argument(
            help="Path to source document file (.md, .txt) to simulate routing for.",
            exists=True,
            dir_okay=False,
        ),
    ],
    policy: Annotated[
        str,
        typer.Option(
            "--policy",
            "-p",
            help="Routing policy ('pareto', 'quality_first', 'cost_first').",
        ),
    ] = "pareto",
    budget_model: Annotated[
        str,
        typer.Option(
            "--budget-model",
            help="Budget model for trivial/standard narrative blocks.",
        ),
    ] = "deepseek-chat",
    standard_model: Annotated[
        str,
        typer.Option(
            "--standard-model",
            help="Standard draft model for routine technical blocks.",
        ),
    ] = "deepseek-chat",
    flagship_model: Annotated[
        str,
        typer.Option(
            "--flagship-model",
            help="Flagship model for extreme mathematical/code proofs.",
        ),
    ] = "claude-3-7-sonnet",
    glossary: Annotated[
        Path | None,
        typer.Option(
            "--glossary",
            "-g",
            help="Optional external glossary file (.csv, .json) to extract domain terms.",
        ),
    ] = None,
    markdown_output: Annotated[
        bool,
        typer.Option(
            "--markdown",
            "-m",
            help="Output GitHub Flavored Markdown summary report.",
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output machine-readable JSON simulation report.",
        ),
    ] = False,
) -> None:
    """Simulate document block routing distribution, estimated expenditure, and savings."""
    blocks = _extract_blocks_from_text(file_path)
    if not blocks:
        console.print(f"[bold red]Error:[/] No blocks parsed from {file_path}")
        raise typer.Exit(code=1)

    glossary_terms: list[str] = []
    if glossary and glossary.exists():
        entries = load_external_glossary(glossary)
        glossary_terms = [e.source for e in entries if e.source]

    try:
        policy_enum = DynamicRoutingPolicy(policy.lower())
    except ValueError:
        console.print(
            f"[bold red]Error:[/] Invalid policy '{policy}'. Must be pareto, quality_first, or cost_first."
        )
        raise typer.Exit(code=1) from None

    scheduler = DynamicParetoScheduler(
        policy=policy_enum,
        budget_model=budget_model,
        standard_model=standard_model,
        flagship_model=flagship_model,
    )

    report = scheduler.simulate_document_routing(blocks, glossary_terms=glossary_terms)

    if json_output:
        console.print_json(scheduler.render_simulation_json(report))
        return

    if markdown_output:
        console.print(scheduler.render_simulation_markdown(report))
        return

    scheduler.render_simulation_terminal(report, console=console)


@route_app.command(name="inspect")
def route_inspect_command(
    file_path: Annotated[
        Path,
        typer.Argument(
            help="Path to source document file (.md, .txt) to inspect.",
            exists=True,
            dir_okay=False,
        ),
    ],
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            "-n",
            help="Maximum number of blocks to display.",
        ),
    ] = 25,
    tier: Annotated[
        str | None,
        typer.Option(
            "--tier",
            "-t",
            help="Filter by complexity tier ('trivial', 'standard', 'complex', 'extreme').",
        ),
    ] = None,
    budget_model: Annotated[
        str,
        typer.Option(
            "--budget-model",
            help="Budget model name.",
        ),
    ] = "deepseek-chat",
    standard_model: Annotated[
        str,
        typer.Option(
            "--standard-model",
            help="Standard draft model name.",
        ),
    ] = "deepseek-chat",
    flagship_model: Annotated[
        str,
        typer.Option(
            "--flagship-model",
            help="Flagship model name.",
        ),
    ] = "claude-3-7-sonnet",
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output machine-readable JSON inspection list.",
        ),
    ] = False,
) -> None:
    """Inspect block-level complexity scores, feature diagnostics, and assigned model tiers."""
    blocks = _extract_blocks_from_text(file_path)
    if not blocks:
        console.print(f"[bold red]Error:[/] No blocks parsed from {file_path}")
        raise typer.Exit(code=1)

    scheduler = DynamicParetoScheduler(
        budget_model=budget_model,
        standard_model=standard_model,
        flagship_model=flagship_model,
    )

    assignments = [scheduler.resolve_model_for_block(b) for b in blocks]

    if tier:
        target_tier = tier.strip().lower()
        assignments = [a for a in assignments if a.tier.value == target_tier]

    displayed = assignments[:limit]

    if json_output:
        data = [a.model_dump() for a in displayed]
        console.print_json(json.dumps(data, ensure_ascii=False))
        return

    table = Table(
        title=f"Block Complexity Inspection ({len(displayed)} of {len(assignments)} shown)",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Block ID", style="bold")
    table.add_column("Type", style="cyan")
    table.add_column("Chars", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Tier", style="bold")
    table.add_column("Assigned Model", style="bold green")
    table.add_column("Primary Reason", style="dim")

    tier_colors = {
        ComplexityTier.TRIVIAL: "blue",
        ComplexityTier.STANDARD: "green",
        ComplexityTier.COMPLEX: "yellow",
        ComplexityTier.EXTREME: "bold red",
    }

    for a in displayed:
        t_color = tier_colors.get(a.tier, "white")
        table.add_row(
            a.block_id,
            a.metrics.block_type,
            str(a.metrics.char_count),
            f"{a.metrics.score:.2f}",
            f"[{t_color}]{a.tier.value.upper()}[/{t_color}]",
            a.assigned_model,
            a.reason,
        )

    console.print(table)


__all__ = ["route_app", "route_inspect_command", "route_simulate_command"]
