"""CLI command for executing bidirectional round-trip translation verification."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from ubt.cli.commands import resolve_db_dir

console = Console()


def roundtrip_command(
    job_id: Annotated[
        str | None,
        typer.Option(
            "--job-id", help="Audit translation blocks from an existing SQLite job ledger"
        ),
    ] = None,
    pairs_file: Annotated[
        Path | None,
        typer.Option(
            "--pairs-file",
            help="Path to JSON file containing [{'src': ..., 'mt': ..., 'back': ...}] pairs",
        ),
    ] = None,
    source_text: Annotated[
        str | None,
        typer.Option("--source", "-S", help="Single source text string to audit"),
    ] = None,
    target_text: Annotated[
        str | None,
        typer.Option("--target", "-T", help="Single target translation text string to audit"),
    ] = None,
    back_text: Annotated[
        str | None,
        typer.Option("--back", "-B", help="Optional back-translated text string"),
    ] = None,
    source_lang: Annotated[
        str,
        typer.Option("--source-lang", "-s", help="Source language code (default: en)"),
    ] = "en",
    target_lang: Annotated[
        str,
        typer.Option("--target-lang", "-l", help="Target language code (default: zh)"),
    ] = "zh",
    threshold: Annotated[
        float,
        typer.Option("--threshold", help="Passing roundtrip score threshold (default: 0.65)"),
    ] = 0.65,
    db_dir: Annotated[
        Path | None,
        typer.Option(
            "--db-dir", help="Directory holding job ledgers (default: config / UBT_DB_DIR)"
        ),
    ] = None,
    markdown_out: Annotated[
        Path | None,
        typer.Option("--markdown-out", help="Export GitHub Flavored Markdown report path"),
    ] = None,
    json_out: Annotated[
        bool,
        typer.Option("--json", help="Output machine-readable JSON to stdout"),
    ] = False,
) -> None:
    """Audit translation quality and semantic invariants via bidirectional round-trip verification.

    Performs smoothed BLEU, entity/number conservation checks, polarity flip guards,
    and MQM defect classifications across language pairs.
    """
    from ubt.core.engine.ledger import SQLiteJobLedger
    from ubt.core.qe.roundtrip import RoundTripVerifier
    from ubt.core.qe.roundtrip_matrix import (
        MultiLanguageMatrixAuditor,
        render_json_matrix,
        render_markdown_matrix,
        render_terminal_matrix,
    )

    verifier = RoundTripVerifier(threshold=threshold)
    auditor = MultiLanguageMatrixAuditor(verifier=verifier)

    eval_pairs: list[tuple[str, str, str | None]] = []

    # Case 1: Direct single string pair
    if source_text and target_text:
        eval_pairs.append((source_text, target_text, back_text))

    # Case 2: JSON file of pairs
    elif pairs_file and pairs_file.exists():
        data = json.loads(pairs_file.read_text(encoding="utf-8"))
        for item in data:
            s = item.get("src") or item.get("source", "")
            t = item.get("mt") or item.get("target", "")
            b = item.get("back") or item.get("back_translation")
            if s and t:
                eval_pairs.append((s, t, b))

    # Case 3: Read from SQLite ledger for job_id
    elif job_id:
        resolve_fn = resolve_db_dir()
        db_path = resolve_fn(db_dir) / f"{job_id}.sqlite"
        if not db_path.exists():
            console.print(f"[bold red]Error:[/] Job ledger not found at {db_path}")
            raise typer.Exit(code=1)

        ledger = SQLiteJobLedger(db_path, read_only=True)
        try:
            blocks = ledger.get_all_blocks(job_id)
            for b in blocks:
                src = (b.source_text or "").strip()
                mt = (b.target_text or b.draft_text or "").strip()
                if src and mt and not mt.startswith("<mark"):
                    eval_pairs.append((src, mt, None))
        finally:
            ledger.close()

    else:
        console.print(
            "[yellow]Notice:[/] No inputs specified. Use --job-id, --pairs-file, or --source and --target."
        )
        raise typer.Exit(code=1)

    if not eval_pairs:
        console.print("[yellow]Warning:[/] No non-empty translation pairs found for evaluation.")
        raise typer.Exit(code=0)

    summary = asyncio.run(
        auditor.audit_pairs(
            eval_pairs,
            src_lang=source_lang,
            tgt_lang=target_lang,
        )
    )

    summaries = [summary]

    if json_out:
        print(json.dumps(render_json_matrix(summaries), indent=2, ensure_ascii=False))
        return

    render_terminal_matrix(summaries, console=console)

    if markdown_out:
        md = render_markdown_matrix(summaries)
        markdown_out.write_text(md, encoding="utf-8")
        console.print(f"[bold green]✓ Exported Markdown report to {markdown_out}[/]")
