"""CLI commands for inspecting documents, checking job checkpoint status, and importing PE revisions."""

import asyncio
import json
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from ubt.cli.commands import resolve_db_dir
from ubt.core.config import UBTConfig
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.pe_import import PEImportError, import_pe_revisions
from ubt.core.engine.writer_lock import LedgerWriterLock
from ubt.core.exceptions import DocumentParseError, LedgerWriterLockConflictError
from ubt.core.ir.models import BlockStatus
from ubt.core.job_options import JOB_ID_MAX_LEN, job_id_is_valid

console = Console()


def recheck_gates(job_id: str, *, db_dir: Any = None) -> dict[str, Any]:
    """Re-run today's gates over every quarantined block's preserved draft.

    Triage replaces a quarantined block's ``target_text`` with a
    ``<mark class="ubt-blocked-human">`` placeholder (so no machine output
    ships), and the rejected translation survives only in ``draft_text``.
    Anything asking "would this pass now?" — after a gate fix, a model swap, or
    a prompt change — must read ``draft_text``: ``target_text`` is an HTML
    marker containing no LaTeX at all, so a report built on it declares every
    quarantined block clean and invents a pass rate.

    Returns counts plus a per-block breakdown; never raises for a missing job.
    """
    from ubt.core.qe.fast_pass import FastPassFilter

    report: dict[str, Any] = {
        "job_id": job_id,
        "total": 0,
        "would_pass": 0,
        "still_failing": 0,
        "no_draft": 0,
        "blocks": [],
    }
    if not job_id_is_valid(job_id):
        # Mirror job_status/pe_import: never build a path from an unvalidated id.
        report["error"] = f"invalid job id {job_id!r}"
        return report
    try:
        db_dir = resolve_db_dir()(db_dir)
        db_path = Path(db_dir) / f"{job_id}.sqlite"
        if not db_path.exists():
            # SQLite would silently create an empty store here, making a typo'd
            # job id indistinguishable from a job with nothing quarantined.
            report["error"] = f"no ledger at {db_path}"
            return report
        ledger = SQLiteJobLedger(db_path, read_only=True)
    except Exception as exc:  # bad job id, unreadable store
        report["error"] = str(exc)
        return report

    try:
        quarantined = [
            *ledger.fetch_blocks_by_status(job_id, BlockStatus.BLOCKED_HUMAN),
            *ledger.fetch_blocks_by_status(job_id, BlockStatus.NEEDS_HUMAN),
        ]
        # The gate is language-pair specific: without the job's own target the
        # filter used the zh default and mis-scored, say, an en→ja job's drafts.
        target_lang = ledger.get_job_target_lang(job_id)
    except Exception as exc:
        report["error"] = str(exc)
        return report
    finally:
        ledger.close()

    gate = FastPassFilter(target_lang=target_lang) if target_lang else FastPassFilter()
    for block in quarantined:
        report["total"] += 1
        # draft_text, never target_text — see the docstring.
        draft = block.draft_text or ""
        if not draft.strip() or "ubt-blocked-human" in draft:
            report["no_draft"] += 1
            continue
        # Structural recheck by contract (see the command's test): reading
        # draft_text is what lets a Unicode-aware structural fix show through.
        decision = gate.validate_structural_invariants(block.source_text or "", draft)
        if decision.passed:
            report["would_pass"] += 1
        else:
            report["still_failing"] += 1
        report["blocks"].append(
            {
                "block_id": block.id,
                "status": block.status.value,
                "would_pass": decision.passed,
                "reason": decision.reason[:200],
            }
        )
    return report


def recheck_gates_cmd(
    job_id: Annotated[str, typer.Argument(help="Job ID whose quarantined blocks to re-verify")],
    db_dir: Annotated[
        Path | None,
        typer.Option(
            "--db-dir",
            help="SQLite ledger storage directory (default: from config / UBT_DB_DIR)",
            show_default=False,
        ),
    ] = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="Machine-readable mode: stdout carries the report JSON")
    ] = False,
    limit: Annotated[
        int,
        typer.Option("--limit", help="Show at most N failing blocks in the detail table", min=0),
    ] = 20,
) -> None:
    """Re-run today's quality gates over every quarantined block's draft.

    After a gate fix, a model swap, or a prompt change, this answers "which
    previously-quarantined blocks would pass today?". It reads the preserved
    ``draft_text`` — a quarantined block's ``target_text`` is an HTML
    placeholder, so judging that would report everything clean.
    """
    report = recheck_gates(job_id, db_dir=db_dir)
    if json_output:
        print(json.dumps(report, ensure_ascii=False))
        if report.get("error"):
            raise typer.Exit(code=1)
        return
    if report.get("error"):
        console.print(f"[bold red]Cannot recheck:[/] {escape(str(report['error']))}")
        raise typer.Exit(code=1)
    table = Table(title=f"Gate recheck — {job_id}", show_header=True, header_style="bold")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Quarantined blocks", str(report["total"]))
    table.add_row(
        "[green]Would pass now[/]",
        str(report["would_pass"]),
    )
    table.add_row("[red]Still failing[/]", str(report["still_failing"]))
    table.add_row("[yellow]No draft to judge[/]", str(report["no_draft"]))
    console.print(table)
    failing = [b for b in report["blocks"] if not b["would_pass"]]
    if failing:
        detail = Table(title=f"Still failing (showing {min(limit, len(failing))})")
        detail.add_column("Block", overflow="fold")
        detail.add_column("Reason", overflow="fold")
        for b in failing[:limit]:
            detail.add_row(b["block_id"], escape(b["reason"])[:160])
        console.print(detail)
    if report["would_pass"]:
        console.print(
            f"[green]{report['would_pass']} block(s) would now pass.[/] "
            "Reset them to re-enter the pipeline:\n"
            f"  [dim]ubt translate <input> --job-id {job_id} …[/]"
        )


def inspect_book(
    input_path: Annotated[
        Path, typer.Argument(help="Path to input document (.epub, .md, .pdf) or job ID")
    ],
    db_dir: Annotated[
        Path | None,
        typer.Option(
            "--db-dir",
            help="SQLite ledger storage directory (default: from config / UBT_DB_DIR)",
            show_default=False,
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Machine-readable mode: stdout carries the manifest JSON object",
        ),
    ] = False,
) -> None:
    """Inspect book metadata, TOC structure, or query job status."""
    resolve_fn = resolve_db_dir()
    db_dir = resolve_fn(db_dir)
    raw_str = str(input_path)
    if not input_path.exists():
        if raw_str.startswith("job_") or (db_dir / f"{raw_str}.sqlite").exists():
            if not json_output:
                console.print(
                    f"[bold yellow]Hint:[/] '{raw_str}' recognized as a Job ID. Displaying job status (hint: 'ubt status {raw_str}'):"
                )
            job_status(job_id=raw_str, db_dir=db_dir, json_output=json_output)
            return
        if json_output:
            print(json.dumps({"status": "failed", "error": f"Input file not found: {input_path}"}))
            raise typer.Exit(code=1)
        console.print(f"[bold red]Error:[/] Input file not found: {input_path}")
        raise typer.Exit(code=1)

    try:
        # Deferred: see the note in doctor_command — a module-scope adapter
        # import makes every CLI invocation pay the PDF import graph.
        from ubt.adapters.factory import get_adapter_for_path

        adapter = get_adapter_for_path(input_path, pdf_engine=UBTConfig.from_env().pdf_engine)
        manifest = asyncio.run(adapter.extract_manifest(input_path))
        if json_output:
            print(manifest.model_dump_json())
            return

        table = Table(title=f"Book TOC Structure: {manifest.title}", border_style="cyan")
        table.add_column("Spine", justify="right", style="cyan", no_wrap=True)
        table.add_column("Chapter ID", style="magenta")
        table.add_column("Title", style="green")
        table.add_column("Source File", style="dim")

        for ch in manifest.chapters:
            table.add_row(
                str(ch.spine_index),
                ch.chapter_id,
                ch.title,
                ch.source_file or "-",
            )

        console.print(
            Panel(
                f"[bold]Title:[/] {manifest.title}\n"
                f"[bold]Document ID:[/] {manifest.doc_id}\n"
                f"[bold]Total Chapters:[/] {manifest.total_chapters}\n"
                f"[bold]Source Path:[/] {manifest.source_path}",
                title="Manifest Summary",
                border_style="green",
            )
        )
        console.print(table)
    except DocumentParseError as exc:
        if json_output:
            print(json.dumps({"status": "failed", "error": str(exc)}))
            raise typer.Exit(code=1) from exc
        console.print(f"[bold red]Failed to inspect document:[/] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc


def job_status(
    job_id: Annotated[str, typer.Argument(help="Job ID or document hash to query")],
    db_dir: Annotated[
        Path | None,
        typer.Option(
            "--db-dir",
            help="SQLite ledger storage directory (default: from config / UBT_DB_DIR)",
            show_default=False,
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Machine-readable mode: stdout carries the stats JSON object",
        ),
    ] = False,
) -> None:
    """Display real-time checkpoint statistics from SQLite ledger."""
    resolve_fn = resolve_db_dir()
    db_dir = resolve_fn(db_dir)
    # Validate job_id before building ledger path to prevent directory traversal.
    if not job_id_is_valid(job_id):
        if json_output:
            print(
                json.dumps(
                    {
                        "status": "error",
                        "job_id": job_id,
                        "error": (
                            f"Invalid job id {job_id!r}: letters, digits, "
                            f"'-' or '_' only, up to {JOB_ID_MAX_LEN} characters."
                        ),
                    }
                )
            )
            raise typer.Exit(code=2)
        console.print(
            f"[bold red]Error:[/] Invalid job id {job_id!r}: letters, digits, "
            f"'-' or '_' only, up to {JOB_ID_MAX_LEN} characters."
        )
        raise typer.Exit(code=2)
    db_path = db_dir / f"{job_id}.sqlite"
    if not db_path.exists():
        if json_output:
            print(
                json.dumps(
                    {
                        "status": "unknown",
                        "job_id": job_id,
                        "error": f"Ledger database not found: {db_path}",
                    }
                )
            )
            raise typer.Exit(code=1)
        console.print(f"[bold red]Error:[/] Ledger database not found: {db_path}")
        raise typer.Exit(code=1)

    ledger = SQLiteJobLedger(db_path, read_only=True)
    try:
        stats = ledger.get_job_stats(job_id)
        if not stats:
            if json_output:
                print(json.dumps({"status": "empty", "job_id": job_id}))
                return
            console.print(f"[yellow]No block records recorded yet for job: {job_id}[/]")
            return

        total = int(stats.get("total", 0))
        completed = int(stats.get("completed", 0))
        # Terminal-state numerator, matching ProgressSnapshot.processed_blocks: a run
        # whose blocks all reached terminal state is fully processed, reflecting true workflow completion.
        processed = (
            completed
            + int(stats.get("failed", 0))
            + int(stats.get("needs_human", 0))
            + int(stats.get("blocked_human", 0))
        )
        pct = round(processed / total * 100, 1) if total > 0 else 0.0
        visual = ledger.get_visual_report(job_id)
        visual_summary: dict[str, object] | None = None
        if visual:
            findings = visual.get("findings", [])
            majors = sum(1 for f in findings if f.get("severity") in ("major", "critical"))
            visual_summary = {
                "passed": visual.get("passed"),
                "sampled_pages": visual.get("sampled_pages", []),
                "major_critical": majors,
            }
        # Terminology conflicts: concepts the glossary gave more than one
        # rendering. An unresolved one explains a cluster of low scores that no
        # translation fix can clear, so it belongs in the operator's status view
        # rather than only in the run log.
        raw_conflicts = ledger.get_job_metadata_value(job_id, "terminology_conflicts")
        conflicts: list[dict[str, Any]] = (
            [c for c in raw_conflicts if isinstance(c, dict)]
            if isinstance(raw_conflicts, list)
            else []
        )
        unresolved_conflicts = [c for c in conflicts if not c.get("resolved")]
        if json_output:
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "job_id": job_id,
                        "total": total,
                        "completed": completed,
                        "processed_blocks": processed,
                        "progress_pct": pct,
                        "repaired": int(stats.get("repaired", 0)),
                        "failed": int(stats.get("failed", 0)),
                        "avg_qe_score": float(stats.get("avg_qe_score", 0.0)),
                        "bottom_15_avg_qe": float(stats.get("bottom_15_avg_qe", 0.0)),
                        "visual": visual_summary,
                        "terminology_conflicts": conflicts,
                        "unresolved_terminology_conflicts": len(unresolved_conflicts),
                    }
                )
            )
            return

        table = Table(title=f"Job Ledger Checkpoint Status: {job_id}", border_style="blue")
        table.add_column("Metric", style="bold cyan")
        table.add_column("Value", style="bold green")

        table.add_row("Total Blocks", str(total))
        table.add_row("Completed Blocks", str(completed))
        table.add_row("Processed (terminal states)", f"{processed} ({pct}%)")
        table.add_row("Repaired Blocks", str(stats.get("repaired", 0)))
        table.add_row("Failed Blocks", str(stats.get("failed", 0)))
        # A 0.0 avg is the "nothing scored" sentinel under the heuristic QE
        # engine, not a failing measurement — say which one it is instead of
        # printing a number that reads as quality data.
        qe_measured = bool(stats.get("qe_measured"))
        table.add_row(
            "Average MTQE Score",
            str(stats.get("avg_qe_score", 0.0)) if qe_measured else "not measured (heuristic QE)",
        )
        table.add_row(
            "Bottom 15% MTQE Score",
            str(stats.get("bottom_15_avg_qe", 0.0)) if qe_measured else "not measured",
        )
        if visual_summary is not None:
            table.add_row(
                "Visual Gate",
                f"passed={visual_summary['passed']} "
                f"sampled={visual_summary['sampled_pages']} "
                f"major/critical={visual_summary['major_critical']}",
            )
        else:
            table.add_row("Visual Gate", "n/a (no visual report recorded)")

        if conflicts:
            table.add_row(
                "Terminology Conflicts",
                f"{len(conflicts)} concept(s), {len(unresolved_conflicts)} unresolved",
            )

        console.print(table)

        if unresolved_conflicts:
            # Not a score line: a paragraph explaining why a reader should not
            # treat these blocks as translation failures.
            lines: list[str] = []
            for conflict in unresolved_conflicts:
                losers = ", ".join(
                    str(s.get("rendering", ""))
                    for s in conflict.get("superseded", [])
                    if isinstance(s, dict)
                )
                lines.append(
                    f"[bold]{escape(str(conflict.get('concept', '')))}[/] is enforced as "
                    f"[green]{escape(str(conflict.get('preferred', '')))}[/] "
                    f"but also appears as [yellow]{escape(losers)}[/]"
                )
            console.print(
                Panel(
                    "\n".join(lines)
                    + "\n\nA translation can only satisfy one rendering, so blocks using "
                    "another read as terminology drift no matter how well they are "
                    "translated. Keep one entry, or list the rejected rendering as an "
                    "admitted variant in the glossary.",
                    title="[yellow]Unresolved terminology conflicts[/]",
                    border_style="yellow",
                )
            )
        elif conflicts:
            console.print(
                f"[dim]{len(conflicts)} terminology conflict(s) auto-resolved by "
                f"precedence; both renderings are accepted.[/]"
            )
    finally:
        ledger.close()


def rewrite_flag_text(
    db_dir: Annotated[
        Path | None,
        typer.Option(
            "--db-dir",
            help="SQLite ledger storage directory (default: from config / UBT_DB_DIR)",
            show_default=False,
        ),
    ] = None,
    apply: Annotated[
        bool,
        typer.Option(
            "--apply",
            help="Write the rewritten flags (default: dry run, print what would change)",
        ),
    ] = False,
    limit: Annotated[
        int,
        typer.Option("--limit", help="Show at most N blocks in the detail table", min=0),
    ] = 10,
    json_output: Annotated[
        bool, typer.Option("--json", help="Machine-readable mode: stdout carries the plan JSON")
    ] = False,
) -> None:
    """Rewrite stored QA flag texts that an older build phrased misleadingly.

    A ledger is live input, not an archive: the quality report, the XLIFF
    companion a reviewer opens and every later reader take their wording from it.
    Fixing a gate's message therefore only half-repairs the record -- the flags
    already written keep misleading people until the block is re-run, which for a
    delivered job may be never.

    Only the sentence changes. Status, severity, score and repair budget are not
    touched, and a flag that does not match the known old form is left alone, so
    running this twice is a no-op.
    """
    resolve_fn = resolve_db_dir()
    resolved = resolve_fn(db_dir)
    from ubt.core.engine.flag_rewrite import apply as apply_rewrites
    from ubt.core.engine.flag_rewrite import plan

    hits = plan(resolved)
    if json_output:
        print(
            json.dumps(
                {
                    "db_dir": str(resolved),
                    "applied": apply,
                    "blocks": len(hits),
                    "flags": sum(hit.changed for hit in hits),
                    "items": [
                        {
                            "db": hit.db.name,
                            "block_id": hit.block_id,
                            "old": list(hit.old_flags),
                            "new": list(hit.new_flags),
                        }
                        for hit in hits
                    ],
                },
                ensure_ascii=False,
            )
        )
        if apply and hits:
            apply_rewrites(hits)
        return

    if not hits:
        console.print(
            "[green]Nothing to rewrite:[/] every stored flag already uses current wording."
        )
        return
    flags = sum(hit.changed for hit in hits)
    table = Table(title="Stored QA flags with outdated wording", show_header=True)
    table.add_column("Ledger", overflow="fold")
    table.add_column("Block", overflow="fold")
    table.add_column("Flags", justify="right")
    per_db: dict[str, tuple[int, int]] = {}
    for hit in hits:
        blocks, changed = per_db.get(hit.db.name, (0, 0))
        per_db[hit.db.name] = (blocks + 1, changed + hit.changed)
    for name, (blocks, changed) in sorted(per_db.items()):
        table.add_row(name, str(blocks), str(changed))
    console.print(table)
    console.print(f"[bold]{flags}[/] flag(s) across [bold]{len(hits)}[/] block(s).")
    if hits and limit:
        detail = Table(title=f"Examples (showing {min(limit, len(hits))})")
        detail.add_column("Block", overflow="fold")
        detail.add_column("New text", overflow="fold")
        for hit in hits[:limit]:
            changed_text = [
                new for old, new in zip(hit.old_flags, hit.new_flags, strict=True) if old != new
            ]
            detail.add_row(hit.block_id, escape(changed_text[0])[:170])
        console.print(detail)

    if not apply:
        console.print(
            "\n[yellow]Dry run.[/] Nothing written. Back up the ledger directory "
            "first, then re-run with [bold]--apply[/]."
        )
        return
    changed = apply_rewrites(hits)
    console.print(f"[green]✓ Rewrote {changed} flag(s).[/]")


def upgrade_flags(
    db_dir: Annotated[
        Path | None,
        typer.Option(
            "--db-dir",
            help="SQLite ledger storage directory (default: from config / UBT_DB_DIR)",
            show_default=False,
        ),
    ] = None,
    apply: Annotated[
        bool,
        typer.Option(
            "--apply",
            help="Write the upgraded flags (default: dry run, print what would change)",
        ),
    ] = False,
    limit: Annotated[
        int,
        typer.Option("--limit", help="Show at most N blocks in the detail table", min=0),
    ] = 10,
    json_output: Annotated[
        bool, typer.Option("--json", help="Machine-readable mode: stdout carries the plan JSON")
    ] = False,
) -> None:
    """Upgrade stored flags to the structured ``category:severity:detail`` form.

    Legacy flags are classified by substring tables, so a rename in one stage
    can silently change a verdict in another. The structured form carries its
    own category and severity. This rewrites only the flags whose verdict is
    provably unchanged — a flag the tables cannot place is left exactly as it
    is, and a second run is a no-op.

    Separate from ``rewrite-flags``, which fixes *wording* while keeping the
    flag's encoding. Run this one first: an upgraded flag is still matched by
    the old tables (its category is a structural category), so the two passes
    are independent.
    """
    resolve_fn = resolve_db_dir()
    resolved = resolve_fn(db_dir)
    from ubt.core.engine.flag_upgrade import apply as apply_upgrades
    from ubt.core.engine.flag_upgrade import plan as plan_upgrades

    hits = plan_upgrades(resolved)
    if json_output:
        print(
            json.dumps(
                {
                    "db_dir": str(resolved),
                    "applied": apply,
                    "blocks": len(hits),
                    "flags": sum(hit.changed for hit in hits),
                    "items": [
                        {
                            "db": hit.db.name,
                            "block_id": hit.block_id,
                            "old": list(hit.old_flags),
                            "new": list(hit.new_flags),
                        }
                        for hit in hits
                    ],
                },
                ensure_ascii=False,
            )
        )
        if apply and hits:
            apply_upgrades(hits)
        return

    if not hits:
        console.print(
            "[green]Nothing to upgrade:[/] every stored flag already uses the "
            "structured form, or is one this tool will not guess at."
        )
        return
    flags = sum(hit.changed for hit in hits)
    table = Table(title="Stored QA flags to upgrade", show_header=True)
    table.add_column("Ledger", overflow="fold")
    table.add_column("Blocks", justify="right")
    table.add_column("Flags", justify="right")
    per_db: dict[str, tuple[int, int]] = {}
    for hit in hits:
        blocks, changed = per_db.get(hit.db.name, (0, 0))
        per_db[hit.db.name] = (blocks + 1, changed + hit.changed)
    for name, (blocks, changed) in sorted(per_db.items()):
        table.add_row(name, str(blocks), str(changed))
    console.print(table)
    console.print(f"[bold]{flags}[/] flag(s) across [bold]{len(hits)}[/] block(s).")
    if hits and limit:
        detail = Table(title=f"Examples (showing {min(limit, len(hits))})")
        detail.add_column("Block", overflow="fold")
        detail.add_column("New flag", overflow="fold")
        for hit in hits[:limit]:
            changed_text = [
                new for old, new in zip(hit.old_flags, hit.new_flags, strict=True) if old != new
            ]
            detail.add_row(hit.block_id, escape(changed_text[0])[:170])
        console.print(detail)

    if not apply:
        console.print(
            "\n[yellow]Dry run.[/] Nothing written. Back up the ledger directory "
            "first, then re-run with [bold]--apply[/]."
        )
        return
    changed = apply_upgrades(hits)
    console.print(f"[green]✓ Upgraded {changed} flag(s).[/]")


def pe_import(
    job_id: Annotated[str, typer.Argument(help="Job ID whose ledger receives the revisions")],
    file: Annotated[
        Path,
        typer.Option(
            "--file",
            "-f",
            help="Post-edited PE queue file (.csv with 'revised_translation' column, or .xliff/.xlf)",
        ),
    ],
    db_dir: Annotated[
        Path | None,
        typer.Option(
            "--db-dir",
            help="SQLite ledger storage directory (default: from config / UBT_DB_DIR)",
            show_default=False,
        ),
    ] = None,
    write_tm: Annotated[
        bool,
        typer.Option(
            "--write-tm/--no-write-tm",
            help="Feed accepted revisions into the shared translation memory "
            "(provenance='human_pe'; the language pair is taken from the job ledger)",
        ),
    ] = True,
) -> None:
    """Re-import human post-edited PE queue revisions into the ledger and TM.

    The PE file must carry the job binding it was exported with; revisions are
    only applied to that job's ledger. The TM language pair always comes from
    the ledger, never from CLI flags.
    """
    resolve_fn = resolve_db_dir()
    db_dir = resolve_fn(db_dir)
    if not file.exists():
        console.print(f"[bold red]Error:[/] PE queue file not found: {file}")
        raise typer.Exit(code=1)
    if not job_id_is_valid(job_id):
        # ``job_status`` above validates the same argument; without this a
        # traversal like ``../../tmp/other`` resolved the ledger path outside
        # db_dir and ran the schema migrations on an arbitrary SQLite file.
        console.print(f"[bold red]Error:[/] Invalid job id: {job_id!r}")
        raise typer.Exit(code=1)

    db_path = db_dir / f"{job_id}.sqlite"
    if not db_path.exists():
        console.print(f"[bold red]Error:[/] Ledger database not found: {db_path}")
        raise typer.Exit(code=1)

    ledger = SQLiteJobLedger(db_path)
    tm = None
    # ``pe-import`` writes paid block state (``target_text``/``status``) straight
    # into the ledger, so it is a writer like the orchestrator: it must take the
    # same job-level lock, or it can race a concurrent resume and have its human
    # revisions overwritten (or overwrite the pipeline's checkpoints).
    writer_lock = LedgerWriterLock(db_path, job_id)
    try:
        writer_lock.acquire()
    except LedgerWriterLockConflictError as exc:
        console.print(
            f"[bold red]Job {job_id} is being written by another process; "
            f"refusing to import PE revisions concurrently.[/] {escape(str(exc))}"
        )
        raise typer.Exit(code=1) from exc
    try:
        if write_tm:
            from ubt.core.memory.tm import TranslationMemory

            db_dir.mkdir(parents=True, exist_ok=True)
            tm = TranslationMemory(db_dir / "tm.sqlite")
        result = import_pe_revisions(
            job_id=job_id,
            file_path=file,
            ledger=ledger,
            tm=tm,
        )
    except PEImportError as exc:
        console.print(f"[bold red]PE import failed:[/] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc
    finally:
        ledger.close()
        if tm is not None:
            tm.close()
        writer_lock.release()

    table = Table(title=f"PE Re-import: {job_id}", border_style="green")
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", style="bold green", justify="right")
    table.add_row("Revisions in file", str(result.total_rows))
    table.add_row("Imported", str(result.imported))
    table.add_row("Skipped (unchanged/unknown)", str(result.skipped))
    table.add_row("TM entries written (human_pe)", str(result.tm_written))
    console.print(table)
    if result.revised_block_ids:
        preview = ", ".join(result.revised_block_ids[:10])
        more = (
            f" … (+{len(result.revised_block_ids) - 10})"
            if len(result.revised_block_ids) > 10
            else ""
        )
        console.print(f"[dim]Revised blocks:[/] {preview}{more}")
    console.print("[bold green]✓ Human PE revisions applied.[/]")
