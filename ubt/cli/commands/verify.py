"""``ubt verify`` -- the delivery-contract reconciliation gate.

Reconciles the content and asset ledgers of a delivery and exits non-zero on any
error-severity violation (dropped/mislaid text, lost assets). It is
engine-agnostic: it reads the contract, not the render path, so a regression on
any path surfaces as an unbalanced book.

Usage::

    ubt verify out/book_mono.pdf            # read the artifact's *_contract.json
    ubt verify --job job_abc_zh            # re-derive from the ledger (cross-check)
    ubt verify --corpus corpus             # every case in corpus/cases.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from ubt.core.content.contract import ReconciliationReport
from ubt.core.content.verify import (
    contract_from_ledger,
    evaluate_expectations,
    load_contract,
    load_corpus,
)
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.job_options import job_id_is_valid

console = Console()


def _get_resolve_db_dir() -> Any:
    # Single owner: main._resolve_db_dir (same seam the status command uses).
    from ubt.cli.main import _resolve_db_dir

    return _resolve_db_dir


def _render(report: ReconciliationReport, *, title: str) -> None:
    console.print(f"\n[bold]{escape(title)}[/]")
    console.print(f"  {escape(report.summary_line())}")
    if report.violations:
        table = Table(show_header=True, header_style="bold")
        table.add_column("severity")
        table.add_column("kind")
        table.add_column("node")
        table.add_column("detail")
        for v in report.violations[:20]:
            colour = "red" if v.severity.value == "error" else "yellow"
            table.add_row(
                f"[{colour}]{v.severity.value}[/]",
                v.kind.value,
                escape(v.node_id),
                escape(v.detail),
            )
        console.print(table)


def verify_command(
    artifact: Annotated[
        Path | None,
        typer.Argument(help="Delivered artifact to verify (reads its *_contract.json sidecar)"),
    ] = None,
    job: Annotated[
        str | None,
        typer.Option("--job", help="Re-reconcile this finished job's ledger instead"),
    ] = None,
    corpus: Annotated[
        Path | None,
        typer.Option("--corpus", help="Directory containing cases.json to verify"),
    ] = None,
    engine: Annotated[
        str,
        typer.Option(
            "--engine",
            help="Engine hint for --job re-derivation (rigid | publication)",
        ),
    ] = "publication",
    db_dir: Annotated[
        Path | None,
        typer.Option(
            "--db-dir",
            help="SQLite ledger storage directory (default: from config / UBT_DB_DIR)",
            show_default=False,
        ),
    ] = None,
    require_all: Annotated[
        bool,
        typer.Option(
            "--require-all",
            help="Corpus: a missing artifact/job is a failure, not a skip",
        ),
    ] = False,
    json_output: Annotated[
        bool, typer.Option("--json", help="Machine-readable mode: stdout carries JSON")
    ] = False,
) -> None:
    """Reconcile delivery content/asset ledgers; non-zero exit on violations."""
    if corpus is not None:
        _verify_corpus(corpus, db_dir=db_dir, require_all=require_all, json_output=json_output)
        return
    if job is not None:
        report = _verify_job(job, engine=engine, db_dir=db_dir, json_output=json_output)
    elif artifact is not None:
        report = _verify_artifact(artifact, json_output=json_output)
    else:
        if json_output:
            print(json.dumps({"status": "error", "error": "pass an ARTIFACT, --job, or --corpus"}))
        else:
            console.print("[bold red]Nothing to verify:[/] pass an ARTIFACT, --job, or --corpus.")
        raise typer.Exit(code=2)
    raise typer.Exit(code=0 if report.passed else 1)


def _verify_artifact(artifact: Path, *, json_output: bool) -> ReconciliationReport:
    from ubt.core.content.verify import contract_path_for_artifact

    if not artifact.exists():
        if json_output:
            print(json.dumps({"status": "error", "error": f"no artifact at {artifact}"}))
        else:
            console.print(f"[bold red]No artifact at[/] {artifact}")
        raise typer.Exit(code=2)
    try:
        report = load_contract(artifact)
    except FileNotFoundError:
        want = contract_path_for_artifact(artifact)
        if json_output:
            print(json.dumps({"status": "error", "error": f"no contract at {want}"}))
        else:
            console.print(
                f"[bold red]No delivery contract at[/] {want}\n"
                "  Re-run the translation (the contract is written at export), or use "
                "--job to re-derive from the ledger."
            )
        raise typer.Exit(code=2) from None
    if json_output:
        print(json.dumps({"status": "pass" if report.passed else "fail", **_dump(report)}))
    else:
        _render(report, title=f"Delivery contract — {artifact.name}")
    return report


def _verify_job(job: str, *, engine: str, db_dir: Any, json_output: bool) -> ReconciliationReport:
    if not job_id_is_valid(job):
        if json_output:
            print(json.dumps({"status": "error", "error": f"invalid job id {job!r}"}))
        else:
            console.print(f"[bold red]Invalid job id:[/] {escape(job)}")
        raise typer.Exit(code=2)
    resolve_fn = _get_resolve_db_dir()
    resolved = resolve_fn(db_dir)
    db_path = Path(resolved) / f"{job}.sqlite"
    if not db_path.exists():
        if json_output:
            print(json.dumps({"status": "error", "error": f"no ledger at {db_path}"}))
        else:
            console.print(f"[bold red]No ledger at[/] {db_path}")
        raise typer.Exit(code=2)
    ledger = SQLiteJobLedger(db_path, read_only=True)
    try:
        report = contract_from_ledger(ledger, job, engine=engine)
    finally:
        ledger.close()
    if json_output:
        print(json.dumps({"status": "pass" if report.passed else "fail", **_dump(report)}))
    else:
        _render(report, title=f"Delivery contract (ledger re-derivation) — {job}")
    return report


def _verify_corpus(corpus_dir: Path, *, db_dir: Any, require_all: bool, json_output: bool) -> None:
    if not corpus_dir.exists():
        if json_output:
            print(json.dumps({"status": "error", "error": f"no corpus at {corpus_dir}"}))
        else:
            console.print(f"[bold red]No corpus directory at[/] {corpus_dir}")
        raise typer.Exit(code=2)
    try:
        cases = load_corpus(corpus_dir)
    except FileNotFoundError as exc:
        if json_output:
            print(json.dumps({"status": "error", "error": str(exc)}))
        else:
            console.print(f"[bold red]{escape(str(exc))}[/]")
        raise typer.Exit(code=2) from None

    results: list[dict[str, Any]] = []
    failed = 0
    for case in cases:
        entry: dict[str, Any] = {"id": case.id, "description": case.description}
        try:
            report = _resolve_case(case, corpus_dir=corpus_dir, db_dir=db_dir)
        except _CaseSkipped as skip:
            entry["status"] = "fail" if require_all else "skip"
            entry["reason"] = skip.reason
            if require_all:
                failed += 1
            results.append(entry)
            continue
        failures = evaluate_expectations(report, case.expect)
        ok = report.passed and not failures
        entry.update({"status": "pass" if ok else "fail", **_dump(report)})
        if failures:
            entry["expectation_failures"] = failures
        if not ok:
            failed += 1
        results.append(entry)

    payload = {
        "status": "fail" if failed else "pass",
        "corpus": str(corpus_dir),
        "cases": results,
        "failed": failed,
    }
    if json_output:
        print(json.dumps(payload, ensure_ascii=False))
    else:
        table = Table(title=f"Delivery contract corpus — {corpus_dir}", header_style="bold")
        table.add_column("id")
        table.add_column("status")
        table.add_column("contract")
        for entry in results:
            colour = {"pass": "green", "fail": "red", "skip": "yellow"}[entry["status"]]
            table.add_row(
                escape(entry["id"]),
                f"[{colour}]{entry['status']}[/]",
                escape(entry.get("summary") or entry.get("reason", "")),
            )
        console.print(table)
    if failed:
        raise typer.Exit(code=1)


class _CaseSkipped(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _resolve_case(case: Any, *, corpus_dir: Path, db_dir: Any) -> ReconciliationReport:
    if case.artifact:
        artifact = Path(case.artifact)
        if not artifact.is_absolute():
            artifact = corpus_dir / artifact
        if not artifact.exists():
            raise _CaseSkipped(f"artifact missing: {artifact}")
        return load_contract(artifact)
    if case.job:
        resolve_fn = _get_resolve_db_dir()
        resolved = resolve_fn(db_dir)
        db_path = Path(resolved) / f"{case.job}.sqlite"
        if not db_path.exists():
            raise _CaseSkipped(f"ledger missing: {db_path}")
        ledger = SQLiteJobLedger(db_path, read_only=True)
        try:
            return contract_from_ledger(ledger, case.job)
        finally:
            ledger.close()
    raise _CaseSkipped("case declares neither artifact nor job")


def _dump(report: ReconciliationReport) -> dict[str, Any]:
    data = report.model_dump(mode="json")
    data["summary"] = report.summary_line()
    return data
