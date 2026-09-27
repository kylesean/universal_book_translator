"""Black-box end-to-end smoke: the real ``ubt`` CLI over a real ledger and disk.

There is no third-party mock in this module. The run goes through the installed
CLI in a child process and uses the project's own zero-cost dry-run (echo)
provider, so it needs neither network nor an API key. Every assertion reads the
*system under test* — the rows the pipeline committed to the SQLite ledger and
the artifacts it wrote to disk — so a run that reported success without writing
either would fail here.

Re-run cold with::

    uv run pytest tests/e2e -q
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

from ubt.core.ir.models import TERMINAL_STATUSES, BlockStatus

#: A source sentence the dry-run provider mirrors back into the deliverable.
SOURCE_SENTENCE = "Alpha paragraph one is here."
#: The dry-run provider's stand-in for a translated draft.
DRY_RUN_TARGET_MARKER = "模拟翻译"


def _run_cli(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    # Strip provider credentials so the child can never reach a live endpoint,
    # even when the developer's shell exports one; ``--dry-run`` is the belt.
    env = {k: v for k, v in os.environ.items() if "API_KEY" not in k.upper()}
    return subprocess.run(
        [sys.executable, "-m", "ubt", *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_cli_dry_run_writes_a_ledger_and_real_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "smoke.md"
    source.write_text(
        f"# Smoke\n\n{SOURCE_SENTENCE}\n\nBeta paragraph two is here.\n\n"
        "Gamma paragraph three is here.\n",
        encoding="utf-8",
    )
    output = tmp_path / "out" / "smoke_bilingual.md"
    job_id = "e2e-cli-smoke"

    result = _run_cli(
        [
            "translate",
            str(source),
            "--dry-run",
            "--job-id",
            job_id,
            "-l",
            "zh",
            "-o",
            str(output),
            "--json",
        ],
        tmp_path,
    )
    assert result.returncode == 0, (
        f"CLI failed ({result.returncode}):\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["status"] == "completed"
    assert payload["output_file"] == str(output)
    assert payload["quality_report"] is not None

    # --- real artifacts on disk ------------------------------------------
    assert output.exists() and output.stat().st_size > 0, "no bilingual deliverable"
    deliverable = output.read_text(encoding="utf-8")
    assert SOURCE_SENTENCE in deliverable, "source text lost from the deliverable"
    assert DRY_RUN_TARGET_MARKER in deliverable, "no translated draft in the deliverable"

    report_path = Path(payload["quality_report"])
    assert report_path.exists(), "quality report missing beside the deliverable"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    from ubt.core.engine.reporter import QUALITY_REPORT_SCHEMA_VERSION

    assert report["schema_version"] == QUALITY_REPORT_SCHEMA_VERSION
    summary = report["summary"]
    total = summary["total_blocks"]
    assert total >= 1
    # ``completed_blocks`` already folds in repaired blocks: the ledger counts
    # ``status IN ('mtqe_passed', 'repaired')`` as completed (ledger_mixins.py)
    # and reporter.py passes that through unchanged. Adding ``repaired_blocks``
    # on top therefore double-counts it — the old sum did exactly that and
    # stayed green only because the dry-run never repairs a block. The first run
    # with an actual repair (3 repaired on the synthetic-duo PDF) read
    # 312 + 3 + 0 + 0 + 3 = 318 against a total of 315. The partition below is
    # the invariant; the subset check pins the inclusive reading so a re-added
    # ``repaired`` term fails here rather than on a live corpus.
    assert summary["repaired_blocks"] <= summary["completed_blocks"], (
        "repaired blocks are a subset of completed, not a sibling bucket"
    )
    accounted = (
        summary["completed_blocks"]
        + summary["failed_blocks"]
        + summary["needs_human_blocks"]
        + summary["blocked_human_blocks"]
    )
    assert accounted == total, f"report accounting {accounted} != total {total}"

    # --- real ledger rows -------------------------------------------------
    db = tmp_path / ".ubt" / "ledgers" / f"{job_id}.sqlite"
    assert db.exists(), "pipeline never opened a ledger"
    with sqlite3.connect(db) as conn:
        job = conn.execute(
            "SELECT status, total_blocks FROM job_meta WHERE job_id = ?", (job_id,)
        ).fetchone()
        assert job is not None, "job_meta row missing"
        assert job[0] == "completed", f"job not finalized: {job[0]!r}"
        assert job[1] == total, "ledger and report disagree on total_blocks"
        statuses = [
            BlockStatus(row[0])
            for row in conn.execute("SELECT status FROM blocks WHERE job_id = ?", (job_id,))
        ]
    assert len(statuses) == total, "block rows do not match the reported total"
    assert set(statuses) <= TERMINAL_STATUSES, sorted(
        s.value for s in set(statuses) - TERMINAL_STATUSES
    )


def test_cli_empty_document_fails_loudly_and_writes_no_deliverable(tmp_path: Path) -> None:
    """An empty ``.md``/``.txt`` is a hard failure, never a "completed" empty book.

    ``MarkdownAdapter.parse_stream`` yields no chapter for empty content, so the
    ingest empty-book guard must still fire. Before the fix the run finalized
    ``completed`` with a ~1-byte deliverable and ``--strict`` exited 0 — a
    silent false success on a truncated download or a mistyped path.
    """
    source = tmp_path / "empty.md"
    source.write_text("", encoding="utf-8")
    output = tmp_path / "out" / "empty_bilingual.md"
    job_id = "e2e-cli-empty"

    result = _run_cli(
        ["translate", str(source), "--dry-run", "--strict", "--job-id", job_id, "-o", str(output)],
        tmp_path,
    )
    assert result.returncode != 0, (
        "an empty document must not report success:\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert not output.exists(), "no deliverable may be written for an empty document"

    # The ledger must record the failure, not a completed job.
    db = tmp_path / ".ubt" / "ledgers" / f"{job_id}.sqlite"
    assert db.exists(), "pipeline never opened a ledger"
    with sqlite3.connect(db) as conn:
        row = conn.execute("SELECT status FROM job_meta WHERE job_id = ?", (job_id,)).fetchone()
    assert row is not None and row[0] == "failed", f"job not marked failed: {row}"
