"""``translate --strict`` delivery-integrity gate.

The run prints "completed" even when blocks failed, are waiting for a human, or
were fail-closed skipped at render (source text left on the page). ``--strict``
turns the already-written quality report into an exit code; this pins the
report-reading half so the gate cannot silently stop firing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from ubt.cli.main import _strict_failures, app
from ubt.core.job_options import sidecar_path


def _report(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "book_quality_report.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_clean_report_passes(tmp_path: Path) -> None:
    path = _report(
        tmp_path,
        {
            "summary": {
                "failed_blocks": 0,
                "needs_human_blocks": 0,
                "blocked_human_blocks": 0,
            },
            "render_coverage": {"skipped_blocks": 0},
        },
    )
    assert _strict_failures(path) == []


def test_failed_and_human_blocks_fail(tmp_path: Path) -> None:
    path = _report(
        tmp_path,
        {
            "summary": {
                "failed_blocks": 2,
                "needs_human_blocks": 1,
                "blocked_human_blocks": 3,
            },
            "render_coverage": {"skipped_blocks": 0},
        },
    )
    failures = _strict_failures(path)
    assert len(failures) == 3
    assert any("2 block(s) failed" in f for f in failures)
    assert any("1 block(s) waiting for human review" in f for f in failures)
    assert any("3 block(s) blocked from shipping" in f for f in failures)


def test_render_skips_fail(tmp_path: Path) -> None:
    path = _report(tmp_path, {"summary": {}, "render_coverage": {"skipped_blocks": 4}})
    failures = _strict_failures(path)
    assert len(failures) == 1
    assert "4 fail-closed render skip(s)" in failures[0]


def test_missing_or_malformed_report_fails_the_gate(tmp_path: Path) -> None:
    """An unverifiable report must fail, not pass.

    ``--strict`` exists to refuse shipping when integrity cannot be verified.
    Returning ``[]`` for an absent or corrupt report made the gate pass in
    exactly that case — a crash after export, a report path mismatch, or a
    truncated write all left the run green with nothing actually checked.
    """
    absent = _strict_failures(tmp_path / "absent.json")
    assert len(absent) == 1
    assert "unreadable" in absent[0]

    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    malformed = _strict_failures(bad)
    assert len(malformed) == 1
    assert "not valid JSON" in malformed[0]

    # A well-formed report of the wrong shape is equally unverifiable.
    shaped = _report(tmp_path, ["not", "an", "object"])
    assert len(_strict_failures(shaped)) == 1

    partial = _report(tmp_path, {"summary": "oops"})
    assert len(_strict_failures(partial)) == 1


# ---------------------------------------------------------------------------
# End-to-end gate through the `translate` command: the strict refusal must
# survive the command's own broad ``except Exception``. typer.Exit subclasses
# RuntimeError, so the handler used to swallow the gate's exit and print a
# second, fabricated JSON object on stdout — breaking the one-JSON-object
# contract that --json consumers rely on.
# ---------------------------------------------------------------------------

runner = CliRunner()


def _run_translate_with_failing_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, json_output: bool
) -> Any:
    """Drive `translate` with a mocked run whose quality report has failures."""
    src = tmp_path / "book.md"
    src.write_text("# Chapter\n\nSome prose to translate.\n", encoding="utf-8")
    out = tmp_path / "book_zh.md"

    async def fake_run_translation(**kwargs: Any) -> Path:
        out.write_text("译文", encoding="utf-8")
        report = sidecar_path(out, "quality_report.json")
        report.write_text(
            json.dumps(
                {
                    "summary": {"failed_blocks": 2},
                    "render_coverage": {"skipped_blocks": 0},
                }
            ),
            encoding="utf-8",
        )
        return out

    monkeypatch.setattr("ubt.cli.main._run_translation", fake_run_translation)
    argv = ["translate", str(src), "--strict"]
    if json_output:
        argv.append("--json")
    return runner.invoke(app, argv)


def test_strict_json_failure_prints_exactly_one_json_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--strict --json must emit ONE object: strict_failed, then exit 1.

    Regression: the swallowed typer.Exit printed a
    second {"status": "failed"} line after the strict_failed one.
    """
    result = _run_translate_with_failing_report(tmp_path, monkeypatch, json_output=True)
    assert result.exit_code == 1
    json_lines = [line for line in result.stdout.splitlines() if line.strip().startswith("{")]
    assert len(json_lines) == 1, f"expected exactly one JSON object, got: {result.stdout!r}"
    payload = json.loads(json_lines[0])
    assert payload["status"] == "strict_failed"
    assert payload["strict_failures"]


def test_strict_human_failure_has_no_fabricated_pipeline_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-JSON mode must report the strict gate, not a bogus pipeline failure."""
    result = _run_translate_with_failing_report(tmp_path, monkeypatch, json_output=False)
    assert result.exit_code == 1
    assert "Strict gate failed" in result.stdout
    # The old bug echoed "Pipeline Execution Failed" with an empty error
    # because the gate's own typer.Exit message had been consumed.
    assert "Pipeline Execution Failed" not in result.stdout
