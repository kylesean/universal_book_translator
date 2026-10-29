"""Unit tests for the 'ubt patch' CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ubt.cli.main import app

pytestmark = pytest.mark.fast

runner = CliRunner()


def test_patch_diff_command(tmp_path: Path) -> None:
    """Verify ubt patch diff calculates reusable blocks and outputs table/json/markdown."""
    base_file = tmp_path / "v1.md"
    base_file.write_text(
        "# Heading 1\n\nThis is paragraph one.\n\nThis is paragraph two to delete.\n",
        encoding="utf-8",
    )

    rev_file = tmp_path / "v2.md"
    rev_file.write_text(
        "# Heading 1\n\n"
        "This is paragraph one with slight modification.\n\n"
        "This is a brand new paragraph.\n",
        encoding="utf-8",
    )

    # 1. Terminal table output
    res_term = runner.invoke(app, ["patch", "diff", str(base_file), str(rev_file)])
    assert res_term.exit_code == 0
    assert "UBT Book Semantic Diff & Patch Summary" in res_term.stdout
    assert "UNCHANGED" in res_term.stdout

    # 2. Markdown output
    res_md = runner.invoke(app, ["patch", "diff", str(base_file), str(rev_file), "--markdown"])
    assert res_md.exit_code == 0
    assert "### Universal Book Translator: Semantic Revision Patch Summary" in res_md.stdout
    assert "| `unchanged` |" in res_md.stdout

    # 3. JSON output
    res_json = runner.invoke(app, ["patch", "diff", str(base_file), str(rev_file), "--json"])
    assert res_json.exit_code == 0
    data = json.loads(res_json.stdout)
    assert data["total_base_blocks"] == 3
    assert data["total_revised_blocks"] == 3
    assert data["unchanged_count"] == 1

    # 4. Save patch file output
    out_patch = tmp_path / "diff_result.patch.json"
    res_file = runner.invoke(
        app,
        ["patch", "diff", str(base_file), str(rev_file), "--output-patch", str(out_patch)],
    )
    assert res_file.exit_code == 0
    assert out_patch.exists()
    file_data = json.loads(out_patch.read_text(encoding="utf-8"))
    assert file_data["unchanged_count"] == 1


def test_patch_apply_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify ubt patch apply creates seeded ledger job in dry-run and commit mode."""
    db_dir = tmp_path / "ledgers"
    db_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("UBT_DB_DIR", str(db_dir))

    base_file = tmp_path / "base.md"
    base_file.write_text(
        "# Architecture Overview\n\nThe core pipeline consists of five stages.\n",
        encoding="utf-8",
    )

    rev_file = tmp_path / "revised.md"
    rev_file.write_text(
        "# Architecture Overview\n\n"
        "The core pipeline consists of six robust stages.\n\n"
        "A new chapter is born.\n",
        encoding="utf-8",
    )

    # 1. Dry run mode
    res_dry = runner.invoke(
        app,
        ["patch", "apply", str(base_file), str(rev_file), "--dry-run"],
    )
    assert res_dry.exit_code == 0
    assert "Dry-run simulation only" in res_dry.stdout

    # 2. Real application with explicit job id and json output
    res_real = runner.invoke(
        app,
        [
            "patch",
            "apply",
            str(base_file),
            str(rev_file),
            "--new-job-id",
            "job_test_v2",
            "--json",
        ],
    )
    assert res_real.exit_code == 0
    resp = json.loads(res_real.stdout)
    assert resp["status"] == "success"
    assert resp["job_id"] == "job_test_v2"
    assert resp["summary"]["unchanged_count"] == 1
    assert resp["summary"]["total_revised_blocks"] == 3
