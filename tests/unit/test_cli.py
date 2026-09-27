"""Unit tests for Typer and Rich CLI."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from ubt.cli.main import app
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.ir.models import BookManifest, ChapterMeta

runner = CliRunner()


@pytest.fixture
def sample_book_md(tmp_path: Path) -> Path:
    f = tmp_path / "cli_book.md"
    f.write_text(
        "# Chapter 1: Introduction\n\nWelcome to the testing universe.\n", encoding="utf-8"
    )
    return f


def test_cli_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "Universal Book Translator" in result.stdout
    assert "0.1.0" in result.stdout


def test_cli_inspect(sample_book_md: Path) -> None:
    result = runner.invoke(app, ["inspect", str(sample_book_md)])
    assert result.exit_code == 0
    assert "Manifest Summary" in result.stdout
    assert "cli_book" in result.stdout
    assert "Chapter 1: Introduction" in result.stdout


def test_cli_inspect_missing_file(tmp_path: Path) -> None:
    result = runner.invoke(app, ["inspect", str(tmp_path / "missing.md")])
    assert result.exit_code == 1
    assert "Error" in result.stdout


def test_cli_status(tmp_path: Path) -> None:
    db_dir = tmp_path / "ledgers"
    db_dir.mkdir()
    db_path = db_dir / "my_job.sqlite"
    ledger = SQLiteJobLedger(db_path)
    manifest = BookManifest(
        doc_id="doc_xyz",
        title="Status Book",
        source_path="status.md",
        chapters=[ChapterMeta(chapter_id="c1", title="Ch1", spine_index=1)],
    )
    ledger.init_job_from_manifest("my_job", manifest)

    result = runner.invoke(app, ["status", "my_job", "--db-dir", str(db_dir)])
    assert result.exit_code == 0
    assert "Job Ledger Checkpoint Status" in result.stdout
    assert "my_job" in result.stdout


def test_cli_status_nonexistent(tmp_path: Path) -> None:
    result = runner.invoke(app, ["status", "nonexistent_job", "--db-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "Error" in result.stdout


def test_cli_pe_import_rejects_a_traversal_job_id(tmp_path: Path) -> None:
    """A traversal job id must not resolve the ledger outside ``--db-dir``.

    Without the guard ``db_dir / "../outside.sqlite"`` opened (and ran the
    schema migrations on) an arbitrary SQLite file that happened to exist.
    """
    import sqlite3

    db_dir = tmp_path / "ledgers"
    db_dir.mkdir()
    outside = tmp_path / "outside.sqlite"
    con = sqlite3.connect(outside)
    con.execute("CREATE TABLE marker (x INTEGER)")
    con.execute("INSERT INTO marker VALUES (42)")
    con.commit()
    con.close()
    before = outside.read_bytes()

    pe_file = tmp_path / "pe.csv"
    pe_file.write_text("block_id,revised_translation\nb1,译文\n", encoding="utf-8")

    result = runner.invoke(
        app,
        ["pe-import", "../outside", "--file", str(pe_file), "--db-dir", str(db_dir)],
    )

    assert result.exit_code == 1
    assert "Invalid job id" in result.stdout
    assert outside.read_bytes() == before, "the traversal target was opened and mutated"


def test_cli_pe_import_refuses_while_another_writer_holds_the_lock(tmp_path: Path) -> None:
    """``pe-import`` writes paid block state, so it must take the job writer lock.

    Regression: it opened the ledger directly and could race a concurrent resume
    — its human revisions and the pipeline's checkpoints overwriting one another.
    """
    from ubt.core.engine.writer_lock import LedgerWriterLock

    db_dir = tmp_path / "ledgers"
    db_dir.mkdir()
    job_id = "job_locked01"
    db_path = db_dir / f"{job_id}.sqlite"
    # A present-but-empty ledger file is enough to reach the lock acquisition.
    db_path.write_bytes(b"")

    pe_file = tmp_path / "pe.csv"
    pe_file.write_text("block_id,revised_translation\nb1,译文\n", encoding="utf-8")

    holder = LedgerWriterLock(db_path, job_id)
    holder.acquire()
    try:
        result = runner.invoke(
            app,
            ["pe-import", job_id, "--file", str(pe_file), "--db-dir", str(db_dir)],
        )
    finally:
        holder.release()

    assert result.exit_code == 1
    assert "being written by another process" in result.stdout


def test_cli_translate_dry_run(sample_book_md: Path, tmp_path: Path) -> None:
    out_file = tmp_path / "cli_out.md"
    db_dir = tmp_path / "cli_ledgers"
    result = runner.invoke(
        app,
        [
            "translate",
            str(sample_book_md),
            "-o",
            str(out_file),
            "--source-lang",
            "en",
            "--target-lang",
            "zh",
            "--dry-run",
            "--db-dir",
            str(db_dir),
        ],
    )
    assert result.exit_code == 0
    assert "Translation Completed Successfully" in result.stdout
    assert out_file.exists()
    assert out_file.read_text(encoding="utf-8").count("[模拟翻译]") >= 1


def test_cli_inspect_job_id_redirection(tmp_path: Path) -> None:
    db_dir = tmp_path / "ledgers"
    db_dir.mkdir()
    db_path = db_dir / "job_abc123.sqlite"
    ledger = SQLiteJobLedger(db_path)
    manifest = BookManifest(
        doc_id="doc_test",
        title="Inspected Book",
        source_path="test.md",
        chapters=[ChapterMeta(chapter_id="c1", title="Ch1", spine_index=1)],
    )
    ledger.init_job_from_manifest("job_abc123", manifest)

    result = runner.invoke(app, ["inspect", "job_abc123", "--db-dir", str(db_dir)])
    assert result.exit_code == 0
    assert "recognized as a Job ID" in result.stdout
    assert "Job Ledger Checkpoint Status" in result.stdout


def test_cli_translate_with_chapter_bounds(sample_book_md: Path, tmp_path: Path) -> None:
    out_file = tmp_path / "cli_bounded_out.md"
    db_dir = tmp_path / "cli_bounded_ledgers"
    result = runner.invoke(
        app,
        [
            "translate",
            str(sample_book_md),
            "-o",
            str(out_file),
            "--dry-run",
            "--db-dir",
            str(db_dir),
            "--start-chapter",
            "1",
            "--max-chapters",
            "1",
        ],
    )
    assert result.exit_code == 0
    assert "Translation Completed Successfully" in result.stdout
    assert out_file.exists()


def test_doctor_ok_with_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UBT_LLM_API_KEY", "test-key-12345678")
    # An explicit model: the shipped benchmark default is now a deliberate WARN
    # (test_doctor_warns_when_models_are_the_shipped_default), so a healthy run
    # means the operator has actually chosen one.
    monkeypatch.setenv("UBT_DRAFT_MODEL", "deepseek-chat")
    monkeypatch.setenv("UBT_REPAIR_MODEL", "deepseek-chat")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "All checks passed" in result.stdout


def test_doctor_warns_when_models_are_the_shipped_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shipped default names the author's benchmark target, not a model a
    normal credential can call; leaving it in place must be called out, not
    printed as OK."""
    monkeypatch.setenv("UBT_LLM_API_KEY", "test-key-12345678")
    monkeypatch.delenv("UBT_DRAFT_MODEL", raising=False)
    monkeypatch.delenv("UBT_REPAIR_MODEL", raising=False)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "shipped benchmark default" in result.stdout


def test_doctor_accepts_a_providers_own_default_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A selected provider supplies its models on purpose — even when they match
    the shipped default (opencode's do), which is not an oversight to WARN about."""
    monkeypatch.setenv("UBT_LLM_API_KEY", "test-key-12345678")
    monkeypatch.setenv("UBT_PROVIDER", "opencode")
    monkeypatch.delenv("UBT_DRAFT_MODEL", raising=False)
    monkeypatch.delenv("UBT_REPAIR_MODEL", raising=False)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["doctor", "--json"])

    payload = json.loads(result.stdout)
    models = next(check for check in payload["checks"] if check["name"] == "Models")
    assert models["status"] == "OK"
    assert models["detail"] == (
        "draft=muse-spark-1.3-contributor repair=muse-spark-1.3-contributor"
    )


def test_doctor_fails_without_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UBT_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "UBT_LLM_API_KEY" in result.stdout


def test_doctor_rejects_invalid_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UBT_QE_ENGINE", "bogus")
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 2
    assert "qe_engine" in result.stdout


def test_doctor_json_emits_one_object_with_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--json`` must put exactly one parseable object on stdout, no Rich chrome.

    Doctor was the only command with no machine-readable mode, so a CI gate had
    to scrape the human checklist to learn whether the environment was ready.
    """
    monkeypatch.setenv("UBT_LLM_API_KEY", "test-key-12345678")
    monkeypatch.setenv("UBT_DRAFT_MODEL", "deepseek-chat")
    monkeypatch.setenv("UBT_REPAIR_MODEL", "deepseek-chat")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["doctor", "--json"])

    # json.loads raises if any table border / summary line leaked into stdout.
    payload = json.loads(result.stdout)
    assert result.exit_code == 0
    assert payload["status"] != "fail"
    assert payload["summary"]["fail"] == 0
    assert payload["summary"]["ok"] > 0
    names = {check["name"] for check in payload["checks"]}
    assert {"API key", "Base URL", "Models"} <= names
    for check in payload["checks"]:
        assert {"name", "status", "detail"} <= set(check)


def test_doctor_json_reports_fail_and_keeps_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("UBT_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["doctor", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "fail"
    assert payload["summary"]["fail"] >= 1
    api_key = next(check for check in payload["checks"] if check["name"] == "API key")
    assert api_key["status"] == "FAIL"


def test_doctor_json_on_invalid_config_is_one_error_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A config error under ``--json`` is still one object, not a traceback."""
    monkeypatch.setenv("UBT_QE_ENGINE", "bogus")

    result = runner.invoke(app, ["doctor", "--json"])

    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert payload["code"] == "config_invalid"
    assert "qe_engine" in result.stdout


def test_doctor_collapses_ledger_fix_to_scan_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The chmod fix names the scan roots, not every per-asset hash directory.

    The docling cache keeps one hash directory per asset, so listing each
    exposed file's immediate parent printed a near-identical long path per
    asset and buried the one command the operator must actually run. The root
    also stays correct as new hash directories appear.
    """
    monkeypatch.setenv("UBT_LLM_API_KEY", "test-key-12345678")
    monkeypatch.setenv("UBT_DRAFT_MODEL", "deepseek-chat")
    monkeypatch.setenv("UBT_REPAIR_MODEL", "deepseek-chat")
    monkeypatch.chdir(tmp_path)
    output_dir = tmp_path / "output"
    monkeypatch.setenv("UBT_OUTPUT_DIR", str(output_dir))

    cache_asset = tmp_path / ".ubt" / "docling_cache" / "assets" / "deadbeef"
    cache_asset.mkdir(parents=True)
    (cache_asset / "page.png").write_text("x", encoding="utf-8")
    (cache_asset / "page.png").chmod(0o644)
    output_dir.mkdir()
    (output_dir / "book_bilingual.md").write_text("x", encoding="utf-8")
    (output_dir / "book_bilingual.md").chmod(0o644)

    result = runner.invoke(app, ["doctor"])

    # Strip every whitespace run, not just newlines: Rich folds long paths
    # mid-word at the terminal width, so a " ".join would re-split them.
    compact = "".join(result.stdout.split())
    assert "chmod-Rgo-rwx" in compact
    fix_command = compact.split("chmod-Rgo-rwx", 1)[1]
    assert "'.ubt/docling_cache'" in fix_command
    assert f"'{output_dir}'" in fix_command
    assert "deadbeef" not in fix_command


def test_require_api_key_strict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ubt.core.config import require_api_key

    monkeypatch.setenv("UBT_LLM_API_KEY", "test-key-12345678")
    assert require_api_key() == "test-key-12345678"
    monkeypatch.delenv("UBT_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        require_api_key()


def test_cli_translate_help_lists_fresh() -> None:
    """--fresh must be discoverable: stale-resume is silent otherwise."""
    result = runner.invoke(app, ["translate", "--help"])
    assert result.exit_code == 0
    assert "--fresh" in result.stdout


def test_cli_translate_rolling_summary_defaults(
    monkeypatch: pytest.MonkeyPatch, sample_book_md: Path
) -> None:
    captured_rolling: list[object] = []

    async def _mock_run_translation(**kwargs: object) -> Path:
        captured_rolling.append(kwargs.get("enable_rolling_summary"))
        return sample_book_md

    monkeypatch.setattr("ubt.cli.main._run_translation", _mock_run_translation)

    # 1. Default invocation (no rolling-summary flag)
    runner.invoke(app, ["translate", str(sample_book_md)])
    assert captured_rolling[-1] is None

    # 2. Explicit --no-rolling-summary
    runner.invoke(app, ["translate", str(sample_book_md), "--no-rolling-summary"])
    assert captured_rolling[-1] is False

    # 3. Explicit --rolling-summary
    runner.invoke(app, ["translate", str(sample_book_md), "--rolling-summary"])
    assert captured_rolling[-1] is True


def test_cli_config_backed_flags_arrive_unset_as_none(
    monkeypatch: pytest.MonkeyPatch, sample_book_md: Path
) -> None:
    """Regression: typer literals ("inline"/"auto"/False/
    Path(".ubt/ledgers")) used to reach the engine as if the user had typed
    them, silently erasing UBT_* env and ubt.toml values."""
    captured: list[dict[str, Any]] = []

    async def _mock_run_translation(**kwargs: Any) -> Path:
        captured.append(kwargs)
        return sample_book_md

    monkeypatch.setattr("ubt.cli.main._run_translation", _mock_run_translation)

    runner.invoke(app, ["translate", str(sample_book_md)])
    defaults = captured[-1]
    for key in (
        "budget_usd",
        "db_dir",
        "dual_mode",
        "translate_chrome",
        "facing_spread",
        "cover_mode",
        "fresh",
        "ocr_mode",
        "formula_mode",
    ):
        assert defaults[key] is None, (
            f"unset --{key.replace('_', '-')} must reach the engine as None"
        )

    runner.invoke(
        app,
        [
            "translate",
            str(sample_book_md),
            "--fresh",
            "--dual-mode",
            "monolingual",
            "--ocr",
            "off",
            "--db-dir",
            "/tmp/wherever",
            "--formula-mode",
            "strict",
            "--translate-chrome",
            "--budget-usd",
            "5",
        ],
    )
    explicit = captured[-1]
    assert explicit["budget_usd"] == 5.0
    assert explicit["fresh"] is True
    assert explicit["dual_mode"] == "monolingual"
    assert explicit["ocr_mode"] == "off"
    assert explicit["db_dir"] == Path("/tmp/wherever")
    assert explicit["formula_mode"] == "strict"
    assert explicit["translate_chrome"] is True


def test_cli_unset_flags_preserve_env_config(
    monkeypatch: pytest.MonkeyPatch, sample_book_md: Path, tmp_path: Path
) -> None:
    """End-to-end through the real _run_translation: unset flags must not
    appear in the override set, and env-configured values must survive."""
    from ubt.cli import main as cli_main

    monkeypatch.setenv("UBT_DUAL_MODE", "monolingual")
    monkeypatch.setenv("UBT_DB_DIR", str(tmp_path / "env_ledgers"))

    seen_overrides: dict[str, object] = {}
    built: list[Any] = []
    real_build = cli_main._build_config

    def spy(overrides: dict[str, Any]) -> Any:
        seen_overrides.update(overrides)
        cfg = real_build(overrides)
        built.append(cfg)
        return cfg

    monkeypatch.setattr(cli_main, "_build_config", spy)

    out = tmp_path / "env_out.md"
    result = runner.invoke(app, ["translate", str(sample_book_md), "-o", str(out), "--dry-run"])
    assert result.exit_code == 0
    assert built, "_build_config was never called"
    for key in ("dual_mode", "db_dir", "fresh", "ocr_mode", "cover_mode", "formula_mode"):
        assert key not in seen_overrides, f"unset --{key} leaked into overrides"
    assert built[-1].dual_mode == "monolingual"
    assert built[-1].db_dir == tmp_path / "env_ledgers"


def test_cli_dual_mode_facing_derives_spread(
    monkeypatch: pytest.MonkeyPatch, sample_book_md: Path, tmp_path: Path
) -> None:
    """--dual-mode facing implies facing_spread=True; an explicit
    --no-facing-spread still wins over the implication."""
    from ubt.cli import main as cli_main

    seen: dict[str, object] = {}
    real_build = cli_main._build_config

    def spy(overrides: dict[str, Any]) -> Any:
        seen.clear()
        seen.update(overrides)
        return real_build(overrides)

    monkeypatch.setattr(cli_main, "_build_config", spy)

    out = tmp_path / "face_out.md"
    result = runner.invoke(
        app,
        ["translate", str(sample_book_md), "-o", str(out), "--dry-run", "--dual-mode", "facing"],
    )
    assert result.exit_code == 0
    assert seen.get("facing_spread") is True

    result = runner.invoke(
        app,
        [
            "translate",
            str(sample_book_md),
            "-o",
            str(out),
            "--dry-run",
            "--dual-mode",
            "facing",
            "--no-facing-spread",
        ],
    )
    assert result.exit_code == 0
    assert seen.get("facing_spread") is False


def test_cli_status_follows_UBT_DB_DIR_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without --db-dir, status/inspect must look in the configured dir."""
    env_db = tmp_path / "ledgers_elsewhere"
    env_db.mkdir()
    ledger = SQLiteJobLedger(env_db / "envjob.sqlite")
    manifest = BookManifest(
        doc_id="doc_env",
        title="Env Book",
        source_path="env.md",
        chapters=[ChapterMeta(chapter_id="c1", title="Ch1", spine_index=1)],
    )
    ledger.init_job_from_manifest("envjob", manifest)
    ledger.close()

    monkeypatch.setenv("UBT_DB_DIR", str(env_db))
    monkeypatch.chdir(tmp_path)  # no ./.ubt/ledgers here
    result = runner.invoke(app, ["status", "envjob"])
    assert result.exit_code == 0
    assert "envjob" in result.stdout


def test_cli_out_of_range_concurrency_override_is_rejected() -> None:
    """Risk: ``--concurrency 0`` bypassed pydantic through ``model_copy``,
    reached ``asyncio.Semaphore(0)`` and the run hung forever — no error, no
    log, no way for the user to know the job would never start."""
    from pydantic import ValidationError

    from ubt.cli.main import _build_config

    with pytest.raises(ValidationError, match="max_concurrency"):
        _build_config({"max_concurrency": 0})


def test_cli_other_out_of_range_override_is_rejected() -> None:
    """Risk: the same unchecked override path silently accepted any out-of-range
    value (e.g. zero pages for the short chain) instead of failing fast."""
    from pydantic import ValidationError

    from ubt.cli.main import _build_config

    with pytest.raises(ValidationError, match="short_max_pages"):
        _build_config({"short_max_pages": 0})


def test_cli_valid_overrides_still_apply_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Risk: the validation fix must not change override precedence — an
    explicit flag still beats the environment, and untouched fields keep it."""
    from ubt.cli.main import _build_config

    monkeypatch.setenv("UBT_MAX_CONCURRENCY", "9")
    cfg = _build_config({"max_concurrency": 4, "batch_limit": 7, "pages": "2-4"})
    assert cfg.max_concurrency == 4
    assert cfg.batch_limit == 7
    assert cfg.pages == "2-4"


def test_cli_overrides_run_the_credential_invariant(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Risk: the override path skipped the model_validator, so a config that
    reused the inbound X-API-Key gate secret as the outbound LLM key was
    accepted and every API client ended up holding the provider credential."""
    from pydantic import ValidationError

    from ubt.cli.main import _build_config
    from ubt.core.config import MOCK_API_KEY

    monkeypatch.delenv("UBT_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    with pytest.raises(ValidationError, match="must differ"):
        _build_config({"service_api_key": MOCK_API_KEY})


def test_cli_overrides_run_the_qe_engine_normalizer() -> None:
    """Risk: ``model_copy`` skipped the field validators, so the deprecated
    ``--qe-engine comet`` reached the run config unconverted instead of the
    canonical ``subprocess`` scorer it is documented to mean."""
    from ubt.cli.main import _build_config

    cfg = _build_config({"qe_engine": "comet"})
    assert cfg.qe_engine == "subprocess"


def test_cli_translate_reports_out_of_range_override(sample_book_md: Path, tmp_path: Path) -> None:
    """Risk: an out-of-range flag must surface as a clear error and a non-zero
    exit code before the run starts (it used to be applied unvalidated)."""
    result = runner.invoke(
        app,
        [
            "translate",
            str(sample_book_md),
            "--dry-run",
            "--db-dir",
            str(tmp_path / "ledgers"),
            "--short-max-pages",
            "0",
        ],
    )
    assert result.exit_code == 1
    assert "short_max_pages" in result.stdout


def test_safety_gates_ship_closed() -> None:
    """Three defaults that were once open, and stay shut unless asked.

    A blocking visual gate, page images leaving the machine, and an export that
    would otherwise render a half-translated book and call it completed. The
    visual gate is forced on for the short chain; the other two are read off the
    job's config at use time, so shipping them open means changing the default
    here — which is what this pins. The behaviour each one
    guards is tested where it happens (``test_audit_2026_09_20_regressions`` for
    the export completion ratio, ``test_pluggable_ocr`` for page egress).
    """
    from ubt.core.config import UBTConfig

    config = UBTConfig()
    assert config.visual_blocking_gate_enabled is False
    assert config.allow_page_upload is False
    assert config.export_min_completion_ratio == 0.5


def test_cli_verbose_logging_and_model_copy(sample_book_md: Path, tmp_path: Path) -> None:
    out_file = tmp_path / "cli_out_verbose.md"
    db_dir = tmp_path / "cli_ledgers_verbose"
    result = runner.invoke(
        app,
        [
            "-v",
            "translate",
            str(sample_book_md),
            "-o",
            str(out_file),
            "--dry-run",
            "--db-dir",
            str(db_dir),
            "-v",
        ],
    )
    assert result.exit_code == 0
    assert out_file.exists()
    assert out_file.read_text(encoding="utf-8").count("[模拟翻译]") >= 1


def test_cli_page_range_options(sample_book_md: Path, tmp_path: Path) -> None:
    """--pages / --page-range parse on the CLI, but are rejected at ingest for a
    non-PDF input (which has no page geometry) instead of silently translating —
    and billing — the whole document. Markdown is such an input, so a page-ranged
    run must fail loudly with the PDF-only message."""
    out_file = tmp_path / "cli_out_pages.md"
    db_dir = tmp_path / "cli_ledgers_pages"
    # Test --pages
    result = runner.invoke(
        app,
        [
            "translate",
            str(sample_book_md),
            "-o",
            str(out_file),
            "--dry-run",
            "--db-dir",
            str(db_dir),
            "--pages",
            "1-2",
        ],
    )
    assert result.exit_code != 0
    assert "only supported for PDF" in result.output

    # Test --page-range alias
    out_file2 = tmp_path / "cli_out_pages2.md"
    result2 = runner.invoke(
        app,
        [
            "translate",
            str(sample_book_md),
            "-o",
            str(out_file2),
            "--dry-run",
            "--db-dir",
            str(db_dir),
            "--page-range",
            "1-2",
        ],
    )
    assert result2.exit_code != 0
    assert "only supported for PDF" in result2.output


def test_parse_page_ranges_unit(monkeypatch: pytest.MonkeyPatch) -> None:
    from ubt.core.config import UBTConfig, parse_page_ranges

    assert parse_page_ranges(None) is None
    assert parse_page_ranges("") is None
    assert parse_page_ranges("   ") is None
    assert parse_page_ranges("1-3") == {1, 2, 3}
    assert parse_page_ranges("1, 3, 5") == {1, 3, 5}
    assert parse_page_ranges("1-2, 5, 7-8") == {1, 2, 5, 7, 8}
    assert parse_page_ranges("4") == {4}

    # UBT_PAGES is the canonical env name (a bare PAGES must be ignored).
    monkeypatch.setenv("UBT_PAGES", "2-4")
    cfg = UBTConfig()
    assert cfg.get_selected_pages() == {2, 3, 4}

    with pytest.raises(ValueError, match="Invalid page range"):
        parse_page_ranges("5-2")
    with pytest.raises(ValueError, match="Page numbers must be >= 1"):
        parse_page_ranges("0-2")
    with pytest.raises(ValueError, match="Invalid page range specification"):
        parse_page_ranges("abc-xyz")
    # The raw spec is length-capped before it is split (body-size DoS guard).
    with pytest.raises(ValueError, match="Page specification too long"):
        parse_page_ranges(",".join(str(i) for i in range(1, 100_002)))
    # The materialized set is bounded in TOTAL, not just per contiguous span:
    # two in-cap spans whose union exceeds the cap must still be refused.
    with pytest.raises(ValueError, match="Too many pages"):
        parse_page_ranges("1-60000,70000-110001")


def test_cli_translate_help_lists_credential_options() -> None:
    result = runner.invoke(app, ["translate", "--help"], env={"COLUMNS": "160"})
    assert result.exit_code == 0
    assert "--api-key" in result.stdout
    assert "--base-url" in result.stdout
    assert "--api-mode" in result.stdout
    assert "--provider" in result.stdout


def test_cli_translate_forwards_credentials_and_profile(
    monkeypatch: pytest.MonkeyPatch, sample_book_md: Path
) -> None:
    captured: dict[str, Any] = {}

    async def _mock_run_translation(**kwargs: Any) -> Path:
        captured.update(kwargs)
        return sample_book_md

    monkeypatch.setattr("ubt.cli.main._run_translation", _mock_run_translation)

    result = runner.invoke(
        app,
        [
            "translate",
            str(sample_book_md),
            "--api-key",
            "sk-custom-cli-key",
            "--base-url",
            "https://generativelanguage.googleapis.com/v1beta/openai",
            "--api-mode",
            "openai-chat",
            "--provider",
            "gemini",
        ],
    )
    assert result.exit_code == 0
    assert captured["api_key"] == "sk-custom-cli-key"
    assert captured["base_url"] == "https://generativelanguage.googleapis.com/v1beta/openai"
    assert captured["api_mode"] == "openai-chat"
    assert captured["provider"] == "gemini"


def test_doctor_warns_when_configured_models_are_unpriced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unpriced model makes the money features inert, and must say so.

    With no price-table entry the cost reports as unknown, so `--budget-usd` can
    never trip and the pre-flight estimate prints "unknown" — silently true for
    the project's own bundled default model.
    """
    monkeypatch.setenv("UBT_LLM_API_KEY", "test-key-12345678")
    monkeypatch.setenv("UBT_DRAFT_MODEL", "brand-new-model-9000")
    monkeypatch.setenv("UBT_REPAIR_MODEL", "brand-new-model-9000")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["doctor"])

    assert "Cost pricing" in result.stdout
    assert "brand-new-model-9000" in result.stdout
    assert "WARN" in result.stdout
    assert "MODEL_PRICES_USD_PER_MTOK" in result.stdout


def test_doctor_reports_local_endpoint_as_free_not_unpriced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Self-hosted endpoints are free by construction, not unpriced.

    WARNing "no price-table entry" for ``127.0.0.1`` tells an Ollama/llama.cpp
    user their money features are broken when the run in fact costs $0 — the
    false alarm that made the diagnostic distrust its own local path.
    """
    monkeypatch.setenv("UBT_LLM_API_KEY", "test-key-12345678")
    monkeypatch.setenv("UBT_DRAFT_MODEL", "translategemma:4b")
    monkeypatch.setenv("UBT_REPAIR_MODEL", "translategemma:4b")
    monkeypatch.setenv("UBT_BASE_URL", "http://127.0.0.1:9090/v1")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["doctor"])

    line = next(row for row in result.stdout.splitlines() if "Cost pricing" in row)
    assert "OK" in line and "WARN" not in line, line
    assert "local" in line.lower(), line


def test_over_long_job_id_is_refused_on_every_surface() -> None:
    """The length cap is shared, not API-only.

    ``job_options`` claimed "each keeps its own length cap", but only the REST
    API had one. A 100k-char ``--job-id`` became ``<job_id>.sqlite`` and failed
    with ENAMETOOLONG mid-run instead of a clean validation error.
    """
    from ubt.core.exceptions import UBTError
    from ubt.core.job_options import JOB_ID_MAX_LEN, job_id_is_valid
    from ubt.mcp.server import _check_job_id

    assert job_id_is_valid("job_ok-1")
    assert not job_id_is_valid("")
    assert not job_id_is_valid("../escape")
    assert job_id_is_valid("a" * JOB_ID_MAX_LEN)
    assert not job_id_is_valid("a" * (JOB_ID_MAX_LEN + 1))

    # The MCP surface enforces the same rule, not just the pattern.
    with pytest.raises(UBTError):
        _check_job_id("a" * (JOB_ID_MAX_LEN + 1))


def test_doctor_reports_pricing_covered_for_priced_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UBT_LLM_API_KEY", "test-key-12345678")
    monkeypatch.setenv("UBT_DRAFT_MODEL", "deepseek-chat")
    monkeypatch.setenv("UBT_REPAIR_MODEL", "deepseek-chat")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["doctor"])

    line = next(row for row in result.stdout.splitlines() if "Cost pricing" in row)
    assert "OK" in line and "WARN" not in line, line


def test_doctor_warns_on_missing_toolchain_that_silently_degrades(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The probes must name what degrades, not print OK for a broken install.

    ``pdf_engine`` defaults to "auto", which routes layout-heavy PDFs to
    docling — an optional extra the README quickstart does not install, and the
    degraded result (one fused block per page) reads downstream as a book that
    cannot be translated. MathJax is the advertised vector-formula backend and
    needs Node plus ``scripts/mathjax/node_modules``. Both rows used to be OK.
    """
    monkeypatch.setenv("UBT_LLM_API_KEY", "test-key-12345678")
    monkeypatch.setenv("UBT_MATH_BACKEND", "mathjax")
    monkeypatch.chdir(tmp_path)
    import importlib.util

    real_find_spec = importlib.util.find_spec

    def _no_docling(name: str, *args: object, **kwargs: object) -> object:
        return None if name == "docling" else real_find_spec(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(importlib.util, "find_spec", _no_docling)
    from ubt.adapters.pdf.math_renderer import MathjaxRenderer

    monkeypatch.setattr(MathjaxRenderer, "available", lambda self: False)

    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "PDF engine" in result.stdout
    assert "docling" in result.stdout
    assert "MathJax" in result.stdout
    assert "warning(s)" in result.stdout


def test_cli_request_surface_is_recognized_by_job_options(
    monkeypatch: pytest.MonkeyPatch, sample_book_md: Path
) -> None:
    """Every forwarded key must live in the shared request schema.

    ``overrides_from_request`` drops unrecognized keys silently (an API
    payload may carry extras), which is exactly the drift a hand-mirrored CLI
    parameter list used to allow: one surface missing one field, invisible
    until a user hits it.
    """
    captured: list[dict[str, Any]] = []

    async def _mock_run_translation(**kwargs: Any) -> Path:
        captured.append(kwargs)
        return sample_book_md

    monkeypatch.setattr("ubt.cli.main._run_translation", _mock_run_translation)
    runner.invoke(app, ["translate", str(sample_book_md)])

    from ubt.core.config import UBTConfig
    from ubt.core.job_options import RUN_ONLY_KEYS

    recognized = (
        set(UBTConfig.model_fields)
        | set(RUN_ONLY_KEYS)
        | {"glossary"}  # alias for glossary_path
        | {"preset"}  # preset bundle, consumed by overrides_from_request
        | {"dry_run", "quiet"}  # CLI display/lifecycle, never config
    )
    unknown = set(captured[-1]) - recognized
    assert not unknown, f"CLI forwards request keys the shared mapping drops: {sorted(unknown)}"


def test_translate_rejects_malformed_lang_codes(sample_book_md: Path) -> None:
    """API and MCP reject non-ISO-ish language tags up front; the CLI used
    to pass them into derive_job_id / the ledger file name and fail deep
    with an opaque sqlite or Typst error."""
    for bad in ("zh ;", "../../etc/passwd", "中文"):
        result = runner.invoke(app, ["translate", str(sample_book_md), "-l", bad])
        assert result.exit_code != 0
        assert "Invalid target-lang" in result.output, (bad, result.output)
        result_src = runner.invoke(app, ["translate", str(sample_book_md), "-s", bad, "--dry-run"])
        assert result_src.exit_code != 0
        assert "Invalid source-lang" in result_src.output, (bad, result_src.output)
    # A legit code passes the shape gate and the language-profile layer
    # downstream (dry-run reaches completion).
    result_ok = runner.invoke(app, ["translate", str(sample_book_md), "-l", "ja", "--dry-run"])
    assert result_ok.exit_code == 0, result_ok.output


def test_cli_translate_handles_none_result_path(tmp_path: Path) -> None:
    import json
    from unittest.mock import AsyncMock, patch

    fake_book = tmp_path / "book.md"
    fake_book.write_text("# Chapter 1\nHello", encoding="utf-8")

    # Mock _run_translation to return None
    with patch("ubt.cli.commands.translate._get_run_translation") as mock_get_run:
        mock_run = AsyncMock(return_value=None)
        mock_get_run.return_value = mock_run

        result = runner.invoke(app, ["translate", str(fake_book), "--json"])
        # Should exit with code 1 cleanly, not crash with AttributeError
        assert result.exit_code == 1
        data = json.loads(result.stdout)
        assert data["status"] == "failed"


def test_cli_worker_command_passes_db_dir_to_config(tmp_path: Path) -> None:
    from unittest.mock import AsyncMock, MagicMock, patch

    from ubt.cli.commands.worker import worker_command

    custom_db_dir = tmp_path / "custom_dbs"
    custom_db_dir.mkdir()

    with (
        patch("ubt.core.engine.job_queue.JobQueue"),
        patch("ubt.core.engine.job_worker.JobWorker") as mock_worker_cls,
        patch("ubt.core.router.rate_limiter.build_rate_limiter"),
    ):
        mock_worker = MagicMock()
        mock_worker.run_until_idle = AsyncMock(return_value=0)
        mock_worker.failed_jobs = 0
        mock_worker_cls.return_value = mock_worker

        worker_command(db_dir=custom_db_dir, once=True)

        # JobWorker must have been called with a config whose db_dir == custom_db_dir
        call_config = mock_worker_cls.call_args[0][1]
        assert call_config.db_dir == custom_db_dir


@pytest.mark.fast
def test_cli_translate_invalid_lang_json_output() -> None:
    from typer.testing import CliRunner

    from ubt.cli.main import app

    r = CliRunner()
    result = r.invoke(
        app, ["translate", "nonexistent.md", "--target-lang", "invalid_lang_code!!", "--json"]
    )
    assert result.exit_code != 0
    # stdout must be parseable JSON
    data = json.loads(result.stdout)
    assert data.get("success") is False or data.get("status") == "failed"
    assert "invalid" in data.get("error", "").lower()


@pytest.mark.fast
def test_cli_status_invalid_job_id_json_output() -> None:
    from typer.testing import CliRunner

    from ubt.cli.main import app

    r = CliRunner()
    result = r.invoke(app, ["status", "invalid/path/job", "--json"])
    assert result.exit_code == 2
    data = json.loads(result.stdout)
    assert data.get("status") == "error"
    assert "invalid" in data.get("error", "").lower()


@pytest.mark.fast
def test_cli_translate_domain_profile_alias_and_validation(sample_book_md: Path) -> None:
    """--domain-profile must be accepted as alias for --profile, and invalid profiles rejected."""
    # 1. Check --help shows --domain-profile (ensure wide enough column width so Rich doesn't truncate)
    runner_wide = CliRunner(env={"COLUMNS": "200"})
    res_help = runner_wide.invoke(app, ["translate", "--help"])
    assert res_help.exit_code == 0
    assert "--domain-profile" in res_help.stdout

    # 2. Reject a path-shaped profile (the shared safe-name pattern). The old
    #    narrow allowlist rejected name-shaped profiles that the API/MCP accept
    #    — including every profile that actually carries glossary seeds.
    res_invalid = runner.invoke(
        app, ["translate", str(sample_book_md), "--domain-profile", "../evil"]
    )
    assert res_invalid.exit_code != 0
    assert "invalid domain profile" in res_invalid.stdout.lower()

    # 3. Accept a valid profile via --domain-profile alias, and a seeded profile
    #    the API/MCP already accepted (parity).
    for profile in ("textbook", "semiconductor"):
        res_valid = runner.invoke(
            app, ["translate", str(sample_book_md), "--domain-profile", profile, "--dry-run"]
        )
        assert "invalid domain profile" not in res_valid.stdout.lower(), profile


def test_json_stdout_is_pure_json_in_a_fresh_process(tmp_path: Path) -> None:
    import json
    import subprocess
    import sys

    book = tmp_path / "probe.md"
    book.write_text("# Chapter 1\n\nA short technical note.\n", encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, "-m", "ubt", "assess", str(book), "--json"],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)  # exactly one object, no log preamble
    assert payload["status"] == "ok"


@pytest.mark.fast
def test_rigid_engine_warning_keeps_json_stdout_pure(tmp_path: Path) -> None:
    """The advisory warning must not precede the JSON object on stdout."""
    book = tmp_path / "probe.md"
    book.write_text("# Chapter 1\n\nA short technical note.\n", encoding="utf-8")
    env = {
        **os.environ,
        "UBT_RENDER_ENGINE": "rigid",
        "UBT_OUTPUT_DIR": str(tmp_path / "out"),
    }
    # Under Profile-Aware adaptive defaults, an unset --dual-mode adapts cleanly
    # to 'monolingual' for rigid engines without a spurious warning. Pass
    # '--dual-mode inline' explicitly to trigger the downgrade advisory and verify
    # that it routes to stderr without corrupting the JSON payload on stdout.
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ubt",
            "translate",
            str(book),
            "--dual-mode",
            "inline",
            "--dry-run",
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=tmp_path,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)  # exactly one JSON object, no preamble
    assert payload["status"] == "completed"
    # The warning is still delivered -- just not on the machine-readable stream.
    assert "monolingual-only" in proc.stderr


@pytest.mark.fast
def test_translate_cli_and_main_share_same_rich_console() -> None:
    """translate.py and main.py must share one Rich Console instance so RichHandler
    does not tear the Progress live bar."""
    import ubt.cli.commands.translate as translate_mod
    import ubt.cli.main as main_mod

    assert translate_mod.console is main_mod.console


def test_server_bind_flags_are_actually_parsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from ubt.api.app import _resolve_bind

    monkeypatch.setattr(sys, "argv", ["ubt-api", "--host", "0.0.0.0", "--port", "9000"])
    assert _resolve_bind(None, None) == ("0.0.0.0", 9000)
    monkeypatch.setattr(sys, "argv", ["ubt-api"])
    assert _resolve_bind(None, None) == ("127.0.0.1", 8000)
    # Explicit arguments still win (the bind-guard regression calls this way).
    assert _resolve_bind("0.0.0.0", 1) == ("0.0.0.0", 1)


def test_translate_accepts_a_seeded_domain_profile(tmp_path: Path) -> None:
    """CLI's old allowlist rejected every profile that had glossary seeds.

    'semiconductor' is a packaged glossary directory (seed_entries_for_profile),
    so the CLI must accept it like the API/MCP do; the run then fails on the
    missing input, not on the profile.
    """
    missing = tmp_path / "nope.md"
    result = runner.invoke(app, ["translate", str(missing), "--profile", "semiconductor"])
    assert "Invalid domain profile" not in result.output
    assert "Input file not found" in result.output


def test_cli_tm_scan_and_evict_round_trip(tmp_path: Path) -> None:
    """``TM.scan()``/``evict_ids()`` are reachable from the CLI.

    Reusable memory that can only grow and never be corrected is a liability; a
    poisoned entry is served verbatim on every later run.
    """
    import json

    from ubt.core.memory.tm import TMPendingEntry, TranslationMemory

    db_dir = tmp_path / "ledgers"
    db_dir.mkdir()
    tm = TranslationMemory(db_dir / "tm.sqlite")
    try:
        tm.writeback(
            [
                TMPendingEntry(
                    src_lang="en",
                    tgt_lang="zh",
                    source_text="A poisoned source.",
                    target_text="被污染的译文。",
                )
            ]
        )
        entries = tm.scan()
    finally:
        tm.close()
    assert len(entries) == 1
    entry_id = entries[0].id

    scanned = runner.invoke(app, ["tm", "scan", "--db-dir", str(db_dir), "--json"])
    assert scanned.exit_code == 0, scanned.stdout
    payload = json.loads(scanned.stdout.strip().splitlines()[-1])
    assert payload[0]["source_text"] == "A poisoned source."

    evicted = runner.invoke(app, ["tm", "evict", str(entry_id), "--db-dir", str(db_dir), "--yes"])
    assert evicted.exit_code == 0, evicted.stdout
    assert "Evicted 1" in evicted.stdout
    tm2 = TranslationMemory(db_dir / "tm.sqlite")
    try:
        assert tm2.scan() == []
    finally:
        tm2.close()
