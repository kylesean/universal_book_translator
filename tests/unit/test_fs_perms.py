"""``fs_perms``: owner-only modes for the files that carry manuscript text.

Covers the two additions from the 2026-09 review: ``.env`` convergence (M1 —
dotenv files are created 0644 by editors/``cp`` while holding API credentials)
and the ``extra_dirs`` extension of the world-readable scan (M2 — the pipeline
only ever walked ``db_dir``, so exported PDFs and quality reports were never
reported).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ubt.core.fs_perms import (
    is_sensitive_path_part,
    restrict_env_file,
    world_readable_files,
)


def test_restrict_env_file_converges_to_owner_only(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("UBT_LLM_API_KEY=sk-secret\n", encoding="utf-8")
    env.chmod(0o644)  # what an editor or `cp` leaves behind

    assert restrict_env_file(env) == env
    assert env.stat().st_mode & 0o777 == 0o600
    # Content is untouched: this is a chmod, not a rewrite.
    assert "sk-secret" in env.read_text(encoding="utf-8")


def test_restrict_env_file_is_idempotent_and_quiet_once_fixed(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("X=1\n", encoding="utf-8")
    env.chmod(0o644)
    assert restrict_env_file(env) is not None
    # Second call converges to the same mode (and logs nothing new).
    assert restrict_env_file(env) == env
    assert env.stat().st_mode & 0o777 == 0o600


def test_restrict_env_file_defaults_to_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """pydantic-settings resolves ``env_file=".env"`` against the cwd, so that
    is the file the boot-time call has to converge."""
    monkeypatch.chdir(tmp_path)
    env = tmp_path / ".env"
    env.write_text("X=1\n", encoding="utf-8")
    env.chmod(0o644)

    assert restrict_env_file() == tmp_path / ".env"
    assert env.stat().st_mode & 0o777 == 0o600


def test_restrict_env_file_absent_is_a_noop(tmp_path: Path) -> None:
    assert restrict_env_file(tmp_path / ".env") is None
    # A directory is not a dotenv file either: no chmod, no exception.
    (tmp_path / ".env").mkdir()
    assert restrict_env_file(tmp_path / ".env") is None


def test_restrict_env_file_never_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A filesystem that refuses the chmod must not abort the boot (M1: warn only)."""
    env = tmp_path / ".env"
    env.write_text("X=1\n", encoding="utf-8")

    def _refuse(self: Path, mode: int) -> None:
        raise OSError("read-only mount")

    monkeypatch.setattr(Path, "chmod", _refuse)
    assert restrict_env_file(env) is None  # warns, returns, never raises


def test_world_readable_files_scans_extra_dirs(tmp_path: Path) -> None:
    ledger_dir = tmp_path / "ledgers"
    ledger_dir.mkdir()
    exposed_ledger = ledger_dir / "job.sqlite"
    exposed_ledger.write_text("text", encoding="utf-8")
    exposed_ledger.chmod(0o644)
    private_ledger = ledger_dir / "private.sqlite"
    private_ledger.write_text("text", encoding="utf-8")
    private_ledger.chmod(0o600)

    export_dir = tmp_path / "export"
    export_dir.mkdir()
    exposed_pdf = export_dir / "book_bilingual.pdf"
    exposed_pdf.write_text("text", encoding="utf-8")
    exposed_pdf.chmod(0o644)

    # Backwards compatible: one argument scans exactly that tree...
    assert world_readable_files(ledger_dir) == [exposed_ledger]
    # ...and the extension reaches the artifact trees db_dir never covered (M2).
    assert sorted(world_readable_files(ledger_dir, extra_dirs=[export_dir])) == sorted(
        [exposed_ledger, exposed_pdf]
    )
    # Overlapping roots are reported once, not twice.
    assert world_readable_files(ledger_dir, extra_dirs=[ledger_dir]) == [exposed_ledger]
    # Missing roots are skipped, not fatal.
    assert world_readable_files(tmp_path / "never-created") == []


def test_sensitive_path_parts_are_casefolded() -> None:
    """L1: the deny list is matched casefolded — ``.SSH`` is ``.ssh`` on a
    case-insensitive filesystem, and the old ``in`` check allowed the read."""
    for name in (".ssh", ".SSH", ".Env", "CREDENTIALS.JSON", ".Git-Credentials"):
        assert is_sensitive_path_part(name), name
    assert not is_sensitive_path_part("book.pdf")
    assert not is_sensitive_path_part("credentials.json.bak")
