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
from fastapi import HTTPException

from ubt.api.app import SYSTEM_DISALLOWED_PREFIXES, resolve_secure_path
from ubt.core.config import UBTConfig
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


def test_validate_safe_path_security_checks(tmp_path: Path) -> None:
    """Verify resolve_secure_path rejects system paths, traversals, and sensitive files."""
    # Disallowed system directories
    for bad in [
        "/etc/shadow",
        "/root/secrets.txt",
        "/proc/cpuinfo",
        "/sys/kernel",
        "/var/log/syslog",
    ]:
        with pytest.raises(HTTPException) as exc:
            resolve_secure_path(Path(bad), must_exist=False)
        assert exc.value.status_code == 403
        assert "restricted system directory" in exc.value.detail

        # Sensitive filenames / hidden files
    for sensitive in [".ssh/id_rsa", ".env", ".aws/credentials", ".bashrc", ".git/config"]:
        with pytest.raises(HTTPException) as exc:
            resolve_secure_path(tmp_path / sensitive, must_exist=False)
        assert exc.value.status_code == 403
        assert "sensitive configuration directory or file" in exc.value.detail

        # Directory traversal
    with pytest.raises(HTTPException) as exc:
        resolve_secure_path("../traversal/file.epub", must_exist=False)
    assert exc.value.status_code == 403
    assert "Directory traversal" in exc.value.detail

    # Safe path succeeds when it is inside an allowed base
    safe = tmp_path / "valid_doc.epub"
    safe.touch()
    assert (
        resolve_secure_path(safe, must_exist=True, allowed_bases=[tmp_path.resolve()])
        == safe.resolve()
    )


def test_operator_whitelist_unblocks_paths_inside_a_system_prefix(tmp_path: Path) -> None:
    """Risk: the hard-coded system deny list was checked before containment and
    overrode an explicit whitelist, so ``UBT_ALLOWED_DIRS=/var/lib/ubt/books``
    was unusable with 403 "restricted system directory" — the whitelist the
    non-loopback startup guardrail demands could not be satisfied with the
    conventional ``/var/lib/ubt`` data location."""
    config = UBTConfig(allowed_dirs="/var/lib/ubt/books", db_dir=tmp_path / "ledgers")

    allowed = resolve_secure_path("/var/lib/ubt/books/book.pdf", must_exist=False, config=config)
    assert allowed.name == "book.pdf"

    # Containment stays the authority: everything outside the whitelist is still
    # refused, including the system dirs the deny list covers.
    for denied in ("/etc/passwd", "/root/.bashrc", str(tmp_path / "outside.md")):
        with pytest.raises(HTTPException) as exc:
            resolve_secure_path(denied, must_exist=False, config=config)
        assert exc.value.status_code == 403


def test_system_deny_list_still_applies_without_a_whitelist(tmp_path: Path) -> None:
    """Risk: lifting the deny list for whitelisted paths must not weaken the
    default posture — with no ``UBT_ALLOWED_DIRS`` the system directories stay
    the fallback sandbox (fail closed)."""
    config = UBTConfig(db_dir=tmp_path / "ledgers")
    for denied in ("/etc/shadow", "/proc/cpuinfo", "/dev/null", "/usr/bin/env"):
        with pytest.raises(HTTPException) as exc:
            resolve_secure_path(denied, must_exist=False, config=config)
        assert exc.value.status_code == 403
        assert "restricted system directory" in exc.value.detail


def test_system_deny_prefixes_are_pre_resolved() -> None:
    """Risk (macOS): /etc, /var and /tmp are symlinks into /private, so a deny
    list built from unresolved paths never matched the resolved request path and
    the system-directory branch silently allowed those reads."""
    assert SYSTEM_DISALLOWED_PREFIXES
    assert all(prefix == prefix.resolve() for prefix in SYSTEM_DISALLOWED_PREFIXES)
    assert all(prefix.is_absolute() for prefix in SYSTEM_DISALLOWED_PREFIXES)


def test_sensitive_list_is_casefolded_and_complete(tmp_path: Path) -> None:
    """L1: the deny list must cover today's credential stores, case-insensitively.

    Two gaps: the comparison was case-sensitive (so ``.SSH``/``.ENV`` slipped
    through on case-insensitive filesystems), and the list predated
    ``.git-credentials``/``.gnupg``/``.password-store``/``.bash_history``/
    ``.env.local``/``secrets.env``/``credentials.json``.
    """
    from ubt.core.fs_perms import SENSITIVE_FILENAME_PARTS

    required = {
        ".git-credentials",
        ".gnupg",
        ".password-store",
        ".bash_history",
        ".env.local",
        "secrets.env",
        "credentials.json",
    }
    assert required <= set(SENSITIVE_FILENAME_PARTS), "review L1 additions went missing"

    probes = [
        ".SSH/id_rsa",
        ".GNUPG/pubring.kbx",
        ".password-store/.gpg-id",
        ".bash_history",
        ".env.local",
        "secrets.env",
        "credentials.json",
        "nested/.Git-Credentials",
    ]
    for name in probes:
        with pytest.raises(HTTPException) as exc:
            resolve_secure_path(tmp_path / name, must_exist=False)
        assert exc.value.status_code == 403, name
        assert "sensitive configuration directory or file" in exc.value.detail, name


def test_sensitive_deny_beats_an_explicit_whitelist(tmp_path: Path) -> None:
    """The precedence is deliberate, not a contradiction (review L1).

    ``resolve_secure_path`` docstring point 3 exempts a whitelisted path from
    the *system* deny list; point 4 says the *sensitive-name* deny list still
    wins. An allowlist is a scope decision ("these books"), not a licence to
    serve the ``credentials.json`` that happens to live beside them — and this
    route is reachable without credentials by default.
    """
    config = UBTConfig(allowed_dirs=str(tmp_path), db_dir=tmp_path / "ledgers")

    # The allowlist admits an ordinary file in the same directory...
    ok = tmp_path / "book.md"
    ok.write_text("# Title\n", encoding="utf-8")
    assert resolve_secure_path(ok, must_exist=True, config=config) == ok.resolve()

    # ...but not the credential file beside it, whitelist or no whitelist.
    secret = tmp_path / "credentials.json"
    secret.write_text("{}", encoding="utf-8")
    with pytest.raises(HTTPException) as exc:
        resolve_secure_path(secret, must_exist=True, config=config)
    assert exc.value.status_code == 403
    assert "sensitive configuration directory or file" in exc.value.detail
