"""``fs_perms``: owner-only modes for the files that carry manuscript text.

Covers the ``extra_dirs`` extension of the world-readable scan (the pipeline
only ever walked ``db_dir``, so exported PDFs and quality reports were never
reported) and the sensitive-path deny-list the API sandbox relies on.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

from ubt.api.app import SYSTEM_DISALLOWED_PREFIXES, resolve_secure_path
from ubt.core.config import UBTConfig
from ubt.core.fs_perms import (
    is_sensitive_path_part,
    world_readable_files,
)


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
    # ...and the extension reaches the artifact trees db_dir never covered.
    assert sorted(world_readable_files(ledger_dir, extra_dirs=[export_dir])) == sorted(
        [exposed_ledger, exposed_pdf]
    )
    # Overlapping roots are reported once, not twice.
    assert world_readable_files(ledger_dir, extra_dirs=[ledger_dir]) == [exposed_ledger]
    # Missing roots are skipped, not fatal.
    assert world_readable_files(tmp_path / "never-created") == []


def test_sensitive_path_parts_are_casefolded() -> None:
    """The deny list is matched casefolded — ``.SSH`` is ``.ssh`` on a
    case-insensitive filesystem, and the old ``in`` check allowed the read."""
    for name in (".ssh", ".SSH", ".Env", "CREDENTIALS.JSON", ".Git-Credentials"):
        assert is_sensitive_path_part(name), name
    assert not is_sensitive_path_part("book.pdf")
    assert not is_sensitive_path_part("credentials.json.bak")


def test_per_environment_dotenv_variants_are_sensitive() -> None:
    """``.env.production`` / ``.envrc`` hold the same secrets as ``.env``."""
    for name in (".env.production", ".env.development", ".envrc", ".env.example"):
        assert is_sensitive_path_part(name), name


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


def test_allowed_bases_are_resolved_before_containment(tmp_path: Path) -> None:
    """An unresolved allowlist entry must still contain its own files.

    Containment compares the *resolved* candidate against each base. A base
    handed in unnormalized (``sub/..``, a symlinked directory) used to be
    compared verbatim, so it never matched and every path inside it 403'd.
    """
    (tmp_path / "sub").mkdir()
    target = tmp_path / "book.md"
    target.touch()
    base = tmp_path / "sub" / ".."  # resolves to tmp_path, but is not normalized as-is

    resolved = resolve_secure_path(target, must_exist=True, allowed_bases=[base])
    assert resolved == target.resolve()


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
    """The deny list must cover today's credential stores, case-insensitively.

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
    assert required <= set(SENSITIVE_FILENAME_PARTS), "sensitive-name additions went missing"

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
    """The precedence is deliberate, not a contradiction.

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


def test_unreadable_subtree_is_surfaced_not_silently_skipped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A directory the scan cannot enter must warn, not read as all-clear."""
    import logging
    import os

    if os.geteuid() == 0:
        pytest.skip("root bypasses directory permissions")
    root = tmp_path / "root"
    blocked = root / "blocked"
    blocked.mkdir(parents=True)
    (blocked / "hidden.txt").write_text("manuscript", encoding="utf-8")
    (blocked / "hidden.txt").chmod(0o644)
    visible = root / "visible.txt"
    visible.write_text("manuscript", encoding="utf-8")
    visible.chmod(0o644)
    blocked.chmod(0o000)
    try:
        with caplog.at_level(logging.WARNING):
            exposed = world_readable_files(root)
        assert visible in exposed
        assert any("could not enter" in rec.message for rec in caplog.records)
    finally:
        blocked.chmod(0o755)


def test_system_disallowed_prefixes_cover_etc() -> None:
    from ubt.core.fs_perms import SYSTEM_DISALLOWED_PREFIXES

    assert Path("/etc").resolve() in SYSTEM_DISALLOWED_PREFIXES
