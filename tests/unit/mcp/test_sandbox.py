"""The MCP path sandbox: every tool path input passes ``_sandbox_path``.

The stdio MCP server trusts the local user but not the calling agent: a
prompt-injected agent can name any path it likes, so every path argument is
resolved through the same allowlist the REST API enforces — re-implemented
locally (no fastapi import), sharing the sensitive-name primitives. The guard
is the security boundary for the whole MCP surface, and until this module it
had zero tests: a relaxed comparison (``str.startswith``), a dropped
sensitive-part check, or a base-fallback change would all have shipped
silently.

The layers, each pinned here:

1. ``..`` anywhere in the path is refused before any resolution.
2. System directories (/etc, /proc, /sys, ...) are refused unless the
   operator explicitly set ``UBT_ALLOWED_DIRS``.
3. Paths outside the allowlist bases (``UBT_ALLOWED_DIRS``, else cwd +
   ``db_dir``) are refused, with the caller's own path echoed — not the
   server-side bases.
4. Sensitive names (.ssh, .env, credentials...) are refused *even inside an
   allowed base* — a server started in $HOME would otherwise expose them.
5. ``must_exist`` distinguishes input (must exist) from output (created later).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ubt.core.exceptions import UBTError
from ubt.mcp.server import _resolve_input, _safe_output_path, _sandbox_path

pytestmark = pytest.mark.fast


@pytest.fixture()
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A sandbox rooted at tmp_path, with a cwd also under tmp_path."""
    cwd = tmp_path / "workdir"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("UBT_DB_DIR", str(tmp_path / "ledgers"))
    (tmp_path / "ledgers").mkdir()
    monkeypatch.delenv("UBT_ALLOWED_DIRS", raising=False)
    return tmp_path


def test_an_empty_path_is_refused() -> None:
    with pytest.raises(UBTError, match="Empty path"):
        _sandbox_path("   ", must_exist=False)


@pytest.mark.parametrize("raw", ["../escape.md", "sub/../../escape.md", ".."])
def test_parent_traversal_is_refused(sandbox: Path, raw: str) -> None:
    with pytest.raises(UBTError, match=r"\.\."):
        _sandbox_path(raw, must_exist=False)


def test_a_traversal_that_resolves_inside_the_bases_is_still_refused(
    sandbox: Path,
) -> None:
    """The '..' veto is defense in depth, not redundant with the allowlist.

    ``book.md`` exists only as ``<workdir>/book.md``, so the segment loop the
    '..' check guards is reachable: a crafted path can traverse out *and back*
    into an allowed base while no resolved path ever leaves it.
    """
    target = sandbox / "workdir" / "book.md"
    target.write_text("hello")
    # Resolves to the allowed <workdir>/book.md -- only the '..' veto refuses it.
    with pytest.raises(UBTError, match=r"\.\."):
        _sandbox_path("../workdir/book.md", must_exist=True)


def test_a_path_inside_the_default_bases_is_accepted(sandbox: Path) -> None:
    source = sandbox / "workdir" / "book.md"
    source.write_text("hello")
    resolved = _resolve_input(str(source))
    assert resolved == source.resolve()


def test_a_relative_path_resolves_against_the_base(sandbox: Path) -> None:
    source = sandbox / "workdir" / "book.md"
    source.write_text("hello")
    assert _resolve_input("book.md") == source.resolve()


def test_a_path_outside_the_bases_is_refused_and_echoes_the_caller_path(
    sandbox: Path,
) -> None:
    outside = Path("/tmp") / "not-a-base" / "book.md"
    with pytest.raises(UBTError, match="outside the allowed directories") as exc:
        _sandbox_path(str(outside), must_exist=False)
    # The caller's own path is echoed; the server-side bases are not.
    assert str(outside) in str(exc.value)


def test_system_directories_are_refused_without_an_explicit_allowlist(
    sandbox: Path,
) -> None:
    with pytest.raises(UBTError, match="restricted system directory"):
        _sandbox_path("/etc/passwd", must_exist=False)


def test_an_explicit_allowlist_can_widen_into_a_system_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UBT_ALLOWED_DIRS is the operator's explicit widening: it lifts the
    system-prefix veto (an operator pointing the sandbox at /srv/data gets
    exactly that), but never the sensitive-name rule."""
    widened = tmp_path / "widened"
    widened.mkdir()
    monkeypatch.setenv("UBT_ALLOWED_DIRS", str(widened))
    target = widened / "book.md"
    target.write_text("hello")
    assert _resolve_input(str(target)) == target.resolve()


def test_sensitive_names_are_refused_even_inside_an_allowed_base(
    sandbox: Path,
) -> None:
    secret = sandbox / "workdir" / ".env"
    secret.write_text("KEY=1")
    with pytest.raises(UBTError, match="sensitive"):
        _resolve_input(str(secret))
    ssh_dir = sandbox / "workdir" / ".ssh" / "id_rsa"
    ssh_dir.parent.mkdir()
    ssh_dir.write_text("key")
    with pytest.raises(UBTError, match="sensitive"):
        _resolve_input(str(ssh_dir))


def test_must_exist_refuses_a_missing_input(sandbox: Path) -> None:
    with pytest.raises(UBTError, match="not found"):
        _resolve_input(str(sandbox / "workdir" / "missing.md"))
    # The same path is fine as an output (created by the pipeline later).
    resolved = _safe_output_path(str(sandbox / "workdir" / "missing.md"))
    assert resolved == (sandbox / "workdir" / "missing.md").resolve()


__all__ = ["pytest"]


# --------------------------------------------------------------------------- #
# The tools behind the sandbox: status fallback, cancel idempotence, inspect
# --------------------------------------------------------------------------- #


async def test_job_status_disk_fallback_reads_a_real_ledger(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The SQLite fallback answers from the ledger; unknown jobs say unknown."""
    from ubt.core.engine.ledger import SQLiteJobLedger
    from ubt.core.ir.models import BookManifest
    from ubt.mcp import server as mcp_server

    job_id = "status_fallback_job"
    db_path = sandbox / "ledgers" / f"{job_id}.sqlite"
    with SQLiteJobLedger(db_path) as ledger:
        ledger.init_job_from_manifest(
            job_id,
            BookManifest(
                doc_id=job_id,
                title="t",
                source_path=str(sandbox / "in.md"),
                source_lang="en",
                target_lang="zh",
            ),
        )

    status = await mcp_server.ubt_job_status(job_id=job_id, db_dir=str(sandbox / "ledgers"))
    assert status["job_id"] == job_id
    # A freshly initialized ledger reads as running/pending (never "completed"):
    # the fallback must not fabricate progress for a job that did none.
    assert status["status"] in {"initialized", "pending", "running"}
    assert "delivery_verdict" in status
    assert "needs_human" in status
    assert "blocked_human" in status

    # A live writer lock keeps a quiet ledger marked as RUNNING even if stale
    import os
    import time

    from ubt.core.engine.writer_lock import LedgerWriterLock

    old = time.time() - 1000.0
    os.utime(db_path, (old, old))
    lock = LedgerWriterLock(db_path, job_id)
    lock.acquire()
    try:
        live_status = await mcp_server.ubt_job_status(
            job_id=job_id, db_dir=str(sandbox / "ledgers")
        )
        assert live_status["status"] == "running"
    finally:
        lock.release()

    unknown = await mcp_server.ubt_job_status(
        job_id="never_seen_job", db_dir=str(sandbox / "ledgers")
    )
    assert unknown["status"] == "unknown"


async def test_job_status_db_dir_goes_through_the_sandbox(sandbox: Path) -> None:
    """A system directory as db_dir is refused before any ledger opens there.

    The tool decorator converts the domain error into ``ToolError`` for a clean
    MCP error response — the boundary the calling agent actually sees.
    """
    from mcp.server.mcpserver.exceptions import ToolError

    from ubt.mcp import server as mcp_server

    with pytest.raises(ToolError, match="restricted system directory"):
        await mcp_server.ubt_job_status(job_id="valid_job_id", db_dir="/etc")


async def test_inspect_book_reports_the_manifest(sandbox: Path) -> None:
    from ubt.mcp import server as mcp_server

    source = sandbox / "workdir" / "book.md"
    source.write_text("# Heading\n\nA paragraph of body text.\n")
    info = await mcp_server.ubt_inspect_book(input_path=str(source))
    assert info["title"], info
    assert info["doc_id"]
