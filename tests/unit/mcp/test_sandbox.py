from __future__ import annotations

from pathlib import Path

import pytest

from ubt.core.exceptions import UBTError
from ubt.mcp.server import _resolve_input, _safe_output_path, _sandbox_path

pytestmark = pytest.mark.fast


@pytest.fixture()
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
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
    target = sandbox / "workdir" / "book.md"
    target.write_text("hello")
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
    assert str(outside) in str(exc.value)


def test_system_directories_are_refused_without_an_explicit_allowlist(
    sandbox: Path,
) -> None:
    with pytest.raises(UBTError, match="restricted system directory"):
        _sandbox_path("/etc/passwd", must_exist=False)


def test_an_explicit_allowlist_can_widen_into_a_system_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    resolved = _safe_output_path(str(sandbox / "workdir" / "missing.md"))
    assert resolved == (sandbox / "workdir" / "missing.md").resolve()


__all__ = ["pytest"]


async def test_job_status_disk_fallback_reads_a_real_ledger(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    assert status["status"] in {"initialized", "pending", "running"}
    assert "delivery_verdict" in status
    assert "needs_human" in status
    assert "blocked_human" in status

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
