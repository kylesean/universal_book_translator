"""The validation-to-open TOCTOU guard for submitted input documents.

F3 regression: ``resolve_secure_path`` validates a submitted path at submit
time but the pipeline opens the stored path later; a local writer replacing
the validated file with a symlink in between used to be followed at open.
``open_verified`` (wired at the adapter factory) refuses that swap fail-closed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from ubt.core.exceptions import DocumentParseError
from ubt.core.fs_perms import VerifiedOpenError, open_verified

pytestmark = pytest.mark.fast


def test_a_plain_file_opens_and_reads(tmp_path: Path) -> None:
    doc = tmp_path / "book.md"
    doc.write_bytes(b"hello book")
    with open_verified(doc) as fh:
        assert fh.read() == b"hello book"


def test_a_final_component_symlink_is_refused(tmp_path: Path) -> None:
    real = tmp_path / "book.md"
    real.write_text("real", encoding="utf-8")
    link = tmp_path / "link.md"
    link.symlink_to(real)
    with pytest.raises(VerifiedOpenError, match="refusing to open"):
        open_verified(link)


def test_a_file_swapped_for_a_symlink_after_validation_is_refused(tmp_path: Path) -> None:
    """The F3 scenario itself: validated file -> local writer swaps a symlink in."""
    doc = tmp_path / "book.md"
    doc.write_text("real book", encoding="utf-8")
    # The "validation" would have happened here.
    stolen = tmp_path / "stolen.txt"
    stolen.write_text("attacker-chosen target", encoding="utf-8")
    doc.unlink()
    doc.symlink_to(stolen)
    with pytest.raises(VerifiedOpenError, match="symlink"):
        open_verified(doc)


def test_a_directory_component_symlink_outside_the_roots_is_refused(tmp_path: Path) -> None:
    """The final component is honest; a swapped parent steers the read outside."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("top secret", encoding="utf-8")
    link_dir = sandbox / "alias"
    link_dir.symlink_to(outside, target_is_directory=True)
    candidate = link_dir / "secret.txt"
    with pytest.raises(VerifiedOpenError, match="sandbox"):
        open_verified(candidate, [sandbox])


def test_without_roots_the_containment_check_is_skipped(tmp_path: Path) -> None:
    """Callers that do not know a sandbox still get the anti-symlink verification."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("top secret", encoding="utf-8")
    link_dir = sandbox / "alias"
    link_dir.symlink_to(outside, target_is_directory=True)
    # No allowed_roots: only the final-component/symlink contract applies, so
    # this opens (the caller opted out of containment explicitly).
    with open_verified(link_dir / "secret.txt") as fh:
        assert fh.read() == b"top secret"


@pytest.mark.skipif(sys.platform == "win32", reason="os.mkfifo is POSIX-only")
def test_a_swapped_fifo_is_refused_not_hung(tmp_path: Path) -> None:
    fifo = tmp_path / "book.md"
    os.mkfifo(fifo)
    with pytest.raises(VerifiedOpenError, match="regular file"):
        open_verified(fifo)


def test_a_directory_is_refused(tmp_path: Path) -> None:
    with pytest.raises(VerifiedOpenError, match="regular file"):
        open_verified(tmp_path)


def test_a_missing_file_is_refused_with_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(VerifiedOpenError, match="ghost.md"):
        open_verified(tmp_path / "ghost.md")


def test_the_adapter_factory_refuses_a_swapped_input(tmp_path: Path) -> None:
    """The wired gate: submit-validate, swap, then resolve the adapter."""
    from ubt.adapters.factory import get_adapter_for_path
    from ubt.api.security import resolve_secure_path
    from ubt.core.config import UBTConfig

    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    doc = sandbox / "book.md"
    doc.write_text("# Chapter 1\n\nHello.\n", encoding="utf-8")
    config = UBTConfig(db_dir=tmp_path / "db", allowed_dirs=str(sandbox))
    resolved = resolve_secure_path(doc, config=config)
    assert resolved == doc.resolve()

    # The local writer swaps the validated file for a symlink while the job
    # waits; the factory must refuse at the moment the pipeline would open it.
    outside = tmp_path / "outside.md"
    outside.write_text("attacker content", encoding="utf-8")
    doc.unlink()
    doc.symlink_to(outside)
    with pytest.raises(DocumentParseError, match="symlink"):
        get_adapter_for_path(resolved, pdf_engine="auto")


def test_the_adapter_factory_still_resolves_a_real_file(tmp_path: Path) -> None:
    from ubt.adapters.codec_adapter import DocumentCodecAdapter
    from ubt.adapters.factory import get_adapter_for_path

    doc = tmp_path / "book.md"
    doc.write_text("# Chapter 1\n\nHello.\n", encoding="utf-8")
    adapter = get_adapter_for_path(doc, pdf_engine="auto")
    assert isinstance(adapter, DocumentCodecAdapter)
