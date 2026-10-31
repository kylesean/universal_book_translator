"""ensure_private_dir: tighten only a directory UBT itself creates."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from ubt.core.fs_perms import ensure_private_dir, restrict_dir_to_owner
from ubt.core.policy.layout_policy import BOOK_TEXT_DIR_MODE

pytestmark = pytest.mark.fast


def test_ensure_private_dir_restricts_a_new_directory(tmp_path: Path) -> None:
    target = tmp_path / "new" / "nested"
    ensure_private_dir(target)
    assert target.is_dir()
    assert stat.S_IMODE(target.stat().st_mode) == BOOK_TEXT_DIR_MODE


def test_ensure_private_dir_leaves_an_existing_directory_mode_alone(tmp_path: Path) -> None:
    # A user-chosen db_dir that already exists (a shared workspace) must not be
    # silently rewritten to owner-only.
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o755)

    ensure_private_dir(shared)

    assert stat.S_IMODE(shared.stat().st_mode) == 0o755


def test_restrict_dir_to_owner_always_chmods(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o755)
    restrict_dir_to_owner(shared)
    assert stat.S_IMODE(shared.stat().st_mode) == BOOK_TEXT_DIR_MODE


def test_sanitize_display_path() -> None:
    from ubt.core.fs_perms import sanitize_display_path

    assert sanitize_display_path(None) is None
    home = Path.home().resolve()
    assert sanitize_display_path(home) == "~"
    assert sanitize_display_path(home / "books" / "sample.pdf") == "~/books/sample.pdf"
    assert sanitize_display_path("/tmp/outside.txt") == "/tmp/outside.txt"


def test_world_readable_scan_skips_non_traversable_subtrees(tmp_path: Path) -> None:
    """Exposure needs traversal: a 0700 build dir walls off its contents.

    UBT's build directories are born owner-only precisely so the manuscripts
    inside stay private; the scan must not list files it cannot be opened for,
    or every private build dir would read as a leak.
    """
    from ubt.core.fs_perms import world_readable_files

    private = restrict_dir_to_owner(tmp_path / "build")
    (private / "manuscript.txt").write_text("全书译文", encoding="utf-8")
    (private / "manuscript.txt").chmod(0o644)  # mode says readable; the dir says no

    shared = tmp_path / "shared"
    shared.mkdir(mode=0o755)
    (shared / "leaked.txt").write_text("泄露的文本", encoding="utf-8")
    (shared / "leaked.txt").chmod(0o644)

    found = world_readable_files(tmp_path)

    assert shared / "leaked.txt" in found
    assert private / "manuscript.txt" not in found
