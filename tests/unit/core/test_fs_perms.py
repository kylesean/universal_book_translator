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
