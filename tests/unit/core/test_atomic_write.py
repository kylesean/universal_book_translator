"""Unit tests for atomic write utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from ubt.core.atomic import (
    atomic_save,
    atomic_write_bytes,
    atomic_write_path,
    atomic_write_text,
)

pytestmark = pytest.mark.fast


def test_atomic_write_text_success(tmp_path: Path) -> None:
    dest = tmp_path / "subdir" / "test.txt"
    atomic_write_text(dest, "hello world 123", encoding="utf-8")

    assert dest.exists()
    assert dest.read_text(encoding="utf-8") == "hello world 123"
    # Ensure no lingering temp files
    assert list(tmp_path.glob(".*.tmp")) == []


def test_atomic_write_bytes_success(tmp_path: Path) -> None:
    dest = tmp_path / "test.bin"
    atomic_write_bytes(dest, b"\x00\x01\x02\xff")

    assert dest.exists()
    assert dest.read_bytes() == b"\x00\x01\x02\xff"
    assert list(tmp_path.glob(".*.tmp")) == []


def test_atomic_save_success(tmp_path: Path) -> None:
    dest = tmp_path / "test.custom"

    def custom_saver(target: Path) -> None:
        target.write_text("saved via custom callback", encoding="utf-8")

    atomic_save(dest, custom_saver)
    assert dest.exists()
    assert dest.read_text(encoding="utf-8") == "saved via custom callback"
    assert list(tmp_path.glob(".*.tmp")) == []


def test_atomic_write_cleans_up_on_failure(tmp_path: Path) -> None:
    dest = tmp_path / "will_fail.txt"
    dest.write_text("existing content", encoding="utf-8")

    with pytest.raises(RuntimeError, match="simulated failure"), atomic_write_path(dest) as tmp:
        tmp.write_text("corrupted content", encoding="utf-8")
        raise RuntimeError("simulated failure")

    # Destination retains old content
    assert dest.read_text(encoding="utf-8") == "existing content"
    # Temp file cleaned up
    assert list(tmp_path.glob(".*.tmp")) == []
