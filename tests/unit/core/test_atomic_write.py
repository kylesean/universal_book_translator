from __future__ import annotations

from pathlib import Path

import pytest

from ubt.core.atomic import (
    atomic_write_path,
    atomic_write_text,
)

pytestmark = pytest.mark.fast


def test_atomic_write_text_success(tmp_path: Path) -> None:
    dest = tmp_path / "subdir" / "test.txt"
    atomic_write_text(dest, "hello world 123", encoding="utf-8")

    assert dest.exists()
    assert dest.read_text(encoding="utf-8") == "hello world 123"
    assert list(tmp_path.glob(".*.tmp")) == []


def test_atomic_write_cleans_up_on_failure(tmp_path: Path) -> None:
    dest = tmp_path / "will_fail.txt"
    dest.write_text("existing content", encoding="utf-8")

    with pytest.raises(RuntimeError, match="simulated failure"), atomic_write_path(dest) as tmp:
        tmp.write_text("corrupted content", encoding="utf-8")
        raise RuntimeError("simulated failure")

    assert dest.read_text(encoding="utf-8") == "existing content"
    assert list(tmp_path.glob(".*.tmp")) == []
