"""``allowed_base_dirs`` separator parsing.

The allowlist is the path sandbox for the REST and MCP servers, so how the
``allowed_dirs`` string splits into bases is security-relevant: a wrong split
either widens or narrows the sandbox. It must accept comma/semicolon and the
platform path separator, while never splitting a Windows drive letter's colon.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ubt.core.config import UBTConfig

pytestmark = pytest.mark.fast


def test_splits_on_comma_and_semicolon(tmp_path: Path) -> None:
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    cfg = UBTConfig(allowed_dirs=f"{a},{b};{c}")
    assert cfg.allowed_base_dirs() == [a.resolve(), b.resolve(), c.resolve()]


def test_windows_pathsep_does_not_split_drive_letters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # On Windows os.pathsep is ';', already covered by the comma/semicolon set;
    # the ':' in 'C:\Books' must NOT become a split point.
    monkeypatch.setattr(os, "pathsep", ";")
    cfg = UBTConfig(allowed_dirs=r"C:\Books;D:\Papers")
    assert len(cfg.allowed_base_dirs()) == 2


@pytest.mark.skipif(os.pathsep != ":", reason="POSIX pathsep only")
def test_posix_pathsep_splits_colon(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    cfg = UBTConfig(allowed_dirs=f"{a}:{b}")
    assert cfg.allowed_base_dirs() == [a.resolve(), b.resolve()]
