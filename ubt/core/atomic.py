"""Atomic file write operations with temporary files, fsync, and atomic replace.

Guarantees that final delivery artifacts (PDF, EPUB, HTML, reports, contracts)
are never observed partially written or corrupted following crashes, SIGKILL,
or disk write failures.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


@contextmanager
def atomic_write_path(dest: str | Path) -> Generator[Path, None, None]:
    """Provide a temporary file path in dest.parent that is atomically renamed on success.

    Guarantees:
    - Writes to a hidden temporary file in the same directory (same filesystem).
    - Flushes and fsyncs data to disk before renaming.
    - Atomically replaces dest via os.replace.
    - Cleans up the temporary file if an exception or cancellation occurs.
    """
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path_str = tempfile.mkstemp(
        dir=dest_path.parent, prefix=f".{dest_path.name}.", suffix=".tmp"
    )
    os.close(tmp_fd)
    tmp_path = Path(tmp_path_str)
    try:
        yield tmp_path
        if tmp_path.exists():
            with tmp_path.open("a+b") as f:
                f.flush()
                os.fsync(f.fileno())
            tmp_path.replace(dest_path)
    except BaseException:
        if tmp_path.exists():
            with contextlib.suppress(OSError):
                tmp_path.unlink()
        raise


def atomic_write_bytes(dest: str | Path, content: bytes) -> None:
    """Atomically write binary content to dest."""
    with atomic_write_path(dest) as tmp_path:
        tmp_path.write_bytes(content)


def atomic_write_text(dest: str | Path, content: str, encoding: str = "utf-8") -> None:
    """Atomically write text content to dest."""
    with atomic_write_path(dest) as tmp_path:
        tmp_path.write_text(content, encoding=encoding)


def atomic_save(dest: str | Path, save_fn: Callable[[Path], Any]) -> None:
    """Atomically run save_fn(tmp_path) and rename to dest."""
    with atomic_write_path(dest) as tmp_path:
        save_fn(tmp_path)


__all__ = [
    "atomic_save",
    "atomic_write_bytes",
    "atomic_write_path",
    "atomic_write_text",
]
