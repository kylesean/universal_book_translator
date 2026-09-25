"""Fingerprint cache: one read per unchanged file, a fresh digest after an edit."""

from __future__ import annotations

from pathlib import Path

import pytest

import ubt.core.ir.serializer as serializer
from ubt.core.ir.serializer import compute_file_sha256_cached

pytestmark = pytest.mark.fast


def test_fingerprint_is_read_once_until_the_file_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same PDF is fingerprinted at two layers (manifest doc_id + Docling
    cache key); re-reading it per layer is the waste this cache removes. The
    check counts real reads, so a cache that never misses and one that never
    hits both fail: unchanged file -> one read, edited file -> one more.
    """
    target = tmp_path / "book.pdf"
    target.write_bytes(b"first revision")
    real = serializer.compute_file_sha256
    calls = 0

    def counting(path: Path, chunk_size: int = 65536) -> str:
        nonlocal calls
        calls += 1
        return real(path, chunk_size)

    monkeypatch.setattr(serializer, "compute_file_sha256", counting)
    serializer._sha256_for_stat.cache_clear()

    first = compute_file_sha256_cached(target)
    assert first == real(target)
    assert compute_file_sha256_cached(target) == first
    assert calls == 1, "an unchanged file must be fingerprinted exactly once"

    target.write_bytes(b"second revision, different bytes")  # size and mtime move
    second = compute_file_sha256_cached(target)
    assert second != first
    assert calls == 2, "an edited file must be re-hashed, never served from a stale digest"
