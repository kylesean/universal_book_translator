"""Shared source identity for the readers.

Every reader names its ``CanonicalSource.doc_id`` by the file's content, so the
same source is one document across runs and formats. This is the one place that
digest is computed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

#: Hex digits kept: long enough to collide only on a deliberate attack, short
#: enough to read in a ledger row.
_DIGEST_CHARS = 16


def file_digest(path: str | Path) -> str:
    """A stable content hash of a source file (first :data:`_DIGEST_CHARS` hex)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:_DIGEST_CHARS]


__all__ = ["file_digest"]
