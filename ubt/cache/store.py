"""Content-addressed step cache (ADR-0001 Phase 4).

A step's key is a hash of its kind, its inputs and its params: equal keys mean
the compute is skipped and the stored value returned, so a resumed or re-run job
recomputes only what actually changed. The ADR's guardrail is explicit -- this
wraps expensive *pure* steps and nothing else. There is no scheduler, no
dependency graph and no event log (ADR-0001 §8.2).

Values are text (JSON written by the caller), never pickles: a cache file stays
inspectable, and a corrupt one is a *miss*, not an executable payload. The store
is fail-open on IO -- a cache must never break a render -- so an unreadable or
unwritable entry simply computes.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)


def step_key(kind: str, inputs: Sequence[str], params: Mapping[str, object]) -> str:
    """A stable content key for one step invocation.

    ``inputs`` are the step's identity-bearing values (in order); ``params`` are
    the knobs that change the result without changing the inputs (a dpi, a
    compiler binary). Both are folded into one SHA-256 so a key is a fixed-size
    filename and equal keys mean equal work.
    """
    payload = json.dumps(
        {"k": kind, "i": list(inputs), "p": dict(params)},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CacheStore(Protocol):
    """A content-addressed store of text values."""

    def get_or_compute(self, key: str, compute: Callable[[], str]) -> str:
        """The stored value for ``key``, or ``compute()`` once and store it."""
        ...


class DiskCacheStore:
    """One JSON-text file per key, two levels deep under ``root``.

    The two-character shard directory keeps a large cache from putting thousands
    of entries in one directory; the temp-file + rename write makes a concurrent
    reader see either the old value or the new one, never a partial file.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get_or_compute(self, key: str, compute: Callable[[], str]) -> str:
        path = self._path(key)
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            pass  # absent or unreadable is a miss, never an error
        value = compute()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.parent / f"{path.name}.tmp"
            tmp.write_text(value, encoding="utf-8")
            tmp.replace(path)
        except OSError as exc:
            logger.debug("cache write failed for %s: %s", key, exc)
        return value


__all__ = ["CacheStore", "DiskCacheStore", "step_key"]
