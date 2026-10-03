"""Content-addressed step cache (content-addressed cache layer).

A step's key is a hash of its kind, its inputs and its params: equal keys mean
the compute is skipped and the stored value returned, so a resumed or re-run job
recomputes only what actually changed. The architectural guardrail is explicit -- this
wraps expensive *pure* steps and nothing else. There is no scheduler, no
dependency graph and no event log (content-addressed cache store).

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
from uuid import uuid4

from ubt.core.fs_perms import restrict_dir_to_owner, restrict_file_to_owner

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

    def get(self, key: str) -> str | None:
        """The stored value for ``key``, or ``None`` on a miss."""
        ...

    def put(self, key: str, value: str) -> None:
        """Store ``value`` under ``key`` (fail-open)."""
        ...

    def get_or_compute(self, key: str, compute: Callable[[], str]) -> str:
        """The stored value for ``key``, or ``compute()`` once and store it."""
        ...


class DiskCacheStore:
    """One owner-restricted JSON-text file per key, two levels deep under ``root``.

    The two-character shard directory keeps a large cache from putting thousands
    of entries in one directory; the temp-file + rename write makes a concurrent
    reader see either the old value or the new one, never a partial file. Cache
    values are derived from (possibly sensitive) book text, so the shard
    directory and every file are limited to their owner.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> str | None:
        try:
            return self._path(key).read_text(encoding="utf-8")
        except OSError:
            return None  # absent or unreadable is a miss, never an error

    def put(self, key: str, value: str) -> None:
        path = self._path(key)
        try:
            restrict_dir_to_owner(path.parent)
            # Unique temp name: two processes writing the same key would
            # otherwise interleave on one fixed .tmp file and rename a torn
            # value into place.
            tmp = path.parent / f"{path.name}.{uuid4().hex}.tmp"
            tmp.write_text(value, encoding="utf-8")
            restrict_file_to_owner(tmp)
            tmp.replace(path)
        except OSError as exc:
            logger.debug("cache write failed for %s: %s", key, exc)

    def get_or_compute(self, key: str, compute: Callable[[], str]) -> str:
        cached = self.get(key)
        if cached is not None:
            return cached
        value = compute()
        self.put(key, value)
        return value


__all__ = ["CacheStore", "DiskCacheStore", "step_key"]
