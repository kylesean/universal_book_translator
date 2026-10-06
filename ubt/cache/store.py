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
import os
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from ubt.core.fs_perms import restrict_dir_to_owner, restrict_file_to_owner

logger = logging.getLogger(__name__)

#: Default ceiling on the number of entries one :class:`DiskCacheStore` keeps.
#: The draft cache writes one file per translated block and never deleted one,
#: so a machine that translated many books accumulated entries without bound.
DEFAULT_MAX_ENTRIES = 50_000

#: A full directory scan is O(entries); run it every this many writes instead of
#: on every ``put`` so the cap costs an amortized constant per write.
_PRUNE_INTERVAL_WRITES = 1024

#: Prune down to this share of the cap, so pruning is not re-triggered on the
#: very next write (a hysteresis band).
_PRUNE_TARGET_RATIO = 0.9

#: The only paths prune() may ever delete: this store's own layout
#: (``root/<2-hex shard>/<sha256 key>.json``). Everything else under the
#: operator-configurable root belongs to someone else.
_SHARD_DIR_RE = re.compile(r"[0-9a-f]{2}")
_ENTRY_FILE_RE = re.compile(r"[0-9a-f]{32,}\.json")


def _max_entries_from_env() -> int:
    raw = os.environ.get("UBT_CACHE_MAX_ENTRIES", "").strip()
    if not raw:
        return DEFAULT_MAX_ENTRIES
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_ENTRIES
    return value if value > 0 else DEFAULT_MAX_ENTRIES


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

    def __init__(self, root: str | Path, max_entries: int | None = None) -> None:
        self.root = Path(root)
        self.max_entries = max_entries if max_entries is not None else _max_entries_from_env()
        self._writes = 0
        # Prune on open as well as on the write interval: a shared cache dir
        # fed by many short-lived runs (each under the write interval) would
        # otherwise never see a prune at all.
        self.prune()

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
            return
        self._writes += 1
        if self._writes % _PRUNE_INTERVAL_WRITES == 0:
            self.prune()

    def prune(self) -> int:
        """Delete the oldest entries until at most ``max_entries`` remain.

        Best-effort and never raises: a concurrent reader that loses its file to
        a prune simply recomputes (a cache miss). Returns the number deleted.
        """
        if self.max_entries <= 0 or not self.root.is_dir():
            return 0
        entries: list[tuple[float, Path]] = []
        try:
            for dirpath, _dirnames, filenames in os.walk(self.root):
                # Only this store's own layout (root/<2 hex shard>/<sha256>.json)
                # is prunable: cache_dir is operator-configurable, and an
                # unrelated tool's .json files in a shared root must never be
                # evicted by a UBT cap.
                shard = Path(dirpath).name
                if shard == self.root.name or not _SHARD_DIR_RE.fullmatch(shard):
                    continue
                for name in filenames:
                    if not _ENTRY_FILE_RE.fullmatch(name):
                        continue
                    child = Path(dirpath) / name
                    try:
                        entries.append((child.stat().st_mtime, child))
                    except OSError:
                        continue
        except OSError as exc:
            logger.debug("cache prune scan failed under %s: %s", self.root, exc)
            return 0
        if len(entries) <= self.max_entries:
            return 0
        keep = int(self.max_entries * _PRUNE_TARGET_RATIO)
        entries.sort(key=lambda item: item[0])  # oldest first
        deleted = 0
        for _mtime, child in entries[: len(entries) - keep]:
            try:
                child.unlink()
                deleted += 1
            except OSError:
                continue
        if deleted:
            logger.debug(
                "cache prune removed %d entr(ies) under %s (cap %d)",
                deleted,
                self.root,
                self.max_entries,
            )
        return deleted

    def get_or_compute(self, key: str, compute: Callable[[], str]) -> str:
        cached = self.get(key)
        if cached is not None:
            return cached
        value = compute()
        self.put(key, value)
        return value


__all__ = ["CacheStore", "DiskCacheStore", "step_key"]
