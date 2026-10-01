"""Content-addressed caching for the analyze step (ADR-0001 Phase 4).

``read_pdf`` is a pure transform of a PDF's bytes into a ``Document`` (and then
into ``IRBlock``s): the same file and page range always yield the same blocks.
It is also the first expensive step of every run, so a resumed or re-run job can
reuse the extraction instead of re-reading the whole geometry.

The key covers every input the extraction reads: the source file's identity, the
page range, and the *reader's own source identity* (its file size and mtime), so
changing a paragraph/heading/chrome rule invalidates old entries automatically
rather than caching a verdict the current reader would not make. A cache is
fail-open by contract: a miss computes, and an undecodable entry is a miss too --
the cache must never change what the pipeline sees.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from pydantic import TypeAdapter

from ubt.cache.store import CacheStore, step_key
from ubt.core.ir.models import IRBlock

logger = logging.getLogger(__name__)

#: Bumped alongside a format change in the cached payload.
_ANALYZE_SCHEMA = "1"

_BLOCKS = TypeAdapter(list[IRBlock])


def _file_identity(path: Path) -> str:
    try:
        stat = path.stat()
    except OSError:
        return str(path)
    return f"{path}:{stat.st_size}:{stat.st_mtime_ns}"


def _reader_identity() -> str:
    """Size+mtime of the reader module, so a rule change invalidates the cache."""
    from ubt.analyze import reader_pdf

    module_file = getattr(reader_pdf, "__file__", "")
    if not module_file:
        return "unknown"
    return _file_identity(Path(module_file))


def cached_blocks(
    store: CacheStore | None,
    *,
    path: Path,
    page_range: tuple[int, int] | None,
    compute: Callable[[], list[IRBlock]],
) -> list[IRBlock]:
    """``compute()`` through ``store`` (or straight through when ``store`` is None)."""
    if store is None:
        return compute()
    key = step_key(
        "analyze",
        [_ANALYZE_SCHEMA, _reader_identity(), _file_identity(path), repr(page_range)],
        {},
    )
    try:
        raw = store.get_or_compute(key, lambda: _BLOCKS.dump_json(compute()).decode("utf-8"))
        return _BLOCKS.validate_json(raw)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.debug("analyze cache entry unusable for %s: %s", key, exc)
        return compute()


__all__ = ["cached_blocks"]
