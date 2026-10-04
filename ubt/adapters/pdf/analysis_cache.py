"""Content-addressed caching for the analyze step (content-addressed cache layer).

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

import importlib
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
    """Size+mtime of the reader's rule modules, so a rule change invalidates the cache.

    The reader imports its list/heading/chrome rules from ``structure`` and its
    text canonicalization from ``normalize``; keying only on ``reader_pdf``
    would serve stale blocks after a change to either.
    """
    return _module_identity(
        "ubt.analyze.reader_pdf",
        "ubt.analyze.structure",
        "ubt.analyze.normalize",
    )


def _module_identity(*modules: str) -> str:
    """Size+mtime of each named module, so a rule change invalidates the cache."""
    parts: list[str] = []
    for name in modules:
        try:
            module_file = getattr(importlib.import_module(name), "__file__", "")
        except ImportError:
            module_file = ""
        parts.append(_file_identity(Path(module_file)) if module_file else name)
    return "|".join(parts)


def docling_identity() -> str:
    """The Docling parser's own identity (its two rule modules)."""
    return _module_identity("ubt.adapters.pdf.docling_parser", "ubt.adapters.pdf.docling_blocks")


def cached_blocks(
    store: CacheStore | None,
    *,
    path: Path,
    page_range: tuple[int, int] | None,
    compute: Callable[[], list[IRBlock]],
    identity: str | None = None,
    extra: str = "",
) -> list[IRBlock]:
    """``compute()`` through ``store`` (or straight through when ``store`` is None).

    ``identity`` names the extractor whose code produced the blocks (defaults to
    the native PDF reader); ``extra`` folds any remaining output-bearing knob
    (e.g. Docling's enrichment policy) into the key, so two runs that differ on
    it do not share an entry.
    """
    if store is None:
        return compute()
    key = step_key(
        "analyze",
        [
            _ANALYZE_SCHEMA,
            identity or _reader_identity(),
            _file_identity(path),
            repr(page_range),
            extra,
        ],
        {},
    )
    try:
        raw = store.get_or_compute(key, lambda: _BLOCKS.dump_json(compute()).decode("utf-8"))
        return _BLOCKS.validate_json(raw)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.debug("analyze cache entry unusable for %s: %s", key, exc)
        return compute()


__all__ = ["cached_blocks", "docling_identity"]
