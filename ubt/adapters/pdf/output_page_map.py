"""Recover each IR block's output page from a reflowed artifact's text layer.

A publication reflow rebuilds every page and re-flows blocks across them, so an
IR block's ``bbox.page`` (its *source* page) says nothing about where its text
landed -- while the visual gate reports findings against *output* pages. The
historical quarantine matched the two directly and flagged arbitrary blocks.

This module matches a block's rendered target text against the artifact's
per-page text and returns the page it appears on. Only blocks whose target text
is long enough to be unambiguous and that appear on exactly one page are mapped;
everything else is omitted so the caller keeps its own fallback.

Limitation: a paragraph split across a page boundary is not found on any single
page and is left unmapped rather than guessed at -- a missed quarantine is
recoverable, a wrong one silently ships a broken page under a human-review flag.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

#: Normalized target text shorter than this is too common to place safely
#: (e.g. "Yes.", a running head, a page number); matching it would quarantine
#: the wrong block, which is worse than not quarantining at all.
_MIN_MATCH_CHARS = 24

_WHITESPACE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Collapse whitespace so extracted line breaks do not break containment."""
    return _WHITESPACE.sub(" ", text).strip()


def map_blocks_to_output_pages(blocks: Sequence[object], pages: Sequence[str]) -> dict[str, int]:
    """Map block id -> 1-based output page its target text appears on.

    ``pages`` is per-page text in 1-based order (index 0 is page 1), as returned
    by the renderer's text extractor. A block is mapped only when its normalized
    target text is at least :data:`_MIN_MATCH_CHARS` characters long and occurs
    on exactly one page; ambiguous or short text is omitted so the caller can
    fall back rather than mis-attribute the finding.
    """
    normalized_pages = [_normalize(page) for page in pages]
    mapping: dict[str, int] = {}
    for block in blocks:
        target = getattr(block, "target_text", None)
        if not isinstance(target, str):
            continue
        needle = _normalize(target)
        if len(needle) < _MIN_MATCH_CHARS:
            continue
        hits = [index + 1 for index, page in enumerate(normalized_pages) if needle in page]
        if len(hits) != 1:
            continue
        block_id = str(getattr(block, "id", "") or "")
        if block_id:
            mapping[block_id] = hits[0]
    return mapping


__all__ = ["map_blocks_to_output_pages"]
