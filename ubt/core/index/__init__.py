"""Publication-grade bilingual concordance and index generator."""

from __future__ import annotations

from ubt.core.index.extractor import extract_index_from_blocks
from ubt.core.index.generator import (
    render_index_html,
    render_index_markdown,
    render_index_typst,
)
from ubt.core.index.models import ConcordanceIndex, IndexEntry, IndexEntryLocation
from ubt.core.index.phonetics import get_latin_initial, get_pinyin_initial

__all__ = [
    "ConcordanceIndex",
    "IndexEntry",
    "IndexEntryLocation",
    "extract_index_from_blocks",
    "get_latin_initial",
    "get_pinyin_initial",
    "render_index_html",
    "render_index_markdown",
    "render_index_typst",
]
