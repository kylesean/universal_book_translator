"""4-Layer Consistency Defense and Translation Memory Subsystem."""

from ubt.core.memory.bible import (
    BibleEntry,
    BookBible,
    clean_bible_entry,
    merge_bible_entries,
)
from ubt.core.memory.cjk_matcher import (
    contains_cjk,
    count_term_in_text,
    format_terms_markdown_table,
    select_terms_for_chunk,
    term_appears_in_text,
)
from ubt.core.memory.neighbor_window import NeighborContextBuilder

__all__ = [
    "BibleEntry",
    "BookBible",
    "NeighborContextBuilder",
    "clean_bible_entry",
    "contains_cjk",
    "count_term_in_text",
    "format_terms_markdown_table",
    "merge_bible_entries",
    "select_terms_for_chunk",
    "term_appears_in_text",
]
