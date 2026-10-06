"""Consistency defenses and Translation Memory subsystem.

The mined book bible and the structural + QE quality gate run on every job. The
hierarchical context memory below (L1 neighbours / L2 chapter snapshots / L3
epoch summaries) only runs on the multi-chapter long route: a single-chapter
document, anything over 40 chapters, and academic profiles get no rolling
summary (``ubt.core.engine.stages.draft.resolve_draft_policy``). The export-time
deterministic glossary enforcement is short-document only; see
``ubt.core.policy.adaptive_policy``, which keys it off ``route.mode == "short"``.
"""

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
