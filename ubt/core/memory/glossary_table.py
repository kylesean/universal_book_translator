"""Glossary markdown tables: the per-chunk retrieval table and the capped
book-level decision sheet.

Draft, repair, and triage each built the chunk table independently; the copies
drifted (term selection and abbreviation merging were re-implemented three
times). All stages now call these helpers, and so does the TM context hash —
the sheet the prompt carries and the sheet the fingerprint records must be the
same bytes, or a glossary change fails to invalidate cached translations.
"""

from typing import Any

from ubt.core.memory.abbreviation_miner import format_abbreviations_markdown_table
from ubt.core.memory.cjk_matcher import (
    format_terms_markdown_table,
    select_terms_for_chunk,
    term_appears_in_text,
)


def build_global_glossary_table(
    glossary_dicts: list[dict[str, Any]],
    max_entries: int,
) -> str:
    """Build the book-level decision sheet, capped to ``max_entries``.

    The per-chunk table above is the retrieval path; this one is the part that
    cannot be retrieved — a name the segment mentions only by an alias, a term
    the editor decided against the obvious rendering. Sending *all* of it on
    every block made the dictionary the dominant cost of a book run (600 rows is
    roughly 20k characters riding along with every one of a few thousand blocks),
    so it is truncated by rank and ``max_entries`` (0 disables the sheet and
    leaves terminology to retrieval plus the export-time enforcer).

    Rank: proper names first (alias-only mentions defeat retrieval), then
    attest frequency — which also puts a user-supplied glossary on top, since
    ``load_external_glossary`` seeds those entries with a frequency no mined
    term reaches. Source text breaks ties so the same run always sends the same
    sheet (the TM context hash is derived from it).
    """
    if max_entries <= 0 or not glossary_dicts:
        return ""
    ranked = sorted(
        (t for t in glossary_dicts if t.get("source")),
        key=lambda t: (
            0 if str(t.get("kind", "term")) in ("person", "place") else 1,
            -int(t.get("frequency", 0) or 0),
            str(t.get("source", "")),
        ),
    )
    return format_terms_markdown_table(ranked[:max_entries])


def build_chunk_glossary_table(
    glossary_dicts: list[dict[str, Any]],
    abbreviation_entries: list[dict[str, Any]],
    source_text: str,
) -> str:
    """Build the per-block glossary markdown table with local abbreviations."""
    relevant_terms = select_terms_for_chunk(glossary_dicts, source_text)
    table = format_terms_markdown_table(relevant_terms)
    local_abbrevs = [e for e in abbreviation_entries if term_appears_in_text(e, source_text)]
    abbrev_block = format_abbreviations_markdown_table(local_abbrevs)
    if abbrev_block:
        table = f"{table}\n\n{abbrev_block}".strip()
    return table
