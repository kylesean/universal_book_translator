"""Glossary markdown tables: the per-chunk retrieval table and the capped
book-level decision sheet.

Draft, repair, and triage each built the chunk table independently; the copies
drifted (term selection and abbreviation merging were re-implemented three
times). All stages now call these helpers, and so does the TM context hash —
the sheet the prompt carries and the sheet the fingerprint records must be the
same bytes, or a glossary change fails to invalidate cached translations.

The dict shape these helpers accept is not a second glossary schema: every
entry is a :class:`~ubt.core.memory.bible.BibleEntry` projection. Dicts are
born only in the bible stage's ``_split_bible_entries`` (fed by validated
models) and read back from the bible cache through ``BibleEntry.model_validate``
— external glossary files are parsed into ``BibleEntry`` at load — so the
model is the single schema and these ``dict`` parameters are its serialized
form.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ubt.core.memory.abbreviation_miner import format_abbreviations_markdown_table
from ubt.core.memory.cjk_matcher import (
    format_terms_markdown_table,
    select_terms_for_chunk,
    term_appears_in_text,
)

if TYPE_CHECKING:
    from ubt.core.memory.scoped_glossary import ScopeContext


def build_global_glossary_table(
    glossary_dicts: list[dict[str, Any]],
    max_entries: int,
    context: ScopeContext | None = None,
) -> str:
    """Build the book-level decision sheet, capped to ``max_entries``."""
    if context is not None:
        from ubt.core.memory.scoped_glossary import resolve_effective_glossary

        effective_dicts = resolve_effective_glossary(glossary_dicts, context)
    else:
        effective_dicts = glossary_dicts

    if max_entries <= 0 or not effective_dicts:
        return ""
    ranked = sorted(
        (t for t in effective_dicts if t.get("source")),
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
    context: ScopeContext | None = None,
) -> str:
    """Build the per-block glossary markdown table with local abbreviations."""
    if context is not None:
        from ubt.core.memory.scoped_glossary import resolve_effective_glossary

        effective_dicts = resolve_effective_glossary(glossary_dicts, context)
    else:
        effective_dicts = glossary_dicts

    relevant_terms = select_terms_for_chunk(effective_dicts, source_text)
    table = format_terms_markdown_table(relevant_terms)
    local_abbrevs = [e for e in abbreviation_entries if term_appears_in_text(e, source_text)]
    abbrev_block = format_abbreviations_markdown_table(local_abbrevs)
    if abbrev_block:
        table = f"{table}\n\n{abbrev_block}".strip()
    return table
