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
    *,
    include_scoped: bool = True,
) -> str:
    """Build the book-level decision sheet.

    ``max_entries`` is a *ranking* cap, not a truncate-what-happens-to-be-first
    one: persons and places lead, then attestation frequency, so a sheet that
    has to shrink loses the terms a reader is least likely to meet.

    ``include_scoped=False`` is the prompt-facing mode. A stage that hands this
    sheet to every block has no single chapter context; including scoped rows
    then puts both ``chapter:2`` and ``chapter:5`` renderings of one source in
    the same system prompt. Prompt callers pass ``False`` and let the per-block
    ``build_chunk_glossary_table(..., context=...)`` carry the scoped entries.
    The default stays ``True`` for the TM fingerprint: a change to any scoped
    entry must still invalidate cached translations, so that hash has to see
    the whole glossary.
    """
    if context is not None:
        from ubt.core.memory.scoped_glossary import resolve_effective_glossary

        effective_dicts = resolve_effective_glossary(glossary_dicts, context)
    elif include_scoped:
        effective_dicts = glossary_dicts
    else:
        from ubt.core.memory.scoped_glossary import (
            ScopeLevel,
            parse_scope_spec,
            resolve_effective_glossary,
        )

        global_entries = [
            entry
            for entry in glossary_dicts
            if parse_scope_spec(str(entry.get("scope") or ""))[0] is ScopeLevel.GLOBAL
        ]
        # Resolve duplicates among the global rows too: two global renderings
        # for one source are a genuine conflict, but only one can be canonical.
        effective_dicts = resolve_effective_glossary(global_entries)

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


def merge_glossary_tables(tables: list[str]) -> str:
    """Merge rendered terminology tables without duplicating term rows.

    ``dict.fromkeys`` over whole table strings only deduplicates *identical*
    tables. Two blocks in the same macro chunk almost always retrieve a
    different local set (and a different order), so their shared global rows
    were emitted once per block. This merges at row granularity instead:
    the first occurrence of a source term (or acronym) wins, order preserved.
    """
    sections: list[tuple[str, list[str]]] = []
    section_index: dict[str, int] = {}
    seen: dict[str, set[str]] = {}
    current: str | None = None
    for table in tables:
        for raw in (table or "").splitlines():
            line = raw.strip()
            if not line or not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            first = cells[0] if cells else ""
            if first in ("原文", "缩写"):
                if first not in section_index:
                    section_index[first] = len(sections)
                    sections.append((first, []))
                    seen[first] = set()
                current = first
                sections[section_index[first]][1].append(line)
                continue
            if all(ch in "|:- " for ch in line):
                # delimiter row
                if current is not None:
                    sections[section_index[current]][1].append(line)
                continue
            if current is None or not first:
                continue
            if first in seen[current]:
                continue
            seen[current].add(first)
            sections[section_index[current]][1].append(line)
    return "\n\n".join("\n".join(rows) for _, rows in sections).strip()


def build_chunk_glossary_table(
    glossary_dicts: list[dict[str, Any]],
    abbreviation_entries: list[dict[str, Any]],
    source_text: str,
    context: ScopeContext | None = None,
    *,
    include_global_topup: bool = True,
) -> str:
    """The retrieval table for ``source_text``: terms and abbreviations appearing in it.

    "Chunk" is the scope the table is designed for, and every caller passes the
    text of the unit it is about to draft, repair or escalate — the draft,
    repair, triage and consistency stages currently hand in one *block's*
    source text each. Either way the table and the fingerprint are derived from
    the same argument, which is the property that has to hold.
    """
    if context is not None:
        from ubt.core.memory.scoped_glossary import resolve_effective_glossary

        effective_dicts = resolve_effective_glossary(glossary_dicts, context)
    else:
        effective_dicts = glossary_dicts

    relevant_terms = select_terms_for_chunk(
        effective_dicts,
        source_text,
        top_n=20 if include_global_topup else 0,
    )
    table = format_terms_markdown_table(relevant_terms)
    local_abbrevs = [e for e in abbreviation_entries if term_appears_in_text(e, source_text)]
    abbrev_block = format_abbreviations_markdown_table(local_abbrevs)
    if abbrev_block:
        table = f"{table}\n\n{abbrev_block}".strip()
    return table
