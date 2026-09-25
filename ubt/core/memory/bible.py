from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ubt.core.memory.cjk_matcher import contains_cjk


class BibleEntry(BaseModel):
    """Single proper name or terminology mapping in the Translation Bible."""

    model_config = ConfigDict(extra="ignore")

    source: str
    translation: str
    aliases: list[str] = Field(default_factory=list)
    kind: str = "term"  # person, place, term
    frequency: int = 0  # Global attest count driving top-N ranking


class BookBible(BaseModel):
    """Whole-book translation bible containing characters, terms, tone, and system prompt."""

    model_config = ConfigDict(extra="ignore")

    doc_id: str
    language: str = "zh"
    tone: str = ""
    glossary: list[BibleEntry] = Field(default_factory=list)
    system_instruction: str = ""


def clean_bible_entry(
    source: str,
    translation: str,
    aliases: list[str] | None = None,
    kind: str = "term",
) -> BibleEntry | None:
    """Clean extracted entry using 0-token deterministic guardrails."""
    src = source.strip()
    # Strip CJK book quotes (simplified + Japanese style), angle brackets, and quotes
    tr = translation.strip().strip("《》〈〉「」『』\"'“”")

    if not src or not tr:
        return None

    # Guardrail: overly long items are likely slogans/captions, drop them.
    # Whitespace tokenization only works for alphabetic source scripts; CJK
    # sources are length-capped in characters instead.
    if contains_cjk(src):
        if len(src) > 12:
            return None
    elif len(src.split()) > 4:
        return None

    cleaned_aliases: list[str] = []
    if aliases:
        for a in aliases:
            cleaned_a = a.strip()
            if cleaned_a and cleaned_a.lower() != src.lower() and cleaned_a not in cleaned_aliases:
                cleaned_aliases.append(cleaned_a)

    return BibleEntry(source=src, translation=tr, aliases=cleaned_aliases, kind=kind)


def merge_bible_entries(entries: list[BibleEntry]) -> list[BibleEntry]:
    """Merge entries by source.lower(): first translation wins, aliases unioned, max frequency."""
    merged: dict[str, dict[str, Any]] = {}
    for e in entries:
        key = e.source.lower()
        if key not in merged:
            merged[key] = {
                "source": e.source,
                "translation": e.translation,
                "aliases": set(e.aliases),
                "kind": e.kind,
                "frequency": e.frequency,
            }
        else:
            merged[key]["aliases"] |= set(e.aliases)
            if not merged[key]["translation"] and e.translation:
                merged[key]["translation"] = e.translation
            merged[key]["frequency"] = max(int(merged[key]["frequency"]), e.frequency)

    result: list[BibleEntry] = []
    for key in sorted(merged):
        m = merged[key]
        result.append(
            BibleEntry(
                source=m["source"],
                translation=m["translation"],
                aliases=sorted(m["aliases"]),
                kind=m["kind"],
                frequency=int(m["frequency"]),
            )
        )
    return result
