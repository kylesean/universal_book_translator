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


def normalize_glossary_spacing(translation: str, source: str = "") -> str:
    """Insert one half-width space at every CJK<->Latin/digit boundary of a rendering.

    The canonical form the glossary enforces must match the typography the
    pipeline itself ships: the export stage applies Pangu spacing
    (:func:`ubt.core.cleaners.cjk_spacing.apply_pangu_spacing`) to every
    target, so a rendering mined without the space ("切换Transformer") fights
    the document's own convention ("切换 Transformer") for the rest of the
    book — measured on the arXiv 2101.03961 run: the unspaced entry sent three
    blocks into repair loops for "drifting" away from a form the enforcer's
    own exporter would rewrite. Matching is space-tolerant both ways (see
    ``_term_pattern``), so normalizing here only aligns the *canonical
    surface* shown to models and written by the enforcer. Latin-Latin
    spacing ("New York") is never touched; spaces already present at a
    boundary are preserved, not doubled.
    """
    tr = translation or ""
    if not tr or not source:
        return tr
    # Only normalize when the source term is Latin-script: a CJK or Cyrillic
    # source never implies an en->zh typographic convention.
    from ubt.core.validators.glossary_enforcer import is_cjk_char

    if is_cjk_char(source[0]) or is_cjk_char(source[-1]):
        return tr

    out: list[str] = []
    # Boundary pairs the export stage's Pangu rules also space: CJK<->Latin
    # letter/digit, and CJK<->half-width bracket (式(3.1) -> 式 (3.1)). A space
    # is never inserted against full-width punctuation — "多头注意力（MHA）"
    # stays glued, matching apply_pangu_spacing exactly.
    _latin = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    _open_half = "([{"
    _close_half = ")]}"
    for ch in tr:
        prev = out[-1] if out else ""
        if not prev or prev in " \t" or ch in " \t" or is_cjk_char(prev) == is_cjk_char(ch):
            out.append(ch)
            continue
        prev_half = prev in _latin or prev in _close_half
        ch_half = ch in _latin or ch in _open_half
        if prev_half or ch_half:
            out.append(" ")
        out.append(ch)
    return "".join(out)


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

    # Canonical typography: the enforcer writes this string into targets, so
    # it must carry the same CJK<->Latin spacing the export stage ships
    # (Pangu). Applied after the length guards so a spaced rendering can
    # never push a marginal entry over the cap retroactively.
    tr = normalize_glossary_spacing(tr, source=src)

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
