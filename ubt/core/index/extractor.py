"""Concordance extractor: builds bilingual index from document IR blocks and glossaries."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Final

from ubt.core.index.models import ConcordanceIndex, IndexEntry, IndexEntryLocation
from ubt.core.ir.models import BlockType, IRBlock
from ubt.model.ast import Heading

_ZH_EN_PAREN_RE: Final[re.Pattern[str]] = re.compile(
    r"([\u4e00-\u9fa5]{2,15})\s*[（(]([A-Za-z][A-Za-z0-9_\-\s]{1,40})[）)]"
)
_EN_ZH_PAREN_RE: Final[re.Pattern[str]] = re.compile(
    r"\b([A-Za-z][A-Za-z0-9_\-\s]{1,40})\s*[（(]([\u4e00-\u9fa5]{2,15})[）)]"
)

# Common non-term English noise in parentheses
_EXCLUDED_PAREN_WORDS: Final[set[str]] = {
    "and",
    "or",
    "see",
    "e.g.",
    "i.e.",
    "etc",
    "note",
    "left",
    "right",
    "top",
    "bottom",
    "true",
    "false",
    "none",
    "null",
}

_LEAD_STOPWORDS: Final[tuple[str, ...]] = (
    "本章主要讨论",
    "本章讨论",
    "本节讨论",
    "我们讨论",
    "另外还涉及",
    "主要涉及",
    "主要讨论",
    "本章",
    "主要",
    "我们",
    "另外",
    "涉及",
    "讨论",
    "学习",
    "提出",
    "使用",
    "引入",
    "基于",
    "支持",
    "对于",
    "关于",
    "称为",
    "即是",
    "即",
    "和",
    "与",
    "及",
    "或",
    "还",
    "在",
    "由",
    "对",
    "将",
    "为",
    "并",
    "则",
    "该",
    "此",
    "这",
    "那",
    "个",
    "种",
    "项",
    "类",
)


def clean_zh_term(term: str) -> str:
    """Strip leading prose verbs, conjunctions, and prepositions to isolate Chinese noun phrase."""
    res = term.strip()
    changed = True
    while changed:
        changed = False
        for p in _LEAD_STOPWORDS:
            if res.startswith(p) and len(res) - len(p) >= 2:
                res = res[len(p) :].strip()
                changed = True
                break
    return res


def extract_index_from_blocks(
    blocks: Sequence[IRBlock],
    glossary: Sequence[dict[str, Any]] | None = None,
    min_occurrences: int = 1,
    doc_id: str = "",
    title: str = "",
) -> ConcordanceIndex:
    """Extract a bilingual concordance index from IRBlocks and optional glossary."""
    active_section_title = ""
    active_section_anchor = ""
    heading_counter = 0

    # Map of canonical_term_key -> IndexEntry
    term_entries: dict[str, IndexEntry] = {}
    term_seen_sections: dict[str, set[str]] = defaultdict(set)

    # Pre-register known glossary entries if supplied
    compiled_glossary: list[tuple[str, str, str, str, re.Pattern[str], re.Pattern[str]]] = []
    if glossary:
        for item in glossary:
            src = str(item.get("source", "")).strip()
            tgt = str(item.get("target", "")).strip()
            cat = str(item.get("category", item.get("domain", ""))).strip()
            if src or tgt:
                key = src.lower() if src else tgt.lower()
                if key not in term_entries:
                    term_entries[key] = IndexEntry(
                        source_term=src,
                        target_term=tgt,
                        category=cat,
                        locations=[],
                        occurrences_count=0,
                    )
                # Compile regex for fast text scanning
                src_pat = (
                    re.compile(r"\b" + re.escape(src) + r"\b", re.IGNORECASE)
                    if src
                    else re.compile(r"$^")
                )
                tgt_pat = re.compile(re.escape(tgt)) if tgt else re.compile(r"$^")
                compiled_glossary.append((key, src, tgt, cat, src_pat, tgt_pat))

    for block in blocks:
        # ``source_text`` projects ``element`` by kind; the old ``element.text``
        # fallback could not type-check against the ``Element`` union and was
        # unreachable for every non-prose arm.
        text = block.target_text or block.source_text or ""
        source_text = block.source_text or ""
        block_id = (
            getattr(block, "id", None)
            or (block.element.id if getattr(block, "element", None) else None)
            or f"blk-{id(block)}"
        )
        page = (
            getattr(block, "page", None)
            or getattr(block.element, "page", None)
            or (
                getattr(block.element.span, "page", None)
                if getattr(block.element, "span", None)
                else None
            )
        )

        # Track active heading
        el = getattr(block, "element", None)
        is_heading = (
            isinstance(el, Heading) or getattr(block, "block_type", None) == BlockType.HEADING
        )
        if is_heading and (text.strip() or source_text.strip()):
            heading_counter += 1
            active_section_title = text.strip() or source_text.strip()
            active_section_anchor = f"sec-{heading_counter}"

        # 1. Check pre-compiled glossary terms
        for key, _src, _tgt, _cat, src_pat, tgt_pat in compiled_glossary:
            count = len(src_pat.findall(source_text)) + len(tgt_pat.findall(text))
            if count > 0:
                entry = term_entries[key]
                entry.occurrences_count += count
                loc_key = active_section_title or block_id
                if loc_key not in term_seen_sections[key]:
                    term_seen_sections[key].add(loc_key)
                    loc = IndexEntryLocation(
                        block_id=block_id,
                        section_title=active_section_title,
                        section_anchor=active_section_anchor or f"loc-{block_id}",
                        page=page,
                    )
                    entry.locations.append(loc)
                    if not entry.primary_anchor:
                        entry.primary_anchor = loc.section_anchor

        # 2. Mine explicit bilingual term annotations in target text (e.g. '共识算法 (Consensus)')
        for m in _ZH_EN_PAREN_RE.finditer(text):
            tgt_t = clean_zh_term(m.group(1).strip())
            src_t = m.group(2).strip()
            if src_t.lower() in _EXCLUDED_PAREN_WORDS or len(src_t) <= 1:
                continue
            key = src_t.lower()
            if key not in term_entries:
                term_entries[key] = IndexEntry(
                    source_term=src_t,
                    target_term=tgt_t,
                    locations=[],
                    occurrences_count=0,
                )
            entry = term_entries[key]
            if not entry.target_term or len(tgt_t) < len(entry.target_term):
                entry.target_term = tgt_t
            entry.occurrences_count += 1
            loc_key = active_section_title or block_id
            if loc_key not in term_seen_sections[key]:
                term_seen_sections[key].add(loc_key)
                loc = IndexEntryLocation(
                    block_id=block_id,
                    section_title=active_section_title,
                    section_anchor=active_section_anchor or f"loc-{block_id}",
                    page=page,
                )
                entry.locations.append(loc)
                if not entry.primary_anchor:
                    entry.primary_anchor = loc.section_anchor

        for m in _EN_ZH_PAREN_RE.finditer(text):
            src_t = m.group(1).strip()
            tgt_t = clean_zh_term(m.group(2).strip())
            if src_t.lower() in _EXCLUDED_PAREN_WORDS or len(src_t) <= 1:
                continue
            key = src_t.lower()
            if key not in term_entries:
                term_entries[key] = IndexEntry(
                    source_term=src_t,
                    target_term=tgt_t,
                    locations=[],
                    occurrences_count=0,
                )
            entry = term_entries[key]
            if not entry.target_term or len(tgt_t) < len(entry.target_term):
                entry.target_term = tgt_t
            entry.occurrences_count += 1
            loc_key = active_section_title or block_id
            if loc_key not in term_seen_sections[key]:
                term_seen_sections[key].add(loc_key)
                loc = IndexEntryLocation(
                    block_id=block_id,
                    section_title=active_section_title,
                    section_anchor=active_section_anchor or f"loc-{block_id}",
                    page=page,
                )
                entry.locations.append(loc)
                if not entry.primary_anchor:
                    entry.primary_anchor = loc.section_anchor

    valid_entries = [e for e in term_entries.values() if e.occurrences_count >= min_occurrences]

    return ConcordanceIndex(
        entries=valid_entries,
        doc_id=doc_id,
        title=title,
    )
