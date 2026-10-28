"""Detector and extractor for chapter epigraphs, quotations, and introductory abstracts."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ubt.core.epigraph.models import ChapterAbstract, Epigraph

if TYPE_CHECKING:
    from ubt.core.ir.models import IRBlock

_ATTRIBUTION_SPLIT_RE = re.compile(
    r"(?:\n|\r\n)\s*(?:>)?\s*(?:——|--|—|by\s+|——\s*|--\s*)\s*(.*?)$",
    re.IGNORECASE,
)

_INLINE_ATTRIBUTION_RE = re.compile(
    r"^(.*?)\s*(?:——|--|—)\s*([^—–—\n\r]+)$",
)

_ABSTRACT_PREFIX_RE = re.compile(
    r"^\s*(?:>)?\s*(?:\*\*)?(?:【)?(?:本章导读|内容提要|本章提要|章节导读|导读|Chapter Summary|Abstract|Summary)(?:】)?(?:\*\*)?[:：\s]*(.*)$",
    re.IGNORECASE | re.DOTALL,
)


def _clean_markdown_quote(text: str) -> str:
    """Remove leading '>' characters and extra whitespace from markdown quote lines."""
    lines = [re.sub(r"^\s*>\s*", "", line) for line in text.splitlines()]
    return "\n".join(lines).strip()


def parse_epigraph_text(text: str) -> tuple[str, str] | None:
    """Parse quote body and attribution source from text if it has epigraph structure."""
    cleaned = _clean_markdown_quote(text)
    if not cleaned:
        return None

    # Multi-line attribution match (e.g. quote on line 1, author on line 2)
    m_split = _ATTRIBUTION_SPLIT_RE.search(cleaned)
    if m_split:
        quote = cleaned[: m_split.start()].strip()
        source = m_split.group(1).strip()
        quote = re.sub(r"^[“\"「『\s]+|[”\"」』\s]+$", "", quote).strip()
        if quote and source:
            return quote, source

    # Single-line attribution match (e.g. "Quote text" —— Author)
    m_inline = _INLINE_ATTRIBUTION_RE.match(cleaned)
    if m_inline:
        quote = m_inline.group(1).strip()
        source = m_inline.group(2).strip()
        quote = re.sub(r"^[“\"「『\s]+|[”\"」』\s]+$", "", quote).strip()
        if quote and source and len(source) <= 60:
            return quote, source

    # If text is enclosed in quotation marks and short (<160 chars) with parenthetical author
    m_paren = re.match(r'^[“"「『](.*?)[”"」』]\s*[（\(](.*?)[）\)]$', cleaned, re.DOTALL)
    if m_paren:
        return m_paren.group(1).strip(), m_paren.group(2).strip()

    return None


def parse_abstract_text(text: str) -> tuple[str, str] | None:
    """Parse chapter abstract title and content if block begins with abstract keywords."""
    cleaned = _clean_markdown_quote(text)
    if not cleaned:
        return None

    m = _ABSTRACT_PREFIX_RE.match(cleaned)
    if not m:
        return None

    body = m.group(1).strip()
    if not body:
        return None

    # Determine canonical title
    prefix = cleaned[: m.start(1)].strip()
    title = re.sub(r"^[\s*【】]+|[\s*【】:：]+$", "", prefix).strip() or "本章导读"
    return title, body


def detect_chapter_epigraphs_and_abstracts(
    blocks: list[IRBlock],
) -> tuple[dict[str, Epigraph], dict[str, ChapterAbstract]]:
    """Scan blocks and detect chapter-opening epigraphs and abstracts.

    Returns:
        (epigraphs_by_block_id, abstracts_by_block_id)
    """
    epigraphs: dict[str, Epigraph] = {}
    abstracts: dict[str, ChapterAbstract] = {}

    current_chapter = 0
    in_chapter_preamble = False

    for b in blocks:
        btype = str(
            getattr(
                getattr(b, "block_type", None),
                "value",
                getattr(b, "block_type", getattr(b, "type", "")),
            )
            or ""
        ).lower()
        level = getattr(getattr(b, "element", None), "level", getattr(b, "level", 1))

        if btype == "heading" and level == 1:
            current_chapter += 1
            in_chapter_preamble = True
            continue

        if not in_chapter_preamble:
            continue

        text = (b.target_text or b.source_text or "").strip()
        if not text:
            continue

        # Try abstract match first
        abs_res = parse_abstract_text(text)
        if abs_res is not None:
            title, abs_body = abs_res
            abstracts[b.id] = ChapterAbstract(
                title=title,
                text=abs_body,
                chapter_index=current_chapter,
                block_id=b.id,
            )
            continue

        # Try epigraph match
        epi_res = parse_epigraph_text(text)
        if epi_res is not None:
            q_text, q_src = epi_res
            epigraphs[b.id] = Epigraph(
                text=q_text,
                source=q_src,
                chapter_index=current_chapter,
                block_id=b.id,
            )
            continue

        # Encountered standard narrative body text: preamble ends
        in_chapter_preamble = False

    return epigraphs, abstracts
