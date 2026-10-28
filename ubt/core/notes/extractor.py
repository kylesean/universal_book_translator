"""Extraction and correlation engine for footnote callouts and definitions."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Final

from ubt.core.ir.models import IRBlock
from ubt.core.notes.models import (
    FootnoteItem,
    NoteNumberingScope,
    NotesConfig,
    NotesTopology,
)

#: Standard Markdown footnote definition: [^1]: Text... or [^label]: Text...
_MD_DEF_RE: Final[re.Pattern[str]] = re.compile(
    r"^\[\^([a-zA-Z0-9_\-]+)\]:\s*([\s\S]*)",
)

#: Generic numbered/symbolic footnote definition at start of block:
#: e.g. "1. Text...", "[1] Text...", "① Text...", "* Text...", "† Text..."
_GENERIC_DEF_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:\[(\d{1,3})\]|(\d{1,3})\.|\(([0-9]{1,3})\)|([①-⑳])|(\*+|†+|‡+|§+))\s*[:：]?\s+([\s\S]*)",
)

#: Markdown footnote callout in prose: [^1], [^label]
_CALLOUT_MD_RE: Final[re.Pattern[str]] = re.compile(
    r"\[\^([a-zA-Z0-9_\-]+)\]",
)

#: Circled number callout in prose: ①, ②...
_CALLOUT_CIRCLED_RE: Final[re.Pattern[str]] = re.compile(
    r"([①-⑳])",
)

#: Symbol callout in prose: *, †, ‡, §
_CALLOUT_SYMBOL_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<=\S)([\*†‡§]{1,3})(?=\s|[，。；！？,.?!]|$)",
)

#: Unicode superscript numbers in prose: ¹, ², ³, ⁴...
_CALLOUT_SUPERSCRIPT_RE: Final[re.Pattern[str]] = re.compile(
    r"([¹²³⁴⁵⁶⁷⁸⁹⁰]+)",
)

_SUPERSCRIPT_MAP: Final[dict[str, str]] = {
    "¹": "1",
    "²": "2",
    "³": "3",
    "⁴": "4",
    "⁵": "5",
    "⁶": "6",
    "⁷": "7",
    "⁸": "8",
    "⁹": "9",
    "⁰": "0",
}

_CIRCLED_MAP: Final[dict[str, int]] = {
    c: i + 1
    for i, c in enumerate(
        [
            "①",
            "②",
            "③",
            "④",
            "⑤",
            "⑥",
            "⑦",
            "⑧",
            "⑨",
            "⑩",
            "⑪",
            "⑫",
            "⑬",
            "⑭",
            "⑮",
            "⑯",
            "⑰",
            "⑱",
            "⑲",
            "⑳",
        ]
    )
}


def _block_text(block: IRBlock) -> str:
    return (
        block.target_text
        or block.source_text
        or (block.element.text if block.element else "")
        or ""
    ).strip()


def _is_heading_1(block: IRBlock) -> bool:
    b_type = str(getattr(block.block_type, "value", block.block_type) or "")
    if b_type != "heading":
        return False
    el = getattr(block, "element", None)
    lvl = getattr(el, "level", None) or getattr(block, "level", None) or 1
    return lvl == 1


def _parse_definition_starter(text: str) -> tuple[str, str] | None:
    """Parse leading footnote marker from a definition string: (key, note_content)."""
    m_md = _MD_DEF_RE.match(text)
    if m_md:
        return m_md.group(1).strip(), m_md.group(2).strip()

    m_gen = _GENERIC_DEF_RE.match(text)
    if m_gen:
        key = m_gen.group(1) or m_gen.group(2) or m_gen.group(3) or m_gen.group(4) or m_gen.group(5)
        content = m_gen.group(6).strip()
        return key.strip(), content

    return None


def extract_notes_topology(
    blocks: Sequence[IRBlock], config: NotesConfig | None = None
) -> NotesTopology:
    """Extract footnote callouts and definitions, correlating them into a document topology."""
    cfg = config or NotesConfig()

    topology = NotesTopology(
        placement=cfg.placement,
        numbering_scope=cfg.numbering_scope,
        numbering_style=cfg.numbering_style,
    )

    current_chapter_idx = 1
    current_chapter_title = ""

    # Phase 1: Scan all blocks and collect candidates
    # chapter_defs: chapter_idx -> list[(key, content, block_id, page)]
    chapter_defs: dict[int, list[tuple[str, str, str, int | None]]] = {}
    # chapter_callouts: chapter_idx -> list[(raw_match, normalized_key, block_id, span_start, span_end)]
    chapter_callouts: dict[int, list[tuple[str, str, str, int, int]]] = {}

    for block in blocks:
        text = _block_text(block)
        if not text:
            continue

        if _is_heading_1(block):
            if chapter_callouts or chapter_defs:
                current_chapter_idx += 1
            current_chapter_title = text
            chapter_defs.setdefault(current_chapter_idx, [])
            chapter_callouts.setdefault(current_chapter_idx, [])
            continue

        chapter_defs.setdefault(current_chapter_idx, [])
        chapter_callouts.setdefault(current_chapter_idx, [])

        b_type = str(getattr(block.block_type, "value", block.block_type) or "")
        f_id = str(getattr(block, "flow_id", None) or "")
        page_num = getattr(block, "page", None) or (
            getattr(block.element.span, "page", None)
            if getattr(block, "element", None) and getattr(block.element, "span", None)
            else None
        )

        is_explicit_fn_block = (
            b_type in ("footnote", "BlockType.FOOTNOTE")
            or f_id == "footnote"
            or text.startswith("[^")
        )

        # Check if block is a footnote definition
        def_info = _parse_definition_starter(text)
        if def_info:
            key, body = def_info
            chapter_defs[current_chapter_idx].append((key, body, block.id, page_num))
            topology.def_block_ids.add(block.id)
            continue
        elif is_explicit_fn_block:
            # Explicit footnote block without clear numerical starter
            chapter_defs[current_chapter_idx].append(
                (f"auto_{len(chapter_defs[current_chapter_idx]) + 1}", text, block.id, page_num)
            )
            topology.def_block_ids.add(block.id)
            continue

        # Look for callouts inside prose blocks
        # 1. Markdown callouts: [^label]
        for m in _CALLOUT_MD_RE.finditer(text):
            chapter_callouts[current_chapter_idx].append(
                (m.group(0), m.group(1), block.id, m.start(), m.end())
            )

        # 2. Circled callouts: ①, ②...
        for m in _CALLOUT_CIRCLED_RE.finditer(text):
            val = str(_CIRCLED_MAP.get(m.group(1), m.group(1)))
            chapter_callouts[current_chapter_idx].append(
                (m.group(0), val, block.id, m.start(), m.end())
            )

        # 3. Superscript digits: ¹, ²...
        for m in _CALLOUT_SUPERSCRIPT_RE.finditer(text):
            digits = "".join(_SUPERSCRIPT_MAP.get(ch, ch) for ch in m.group(1))
            chapter_callouts[current_chapter_idx].append(
                (m.group(0), digits, block.id, m.start(), m.end())
            )

        # 4. Symbols: *, †, ‡
        for m in _CALLOUT_SYMBOL_RE.finditer(text):
            chapter_callouts[current_chapter_idx].append(
                (m.group(1), m.group(1), block.id, m.start(), m.end())
            )

    # Phase 2: Correlate callouts with definitions
    global_note_counter = 0

    for ch_idx, defs in chapter_defs.items():
        callouts = chapter_callouts.get(ch_idx, [])
        callout_by_key: dict[str, list[tuple[str, str, str, int, int]]] = {}
        for c in callouts:
            raw, norm_key, b_id, s_s, s_e = c
            callout_by_key.setdefault(norm_key, []).append(c)

        used_callout_keys: set[str] = set()

        for ch_note_counter, def_item in enumerate(defs, 1):
            d_key, d_content, d_blk_id, d_page = def_item
            global_note_counter += 1

            index_num = (
                ch_note_counter
                if cfg.numbering_scope == NoteNumberingScope.PER_CHAPTER
                else global_note_counter
            )

            # Look for matching callout
            matched_callout = None
            if d_key in callout_by_key and callout_by_key[d_key]:
                matched_callout = callout_by_key[d_key].pop(0)
                used_callout_keys.add(d_key)
            elif (
                (d_key.isdigit() or d_key.startswith("auto_"))
                and str(ch_note_counter) in callout_by_key
                and callout_by_key[str(ch_note_counter)]
            ):
                # Fallback to positional numeric match (e.g. 1st def matches 1st numbered callout)
                matched_callout = callout_by_key[str(ch_note_counter)].pop(0)
                used_callout_keys.add(str(ch_note_counter))
            elif d_key.startswith("auto_") and callouts:
                for c in callouts:
                    if c[1] not in used_callout_keys:
                        matched_callout = c
                        used_callout_keys.add(c[1])
                        break

            note_id = (
                f"fn-ch{ch_idx}-{index_num}"
                if cfg.numbering_scope == NoteNumberingScope.PER_CHAPTER
                else f"fn-{index_num}"
            )

            src_blk_id = matched_callout[2] if matched_callout else None
            callout_marker = matched_callout[0] if matched_callout else f"[{d_key}]"

            fn_item = FootnoteItem(
                note_id=note_id,
                index_num=index_num,
                callout_marker=callout_marker,
                content=d_content,
                chapter_index=ch_idx,
                chapter_title=current_chapter_title,
                source_block_id=src_blk_id,
                def_block_id=d_blk_id,
                page=d_page,
            )
            topology.notes.append(fn_item)

            if not matched_callout:
                topology.orphan_definitions.append(
                    {
                        "chapter": ch_idx,
                        "key": d_key,
                        "block_id": d_blk_id,
                        "content_preview": d_content[:60],
                    }
                )

        # Check remaining unmatched callouts
        for c in callouts:
            if c[1] not in used_callout_keys:
                topology.orphan_callouts.append(
                    {
                        "chapter": ch_idx,
                        "raw_marker": c[0],
                        "key": c[1],
                        "block_id": c[2],
                    }
                )

    return topology
