"""Cross-chapter citation topology builder and hyperlink injector."""

from __future__ import annotations

import re
from collections.abc import Sequence

from ubt.core.ir.models import BlockType, IRBlock
from ubt.core.references.extractor import (
    _CITE_BRACKET_RE,
    _PROSE_EQUATION_REF_RE,
    _PROSE_FIGURE_REF_RE,
    _PROSE_SECTION_REF_RE,
    _PROSE_TABLE_REF_RE,
    extract_document_anchors,
    find_prose_citations,
)
from ubt.core.references.models import (
    CitationTopology,
    ReferenceKind,
)

#: Delimiters where hyperlink injection must be skipped
_PROTECTED_SPAN_RE = re.compile(
    r"(?:\$[^\$]+\$)|(?:`[^`]+`)|(?:#link\([^)]+\)\[[^\]]+\])|(?:<mark[^>]*>.*?</mark>)",
    re.DOTALL,
)


def build_citation_topology(blocks: Sequence[IRBlock]) -> CitationTopology:
    """Build complete cross-chapter citation topology from document blocks."""
    topology = extract_document_anchors(blocks)

    for block in blocks:
        if block.block_type is BlockType.NARRATIVE or block.block_type is BlockType.HEADING:
            txt = (block.source_text or "").strip()
            if txt:
                block_links = find_prose_citations(txt, block.id, topology)
                topology.links.extend(block_links)

    return topology


def _inject_typst_link(text: str, topology: CitationTopology) -> str:
    """Inject Typst #link(<anchor>)[text] for resolved reference mentions."""

    def _replace_cite(match: re.Match[str]) -> str:
        full_match = match.group(0)
        raw_inside = match.group(1)
        items = re.split(r"[,;]", raw_inside)
        # Check if first item matches a known bibliography anchor
        if items:
            first_key = items[0].strip().lower()
            anchor_id = topology.key_to_anchor.get((ReferenceKind.BIBLIOGRAPHY, first_key))
            if anchor_id:
                return f"#link(<{anchor_id}>)[{full_match}]"
        return full_match

    def _replace_sec(match: re.Match[str]) -> str:
        full_match = match.group(0)
        val = match.group(1) or match.group(2) or match.group(3)
        if val:
            key = val.strip().lower()
            anchor_id = topology.key_to_anchor.get((ReferenceKind.HEADING, key))
            if anchor_id:
                return f"#link(<{anchor_id}>)[{full_match}]"
        return full_match

    def _replace_fig(match: re.Match[str]) -> str:
        full_match = match.group(0)
        val = match.group(1) or match.group(2)
        if val:
            key = val.strip().lower()
            anchor_id = topology.key_to_anchor.get((ReferenceKind.FIGURE, key))
            if anchor_id:
                return f"#link(<{anchor_id}>)[{full_match}]"
        return full_match

    def _replace_tbl(match: re.Match[str]) -> str:
        full_match = match.group(0)
        val = match.group(1) or match.group(2)
        if val:
            key = val.strip().lower()
            anchor_id = topology.key_to_anchor.get((ReferenceKind.TABLE, key))
            if anchor_id:
                return f"#link(<{anchor_id}>)[{full_match}]"
        return full_match

    def _replace_eq(match: re.Match[str]) -> str:
        full_match = match.group(0)
        val = match.group(1) or match.group(2)
        if val:
            key = val.strip().lower()
            anchor_id = topology.key_to_anchor.get((ReferenceKind.EQUATION, key))
            if anchor_id:
                return f"#link(<{anchor_id}>)[{full_match}]"
        return full_match

    # Apply substitutions in order
    res = _CITE_BRACKET_RE.sub(_replace_cite, text)
    res = _PROSE_SECTION_REF_RE.sub(_replace_sec, res)
    res = _PROSE_FIGURE_REF_RE.sub(_replace_fig, res)
    res = _PROSE_TABLE_REF_RE.sub(_replace_tbl, res)
    res = _PROSE_EQUATION_REF_RE.sub(_replace_eq, res)
    return res


def inject_hyperlinks(
    text: str,
    topology: CitationTopology,
    target_format: str = "typst",
) -> str:
    """Inject clickable cross-reference links into prose text, protecting inline code and math."""
    if not text.strip() or not topology.anchors:
        return text

    # Split text into protected spans ($math$, `code`, #link) and regular prose
    spans: list[tuple[int, int]] = []
    for m in _PROTECTED_SPAN_RE.finditer(text):
        spans.append((m.start(), m.end()))

    if not spans:
        if target_format == "typst":
            return _inject_typst_link(text, topology)
        return text

    # Segment and assemble
    out_parts: list[str] = []
    last_idx = 0
    for start, end in spans:
        if start > last_idx:
            prose_part = text[last_idx:start]
            if target_format == "typst":
                out_parts.append(_inject_typst_link(prose_part, topology))
            else:
                out_parts.append(prose_part)
        out_parts.append(text[start:end])
        last_idx = end

    if last_idx < len(text):
        tail = text[last_idx:]
        if target_format == "typst":
            out_parts.append(_inject_typst_link(tail, topology))
        else:
            out_parts.append(tail)

    return "".join(out_parts)
