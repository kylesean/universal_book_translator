"""Cross-chapter citation topology builder and hyperlink injector."""

from __future__ import annotations

import re
from collections.abc import Sequence

from ubt.core.ir.models import BlockType, IRBlock
from ubt.core.references.extractor import (
    _CITE_BRACKET_RE,
    _PROSE_ALGORITHM_REF_RE,
    _PROSE_EQUATION_REF_RE,
    _PROSE_FIGURE_REF_RE,
    _PROSE_LISTING_REF_RE,
    _PROSE_PAGE_REF_RE,
    _PROSE_SECTION_REF_RE,
    _PROSE_TABLE_REF_RE,
    extract_document_anchors,
    find_prose_citations,
)
from ubt.core.references.models import (
    CitationTopology,
    CrossRefConfig,
    CrossRefPrefixScheme,
    CrossRefStyle,
    ReferenceKind,
)

#: Delimiters where hyperlink injection must be skipped
_PROTECTED_SPAN_RE = re.compile(
    r"(?:\$[^\$]+\$)|(?:`[^`]+`)|(?:#link\([^)]+\)\[[^\]]+\])|(?:<mark[^>]*>.*?</mark>)|(?:<a\s+[^>]*>.*?</a>)|(?:\[[^\]]+\]\([^)]+\))",
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


def format_mention_text(
    original_text: str,
    kind: ReferenceKind,
    key: str,
    scheme: CrossRefPrefixScheme,
    lang: str = "zh",
) -> str:
    """Format reference mention text according to standard schemes."""
    if scheme is CrossRefPrefixScheme.ORIGINAL:
        return original_text

    is_zh = lang.lower().startswith("zh")

    if scheme is CrossRefPrefixScheme.CHINESE_STANDARD or (
        scheme is CrossRefPrefixScheme.ORIGINAL and is_zh
    ):
        if kind is ReferenceKind.FIGURE:
            return f"图 {key}"
        elif kind is ReferenceKind.TABLE:
            return f"表 {key}"
        elif kind is ReferenceKind.HEADING:
            return f"第 {key} 章" if key.isdigit() else f"第 {key} 节"
        elif kind is ReferenceKind.EQUATION:
            return f"式 ({key})"
        elif kind is ReferenceKind.PAGE:
            return f"第 {key} 页"
        elif kind is ReferenceKind.LISTING:
            return f"代码清单 {key}"
        elif kind is ReferenceKind.ALGORITHM:
            return f"算法 {key}"
        elif kind is ReferenceKind.BIBLIOGRAPHY:
            return f"[{key}]"

    elif scheme is CrossRefPrefixScheme.ACADEMIC_FORMAL:
        if kind is ReferenceKind.FIGURE:
            return f"Figure {key}"
        elif kind is ReferenceKind.TABLE:
            return f"Table {key}"
        elif kind is ReferenceKind.HEADING:
            return f"Chapter {key}" if key.isdigit() else f"Section {key}"
        elif kind is ReferenceKind.EQUATION:
            return f"Equation ({key})"
        elif kind is ReferenceKind.PAGE:
            return f"page {key}"
        elif kind is ReferenceKind.LISTING:
            return f"Listing {key}"
        elif kind is ReferenceKind.ALGORITHM:
            return f"Algorithm {key}"
        elif kind is ReferenceKind.BIBLIOGRAPHY:
            return f"[{key}]"

    elif scheme is CrossRefPrefixScheme.COMPACT:
        if kind is ReferenceKind.FIGURE:
            return f"Fig. {key}"
        elif kind is ReferenceKind.TABLE:
            return f"Tab. {key}"
        elif kind is ReferenceKind.HEADING:
            return f"Sec. {key}"
        elif kind is ReferenceKind.EQUATION:
            return f"Eq. ({key})"
        elif kind is ReferenceKind.PAGE:
            return f"p. {key}"
        elif kind is ReferenceKind.LISTING:
            return f"List. {key}"
        elif kind is ReferenceKind.ALGORITHM:
            return f"Algo. {key}"
        elif kind is ReferenceKind.BIBLIOGRAPHY:
            return f"[{key}]"

    return original_text


def _format_link_element(
    anchor_id: str,
    matched_text: str,
    kind: ReferenceKind,
    key: str,
    config: CrossRefConfig,
    target_format: str,
) -> str:
    display_text = format_mention_text(
        matched_text,
        kind,
        key,
        config.prefix_scheme,
        lang=config.target_lang,
    )

    if target_format == "typst":
        if config.style is CrossRefStyle.NATIVE_REF and kind is not ReferenceKind.PAGE:
            return f"#ref(<{anchor_id}>)"

        if config.style is CrossRefStyle.PAGE_ONLY:
            if kind is ReferenceKind.PAGE:
                return f"#link(<{anchor_id}>)[{display_text}]"
            p_prefix = "第 " if config.target_lang.startswith("zh") else "p. "
            p_suffix = " 页" if config.target_lang.startswith("zh") else ""
            return (
                f"#link(<{anchor_id}>)[#context {{"
                f" let loc = query(selector(<{anchor_id}>));"
                f" if loc.len() > 0 [{p_prefix}#loc.first().location().page(){p_suffix}]"
                f" else [{display_text}]"
                f" }}]"
            )

        if (
            config.style is CrossRefStyle.WITH_PAGE or config.show_page_number
        ) and kind is not ReferenceKind.PAGE:
            p_prefix = " (第 " if config.target_lang.startswith("zh") else " (p. "
            p_suffix = " 页)" if config.target_lang.startswith("zh") else ")"
            return (
                f"#link(<{anchor_id}>)[{display_text}]#context {{"
                f" let loc = query(selector(<{anchor_id}>));"
                f" if loc.len() > 0 and loc.first().location().page() != here().page() [{p_prefix}#loc.first().location().page(){p_suffix}]"
                f" }}"
            )

        return f"#link(<{anchor_id}>)[{display_text}]"

    elif target_format in ("html", "epub"):
        target_attr = ' target="_blank" rel="noopener noreferrer"' if config.open_in_new_tab else ""
        css = f"{config.link_css_class} {config.link_css_class}-{kind.value}"
        return f'<a href="#{anchor_id}" class="{css}" data-ref-kind="{kind.value}"{target_attr}>{display_text}</a>'

    elif target_format == "markdown":
        return f"[{display_text}](#{anchor_id})"

    return display_text


def _inject_prose_links(
    text: str,
    topology: CitationTopology,
    target_format: str,
    config: CrossRefConfig,
    skip_anchors: set[str] | None = None,
) -> str:
    """Inject formatted links into prose text according to topology and config."""
    skipped = skip_anchors or set()

    def _replace_cite(match: re.Match[str]) -> str:
        full_match = match.group(0)
        raw_inside = match.group(1)
        items = re.split(r"[,;]", raw_inside)
        if items:
            first_key = items[0].strip().lower()
            anchor_id = topology.key_to_anchor.get((ReferenceKind.BIBLIOGRAPHY, first_key))
            if anchor_id and anchor_id not in skipped:
                return _format_link_element(
                    anchor_id,
                    full_match,
                    ReferenceKind.BIBLIOGRAPHY,
                    first_key,
                    config,
                    target_format,
                )
        return full_match

    def _replace_sec(match: re.Match[str]) -> str:
        full_match = match.group(0)
        val = match.group(1) or match.group(2) or match.group(3)
        if val:
            key = val.strip().lower()
            anchor_id = topology.key_to_anchor.get((ReferenceKind.HEADING, key))
            if anchor_id and anchor_id not in skipped:
                return _format_link_element(
                    anchor_id,
                    full_match,
                    ReferenceKind.HEADING,
                    key,
                    config,
                    target_format,
                )
        return full_match

    def _replace_fig(match: re.Match[str]) -> str:
        full_match = match.group(0)
        val = match.group(1) or match.group(2)
        if val:
            key = val.strip().lower()
            anchor_id = topology.key_to_anchor.get((ReferenceKind.FIGURE, key))
            if anchor_id and anchor_id not in skipped:
                return _format_link_element(
                    anchor_id,
                    full_match,
                    ReferenceKind.FIGURE,
                    key,
                    config,
                    target_format,
                )
        return full_match

    def _replace_tbl(match: re.Match[str]) -> str:
        full_match = match.group(0)
        val = match.group(1) or match.group(2)
        if val:
            key = val.strip().lower()
            anchor_id = topology.key_to_anchor.get((ReferenceKind.TABLE, key))
            if anchor_id and anchor_id not in skipped:
                return _format_link_element(
                    anchor_id,
                    full_match,
                    ReferenceKind.TABLE,
                    key,
                    config,
                    target_format,
                )
        return full_match

    def _replace_eq(match: re.Match[str]) -> str:
        full_match = match.group(0)
        val = match.group(1) or match.group(2)
        if val:
            key = val.strip().lower()
            anchor_id = topology.key_to_anchor.get((ReferenceKind.EQUATION, key))
            if anchor_id and anchor_id not in skipped:
                return _format_link_element(
                    anchor_id,
                    full_match,
                    ReferenceKind.EQUATION,
                    key,
                    config,
                    target_format,
                )
        return full_match

    def _replace_page(match: re.Match[str]) -> str:
        if not config.enable_page_references:
            return match.group(0)
        full_match = match.group(0)
        val = match.group(1) or match.group(2)
        if val:
            key = val.strip().lower()
            anchor_id = topology.key_to_anchor.get((ReferenceKind.PAGE, key))
            if anchor_id and anchor_id not in skipped:
                return _format_link_element(
                    anchor_id,
                    full_match,
                    ReferenceKind.PAGE,
                    key,
                    config,
                    target_format,
                )
        return full_match

    def _replace_listing(match: re.Match[str]) -> str:
        full_match = match.group(0)
        val = match.group(1) or match.group(2)
        if val:
            key = val.strip().lower()
            anchor_id = topology.key_to_anchor.get((ReferenceKind.LISTING, key))
            if anchor_id and anchor_id not in skipped:
                return _format_link_element(
                    anchor_id,
                    full_match,
                    ReferenceKind.LISTING,
                    key,
                    config,
                    target_format,
                )
        return full_match

    def _replace_algorithm(match: re.Match[str]) -> str:
        full_match = match.group(0)
        val = match.group(1) or match.group(2)
        if val:
            key = val.strip().lower()
            anchor_id = topology.key_to_anchor.get((ReferenceKind.ALGORITHM, key))
            if anchor_id and anchor_id not in skipped:
                return _format_link_element(
                    anchor_id,
                    full_match,
                    ReferenceKind.ALGORITHM,
                    key,
                    config,
                    target_format,
                )
        return full_match

    # Apply substitutions in order
    res = _CITE_BRACKET_RE.sub(_replace_cite, text)
    res = _PROSE_SECTION_REF_RE.sub(_replace_sec, res)
    res = _PROSE_FIGURE_REF_RE.sub(_replace_fig, res)
    res = _PROSE_TABLE_REF_RE.sub(_replace_tbl, res)
    res = _PROSE_EQUATION_REF_RE.sub(_replace_eq, res)
    res = _PROSE_PAGE_REF_RE.sub(_replace_page, res)
    res = _PROSE_LISTING_REF_RE.sub(_replace_listing, res)
    res = _PROSE_ALGORITHM_REF_RE.sub(_replace_algorithm, res)
    return res


def inject_hyperlinks(
    text: str,
    topology: CitationTopology,
    target_format: str = "typst",
    config: CrossRefConfig | None = None,
    skip_anchors: set[str] | None = None,
) -> str:
    """Inject clickable cross-reference links into prose text, protecting inline code and math."""
    if not text.strip() or not topology.anchors:
        return text

    active_config = config or CrossRefConfig(target_format=target_format)
    fmt = target_format or active_config.target_format

    # Split text into protected spans ($math$, `code`, #link, etc.) and regular prose
    spans: list[tuple[int, int]] = []
    for m in _PROTECTED_SPAN_RE.finditer(text):
        spans.append((m.start(), m.end()))

    if not spans:
        return _inject_prose_links(text, topology, fmt, active_config, skip_anchors=skip_anchors)

    # Segment and assemble
    out_parts: list[str] = []
    last_idx = 0
    for start, end in spans:
        if start > last_idx:
            prose_part = text[last_idx:start]
            out_parts.append(
                _inject_prose_links(
                    prose_part, topology, fmt, active_config, skip_anchors=skip_anchors
                )
            )
        out_parts.append(text[start:end])
        last_idx = end

    if last_idx < len(text):
        tail = text[last_idx:]
        out_parts.append(
            _inject_prose_links(tail, topology, fmt, active_config, skip_anchors=skip_anchors)
        )

    return "".join(out_parts)
