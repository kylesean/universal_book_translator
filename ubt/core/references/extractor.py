"""Anchor and citation extraction engine for technical documents."""

from __future__ import annotations

import re
from collections.abc import Sequence

from ubt.core.engine.stages.ingest import _BIB_SECTION_HEADING_RE
from ubt.core.ir.models import BlockType, IRBlock
from ubt.core.references.models import (
    AnchorTarget,
    CitationLink,
    CitationTopology,
    ReferenceKind,
)
from ubt.model.ast import RegionKind

#: Citation brackets: [1], [1, 2], [1-3], [Vaswani17], [He+16], and Typst-escaped \\[1\\]
_CITE_BRACKET_RE = re.compile(
    r"\\?\[\s*([A-Za-z0-9_\-\+]+(?:\s*(?:,\s*|\s*[-–—]\s*)[A-Za-z0-9_\-\+]+)*)\s*\\?\]"
)

#: Section numbers: 1. Introduction, 2.3 Methodology, Chapter 4, Section 2
_HEADING_NUMBER_RE = re.compile(
    r"^(?:(?:chapter|chap\.|section|sec\.|appendix|附录|第)\s*(\d+(?:\.\d+)*|[A-Z])(?:\s*(?:章|节|[\.\-–—:]))?|"
    r"(\d+(?:\.\d+)+)(?:\s*[\.\-–—:]\s*|\s+)|"
    r"(\d+|[IVXLCDM]+|[A-Z])\s*[\.\-–—:])",
    re.IGNORECASE,
)

#: Captions: Figure 1, Fig. 2(a), 图 3
_FIGURE_CAPTION_RE = re.compile(
    r"^(?:figure|fig\.|图)\s*(\d+(?:\.\d+)*[a-z]?)",
    re.IGNORECASE,
)

#: Captions: Table 1, Tab. 2, 表 3
_TABLE_CAPTION_RE = re.compile(
    r"^(?:table|tab\.|表)\s*(\d+(?:\.\d+)*)",
    re.IGNORECASE,
)

#: Equation labels: (1), (2.3), \tag{1}, Equation (4)
_EQUATION_LABEL_RE = re.compile(
    r"(?:(?:\(|\\tag\{)(\d+(?:\.\d+)*)(?:\)|\})|^(?:(?:equation|eq\.|式|公式)\s*)?\((\d+(?:\.\d+)*)\))",
    re.IGNORECASE,
)

#: Bibliography item starters: [1], [12], [Vaswani17], 1.
_BIB_ITEM_STARTER_RE = re.compile(
    r"^(?:\\?\[\s*([A-Za-z0-9_\-\+]+)\s*\\?\]|(\d+)\.\s+)",
)

#: Cross-reference mentions in prose:
_PROSE_SECTION_REF_RE = re.compile(
    r"\b(?:(?:see\s+)?(?:section|sec\.|chapter|chap\.|appendix)\s+(\d+(?:\.\d+)*|[A-Z]))|"
    r"(?:第\s*(\d+(?:\.\d+)*)\s*(?:章|节)|(?:参见|见)?(?:章节|小节|附录)\s*(\d+(?:\.\d+)*|[A-Z]))",
    re.IGNORECASE,
)

_PROSE_FIGURE_REF_RE = re.compile(
    r"\b(?:figure|fig\.)\s*(\d+(?:\.\d+)*[a-z]?)|(?:如图|图)\s*(\d+(?:\.\d+)*[a-z]?)",
    re.IGNORECASE,
)

_PROSE_TABLE_REF_RE = re.compile(
    r"\b(?:table|tab\.)\s*(\d+(?:\.\d+)*)|(?:如表|表)\s*(\d+(?:\.\d+)*)",
    re.IGNORECASE,
)

_PROSE_EQUATION_REF_RE = re.compile(
    r"\b(?:equation|eq\.)\s*\((\d+(?:\.\d+)*)\)|(?:公式|式)\s*\((\d+(?:\.\d+)*)\)",
    re.IGNORECASE,
)

#: Page mentions: 第 15 页, 见第 3 页, 参见第 4 页, page 12, p. 5, see page 20, (page 10)
_PROSE_PAGE_REF_RE = re.compile(
    r"(?:(?:(?:参见|见)?(?:第\s*)?(\d+)\s*页)|\b(?:see\s+)?(?:page|p\.)\s*(\d+)\b)",
    re.IGNORECASE,
)

#: Listing captions and references
_LISTING_CAPTION_RE = re.compile(
    r"^(?:listing|list\.|代码清单)\s*(\d+(?:\.\d+)*)",
    re.IGNORECASE,
)

_PROSE_LISTING_REF_RE = re.compile(
    r"\b(?:listing|list\.)\s*(\d+(?:\.\d+)*)|(?:代码清单)\s*(\d+(?:\.\d+)*)",
    re.IGNORECASE,
)

#: Algorithm captions and references
_ALGORITHM_CAPTION_RE = re.compile(
    r"^(?:algorithm|algo\.|算法)\s*(\d+(?:\.\d+)*)",
    re.IGNORECASE,
)

_PROSE_ALGORITHM_REF_RE = re.compile(
    r"\b(?:algorithm|algo\.)\s*(\d+(?:\.\d+)*)|(?:算法)\s*(\d+(?:\.\d+)*)",
    re.IGNORECASE,
)


def _clean_anchor_id(prefix: str, key: str) -> str:
    """Produce a safe Typst/HTML anchor ID."""
    sanitized = re.sub(r"[^a-zA-Z0-9_\-]", "-", key.strip()).strip("-").lower()
    return f"{prefix}-{sanitized}"


def _get_block_page(block: IRBlock) -> int | None:
    """Extract page number from block bbox or element span."""
    if block.bbox is not None and block.bbox.page is not None:
        return block.bbox.page
    element = getattr(block, "element", None)
    if element is not None:
        span = getattr(element, "span", None)
        if span is not None and getattr(span, "page", None) is not None:
            return span.page
    return None


def extract_document_anchors(blocks: Sequence[IRBlock]) -> CitationTopology:
    """Extract all potential anchor targets from a sequence of document blocks."""
    topology = CitationTopology()
    in_bib_section = False

    for block in blocks:
        src = (block.source_text or "").strip()
        if not src:
            continue
        page = _get_block_page(block)

        # Register page anchor for cross-referencing pages
        if page is not None:
            page_key = str(page)
            page_anchor_id = f"page-{page_key}"
            if page_anchor_id not in topology.anchors:
                anchor = AnchorTarget(
                    anchor_id=page_anchor_id,
                    kind=ReferenceKind.PAGE,
                    key=page_key,
                    block_id=block.id,
                    page=page,
                    title=f"Page {page}",
                )
                topology.anchors[page_anchor_id] = anchor
                topology.key_to_anchor[(ReferenceKind.PAGE, page_key)] = page_anchor_id

        # Check for bibliography section heading
        if block.block_type is BlockType.HEADING:
            in_bib_section = bool(_BIB_SECTION_HEADING_RE.search(src))

            # Extract heading number anchor
            m_sec = _HEADING_NUMBER_RE.match(src)
            if m_sec:
                key = (m_sec.group(1) or m_sec.group(2) or m_sec.group(3)).lower()
                anchor_id = _clean_anchor_id("sec", key)
                anchor = AnchorTarget(
                    anchor_id=anchor_id,
                    kind=ReferenceKind.HEADING,
                    key=key,
                    block_id=block.id,
                    page=page,
                    title=src[:60],
                )
                topology.anchors[anchor_id] = anchor
                topology.key_to_anchor[(ReferenceKind.HEADING, key)] = anchor_id

        # In bibliography section or bibliography-shaped blocks
        elif in_bib_section or block.region is RegionKind.BODY:
            m_bib = _BIB_ITEM_STARTER_RE.match(src)
            if m_bib:
                key = (m_bib.group(1) or m_bib.group(2)).lower()
                anchor_id = _clean_anchor_id("bib", key)
                anchor = AnchorTarget(
                    anchor_id=anchor_id,
                    kind=ReferenceKind.BIBLIOGRAPHY,
                    key=key,
                    block_id=block.id,
                    page=page,
                    title=src[:60],
                )
                topology.anchors[anchor_id] = anchor
                topology.key_to_anchor[(ReferenceKind.BIBLIOGRAPHY, key)] = anchor_id

        # Figures and figure captions
        if (
            block.block_type is BlockType.IMAGE
            or block.region is RegionKind.CAPTION
            or _FIGURE_CAPTION_RE.search(src)
        ):
            m_fig = _FIGURE_CAPTION_RE.search(src)
            if m_fig:
                key = m_fig.group(1).lower()
                anchor_id = _clean_anchor_id("fig", key)
                anchor = AnchorTarget(
                    anchor_id=anchor_id,
                    kind=ReferenceKind.FIGURE,
                    key=key,
                    block_id=block.id,
                    page=page,
                    title=src[:60],
                )
                topology.anchors[anchor_id] = anchor
                topology.key_to_anchor[(ReferenceKind.FIGURE, key)] = anchor_id

        # Tables and table captions
        if (
            block.block_type is BlockType.TABLE
            or block.region is RegionKind.CAPTION
            or _TABLE_CAPTION_RE.search(src)
        ):
            m_tbl = _TABLE_CAPTION_RE.search(src)
            if m_tbl:
                key = m_tbl.group(1).lower()
                anchor_id = _clean_anchor_id("tbl", key)
                anchor = AnchorTarget(
                    anchor_id=anchor_id,
                    kind=ReferenceKind.TABLE,
                    key=key,
                    block_id=block.id,
                    page=page,
                    title=src[:60],
                )
                topology.anchors[anchor_id] = anchor
                topology.key_to_anchor[(ReferenceKind.TABLE, key)] = anchor_id

        # Display equations
        if block.block_type is BlockType.FORMULA:
            m_eq = _EQUATION_LABEL_RE.search(src)
            if m_eq:
                key = (m_eq.group(1) or m_eq.group(2)).lower()
                anchor_id = _clean_anchor_id("eq", key)
                anchor = AnchorTarget(
                    anchor_id=anchor_id,
                    kind=ReferenceKind.EQUATION,
                    key=key,
                    block_id=block.id,
                    page=page,
                    title=src[:60],
                )
                topology.anchors[anchor_id] = anchor
                topology.key_to_anchor[(ReferenceKind.EQUATION, key)] = anchor_id

        # Code listings
        if block.block_type is BlockType.CODE or block.region is RegionKind.CAPTION:
            m_lst = _LISTING_CAPTION_RE.search(src)
            if m_lst:
                key = m_lst.group(1).lower()
                anchor_id = _clean_anchor_id("lst", key)
                anchor = AnchorTarget(
                    anchor_id=anchor_id,
                    kind=ReferenceKind.LISTING,
                    key=key,
                    block_id=block.id,
                    page=page,
                    title=src[:60],
                )
                topology.anchors[anchor_id] = anchor
                topology.key_to_anchor[(ReferenceKind.LISTING, key)] = anchor_id

        # Algorithms
        if block.region is RegionKind.CAPTION or block.block_type in (
            BlockType.CODE,
            BlockType.NARRATIVE,
        ):
            m_alg = _ALGORITHM_CAPTION_RE.search(src)
            if m_alg:
                key = m_alg.group(1).lower()
                anchor_id = _clean_anchor_id("algo", key)
                anchor = AnchorTarget(
                    anchor_id=anchor_id,
                    kind=ReferenceKind.ALGORITHM,
                    key=key,
                    block_id=block.id,
                    page=page,
                    title=src[:60],
                )
                topology.anchors[anchor_id] = anchor
                topology.key_to_anchor[(ReferenceKind.ALGORITHM, key)] = anchor_id

    return topology


def find_prose_citations(
    text: str,
    source_block_id: str,
    topology: CitationTopology,
) -> list[CitationLink]:
    """Find citation and reference occurrences in prose and map them to known anchors."""
    links: list[CitationLink] = []

    # 1. Bracketed citations [1], [1, 2], [Vaswani17]
    for match in _CITE_BRACKET_RE.finditer(text):
        raw_inside = match.group(1)
        full_match = match.group(0)
        # Parse items (could be comma-separated or range)
        items = re.split(r"[,;]", raw_inside)
        for item in items:
            item_clean = item.strip().lower()
            if not item_clean:
                continue
            anchor_id = topology.key_to_anchor.get((ReferenceKind.BIBLIOGRAPHY, item_clean))
            topology.total_citations_found += 1
            if anchor_id:
                topology.resolved_citations_count += 1
                links.append(
                    CitationLink(
                        source_block_id=source_block_id,
                        target_anchor_id=anchor_id,
                        matched_text=full_match,
                        kind=ReferenceKind.BIBLIOGRAPHY,
                        span_start=match.start(),
                        span_end=match.end(),
                    )
                )
            else:
                topology.unresolved_mentions.append(
                    {
                        "source_block_id": source_block_id,
                        "kind": ReferenceKind.BIBLIOGRAPHY.value,
                        "text": item_clean,
                        "raw": full_match,
                    }
                )

    # 2. Section references
    for match in _PROSE_SECTION_REF_RE.finditer(text):
        val = match.group(1) or match.group(2) or match.group(3)
        if not val:
            continue
        key = val.strip().lower()
        anchor_id = topology.key_to_anchor.get((ReferenceKind.HEADING, key))
        topology.total_citations_found += 1
        if anchor_id:
            topology.resolved_citations_count += 1
            links.append(
                CitationLink(
                    source_block_id=source_block_id,
                    target_anchor_id=anchor_id,
                    matched_text=match.group(0),
                    kind=ReferenceKind.HEADING,
                    span_start=match.start(),
                    span_end=match.end(),
                )
            )
        else:
            topology.unresolved_mentions.append(
                {
                    "source_block_id": source_block_id,
                    "kind": ReferenceKind.HEADING.value,
                    "text": key,
                    "raw": match.group(0),
                }
            )

    # 3. Figure references
    for match in _PROSE_FIGURE_REF_RE.finditer(text):
        val = match.group(1) or match.group(2)
        if not val:
            continue
        key = val.strip().lower()
        anchor_id = topology.key_to_anchor.get((ReferenceKind.FIGURE, key))
        topology.total_citations_found += 1
        if anchor_id:
            topology.resolved_citations_count += 1
            links.append(
                CitationLink(
                    source_block_id=source_block_id,
                    target_anchor_id=anchor_id,
                    matched_text=match.group(0),
                    kind=ReferenceKind.FIGURE,
                    span_start=match.start(),
                    span_end=match.end(),
                )
            )
        else:
            topology.unresolved_mentions.append(
                {
                    "source_block_id": source_block_id,
                    "kind": ReferenceKind.FIGURE.value,
                    "text": key,
                    "raw": match.group(0),
                }
            )

    # 4. Table references
    for match in _PROSE_TABLE_REF_RE.finditer(text):
        val = match.group(1) or match.group(2)
        if not val:
            continue
        key = val.strip().lower()
        anchor_id = topology.key_to_anchor.get((ReferenceKind.TABLE, key))
        topology.total_citations_found += 1
        if anchor_id:
            topology.resolved_citations_count += 1
            links.append(
                CitationLink(
                    source_block_id=source_block_id,
                    target_anchor_id=anchor_id,
                    matched_text=match.group(0),
                    kind=ReferenceKind.TABLE,
                    span_start=match.start(),
                    span_end=match.end(),
                )
            )
        else:
            topology.unresolved_mentions.append(
                {
                    "source_block_id": source_block_id,
                    "kind": ReferenceKind.TABLE.value,
                    "text": key,
                    "raw": match.group(0),
                }
            )

    # 5. Equation references
    for match in _PROSE_EQUATION_REF_RE.finditer(text):
        val = match.group(1) or match.group(2)
        if not val:
            continue
        key = val.strip().lower()
        anchor_id = topology.key_to_anchor.get((ReferenceKind.EQUATION, key))
        topology.total_citations_found += 1
        if anchor_id:
            topology.resolved_citations_count += 1
            links.append(
                CitationLink(
                    source_block_id=source_block_id,
                    target_anchor_id=anchor_id,
                    matched_text=match.group(0),
                    kind=ReferenceKind.EQUATION,
                    span_start=match.start(),
                    span_end=match.end(),
                )
            )
        else:
            topology.unresolved_mentions.append(
                {
                    "source_block_id": source_block_id,
                    "kind": ReferenceKind.EQUATION.value,
                    "text": key,
                    "raw": match.group(0),
                }
            )

    # 6. Page references
    for match in _PROSE_PAGE_REF_RE.finditer(text):
        val = match.group(1) or match.group(2)
        if not val:
            continue
        key = val.strip().lower()
        anchor_id = topology.key_to_anchor.get((ReferenceKind.PAGE, key))
        topology.total_citations_found += 1
        if anchor_id:
            topology.resolved_citations_count += 1
            links.append(
                CitationLink(
                    source_block_id=source_block_id,
                    target_anchor_id=anchor_id,
                    matched_text=match.group(0),
                    kind=ReferenceKind.PAGE,
                    span_start=match.start(),
                    span_end=match.end(),
                )
            )
        else:
            topology.unresolved_mentions.append(
                {
                    "source_block_id": source_block_id,
                    "kind": ReferenceKind.PAGE.value,
                    "text": key,
                    "raw": match.group(0),
                }
            )

    # 7. Listing references
    for match in _PROSE_LISTING_REF_RE.finditer(text):
        val = match.group(1) or match.group(2)
        if not val:
            continue
        key = val.strip().lower()
        anchor_id = topology.key_to_anchor.get((ReferenceKind.LISTING, key))
        topology.total_citations_found += 1
        if anchor_id:
            topology.resolved_citations_count += 1
            links.append(
                CitationLink(
                    source_block_id=source_block_id,
                    target_anchor_id=anchor_id,
                    matched_text=match.group(0),
                    kind=ReferenceKind.LISTING,
                    span_start=match.start(),
                    span_end=match.end(),
                )
            )
        else:
            topology.unresolved_mentions.append(
                {
                    "source_block_id": source_block_id,
                    "kind": ReferenceKind.LISTING.value,
                    "text": key,
                    "raw": match.group(0),
                }
            )

    # 8. Algorithm references
    for match in _PROSE_ALGORITHM_REF_RE.finditer(text):
        val = match.group(1) or match.group(2)
        if not val:
            continue
        key = val.strip().lower()
        anchor_id = topology.key_to_anchor.get((ReferenceKind.ALGORITHM, key))
        topology.total_citations_found += 1
        if anchor_id:
            topology.resolved_citations_count += 1
            links.append(
                CitationLink(
                    source_block_id=source_block_id,
                    target_anchor_id=anchor_id,
                    matched_text=match.group(0),
                    kind=ReferenceKind.ALGORITHM,
                    span_start=match.start(),
                    span_end=match.end(),
                )
            )
        else:
            topology.unresolved_mentions.append(
                {
                    "source_block_id": source_block_id,
                    "kind": ReferenceKind.ALGORITHM.value,
                    "text": key,
                    "raw": match.group(0),
                }
            )

    return links
