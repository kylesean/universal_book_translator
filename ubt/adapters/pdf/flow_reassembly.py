"""Evidence-based flow repair: chrome, debris, listings, headings, coalescing.

Extraction is a *prior*, not ground truth -- whether it comes from Docling's
layout model or the native reader's geometric grouping. Both can report
algorithm listings and sentence fragments as prose, split one paragraph into
several one-line blocks, or surface page furniture as body text. A fragment has
no region the rigid engine can typeset its translation into, so the engine keeps
the source visible (``render:no_zone``): the reader loses a translation, which is
exactly the account the delivery contract is meant to keep balanced.

This module repairs the *flow* before zoning. Every rule fires only on
checkable evidence; a block that is not provably a fragment is left exactly as
extraction typed it:

1. **Ground-truth typography** -- pdfium ``font_size`` / ``bold`` for the lines
   under each prose block is stamped onto ``block.style`` / ``provenance``, and
   the page's body size and height are measured. This is the page's real
   typography and geometry, independent of the extractor's guess.
2. **Chrome** -- a bare page number is preserved (never translated or merged)
   only when it sits in the page's top or bottom margin band. A bare number
   *inside* the body is content (a table cell, a TOC leader, an index entry).
3. **Math debris** -- a short token that is a name, operator or equation
   fragment is held byte-identical rather than translated into nonsense.
4. **Listings** -- text carrying unambiguous program syntax is moved to
   ``BlockType.CODE`` (preserved verbatim). A listing is not prose; leaving it
   to the rigid engine as a "heading" is what produced the loss.
5. **Heading legitimacy** -- a ``HEADING`` that is *smaller than the body text*
   it heads, or that is a lowercase/math/sentence fragment, is demoted to
   ``NARRATIVE``. A heading cannot be smaller than its body, and prose fragments
   are not titles; these are invariant facts, not tuned thresholds.
6. **Fragment coalescing** -- adjacent prose fragments in the same column within
   normal leading, where the second continues the first (lowercase opener, open
   sentence, hyphenation), are reunited into one translation unit.

The pass is fail-open: any measurement problem leaves the block list untouched.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ubt.analyze.structure import is_bare_page_number, looks_like_debris, looks_like_listing
from ubt.core.ir.models import (
    BlockType,
    BoundingBox,
    FlowID,
    IRBlock,
    StyleMeta,
)
from ubt.core.policy.layout_policy import FOOTER_BAND_PT, HEADER_BAND_PT, PROSE_BLOCK_TYPES
from ubt.model.ast import RegionKind

if TYPE_CHECKING:
    from ubt.adapters.pdf.textgeom import LineBox

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Heading legitimacy
# --------------------------------------------------------------------------- #
#: Heading shapes that are always genuine, whatever the typography.
_NUMBERED_HEADING = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?|Chapter\s+\d+|[IVXLC]+\.)\s+\S", re.IGNORECASE
)
_STRUCTURAL_HEADING = re.compile(
    r"^\s*(?:abstract|contents|references|bibliography|acknowledge?ments?|appendix|"
    r"introduction|conclusion|preface|foreword|index|glossary|summary)\s*$",
    re.IGNORECASE,
)
#: Run-in result/theorem labels are statements, not titles: "Definition 18.
#: Define ..." heads a paragraph and must be translated as prose.
_RUNIN_STATEMENT = re.compile(
    r"^\s*(?:definition|theorem|lemma|proof|corollary|proposition|claim|remark|"
    r"example|algorithm|figure|table)\b",
    re.IGNORECASE,
)
#: A heading is never smaller than the body text it heads (ratio < 1 by mark).
_HEADING_MIN_SIZE_RATIO = 0.95
#: A heading is at least this much larger than body, or it must be bold /
#: numbered / structurally named to count as a heading.
_HEADING_SIZE_RATIO = 1.08
#: Inline math marks a sentence fragment, never a title.
_HAS_MATH = re.compile(r"[←→↦∘ΣΓ𝔐ℑ𝑧𝜔𝜆∈∪⊥≃≤≥≠∀∃√∫∑∏]|\$")
#: Terminal punctuation that a title essentially never carries.
_TERMINAL_FRAGMENT = (".", ",", ";", "=", "，", "、", "；")
#: Above this length an unbolded, unnumbered, body-sized "heading" is really a
#: sentence (a run-in statement or a paragraph fragment).
_HEADING_MAX_WORDS = 8
_CONTINUATION_START = re.compile(
    r"^(?:[a-z]|[,;:)\]，；：）]|"
    r"(?:and|or|but|for|with|in|to|that|as|by|from|is|are|was|were|has|have|had|"
    r"where|which|when|if|of|the|we|it|this|they|then|so|thus|hence)\b)"
)
_MERGE_GAP_FACTOR = 2.0  # vertical gap allowance, in body-line heights


@dataclass
class ReassemblyStats:
    """What the pass changed, for the audit trail."""

    listings: int = 0
    demoted_headings: int = 0
    coalesced: int = 0
    chrome: int = 0
    debris: int = 0
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"flow reassembly: {self.chrome} chrome, {self.debris} math-debris, "
            f"{self.listings} listing(s), {self.demoted_headings} demoted heading(s), "
            f"{self.coalesced} coalesced fragment(s)"
        )


# --------------------------------------------------------------------------- #
# Step 1 -- ground-truth typography
# --------------------------------------------------------------------------- #
def _page_lines_and_height(
    pdf_path: Path, page: int, cache: dict[int, tuple[list[LineBox], float]]
) -> tuple[list[LineBox], float]:
    """Extract a page's lines and height once; extraction never breaks the pipeline."""
    entry = cache.get(page)
    if entry is None:
        from ubt.adapters.pdf.textgeom import extract_lines

        try:
            lines, (_width, height) = extract_lines(pdf_path, page)
        except Exception as exc:  # extraction must never break the pipeline
            logger.debug("flow reassembly: line extraction failed on page %d: %s", page, exc)
            lines, height = [], 0.0
        entry = (lines, float(height))
        cache[page] = entry
    return entry


def _body_size_by_page(lines_by_page: dict[int, list[LineBox]]) -> dict[int, float]:
    """Char-weighted modal font size of the prose lines on each page."""
    body: dict[int, float] = {}
    for page, lines in lines_by_page.items():
        weights: Counter[float] = Counter()
        for ln in lines:
            text = (ln.text or "").strip()
            if not text or ln.table_band or ln.font_size < 4.5:
                continue
            weights[round(ln.font_size, 1)] += len(text)
        if weights:
            body[page] = max(weights.items(), key=lambda kv: (kv[1], kv[0]))[0]
    return body


@dataclass
class PageTruth:
    """The page's own geometry: char-weighted body size and height, by page.

    Both are measured from the page, never guessed. ``body_size`` decides
    headings; ``height`` places page furniture (a bare number is a page number
    only inside the margin band, never mid-page).
    """

    body_size: dict[int, float] = field(default_factory=dict)
    height: dict[int, float] = field(default_factory=dict)


def stamp_ground_truth_typography(blocks: list[IRBlock], pdf_path: Path | None) -> PageTruth:
    """Stamp pdfium font size / weight on every prose block; return the page truth."""
    truth = PageTruth()
    if pdf_path is None:
        return truth
    pages = sorted({b.bbox.page for b in blocks if b.bbox and b.bbox.page > 0})
    if not pages:
        return truth
    from ubt.adapters.pdf.textgeom import aggregate_line_styles

    cache: dict[int, tuple[list[LineBox], float]] = {}
    lines_by_page: dict[int, list[LineBox]] = {}
    for page in pages:
        lines, height = _page_lines_and_height(pdf_path, page, cache)
        lines_by_page[page] = lines
        truth.height[page] = height
    truth.body_size = _body_size_by_page(lines_by_page)

    for block in blocks:
        if block.bbox is None or block.block_type not in PROSE_BLOCK_TYPES:
            continue
        bbox = block.bbox
        lines = [
            ln
            for ln in lines_by_page.get(bbox.page, [])
            if ln.rect[3] > bbox.y0 - 1
            and ln.rect[1] < bbox.y1 + 1
            and ln.rect[0] < bbox.x1 + 1
            and ln.rect[2] > bbox.x0 - 1
        ]
        if not lines:
            continue
        size, bold, italic = aggregate_line_styles(lines)
        if size > 0:
            block.style = StyleMeta(font_size=size)
        block.provenance["gt_bold"] = bold
        block.provenance["gt_italic"] = italic
    return truth


# --------------------------------------------------------------------------- #
# Step 2 -- chrome
# --------------------------------------------------------------------------- #
def _in_margin_band(bbox: BoundingBox | None, page_heights: dict[int, float]) -> bool:
    """True when a box sits in the page's top or bottom margin band."""
    if bbox is None:
        return False
    height = page_heights.get(bbox.page, 0.0)
    if height <= 0:
        return False
    return bbox.y0 < FOOTER_BAND_PT or bbox.y1 > height - HEADER_BAND_PT


def classify_chrome_blocks(blocks: list[IRBlock], page_heights: dict[int, float]) -> int:
    """Mark bare page numbers *in the page margins* as preserved chrome.

    The plain-text extractors have no page-furniture model, so a page number
    arrives as prose -- and, lacking terminal punctuation, as a heading. Left in
    the flow it is translated, and the fragment coalescer would then anchor a
    whole page onto it.

    A bare number is page furniture only where page furniture lives: the top or
    bottom margin band. A bare number *inside* the text body is content -- a
    table cell, a TOC leader, an index entry -- and is left alone. When the page
    height is unknown the number cannot be placed, so it is left alone too: the
    safe direction is to translate it, never to keep it silently.
    """
    marked = 0
    for block in blocks:
        if block.block_type not in PROSE_BLOCK_TYPES or block.skip_translate:
            continue
        if not is_bare_page_number(block.source_text or ""):
            continue
        if not _in_margin_band(block.bbox, page_heights):
            continue
        block.skip_translate = True
        block.layout_role = RegionKind.PAGE_NUMBER
        block.provenance["flow_reassembly"] = "page_number"
        marked += 1
    return marked


def classify_debris_blocks(blocks: list[IRBlock]) -> int:
    """Hold short math/algorithm tokens byte-identical (preserved, not prose)."""
    marked = 0
    for block in blocks:
        if block.block_type not in PROSE_BLOCK_TYPES or block.skip_translate:
            continue
        if looks_like_debris(block.source_text or ""):
            block.block_type = BlockType.FORMULA
            block.skip_translate = True
            block.provenance["flow_reassembly"] = "math_debris"
            marked += 1
    return marked


def classify_listing_blocks(blocks: list[IRBlock]) -> int:
    """Move unambiguous algorithm/listing blocks to ``CODE`` (preserved)."""
    marked = 0
    for block in blocks:
        if block.block_type not in PROSE_BLOCK_TYPES or block.skip_translate:
            continue
        if looks_like_listing(block.source_text or ""):
            block.block_type = BlockType.CODE
            block.skip_translate = True
            block.provenance["flow_reassembly"] = "listing"
            marked += 1
    return marked


# --------------------------------------------------------------------------- #
# Step 3 -- heading legitimacy
# --------------------------------------------------------------------------- #
def _looks_title_shape(text: str) -> bool:
    words = text.split()
    if not words or len(words) > 12:
        return False
    content = [w for w in words if w[:1].isupper() and w.isalpha()]
    return len(content) >= max(1, len(words) // 2)


def heading_is_fragment(
    text: str, *, size: float | None, body_size: float | None, bold: bool
) -> bool:
    """True when a Docling ``SECTION_HEADER`` cannot be a real heading.

    Returns False (trust Docling) unless there is positive evidence that the
    block is a fragment. Explicit heading shapes and typographically-distinct
    lines are always trusted.
    """
    body = " ".join((text or "").split())
    if not body:
        return False
    if _NUMBERED_HEADING.match(body) or _STRUCTURAL_HEADING.match(body) or bold:
        return False
    if size and body_size and size >= body_size * _HEADING_SIZE_RATIO:
        return False
    # A heading is never smaller than its body: provably a fragment.
    if size and body_size and size < body_size * _HEADING_MIN_SIZE_RATIO:
        return True
    # Body-sized, unbolded, unnumbered: demote only on fragment evidence.
    if _HAS_MATH.search(body):
        return True
    if body[:1].islower() or not body[:1].isalpha():
        return True
    if body.endswith(_TERMINAL_FRAGMENT) or body.endswith("-"):
        return True
    if _RUNIN_STATEMENT.match(body) and len(body.split()) >= 4:
        return True
    words = body.split()
    if len(words) > _HEADING_MAX_WORDS and not _looks_title_shape(body):
        return True
    return bool(body.endswith(":") and len(words) >= 4)


def demote_fragment_headings(blocks: list[IRBlock], body_by_page: dict[int, float]) -> int:
    """Demote ``MAIN_STORY`` headings that are provably not headings."""
    demoted = 0
    for block in blocks:
        if (
            block.block_type != BlockType.HEADING
            or block.flow_id != FlowID.MAIN_STORY
            or block.skip_translate
            or block.provenance.get("toc_entry")
        ):
            continue
        body_size = None
        if block.bbox is not None:
            body_size = body_by_page.get(block.bbox.page)
        size = block.style.font_size if block.style else None
        if heading_is_fragment(
            block.source_text or "",
            size=size,
            body_size=body_size,
            bold=bool(block.provenance.get("gt_bold")),
        ):
            block.block_type = BlockType.NARRATIVE
            block.provenance["flow_reassembly"] = "demoted_heading"
            demoted += 1
    return demoted


# --------------------------------------------------------------------------- #
# Step 4 -- fragment coalescing
# --------------------------------------------------------------------------- #
def _is_flow_prose(block: IRBlock) -> bool:
    return (
        block.block_type == BlockType.NARRATIVE
        and block.flow_id == FlowID.MAIN_STORY
        and not block.skip_translate
        and not block.provenance.get("toc_entry")
        and block.bbox is not None
    )


def _same_column(a: BoundingBox, b: BoundingBox) -> bool:
    overlap = min(a.x1, b.x1) - max(a.x0, b.x0)
    narrower = min(a.x1 - a.x0, b.x1 - b.x0)
    return overlap > 0.3 * max(narrower, 1e-3)


def _textually_continuous(curr_text: str, next_text: str) -> bool:
    """Whether ``next_text`` continues ``curr_text`` within one paragraph.

    Deliberately strict: the plain-text extractors already emit fragments, so a
    weak test ("current does not end in punctuation") folds a whole page into
    one block. A merge needs *positive* proof of mid-sentence continuation.
    """
    if curr_text.endswith("-"):
        return True  # hard-hyphenated line break
    if curr_text.endswith((",", ";", "，", "；")):
        return True  # clause continues
    # The next unit opens mid-sentence: it cannot start a paragraph.
    return bool(next_text[:1].islower() or _CONTINUATION_START.match(next_text))


def _merge_into(target: IRBlock, fragment: IRBlock) -> None:
    left = (target.source_text or "").strip()
    right = (fragment.source_text or "").strip()
    if left.endswith("-"):
        target.source_text = left[:-1] + right
    else:
        target.source_text = f"{left} {right}".strip()
    if target.bbox is not None and fragment.bbox is not None:
        target.bbox = BoundingBox(
            page=target.bbox.page,
            x0=min(target.bbox.x0, fragment.bbox.x0),
            y0=min(target.bbox.y0, fragment.bbox.y0),
            x1=max(target.bbox.x1, fragment.bbox.x1),
            y1=max(target.bbox.y1, fragment.bbox.y1),
        )


def coalesce_flow_fragments(
    blocks: list[IRBlock],
    *,
    allow_cross_page: bool = True,
    body_by_page: dict[int, float] | None = None,
) -> int:
    """Reunite adjacent prose fragments in the same column into one unit."""
    body_by_page = body_by_page or {}
    merged = 0
    i = 0
    while i < len(blocks) - 1:
        curr, nxt = blocks[i], blocks[i + 1]
        if not _is_flow_prose(curr) or not _is_flow_prose(nxt):
            i += 1
            continue
        assert curr.bbox is not None and nxt.bbox is not None
        same_page = curr.bbox.page == nxt.bbox.page
        if not same_page and not allow_cross_page:
            i += 1
            continue
        if same_page:
            if not _same_column(curr.bbox, nxt.bbox):
                i += 1
                continue
            body = body_by_page.get(curr.bbox.page, 0.0)
            gap_limit = max(_MERGE_GAP_FACTOR * body, 12.0)
            gap = curr.bbox.y0 - nxt.bbox.y1
            if gap < -2.0 or gap > gap_limit:
                i += 1
                continue
        if not _textually_continuous(
            (curr.source_text or "").strip(), (nxt.source_text or "").strip()
        ):
            i += 1
            continue
        _merge_into(curr, nxt)
        del blocks[i + 1]
        merged += 1
        # Stay on ``i`` so a run of fragments folds into one unit.
    return merged


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def reassemble_flow(
    blocks: list[IRBlock],
    *,
    pdf_path: Path | None,
    allow_cross_page: bool = True,
) -> tuple[list[IRBlock], ReassemblyStats]:
    """Run the reassembly passes in order; fail-open on any error."""
    stats = ReassemblyStats()
    try:
        truth = stamp_ground_truth_typography(blocks, pdf_path)
        stats.chrome = classify_chrome_blocks(blocks, truth.height)
        stats.debris = classify_debris_blocks(blocks)
        stats.listings = classify_listing_blocks(blocks)
        stats.demoted_headings = demote_fragment_headings(blocks, truth.body_size)
        stats.coalesced = coalesce_flow_fragments(
            blocks, allow_cross_page=allow_cross_page, body_by_page=truth.body_size
        )
    except Exception as exc:  # never let extraction repair sink extraction
        logger.warning("flow reassembly skipped after error: %s", exc)
        return blocks, stats
    if stats.chrome or stats.debris or stats.listings or stats.demoted_headings or stats.coalesced:
        logger.info("%s", stats.summary())
    return blocks, stats


__all__ = [
    "PageTruth",
    "ReassemblyStats",
    "classify_chrome_blocks",
    "classify_debris_blocks",
    "classify_listing_blocks",
    "coalesce_flow_fragments",
    "demote_fragment_headings",
    "heading_is_fragment",
    "reassemble_flow",
    "stamp_ground_truth_typography",
]
