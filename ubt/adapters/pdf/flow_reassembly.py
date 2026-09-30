"""Evidence-based flow reassembly for Docling extraction.

Docling's item labels are a *prior*, not ground truth. On a math-heavy paper
its layout model routinely reports algorithm listings and sentence fragments as
``SECTION_HEADER`` items, and splits one paragraph into several one-line blocks.
A fragment has no region the rigid engine can typeset its translation into, so
the engine keeps the source visible (``render:no_zone``): the reader loses a
translation, which is exactly the account the delivery contract is meant to keep
balanced.

This module repairs the *flow* before zoning. Every rule fires only on
checkable evidence; a block that is not provably a fragment is left exactly as
Docling labelled it:

1. **Ground-truth typography** -- pdfium ``font_size`` / ``bold`` for the lines
   under each prose block is stamped onto ``block.style`` / ``provenance``. This
   is the page's real typography, independent of Docling's guess.
2. **Listings** -- text carrying unambiguous program syntax is moved to
   ``BlockType.CODE`` (preserved verbatim). A listing is not prose; leaving it
   to the rigid engine as a "heading" is what produced the loss.
3. **Heading legitimacy** -- a ``HEADING`` that is *smaller than the body text*
   it heads, or that is a lowercase/math/sentence fragment, is demoted to
   ``NARRATIVE``. A heading cannot be smaller than its body, and prose fragments
   are not titles; these are invariant facts, not tuned thresholds.
4. **Fragment coalescing** -- adjacent prose fragments in the same column within
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

from ubt.core.ir.models import (
    BlockType,
    BoundingBox,
    FlowID,
    IRBlock,
    LayoutRole,
    StyleMeta,
)
from ubt.core.policy.layout_policy import PROSE_BLOCK_TYPES

if TYPE_CHECKING:
    from ubt.adapters.pdf.textgeom import LineBox

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Listing detection
# --------------------------------------------------------------------------- #
# Unambiguous program/algorithm syntax. Deliberately narrow: a false positive
# silently stops a real paragraph from being translated, so only forms that do
# not occur in running prose qualify. A listed line almost always carries an
# assignment arrow or a definition/call keyword; weaker hints ("for ... (",
# "obj.method(") were removed because prose ("for the number ... (No deadlock.)")
# matched them and was silently kept untranslated.
_LISTING_FORMS = re.compile(
    r"←|⟵|↤|▷"  # assignment / dataflow / comment markers
    r"|\bawait\s+\w[\w.]*\s*\("  # await call(...)
    r"|\bdef\s+\w+\s*\(|\bclass\s+\w+\s*[:\(]|\bfunction\s+\w+\s*\("
    r"|\bfrom\s+[\w.]+\s+import\b"
    r"|\b(?:else|elif|repeat|until)\s+\d{1,3}\b"  # bare statement + line number
)
_LISTING_MAX_CHARS = 400

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

#: A bare page number (arabic or roman) is chrome, never content.
_BARE_PAGE_NUMBER = re.compile(r"^\s*(?:\d{1,4}|[ivxlcdm]{1,7})\s*$", re.IGNORECASE)

# --------------------------------------------------------------------------- #
# Non-translatable math / algorithm debris
# --------------------------------------------------------------------------- #
_MATH_SYMBOL_CHARS = frozenset("←⟵↤↦∘≔≃≅≤≥≠⊤⊥⊢⊣∈∉⊆⊂∪∩∀∃∧∨¬≡∑∏√∫∞∂∇⋯⋃⋂′″⟨⟩")
#: Algorithm/typing tokens: ``L-Iter``, ``pr 1``, ``id Γ``. These are names, not
#: prose; translating them is meaningless and painting over them corrupts the
#: listing, so they are preserved.
_ALG_TOKEN = re.compile(
    r"^(?:[A-Z][A-Za-z]*[-_][A-Z][a-z]+"
    r"|[a-z]{1,3}\s+\d{1,3}"
    r"|[a-z]{1,3}\s*[\U0001D400-\U0001D7FFΓΔΘΛΞΠΣΦΨΩ])$"
)


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
def _page_lines(pdf_path: Path, page: int, cache: dict[int, list[LineBox]]) -> list[LineBox]:
    lines = cache.get(page)
    if lines is None:
        from ubt.adapters.pdf.textgeom import extract_lines

        try:
            lines = extract_lines(pdf_path, page)[0]
        except Exception as exc:  # extraction must never break the pipeline
            logger.debug("flow reassembly: line extraction failed on page %d: %s", page, exc)
            lines = []
        cache[page] = lines
    return lines


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


def stamp_ground_truth_typography(blocks: list[IRBlock], pdf_path: Path | None) -> dict[int, float]:
    """Stamp pdfium font size / weight on every prose block; return page body sizes."""
    if pdf_path is None:
        return {}
    pages = sorted({b.bbox.page for b in blocks if b.bbox and b.bbox.page > 0})
    if not pages:
        return {}
    from ubt.adapters.pdf.textgeom import aggregate_line_styles

    cache: dict[int, list[LineBox]] = {}
    for page in pages:
        _page_lines(pdf_path, page, cache)
    body = _body_size_by_page(cache)

    for block in blocks:
        if block.bbox is None or block.block_type not in PROSE_BLOCK_TYPES:
            continue
        bbox = block.bbox
        lines = [
            ln
            for ln in cache.get(bbox.page, [])
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
    return body


# --------------------------------------------------------------------------- #
# Step 2 -- listings
# --------------------------------------------------------------------------- #
def _looks_like_listing(text: str) -> bool:
    body = (text or "").strip()
    if not body or len(body) > _LISTING_MAX_CHARS:
        return False
    return bool(_LISTING_FORMS.search(body))


def classify_chrome_blocks(blocks: list[IRBlock]) -> int:
    """Mark bare page numbers as preserved chrome (never translated or merged).

    The plain-text extractors have no page-furniture model, so a bare page
    number arrives as prose -- and, lacking terminal punctuation, as a heading.
    Left in the flow it is translated, and the fragment coalescer would then
    anchor a whole page onto it. It is removed before any of that happens.
    """
    marked = 0
    for block in blocks:
        if block.block_type not in PROSE_BLOCK_TYPES or block.skip_translate:
            continue
        if _BARE_PAGE_NUMBER.match(block.source_text or ""):
            block.skip_translate = True
            block.layout_role = LayoutRole.PAGE_NUMBER
            block.provenance["flow_reassembly"] = "page_number"
            marked += 1
    return marked


def _has_math(text: str) -> bool:
    return any(ch in _MATH_SYMBOL_CHARS or 0x1D400 <= ord(ch) <= 0x1D7FF for ch in text)


def _looks_like_debris(text: str) -> bool:
    """True for a short math/algorithm token that is not prose.

    Axiom B preserves non-translatable content explicitly: these are names,
    operators and equation fragments the extractor typed as prose. Translating
    them yields nonsense and repainting them corrupts the listing, so they are
    held byte-identical instead. A sentence fragment (terminal punctuation) is
    never debris -- it belongs to a paragraph and must be translated.
    """
    body = (text or "").strip()
    if not body or len(body) > 80 or body.endswith((".", "!", "?", "。")):
        return False
    if _ALG_TOKEN.fullmatch(body):
        return True
    tokens = body.split()
    if len(tokens) > 10:
        return False
    wordy = sum(1 for w in tokens if w.isalpha() and len(w) >= 3)
    if wordy <= 1 and (body.endswith(("=", "→", "↦", "≔", "<", ">")) or _has_math(body)):
        return True
    return bool(wordy <= 2 and len(tokens) <= 6 and _has_math(body))


def classify_debris_blocks(blocks: list[IRBlock]) -> int:
    """Hold short math/algorithm tokens byte-identical (preserved, not prose)."""
    marked = 0
    for block in blocks:
        if block.block_type not in PROSE_BLOCK_TYPES or block.skip_translate:
            continue
        if _looks_like_debris(block.source_text or ""):
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
        if _looks_like_listing(block.source_text or ""):
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
        body_by_page = stamp_ground_truth_typography(blocks, pdf_path)
        stats.chrome = classify_chrome_blocks(blocks)
        stats.debris = classify_debris_blocks(blocks)
        stats.listings = classify_listing_blocks(blocks)
        stats.demoted_headings = demote_fragment_headings(blocks, body_by_page)
        stats.coalesced = coalesce_flow_fragments(
            blocks, allow_cross_page=allow_cross_page, body_by_page=body_by_page
        )
    except Exception as exc:  # never let extraction repair sink extraction
        logger.warning("flow reassembly skipped after error: %s", exc)
        return blocks, stats
    if stats.chrome or stats.debris or stats.listings or stats.demoted_headings or stats.coalesced:
        logger.info("%s", stats.summary())
    return blocks, stats


__all__ = [
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
