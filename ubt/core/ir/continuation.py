"""Continuation runs: prose blocks that are one element split across boxes.

A born-digital PDF has no notion of "paragraph": a paragraph crossing a page
boundary (or a column break) is extracted as two or more blocks. This module
detects those runs from already-parsed IR blocks by reading order and the
extractors' own signals -- same flow and region, the earlier block not ending a
sentence, and the later block starting a new page *or a new column* (with a
lowercase start, for Latin). The composite renderer then flows one translation
across the chain.

The heuristic is deliberately conservative: it only ever *joins* blocks that
already read as one, and every consumer treats a missing run as the ordinary
one-box-per-block case.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Sequence
from dataclasses import dataclass

from ubt.core.cjk_ranges import is_cjk_wide_char
from ubt.core.ir.bifurcation import SEMANTIC_BREAK_FLAG
from ubt.core.ir.models import (
    BlockType,
    BoundingBox,
    IRBlock,
    _with_element_source,
)
from ubt.model.ast import RegionKind
from ubt.model.span import CompositeSpan, PhysicalBox

#: Block types that carry flowing prose (the ones a paragraph can continue into).
_PROSE_TYPES: frozenset[BlockType] = frozenset(
    {BlockType.NARRATIVE, BlockType.DIALOGUE, BlockType.LIST_ITEM}
)
#: End-of-sentence punctuation: a block ending on one of these is complete.
_TERMINAL = ".!?。！？…:;：；"
#: Closing marks that may trail the terminal punctuation.
_CLOSERS = "\"')]}」』》】”’）〕»"
#: Common abbreviations that end in a dot but do not terminate a sentence.
_ABBREVIATIONS = ("e.g.", "i.e.", "et al.", "fig.", "vs.", "dr.", "prof.", "etc.")
#: Slack (pt) absorbed by line-box rounding in the column-jump geometry test.
_COLUMN_TOL = 2.0
#: Patterns indicating a new list item or bullet (not a continuation).
_LIST_MARKER_RE = re.compile(
    r"^(\(\w+\)|\[\w+\]|\d+[\.\)]|[a-zA-Z][\.\)]|[\u2022\u25e6\u2023\-\*\u00b7\u2013])\s*"
)
#: Bare number or roman numeral (e.g. standalone page number or section number).
_BARE_NUMBER_RE = re.compile(r"^(\d+|[ivxlcdm]+)$", re.IGNORECASE)
#: Regions representing page furniture.
_FURNITURE_REGIONS = frozenset(
    {
        RegionKind.HEADER,
        RegionKind.FOOTER,
        RegionKind.PAGE_NUMBER,
        "header",
        "footer",
        "page_number",
    }
)


@dataclass(frozen=True, slots=True)
class ContinuationRun:
    """One element's reading-order chain: block ids plus the boxes they occupy."""

    block_ids: tuple[str, ...]
    boxes: tuple[PhysicalBox, ...]

    @property
    def is_composite(self) -> bool:
        return len(self.block_ids) > 1


def join_continuous_text(pieces: Sequence[str]) -> str:
    """Join text fragments of a continuation run.

    Inserts a space between pieces unless the boundary connects two CJK
    characters (ideographs, punctuation, kana, or hangul), avoiding unnatural
    gaps in Chinese/Japanese prose.
    """
    clean = [p.strip() for p in pieces if p and p.strip()]
    if not clean:
        return ""
    out = clean[0]
    for piece in clean[1:]:
        if out and piece and is_cjk_wide_char(out[-1]) and is_cjk_wide_char(piece[0]):
            out += piece
        elif out.endswith("-") and piece[:1].isalpha():
            # A page-final hard hyphen must not grow a space ("atten- tion");
            # keep the hyphen rather than dehyphenate, which would corrupt
            # real compounds ("well-" + "known").
            out += piece
        else:
            out = f"{out} {piece}"
    return out


def _ends_sentence(text: str) -> bool:
    stripped = text.rstrip()
    while stripped and stripped[-1] in _CLOSERS:
        stripped = stripped[:-1]
    if not stripped or stripped[-1] not in _TERMINAL:
        return False
    lower = stripped.lower()
    return not any(lower.endswith(abbr) for abbr in _ABBREVIATIONS)


def _is_furniture(block: IRBlock) -> bool:
    """True for page furniture (header, footer, page number) that should not break prose flow."""
    if block.region in _FURNITURE_REGIONS:
        return True
    text = block.source_text.strip()
    return bool(_BARE_NUMBER_RE.match(text))


def _candidate(block: IRBlock) -> bool:
    """A block that can start or extend a run: flowing prose with real geometry."""
    if block.block_type not in _PROSE_TYPES or block.skip_translate:
        return False
    # A block a semantic break split out is *not* a continuation: the break is
    # the decision that the layout parser glued separate passages together.
    # Re-fusing here would undo the bifurcation before the compositor sees it.
    if SEMANTIC_BREAK_FLAG in block.error_flags or "bifurcated_from" in block.provenance:
        return False
    text = block.source_text.strip()
    if not text or _BARE_NUMBER_RE.match(text):
        return False
    box = block.bbox
    return box is not None and box.page > 0


def _lowercase_start(text: str) -> bool:
    """False for a block that opens with a capital, a list marker, or a number.

    Case is the cheap continuation signal cased scripts give us. Proper nouns
    lose the flow but are still delivered, just not merged. ``isupper`` is
    script-agnostic, so a Cyrillic/Greek/accented-Latin capital vetoes fusion
    exactly like a Latin one; CJK has no case and is never rejected here.
    """
    stripped = text.lstrip(" \"'“‘([<{«\t\r\n")
    if not stripped:
        return False
    if _LIST_MARKER_RE.match(text.strip()):
        return False
    first = stripped[:1]
    if first.isdigit():
        return False
    return not (first.isalpha() and first.isupper())


def _column_jump(previous: BoundingBox, following: BoundingBox) -> bool:
    """True when ``following`` starts a new column to the right on the same page.

    The signature requires that ``following`` begins to the right of the
    previous block's right edge. In multi-column reading order, a continuation
    jumps from the end of the previous column to the start of the next column,
    which sits at or above the bottom of the previous column.
    """
    to_the_right = following.x0 >= previous.x1 - _COLUMN_TOL
    not_below = following.y1 >= previous.y0 - _COLUMN_TOL
    return to_the_right and not_below


def _continues(previous: IRBlock, following: IRBlock) -> bool:
    if not (_candidate(previous) and _candidate(following)):
        return False
    if previous.flow_id != following.flow_id or previous.region != following.region:
        return False
    if _ends_sentence(previous.source_text):
        return False
    prev_box = previous.bbox
    next_box = following.bbox
    assert prev_box is not None and next_box is not None  # _candidate guarantees geometry
    if not _lowercase_start(following.source_text):
        return False
    if next_box.page == prev_box.page + 1:
        return True
    if next_box.page == prev_box.page:
        return _column_jump(prev_box, next_box)
    return False


def find_continuation_runs(blocks: Sequence[IRBlock]) -> tuple[ContinuationRun, ...]:
    """Group ``blocks`` (in reading order) into multi-box continuation runs.

    An empty tuple means every block stands alone -- the ordinary case. Only runs
    of two or more blocks are returned.
    """
    runs: list[ContinuationRun] = []
    run_ids: list[str] = []
    run_boxes: list[PhysicalBox] = []
    previous: IRBlock | None = None

    def flush() -> None:
        if len(run_ids) >= 2:
            runs.append(ContinuationRun(tuple(run_ids), tuple(run_boxes)))

    for block in blocks:
        if _is_furniture(block):
            # Page furniture between pages does not break continuation of an ongoing paragraph.
            continue
        if not _candidate(block):
            flush()
            run_ids, run_boxes, previous = [], [], None
            continue
        bbox = block.bbox
        assert bbox is not None  # _candidate guarantees geometry
        box = PhysicalBox.of(bbox.page, (bbox.x0, bbox.y0, bbox.x1, bbox.y1))
        if previous is not None and _continues(previous, block):
            run_ids.append(block.id)
            run_boxes.append(box)
        else:
            flush()
            run_ids, run_boxes = [block.id], [box]
        previous = block

    flush()
    return tuple(runs)


def fuse_continuation_blocks(blocks: Sequence[IRBlock]) -> list[IRBlock]:
    """Fuse continuation runs into single IRBlocks carrying CompositeSpan.

    When adjacent blocks form an unbroken prose continuation (e.g. crossing a
    page break or column break), they are merged into a single semantic element
    so that translation, QE, and validators process the unbroken paragraph as a
    whole. The resulting block carries a CompositeSpan containing all physical
    boxes in reading order, enabling BreakageSolver to flow the translation at
    render time.
    """
    materialized = list(blocks)
    runs = find_continuation_runs(materialized)
    if not runs:
        return materialized

    run_by_first_id: dict[str, ContinuationRun] = {run.block_ids[0]: run for run in runs}
    subsumed_ids: set[str] = {block_id for run in runs for block_id in run.block_ids[1:]}
    by_id: dict[str, IRBlock] = {b.id: b for b in materialized}

    output: list[IRBlock] = []
    for block in materialized:
        if block.id in subsumed_ids:
            continue
        run = run_by_first_id.get(block.id)
        if run is None:
            output.append(block)
            continue

        run_blocks = [by_id[b_id] for b_id in run.block_ids if b_id in by_id]
        if len(run_blocks) < 2:
            output.append(block)
            continue

        joined_source = join_continuous_text([b.source_text for b in run_blocks])
        joined_target = join_continuous_text([b.target_text for b in run_blocks if b.target_text])

        elem = _with_element_source(block.element, joined_source)
        elem = dataclasses.replace(elem, span=CompositeSpan(boxes=run.boxes))

        fused = IRBlock(element=elem)
        fused.provenance = {
            **block.provenance,
            "fused_block_ids": list(run.block_ids),
            "fused_sources": [b.source_text for b in run_blocks],
            # The box chain must survive the ledger round-trip: ``_row_to_block``
            # rebuilds the CompositeSpan from ``physical_boxes`` alone, so without
            # this the fused paragraph comes back as a single first-box overlay and
            # its whole target is squeezed into one line (a jarring shrink).
            "physical_boxes": [{"page": box.page, "bbox": list(box.bbox)} for box in run.boxes],
        }
        if joined_target:
            fused.target_text = joined_target
        output.append(fused)

    return output


__all__ = [
    "ContinuationRun",
    "find_continuation_runs",
    "fuse_continuation_blocks",
    "join_continuous_text",
]
