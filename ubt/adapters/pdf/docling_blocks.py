"""Pure IRBlock shaping helpers for the Docling adapter.

These are the Docling adapter's raw item helpers: read a Docling item's own
provenance (page spans, geometry) and render its grid to markup. They never
classify or repair -- the analyzer types directly from Docling's labels in
:func:`ubt.adapters.pdf.docling_parser.extract_with_docling` (native analyzer AST type production).
The caption fuse/decouple/latch heuristics that used to live here are deleted:
Docling labels captions (``CAPTION``, ``PICTURE``, a ``FIG. N`` title), and the
analyzer trusts those labels rather than re-guessing the boundaries afterwards.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from ubt.analyze.structure import looks_like_debris, looks_like_listing
from ubt.core.ir.models import BlockType, BoundingBox, IRBlock

logger = logging.getLogger(__name__)

#: Unambiguous tabular/code punctuation for absorbing a table's stray bottom
#: fragments. A bare ":" was removed: it matched ordinary prose.
_TABULAR_MARKERS = ("->", "←→", "| None", "tuple[", "()", "|")

#: A single code-shaped token: CamelCase (``ScheduleSource``), snake_case
#: (``next_fire``), or carrying a digit. Anchored to the whole fragment, so any
#: whitespace disqualifies it — a plain one-word prose fragment is never
#: absorbed, while a stray table signature still is.
_CODE_IDENTIFIER_RE = re.compile(
    r"^(?:[A-Za-z_][A-Za-z0-9_]*[a-z][A-Z][A-Za-z0-9_]*"
    r"|[A-Za-z]+_[A-Za-z0-9_]+"
    r"|[A-Za-z_][A-Za-z0-9_]*\d[A-Za-z0-9_]*)$"
)

_DIGIT_RUN_RE = re.compile(r"\d+")
_BARE_HANDLE_RE = re.compile(r"^@[\w.\-]+$")
_TOC_LEADER_LINE_RE = re.compile(
    r"^(?P<title>.+?)\s*(?:(?:\.\s*){3,}|[·…⋯]{2,})\s*(?P<page>[0-9ivxlcdmIVXLCDM]+)\s*$"
)


def parse_toc_entry_line(line_text: str) -> tuple[str, str] | None:
    """Parse a Table of Contents line with dot leaders into (title, page_number)."""
    m = _TOC_LEADER_LINE_RE.match((line_text or "").strip())
    if not m:
        return None
    title = m.group("title").strip()
    page = m.group("page").strip()
    if not title or not page:
        return None
    return title, page


def chrome_key(text: str) -> str:
    """Normalize running-head text for frequency counting.

    Running heads differ per page only by their page number
    ("Handbook 02 10" vs "Handbook 02 11"); without digit normalization
    every instance counts 1 and the >3 repeat filter never fires.
    """
    return _DIGIT_RUN_RE.sub("#", text.strip())


def is_repeat_handle(text: str, seen: set[str]) -> bool:
    """Whether a bare social-handle watermark was already kept once.

    The first occurrence (typically the cover credit line) is preserved;
    repeats are running chrome and must not become content blocks. Returns
    True when the caller should drop the item.
    """
    handle = text.strip()
    if _BARE_HANDLE_RE.match(handle) is None:
        return False
    if handle in seen:
        return True
    seen.add(handle)
    return False


def split_prov_spans(item: Any) -> tuple[str, str, BoundingBox | None] | None:
    """Split one Docling item whose text spans two pages.

    Docling sometimes joins a cross-page paragraph fragment with the next
    page's figure-caption body into a single text item. The item's prov
    charspans expose both regions; return ``(head_text, tail_text,
    tail_bbox)`` when the trailing span is a sentence-like caption body on
    a different page, otherwise ``None`` so the item stays untouched. This
    reads Docling's own provenance, so it is typing from evidence, not a
    post-hoc guess.
    """
    provs = list(getattr(item, "prov", []) or [])
    if len(provs) < 2:
        return None
    first, last = provs[0], provs[-1]
    if getattr(last, "page_no", None) == getattr(first, "page_no", None):
        return None
    charspan = getattr(last, "charspan", None)
    text = getattr(item, "text", "") or ""
    if not charspan or charspan[0] <= 0 or charspan[1] > len(text):
        return None
    head = text[: charspan[0]].strip()
    tail = text[charspan[0] : charspan[1]].strip()
    if not head or not tail:
        return None
    # Caption bodies are standalone sentences: capitalized opener plus
    # terminal punctuation. Mid-sentence page continuations (the other
    # multi-page span shape) match neither side of that test.
    if not tail[:1].isupper() or not tail.endswith((".", "!", "?", "。")):
        return None
    if len(tail.split()) < 5:
        return None
    prov_bbox = getattr(last, "bbox", None)
    tail_bbox = None
    if prov_bbox is not None:
        tail_bbox = BoundingBox(
            page=int(getattr(last, "page_no", 0)),
            x0=float(getattr(prov_bbox, "l", 0.0)),
            y0=float(getattr(prov_bbox, "b", 0.0)),
            x1=float(getattr(prov_bbox, "r", 0.0)),
            y1=float(getattr(prov_bbox, "t", 0.0)),
        )
    return head, tail, tail_bbox


def is_inside_picture(
    page_no: int,
    bbox: Any,
    picture_boxes: list[tuple[int, Any]],
    margin_pt: float = 2.0,
) -> bool:
    """True if bbox lies fully within any collected picture bbox (same page)."""
    for pno, pbox in picture_boxes:
        if pno != page_no:
            continue
        if (
            pbox.l - margin_pt <= bbox.l
            and bbox.r <= pbox.r + margin_pt
            and pbox.b - margin_pt <= bbox.b
            and bbox.t <= pbox.t + margin_pt
        ):
            return True
    return False


def table_to_markdown(table_item: Any, doc: Any = None) -> str:
    """Render a Docling TableItem as markdown with proper merged/colspan cell handling.

    For merged cells (spanning multiple columns or rows), only the top-left primary cell
    emits text; continuation cells emit empty strings to preserve grid dimensions without
    duplicating text across cells or corrupting downstream text/QE.
    """
    grid = getattr(getattr(table_item, "data", None), "grid", None) or []
    if grid:
        seen_cells: set[int] = set()
        covered_coords: set[tuple[int, int]] = set()
        rows: list[list[str]] = []
        for r, row in enumerate(grid):
            row_cells: list[str] = []
            for c, cell in enumerate(row):
                cell_id = id(cell)
                start_r = getattr(cell, "start_row_offset_idx", r)
                start_c = getattr(cell, "start_col_offset_idx", c)
                row_span = getattr(cell, "row_span", 1) or 1
                col_span = getattr(cell, "col_span", 1) or 1

                is_continuation = (
                    (r, c) in covered_coords
                    or (start_r != r or start_c != c)
                    or (cell_id in seen_cells)
                )

                if is_continuation:
                    row_cells.append("")
                else:
                    seen_cells.add(cell_id)
                    if row_span > 1 or col_span > 1:
                        for dr in range(row_span):
                            for dc in range(col_span):
                                if dr != 0 or dc != 0:
                                    covered_coords.add((r + dr, c + dc))

                    raw_text = getattr(cell, "text", "") or ""
                    clean_text = " ".join(raw_text.split()).replace("|", r"\|")
                    row_cells.append(clean_text)
            rows.append(row_cells)

        if rows and any(any(c for c in r) for r in rows):
            num_cols = max(len(r) for r in rows) if rows else 0
            if num_cols > 0:
                padded_rows = [r + [""] * (num_cols - len(r)) for r in rows]
                lines = ["| " + " | ".join(r) + " |" for r in padded_rows]
                lines.insert(1, "|" + "---|" * num_cols)
                return "\n".join(lines)

    # Fallback to export_to_markdown if grid is not available
    try:
        if hasattr(table_item, "export_to_markdown"):
            md = (
                table_item.export_to_markdown(doc)
                if doc is not None
                else table_item.export_to_markdown()
            )
            if md and str(md).strip():
                return str(md).strip()
    except Exception as exc:
        logger.debug("Table export_to_markdown failed: %s", exc)

    return ""


def resolve_overlapping_formula_blocks(blocks: list[IRBlock]) -> list[IRBlock]:
    """Merge consecutive FORMULA blocks on the same page whose bounding boxes overlap.

    Docling's layout detector occasionally emits two vertically overlapping
    FORMULA boxes for an equation paired with a commutative diagram (e.g.
    equation (7) + top half of diagram in one box, and the diagram in a second
    box). When both degrade to source-graphic witness crops, the overlapping
    region renders twice. Merging vertically overlapping formula boxes into a
    single union bounding box ensures the equation+diagram region is cropped
    and rendered exactly once.
    """
    if len(blocks) < 2:
        return blocks
    out: list[IRBlock] = []
    for b in blocks:
        if (
            out
            and b.block_type == BlockType.FORMULA
            and out[-1].block_type == BlockType.FORMULA
            and b.bbox is not None
            and out[-1].bbox is not None
            and b.bbox.page == out[-1].bbox.page
        ):
            prev_box = out[-1].bbox
            cur_box = b.bbox
            y_overlap = min(prev_box.y1, cur_box.y1) - max(prev_box.y0, cur_box.y0)
            x_overlap = min(prev_box.x1, cur_box.x1) - max(prev_box.x0, cur_box.x0)
            if y_overlap > 2.0 and x_overlap > 10.0:
                union_box = BoundingBox(
                    page=prev_box.page,
                    x0=min(prev_box.x0, cur_box.x0),
                    y0=min(prev_box.y0, cur_box.y0),
                    x1=max(prev_box.x1, cur_box.x1),
                    y1=max(prev_box.y1, cur_box.y1),
                )
                prev_txt = (out[-1].source_text or "").strip()
                cur_txt = (b.source_text or "").strip()
                merged_txt = (
                    f"{prev_txt}\n{cur_txt}".strip()
                    if cur_txt and cur_txt not in prev_txt
                    else prev_txt
                )
                out[-1].set_bbox(union_box)
                out[-1].set_source_text(merged_txt)
                if out[-1].target_text is not None:
                    out[-1].target_text = merged_txt
                continue
        out.append(b)
    return out


def merge_table_continuation_fragments(blocks: list[IRBlock]) -> list[IRBlock]:
    """Merge loose code/formula/narrative fragments immediately below a table into the table block.

    Docling occasionally fails to include the bottom row(s) of a complex table (e.g.
    method signatures, key-value rows, or formulas) into the table item, emitting
    them as a constellation of tiny formula/narrative fragments directly underneath
    the table grid. In rigid typesetting, these fragments get translated into
    narrow bboxes causing catastrophic multi-line overlapping and layout wreckage.
    Merging them into the preceding table block extends the table's protective
    envelope and preserves the entire table structure intact.
    """
    if len(blocks) < 2:
        return blocks

    out: list[IRBlock] = []
    i = 0
    n = len(blocks)

    while i < n:
        cur = blocks[i]
        out.append(cur)
        i += 1

        if cur.block_type != BlockType.TABLE or cur.bbox is None:
            continue

        tbl_box = cur.bbox
        cur_bottom = tbl_box.y0
        merged_texts: list[str] = []
        min_x0 = tbl_box.x0
        max_x1 = tbl_box.x1

        while i < n:
            nxt = blocks[i]
            nb = nxt.bbox
            if nb is None or nb.page != tbl_box.page:
                break
            # Hard barriers: never swallow another table, heading, or caption
            if nxt.block_type in (BlockType.TABLE, BlockType.HEADING):
                break
            ntxt = (nxt.source_text or "").strip()
            if ntxt.lower().startswith(("table ", "figure ", "fig. ")):
                break

            # Fragment must be below the original table top
            if nb.y1 > tbl_box.y0 + 5.0:
                break

            # Proximity to the current lowest row of the table
            # Allow same-row blocks (nb.y1 can be slightly above cur_bottom)
            gap = cur_bottom - nb.y1
            if gap > 25.0:
                break

            # Horizontal containment within the table column envelope
            if not (nb.x0 >= tbl_box.x0 - 25.0 and nb.x1 <= tbl_box.x1 + 25.0):
                break

            # Fragment characteristics: formula, code, or a tabular/code-shaped
            # phrase. The old gate also accepted any block under 120 chars or
            # one containing a bare ":", so a short body sentence directly below
            # a table ("The results are shown below:") was swallowed into the
            # table block and shipped as grid markup instead of translated
            # prose. Only unambiguous structural markers qualify now.
            is_fragment = (
                nxt.block_type in (BlockType.FORMULA, BlockType.CODE)
                or looks_like_debris(ntxt)
                or looks_like_listing(ntxt)
                or _CODE_IDENTIFIER_RE.match(ntxt) is not None
                or (len(ntxt) < 120 and any(sym in ntxt for sym in _TABULAR_MARKERS))
            )
            if not is_fragment:
                break

            # Absorb fragment
            min_x0 = min(min_x0, nb.x0)
            max_x1 = max(max_x1, nb.x1)
            cur_bottom = min(cur_bottom, nb.y0)
            if ntxt:
                merged_texts.append(ntxt)
            i += 1

        if merged_texts:
            new_box = BoundingBox(
                page=tbl_box.page,
                x0=min_x0,
                y0=cur_bottom,
                x1=max_x1,
                y1=tbl_box.y1,
            )
            cur.set_bbox(new_box)
            prev_txt = (cur.source_text or "").strip()
            extra_txt = "\n".join(merged_texts)
            merged_full = f"{prev_txt}\n{extra_txt}".strip()
            cur.set_source_text(merged_full)
            if cur.target_text is not None:
                cur.target_text = merged_full

    return out
