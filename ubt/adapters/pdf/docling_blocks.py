"""Pure IRBlock shaping helpers for the Docling adapter.

These are the Docling adapter's raw item helpers: read a Docling item's own
provenance (page spans, geometry) and render its grid to markup. They never
classify or repair -- the analyzer types directly from Docling's labels in
:func:`ubt.adapters.pdf.docling_parser.extract_with_docling` (native analyzer AST type production).
No caption fuse/decouple/latch heuristics live here: Docling labels captions
(``CAPTION``, ``PICTURE``, a ``FIG. N`` title), and the analyzer trusts those
labels rather than re-guessing the boundaries afterwards.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from ubt.analyze.structure import looks_like_debris, looks_like_listing
from ubt.core.ir.models import BlockType, BoundingBox, FlowID, IRBlock, make_element

logger = logging.getLogger(__name__)

#: Unambiguous tabular/code punctuation for absorbing a table's stray bottom
#: fragments. A bare ":" is excluded: it matches ordinary prose.
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
_TOC_SPACE_LINE_RE = re.compile(r"^(?P<title>.+?)\s+(?P<page>[0-9ivxlcdmIVXLCDM]+)\s*$")

#: A code listing Docling collapsed to ONE line is not renderable code. Docling
#: joins a listing's lines into a single space-separated stream, and the source
#: PDF frequently draws the identifier underscores as vector rules that no text
#: layer contains (the arXiv 2101.03961 pseudocode: ``load_balance_loss``
#: extracts as ``load balance loss``). Typesetting that blob as prose loses the
#: line structure and every underscore — the listing is unreadable as code.
#: The threshold is low on purpose: a real listing of any size carries line
#: breaks in any faithful text layer.
_CODE_LOSSY_MIN_CHARS = 120


def _figure_asset_dpi_tag() -> str:
    """The delivered-crop DPI tag, read lazily.

    Asset filenames carry the resolution they were cut at, so raising the DPI
    cuts new files instead of replaying old ones from the sha-keyed asset dir.
    """
    from ubt.adapters.pdf.visual_scalpel import FIGURE_ASSET_DPI_TAG

    return FIGURE_ASSET_DPI_TAG


def code_text_layer_is_lossy(text: str) -> bool:
    """True when a ``code`` block's extracted text cannot be typeset as code.

    The extractor's signal is structural, not a guess about fonts: a listing
    longer than :data:`_CODE_LOSSY_MIN_CHARS` with no line break has had its
    line structure destroyed upstream, so the only faithful representation is
    a crop of the source region.
    """
    body = (text or "").strip()
    return len(body) >= _CODE_LOSSY_MIN_CHARS and "\n" not in body


def crop_region_asset(
    item: Any,
    assets_dir: Path,
    pdf_path: Path | None,
    ordinal: int,
    *,
    kind: str,
    min_width_pt: float,
    min_height_pt: float,
) -> IRBlock | None:
    """Crop one Docling item's source region to a PNG and return an IMAGE block.

    The shared implementation of the two crop recoveries (a code listing whose
    line structure was destroyed upstream, an unreadable formula). They were two
    65-line near-copies — normalized AST similarity 0.9956 — differing only in
    the size floor, the asset name and the block-id prefix, which is exactly the
    shape that drifts: a DPI or naming change made in one place silently left the
    other behind.

    Returns ``None`` when the region cannot be cropped (no source PDF, no
    provenance bbox, or a region smaller than ``min_width_pt`` ×
    ``min_height_pt``), so the caller keeps whatever it had.
    """
    if pdf_path is None or not pdf_path.exists():
        return None
    prov = item.prov[0] if getattr(item, "prov", None) else None
    if prov is None or getattr(prov, "bbox", None) is None:
        return None
    page_no = int(getattr(prov, "page_no", 0) or 0)
    if page_no < 1:
        return None
    b = prov.bbox
    bbox = BoundingBox(
        page=page_no,
        x0=float(getattr(b, "l", 0.0)),
        y0=float(getattr(b, "b", 0.0)),
        x1=float(getattr(b, "r", 0.0)),
        y1=float(getattr(b, "t", 0.0)),
    )
    if bbox.x1 - bbox.x0 < min_width_pt or bbox.y1 - bbox.y0 < min_height_pt:
        return None
    prefix = _CROP_ASSET_PREFIXES[kind]
    asset_path = assets_dir / f"{prefix}_p{page_no}_{ordinal:04d}.{_figure_asset_dpi_tag()}.png"
    try:
        from ubt.adapters.pdf.visual_scalpel import FIGURE_ASSET_DPI, crop_block_pil

        assets_dir.mkdir(parents=True, exist_ok=True)
        # Delivered at print resolution (see FIGURE_ASSET_DPI): the crop is the
        # only raster of this region — the renderer rebuilds the page from the IR
        # and has no source canvas.
        cropped = crop_block_pil(pdf_path, page_no, bbox, dpi=FIGURE_ASSET_DPI)
        if getattr(cropped, "width", 0) < min_width_pt or getattr(cropped, "height", 0) < (
            min_height_pt
        ):
            return None
        cropped.save(asset_path)
    except Exception as exc:
        logger.debug("%s crop failed for page %d: %s", kind, page_no, exc)
        return None
    return IRBlock(
        element=make_element(
            id=f"pdf_main#{prefix}_{ordinal:04d}",
            spine_index=ordinal,
            block_type=BlockType.IMAGE,
            flow_id=FlowID.MAIN_STORY,
            source_text=str(asset_path),
            bbox=bbox,
            skip_translate=True,
        ),
        target_text=str(asset_path),
    )


#: Asset-file prefix and block-id prefix per crop kind. These are the names
#: earlier builds already wrote; changing one orphans the assets of every cached
#: conversion, so they are named here rather than inlined.
_CROP_ASSET_PREFIXES: dict[str, str] = {"code": "code", "formula": "formula"}


def crop_lossy_code_asset(
    item: Any,
    assets_dir: Path,
    pdf_path: Path | None,
    ordinal: int,
    text: str,
) -> IRBlock | None:
    """Recover a line-structure-destroyed code listing by cropping its region.

    A listing whose text layer lost its line structure must be cropped rather
    than delivered as one wrapped prose paragraph; the crop becomes an IMAGE
    block, which the renderer places and the delivery contract already counts.

    ``text`` is accepted for call-site symmetry with :func:`code_text_layer_is_lossy`
    but is not read: the decision was already made by the caller, and the size
    floor (60 × 24pt) is what rejects a one-line fragment.
    """
    return crop_region_asset(
        item,
        assets_dir,
        pdf_path,
        ordinal,
        kind="code",
        min_width_pt=60,
        min_height_pt=24,
    )


def parse_toc_entry_line(line_text: str) -> tuple[str, str, bool] | None:
    """Parse a Table of Contents line into (title, page_number, has_leaders).

    Matches both traditional leader-separated entries ("1.1 Title ... 3") and
    unleaded chapter/part headings with terminal page numbers ("1 Introduction 3",
    "Preface ix", "I Basic techniques 1").
    """
    text = (line_text or "").strip()
    if not text:
        return None
    m = _TOC_LEADER_LINE_RE.match(text)
    if m:
        title = m.group("title").strip()
        page = m.group("page").strip()
        if title and page:
            return title, page, True
    m_sp = _TOC_SPACE_LINE_RE.match(text)
    if m_sp:
        title = m_sp.group("title").strip()
        page = m_sp.group("page").strip()
        if title and page:
            return title, page, False
    return None


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
    # A head that does not *end* a sentence means the tail continues it, not a
    # caption: "…we evaluate RT-" + "DocLayout using downstream benchmarks."
    # is one paragraph split across the page break (measured on arXiv
    # 2606.23344, where the fragment then absorbed the neighbouring table's
    # content in translation). Only a head that ends cleanly is split; a head
    # with a dangling hyphen or no terminal punctuation keeps the item whole.
    if not head.endswith((".", "!", "?", "。")):
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


#: A figure label may hang a few points past the plot area's edge (an axis
#: title's baseline sits just below it); this share of the block's own area
#: overlapping the picture is enough to anchor the label to the figure.
_PICTURE_LABEL_SHARE = 0.5

#: How far past the picture's bbox a text item may sit and still count as the
#: figure's own text. A figure's headers float *above* the plot area, outside
#: Docling's picture bbox: the Figure 6 legend in the two-column corpus sits
#: 6.8pt above it and a panel title 7.3pt, so the old 2pt tolerance let both
#: through as prose -- the legend, squeezed into its 41pt column, kept source
#: and tripped the delivery gate. Measured across the reference corpus the gap
#: is clean: figure-internal items sit within ~8pt of the bbox, the nearest
#: genuine body text 21.9pt away, so 10pt clears every true positive with ~12pt
#: of headroom before it could touch real prose.
_PICTURE_CONTAINMENT_MARGIN_PT = 10.0


def _overlap_share(box: Any, pbox: Any) -> float:
    """Fraction of ``box``'s own area that lies inside ``pbox``."""
    width = min(float(getattr(box, "r", 0.0)), float(getattr(pbox, "r", 0.0))) - max(
        float(getattr(box, "l", 0.0)), float(getattr(pbox, "l", 0.0))
    )
    height = min(float(getattr(box, "t", 0.0)), float(getattr(pbox, "t", 0.0))) - max(
        float(getattr(box, "b", 0.0)), float(getattr(pbox, "b", 0.0))
    )
    area = (float(getattr(box, "r", 0.0)) - float(getattr(box, "l", 0.0))) * (
        float(getattr(box, "t", 0.0)) - float(getattr(box, "b", 0.0))
    )
    if area <= 0 or width <= 0 or height <= 0:
        return 0.0
    return (width * height) / area


def is_inside_picture(
    page_no: int,
    bbox: Any,
    picture_boxes: list[tuple[int, Any]],
    margin_pt: float = _PICTURE_CONTAINMENT_MARGIN_PT,
) -> bool:
    """Whether a Docling item is the picture's own text (same page).

    A figure's text sits *on* the picture rather than neatly inside it: an axis
    title's baseline hangs a few points below the plot area's bbox, a rotated
    axis label straddles its edge, and a legend floats above it. Requiring full
    containment let exactly those labels through as ordinary prose, which then
    reached the translator and was painted into its ~100pt box at the source's
    15pt size -- wrecking the figure's layout and leaving the figure half
    translated, because its tick labels, legend and in-chart annotations stay in
    the preserved source graphic.

    A label belongs to the figure when it is fully inside it (within
    ``margin_pt``, wide enough to cover a header just outside the bbox), when
    most of its own area overlaps it, or when its centre falls inside it. A
    caption is placed *below* the picture with no overlap at all, so captions
    and footnotes stay translatable (the caller keeps them out of this test).
    """
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
        if _overlap_share(bbox, pbox) >= _PICTURE_LABEL_SHARE:
            return True
        centre_x = (float(bbox.l) + float(bbox.r)) / 2.0
        centre_y = (float(bbox.b) + float(bbox.t)) / 2.0
        if pbox.l <= centre_x <= pbox.r and pbox.b <= centre_y <= pbox.t:
            return True
    return False


#: A connective word inside a citation run means the cell is prose (a sentence
#: citing two works), not a stack of names. Keeps ``_split_stacked_cell`` from
#: tearing "SigLIP [12] and CLIP [13]" apart.
_STACKED_CONNECTIVE_RE = re.compile(
    r"\b(?:and|or|to|of|in|for|with|using|via|by|on|at|as|et|al)\b", re.IGNORECASE
)
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_CITATION_RE = re.compile(r"\[[^\]]*\d[^\]]*\]")


def _split_stacked_cell(text: str) -> list[str] | None:
    """Split a cell Docling fused from several visual rows, or ``None``.

    Two shapes, both unambiguous:

    * a run of citation-terminated names — each piece carries ``[n]`` and no
      connective word ("MonkeyOCR [27] MinerU2.5 [5]");
    * a run of pure numbers ("0.169 0.177 0.178").

    Anything else (prose with citations, a single value, a name with a space)
    returns ``None`` so the cell is left exactly as Docling gave it.
    """
    body = (text or "").strip()
    if not body:
        return None
    if len(_CITATION_RE.findall(body)) >= 2 and not _STACKED_CONNECTIVE_RE.search(body):
        pieces = re.split(r"(?<=\])\s+(?=\S)", body)
        if len(pieces) >= 2 and all(_CITATION_RE.search(p) for p in pieces):
            return pieces
        return None
    tokens = body.split()
    if len(tokens) >= 2 and all(_NUMBER_RE.fullmatch(tok) for tok in tokens):
        return tokens
    return None


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
                    # Keep a cell's internal line breaks. Docling emits one text
                    # line per wrapped visual row inside a cell, and collapsing
                    # them to spaces fused a stacked column into one blob: the
                    # benchmark table's "Recognition Method" column (four model
                    # names stacked) became "MonkeyOCR [27] MinerU2.5 [5]
                    # Dolphin-v2 [28] PaddleOCR-VL…" in a single cell, and the
                    # whole table rendered unreadable. Markdown has no in-cell
                    # newline, so use ``<br>`` (the renderer splits on it).
                    lines = [
                        " ".join(line.split()) for line in raw_text.splitlines() if line.strip()
                    ]
                    # Docling sometimes merges *several* visual rows into one
                    # cell WITHOUT a newline, joining them with spaces — the
                    # benchmark table's first row held "MonkeyOCR [27] MinerU2.5
                    # [5]" and "85.10 85.73" in cells that should be two rows
                    # each. Split such a cell back into its stacked values. Kept
                    # conservative: only a run of citation-terminated names
                    # (each piece carries ``[n]``, no connective word) or a run
                    # of pure numbers, so real prose ("SigLIP [12] and CLIP
                    # [13]") is never touched.
                    if len(lines) == 1:
                        stacked = _split_stacked_cell(lines[0])
                        if stacked is not None:
                            lines = stacked
                    clean_text = "<br>".join(lines).replace("|", r"\|")
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
    the table grid. Left alone they would be translated as separate paragraphs
    and lose their column alignment, so the reader would see the table's tail
    scattered as prose. Merging them into the preceding table block keeps the
    whole structure inside one table element.
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
            # phrase. Accepting any block under 120 chars or one containing a
            # bare ":" would swallow a short body sentence directly below
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
