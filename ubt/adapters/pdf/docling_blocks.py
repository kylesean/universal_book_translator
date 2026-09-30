"""Pure IRBlock shaping for the Docling adapter.

These helpers were the Tail of ``docling_adapter.DoclingPDFAdapter``. They
never touch adapter state: each one takes the mapped blocks (or a raw Docling
item) and returns blocks, so they live here as plain functions and the
adapter keeps thin static aliases for its historical call sites. The
composed pipeline is :func:`postprocess_blocks`.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from ubt.core.ir.models import BlockType, BoundingBox, FlowID, IRBlock, LayoutRole
from ubt.core.policy.layout_policy import CAPTION_RE, PAIR_TERMINAL_PUNCT

logger = logging.getLogger(__name__)

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
    page's figure-caption body into a single text item (chapter-3 FIG. 3.1
    and FIG. 3.5 captions were both swallowed this way). The item's prov
    charspans expose both regions; return ``(head_text, tail_text,
    tail_bbox)`` when the trailing span is a sentence-like caption body on
    a different page, otherwise ``None`` so the item stays untouched.
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


def attach_split_caption_tails(blocks: list[IRBlock]) -> list[IRBlock]:
    """Fuse a span-split caption body back onto its bare "FIG. N" label.

    The split tail sits before the label in reading order (docling emits
    the swallowed caption text at the end of the paragraph item, and the
    label item follows the figure), so search backwards from each bare
    caption label for the nearest split tail on the same page. Formula
    blocks may sit between the two; another caption label ends the search.
    """
    removed: set[int] = set()
    for i, head in enumerate(blocks):
        head_text = (head.source_text or "").strip()
        if not (
            CAPTION_RE.fullmatch(head_text)
            or (CAPTION_RE.match(head_text) and len(head_text) <= 20)
        ):
            continue
        for j in range(i - 1, max(i - 8, -1), -1):
            if j in removed:
                continue
            cand = blocks[j]
            cand_text = (cand.source_text or "").strip()
            if not cand.provenance.get("docling_span_split_tail"):
                if CAPTION_RE.match(cand_text) and len(cand_text) <= 20:
                    break  # a different caption label owns nothing here
                continue
            if cand.bbox is not None and head.bbox is not None and cand.bbox.page != head.bbox.page:
                continue
            head.source_text = f"{head_text}: {cand_text}"
            head.flow_id = FlowID.CAPTION
            removed.add(j)
            break
    if not removed:
        return blocks
    return [b for k, b in enumerate(blocks) if k not in removed]


def decouple_embedded_captions(blocks: list[IRBlock]) -> list[IRBlock]:
    """Decouple figure captions that Docling merged into preceding body paragraphs."""
    # Generic caption decoupling only: rigid on explicit "Figure N:" /
    # "Fig. N:" markers. Book-specific caption opener strings are NOT
    # hardcoded here — they leaked single-book calibration into generic
    # Extraction and never matched any other document.
    caption_pattern = re.compile(
        r"(?P<prose>.*?)(?<!\bas shown in)(?<!\bin)(?<!\bfrom)(?<!\bsee)(?<!\busing)\s+"
        r"(?P<caption>(?:FIGURE|Figure|FIG\.|Fig\.)\s+\d+(?:\.\d+)*[.:]?\s+[A-Z].*?\.)\s*$",
        re.DOTALL,
    )
    out: list[IRBlock] = []
    for b in blocks:
        if b.block_type == BlockType.NARRATIVE and b.source_text:
            m = caption_pattern.search(b.source_text)
            if m and len(m.group("prose").strip()) > 20:
                prose_text = m.group("prose").strip()
                cap_text = m.group("caption").strip()
                b.source_text = prose_text
                # Split the merged paragraph's vertical band between the prose
                # and the caption. Handing the caption the parent's *entire*
                # bbox made the two rigid zones overlap: the caption's zone was
                # clipped to zero height and rejected, while the prose
                # translation (which no longer contains the caption) was drawn
                # over the source caption line — the caption vanished from the
                # delivered PDF. Disjoint bands keep both zones placeable.
                #
                # Bounding boxes are bottom-up (y0 = the lower edge; the rigid
                # typesetter converts with ``page_h - y1``), and the caption is
                # the *last* text of the paragraph, so it owns the bottom band.
                # Taking the top band instead painted the translated caption
                # over the source prose and vice versa — the collision this
                # split exists to prevent.
                cap_bbox: BoundingBox | None = None
                if b.bbox is not None:
                    height = max(b.bbox.y1 - b.bbox.y0, 0.0)
                    share = len(cap_text) / max(len(prose_text) + len(cap_text), 1)
                    split_y = b.bbox.y0 + height * share
                    cap_bbox = BoundingBox(
                        page=b.bbox.page,
                        x0=b.bbox.x0,
                        y0=b.bbox.y0,
                        x1=b.bbox.x1,
                        y1=split_y,
                    )
                    b.bbox = BoundingBox(
                        page=b.bbox.page,
                        x0=b.bbox.x0,
                        y0=split_y,
                        x1=b.bbox.x1,
                        y1=b.bbox.y1,
                    )
                out.append(b)
                cap_block = IRBlock(
                    id=f"{b.id}_cap",
                    # Share the parent's spine index: a fresh ``len(out) + 1``
                    # would equal the next block's own index (the caption adds
                    # one extra block, shifting every following position). The
                    # ledger's keyset order is ``(spine_index, block_id)`` and
                    # the ``_cap`` suffix sorts after the parent, so a tie keeps
                    # the caption immediately after its paragraph.
                    spine_index=b.spine_index,
                    block_type=BlockType.NARRATIVE,
                    flow_id=FlowID.CAPTION,
                    source_text=cap_text,
                    bbox=cap_bbox,
                    layout_role=LayoutRole.CAPTION,
                )
                out.append(cap_block)
                continue
        out.append(b)
    return out


def unify_figure_captions(blocks: list[IRBlock]) -> list[IRBlock]:
    """Fuse orphan caption tags (e.g. 'FIG. 3.1') with their descriptive text.

    Docling frequently separates 'FIG. 3.2' and its body 'Fin potential ...'
    into two independent blocks. Unifying them produces coherent LLM translation
    and proper academic caption layout.
    """
    i = 0
    while i < len(blocks):
        head = blocks[i]
        head_text = (head.source_text or "").strip()
        if CAPTION_RE.fullmatch(head_text) or (
            CAPTION_RE.match(head_text) and len(head_text) <= 20
        ):
            head.block_type = BlockType.NARRATIVE
            head.flow_id = FlowID.CAPTION
            merged = False
            # Try next block first (standard order: label then description)
            if i + 1 < len(blocks):
                cand = blocks[i + 1]
                cand_text = (cand.source_text or "").strip()
                if (
                    cand.block_type == BlockType.NARRATIVE
                    and cand.flow_id in (FlowID.CAPTION, FlowID.MAIN_STORY)
                    and (head.bbox is None or cand.bbox is None or head.bbox.page == cand.bbox.page)
                    and not CAPTION_RE.match(cand_text)
                    and len(cand_text) > 5
                ):
                    head.source_text = f"{head_text}: {cand_text}"
                    if (
                        head.bbox is not None
                        and cand.bbox is not None
                        and head.bbox.page == cand.bbox.page
                    ):
                        head.bbox = BoundingBox(
                            page=head.bbox.page,
                            x0=min(head.bbox.x0, cand.bbox.x0),
                            y0=min(head.bbox.y0, cand.bbox.y0),
                            x1=max(head.bbox.x1, cand.bbox.x1),
                            y1=max(head.bbox.y1, cand.bbox.y1),
                        )
                    del blocks[i + 1]
                    merged = True
            # Try previous block (if description preceded the label block on same page)
            if not merged and i > 0:
                prev = blocks[i - 1]
                prev_text = (prev.source_text or "").strip()
                if (
                    prev.block_type == BlockType.NARRATIVE
                    and prev.flow_id in (FlowID.CAPTION, FlowID.MAIN_STORY)
                    and (head.bbox is None or prev.bbox is None or head.bbox.page == prev.bbox.page)
                    and not CAPTION_RE.match(prev_text)
                    and len(prev_text) > 5
                    and not prev_text.endswith((".", "!", "?", "。"))
                ):
                    head.source_text = f"{head_text}: {prev_text}"
                    if (
                        head.bbox is not None
                        and prev.bbox is not None
                        and head.bbox.page == prev.bbox.page
                    ):
                        head.bbox = BoundingBox(
                            page=head.bbox.page,
                            x0=min(head.bbox.x0, prev.bbox.x0),
                            y0=min(head.bbox.y0, prev.bbox.y0),
                            x1=max(head.bbox.x1, prev.bbox.x1),
                            y1=max(head.bbox.y1, prev.bbox.y1),
                        )
                    del blocks[i - 1]
                    i -= 1
                    merged = True
        i += 1
    return blocks


def defragment_narrative_blocks(
    blocks: list[IRBlock], *, allow_cross_page: bool = True
) -> list[IRBlock]:
    """Stitch broken narrative paragraphs across page breaks and figure floats.

    Docling often breaks a sentence across page boundaries or around figures
    (e.g., 'which only has spatial' on page 2 and 'dependence, Nch is...' on page 3).
    This stitches them back together into single coherent translation units.
    """
    i = 0
    while i < len(blocks) - 1:
        curr = blocks[i]
        if (
            curr.block_type != BlockType.NARRATIVE
            or curr.flow_id != FlowID.MAIN_STORY
            or curr.skip_translate
            or curr.provenance.get("toc_entry")
        ):
            i += 1
            continue
        curr_text = (curr.source_text or "").strip()
        if not curr_text:
            i += 1
            continue

        ends_open = curr_text.endswith("-") or (
            not curr_text.endswith(
                (".", "!", "?", "。", "！", "？", ":", "：", "”", '"', "）", ")")
            )
            and len(curr_text) > 10
        )
        if not ends_open:
            i += 1
            continue

        next_narrative_idx = None
        for j in range(i + 1, len(blocks)):
            cand = blocks[j]
            if cand.provenance.get("toc_entry"):
                break
            if (
                cand.block_type == BlockType.NARRATIVE
                and cand.flow_id == FlowID.MAIN_STORY
                and not cand.skip_translate
            ):
                next_narrative_idx = j
                break
            elif cand.block_type in (
                BlockType.HEADING,
                BlockType.CODE,
                BlockType.FORMULA,
                BlockType.TABLE,
            ) or (cand.block_type == BlockType.NARRATIVE and cand.flow_id != FlowID.CAPTION):
                break

        if next_narrative_idx is not None:
            next_block = blocks[next_narrative_idx]
            next_text = (next_block.source_text or "").strip()
            if next_text:
                first_char = next_text[0]
                first_word = next_text.split(None, 1)[0].lower().strip("([\"'")
                is_continuation = (
                    first_char.islower()
                    or first_word
                    in {
                        "dependence",
                        "where",
                        "which",
                        "and",
                        "or",
                        "but",
                        "for",
                        "with",
                        "in",
                        "to",
                        "that",
                        "as",
                        "by",
                        "from",
                        "is",
                        "are",
                        "was",
                        "were",
                        "has",
                        "have",
                        "had",
                    }
                    or first_char in (",", ";", ")", "]", "}", "，", "；", "）")
                )
                math_even = (curr_text + next_text).count("$") % 2 == 0

                if is_continuation and math_even:
                    if (
                        not allow_cross_page
                        and curr.bbox is not None
                        and next_block.bbox is not None
                        and curr.bbox.page != next_block.bbox.page
                    ):
                        i += 1
                        continue
                    if curr_text.endswith("-"):
                        curr.source_text = curr_text[:-1] + next_text
                    else:
                        curr.source_text = f"{curr_text} {next_text}"
                    del blocks[next_narrative_idx]
                    continue
        i += 1
    return blocks


def latch_caption_bodies(blocks: list[IRBlock]) -> list[IRBlock]:
    """Latch caption-body paragraphs onto the caption flow (post-pass).

    Docling labels the caption tag line (``FIG. 3.6``) but frequently
    leaves the caption body itself as plain text, so the body flows as
    body prose: it merges with neighbours and its paint truncates
    (chapter-3 FIG. 3.6 kept only its first clause). After a
    caption-label block, the immediately following narrative run on the
    same page belongs to the figure: latch it onto ``CAPTION`` until a
    terminal-punctuated block (inclusive). The label block itself is
    untouched (it already translates fine as a heading).
    """
    for i, head in enumerate(blocks):
        if not CAPTION_RE.match((head.source_text or "").strip()):
            continue
        for j in range(i + 1, len(blocks)):
            cand = blocks[j]
            if cand.block_type != BlockType.NARRATIVE or cand.flow_id != FlowID.MAIN_STORY:
                break
            if cand.bbox is None or head.bbox is None or cand.bbox.page != head.bbox.page:
                break
            # Do not latch lowercase sentence continuations as captions
            cand_src = (cand.source_text or "").lstrip()
            if (
                not cand_src
                or cand_src[:1].islower()
                or cand_src.startswith(("dependence", "where ", "which ", "and "))
            ):
                break
            cand.flow_id = FlowID.CAPTION
            tail = (cand.source_text or "").rstrip()
            if tail and tail[-1] in PAIR_TERMINAL_PUNCT:
                break
    return blocks


def fuse_chapter_number(blocks: list[IRBlock]) -> list[IRBlock]:
    """Fuse an orphan chapter digit into its CHAPTER heading pre-translation.

    Docling often splits ``CHAPTER`` / title / ``3`` into three blocks;
    translating ``CHAPTER`` alone yields ``第章`` (numberless) while the
    lone ``3`` is later dropped as page chrome. When a bare-digit
    narrative sits within 3 blocks after a CHAPTER heading inside the
    opening window, merge it (``CHAPTER 3`` → ``第 3 章``) and drop the
    digit block. Ids of surviving blocks are untouched.
    """
    if len(blocks) < 3:
        return blocks
    head_idx = next(
        (
            i
            for i, b in enumerate(blocks[:6])
            if b.block_type == BlockType.HEADING
            and re.fullmatch(r"(?i)\s*chapter\s*", b.source_text or "")
        ),
        None,
    )
    if head_idx is None:
        return blocks
    num_idx = next(
        (
            i
            for i in range(head_idx + 1, min(head_idx + 4, len(blocks)))
            if blocks[i].block_type == BlockType.NARRATIVE
            and re.fullmatch(r"\s*\d{1,3}\s*", blocks[i].source_text or "")
        ),
        None,
    )
    if num_idx is None:
        return blocks
    num_block = blocks[num_idx]
    num = (num_block.source_text or "").strip()
    head = blocks[head_idx]
    old_source = (head.source_text or "").strip()
    head.source_text = f"{old_source} {num}"
    head.skip_translate = False
    head.layout_role = LayoutRole.TITLE
    if head.bbox is not None and num_block.bbox is not None:
        head_h = head.bbox.y1 - head.bbox.y0
        num_h = num_block.bbox.y1 - num_block.bbox.y0
        if 0.5 <= num_h / max(head_h, 1e-3) <= 2.0:
            head.bbox = BoundingBox(
                page=head.bbox.page,
                x0=min(head.bbox.x0, num_block.bbox.x0),
                y0=min(head.bbox.y0, num_block.bbox.y0),
                x1=max(head.bbox.x1, num_block.bbox.x1),
                y1=max(head.bbox.y1, num_block.bbox.y1),
            )
    # The former zh preset (``target_text = f"第 {num} 章"``)
    # hardcoded Chinese into a target-language-agnostic extraction stage —
    # non-zh runs could ship a Chinese fragment through MT-tier/verbatim
    # paths. The fused "CHAPTER 3" source is translated normally by
    # whichever tier handles the block, in the actual target language.
    del blocks[num_idx]
    return blocks


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
                out[-1].bbox = union_box
                out[-1].source_text = merged_txt
                if out[-1].target_text is not None:
                    out[-1].target_text = merged_txt
                continue
        out.append(b)
    return out


def postprocess_blocks(
    blocks: list[IRBlock],
    *,
    allow_cross_page: bool = True,
    pdf_path: Path | None = None,
) -> list[IRBlock]:
    """Apply the caption/narrative post-pipeline in its fixed order.

    Order is load-bearing: chapter fuse → embedded-caption decouple →
    caption-body latch → span-tail attach → caption unify → overlapping
    formula merge → flow reassembly → narrative defragment.
    """
    staged = resolve_overlapping_formula_blocks(
        unify_figure_captions(
            attach_split_caption_tails(
                latch_caption_bodies(decouple_embedded_captions(fuse_chapter_number(blocks)))
            )
        )
    )
    from ubt.adapters.pdf.flow_reassembly import reassemble_flow

    staged, _stats = reassemble_flow(staged, pdf_path=pdf_path, allow_cross_page=allow_cross_page)
    return defragment_narrative_blocks(staged, allow_cross_page=allow_cross_page)
