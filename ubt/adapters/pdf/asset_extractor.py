"""Permissive PDF asset and figure extractor (Apache-2.0 / BSD, zero AGPL).

Extracts embedded and vector figures from born-digital PDFs into high-resolution
PNG assets without invoking heavy vision models or violating license guards.
"""

from __future__ import annotations

import logging
import re
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ubt.adapters.pdf.pdfium_gate import pdfium_serialized

logger = logging.getLogger(__name__)

_FIG_CAPTION_RE = re.compile(
    r"(?:^|\n)\s*(FIG(?:URE)?\.?\s*([0-9A-Z]+(?:\.[0-9A-Z]+)?)[^\n]*)",
    re.IGNORECASE,
)
_FIG_ID_RE = re.compile(r"([0-9A-Z]+\.[0-9A-Z]+)", re.IGNORECASE)
_SENTENCE_VERBS_RE = re.compile(
    r"^(?:shows?|illustrates?|displays?|depicts?|is|are|was|were|presents?|compares?|gives?|can)\b",
    re.IGNORECASE,
)


def _close_handles(*handles: Any) -> None:
    """Best-effort close of pypdfium2 native handles (None-safe)."""
    for handle in handles:
        if handle is not None:
            with suppress(Exception):
                handle.close()


@dataclass
class ExtractedFigure:
    """Metadata for an extracted figure asset."""

    fig_id: str
    caption_en: str
    page: int
    image_path: Path
    relative_path: str
    bbox: tuple[float, float, float, float] | None = None


@pdfium_serialized
def extract_pdf_figures(
    pdf_path: Path | str,
    output_assets_dir: Path | str,
    dpi: int = 300,
) -> dict[str, ExtractedFigure]:
    """Extract figures from a PDF document to disk using pypdfium2 (zero-AGPL).

    Scans pages for standard academic/book figure captions ('FIG. X.Y'),
    determines bounding boxes from vector path objects or raster image objects,
    and crops them cleanly at `dpi` resolution.
    """
    import pypdfium2 as pdfium

    path = Path(pdf_path)
    out_dir = Path(output_assets_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        logger.warning("PDF path does not exist for figure extraction: %s", path)
        return {}

    try:
        doc = pdfium.PdfDocument(str(path))
    except Exception as exc:
        logger.warning("Failed to open PDF for asset extraction '%s': %s", path.name, exc)
        return {}

    extracted: dict[str, ExtractedFigure] = {}

    try:
        # Deferred close: the figure loop below is the last statement of the page
        # body, so a close placed after it would run per figure. Instead the
        # previous page's handles are flushed when the next page starts (peak +1
        # page held — negligible vs leaking every page of a 500-page book).
        prev_handles: list[Any] = []
        for page_idx, page in enumerate(doc):
            _close_handles(*prev_handles)
            prev_handles = []
            page_num = page_idx + 1
            textpage = None
            try:
                textpage = page.get_textpage()
                w, h = page.get_size()
                text = textpage.get_text_range()
            except Exception as exc:
                logger.debug("Page %d get_textpage error: %s", page_num, exc)
                _close_handles(textpage, page)
                continue
            prev_handles = [textpage, page]

            try:
                lines = [line.strip() for line in text.split("\n") if line.strip()]
                cap_lines = [
                    line
                    for line in lines
                    if (line.upper().startswith("FIG. ") or line.upper().startswith("FIGURE "))
                    and len(line) < 150
                ]
                if not cap_lines:
                    _close_handles(textpage, page)
                    prev_handles = []
                    continue

                cap_info: list[dict[str, Any]] = []
                for cap in cap_lines:
                    m = _FIG_ID_RE.search(cap)
                    if not m:
                        continue
                    fig_id_raw = m.group(1)
                    # No verb filter here: ``cap_lines`` already requires the
                    # line to *start* with ``FIG.``/``FIGURE``, so the filter
                    # only ever rejected real captions like "Figure 3.2 shows
                    # the pipeline". The candidate search below still skips
                    # verb-led hits and falls back to the next match.
                    # Match the actual caption prefix (e.g. "FIG. 3.2" or "FIGURE 3.2")
                    # to avoid false positives on in-text references like "as shown in Fig. 3.2"
                    prefix_match = re.match(
                        r"(?:FIG(?:URE)?\.?\s*" + re.escape(fig_id_raw) + r")",
                        cap,
                        re.IGNORECASE,
                    )
                    search_query = prefix_match.group(0) if prefix_match else f"FIG. {fig_id_raw}"
                    search = textpage.search(search_query)
                    candidates = []
                    while True:
                        hit = search.get_next()
                        if not hit:
                            break
                        char_start, char_count = hit
                        preceding = text[:char_start]
                        last_nl = max(preceding.rfind("\n"), preceding.rfind("\r"))
                        line_prefix = preceding[last_nl + 1 :] if last_nl != -1 else preceding

                        following = text[char_start + char_count :]
                        next_nl_candidates = [
                            i for i in [following.find("\n"), following.find("\r")] if i != -1
                        ]
                        next_nl = min(next_nl_candidates) if next_nl_candidates else len(following)
                        line_suffix = following[:next_nl].strip()

                        is_line_start = line_prefix.strip() == ""
                        is_verb = bool(_SENTENCE_VERBS_RE.match(line_suffix))
                        if is_line_start and not is_verb:
                            candidates.append(hit)

                    if candidates:
                        match = candidates[0]
                    else:
                        search = textpage.search(search_query)
                        match = search.get_next()
                        if not match:
                            search = textpage.search(cap[: min(len(cap), 20)])
                            match = search.get_next()
                    if not match:
                        continue
                    # count_rects with a char range RE-SCOPES the rect array to that
                    # segment: the get_rect(0) below returns the caption hit's own
                    # first rect, NOT the page's first line (verified against
                    # pypdfium2 with a two-line fixture — a review round misread
                    # this as systematic caption mis-anchoring; do not "fix" it).
                    textpage.count_rects(match[0], match[1])
                    r = textpage.get_rect(0)
                    cap_info.append(
                        {
                            "fig_id": fig_id_raw,
                            "caption": cap,
                            "rect": r,
                            "y_top": r[3],
                            "y_bottom": r[1],
                        }
                    )

                # Sort captions from bottom of page to top (ascending y in PDF points)
                cap_info.sort(key=lambda c: c["y_bottom"])

                img_objs = [
                    obj for obj in page.get_objects() if obj.type == pdfium.raw.FPDF_PAGEOBJ_IMAGE
                ]
                path_objs = [
                    obj for obj in page.get_objects() if obj.type == pdfium.raw.FPDF_PAGEOBJ_PATH
                ]

                for i, c in enumerate(cap_info):
                    fig_id = c["fig_id"]
                    # Figure numbers repeat across chapters ("Figure 3.2" in ch3 and
                    # ch9). The page disambiguates the filename and the dict key, or
                    # the later figure overwrites the earlier PNG and both blocks end
                    # up pointing at one image.
                    fig_key = f"{fig_id}_p{page_num}"
                    safe_id = fig_key.replace(".", "_")
                    y_start = c["y_top"] + 2

                    # Upper bound: next caption bottom or top margin
                    if i + 1 < len(cap_info):
                        y_end = cap_info[i + 1]["y_bottom"] - 10
                    else:
                        y_end = min(h - 35, y_start + 260)

                    # Collect relevant graphical boxes in (c['y_bottom'] - 20, y_end + 20)
                    # AND overlapping the caption's own x-span: a y-only filter
                    # unioned two side-by-side figures at the same height into one
                    # crop (figure A's PNG contained figure B and the gutter). A
                    # full-width caption still overlaps every box, so it is
                    # unchanged.
                    cap_x0, cap_x1 = c["rect"][0], c["rect"][2]

                    # Bind the loop's x-span as defaults: the closure is used
                    # within this iteration, but binding keeps it correct if the
                    # call is ever deferred (ruff B023).
                    def _overlaps_caption_x(
                        b: tuple[float, float, float, float],
                        x0: float = cap_x0,
                        x1: float = cap_x1,
                    ) -> bool:
                        return b[2] >= x0 - 15 and b[0] <= x1 + 15

                    relevant_boxes: list[tuple[float, float, float, float]] = []
                    for obj in img_objs:
                        b = obj.get_bounds()
                        if (
                            b[3] >= c["y_bottom"] - 20
                            and b[1] <= y_end + 20
                            and _overlaps_caption_x(b)
                        ):
                            relevant_boxes.append(b)
                    for obj in path_objs:
                        b = obj.get_bounds()
                        if (
                            (b[2] - b[0]) > 30
                            and (b[3] - b[1]) > 30
                            and b[3] >= c["y_bottom"] - 20
                            and b[1] <= y_end + 20
                            and _overlaps_caption_x(b)
                        ):
                            relevant_boxes.append(b)

                    if relevant_boxes:
                        bx0 = min(b[0] for b in relevant_boxes) - 45
                        # For by0: ensure it stays above the caption to avoid caption text/rule artifacts,
                        # but low enough to capture x-axis labels (which are PDF text objects).
                        candidate_by0 = min(b[1] for b in relevant_boxes) - 28
                        by0 = max(candidate_by0, c["y_top"] + 2)
                        bx1 = max(b[2] for b in relevant_boxes) + 30
                        by1 = max(b[3] for b in relevant_boxes) + 12
                    else:
                        bx0, by0, bx1, by1 = 45, y_start, w - 45, y_end

                    bx0 = max(10, bx0)
                    by0 = max(10, by0)
                    bx1 = min(w - 10, bx1)
                    by1 = min(h - 10, by1)

                    # Avoid degenerate crop
                    if bx1 <= bx0 or by1 <= by0:
                        continue

                    crop = (bx0, by0, w - bx1, h - by1)
                    try:
                        bitmap = page.render(scale=dpi / 72.0, crop=crop)
                        try:
                            pil_img = bitmap.to_pil()
                        finally:
                            _close_handles(bitmap)
                        img_name = f"fig_{safe_id}.png"
                        target_file = out_dir / img_name
                        pil_img.save(target_file)
                        extracted[fig_key] = ExtractedFigure(
                            fig_id=fig_id,
                            caption_en=c["caption"],
                            page=page_num,
                            image_path=target_file,
                            relative_path=f"assets/{img_name}",
                            bbox=(bx0, by0, bx1, by1),
                        )
                        logger.debug(
                            "Extracted figure %s on page %d: %s (size %s)",
                            fig_id,
                            page_num,
                            img_name,
                            pil_img.size,
                        )
                    except Exception as exc:
                        logger.warning(
                            "Failed to crop/render figure %s on page %d: %s",
                            fig_id,
                            page_num,
                            exc,
                        )

            except Exception as exc:
                logger.warning("Page %d figure extraction skipped: %s", page_num, exc)
                continue
    finally:
        # A raise must not cost the whole book its figures or leak the
        # native handles: pending page handles and the doc close here.
        _close_handles(*prev_handles)
        _close_handles(doc)
    return extracted
