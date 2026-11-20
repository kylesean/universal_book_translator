"""Pluggable PDF layout detectors (PP-DocLayoutV2 ONNX default, Apache-2.0).

Separates macro visual layout classification (where are titles, paragraphs,
display formulas, inline formulas, tables, figures, captions, footnotes, and
headers/footers?) from micro PDF content-stream character extraction.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from PIL import Image

from ubt.codecs.pdf.stream_analyzer import PageStreamAnalysis


class RegionKind(StrEnum):
    """Normalized visual layout categories produced by any :class:`LayoutDetector`."""

    PARAGRAPH = "paragraph"
    HEADING = "heading"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    TABLE = "table"
    DISPLAY_FORMULA = "display_formula"
    INLINE_FORMULA = "inline_formula"
    FIGURE = "figure"
    ALGORITHM = "algorithm"
    FURNITURE = "furniture"  # header, footer, page number, arXiv side stamp


@dataclass(frozen=True, slots=True)
class LayoutRegion:
    """One detected layout region in logical PDF bottom-left coordinates."""

    kind: RegionKind
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1) bottom-left
    score: float
    reading_order: int
    raw_label: str = ""


_PP_DOCLAYOUT_LABEL_MAP: dict[str, RegionKind] = {
    "text": RegionKind.PARAGRAPH,
    "abstract": RegionKind.PARAGRAPH,
    "content": RegionKind.PARAGRAPH,
    "reference": RegionKind.HEADING,
    "reference_content": RegionKind.PARAGRAPH,
    "doc_title": RegionKind.HEADING,
    "paragraph_title": RegionKind.HEADING,
    "figure_title": RegionKind.CAPTION,
    "footnote": RegionKind.FOOTNOTE,
    "vision_footnote": RegionKind.FOOTNOTE,
    "table": RegionKind.TABLE,
    "display_formula": RegionKind.DISPLAY_FORMULA,
    "formula_number": RegionKind.DISPLAY_FORMULA,
    "inline_formula": RegionKind.INLINE_FORMULA,
    "image": RegionKind.FIGURE,
    "chart": RegionKind.FIGURE,
    "header_image": RegionKind.FIGURE,
    "footer_image": RegionKind.FIGURE,
    "seal": RegionKind.FIGURE,
    "algorithm": RegionKind.ALGORITHM,
    "header": RegionKind.FURNITURE,
    "footer": RegionKind.FURNITURE,
    "number": RegionKind.FURNITURE,
    "aside_text": RegionKind.FURNITURE,
    "vertical_text": RegionKind.FURNITURE,
}


@runtime_checkable
class LayoutDetector(Protocol):
    """Detects macro layout regions on one PDF page."""

    def detect_page(
        self,
        page_image: Image.Image | None,
        analysis: PageStreamAnalysis,
    ) -> list[LayoutRegion]: ...


_ENGINE_LOCK = threading.Lock()
_CACHED_ENGINES: dict[str, Any] = {}


def _get_rapid_layout_engine(model_type: str, conf_thresh: float) -> Any:
    key = f"{model_type}:{conf_thresh:.2f}"
    with _ENGINE_LOCK:
        engine = _CACHED_ENGINES.get(key)
        if engine is None:
            from rapid_layout import ModelType, RapidLayout

            mt = ModelType(model_type)
            engine = RapidLayout(model_type=mt, conf_thresh=conf_thresh)
            _CACHED_ENGINES[key] = engine
        return engine


class RapidPPDocLayoutDetector:
    """Apache-2.0 PP-DocLayoutV2/V3 ONNX layout detector via ``rapid-layout``."""

    def __init__(
        self,
        *,
        model_type: str = "pp_doc_layoutv2",
        conf_thresh: float = 0.45,
    ) -> None:
        self.model_type = model_type
        self.conf_thresh = conf_thresh

    def detect_page(
        self,
        page_image: Image.Image | None,
        analysis: PageStreamAnalysis,
    ) -> list[LayoutRegion]:
        if page_image is None:
            return HeuristicLayoutDetector().detect_page(None, analysis)

        img_w, img_h = page_image.size
        if img_w <= 0 or img_h <= 0:
            return []

        engine = _get_rapid_layout_engine(self.model_type, self.conf_thresh)
        output = engine(page_image)
        boxes = getattr(output, "boxes", None)
        names = getattr(output, "class_names", None)
        scores = getattr(output, "scores", None)
        if boxes is None or names is None or scores is None or len(boxes) == 0:
            return []

        sx = analysis.page_width / float(img_w)
        sy = analysis.page_height / float(img_h)
        regions: list[LayoutRegion] = []
        for idx, (box, label, score) in enumerate(zip(boxes, names, scores, strict=False)):
            px0, py0, px1, py1 = (float(v) for v in box[:4])
            # Image coordinates have origin at top-left; flip y to PDF bottom-left.
            lx0 = max(0.0, px0 * sx)
            lx1 = min(analysis.page_width, px1 * sx)
            ly0 = max(0.0, analysis.page_height - py1 * sy)
            ly1 = min(analysis.page_height, analysis.page_height - py0 * sy)
            if lx1 <= lx0 or ly1 <= ly0:
                continue
            raw = str(label).strip().lower()
            kind = _PP_DOCLAYOUT_LABEL_MAP.get(raw, RegionKind.PARAGRAPH)
            regions.append(
                LayoutRegion(
                    kind=kind,
                    bbox=(lx0, ly0, lx1, ly1),
                    score=float(score),
                    reading_order=idx,
                    raw_label=raw,
                )
            )
        return regions


class HeuristicLayoutDetector:
    """Zero-model geometric paragraph clusterer for offline tests and simple PDFs."""

    def detect_page(
        self,
        page_image: Image.Image | None,
        analysis: PageStreamAnalysis,
    ) -> list[LayoutRegion]:
        del page_image
        if not analysis.text_ops:
            return []

        # Group text ops into lines by baseline proximity.
        sorted_ops = sorted(analysis.text_ops, key=lambda o: (-o.baseline_y, o.bbox[0]))
        lines: list[list[Any]] = []
        for op in sorted_ops:
            if not op.text.strip():
                continue
            placed = False
            for ln in lines:
                ref = ln[0]
                tol = max(2.0, min(ref.font_size_pt, op.font_size_pt) * 0.45)
                if abs(op.baseline_y - ref.baseline_y) <= tol:
                    # Keep distinct columns separated if there is a wide horizontal gap
                    ln_x0 = min(o.bbox[0] for o in ln)
                    ln_x1 = max(o.bbox[2] for o in ln)
                    if op.bbox[0] <= ln_x1 + max(24.0, op.font_size_pt * 3.0) and op.bbox[
                        2
                    ] >= ln_x0 - max(24.0, op.font_size_pt * 3.0):
                        ln.append(op)
                        placed = True
                        break
            if not placed:
                lines.append([op])

        # Compute line boxes and group adjacent lines into paragraphs.
        line_boxes: list[tuple[tuple[float, float, float, float], float, bool]] = []
        for ln in lines:
            x0 = min(o.bbox[0] for o in ln)
            y0 = min(o.bbox[1] for o in ln)
            x1 = max(o.bbox[2] for o in ln)
            y1 = max(o.bbox[3] for o in ln)
            fs = max(o.font_size_pt for o in ln)
            is_bold = all(o.is_bold for o in ln if o.text.strip())
            line_boxes.append(((x0, y0, x1, y1), fs, is_bold))

        # Order top-to-bottom within columns (simple mid-page check for 2-col).
        mid_x = analysis.page_width * 0.5
        left_cnt = sum(1 for b, _, _ in line_boxes if b[2] < mid_x + 12.0)
        right_cnt = sum(1 for b, _, _ in line_boxes if b[0] > mid_x - 12.0)
        is_two_col = left_cnt >= 3 and right_cnt >= 3

        def _col_key(
            item: tuple[tuple[float, float, float, float], float, bool],
        ) -> tuple[int, float]:
            b, _, _ = item
            if not is_two_col:
                return (0, -b[3])
            if b[0] < mid_x and b[2] > mid_x + 20.0:
                return (0, -b[3])  # full-width banner
            col = 1 if (b[0] + b[2]) * 0.5 < mid_x else 2
            return (col, -b[3])

        line_boxes.sort(key=_col_key)

        body_fs = sorted(fs for _, fs, _ in line_boxes)[len(line_boxes) // 2]
        groups: list[list[tuple[tuple[float, float, float, float], float, bool]]] = []
        for item in line_boxes:
            b, fs, _ = item
            if not groups:
                groups.append([item])
                continue
            prev_b, prev_fs, _ = groups[-1][-1]
            same_col = not is_two_col or (_col_key(item)[0] == _col_key(groups[-1][-1])[0])
            gap = prev_b[1] - b[3]
            max_gap = max(prev_fs, fs) * 0.95
            same_size = abs(fs - prev_fs) <= max(1.5, prev_fs * 0.18)
            if same_col and -4.0 <= gap <= max_gap and same_size:
                groups[-1].append(item)
            else:
                groups.append([item])

        regions: list[LayoutRegion] = []
        for idx, grp in enumerate(groups):
            gx0 = min(b[0] for b, _, _ in grp)
            gy0 = min(b[1] for b, _, _ in grp)
            gx1 = max(b[2] for b, _, _ in grp)
            gy1 = max(b[3] for b, _, _ in grp)
            gfs = max(fs for _, fs, _ in grp)
            gbold = all(bld for _, _, bld in grp)
            kind = (
                RegionKind.HEADING
                if (len(grp) <= 2 and (gfs >= body_fs * 1.18 or gbold))
                else RegionKind.PARAGRAPH
            )
            regions.append(
                LayoutRegion(
                    kind=kind,
                    bbox=(gx0, gy0, gx1, gy1),
                    score=1.0,
                    reading_order=idx,
                    raw_label=kind.value,
                )
            )
        return regions
