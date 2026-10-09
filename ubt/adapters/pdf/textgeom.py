"""PDF text geometry: pdfium line extraction plus row/column primitives.

Pure geometry layer shared by the PDF compositor and the VLM coordinate
bridge: pdfium line rects in column-aware reading order, word-soup row glue,
normalized matching text, and small rect helpers. No Typst, no raster
sampling, no ledger policy — safe to unit-test with synthetic rects.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ubt.adapters.pdf.pdfium_gate import pdfium_serialized
from ubt.core.exceptions import DocumentParseError
from ubt.core.policy.layout_policy import (
    CONTROL_RE,
    FOLD_MAP,
    ROW_MERGE_GAP_PT,
    ROW_MERGE_TALL_FACTOR,
    ROW_MERGE_Y_TOL,
    WS_RE,
)
from ubt.model.span import BBox

logger = logging.getLogger(__name__)


@dataclass
class LineBox:
    """One pdfium text line: text plus its box (bottom-left origin, points)."""

    text: str
    rect: tuple[float, float, float, float]
    table_band: bool = False  # row from a >=3-run band (tabular zone)
    members: tuple[LineBox, ...] = ()  # glued source frags (span slicing)
    font_size: float = 0.0
    bold: bool = False
    italic: bool = False


def box_area(x0: float, y0: float, x1: float, y1: float) -> float:
    """Area of an axis-aligned box, clamped to zero for an inverted span."""
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Intersection-over-union of two boxes, 0.0 when they do not overlap."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = box_area(ix0, iy0, ix1, iy1)
    if inter <= 0:
        return 0.0
    union = box_area(*a) + box_area(*b) - inter
    return inter / union if union > 0 else 0.0


def _aggregate_line_styles(items: Sequence[LineBox]) -> tuple[float, bool, bool]:
    """Compute character-weighted ground-truth font_size, bold, and italic across fragments."""
    valid = [
        (round(ln.font_size, 2), max(1, len((ln.text or "").strip())), ln.bold, ln.italic)
        for ln in items
        if ln.font_size >= 4.5
    ]
    if not valid:
        fallback_sz = max((ln.font_size for ln in items if ln.font_size >= 4.5), default=0.0)
        any_bold = any(ln.bold for ln in items)
        any_italic = any(ln.italic for ln in items)
        return fallback_sz, any_bold, any_italic

    size_weights: dict[float, int] = {}
    total_w = 0
    bold_w = 0
    italic_w = 0
    for fsz, w, is_bold, is_italic in valid:
        size_weights[fsz] = size_weights.get(fsz, 0) + w
        total_w += w
        if is_bold:
            bold_w += w
        if is_italic:
            italic_w += w
    best_size = max(size_weights.items(), key=lambda kv: (kv[1], kv[0]))[0]
    return best_size, (bold_w >= 0.5 * total_w), (italic_w >= 0.5 * total_w)


#: Public alias: the char-weighted ground-truth style of a group of source lines.
#: Extraction-side validation reads this to check a block's role against the
#: typography actually present on the page.
aggregate_line_styles = _aggregate_line_styles


def _probe_rect_font_style(
    textpage: Any, left: float, bottom: float, right: float, top: float
) -> tuple[float, bool, bool]:
    """Read ground-truth font size, weight, and italic style from pdfium C API."""
    import ctypes

    import pypdfium2.raw as pdfium_c

    y_mid = (bottom + top) / 2.0
    width = max(0.0, right - left)
    samples: list[tuple[float, bool, bool]] = []
    buf = ctypes.create_string_buffer(256)
    flags = ctypes.c_int(0)

    for frac in (0.15, 0.35, 0.55, 0.75):
        x = left + max(1.5, width * frac)
        idx = int(pdfium_c.FPDFText_GetCharIndexAtPos(textpage, x, y_mid, 6.0, 6.0))
        if idx < 0:
            continue
        fsize = float(pdfium_c.FPDFText_GetFontSize(textpage, idx))
        if fsize <= 0.0:
            continue
        fweight = int(pdfium_c.FPDFText_GetFontWeight(textpage, idx))
        flags.value = 0
        pdfium_c.FPDFText_GetFontInfo(textpage, idx, buf, 256, ctypes.byref(flags))
        fname = buf.value.decode("utf-8", errors="ignore").lower()
        is_bold = (fweight >= 600) or any(
            k in fname for k in ("bold", "heavy", "black", "demi", "semibold")
        )
        is_italic = any(k in fname for k in ("italic", "oblique", "slanted")) or bool(
            flags.value & 0x40
        )
        samples.append((fsize, is_bold, is_italic))

    if not samples:
        rect_h = max(0.0, top - bottom)
        return (rect_h if rect_h >= 4.5 else 0.0), False, False
    normal = [s for s in samples if s[0] >= 4.5]
    if normal:
        best_size = max(s[0] for s in normal)
    else:
        # FPDFText_GetFontSize returned 1.0 (or <4.5) because font was scaled via text matrix Tm.
        # Fall back to the physical line bounding box height.
        rect_h = max(0.0, top - bottom)
        best_size = rect_h if rect_h >= 4.5 else 0.0
    is_bold = sum(1 for s in samples if s[1]) >= (len(samples) + 1) // 2
    is_italic = sum(1 for s in samples if s[2]) >= (len(samples) + 1) // 2
    return best_size, is_bold, is_italic


@dataclass(frozen=True, slots=True)
class CharStyle:
    """One pdfium character: its box, glyph, and ground-truth style.

    The line-level probes (:func:`_probe_rect_font_style`) collapse a line to a
    single style, losing *within-line* changes -- a blue citation link, a raised
    footnote dagger. This keeps the per-character facts so a caller can recover
    those runs.
    """

    rect: tuple[float, float, float, float]
    text: str
    font_size: float
    bold: bool
    color_hex: str | None


def _hex_if_colored(red: int, green: int, blue: int) -> str | None:
    """``#RRGGBB`` for a non-black fill, ``None`` for default black."""
    if (red, green, blue) == (0, 0, 0):
        return None
    return f"#{red:02x}{green:02x}{blue:02x}"


@pdfium_serialized
def extract_char_styles(pdf_path: Path, page_no: int) -> list[CharStyle]:
    """Per-character box/text/style for one page, straight from pdfium.

    Characters arrive in content-stream order, which for a single block is
    reading order. A character with no box (a synthetic space) is skipped.
    """
    import ctypes

    import pypdfium2 as pdfium
    import pypdfium2.raw as pdfium_c

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        if not 1 <= page_no <= len(pdf):
            raise DocumentParseError(f"page {page_no} out of range in {pdf_path.name}")
        page = pdf[page_no - 1]
        try:
            textpage = page.get_textpage()
            try:
                out: list[CharStyle] = []
                for index in range(textpage.count_chars()):
                    code = pdfium_c.FPDFText_GetUnicode(textpage, index)
                    if code == 0 or code > 0x10FFFF:
                        continue
                    left = ctypes.c_double()
                    right = ctypes.c_double()
                    bottom = ctypes.c_double()
                    top = ctypes.c_double()
                    # pdfium's argument order is (left, right, bottom, top).
                    if not pdfium_c.FPDFText_GetCharBox(
                        textpage,
                        index,
                        ctypes.byref(left),
                        ctypes.byref(right),
                        ctypes.byref(bottom),
                        ctypes.byref(top),
                    ):
                        continue
                    size = float(pdfium_c.FPDFText_GetFontSize(textpage, index))
                    weight = int(pdfium_c.FPDFText_GetFontWeight(textpage, index))
                    red = ctypes.c_uint()
                    green = ctypes.c_uint()
                    blue = ctypes.c_uint()
                    alpha = ctypes.c_uint()
                    pdfium_c.FPDFText_GetFillColor(
                        textpage,
                        index,
                        ctypes.byref(red),
                        ctypes.byref(green),
                        ctypes.byref(blue),
                        ctypes.byref(alpha),
                    )
                    out.append(
                        CharStyle(
                            rect=(left.value, bottom.value, right.value, top.value),
                            text=chr(code),
                            font_size=size,
                            bold=weight >= 600,
                            color_hex=_hex_if_colored(red.value, green.value, blue.value),
                        )
                    )
                return out
            finally:
                textpage.close()
        finally:
            page.close()
    finally:
        pdf.close()


def _char_center_in_box(rect: tuple[float, float, float, float], box: BBox) -> bool:
    cx = (rect[0] + rect[2]) / 2.0
    cy = (rect[1] + rect[3]) / 2.0
    return box[0] - 1.0 <= cx <= box[2] + 1.0 and box[1] - 1.0 <= cy <= box[3] + 1.0


def styled_runs_in_box(
    chars: Sequence[CharStyle],
    box: BBox,
    *,
    superscript_ratio: float = 0.8,
) -> list[tuple[str, bool, bool, bool, str | None]]:
    """Merge a page's characters inside ``box`` into ``(text, bold, italic,
    superscript, color_hex)`` runs of constant style.

    Only non-default runs are returned (bold / superscript / coloured): plain
    body text needs no run and would only bloat the block's style. A raised
    marker is a glyph smaller than ``superscript_ratio`` of the box's median size
    (the footnote daggers are ~0.73x the body); a whitespace glyph is neutral and
    rides the surrounding run rather than starting one of its own (pdfium reports
    synthetic spaces with a degenerate 1pt box).
    """
    inside = [c for c in chars if c.text and _char_center_in_box(c.rect, box)]
    if not inside:
        return []
    sizes = sorted(c.font_size for c in inside if c.font_size >= 4.5 and not c.text.isspace())
    base = sizes[len(sizes) // 2] if sizes else 0.0
    runs: list[tuple[str, bool, bool, bool, str | None]] = []
    buffer: list[str] = []
    style: tuple[bool, bool, bool, str | None] | None = None

    def _flush() -> None:
        nonlocal buffer, style
        # A trailing space rides the run it follows; keep it out of the run text
        # so the render can still locate the span in a CJK target, where the
        # source's space is usually dropped or replaced by full-width punctuation
        # ("50.0% " -> "50.0%；"). Internal spaces are preserved.
        text = "".join(buffer).rstrip()
        if text and style is not None and style != (False, False, False, None):
            runs.append((text, style[0], style[1], style[2], style[3]))
        buffer = []

    for char in inside:
        if char.text.isspace():
            if buffer:
                buffer.append(char.text)
            continue
        raised = base > 0 and char.font_size >= 4.5 and char.font_size < superscript_ratio * base
        current = (char.bold, False, raised, char.color_hex)
        if style is not None and current != style:
            _flush()
        style = current
        buffer.append(char.text)
    _flush()
    return runs


_EXTRACT_LINES_CACHE: dict[tuple[str, int, int], tuple[list[LineBox], tuple[float, float]]] = {}

# --- Vector-figure micro-glyph cleanup -------------------------------------
# A matplotlib/tikz scatter cloud emits thousands of sub-5pt marker glyphs on
# one page. They are not text lines, but they cost like them: every rect is
# laid against every row band in ``merge_row_fragments`` and inside
# ``_glue_run``'s duplicate check, so K≈10^4 is an O(K^2) multi-minute hang on
# a single page. The cleanup below bounds K by dropping rects that cannot be a
# text line run -- and *reports* what it dropped, because a real page can carry
# a footnote marker or a superscript the same size as the debris.
#: A page needs at least this many rects before the cleanup is considered. A
#: page of ordinary prose has tens to a few hundred runs; this keeps the
#: rectangle sweep and the (O(K^2)) glue off every ordinary page.
_CLOUD_MIN_RECTS = 600
#: A run is at least this tall/narrow: a 10pt body line is ~9pt tall, and a
#: glyph run narrower than 3pt is a lone mark, never a line of text.
_DEBRIS_MIN_HEIGHT_PT = 5.0
_DEBRIS_MIN_WIDTH_PT = 3.0
#: A text line is never this tall (that is a ~6-line box); only rules, sidebars
#: and watermarks are.
_DEBRIS_MAX_HEIGHT_PT = 60.0
#: ...and the debris must be this numerous. A page that is merely *dense* -- a
#: CJK page pdfium splits per character, a wide table -- is above the rect
#: count with almost no debris rects and now keeps every one of them; before,
#: any page over the count had its sub-5pt rects dropped (its superscripts and
#: footnote marks) with nothing said about it.
_DEBRIS_MIN_COUNT = 300


def _is_debris_rect(rect: tuple[float, float, float, float]) -> bool:
    """True for a rect the micro-glyph cleanup would drop."""
    height = rect[3] - rect[1]
    width = rect[2] - rect[0]
    return (
        height < _DEBRIS_MIN_HEIGHT_PT
        or width < _DEBRIS_MIN_WIDTH_PT
        or height > _DEBRIS_MAX_HEIGHT_PT
    )


def _drop_scatter_debris(
    rects: list[tuple[float, float, float, float]],
) -> tuple[list[tuple[float, float, float, float]], int]:
    """Drop a vector figure's micro-glyph debris; ``(kept, dropped_count)``.

    ``dropped_count`` is zero unless the page is a genuine marker cloud (many
    rects, hundreds of them debris-sized). A caller that sees a nonzero count
    must report it: those rects, and whatever text the extractor placed under
    them, are gone from the line list.
    """
    if len(rects) <= _CLOUD_MIN_RECTS:
        return rects, 0
    debris = sum(1 for rect in rects if _is_debris_rect(rect))
    if debris < _DEBRIS_MIN_COUNT:
        return rects, 0
    kept = [rect for rect in rects if not _is_debris_rect(rect)]
    return kept, len(rects) - len(kept)


@pdfium_serialized
def extract_text_rects(pdf_path: Path, page_no: int) -> list[tuple[float, float, float, float]]:
    """Raw pdfium text rects for one page, *unmerged*.

    ``extract_lines`` merges rects that share a baseline into one reading-order
    line — which folds two overprinted runs at the same position into a single
    rect and hides the very overlap a render check looks for. This returns the
    raw per-run rects instead.
    """
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        if not 1 <= page_no <= len(pdf):
            raise DocumentParseError(f"page {page_no} out of range in {pdf_path.name}")
        page = pdf[page_no - 1]
        try:
            textpage = page.get_textpage()
            try:
                rects: list[tuple[float, float, float, float]] = []
                for idx in range(textpage.count_rects(0, -1)):
                    x0, y0, x1, y1 = textpage.get_rect(idx)
                    rects.append((float(x0), float(y0), float(x1), float(y1)))
                return rects
            finally:
                textpage.close()
        finally:
            page.close()
    finally:
        pdf.close()


@pdfium_serialized
def extract_lines(pdf_path: Path, page_no: int) -> tuple[list[LineBox], tuple[float, float]]:
    """pdfium line rects in column-aware reading order + page size."""
    import pypdfium2 as pdfium

    cache_key: tuple[str, int, int] | None = None
    try:
        resolved = str(pdf_path.resolve())
        mtime_ns = pdf_path.stat().st_mtime_ns
        cache_key = (resolved, mtime_ns, page_no)
        cached = _EXTRACT_LINES_CACHE.get(cache_key)
        if cached is not None:
            return list(cached[0]), cached[1]
    except OSError:
        cache_key = None

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        if not 1 <= page_no <= len(pdf):
            raise DocumentParseError(f"page {page_no} out of range in {pdf_path.name}")
        page = pdf[page_no - 1]
        # ``get_rect`` returns UNROTATED user-space rects while
        # ``get_width/height`` return the rotated display size. Report the
        # unrotated mediabox size so the returned (lines, size) pair shares one
        # frame; callers that need the display orientation must read
        # ``page.get_rotation()`` themselves.
        mb = page.get_mediabox()
        size = (float(mb[2]) - float(mb[0]), float(mb[3]) - float(mb[1]))
        textpage = page.get_textpage()
        try:
            n_rects = textpage.count_rects(0, -1)
            rects = [textpage.get_rect(i) for i in range(n_rects)]
            rects, dropped_rects = _drop_scatter_debris(rects)
            rects.sort(key=lambda r: (-r[3], r[0]))
            lines = []
            skipped_long = 0
            for left, bottom, right, top in rects:
                text = (textpage.get_text_bounded(left, bottom, right, top) or "").strip()
                if text:
                    if dropped_rects and len(text) > max(240, int((right - left) * 1.5)):
                        # Stacked scatter-plot marker clouds put hundreds of
                        # overlapping point labels inside one bounding box, so
                        # the joined text is a duplication, not a line.
                        skipped_long += 1
                        continue
                    fsz, is_bold, is_italic = _probe_rect_font_style(
                        textpage, left, bottom, right, top
                    )
                    lines.append(
                        LineBox(
                            text,
                            (left, bottom, right, top),
                            font_size=fsz,
                            bold=is_bold,
                            italic=is_italic,
                        )
                    )
            if dropped_rects or skipped_long:
                # Never silent: a sub-5pt rect on a figure page can equally be a
                # superscript or a footnote mark, and this drop is unrecoverable
                # downstream. One line per page, not per rect.
                logger.warning(
                    "textgeom: %s page %s has %d text rects and looks like a "
                    "vector/marker cloud; dropped %d debris rects (<%gpt, <%gpt "
                    "wide, >%gpt tall) and %d over-long labels — text under those "
                    "rects is not extracted",
                    pdf_path.name,
                    page_no,
                    n_rects,
                    dropped_rects,
                    _DEBRIS_MIN_HEIGHT_PT,
                    _DEBRIS_MIN_WIDTH_PT,
                    _DEBRIS_MAX_HEIGHT_PT,
                    skipped_long,
                )
            merged = merge_row_fragments(lines)
            result = (column_order(merged, size[0]), size)
            if cache_key is not None:
                if len(_EXTRACT_LINES_CACHE) >= 256:
                    _EXTRACT_LINES_CACHE.clear()
                _EXTRACT_LINES_CACHE[cache_key] = (list(result[0]), result[1])
            return result
        finally:
            textpage.close()
            page.close()
    finally:
        pdf.close()


# pdfium double emission ("Id Id" for "Ids", "Vd Vd" for "Vd"): collapse
# adjacent identical whitespace-separated words before matching. Word
# chars only, so "[8] [8]" citations and "a, a" lists never collapse.
_DUP_WORD_RE = re.compile(r"\b(\w+) \1\b")

_SUPERSCRIPT_CHARS = {
    "0": "⁰",
    "1": "¹",
    "2": "²",
    "3": "³",
    "4": "⁴",
    "5": "⁵",
    "6": "⁶",
    "7": "⁷",
    "8": "⁸",
    "9": "⁹",
    "+": "⁺",
    "-": "⁻",
    "=": "⁼",
    "(": "⁽",
    ")": "⁾",
    ",": "˒",
}
SUPERSCRIPT_ENCODE_MAP = str.maketrans(_SUPERSCRIPT_CHARS)
SUPERSCRIPT_DECODE_MAP = str.maketrans({v: k for k, v in _SUPERSCRIPT_CHARS.items()})
_SUPERSCRIPT_ELIGIBLE_RE = re.compile(r"^[0-9,+\-=()\s]+$")


def to_unicode_superscript(text: str) -> str:
    """Convert an ASCII numeric/punctuation superscript string into Unicode superscripts."""
    return (text or "").strip().translate(SUPERSCRIPT_ENCODE_MAP)


def dehyph(text: str) -> str:
    """Normalize for cross-engine matching: fold quotes/dashes/ligatures,
    strip control chars, collapse pdfium double-emitted adjacent words
    (``Id Id`` for ``Ids``/``Id``), then strip whitespace AND hyphens
    (pdfium keeps line-break hyphens and drops inter-word spaces the parser
    preserves).

    Dedup is case-sensitive and applies to both rows and block sources,
    so genuine repeats (``that that``) still match their own block.
    """
    folded = (text or "").translate(SUPERSCRIPT_DECODE_MAP).translate(FOLD_MAP)
    folded = CONTROL_RE.sub("", folded)
    # Quote unification: pdfium emits curly doubles while docling normalizes
    # to straight singles — FOLD_MAP folds each family separately, so the two
    # engines still disagree afterwards. Matching only.
    folded = folded.replace('"', "'")
    prev = None
    for _ in range(3):
        collapsed = _DUP_WORD_RE.sub(r"\1", folded)
        if collapsed in (folded, prev):
            break
        prev, folded = folded, collapsed
    return WS_RE.sub("", folded).replace("-", "")


def merge_row_fragments(
    lines: Sequence[LineBox],
    max_gap_pt: float = ROW_MERGE_GAP_PT,
    y_tol: float = ROW_MERGE_Y_TOL,
) -> list[LineBox]:
    """Merge fragmented word boxes back into visual rows for complex pages.

    pdfium emits one rect per glyph run; on table/scan pages a "line" arrives
    as dozens of word fragments, which turns covers into confetti and lets
    substring pairing assemble paragraphs from unrelated words. Two passes:
    normal-height fragments define row bands (adjacent rows never overlap,
    so bands cannot chain); small fragments (sub/superscripts) attach to the
    topmost band they overlap without expanding it. Each band then splits on
    x-gaps wider than ``max_gap_pt`` and glues with spaces (``dehyph``
    strips spaces for matching, so pairing is unaffected). Cross-gutter
    merges cannot happen (real gutters dwarf the gap cap). Runs before
    ``column_order``.

    Fragments far taller than the median row (vertical sidebar text, rotated
    watermarks) are excluded from band formation entirely: seeded as a band,
    every normal row overlapping their huge y-span would join it and collapse
    half the page into one glued row (arXiv 2609.32391). Each stays its own
    sealed band that neither pass may join.
    """
    sized = [(ln.rect[3] - ln.rect[1], ln) for ln in lines]
    heights = sorted(h for h, _ in sized if h > 0)
    med = heights[len(heights) // 2] if heights else 0.0
    # A median of 0 means most fragments carry no height; the tall cut would
    # seal everything, so keep the historical behaviour on such pages.
    tall_cut = ROW_MERGE_TALL_FACTOR * med if med > 0 else float("inf")
    core = [ln for h, ln in sized if 0.7 * med <= h <= tall_cut]
    tall = [ln for h, ln in sized if h > tall_cut]
    small = [ln for h, ln in sized if h < 0.7 * med]
    bands: list[dict[str, Any]] = []  # top-down: {"y0","y1","frags"}
    for ln in sorted(core, key=lambda v: (-v.rect[3], v.rect[0])):
        x0, y0, x1, y1 = ln.rect
        h = y1 - y0
        hit = None
        for band in bands:
            if min(y1, band["y1"]) - max(y0, band["y0"]) >= y_tol * h:
                hit = band
                break
        if hit is None:
            bands.append({"y0": y0, "y1": y1, "frags": [ln]})
        else:
            hit["frags"].append(ln)
    for ln in sorted(small, key=lambda v: (-v.rect[3], v.rect[0])):
        x0, y0, x1, y1 = ln.rect
        h = y1 - y0
        hit = None
        for band in bands:
            if h > 0 and min(y1, band["y1"]) - max(y0, band["y0"]) >= y_tol * h:
                hit = band
                break
        if hit is None:
            bands.append({"y0": y0, "y1": y1, "frags": [ln]})
        else:
            hit["frags"].append(ln)
    # Sealed bands go in last so neither pass above can ever join them.
    for ln in tall:
        bands.append({"y0": ln.rect[1], "y1": ln.rect[3], "frags": [ln]})
    merged: list[LineBox] = []
    for band in bands:
        row = sorted(band["frags"], key=lambda v: v.rect[0])
        runs: list[list[LineBox]] = []
        run: list[LineBox] = []
        for ln in row:
            if run and ln.rect[0] - run[-1].rect[2] > max_gap_pt:
                runs.append(run)
                run = [ln]
            else:
                run.append(ln)
        if run:
            runs.append(run)
        # Table exclusion: a band split into 3+ x-runs is a tabular zone
        # (body columns never exceed 2 runs per band). Its rows stay out of
        # zone matching entirely — tables remain source-visible instead of
        # wearing prose translations at cell positions.
        tabular = len(runs) >= 3
        for run in runs:
            glued = _glue_run(run)
            if tabular:
                glued = LineBox(
                    glued.text,
                    glued.rect,
                    True,
                    glued.members,
                    font_size=glued.font_size,
                    bold=glued.bold,
                    italic=glued.italic,
                )
            merged.append(glued)
    return merged


#: Horizontal slack for treating two fragments as sharing an edge. pdfium's
#: ``get_text_bounded`` selects every character whose box *intersects* the query
#: rect, so a fragment over-reads the glyph sitting on its right edge into the
#: next fragment: the two rects then overlap in x (or abut, gap == 0). Real
#: space-separated words leave a small positive gap (the space's width), so a
#: sub-point tolerance separates a double-count from two adjacent words.
_OVERREAD_X_TOL_PT = 0.05


def _merge_boundary_overreads(fragments: list[LineBox]) -> list[LineBox]:
    """Fold a fragment's duplicated boundary run back into its left neighbour.

    ``get_text_bounded`` returns a character when its box intersects the query
    rect, so two adjacent fragments both report the glyph on their shared edge:
    ``"creating 𝑚"`` + ``"𝑚"`` -> ``"creating 𝑚 𝑚"``, ``"It is D"`` +
    ``"Definition 33"`` -> ``"It is D Definition 33"``. Those fragments overlap
    in x and share a baseline; the longest run where the left text ends with the
    right text's prefix is the double-count, so it is spliced out (no separating
    space) and the two collapse into one.

    The x-overlap gate is what keeps genuine adjacent words apart: "dynamic" and
    "composition" collide on ``c`` but their rects are separated by the space, so
    they are never folded.
    """
    merged: list[LineBox] = []
    for ln in fragments:
        if merged:
            prev = merged[-1]
            px0, py0, px1, py1 = prev.rect
            x0, y0, x1, y1 = ln.rect
            if (
                x0 - px1 <= _OVERREAD_X_TOL_PT
                and min(py1, y1) - max(py0, y0) > 0
                and ln.text
                and prev.text
            ):
                a, b = prev.text, ln.text
                for k in range(min(len(a), len(b)), 0, -1):
                    if a.endswith(b[:k]):
                        merged[-1] = dataclasses.replace(
                            prev,
                            text=a + b[k:],
                            rect=(min(px0, x0), min(py0, y0), max(px1, x1), max(py1, y1)),
                        )
                        break
                else:
                    merged.append(ln)
                continue
        merged.append(ln)
    return merged


def _glue_run(run: list[LineBox]) -> LineBox:
    """One row run into a single LineBox (union box, space-joined text).

    pdfium double-emits sub/superscript glyphs (same or split text twice
    at overlapping boxes, e.g. "Io"+"Ioff"); gluing both would bake "IoIoff"
    into the row and break substring pairing, so a fragment nested in a kept
    box with nested text is dropped (longest wins). Genuine repetitions sit
    apart and survive.
    """
    if len(run) == 1:
        return run[0]
    kept: list[LineBox] = []
    kept_norms: list[str] = []
    for ln in run:
        nl = dehyph(ln.text)
        dup = False
        x0, y0, x1, y1 = ln.rect
        for i, prev in enumerate(kept):
            px0, py0, px1, py1 = prev.rect
            if px1 <= x0 or x1 <= px0:
                continue
            np = kept_norms[i]
            if not nl or not np:
                continue
            inter = max(0.0, min(px1, x1) - max(px0, x0)) * max(0.0, min(py1, y1) - max(py0, y0))
            smaller = min((px1 - px0) * (py1 - py0), (x1 - x0) * (y1 - y0))
            if smaller > 0 and inter / smaller >= 0.4 and (nl in np or np in nl):
                dup = True
                if len(nl) > len(np):
                    kept[i] = ln
                    kept_norms[i] = nl
                break
        if not dup:
            kept.append(ln)
            kept_norms.append(nl)
    kept = _merge_boundary_overreads(kept)
    x0 = min(ln.rect[0] for ln in kept)
    y0 = min(ln.rect[1] for ln in kept)
    x1 = max(ln.rect[2] for ln in kept)
    y1 = max(ln.rect[3] for ln in kept)
    fsz, is_bold, is_italic = _aggregate_line_styles(kept)
    glued_text = _join_fragments_with_superscripts(kept, fsz)
    return LineBox(
        glued_text,
        (x0, y0, x1, y1),
        members=tuple(kept),
        font_size=fsz,
        bold=is_bold,
        italic=is_italic,
    )


def _join_fragments_with_superscripts(kept: Sequence[LineBox], main_fsz: float) -> str:
    """Join row fragments while converting raised, smaller-font fragments into Unicode superscripts."""
    if len(kept) < 2 or main_fsz < 4.5:
        return " ".join(ln.text for ln in kept)
    main_frags = [ln for ln in kept if ln.font_size >= 0.8 * main_fsz]
    if not main_frags:
        return " ".join(ln.text for ln in kept)
    main_y0 = min(ln.rect[1] for ln in main_frags)
    main_y1 = max(ln.rect[3] for ln in main_frags)
    main_h = max(1.0, main_y1 - main_y0)

    tokens: list[str] = []
    attach_next = False
    for idx, ln in enumerate(kept):
        raw = (ln.text or "").strip()
        if not raw:
            continue
        is_super = (
            0.0 < ln.font_size <= 0.78 * main_fsz
            and (ln.rect[1] - main_y0) >= 0.18 * main_h
            and bool(_SUPERSCRIPT_ELIGIBLE_RE.match(raw))
        )
        if not is_super:
            if attach_next and tokens:
                tokens[-1] = tokens[-1] + raw
                attach_next = False
            else:
                tokens.append(raw)
            continue

        sup_str = to_unicode_superscript(raw)
        # Decide prefix vs suffix attachment from horizontal neighbor gaps
        is_prefix = False
        if idx == 0:
            is_prefix = True
        elif idx + 1 < len(kept):
            left_gap = max(0.0, ln.rect[0] - kept[idx - 1].rect[2])
            right_gap = max(0.0, kept[idx + 1].rect[0] - ln.rect[2])
            if left_gap > right_gap + 3.0:
                is_prefix = True

        if is_prefix:
            tokens.append(sup_str)
            attach_next = True
        elif tokens:
            tokens[-1] = tokens[-1] + sup_str
            attach_next = False
        else:
            tokens.append(sup_str)
            attach_next = False
    return " ".join(tokens)


def column_order(lines: Sequence[LineBox], page_width: float) -> list[LineBox]:
    """Column-aware reading order: split at the gutter, then read left-to-right.

    A line only defines column membership when it does not span the gutter;
    full-width lines (headings, captions) are routed to the nearer column by
    x-center so they cannot bridge the two columns into one. Clusters go
    left-to-right, lines inside top-down. Single-column layouts strictly
    preserve top-down reading.
    """
    if len(lines) <= 1:
        return list(lines)
    work = list(lines)
    span = max((ln.rect[2] for ln in work), default=page_width) - min(
        (ln.rect[0] for ln in work), default=0.0
    )
    if span <= 0.0:
        return sorted(work, key=lambda ln: (-ln.rect[3], ln.rect[0]))

    # Full-width lines (w > 0.7 * span) are never column candidates: they would
    # bridge the gutter. They are routed by x-center below.
    candidates = [idx for idx, ln in enumerate(work) if (ln.rect[2] - ln.rect[0]) <= 0.7 * span]
    if len(candidates) < 2:
        return sorted(work, key=lambda ln: (-ln.rect[3], ln.rect[0]))

    def center(idx: int) -> float:
        return (work[idx].rect[0] + work[idx].rect[2]) / 2.0

    def crossings(split: float) -> int:
        return sum(1 for idx in candidates if work[idx].rect[0] < split < work[idx].rect[2])

    def side_span(idxs: list[int]) -> float:
        if not idxs:
            return 0.0
        return max(work[i].rect[2] for i in idxs) - min(work[i].rect[0] for i in idxs)

    # The gutter is the x crossed by the fewest candidate lines. A single-column
    # page has no low-crossing cut (any mid-column split severs many lines); a
    # two-column page dips sharply at the real gutter. Naive connected-component
    # clustering would merge every overlapping interval, so a single
    # gutter-spanning line (a centered caption, a hanging-indented equation line)
    # would chain the two columns into one component and collapse the page to
    # top-down reading order.
    split_points = sorted(
        {work[idx].rect[0] for idx in candidates} | {work[idx].rect[2] for idx in candidates}
    )
    best_split: float | None = None
    best_crossings = len(candidates) + 1
    for split in split_points:
        left = [i for i in candidates if center(i) < split]
        right = [i for i in candidates if center(i) >= split]
        if not left or not right:
            continue
        if side_span(left) <= 0.15 * span or side_span(right) <= 0.15 * span:
            continue
        crossed = crossings(split)
        if crossed < best_crossings:
            best_crossings = crossed
            best_split = split
    # A real gutter is crossed by few lines; a mid-column cut is not. Without a
    # low-crossing split this is a single-column page: keep top-down order.
    if best_split is None or best_crossings > 0.25 * len(candidates):
        return sorted(work, key=lambda ln: (-ln.rect[3], ln.rect[0]))

    left_cands = [i for i in candidates if center(i) < best_split]
    right_cands = [i for i in candidates if center(i) >= best_split]
    # A banner must clear the TOP of BOTH columns (y grows upward, so the
    # higher column's max top); a footer must sit below the BOTTOM of both (the
    # lower column's min bottom). Inverting these would misread the longer
    # column's lower lines as footers when the two columns have unequal extents,
    # pushing them to the end of the page.
    top_ceiling = max(
        max(work[i].rect[3] for i in left_cands), max(work[i].rect[3] for i in right_cands)
    )
    bottom_floor = min(
        min(work[i].rect[1] for i in left_cands), min(work[i].rect[1] for i in right_cands)
    )

    def sort_key(i: int) -> tuple[int, int, float, float]:
        ln = work[i]
        # Lines placed above both columns (banners/headings) read first.
        if ln.rect[1] >= top_ceiling:
            return (0, 0, -ln.rect[3], ln.rect[0])
        # Lines placed below both columns (footers/notes) read last.
        if ln.rect[3] <= bottom_floor:
            return (2, 0, -ln.rect[3], ln.rect[0])
        col = 0 if center(i) < best_split else 1
        return (1, col, -ln.rect[3], ln.rect[0])

    order = sorted(range(len(work)), key=sort_key)
    return [work[i] for i in order]


__all__ = [
    "LineBox",
    "box_area",
    "box_iou",
    "column_order",
    "dehyph",
    "extract_lines",
    "merge_row_fragments",
]
