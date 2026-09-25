"""PDF text geometry: pdfium line extraction plus row/column primitives.

Pure geometry layer shared by the rigid typesetter and the VLM coordinate
bridge: pdfium line rects in column-aware reading order, word-soup row glue,
normalized matching text, and small rect helpers. No Typst, no raster
sampling, no ledger policy — safe to unit-test with synthetic rects.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ubt.adapters.pdf.pdfium_gate import pdfium_serialized
from ubt.core.exceptions import DocumentParseError
from ubt.core.ir.models import IRBlock
from ubt.core.policy.layout_policy import (
    CONTROL_RE,
    FOLD_MAP,
    ROW_MERGE_GAP_PT,
    ROW_MERGE_Y_TOL,
    WS_RE,
)


@dataclass
class LineBox:
    """One pdfium text line: text plus its box (bottom-left origin, points)."""

    text: str
    rect: tuple[float, float, float, float]
    table_band: bool = False  # P0: row from a >=3-run band (tabular zone)
    members: tuple[LineBox, ...] = ()  # P0: glued source frags (span slicing)
    font_size: float = 0.0
    bold: bool = False
    italic: bool = False


def _aggregate_line_styles(items: Sequence[LineBox]) -> tuple[float, bool, bool]:
    """Compute character-weighted ground-truth font_size, bold, and italic across fragments."""
    valid = [
        (round(ln.font_size, 2), max(1, len((ln.text or "").strip())), ln.bold, ln.italic)
        for ln in items
        if ln.font_size >= 4.5
    ]
    if not valid:
        fallback_sz = max((ln.font_size for ln in items if ln.font_size > 0.0), default=0.0)
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
        return 0.0, False, False
    normal = [s for s in samples if s[0] >= 4.5] or samples
    best_size = max(s[0] for s in normal)
    is_bold = sum(1 for s in normal if s[1]) >= (len(normal) + 1) // 2
    is_italic = sum(1 for s in normal if s[2]) >= (len(normal) + 1) // 2
    return best_size, is_bold, is_italic


def synthetic_vlm_lines(blocks: Sequence[IRBlock]) -> list[LineBox]:
    """4: LineBoxes from VLM-measured member lines (textless pages)."""
    from ubt.adapters.pdf.vlm.transcribe import synthetic_vlm_lines as _vlm_lines

    return [LineBox(e.text, e.box) for e in _vlm_lines(blocks)]


def union_area(rects: Sequence[tuple[float, float, float, float]]) -> float:
    """Exact union area of rects (x-sweep; footprint metric primitive)."""
    edges = sorted({e for r in rects for e in (r[0], r[2])})
    total = 0.0
    for x0, x1 in zip(edges, edges[1:], strict=False):
        if x1 <= x0:
            continue
        spans = sorted((r[1], r[3]) for r in rects if r[0] <= x0 and r[2] >= x1)
        y = None
        for lo, hi in spans:
            if y is None:
                y, cur = lo, hi
            elif lo <= cur:
                cur = max(cur, hi)
            else:
                total += (x1 - x0) * (cur - y)
                y, cur = lo, hi
        if y is not None:
            total += (x1 - x0) * (cur - y)
    return total


_EXTRACT_LINES_CACHE: dict[tuple[str, int, int], tuple[list[LineBox], tuple[float, float]]] = {}


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
            if n_rects > 600:
                # Embedded vector scatter plots (e.g. matplotlib/tikz marker clouds)
                # can emit thousands of sub-5pt micro-glyphs on a single page,
                # triggering O(K^2) row-fragment merging and 5-minute hangs.
                rects = [
                    r
                    for r in rects
                    if (r[3] - r[1]) >= 5.0 and (r[2] - r[0]) >= 3.0 and (r[3] - r[1]) <= 60.0
                ]
            rects.sort(key=lambda r: (-r[3], r[0]))
            lines = []
            for left, bottom, right, top in rects:
                text = (textpage.get_text_bounded(left, bottom, right, top) or "").strip()
                if text:
                    if n_rects > 600 and len(text) > max(240, int((right - left) * 1.5)):
                        # Skip stacked scatter-plot marker clouds where hundreds of
                        # overlapping point labels fall inside one bounding box.
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
    """
    sized = [(ln.rect[3] - ln.rect[1], ln) for ln in lines]
    heights = sorted(h for h, _ in sized if h > 0)
    med = heights[len(heights) // 2] if heights else 0.0
    core = [ln for h, ln in sized if h >= 0.7 * med]
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
        # P0 table exclusion: a band split into 3+ x-runs is a tabular zone
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
    """Column-aware reading order: cluster lines by x-overlap, then read.

    Full-width lines (headings, captions) never define clusters — they would
    bridge the gutter and collapse everything into one column; they join the
    nearest cluster by x-center instead. Clusters go left-to-right, lines
    inside top-down. Single-column layouts strictly preserve top-down reading.
    """
    if len(lines) <= 1:
        return list(lines)
    work = list(lines)
    span = max((ln.rect[2] for ln in work), default=page_width) - min(
        (ln.rect[0] for ln in work), default=0.0
    )
    if span <= 0.0:
        return sorted(work, key=lambda ln: (-ln.rect[3], ln.rect[0]))

    deferred: list[int] = []
    candidates: list[int] = []
    for idx, ln in enumerate(work):
        w = ln.rect[2] - ln.rect[0]
        if w > 0.7 * span:
            deferred.append(idx)
        else:
            candidates.append(idx)

    # Connected-component clustering of horizontal spans: merge all overlapping intervals
    clusters: list[list[Any]] = []  # [x0, x1, [indices]]
    for idx in candidates:
        x0, _, x1, _ = work[idx].rect
        matching = [i for i, c in enumerate(clusters) if not (x1 < c[0] or x0 > c[1])]
        if not matching:
            clusters.append([x0, x1, [idx]])
        else:
            new_x0 = min(x0, min(clusters[m][0] for m in matching))
            new_x1 = max(x1, max(clusters[m][1] for m in matching))
            new_indices = [idx]
            for m in matching:
                new_indices.extend(clusters[m][2])
            for m in sorted(matching, reverse=True):
                del clusters[m]
            clusters.append([new_x0, new_x1, new_indices])

    # Valid multi-column layout requires at least 2 distinct non-overlapping columns
    # with substantial span width (> 15% of span) and line membership.
    valid_cols = [c for c in clusters if (c[1] - c[0]) > 0.15 * span and len(c[2]) >= 1]
    if len(valid_cols) < 2:
        return sorted(work, key=lambda ln: (-ln.rect[3], ln.rect[0]))

    valid_cols.sort(key=lambda c: c[0])
    assign: list[int] = [-1] * len(work)
    for c_idx, c in enumerate(valid_cols):
        for idx in c[2]:
            assign[idx] = c_idx

    unassigned = [i for i in range(len(work)) if assign[i] < 0]
    for idx in unassigned:
        cx = (work[idx].rect[0] + work[idx].rect[2]) / 2
        best = min(
            range(len(valid_cols)),
            key=lambda i: (
                0.0
                if valid_cols[i][0] <= cx <= valid_cols[i][1]
                else min(abs(cx - valid_cols[i][0]), abs(cx - valid_cols[i][1]))
            ),
        )
        assign[idx] = best

    order = sorted(
        range(len(work)),
        key=lambda i: (valid_cols[assign[i]][0], -work[i].rect[3]),
    )
    return [work[i] for i in order]


__all__ = [
    "LineBox",
    "column_order",
    "dehyph",
    "extract_lines",
    "merge_row_fragments",
    "synthetic_vlm_lines",
    "union_area",
]
