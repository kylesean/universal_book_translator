"""Vector-diagram region detection on a PDF content stream.

A "diagram" here is a cluster of painted vector paths -- a figure, chart or
schematic -- found by walking the page's content stream rather than by raster
sampling. Detection is pure geometry and needs no external binary, so the
bilingual advisor can ask which pages carry figures before any draft is paid
for (see :mod:`ubt.core.engine.stages.advisory`).

Two entry points:

* :func:`detect_diagram_regions` -- painted-path bboxes for one page, merged
  into regions and filtered so text panels and picture noise do not qualify.
* :func:`detect_figure_pages` -- the page numbers of a document that carry at
  least one such region, with TABLE bboxes excluded (tables have their own
  pipeline).

Label text is located through ``pdftotext -bbox``; :func:`is_text_panel` uses
those word boxes to reject a candidate region that is mostly running text.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ubt.adapters.pdf import pdf_struct
from ubt.adapters.pdf.stream_strip import mul_matrix
from ubt.core.env import subprocess_env

logger = logging.getLogger(__name__)

# Minimum diagram edge length; smaller regions are picture noise, not diagrams.
_MIN_DIAGRAM_EDGE_PT = 20.0


def detect_figure_pages(
    pdf_path: Path | str,
    blocks: list[Any] | None = None,
) -> set[int]:
    """Return 1-based page numbers carrying vector-diagram regions.

    Used by the Phase-1 bilingual advisor (pre-draft, zero-LLM). TABLE
    block bboxes found in ``blocks`` are excluded (tables have their own
    pipeline). Never raises: any read failure yields an empty set.
    """
    from ubt.core.ir.models import BlockType as _BlockType

    try:
        page_count = pdf_struct.page_count(pdf_path)
    except Exception as exc:
        logger.debug("Figure-page detection cannot read '%s': %s", pdf_path, exc)
        return set()
    tables_by_page: dict[int, list[tuple[float, float, float, float]]] = {}
    for block in blocks or []:
        bbox = getattr(block, "bbox", None)
        if (
            getattr(block, "block_type", None) == _BlockType.TABLE
            and bbox is not None
            and getattr(bbox, "page", 0) > 0
            and bbox.x1 > bbox.x0
            and bbox.y1 > bbox.y0
        ):
            tables_by_page.setdefault(bbox.page, []).append((bbox.x0, bbox.y0, bbox.x1, bbox.y1))
    pages: set[int] = set()
    for page_no in range(1, page_count + 1):
        try:
            if detect_diagram_regions(pdf_path, page_no, exclude=tables_by_page.get(page_no)):
                pages.add(page_no)
        except Exception as exc:
            logger.debug("Figure-page detection failed on p%d: %s", page_no, exc)
    return pages


# ---------------------------------------------------------------------------
# Vector-diagram region detection (inline content-stream parsing)
# ---------------------------------------------------------------------------

# Paint operators that commit the current path (``n`` discards it).
_PAINT_OPS = frozenset({"S", "s", "f", "F", "f*", "B", "B*", "b", "b*"})
# Tokenizer for the small operator subset we track.
_NUM_RE = re.compile(rb"^[+\-.]?[\d.]+$")


def _parse_content_drawings(data: bytes) -> list[tuple[float, float, float, float]]:
    """Collect painted-path bboxes from a raw content stream (user space).

    Tracks the CTM through ``q``/``Q``/``cm`` and approximates curves by
    their control/end points. Text (``BT``…``ET``) contributes no drawing
    bbox — labels are located separately via ``pdftotext -bbox``.
    """
    ctm: tuple[float, float, float, float, float, float] = (1, 0, 0, 1, 0, 0)
    stack: list[tuple[float, float, float, float, float, float]] = []
    operand: list[float] = []
    current: list[tuple[float, float]] = []
    committed: list[tuple[float, float, float, float]] = []

    def _xf(pt: tuple[float, float]) -> tuple[float, float]:
        a, b, c, d, e, f = ctm
        x, y = pt
        return (a * x + c * y + e, b * x + d * y + f)

    def _commit() -> None:
        if current:
            pts = [_xf(p) for p in current]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            committed.append((min(xs), min(ys), max(xs), max(ys)))
            current.clear()

    # Strip string literals (…) and hex strings <…> so their bytes are not
    # mistaken for operators; dictionary <<…>> blocks are operator-free.
    cleaned = re.sub(rb"\((?:\\.|[^\\()])*\)", b" ", data)
    cleaned = re.sub(rb"<[0-9a-fA-F\s]+>", b" ", cleaned)
    cleaned = re.sub(rb"<<.*?>>", b" ", cleaned, flags=re.S)

    for tok in re.findall(rb"[^\s]+", cleaned):
        if _NUM_RE.match(tok):
            try:
                operand.append(float(tok))
            except ValueError:
                operand = []
            continue
        op = tok.decode("latin-1")
        if op == "q":
            stack.append(ctm)
        elif op == "Q":
            if stack:
                ctm = stack.pop()
        elif op == "cm" and len(operand) >= 6:
            a2, b2, c2, d2, e2, f2 = operand[-6:]
            ctm = mul_matrix((a2, b2, c2, d2, e2, f2), ctm)
        elif op == "m" and len(operand) >= 2:
            _commit()
            current.append((operand[-2], operand[-1]))
        elif op == "l" and len(operand) >= 2:
            current.append((operand[-2], operand[-1]))
        elif op in ("c", "v", "y"):
            n = 6 if op == "c" else 4
            if len(operand) >= n:
                pts = operand[-n:]
                for i in range(0, len(pts), 2):
                    current.append((pts[i], pts[i + 1]))
        elif op == "re" and len(operand) >= 4:
            _commit()
            x, y, w, h = operand[-4:]
            current.extend([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
            _commit()
        elif op in _PAINT_OPS:
            _commit()
        elif op == "n":
            current.clear()
        operand = []
    return committed


def _merge_rects(
    rects: list[tuple[float, float, float, float]], gap: float = 8.0
) -> list[tuple[float, float, float, float]]:
    """Union rects that overlap or come within ``gap`` points of each other."""
    boxes = list(rects)
    merged: list[tuple[float, float, float, float]] = []
    while boxes:
        base = boxes.pop()
        changed = True
        while changed:
            changed = False
            rest = []
            for other in boxes:
                if (
                    other[0] <= base[2] + gap
                    and other[2] >= base[0] - gap
                    and other[1] <= base[3] + gap
                    and other[3] >= base[1] - gap
                ):
                    base = (
                        min(base[0], other[0]),
                        min(base[1], other[1]),
                        max(base[2], other[2]),
                        max(base[3], other[3]),
                    )
                    changed = True
                else:
                    rest.append(other)
            boxes = rest
        merged.append(base)
    return merged


def _merge_row_bands(
    rects: list[tuple[float, float, float, float]],
    max_x_gap: float = 100.0,
    min_y_overlap: float = 0.6,
) -> list[tuple[float, float, float, float]]:
    """Merge chips sharing one visual row (flow-chart boxes + arrows).

    Structural merging (``_merge_rects``) leaves same-row boxes split when
    arrow gaps exceed the gap tolerance; rendering each chip as its own
    full-width figure then destroys the row. Two regions merge only when
    their vertical overlap covers >= ``min_y_overlap`` of the smaller height
    (vertically stacked boxes, e.g. Layer diagrams, are untouched) and the
    horizontal gap is small (arrows, not separate figures).
    """
    boxes = list(rects)
    merged: list[tuple[float, float, float, float]] = []
    while boxes:
        base = boxes.pop()
        changed = True
        while changed:
            changed = False
            rest = []
            for other in boxes:
                y_top = max(base[1], other[1])
                y_bot = min(base[3], other[3])
                overlap = max(0.0, y_bot - y_top)
                min_h = min(base[3] - base[1], other[3] - other[1])
                x_gap = max(other[0] - base[2], base[0] - other[2], 0.0)
                if min_h > 0 and overlap / min_h >= min_y_overlap and x_gap <= max_x_gap:
                    base = (
                        min(base[0], other[0]),
                        min(base[1], other[1]),
                        max(base[2], other[2]),
                        max(base[3], other[3]),
                    )
                    changed = True
                else:
                    rest.append(other)
            boxes = rest
        merged.append(base)
    return merged


def is_text_panel(
    paths_in_region: int,
    words: list[tuple[float, float, float, float, str]],
    *,
    max_paths: int = 2,
    row_tol: float = 3.0,
    min_words_per_row: int = 3,
    min_rows: int = 2,
) -> bool:
    """Whether a candidate region is formatted prose, not a figure.

    Pure function of path count + word boxes (top-down points). A region
    with trivial vector ink (title background panels, hyperlink rules)
    that contains running-prose rows is decorated text: vectorizing it
    duplicates body content as outlined glyphs and invites glossary
    backfill inside bibliography text (the p33 references defect).
    True diagrams either carry real ink (``paths_in_region`` above
    ``max_paths``) or only short labels (single-row chips, symbol
    annotations), never two multi-word prose rows.
    """
    if paths_in_region > max_paths:
        return False
    rows: list[list[tuple[float, float, float, float, str]]] = []
    for word in sorted(words, key=lambda w: (w[1], w[0])):
        placed = False
        for row in rows:
            if abs(word[1] - row[0][1]) <= row_tol:
                row.append(word)
                placed = True
                break
        if not placed:
            rows.append([word])
    full_rows = sum(1 for row in rows if len(row) >= min_words_per_row)
    return full_rows >= min_rows


_WORD_BOX_RE = re.compile(
    r'<word xMin="([0-9\.]+)" yMin="([0-9\.]+)" xMax="([0-9\.]+)" yMax="([0-9\.]+)">([^<]+)</word>'
)


def _page_word_boxes(
    pdf_path: Path | str, page_no: int
) -> list[tuple[float, float, float, float, str]]:
    """All raw pdftotext word boxes on a page (top-down points, no grouping)."""
    if shutil.which("pdftotext") is None:
        return []
    try:
        proc = subprocess.run(
            ["pdftotext", "-bbox", "-f", str(page_no), "-l", str(page_no), str(pdf_path), "-"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            env=subprocess_env(),
        )
        if proc.returncode != 0:
            return []
        words = []
        for m in _WORD_BOX_RE.finditer(proc.stdout):
            wx0, wy0, wx1, wy1 = (float(m.group(i)) for i in range(1, 5))
            text = m.group(5).strip()
            if text:
                words.append((wx0, wy0, wx1, wy1, text))
        return words
    except (OSError, subprocess.SubprocessError):
        return []


def _rects_overlap(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    """Intersection area over the smaller rect (0..1)."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / small if small > 0 else 0.0


def detect_diagram_regions(
    pdf_path: Path | str,
    page_no: int,
    *,
    exclude: list[tuple[float, float, float, float]] | None = None,
    min_paths: int = 3,
    gap: float = 14.0,
) -> list[tuple[float, float, float, float]]:
    """Find vector-diagram regions on a page (bottom-up PDF points).

    Parses the page content stream for painted paths, merges nearby ones,
    and drops full-bleed hairlines plus anything overlapping ``exclude``
    rects (e.g. Docling TABLE bboxes, which have their own pipeline).
    Form XObjects are not descended into (limitation, documented).
    """
    try:
        with pdf_struct.open_pdf(pdf_path) as pdf:
            if not 1 <= page_no <= len(pdf.pages):
                return []
            page = pdf.pages[page_no - 1]
            width, height = pdf_struct.page_size(page)
            stream = pdf_struct.content_stream_bytes(page)
    except Exception as exc:
        logger.debug("Diagram detection failed to read p%d: %s", page_no, exc)
        return []
    if not stream:
        return []

    paths = _parse_content_drawings(stream)
    # Pre-filter furniture paths before spatial clustering; otherwise full-sheet
    # background fills (e.g. 0 0 W H re f on Elsevier/Springer pages) swallow
    # all legitimate diagram paths on the page and cause them to be dropped.
    filtered_paths = [
        p
        for p in paths
        if not ((p[2] - p[0]) * (p[3] - p[1]) > 0.75 * width * height)
        and not ((p[2] - p[0]) > 0.6 * width and (p[3] - p[1]) < 3.0)
    ]
    if len(filtered_paths) < min_paths:
        return []
    # One pdftotext pass per page for the text-panel gate (empty when
    # poppler is missing — the gate then fail-opens toward rendering).
    page_words = _page_word_boxes(pdf_path, page_no)
    regions: list[tuple[float, float, float, float]] = []
    for rect in _merge_row_bands(_merge_rects(filtered_paths, gap=gap)):
        # Clamp to the mediabox; drop spillover artefacts from exotic CTMs.
        cx0, cy0 = max(rect[0], 0.0), max(rect[1], 0.0)
        cx1, cy1 = min(rect[2], width), min(rect[3], height)
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        if (cx1 - cx0) * (cy1 - cy0) < 0.5 * (rect[2] - rect[0]) * (rect[3] - rect[1]):
            continue
        w, h = cx1 - cx0, cy1 - cy0
        if w < _MIN_DIAGRAM_EDGE_PT or h < _MIN_DIAGRAM_EDGE_PT:
            continue
        if w * h > 0.75 * width * height:
            # Page-background fills (e.g. ``0 0 W H re f`` on Elsevier pages)
            # span the whole sheet; they are furniture, never diagrams.
            continue
        if w > 0.8 * width and h < 3.0:
            continue  # full-bleed hairline (header/footer rule)
        rect = (cx0, cy0, cx1, cy1)
        if exclude and any(_rects_overlap(rect, ex) > 0.3 for ex in exclude):
            continue
        # Text-panel gate: trivial ink + running prose rows = decorated
        # text (title panels, hyperlink rules), never vectorize it.
        ink = sum(1 for box in filtered_paths if _rects_overlap(box, rect) > 0.2)
        tx0, ty0, tx1, ty1 = cx0, height - cy1, cx1, height - cy0
        in_region = [
            w for w in page_words if tx0 <= w[0] and w[2] <= tx1 and ty0 <= w[1] and w[3] <= ty1
        ]
        if is_text_panel(ink, in_region):
            logger.debug("Diagram region on p%d rejected as text panel: %r", page_no, rect)
            continue
        regions.append(rect)
    return regions
