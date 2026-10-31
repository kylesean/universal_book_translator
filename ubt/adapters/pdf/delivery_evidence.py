"""Artifact evidence adapters for the delivery contract.

The contract module is pure: it takes ``ContractEvidence`` and returns a
verdict. This module is the one place that reads a real PDF into that shape, so
the contract never learns about pdfium, and the reader is testable on its own.

Two deliberate choices:

* **A box without a measurable font size stays unmeasured.** pdfium reports
  per-character boxes; a formula's superscript or a table rule has no body
  size. ``font_size_pt=None`` tells the contract "this box cannot establish a
  style or overlap claim", which is why formula pages are not red by
  construction.
* **The body-style sample comes from the renderer, not from the boxes.** A
  document-wide distribution over every box would call a heading a style
  drift. The renderer knows which size it set the body at; the caller passes
  that through ``StyleSample``. When the caller passes nothing, the size is
  read from the content stream (see :func:`_exact_body_sizes`) and only the
  ink-height proxy remains as the last resort — the proxy tracks the *script*
  as much as the size (a CJK glyph fills its em box, a Latin one does not), so
  on a mixed-script page it reports drift that is not there.
"""

from __future__ import annotations

import contextlib
import logging
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pikepdf

from ubt.core.engine.delivery_contract import (
    ContractBlock,
    ContractEvidence,
    ContractTextBox,
    OverflowRecord,
    StyleSample,
)

logger = logging.getLogger(__name__)

#: A text box must hold at least this many characters to be worth a geometry
#: claim. A single punctuation mark's box is ink, not layout.
_MIN_BOX_CHARS = 2


def _normalise_text(text: str) -> str:
    return "".join((text or "").split())


def _exact_body_sizes(
    pdf_path: Path,
    exhibit_hulls: dict[int, list[tuple[float, float, float, float]]] | None = None,
    body_pages: set[int] | None = None,
) -> tuple[StyleSample, ...]:
    """The body size of each page, in points, from the content stream.

    ``Tf`` sets a nominal size and the text matrix scales it, so the size the
    reader sees is ``Tf × |Tm scale|``; the body size of a page is the
    *character-weighted mode* of that value over its show operators. A mode
    rather than a median on purpose: headings, captions and formula
    sub/superscripts are legitimate outliers, while the body is the size most
    glyphs are actually set in.

    Read here rather than inferred from ink boxes because ink height measures
    the script as much as the size. On ``chapter-3.pdf`` the delivered body
    sizes really are 7.5–9.79pt (the legacy renderer solves a size per page),
    but a CJK glyph fills its em box while a Latin one does not, so the ink
    proxy on a mixed page would report drift for a page set entirely at one
    size — a false positive that hides the true ones.

    ``exhibit_hulls`` and ``body_pages`` come from the pdfium rect walk: a show
    operator whose text origin falls inside a table's covering hull is left out
    of the vote (a grid is its own style group, PRD FR-L4, and on the ATC'25
    replay the lmbench page's glyph-weighted mode was the *grid's* size), and a
    page the walk found no body-block text on is not sampled at all -- its
    leftover caption characters must not pose as a body.

    Unparseable pages are skipped rather than failing the read: a broken stream
    must not turn into ``style_unmeasured`` for the whole document.
    """
    hulls_by_page = exhibit_hulls or {}
    try:
        with pikepdf.open(pdf_path) as pdf:
            samples: list[StyleSample] = []
            for index in range(len(pdf.pages)):
                page_no = index + 1
                if body_pages is not None and page_no not in body_pages:
                    continue
                try:
                    size = _page_body_size(pdf.pages[index], hulls_by_page.get(page_no, ()))
                except Exception as exc:  # pragma: no cover - malformed stream
                    logger.debug("Delivery contract: page %d unreadable: %s", index + 1, exc)
                    continue
                if size is not None:
                    samples.append(StyleSample(page=page_no, size_pt=size))
            return tuple(samples)
    except Exception as exc:
        logger.warning("Delivery contract: cannot measure body sizes in %s: %s", pdf_path, exc)
        return ()


def _in_exhibit_hull(
    x: float, y: float, hulls: Sequence[tuple[float, float, float, float]]
) -> bool:
    return any(
        hx0 - 2.0 <= x <= hx1 + 2.0 and hy0 - 2.0 <= y <= hy1 + 2.0 for hx0, hy0, hx1, hy1 in hulls
    )


def _page_body_size(
    page: pikepdf.Page,
    hulls: Sequence[tuple[float, float, float, float]] = (),
) -> float | None:
    """Character-weighted mode of the device font size on one page.

    Text origins are tracked through ``Tm``/``Td``/``TD``/``T*`` well enough to
    attribute a show operator to a region of the page (the exhibit hulls) --
    the mode then weighs only the operators outside them. Typst emits one
    ``Tm`` per text run, so the origin is exact for the runs that matter; a
    ``Td``-relative stream degrades to approximate attribution, which still
    separates a full-width grid from the prose above and below it.
    """
    weights: Counter[float] = Counter()
    tm_scale = 1.0
    tf_size = 0.0
    tx = ty = 0.0
    leading = 0.0
    for instruction in pikepdf.parse_content_stream(page):
        try:
            operands, operator = instruction[0], str(instruction[1])
        except Exception:  # pragma: no cover - lazy parse hazard
            break
        if operator == "Tm" and len(operands) >= 6:
            try:
                tm_scale = abs(float(operands[0])) or 1.0
                tx, ty = float(operands[4]), float(operands[5])
            except (TypeError, ValueError):
                tm_scale = 1.0
        elif operator == "Tf" and len(operands) >= 2:
            try:
                tf_size = float(operands[1])
            except (TypeError, ValueError):
                tf_size = 0.0
        elif operator == "BT":
            tx = ty = 0.0
        elif operator == "Td" and len(operands) >= 2:
            try:
                tx += float(operands[0])
                ty += float(operands[1])
            except (TypeError, ValueError):
                pass
        elif operator == "TD" and len(operands) >= 2:
            try:
                leading = -float(operands[1])
                tx += float(operands[0])
                ty += float(operands[1])
            except (TypeError, ValueError):
                pass
        elif operator == "TL" and operands:
            with contextlib.suppress(TypeError, ValueError):
                leading = float(operands[0])
        elif operator == "T*":
            ty -= leading
        elif operator in ("Tj", "TJ", "'", '"'):
            if tf_size <= 0.0:
                continue
            if hulls and _in_exhibit_hull(tx, ty, hulls):
                continue
            size = round(tf_size * tm_scale, 2)
            if not 1.0 < size < 72.0:
                continue
            weights[size] += max(1, len(_show_glyph_bytes(operands, operator)))
    if not weights:
        return None
    return weights.most_common(1)[0][0]


def _show_glyph_bytes(operands: Any, operator: str) -> bytes:
    """Glyph bytes one show operator paints (used only for its length)."""
    try:
        if operator == "TJ":
            return b"".join(bytes(item) for item in operands[0] if not _is_number(item))
        target = operands[-1] if operator in ("'", '"') else operands[0]
        return bytes(target)
    except Exception:  # pragma: no cover - unusual operand shape
        return b""


def _is_number(item: Any) -> bool:
    try:
        float(item)
    except (TypeError, ValueError):
        return False
    return True


def _page_rect_classes(
    textpage: object, exhibit_norm: tuple[str, ...], body_norm: tuple[str, ...]
) -> tuple[list[tuple[float, float, float, float, str]], bool]:
    """One rect walk classifying the page's text into exhibit / body / other.

    Returns ``(rects, has_body)`` where each classified rect is
    ``(x0, y0, x1, y1, class)`` with class ``"exhibit"`` or ``"body"``, and
    ``has_body`` says whether the page carries any body-block text at all. A
    rect belongs to an exhibit when its normalized text is contained in (or
    contains) one of the table blocks' normalized lines, and to the body on
    the same terms against the prose blocks' lines; unmatched rects carry no
    class and the sampler ignores them.
    """
    rects: list[tuple[float, float, float, float, str]] = []
    has_body = False
    rect_count = int(textpage.count_rects(0, -1))  # type: ignore[attr-defined]
    for index in range(rect_count):
        x0, y0, x1, y1 = textpage.get_rect(index)  # type: ignore[attr-defined]
        text = (textpage.get_text_bounded(x0, y0, x1, y1) or "").strip()  # type: ignore[attr-defined]
        norm = _normalise_text(text)
        if len(norm) < 6:
            continue
        if any(line in norm or norm in line for line in exhibit_norm):
            rects.append((float(x0), float(y0), float(x1), float(y1), "exhibit"))
        elif body_norm and any(line in norm or norm in line for line in body_norm):
            rects.append((float(x0), float(y0), float(x1), float(y1), "body"))
            has_body = True
    return rects, has_body


def _median_height(boxes: Sequence[tuple[float, float, float, float]]) -> float:
    heights = sorted(box[3] - box[1] for box in boxes)
    return heights[len(heights) // 2] if heights else 0.0


def _hull(boxes: Sequence[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _body_size_from_char_fonts(
    textpage: object,
    classified_rects: Sequence[tuple[float, float, float, float, str]] = (),
) -> float | None:
    """The mode of the page's per-character font sizes, exhibits excluded.

    pdfium reports each character's actual font size -- ``Tf`` and the text
    matrix resolved, script-independent, in points. Ink height was the old
    proxy here and it cannot carry the check: a CJK glyph fills its em box
    while a Latin one does not, so a Chinese body and a references page set at
    the *same* size read 8.8pt vs 7.5pt at ink level (measured on the ATC'25
    replay) -- a phantom drift.

    Exclusion is by *size*, decided by where a size's characters live. pdfium
    slices a grid into per-cell rects and most cells are too short to
    text-match ("0.050"), so rect classes alone leave cells unclassified -- but
    a grid is set at ONE size, and its long header/first-column cells do match
    the table corpus. Any font size whose characters sit mostly in
    exhibit-class rects is therefore the grid's size, and every character
    carrying it -- the numeric cells, the continuation headers -- leaves the
    vote together; the body's size sits mostly in body-class rects. Measured
    on the replay: the lmbench page's grid characters (5.3pt) outnumbered the
    prose three to one, which is how the document read as a 4.18pt body drift
    and blocked the export. ``None`` when no character lands in a classified
    rect.
    """
    from bisect import bisect_left, bisect_right

    from pypdfium2 import raw as pdfium_raw

    # Chars sorted by box center y so each rect's population is a slice.
    chars: list[tuple[float, float, float]] = []
    count = int(textpage.count_chars())  # type: ignore[attr-defined]
    for index in range(count):
        char = textpage.get_text_range(index, 1)  # type: ignore[attr-defined]
        if not char or char.isspace() or char in "\r\n":
            continue
        try:
            size = float(pdfium_raw.FPDFText_GetFontSize(textpage, index))
        except Exception:  # pragma: no cover - raw binding mismatch
            return None
        if not 1.0 < size < 72.0:
            continue
        x0, y0, x1, y1 = textpage.get_charbox(index)  # type: ignore[attr-defined]
        chars.append(((float(y0) + float(y1)) / 2.0, (float(x0) + float(x1)) / 2.0, round(size, 2)))
    chars.sort()
    ys = [entry[0] for entry in chars]

    sizes_by_class: dict[str, Counter[float]] = {"exhibit": Counter(), "body": Counter()}
    for rx0, ry0, rx1, ry1, cls in classified_rects:
        lo = bisect_left(ys, ry0 - 1.0)
        hi = bisect_right(ys, ry1 + 1.0)
        counter = sizes_by_class[cls]
        for _cy, cx, size in chars[lo:hi]:
            if rx0 - 1.0 <= cx <= rx1 + 1.0:
                counter[size] += 1

    body_weights: Counter[float] = Counter()
    for size, total in sizes_by_class["body"].items():
        if total > sizes_by_class["exhibit"].get(size, 0):
            body_weights[size] = total
    if not body_weights:
        return None
    return body_weights.most_common(1)[0][0]


def _text_boxes_for_page(page_no: int, textpage: object) -> list[ContractTextBox]:
    boxes: list[ContractTextBox] = []
    rect_count = int(textpage.count_rects(0, -1))  # type: ignore[attr-defined]
    for index in range(rect_count):
        x0, y0, x1, y1 = textpage.get_rect(index)  # type: ignore[attr-defined]
        text = (textpage.get_text_bounded(x0, y0, x1, y1) or "").strip()  # type: ignore[attr-defined]
        if len(_normalise_text(text)) < _MIN_BOX_CHARS:
            continue
        line_count = max(1, sum(1 for ln in text.splitlines() if ln.strip()))
        line_height_pt = (float(y1) - float(y0)) / line_count
        font_size = round(line_height_pt, 2) if line_height_pt >= 4.0 else None
        boxes.append(
            ContractTextBox(
                page=page_no,
                x0=float(x0),
                y0=float(y0),
                x1=float(x1),
                y1=float(y1),
                font_size_pt=font_size,
                text=text,
            )
        )
    return boxes


def read_artifact_evidence(
    pdf_path: Path,
    *,
    contract: str,
    blocks: Sequence[ContractBlock],
    expected_page_count: int | None = None,
    missing_assets: Iterable[str] = (),
    font_issues: Iterable[str] = (),
    overflow_records: Iterable[OverflowRecord] = (),
    style_samples: Iterable[StyleSample] = (),
    read_geometry: bool = True,
) -> ContractEvidence:
    """Read a delivered PDF into ``ContractEvidence``.

    ``read_geometry=False`` still reads page text (the text-realization check),
    but skips the per-character geometry walk. That split exists because the
    text check is cheap and the geometry walk is O(chars); a caller that only
    wants "did every block make it" should not pay for boxes it will not use.

    A PDF that cannot be opened yields ``page_count=0`` and no text; the
    contract turns that into ``unreadable_artifact`` rather than a silent pass.
    """
    from ubt.adapters.pdf.pdfium_gate import open_document, pdfium_serialized

    # Exhibit text the body-size sampler must not count: a table legitimately
    # renders smaller than the body (its own style group, PRD FR-L4), and on a
    # page a dense grid dominates, the page's size statistic is the *table's*.
    # Measured on the ATC'25 replay: the lmbench page read as a 5.32pt body and
    # the appendix pages as 5.25pt -- a 4.18pt "drift" that blocked the export
    # while every prose page sat at one size.
    exhibit_norm = tuple(
        _normalise_text(line)
        for block in blocks
        if block.kind == "table"
        for line in (block.text or "").splitlines()
        if len(_normalise_text(line)) >= 6
    )
    # Body text a page must carry to earn a body-size sample: a page of pure
    # exhibit (grid + caption + footnotes) has no body to measure, and its
    # leftover caption characters must not pose as one.
    body_norm = tuple(
        _normalise_text(line)
        for block in blocks
        if block.kind in ("narrative", "heading", "list_item")
        for line in (block.text or "").splitlines()
        if len(_normalise_text(line)) >= 10
    )

    @pdfium_serialized
    def _read() -> tuple[
        int,
        tuple[str, ...],
        dict[int, tuple[float, float, float, float]],
        tuple[ContractTextBox, ...],
        tuple[StyleSample, ...],
        dict[int, list[tuple[float, float, float, float]]],
        set[int],
    ]:
        with open_document(pdf_path) as pdf:
            page_count = len(pdf)
            texts: list[str] = []
            bounds: dict[int, tuple[float, float, float, float]] = {}
            boxes: list[ContractTextBox] = []
            samples: list[StyleSample] = []
            page_exhibits: dict[int, list[tuple[float, float, float, float]]] = {}
            body_pages: set[int] = set()
            for index in range(page_count):
                page_no = index + 1
                page = pdf[index]
                try:
                    width = float(page.get_width())
                    height = float(page.get_height())
                    bounds[page_no] = (0.0, 0.0, width, height)
                    textpage = page.get_textpage()
                    try:
                        texts.append(textpage.get_text_range() or "")
                        if read_geometry:
                            boxes.extend(_text_boxes_for_page(page_no, textpage))
                            rects, has_body = _page_rect_classes(textpage, exhibit_norm, body_norm)
                            if has_body:
                                body_pages.add(page_no)
                                size = _body_size_from_char_fonts(textpage, rects)
                                if size is not None:
                                    samples.append(StyleSample(page=page_no, size_pt=size))
                            exhibit_boxes = [
                                (rx0, ry0, rx1, ry1)
                                for rx0, ry0, rx1, ry1, cls in rects
                                if cls == "exhibit"
                            ]
                            if exhibit_boxes:
                                gap_tol = 3.0 * _median_height(exhibit_boxes)
                                ordered = sorted(exhibit_boxes, key=lambda box: box[1])
                                hulls: list[tuple[float, float, float, float]] = []
                                cluster: list[tuple[float, float, float, float]] = [ordered[0]]
                                for box in ordered[1:]:
                                    if box[1] - cluster[-1][3] <= gap_tol:
                                        cluster.append(box)
                                    else:
                                        hulls.append(_hull(cluster))
                                        cluster = [box]
                                hulls.append(_hull(cluster))
                                page_exhibits[page_no] = hulls
                    finally:
                        textpage.close()
                finally:
                    page.close()
            return (
                page_count,
                tuple(texts),
                bounds,
                tuple(boxes),
                tuple(samples),
                page_exhibits,
                body_pages,
            )

    try:
        (
            page_count,
            page_texts,
            page_bounds,
            text_boxes,
            measured,
            page_exhibits,
            body_pages,
        ) = _read()
    except Exception as exc:
        logger.warning("Delivery contract: cannot read %s: %s", pdf_path, exc)
        return ContractEvidence(
            contract=contract,  # type: ignore[arg-type]
            page_count=0,
            page_texts=(),
            page_bounds={},
            blocks=tuple(blocks),
            expected_page_count=expected_page_count,
            missing_assets=tuple(missing_assets),
            font_issues=tuple(font_issues),
            overflow_records=tuple(overflow_records),
            style_samples=tuple(style_samples),
        )

    return ContractEvidence(
        contract=contract,  # type: ignore[arg-type]
        page_count=page_count,
        page_texts=page_texts,
        page_bounds=page_bounds,
        blocks=tuple(blocks),
        text_boxes=text_boxes,
        expected_page_count=expected_page_count,
        missing_assets=tuple(missing_assets),
        font_issues=tuple(font_issues),
        overflow_records=tuple(overflow_records),
        # Renderer-reported sizes win (they are the fact). Then the content
        # stream -- exact font sizes, script-independent -- with the pdfium
        # rect walk's exhibit hulls and body-page verdicts steering its vote
        # (a grid is its own style group, PRD FR-L4); the ink-height walk is
        # the same exhibit-aware mode and serves artifacts whose text hides in
        # Form XObjects the content-stream parse cannot see -- Typst output,
        # whose text the ATC'25 replay's export was judged from. Unsteered,
        # both estimators read a table-dominated page as the *grid's* size,
        # which read that replay as a 4.18pt body drift and blocked the export.
        style_samples=tuple(style_samples)
        or _exact_body_sizes(pdf_path, page_exhibits or None, body_pages or None)
        or measured,
    )


__all__ = ["read_artifact_evidence"]
