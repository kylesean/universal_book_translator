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


def _exact_body_sizes(pdf_path: Path) -> tuple[StyleSample, ...]:
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

    Unparseable pages are skipped rather than failing the read: a broken stream
    must not turn into ``style_unmeasured`` for the whole document.
    """
    try:
        with pikepdf.open(pdf_path) as pdf:
            samples: list[StyleSample] = []
            for index in range(len(pdf.pages)):
                try:
                    size = _page_body_size(pdf.pages[index])
                except Exception as exc:  # pragma: no cover - malformed stream
                    logger.debug("Delivery contract: page %d unreadable: %s", index + 1, exc)
                    continue
                if size is not None:
                    samples.append(StyleSample(page=index + 1, size_pt=size))
            return tuple(samples)
    except Exception as exc:
        logger.warning("Delivery contract: cannot measure body sizes in %s: %s", pdf_path, exc)
        return ()


def _page_body_size(page: pikepdf.Page) -> float | None:
    """Character-weighted mode of the device font size on one page."""
    weights: Counter[float] = Counter()
    tm_scale = 1.0
    tf_size = 0.0
    for instruction in pikepdf.parse_content_stream(page):
        try:
            operands, operator = instruction[0], str(instruction[1])
        except Exception:  # pragma: no cover - lazy parse hazard
            break
        if operator == "Tm" and len(operands) >= 6:
            try:
                tm_scale = abs(float(operands[0])) or 1.0
            except (TypeError, ValueError):
                tm_scale = 1.0
        elif operator == "Tf" and len(operands) >= 2:
            try:
                tf_size = float(operands[1])
            except (TypeError, ValueError):
                tf_size = 0.0
        elif operator in ("Tj", "TJ", "'", '"'):
            if tf_size <= 0.0:
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


def _body_size_from_charboxes(
    textpage: object,
) -> float | None:
    """Median char-box height, the body-size proxy the visual gate already uses.

    A single glyph's box is ink height, not font size, but the *median* over a
    page's content characters tracks the body size closely enough to seed a
    style sample (the gate's own ``type_size_findings`` relies on the same
    property). ``None`` when the page carries no measurable characters.
    """
    heights: list[float] = []
    count = int(textpage.count_chars())  # type: ignore[attr-defined]
    for index in range(count):
        char = textpage.get_text_range(index, 1)  # type: ignore[attr-defined]
        if not char or char.isspace() or char in "\r\n":
            continue
        x0, y0, x1, y1 = textpage.get_charbox(index)  # type: ignore[attr-defined]
        height = float(y1) - float(y0)
        if height > 0 and (float(x1) - float(x0)) > 0.05:
            heights.append(height)
    if not heights:
        return None
    heights.sort()
    return heights[len(heights) // 2]


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
        boxes.append(
            ContractTextBox(
                page=page_no,
                x0=float(x0),
                y0=float(y0),
                x1=float(x1),
                y1=float(y1),
                font_size_pt=round(line_height_pt, 2),
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

    @pdfium_serialized
    def _read() -> tuple[
        int,
        tuple[str, ...],
        dict[int, tuple[float, float, float, float]],
        tuple[ContractTextBox, ...],
        tuple[StyleSample, ...],
    ]:
        with open_document(pdf_path) as pdf:
            page_count = len(pdf)
            texts: list[str] = []
            bounds: dict[int, tuple[float, float, float, float]] = {}
            boxes: list[ContractTextBox] = []
            samples: list[StyleSample] = []
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
                            size = _body_size_from_charboxes(textpage)
                            if size is not None:
                                samples.append(StyleSample(page=page_no, size_pt=size))
                    finally:
                        textpage.close()
                finally:
                    page.close()
            return page_count, tuple(texts), bounds, tuple(boxes), tuple(samples)

    try:
        page_count, page_texts, page_bounds, text_boxes, measured = _read()
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
        # Renderer-reported sizes win (they are the fact), then the exact
        # content-stream sizes, and only then the ink-height proxy — which
        # measures the script as much as the size and cannot be trusted on a
        # mixed CJK/Latin page.
        style_samples=tuple(style_samples) or _exact_body_sizes(pdf_path) or measured,
    )


__all__ = ["read_artifact_evidence"]
