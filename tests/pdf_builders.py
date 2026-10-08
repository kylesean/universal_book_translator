"""Synthetic PDF fixtures, written with pypdf (deliberately an independent writer).

The PDF extraction/render engines under test must not also be the thing that
authored the fixture, so these PDFs are built with ``pypdf``'s generic writer --
a small, independent implementation -- rather than with the engine's own
renderer. ``tests/unit/adapters/test_pdf_adapter_e2e.py`` feeds them to the real
adapter; the real-document gate stays in the ``slow`` corpus harnesses
(``tests/integration/test_corpus_acceptance.py``).

Only single-column, born-digital pages with a standard Type1 font are produced.
That keeps the engine probe deterministic -- ``pdf_engine="auto"`` routes these
to the pure-runtime pdfium adapter and never the optional Docling extra -- so a
consumer can run in the ``fast`` tier with no OCR/VLM/typst toolchain.

This module is test-only: ``pypdf`` is a dev dependency and is imported lazily by
nothing else in the engine.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

#: US Letter, matching the page geometry the corpus papers use.
PAGE_WIDTH = 612.0
PAGE_HEIGHT = 792.0
_MARGIN = 54.0
_FONT_SIZE = 12.0
_LEADING = 16.0
#: A marker glyph small enough that pdfium reports a sub-5pt rect for it.
_MARKER_FONT_SIZE = 2.0


def _escape(text: str) -> str:
    """Escape the three characters a PDF literal string treats specially."""
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _content_stream(
    lines: Sequence[str],
    *,
    height: float,
    margin: float = _MARGIN,
    width: float = PAGE_WIDTH,
    background: tuple[float, float, float] | None = None,
) -> DecodedStreamObject:
    """A text-showing content stream: one Helvetica line per input line.

    ``background`` paints a full-page fill under the text (a tinted page), the
    shape a mask that must not be white has to blend into.
    """
    ops: list[str] = []
    if background is not None:
        red, green, blue = background
        ops.append(f"q {red} {green} {blue} rg 0 0 {width} {height} re f Q")
    ops.extend(
        [
            "BT",
            f"/F1 {_FONT_SIZE} Tf",
            f"1 0 0 1 {margin} {height - margin} Tm",
            f"{_LEADING} TL",
        ]
    )
    ops.extend(f"({_escape(line)}) Tj T*" for line in lines)
    ops.append("ET")
    stream = DecodedStreamObject()
    stream.set_data("\n".join(ops).encode("latin-1"))
    return stream


def _attach_font(writer: PdfWriter, page: object) -> None:
    """Give a page its ``/F1`` Helvetica resource so the text operators resolve."""
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_ref = writer._add_object(font)  # noqa: SLF001 - pypdf has no public page-resource API
    page[NameObject("/Resources")] = DictionaryObject(  # type: ignore[index]
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})}
    )


def write_text_pdf(
    path: Path,
    pages: Sequence[Sequence[str]],
    *,
    width: float = PAGE_WIDTH,
    height: float = PAGE_HEIGHT,
    margin: float = _MARGIN,
    background: tuple[float, float, float] | None = None,
) -> Path:
    """Write a PDF with one text line per string, one page per inner sequence.

    ``width``/``height`` let a caller build a deliberately narrow page (a
    multi-column-shaped fixture); ``margin=0`` authors a fragment at an exact
    box size (a fake fragment typesetter for the LayerCompositor tests).
    ``background`` fills every page with a colour before its text -- a page
    printed on something other than white.
    """
    writer = PdfWriter()
    for lines in pages:
        page = writer.add_blank_page(width=width, height=height)
        contents = writer._add_object(  # noqa: SLF001 - see _attach_font
            _content_stream(lines, height=height, margin=margin, width=width, background=background)
        )
        page[NameObject("/Contents")] = contents
        _attach_font(writer, page)
    with path.open("wb") as handle:
        writer.write(handle)
    return path


def write_two_column_pdf(
    path: Path,
    pages: Sequence[tuple[Sequence[str], Sequence[str]]],
    *,
    width: float = PAGE_WIDTH,
    height: float = PAGE_HEIGHT,
    margin: float = _MARGIN,
) -> Path:
    """Write a PDF with a left and right text column per page.

    Each page is a ``(left_lines, right_lines)`` pair; the left column starts at
    ``margin`` and the right column just past the page's midline, leaving a
    gutter wide enough that the extractor sees two discrete column boxes -- the
    shape a paragraph continuing across a column break takes.
    """
    right_x = width / 2 + margin / 2
    writer = PdfWriter()
    for left, right in pages:
        page = writer.add_blank_page(width=width, height=height)
        ops: list[str] = []
        for x, lines in ((margin, left), (right_x, right)):
            ops.append("BT")
            ops.append(f"/F1 {_FONT_SIZE} Tf")
            ops.append(f"1 0 0 1 {x} {height - margin} Tm")
            ops.append(f"{_LEADING} TL")
            ops.extend(f"({_escape(line)}) Tj T*" for line in lines)
            ops.append("ET")
        contents = DecodedStreamObject()
        contents.set_data("\n".join(ops).encode("latin-1"))
        page[NameObject("/Contents")] = writer._add_object(contents)  # noqa: SLF001
        _attach_font(writer, page)
    with path.open("wb") as handle:
        writer.write(handle)
    return path


def write_toc_pdf(
    path: Path,
    entries: Sequence[tuple[str, str]],
    *,
    width: float = PAGE_WIDTH,
    height: float = PAGE_HEIGHT,
    margin: float = _MARGIN,
) -> Path:
    """Write a one-page table of contents: each ``(title, page)`` on one row.

    The page number sits near the right margin on the title's own baseline,
    with the dot-leader gutter between them -- the shape the native reader must
    pair back into a single TOC row (``textgeom`` reads the two as separate
    lines). At least three rows are needed for the reader to call it a TOC.
    """
    page_number_x = width - margin - 90.0
    writer = PdfWriter()
    page = writer.add_blank_page(width=width, height=height)
    ops: list[str] = []
    for index, (title, page_no) in enumerate(entries):
        y = height - margin - index * _LEADING
        for x, text in ((margin, title), (page_number_x, page_no)):
            ops.append("BT")
            ops.append(f"/F1 {_FONT_SIZE} Tf")
            ops.append(f"1 0 0 1 {x} {y} Tm")
            ops.append(f"({_escape(text)}) Tj")
            ops.append("ET")
    contents = DecodedStreamObject()
    contents.set_data("\n".join(ops).encode("latin-1"))
    page[NameObject("/Contents")] = writer._add_object(contents)  # noqa: SLF001
    _attach_font(writer, page)
    with path.open("wb") as handle:
        writer.write(handle)
    return path


def write_marker_cloud_pdf(
    path: Path,
    markers: int = 700,
    *,
    body: Sequence[str] = (),
    width: float = PAGE_WIDTH,
    height: float = PAGE_HEIGHT,
    margin: float = _MARGIN,
) -> Path:
    """A page whose text layer is dominated by sub-5pt marker glyphs.

    Stands in for the text layer of a matplotlib/tikz scatter figure: hundreds
    of tiny glyphs at scattered positions, each its own pdfium rect, plus (when
    ``body`` is given) normal lines the caller can assert survive the cleanup.
    Positions come from a fixed seed so the fixture -- and any test that counts
    its rects -- is deterministic.
    """
    rng = random.Random(0x5CA7)
    ops: list[str] = []
    for _ in range(markers):
        x = margin + rng.random() * (width - 2 * margin)
        y = margin + rng.random() * (height - 2 * margin)
        ops.append("BT")
        ops.append(f"/F1 {_MARKER_FONT_SIZE} Tf")
        ops.append(f"1 0 0 1 {x:.2f} {y:.2f} Tm")
        ops.append("(a) Tj")
        ops.append("ET")
    if body:
        ops.append("BT")
        ops.append(f"/F1 {_FONT_SIZE} Tf")
        ops.append(f"1 0 0 1 {margin} {height - margin} Tm")
        ops.append(f"{_LEADING} TL")
        ops.extend(f"({_escape(line)}) Tj T*" for line in body)
        ops.append("ET")
    writer = PdfWriter()
    page = writer.add_blank_page(width=width, height=height)
    contents = DecodedStreamObject()
    contents.set_data("\n".join(ops).encode("latin-1"))
    page[NameObject("/Contents")] = writer._add_object(contents)  # noqa: SLF001
    _attach_font(writer, page)
    with path.open("wb") as handle:
        writer.write(handle)
    return path


__all__ = [
    "PAGE_HEIGHT",
    "PAGE_WIDTH",
    "write_marker_cloud_pdf",
    "write_text_pdf",
    "write_toc_pdf",
    "write_two_column_pdf",
]
