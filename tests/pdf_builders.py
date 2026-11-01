from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject, NumberObject

PAGE_WIDTH = 612.0
PAGE_HEIGHT = 792.0
_MARGIN = 54.0
_FONT_SIZE = 12.0
_LEADING = 16.0
_MARKER_FONT_SIZE = 2.0


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _content_stream(
    lines: Sequence[str],
    *,
    height: float,
    margin: float = _MARGIN,
    width: float = PAGE_WIDTH,
    background: tuple[float, float, float] | None = None,
) -> DecodedStreamObject:
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
    rotation: int = 0,
) -> Path:
    writer = PdfWriter()
    for lines in pages:
        page = writer.add_blank_page(width=width, height=height)
        contents = writer._add_object(  # noqa: SLF001 - see _attach_font
            _content_stream(lines, height=height, margin=margin, width=width, background=background)
        )
        page[NameObject("/Contents")] = contents
        if rotation:
            page[NameObject("/Rotate")] = NumberObject(rotation)
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


def write_dotted_toc_pdf(
    path: Path,
    rows: int,
    *,
    dots_per_row: int = 70,
    width: float = PAGE_WIDTH,
    height: float = PAGE_HEIGHT,
    margin: float = _MARGIN,
) -> Path:
    number_x = width - margin - 60.0
    dot_lo, dot_hi = 200.0, number_x - 20.0
    writer = PdfWriter()
    page = writer.add_blank_page(width=width, height=height)
    ops: list[str] = []
    for index in range(rows):
        y = height - margin - index * _LEADING
        runs = [(margin, f"{index + 1}. Section Title")]
        runs += [(dot_lo + j * (dot_hi - dot_lo) / dots_per_row, ".") for j in range(dots_per_row)]
        runs.append((number_x, f"{index * 2 + 3}"))
        for x, text in runs:
            ops.append("BT")
            ops.append(f"/F1 {_FONT_SIZE} Tf")
            ops.append(f"1 0 0 1 {x:.2f} {y:.2f} Tm")
            ops.append(f"({_escape(text)}) Tj")
            ops.append("ET")
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
    "write_dotted_toc_pdf",
    "write_marker_cloud_pdf",
    "write_text_pdf",
    "write_toc_pdf",
    "write_two_column_pdf",
]
