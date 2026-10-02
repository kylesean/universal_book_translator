"""The native HTML/EPUB/DOCX readers.

Each reader must give every element a real ``Span.chars`` that slices exactly
its own text out of ``CanonicalSource.text`` -- the reader's core promise, a
*verifiable* span -- and the ``Document`` must round-trip through the bridge
losslessly. The fixtures are built here (a minimal DOCX via python-docx, an
EPUB via zipfile, an HTML string), so the tests need no external sample files.
"""

from __future__ import annotations

import zipfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import pytest

from ubt.analyze.assemble import element_text
from ubt.analyze.bridge import blocks_from_document, document_from_blocks
from ubt.analyze.reader_docx import read_docx
from ubt.analyze.reader_epub import read_epub
from ubt.analyze.reader_html import read_html
from ubt.model.ast import Document

#: A 1x1 transparent PNG, for the DOCX picture fixture.
_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c6360606060000000050001a5f645400000000049454e44ae426082"
)

_HTML = """<!doctype html><html><head><title>t</title></head><body>
<h1>Chapter One</h1>
<p>Hello <b>world</b>, this is prose.</p>
<ul><li>First item</li><li>Second item</li></ul>
<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>
<figure><img src="pic.png"><figcaption>Figure 1</figcaption></figure>
<pre>print("hi")</pre>
</body></html>"""

_EPUB_CONTAINER = """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles><rootfile full-path="OEBPS/content.opf"
    media-type="application/oebps-package+xml"/></rootfiles>
</container>"""

_EPUB_OPF = """<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>T</dc:title></metadata>
  <manifest>
    <item id="c1" href="ch1.xhtml" media-type="application/xhtml+xml"/>
    <item id="c2" href="ch2.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine><itemref idref="c1"/><itemref idref="c2"/></spine>
</package>"""

_EPUB_CH1 = """<html><body><h1>One</h1><p>First chapter.</p>
<img src="images/fig.png" alt="fig"/></body></html>"""
_EPUB_CH2 = """<html><body><h1>Two</h1><p>Second chapter.</p>
<ul><li>a</li><li>b</li></ul></body></html>"""


def _write_epub(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr("META-INF/container.xml", _EPUB_CONTAINER)
        zf.writestr("OEBPS/content.opf", _EPUB_OPF)
        zf.writestr("OEBPS/ch1.xhtml", _EPUB_CH1)
        zf.writestr("OEBPS/ch2.xhtml", _EPUB_CH2)
        zf.writestr("OEBPS/images/fig.png", _PNG)


def _write_docx(path: Path) -> None:
    from docx import Document as DocxDocument

    image = path.with_suffix(".png")
    image.write_bytes(_PNG)
    doc = DocxDocument()
    doc.add_heading("Chapter One", level=1)
    doc.add_paragraph("Hello world, this is prose.")
    doc.add_paragraph("First item", style="List Bullet")
    doc.add_paragraph("Second item", style="List Bullet")
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "A"
    table.cell(0, 1).text = "B"
    table.cell(1, 0).text = "1"
    table.cell(1, 1).text = "2"
    doc.add_picture(str(image))
    doc.save(str(path))


def _build_html(tmp_path: Path) -> Path:
    path = tmp_path / "sample.html"
    path.write_text(_HTML, encoding="utf-8")
    return path


def _build_epub(tmp_path: Path) -> Path:
    path = tmp_path / "sample.epub"
    _write_epub(path)
    return path


def _build_docx(tmp_path: Path) -> Path:
    path = tmp_path / "sample.docx"
    _write_docx(path)
    return path


Case = tuple[str, Callable[[Path], Document], dict[str, int]]

_CASES: tuple[Case, ...] = (
    (
        "html",
        lambda path: read_html(path),
        {
            "heading": 1,
            "paragraph": 1,
            "list_item": 2,
            "table": 1,
            "figure": 1,
            "caption": 1,
            "code_block": 1,
        },
    ),
    (
        "epub",
        lambda path: read_epub(path),
        {"heading": 2, "paragraph": 2, "figure": 1, "list_item": 2},
    ),
    (
        "docx",
        lambda path: read_docx(path),
        {"heading": 1, "paragraph": 1, "list_item": 2, "table": 1, "figure": 1},
    ),
)
_BUILDERS: dict[str, Callable[[Path], Path]] = {
    "html": _build_html,
    "epub": _build_epub,
    "docx": _build_docx,
}


def _span_problems(document: Document) -> list[str]:
    problems: list[str] = []
    for element in document.elements:
        chars = element.span.chars
        if chars is None:
            problems.append(f"{element.id}: missing Span.chars")
            continue
        expected = element_text(element)
        got = document.source.text[chars[0] : chars[1]]
        if got != expected:
            problems.append(f"{element.id}: span slice {got[:30]!r} != {expected[:30]!r}")
    return problems


@pytest.mark.parametrize(("name", "read", "expect"), _CASES, ids=[c[0] for c in _CASES])
def test_every_element_has_an_exact_verifiable_span(
    tmp_path: Path, name: str, read: Callable[[Path], Document], expect: dict[str, int]
) -> None:
    document = read(_BUILDERS[name](tmp_path))
    assert document.elements, f"{name}: the reader extracted nothing"
    assert _span_problems(document) == []


@pytest.mark.parametrize(("name", "read", "expect"), _CASES, ids=[c[0] for c in _CASES])
def test_the_document_round_trips_through_the_bridge(
    tmp_path: Path, name: str, read: Callable[[Path], Document], expect: dict[str, int]
) -> None:
    document = read(_BUILDERS[name](tmp_path))
    round_tripped = document_from_blocks(
        blocks_from_document(document), doc_id=document.source.doc_id
    )
    assert len(round_tripped.elements) == len(document.elements)
    assert Counter(e.kind.value for e in round_tripped.elements) == Counter(
        e.kind.value for e in document.elements
    )


@pytest.mark.parametrize(("name", "read", "expect"), _CASES, ids=[c[0] for c in _CASES])
def test_the_expected_structure_is_present(
    tmp_path: Path, name: str, read: Callable[[Path], Document], expect: dict[str, int]
) -> None:
    document = read(_BUILDERS[name](tmp_path))
    counts = Counter(element.kind.value for element in document.elements)
    for kind, minimum in expect.items():
        assert counts.get(kind, 0) >= minimum, f"{name}: expected >= {minimum} {kind}"
