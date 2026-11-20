from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from ubt.adapters.docx.adapter import DOCXAdapter
from ubt.adapters.epub.adapter import EPUBAdapter
from ubt.adapters.html.adapter import HTMLAdapter
from ubt.core.ir.models import BlockType, IRBlock

pytestmark = pytest.mark.fast

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


@pytest.fixture(scope="module")
def html_book(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("html") / "sample.html"
    path.write_text(_HTML, encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def epub_book(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("epub") / "sample.epub"
    _write_epub(path)
    return path


@pytest.fixture(scope="module")
def docx_book(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("docx") / "sample.docx"
    _write_docx(path)
    return path


async def _collect(adapter: object, book: Path) -> list[IRBlock]:
    blocks: list[IRBlock] = []
    async for chapter in adapter.parse_stream(book):  # type: ignore[attr-defined]
        blocks.extend(chapter.blocks)
    return blocks


async def test_html_inspection_adapter_reads_the_typed_blocks(html_book: Path) -> None:
    adapter = HTMLAdapter()
    manifest = await adapter.extract_manifest(html_book)
    assert manifest.chapters[0].chapter_id == "html_main"
    blocks = await _collect(adapter, html_book)
    assert len(blocks) == 8, "the html fixture has exactly 8 blocks"


async def test_epub_inspection_adapter_reads_spine_blocks_and_figures(epub_book: Path) -> None:
    adapter = EPUBAdapter()
    manifest = await adapter.extract_manifest(epub_book)
    assert len(manifest.chapters) == 2
    blocks = await _collect(adapter, epub_book)
    assert len(blocks) == 7, "the epub fixture has exactly 7 blocks"
    figure_ids = [b.source_text for b in blocks if b.block_type is BlockType.IMAGE]
    assert [Path(a).name for a in figure_ids] == ["fig.png"], figure_ids


async def test_docx_manifest_reads_the_title_without_loading_the_body(
    docx_book: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import docx

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("extract_manifest loaded the whole DOCX package")

    monkeypatch.setattr(docx, "Document", _boom)
    manifest = await DOCXAdapter().extract_manifest(docx_book)
    assert manifest.title
    assert manifest.chapters[0].source_file == docx_book.name


async def test_docx_inspection_adapter_reads_body_blocks(docx_book: Path) -> None:
    blocks = await _collect(DOCXAdapter(), docx_book)
    assert len(blocks) == 6, "the docx fixture has exactly 6 blocks"
    assert any(block.block_type is BlockType.TABLE for block in blocks)
