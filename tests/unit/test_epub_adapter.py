"""Unit tests for EPUBAdapter with native DOM bilingual injection."""

import io
import zipfile
from pathlib import Path
from typing import Any

import pytest
from bs4 import BeautifulSoup

from tests.epub_builders import XHTML_NS, item, opf, page, write_epub
from ubt.adapters.epub.adapter import BLOCK_TAGS, EPUBAdapter, is_leaf_block
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.exceptions import UBTError
from ubt.core.ir.models import BlockStatus, BlockType, BookManifest, ChapterMeta, FlowID, IRBlock
from ubt.core.job_options import overrides_from_request


def create_mock_epub(target_path: Path) -> Path:
    """Two-chapter Dickens book: a heading, prose, and a sidebar aside."""
    return write_epub(
        target_path,
        opf_xml=opf(
            pub_id="urn:uuid:12345-67890",
            title="A Tale of Two Cities",
            creator="Charles Dickens",
            items=[item("ch01", "ch01.xhtml"), item("ch02", "ch02.xhtml")],
        ),
        parts={
            "OEBPS/ch01.xhtml": page(
                "    <h1>Chapter 1: The Period</h1>\n"
                "    <p>It was the best of times, it was the worst of times.</p>\n"
                '    <div class="sidebar_box">\n'
                "        <p>Key Note: The French Revolution context.</p>\n"
                "    </div>",
                title="Chapter 1",
            ),
            "OEBPS/ch02.xhtml": page(
                "    <h2>Chapter 2: The Mail</h2>\n"
                "    <p>It was the age of wisdom, it was the age of foolishness.</p>",
                title="Chapter 2",
            ),
        },
    )


@pytest.mark.asyncio
async def test_epub_adapter_end_to_end(tmp_path: Path) -> None:
    """Validate EPUB manifest extraction, stream parsing, and bilingual DOM injection."""
    epub_file = tmp_path / "mock_book.epub"
    create_mock_epub(epub_file)

    adapter = EPUBAdapter()

    # 1. Manifest extraction
    manifest = await adapter.extract_manifest(epub_file)
    assert manifest.title == "A Tale of Two Cities"
    assert manifest.metadata.get("author") == "Charles Dickens"
    assert len(manifest.chapters) == 2
    assert manifest.chapters[0].source_file == "OEBPS/ch01.xhtml"
    assert manifest.chapters[1].source_file == "OEBPS/ch02.xhtml"

    # 2. Parse stream
    chapters = []
    async for ch in adapter.parse_stream(epub_file):
        chapters.append(ch)

    assert len(chapters) == 2
    ch1 = chapters[0]
    assert ch1.title == "ch01.xhtml"
    assert len(ch1.blocks) == 3
    # h1 heading
    assert ch1.blocks[0].source_text == "Chapter 1: The Period"
    assert ch1.blocks[0].flow_id == FlowID.MAIN_STORY
    # p narrative
    assert "best of times" in ch1.blocks[1].source_text
    assert ch1.blocks[1].flow_id == FlowID.MAIN_STORY
    # sidebar p
    assert "Key Note" in ch1.blocks[2].source_text
    assert ch1.blocks[2].flow_id == FlowID.SIDEBAR_ASIDE  # Properly isolated!

    # 3. Store in SQLiteJobLedger and checkpoint translations
    db_path = tmp_path / "epub_ledger.db"
    with SQLiteJobLedger(db_path) as ledger:
        ledger.init_job_from_manifest(manifest.doc_id, manifest)
        ledger.append_chapter(manifest.doc_id, ch1)
        ledger.append_chapter(manifest.doc_id, chapters[1])

        # Commit translations
        ledger.save_checkpoint(
            block_id=ch1.blocks[0].id,
            status=BlockStatus.MTQE_PASSED,
            target_text="第一章：时代",
        )
        ledger.save_checkpoint(
            block_id=ch1.blocks[1].id,
            status=BlockStatus.MTQE_PASSED,
            target_text="那是最好的时代，那是最坏的时代。",
        )
        ledger.save_checkpoint(
            block_id=ch1.blocks[2].id,
            status=BlockStatus.MTQE_PASSED,
            target_text="核心注释：法国大革命背景。",
        )

        # 4. Render bilingual EPUB
        out_epub = tmp_path / "bilingual_book.epub"
        rendered_path = await adapter.render_output(
            manifest=manifest,
            ledger=ledger,
            target_lang="zh",
            output_path=out_epub,
        )
        assert rendered_path.exists()

        # 5. Verify the synthesized bilingual EPUB
        with zipfile.ZipFile(rendered_path) as zout:
            # Check 1: First entry is uncompressed mimetype
            infolist = zout.infolist()
            assert infolist[0].filename == "mimetype"
            assert infolist[0].compress_type == zipfile.ZIP_STORED

            # Check 2: ch01.xhtml contains sibling bilingual DOM tags
            ch1_html = zout.read("OEBPS/ch01.xhtml").decode("utf-8")
            soup = BeautifulSoup(ch1_html, "html.parser")

            # Check heading injection
            h1 = soup.find("h1")
            assert h1 is not None
            next_sibling = h1.find_next_sibling()
            assert next_sibling is not None
            assert next_sibling.name == "h1"
            assert "ubt-bilingual-target" in str(next_sibling.get("class"))
            assert next_sibling.get_text() == "第一章：时代"

            # Check paragraph injection
            p_orig = soup.find("p", string=lambda s: s and "best of times" in s)
            assert p_orig is not None
            p_trans = p_orig.find_next_sibling()
            assert p_trans is not None
            assert "ubt-bilingual-target" in str(p_trans.get("class"))
            assert p_trans.get_text() == "那是最好的时代，那是最坏的时代。"

            # Styling is a package-wide stylesheet, <link>ed here.
            assert soup.head is not None
            link = soup.head.find("link", attrs={"href": "ubt-bilingual.css"})
            assert link is not None
            with zipfile.ZipFile(rendered_path) as zout:
                css = zout.read("OEBPS/ubt-bilingual.css").decode("utf-8")
            assert ".ubt-bilingual-target" in css


@pytest.mark.asyncio
async def test_bilingual_multiparagraph_targets_are_sanitized(tmp_path: Path) -> None:
    """Multi-paragraph bilingual injection must run the allowlist sanitizer too.

    The single-paragraph branches sanitize before DOM injection; the
    ``\\n\\n`` split branches used to set ``new_tag.string`` from raw LLM
    output, so a prompt-injected ``<script>`` or ``onclick=`` survived into
    the stored EPUB.
    """
    epub_file = tmp_path / "mock_book.epub"
    create_mock_epub(epub_file)
    adapter = EPUBAdapter()
    manifest = await adapter.extract_manifest(epub_file)
    chapters = []
    async for ch in adapter.parse_stream(epub_file):
        chapters.append(ch)
    ch1 = chapters[0]

    with SQLiteJobLedger(tmp_path / "epub_ledger.db") as ledger:
        ledger.init_job_from_manifest(manifest.doc_id, manifest)
        for ch in chapters:
            ledger.append_chapter(manifest.doc_id, ch)
        # One paragraph split into two by the model; paragraph two smuggles
        # an event handler — exactly the multi-paragraph branch under test.
        ledger.save_checkpoint(
            block_id=ch1.blocks[1].id,
            status=BlockStatus.MTQE_PASSED,
            target_text='那是最好的时代。\n\n<img src=x onerror=alert(1)>那是<span onclick="steal()">最坏的</span>时代。',
        )

        out_epub = tmp_path / "bilingual_sanitized.epub"
        rendered = await adapter.render_output(
            manifest=manifest,
            ledger=ledger,
            target_lang="zh",
            output_path=out_epub,
        )

    with zipfile.ZipFile(rendered) as zout:
        soup = BeautifulSoup(zout.read("OEBPS/ch01.xhtml").decode("utf-8"), "html.parser")
    injected = soup.find_all(class_="ubt-bilingual-target")
    rendered_text = "".join(tag.decode() for tag in injected)
    assert "那是最好的时代。" in rendered_text
    # Event handlers must never survive the injection.
    assert "onerror" not in rendered_text
    assert "onclick" not in rendered_text
    # The <span> survives as inert text (the sanitizer escapes what it strips).
    assert "最坏的" in rendered_text


@pytest.mark.asyncio
async def test_epub_manifest_handles_missing_or_invalid_itemref(tmp_path: Path) -> None:
    """Ensure missing or non-matching spine itemref does not cause UnboundLocalError or chapter duplicates."""
    target_path = tmp_path / "broken_spine.epub"
    # itemref has non_existent_item before and after valid ch01
    write_epub(
        target_path,
        opf_xml=opf(
            pub_id="urn:uuid:broken-spine",
            title="Test Broken Spine",
            items=[item("ch01", "ch01.xhtml")],
            spine=["non_existent_first", "ch01", "non_existent_second"],
        ),
        parts={
            "OEBPS/ch01.xhtml": '<html xmlns="http://www.w3.org/1999/xhtml">'
            "<body><p>Hello world</p></body></html>"
        },
    )

    adapter = EPUBAdapter()
    manifest = await adapter.extract_manifest(target_path)
    assert len(manifest.chapters) == 1
    assert manifest.chapters[0].chapter_id == "ch_001_ch01"
    assert manifest.chapters[0].source_file == "OEBPS/ch01.xhtml"


def test_epub_table_cell_injection_preserves_structure() -> None:
    """A translated <td>/<th> is appended *inside* the cell, never as a
    sibling cell that would double the column count and wreck the table."""
    raw = (
        b"<html><head></head><body>"
        b"<table>"
        b"<tr><td>Apple</td><td>Banana</td></tr>"
        b"<tr><td>Cherry</td><td>Date</td></tr>"
        b"</table>"
        b"</body></html>"
    )
    soup = BeautifulSoup(raw.decode("utf-8"), "html.parser")
    block_names = set(BLOCK_TAGS)
    assert soup.body is not None
    leaves = [t for t in soup.body.find_all(BLOCK_TAGS) if is_leaf_block(t, block_names)]
    translation_map = {
        f"ch01#p{idx:04d}": f"译{leaf.get_text()}" for idx, leaf in enumerate(leaves)
    }

    adapter = EPUBAdapter()
    out, _injected = adapter._inject_bilingual_dom(raw, "ch01", translation_map, block_names)
    out_soup = BeautifulSoup(out.decode("utf-8"), "html.parser")

    for tr in out_soup.find_all("tr"):
        # Column count must be unchanged — the bug inserts a sibling <td>.
        direct_cells = tr.find_all(["td", "th"], recursive=False)
        assert len(direct_cells) == 2, direct_cells
        for cell in direct_cells:
            target = cell.find("div", class_="ubt-bilingual-target")
            assert target is not None, cell
            assert target.get_text().startswith("译")


def test_epub_monolingual_multiparagraph_cell_keeps_column_count() -> None:
    """Regression: a monolingual multi-paragraph translation of a <td>
    must nest its extra paragraphs *inside* the cell, never emit a sibling
    <td> that doubles the table's column count."""
    raw = (
        b"<html><head></head><body>"
        b"<table><tr><td>Apple</td><td>Banana</td></tr></table>"
        b"</body></html>"
    )
    soup = BeautifulSoup(raw.decode("utf-8"), "html.parser")
    block_names = set(BLOCK_TAGS)
    assert soup.body is not None
    # First cell gets a two-paragraph translation.
    translation_map = {"ch01#p0000": "第一行。\n\n第二行。", "ch01#p0001": "香蕉"}

    out, injected = EPUBAdapter()._inject_bilingual_dom(
        raw, "ch01", translation_map, block_names, bilingual_mode="monolingual"
    )
    out_soup = BeautifulSoup(out.decode("utf-8"), "html.parser")
    tr = out_soup.find("tr")
    assert tr is not None
    assert len(tr.find_all(["td", "th"], recursive=False)) == 2, tr
    first_cell = tr.find_all(["td", "th"], recursive=False)[0]
    text = first_cell.get_text()
    assert "第一行。" in text and "第二行。" in text
    assert injected == 3  # 2 paragraphs in the first cell + 1 in the second


def test_inject_bilingual_dom_reports_zero_for_id_mismatch() -> None:
    """The render-time untranslated guard keys off this return value: block ids
    that do not match the leaves must yield injected == 0, not a silent pass."""
    raw = b"<html><head></head><body><p>Source paragraph.</p></body></html>"
    out, injected = EPUBAdapter()._inject_bilingual_dom(
        raw,
        "ch01",
        {"totally-different-id": "译文"},
        set(BLOCK_TAGS),
    )
    assert injected == 0
    assert "译文" not in out.decode("utf-8")


# ---------------------------------------------------------------------------
# XHTML via the XML parser + original ZipInfo preservation
# ---------------------------------------------------------------------------


def test_inject_bilingual_dom_preserves_xhtml_namespaced_markup() -> None:
    """Xml parsing keeps SVG camelCase tags/attrs that html.parser lowercases."""
    adapter = EPUBAdapter()
    raw = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">'
        "<head><title>t</title></head><body>"
        '<p epub:type="note">Hello <em>world</em></p>'
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
        '<linearGradient id="g1"><stop offset="0"/></linearGradient></svg>'
        "</body></html>"
    )
    out, _injected = adapter._inject_bilingual_dom(
        raw.encode("utf-8"),
        "ch01",
        {"ch01#p0000": "你好，世界。"},
        set(BLOCK_TAGS),
        None,
    )
    soup = BeautifulSoup(out.decode("utf-8"), "xml")
    # html.parser would lowercase these to `lineargradient` / `viewbox`,
    # silently breaking inline SVG in technical books.
    assert soup.find("linearGradient") is not None
    svg = soup.find("svg")
    assert svg is not None and "viewBox" in svg.attrs
    assert soup.find("p", attrs={"epub:type": "note"}) is not None
    # Translation injection still works.
    assert "你好，世界。" in soup.get_text()


@pytest.mark.asyncio
async def test_render_blocks_reuses_original_zipinfo(tmp_path: Path) -> None:
    """Rewritten chapters keep their original timestamps and unix perms."""
    epub_file = tmp_path / "book.epub"
    create_mock_epub(epub_file)
    stamped = tmp_path / "stamped.epub"
    with zipfile.ZipFile(epub_file) as zin, zipfile.ZipFile(stamped, "w") as zout:
        for info in zin.infolist():
            data = zin.read(info.filename)
            if info.filename == "OEBPS/ch01.xhtml":
                info.date_time = (2024, 3, 4, 5, 6, 6)
                info.external_attr = 0o644 << 16
            stored = zipfile.ZIP_STORED if info.filename == "mimetype" else zipfile.ZIP_DEFLATED
            zout.writestr(info, data, compress_type=stored)

    adapter = EPUBAdapter()
    manifest = await adapter.extract_manifest(stamped)
    out = tmp_path / "out.epub"
    await adapter.render_blocks(manifest, [], "zh", out)

    with zipfile.ZipFile(out) as zf:
        info = zf.getinfo("OEBPS/ch01.xhtml")
    assert info.date_time == (2024, 3, 4, 5, 6, 6)
    assert (info.external_attr >> 16) & 0o777 == 0o644


def create_mock_epub_with_toc(target_path: Path) -> Path:
    """EPUB with NCX + EPUB3 nav + dc:language (fixture)."""
    return write_epub(
        target_path,
        opf_xml=opf(
            pub_id="urn:uuid:12345-67890",
            title="Toc Book",
            items=[
                item("ch01", "ch01.xhtml"),
                item("ncx", "toc.ncx", media_type="application/x-dtbncx+xml"),
                item("nav", "nav.xhtml", properties="nav"),
            ],
            spine=["ch01"],
            toc="ncx",
        ),
        parts={
            "OEBPS/toc.ncx": """<?xml version="1.0" encoding="UTF-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
    <head/>
    <docTitle><text>Toc Book</text></docTitle>
    <navMap>
        <navPoint id="np1" playOrder="1">
            <navLabel><text>Chapter One</text></navLabel>
            <content src="ch01.xhtml"/>
        </navPoint>
    </navMap>
</ncx>""",
            "OEBPS/nav.xhtml": (
                '<?xml version="1.0" encoding="utf-8"?>\n'
                f'<html xmlns="{XHTML_NS}" xmlns:epub="http://www.idpf.org/2007/ops">\n'
                "<head><title>nav</title></head>\n"
                "<body>\n"
                '    <nav epub:type="toc"><ol><li><a href="ch01.xhtml">Chapter One</a></li></ol></nav>\n'
                "</body>\n"
                "</html>"
            ),
            "OEBPS/ch01.xhtml": page(
                "    <h1>Chapter One</h1>\n    <p>It was the best of times.</p>",
                title="Chapter 1",
            ),
        },
    )


def _heading_block() -> Any:
    from ubt.core.ir.models import BlockStatus, BlockType, IRBlock

    return IRBlock(
        id="ch_001_ch01#p0000",
        flow_id=FlowID.MAIN_STORY,
        spine_index=1,
        block_type=BlockType.HEADING,
        source_text="Chapter One",
        target_text="第一章",
        status=BlockStatus.MTQE_PASSED,
    )


@pytest.mark.asyncio
async def test_render_blocks_updates_opf_language_and_css_item(tmp_path: Path) -> None:
    """Dc:language rewritten, CSS shipped as manifest item + link."""
    epub_file = tmp_path / "book.epub"
    create_mock_epub_with_toc(epub_file)
    adapter = EPUBAdapter()
    manifest = await adapter.extract_manifest(epub_file)

    out = tmp_path / "out.epub"
    await adapter.render_blocks(manifest, [_heading_block()], "zh", out)

    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        opf_soup = BeautifulSoup(zf.read("OEBPS/content.opf").decode("utf-8"), "xml")
        chapter_soup = BeautifulSoup(zf.read("OEBPS/ch01.xhtml").decode("utf-8"), "xml")
        css_content = zf.read("OEBPS/ubt-bilingual.css").decode("utf-8")

    assert "OEBPS/ubt-bilingual.css" in names
    lang = opf_soup.find("dc:language")
    assert lang is not None and lang.get_text(strip=True) == "zh"
    css_item = opf_soup.find("item", attrs={"href": "ubt-bilingual.css"})
    assert css_item is not None and css_item.get("media-type") == "text/css"
    assert ".ubt-bilingual-target" in css_content
    # Chapter links the shared stylesheet instead of inlining it.
    link = chapter_soup.find("link", attrs={"href": "ubt-bilingual.css"})
    assert link is not None and link.get("rel") == "stylesheet"
    inline_styles = [
        s for s in chapter_soup.find_all("style") if "ubt-bilingual-target" in s.get_text()
    ]
    assert inline_styles == []


@pytest.mark.asyncio
async def test_render_blocks_bilingual_nav_and_ncx_labels(tmp_path: Path) -> None:
    """TOC entries gain the translated chapter title bilingually."""
    epub_file = tmp_path / "book.epub"
    create_mock_epub_with_toc(epub_file)
    adapter = EPUBAdapter()
    manifest = await adapter.extract_manifest(epub_file)

    out = tmp_path / "out.epub"
    await adapter.render_blocks(manifest, [_heading_block()], "zh", out)

    with zipfile.ZipFile(out) as zf:
        ncx_soup = BeautifulSoup(zf.read("OEBPS/toc.ncx").decode("utf-8"), "xml")
        nav_soup = BeautifulSoup(zf.read("OEBPS/nav.xhtml").decode("utf-8"), "xml")

    nav_point = ncx_soup.find("navPoint")
    assert nav_point is not None
    ncx_text = nav_point.find("text")
    assert ncx_text is not None
    assert ncx_text.get_text(strip=True) == "Chapter One / 第一章"
    nav_a = nav_soup.find("a", attrs={"href": "ch01.xhtml"})
    assert nav_a is not None
    assert nav_a.get_text(strip=True) == "Chapter One / 第一章"


def create_mock_epub_with_code_block(target_path: Path) -> Path:
    """Minimal EPUB 3.0 package whose single chapter carries a <pre> block."""
    return write_epub(
        target_path,
        opf_xml=opf(
            pub_id="urn:uuid:code-book",
            title="Code Book",
            items=[item("ch01", "ch01.xhtml")],
        ),
        parts={
            "OEBPS/ch01.xhtml": page(
                "    <h1>Chapter 1</h1>\n"
                "    <p>Intro paragraph.</p>\n"
                "    <pre>def f(x): return x * 2</pre>",
                title="Chapter 1",
            )
        },
    )


@pytest.mark.asyncio
async def test_render_blocks_does_not_duplicate_verbatim_code(tmp_path: Path) -> None:
    """A3: verbatim (skip_translate) blocks must not be injected as translations.

    ingest.py stamps every CODE/IMAGE/FORMULA/kept block with
    ``target_text = source_text``; because the EPUB renderer built its lookup
    without a ``skip_translate`` filter, it injected the code a second time as a
    ``ubt-bilingual-target`` node. Shipped EPUBs read
    ``<pre>def f(x): return x * 2</pre><pre class="ubt-bilingual-target">def f(x): return x * 2</pre>``
    (code blocks duplicated in the reader) even though nothing needed translating.
    """
    epub_file = tmp_path / "code_book.epub"
    create_mock_epub_with_code_block(epub_file)
    adapter = EPUBAdapter()
    manifest = await adapter.extract_manifest(epub_file)

    chapters = [chapter async for chapter in adapter.parse_stream(epub_file)]
    blocks = chapters[0].blocks
    code_blocks = [b for b in blocks if b.block_type == BlockType.CODE]
    assert len(code_blocks) == 1
    assert code_blocks[0].skip_translate is True

    # Mirror ingest.py: skipped blocks ship verbatim (target_text == source_text),
    # everything else gets a real translation.
    for b in blocks:
        b.target_text = b.source_text if b.skip_translate else f"译:{b.source_text}"

    out = tmp_path / "out.epub"
    await adapter.render_blocks(manifest, blocks, "zh", out)

    with zipfile.ZipFile(out) as zf:
        html = zf.read("OEBPS/ch01.xhtml").decode("utf-8")

    assert html.count("def f(x): return x * 2") == 1
    rendered = BeautifulSoup(html, "xml")
    assert len(rendered.find_all("pre")) == 1
    assert rendered.find_all("pre", class_="ubt-bilingual-target") == []
    # The filter is not a blanket no-op: real translations still get injected.
    assert "译:Intro paragraph." in html


@pytest.mark.asyncio
async def test_epub_spine_href_percent_and_relative_resolution(tmp_path: Path) -> None:
    """OPF hrefs are URL references: %xx escapes and ../ segments must be
    resolved before the zip lookup, or whole chapters vanish silently."""
    body = page("<p>Real body text for the encoded chapter file.</p>")
    # NB: ``page`` supplies the <body> wrapper, so this takes the fragment only.
    epub_file = tmp_path / "encoded.epub"
    write_epub(
        epub_file,
        opf_xml=opf(
            pub_id="urn:uuid:encoded-hrefs",
            title="Encoded Hrefs",
            creator="Test",
            items=[item("a", "text%20one.xhtml"), item("b", "../Book/x=2.xhtml")],
        ),
        # One chapter percent-encoded inside OEBPS/, one reached through ../ —
        # both are URL-reference forms the zip lookup must resolve.
        parts={"OEBPS/text one.xhtml": body, "Book/x=2.xhtml": body},
    )

    adapter = EPUBAdapter()
    manifest = await adapter.extract_manifest(epub_file)
    assert [c.source_file for c in manifest.chapters] == [
        "OEBPS/text one.xhtml",
        "Book/x=2.xhtml",
    ]
    chapters = [ch async for ch in adapter.parse_stream(epub_file)]
    assert len(chapters) == 2
    assert all(ch.blocks for ch in chapters)


@pytest.mark.asyncio
async def test_inline_code_does_not_mark_whole_paragraph_code(tmp_path: Path) -> None:
    """A narrative paragraph containing an inline <code> must stay
    translatable: the CODE verdict used to swallow whole prose blocks and
    stamp them MTQE_PASSED without translating."""
    epub_file = tmp_path / "inline.epub"
    write_epub(
        epub_file,
        opf_xml=opf(
            pub_id="urn:uuid:inline-code",
            title="Inline Code",
            creator="Test",
            items=[item("a", "a.xhtml")],
        ),
        parts={
            "OEBPS/a.xhtml": page(
                "    <p>To set things up, run <code>pip install ubt</code>"
                " and then read the guide.</p>\n"
                "    <pre>pip install ubt\nubt doctor</pre>"
            )
        },
    )

    adapter = EPUBAdapter()
    chapters = [ch async for ch in adapter.parse_stream(epub_file)]
    blocks = [b for ch in chapters for b in ch.blocks]
    prose = [b for b in blocks if "read the guide" in b.source_text]
    code = [b for b in blocks if b.source_text.startswith("pip install ubt")]
    assert prose and prose[0].skip_translate is False
    assert code and code[0].skip_translate is True


@pytest.mark.asyncio
async def test_navigation_document_is_not_mined_as_a_chapter(tmp_path: Path) -> None:
    """EPUB3 lists nav.xhtml in the spine; its links are not reading matter.

    Mining it sent every table-of-contents label to the model a second time and
    re-injected each one as a dead ``<li>`` bullet next to the real link, which
    ``_update_nav_xhtml`` had already bilingualized.
    """
    epub = tmp_path / "nav.epub"
    write_epub(
        epub,
        opf_xml=opf(
            pub_id="urn:uuid:nav",
            title="Nav Book",
            creator="Author",
            items=[
                item("nav", "nav.xhtml", properties="nav"),
                item("ch01", "ch01.xhtml"),
            ],
            spine=["nav", "ch01"],
        ),
        parts={
            "OEBPS/nav.xhtml": page(
                '<nav><ul><li><a href="ch01.xhtml">Chapter One</a></li></ul></nav>',
                title="Contents",
            ),
            "OEBPS/ch01.xhtml": page("<h1>Chapter One</h1><p>Body prose here.</p>", title="c1"),
        },
    )
    adapter = EPUBAdapter()
    manifest = await adapter.extract_manifest(epub)
    assert [c.source_file for c in manifest.chapters] == ["OEBPS/ch01.xhtml"]

    chapters = [ch async for ch in adapter.parse_stream(epub)]
    mined = [b.source_text for ch in chapters for b in ch.blocks]
    assert "Body prose here." in mined
    assert not any("Chapter One" in (t or "") and "a href" in (t or "") for t in mined), mined


def test_toc_labels_match_percent_encoded_hrefs() -> None:
    """A nav href is a URL; ``title_map`` is keyed on the decoded path.

    Without the unquote, "text%20one.xhtml" never found its chapter and the
    reader kept a monolingual table of contents while the body was translated.
    """
    adapter = EPUBAdapter()
    nav = page(
        '<nav><ul><li><a href="text%20one.xhtml">Chapter One</a></li></ul></nav>', title="nav"
    )
    out = adapter._update_nav_xhtml(
        nav.encode("utf-8"), "OEBPS", {"OEBPS/text one.xhtml": "第一章"}, None
    )
    assert "第一章" in out.decode("utf-8")

    ncx = (
        "<ncx><navPoint><navLabel><text>Chapter One</text></navLabel>"
        '<content src="text%20one.xhtml"/></navPoint></ncx>'
    )
    out2 = adapter._update_ncx(
        ncx.encode("utf-8"), "OEBPS", {"OEBPS/text one.xhtml": "第一章"}, None
    )
    assert "第一章" in out2.decode("utf-8")


def test_toc_labels_never_leak_blocked_human_markup() -> None:
    """E2E: toc.xhtml carried the literal *source* of the
    quarantine placeholder (``<mark class="ubt-blocked-human" title="...">…``)
    as its link text — double-escaped grammar-tag leakage that the behaviour
    contract forbids. The label must keep only the visible text."""
    from ubt.core.engine.stages.triage import _blocked_human_target

    adapter = EPUBAdapter()
    source = "Chapter <One> & the 'thing'."
    quarantined = _blocked_human_target(source)
    assert "<mark" in quarantined  # the producer really does embed markup

    manifest = BookManifest(
        doc_id="toc_quarantine_doc",
        title="Toc Book",
        source_path="/tmp/toc_book.epub",
        chapters=[
            ChapterMeta(
                chapter_id="ch01",
                title="ch01",
                spine_index=1,
                source_file="OEBPS/ch01.xhtml",
            )
        ],
    )
    heading = IRBlock(
        id="ch01#p0000",
        flow_id=FlowID.MAIN_STORY,
        spine_index=1,
        block_type=BlockType.HEADING,
        source_text=source,
        target_text=quarantined,
    )
    title_map = adapter._translated_titles(manifest, [heading])
    label = title_map["OEBPS/ch01.xhtml"]
    # No markup of either spelling survives into the label text…
    assert "<mark" not in label
    assert "&lt;mark" not in label
    # …while the visible text — review marker and kept source — does.
    assert "【待人工审校" in label
    assert "Chapter <One> & the 'thing'." in label

    # The same guarantee end-to-end through both TOC writers.
    nav = page('<nav><ul><li><a href="ch01.xhtml">Chapter One</a></li></ul></nav>', title="nav")
    nav_out = adapter._update_nav_xhtml(nav.encode("utf-8"), "OEBPS", title_map, None).decode(
        "utf-8"
    )
    assert "<mark" not in nav_out
    assert "&lt;mark" not in nav_out
    assert "【待人工审校" in nav_out

    ncx = (
        "<ncx><navPoint><navLabel><text>Chapter One</text></navLabel>"
        '<content src="ch01.xhtml"/></navPoint></ncx>'
    )
    ncx_out = adapter._update_ncx(ncx.encode("utf-8"), "OEBPS", title_map, None).decode("utf-8")
    assert "<mark" not in ncx_out
    assert "&lt;mark" not in ncx_out
    assert "【待人工审校" in ncx_out


def test_toc_labels_strip_the_entity_escaped_spelling_too() -> None:
    """The double-escaped variant: the whole placeholder arrives as
    ``&lt;mark…&gt;`` text. Decoding then stripping must leave only text."""
    from html import escape

    adapter = EPUBAdapter()
    escaped = escape(
        '<mark class="ubt-blocked-human" title="quarantined">'
        "【待人工审校 | Human review required】Source heading</mark>"
    )
    label = adapter._translated_titles(
        BookManifest(
            doc_id="toc_escaped_doc",
            title="Toc Book 2",
            source_path="/tmp/toc_book2.epub",
            chapters=[
                ChapterMeta(
                    chapter_id="ch01",
                    title="ch01",
                    spine_index=1,
                    source_file="OEBPS/ch01.xhtml",
                )
            ],
        ),
        [
            IRBlock(
                id="ch01#p0000",
                flow_id=FlowID.MAIN_STORY,
                spine_index=1,
                block_type=BlockType.HEADING,
                source_text="Source heading",
                target_text=escaped,
            )
        ],
    )["OEBPS/ch01.xhtml"]
    assert "<mark" not in label
    assert "&lt;mark" not in label
    assert "【待人工审校" in label
    assert "Source heading" in label


@pytest.mark.fast
def test_epub_bilingual_ordered_list_injects_inside_li() -> None:
    html = b"""<?xml version="1.0" encoding="utf-8"?>
    <html xmlns="http://www.w3.org/1999/xhtml">
    <head><title>Test</title></head>
    <body>
    <ol>
        <li>Item One</li>
        <li>Item Two</li>
    </ol>
    </body>
    </html>"""

    adapter = EPUBAdapter()
    translation_map = {
        "ch001#p0000": "第一项",
        "ch001#p0001": "第二项",
    }
    block_names = {"p", "li", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6"}

    out_bytes, count = adapter._inject_bilingual_dom(
        html,
        "ch001",
        translation_map,
        block_names,
        bilingual_mode="bilingual",
    )
    out_str = out_bytes.decode("utf-8")
    # In an <ol>, inserting sibling <li> doubles the item counter (renders 1, 2, 3, 4).
    # It must NOT create 4 <li> tags under <ol>; it must preserve exactly 2 <li> elements!
    li_count = out_str.count("<li")
    assert li_count == 2, f"Expected 2 <li> elements in <ol>, got {li_count}:\n{out_str}"
    assert "第一项" in out_str
    assert "第二项" in out_str


@pytest.mark.fast
def test_epub_bilingual_unordered_list_injects_inside_li() -> None:
    """A <ul> item is an internal container too; the old guard allowed only <ol>."""
    html = b"""<?xml version="1.0" encoding="utf-8"?>
    <html xmlns="http://www.w3.org/1999/xhtml">
    <head><title>Test</title></head>
    <body>
    <ul>
        <li>Item One</li>
        <li>Item Two</li>
    </ul>
    </body>
    </html>"""

    adapter = EPUBAdapter()
    translation_map = {"ch001#p0000": "A.\n\nB.", "ch001#p0001": "第二项"}
    block_names = {"p", "li", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6"}
    out_bytes, _count = adapter._inject_bilingual_dom(
        html, "ch001", translation_map, block_names, bilingual_mode="bilingual"
    )
    out_str = out_bytes.decode("utf-8")
    # One translation per <li>: a multi-paragraph target nests inside its own
    # item instead of adding bullets.
    assert out_str.count("<li") == 2, out_str
    assert "A." in out_str and "B." in out_str


@pytest.mark.fast
def test_epub_and_html_nested_list_direct_text_preserved() -> None:
    # A list item having direct text AND a nested list
    html = b"""<?xml version="1.0" encoding="utf-8"?>
    <html xmlns="http://www.w3.org/1999/xhtml">
    <body>
    <ul>
        <li>Parent Item
            <ul>
                <li>Child Item</li>
            </ul>
        </li>
    </ul>
    </body>
    </html>"""
    adapter = EPUBAdapter()
    translation_map = {
        "ch001#p0000": "父项译文",
        "ch001#p0001": "子项译文",
    }
    block_names = {"p", "li", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6"}
    out_bytes, _ = adapter._inject_bilingual_dom(
        html,
        "ch001",
        translation_map,
        block_names,
        bilingual_mode="bilingual",
    )
    out_str = out_bytes.decode("utf-8")
    assert "Parent Item" in out_str
    assert "父项译文" in out_str
    assert "Child Item" in out_str
    assert "子项译文" in out_str


@pytest.mark.fast
def test_toc_labels_with_anchors_and_subheadings() -> None:
    """TOC navPoints pointing to subheadings/anchors must translate to their respective subheadings, not the chapter title."""
    adapter = EPUBAdapter()
    manifest = BookManifest(
        doc_id="test_toc_anchors",
        title="Test Book",
        source_path="/tmp/book.epub",
        chapters=[
            ChapterMeta(
                chapter_id="ch01", spine_index=1, title="Chapter 1", source_file="OEBPS/ch01.xhtml"
            )
        ],
    )
    blocks = [
        IRBlock(
            id="ch01#p0000",
            spine_index=1,
            block_type=BlockType.HEADING,
            source_text="Chapter 1: The Beginning",
            target_text="第一章：开端",
        ),
        IRBlock(
            id="ch01#p0001",
            spine_index=2,
            block_type=BlockType.HEADING,
            source_text="Section 1.1: Foundations",
            target_text="第 1.1 节：基础",
            provenance={"html_id": "sec1_1"},
        ),
        IRBlock(
            id="ch01#p0002",
            spine_index=3,
            block_type=BlockType.HEADING,
            source_text="Section 1.2: Advanced",
            target_text="第 1.2 节：进阶",
            provenance={"html_id": "sec1_2"},
        ),
    ]
    title_maps = adapter._translated_titles(manifest, blocks)

    ncx = (
        "<ncx>"
        '<navPoint id="np1"><navLabel><text>Chapter 1: The Beginning</text></navLabel><content src="ch01.xhtml"/></navPoint>'
        '<navPoint id="np2"><navLabel><text>Section 1.1: Foundations</text></navLabel><content src="ch01.xhtml#sec1_1"/></navPoint>'
        '<navPoint id="np3"><navLabel><text>Section 1.2: Advanced</text></navLabel><content src="ch01.xhtml#sec1_2"/></navPoint>'
        "</ncx>"
    )
    out_ncx = adapter._update_ncx(
        ncx.encode("utf-8"), "OEBPS", title_maps, bilingual_mode="target"
    ).decode("utf-8")
    assert "第一章：开端" in out_ncx
    assert "第 1.1 节：基础" in out_ncx
    assert "第 1.2 节：进阶" in out_ncx


@pytest.mark.fast
def test_overrides_from_request_blocks_ocr_endpoint_and_epub_locates_single_quoted_opf() -> None:
    """overrides_from_request must block ocr_endpoint when
    allow_provider_keys=False, and EPUBAdapter._locate_opf must parse single-quoted full-path."""
    with pytest.raises(UBTError, match="ocr_endpoint"):
        overrides_from_request(
            {"ocr_endpoint": "http://169.254.169.254/latest"}, allow_provider_keys=False
        )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "META-INF/container.xml",
            "<?xml version='1.0'?><container><rootfiles>"
            "<rootfile full-path='OEBPS/content.opf' media-type='application/oebps-package+xml'/>"
            "</rootfiles></container>",
        )
    buf.seek(0)
    with zipfile.ZipFile(buf, "r") as zf:
        adapter = EPUBAdapter()
        assert adapter._locate_opf(zf) == "OEBPS/content.opf"


@pytest.mark.asyncio
async def test_parse_stream_runs_off_the_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The per-chapter zip read + BeautifulSoup parse must not block the loop."""
    import asyncio
    import time

    import ubt.adapters.epub.adapter as epub_mod

    epub_file = create_mock_epub(tmp_path / "offload.epub")
    calls: list[int] = []

    def slow(*_args: Any, **_kwargs: Any) -> list[IRBlock]:
        calls.append(1)
        time.sleep(0.1)
        return []

    monkeypatch.setattr(epub_mod, "_parse_chapter_blocks", slow)

    ticks = 0
    stop = False

    async def heartbeat() -> None:
        nonlocal ticks
        while not stop:
            ticks += 1
            await asyncio.sleep(0.005)

    task = asyncio.create_task(heartbeat())
    try:
        adapter = EPUBAdapter()
        async for _chapter in adapter.parse_stream(epub_file):
            pass
    finally:
        stop = True
        await task
    assert calls, "parse_stream did not route through the offloaded parser"
    assert ticks >= 5, f"event loop stalled during parse_stream (ticks={ticks})"


def test_unsafe_epub_member_names_are_rejected() -> None:
    from ubt.adapters.epub.adapter import _is_safe_epub_member_name

    assert _is_safe_epub_member_name("OEBPS/ch01.xhtml")
    assert not _is_safe_epub_member_name("../evil.xhtml")
    assert not _is_safe_epub_member_name("a/../../b.xhtml")
    assert not _is_safe_epub_member_name("/etc/passwd")
    assert not _is_safe_epub_member_name("C:evil.xhtml")


def test_epub_named_entities_not_swallowed_during_xml_parse() -> None:
    """EPUB XHTML parser must preserve HTML named entities instead of lxml recover swallowing them."""
    from ubt.adapters.epub.adapter import _parse_xhtml

    raw_xhtml = '<p id="p1">Hello&nbsp;world&mdash;&ldquo;Quote&rdquo;&copy;&hellip;</p>'
    soup = _parse_xhtml(raw_xhtml)
    text = soup.get_text()
    assert "\xa0" in text, f"Non-breaking space was swallowed: {text!r}"
    assert "—" in text, f"M-dash was swallowed: {text!r}"
    assert "“" in text and "”" in text, f"Quotes were swallowed: {text!r}"
    assert "©" in text, f"Copyright was swallowed: {text!r}"
    assert "…" in text, f"Ellipsis was swallowed: {text!r}"


def test_epub_injects_head_if_missing() -> None:
    """If chapter XHTML lacks a <head>, bilingual CSS injection must create <head> and link CSS."""
    from ubt.adapters.epub.adapter import EPUBAdapter

    adapter = EPUBAdapter()
    raw_html = b'<html xmlns="http://www.w3.org/1999/xhtml"><body><p>Hello world.</p></body></html>'
    translation_map = {"ch01#p0000": "你好世界。"}
    block_names = {"p"}

    rewritten, count = adapter._inject_bilingual_dom(
        raw_html=raw_html,
        chapter_id="ch01",
        translation_map=translation_map,
        block_names=block_names,
        stylesheet_href="../ubt-bilingual.css",
    )
    assert count == 1
    assert b"<head>" in rewritten
    assert b'href="../ubt-bilingual.css"' in rewritten


def test_epub_bilingual_retains_source_classes() -> None:
    """Bilingual sibling node in EPUB must retain source element's CSS classes."""
    from bs4 import BeautifulSoup

    from ubt.adapters.base import BILINGUAL_TARGET_CLASS
    from ubt.adapters.epub.adapter import EPUBAdapter

    adapter = EPUBAdapter()
    raw_html = (
        b'<html xmlns="http://www.w3.org/1999/xhtml"><head></head>'
        b'<body><h2 class="chapter-sub title-heavy">Chapter Subtitle</h2></body></html>'
    )
    translation_map = {"ch01#p0000": "章节副标题"}
    block_names = {"h2"}

    rewritten, count = adapter._inject_bilingual_dom(
        raw_html=raw_html,
        chapter_id="ch01",
        translation_map=translation_map,
        block_names=block_names,
    )
    assert count == 1
    soup = BeautifulSoup(rewritten, "xml")
    h2s = soup.find_all("h2")
    assert len(h2s) == 2
    target_h2 = h2s[1]
    raw_classes = target_h2.get("class")
    classes = (
        list(raw_classes)
        if isinstance(raw_classes, list)
        else (str(raw_classes).split() if raw_classes else [])
    )
    assert "chapter-sub" in classes
    assert "title-heavy" in classes
    assert BILINGUAL_TARGET_CLASS in classes


def test_epub_labels_unresolved_drafts() -> None:
    """An unresolved draft must stay visible *and labelled* (ubt.adapters.unresolved).

    Regression: ``_inject_bilingual_dom`` injected the bare machine draft exactly
    like an approved translation, so a reader/reviewer could not tell them apart.
    """
    from bs4 import BeautifulSoup as _BS

    adapter = EPUBAdapter()
    raw_html = (
        b'<html xmlns="http://www.w3.org/1999/xhtml"><head></head>'
        b"<body><p>Source english paragraph.</p></body></html>"
    )
    translation_map = {"ch01#p0000": "未经审核的机器草稿。"}
    unresolved_notes = {"ch01#p0000": "[UBT] draft for review:"}

    rewritten, count = adapter._inject_bilingual_dom(
        raw_html=raw_html,
        chapter_id="ch01",
        translation_map=translation_map,
        block_names={"p"},
        unresolved_notes=unresolved_notes,
    )
    # The note is its own paragraph before the draft, so both are injected.
    assert count == 2
    soup = _BS(rewritten, "xml")
    text = soup.get_text(" ", strip=True)
    assert "未经审核的机器草稿。" in text
    assert "[UBT]" in text


def test_epub_monolingual_preserves_inline_media_and_anchors() -> None:
    """The monolingual rewrite must keep inline images and footnote anchors."""
    adapter = EPUBAdapter()
    raw_html = (
        b'<html xmlns="http://www.w3.org/1999/xhtml"><head></head><body>'
        b'<p>See <img src="fig.png" alt="fig"/> and '
        b'<a href="#fn1" id="ref1">[1]</a> for details.</p></body></html>'
    )
    rewritten, count = adapter._inject_bilingual_dom(
        raw_html=raw_html,
        chapter_id="ch01",
        translation_map={"ch01#p0000": "详见插图与脚注。"},
        block_names={"p"},
        bilingual_mode="monolingual",
    )
    assert count == 1
    out = rewritten.decode("utf-8")
    assert "详见插图与脚注。" in out
    assert "fig.png" in out
    assert 'href="#fn1"' in out
    assert 'id="ref1"' in out


@pytest.mark.asyncio
async def test_epub_external_stylesheet_is_scrubbed(tmp_path: Path) -> None:
    """A copied ``.css`` member must be gated like an inline ``<style>`` body.

    Regression: only ``.xhtml``/``.html``/``.xml``/``.svg`` members went through
    the scrub, so a malicious source could ship a remote ``@import`` (network
    fetch) or a ``url(javascript:…)`` the reader would honour.
    """
    epub_file = tmp_path / "styled.epub"
    write_epub(
        epub_file,
        opf_xml=opf(
            pub_id="urn:uuid:styled",
            title="Styled",
            items=[
                item("ch1", "OEBPS/ch1.xhtml"),
                item("css", "OEBPS/style.css", media_type="text/css"),
            ],
        ),
        parts={
            "OEBPS/ch1.xhtml": page("<p>Hello world.</p>", title="ch1"),
            "OEBPS/style.css": (
                '@import url("https://evil.example/x.css");\n'
                "p { background: url(javascript:alert(1)); }\n"
            ),
        },
    )
    adapter = EPUBAdapter()
    manifest = await adapter.extract_manifest(epub_file)

    out = tmp_path / "out.epub"
    with SQLiteJobLedger(tmp_path / "led.db") as ledger:
        ledger.init_job_from_manifest(manifest.doc_id, manifest)
        await adapter.render_output(
            manifest=manifest, ledger=ledger, target_lang="zh", output_path=out
        )

    with zipfile.ZipFile(out) as zout:
        css = zout.read("OEBPS/style.css").decode("utf-8")
    assert "evil.example" not in css
    assert "javascript" not in css


def test_epub_refuses_to_read_an_oversized_member() -> None:
    """A zip-bomb member (huge declared uncompressed size) must not be read.

    ``zipfile.read`` stops at the declared size, so capping it before the read
    is what keeps one crafted member from exhausting memory while still
    delivering the rest of the book.
    """
    import io

    from ubt.adapters.epub.adapter import _MAX_EPUB_MEMBER_BYTES, _read_epub_member

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("small.txt", b"ok")
        zf.writestr("bomb.bin", b"x" * 1024)
    buf.seek(0)
    with zipfile.ZipFile(buf) as zf:
        assert _read_epub_member(zf, "small.txt") == b"ok"
        assert _read_epub_member(zf, "missing") is None
        # Simulate the bomb declaration without allocating 300 MB.
        zf.getinfo("bomb.bin").file_size = _MAX_EPUB_MEMBER_BYTES + 1
        assert _read_epub_member(zf, "bomb.bin") is None


@pytest.mark.asyncio
async def test_epub_non_utf8_chapter_is_decoded_not_dropped(tmp_path: Path) -> None:
    """A chapter encoded in a non-UTF-8 charset must keep its text.

    Regression: every member was decoded as UTF-8 with ``errors="ignore"``, so a
    gb2312-encoded chapter silently lost all of its Chinese text.
    """
    from tests.epub_builders import CONTAINER_XML

    body = (
        '<?xml version="1.0" encoding="gb2312"?>\n'
        f'<html xmlns="{XHTML_NS}"><head><title>t</title></head>'
        "<body><p>中文字符串测试。</p></body></html>"
    )
    opf_xml = opf(pub_id="urn:uuid:gbk", title="GBK", items=[item("ch1", "ch1.xhtml")])
    path = tmp_path / "gbk.epub"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(
            zipfile.ZipInfo("mimetype"),
            b"application/epub+zip",
            compress_type=zipfile.ZIP_STORED,
        )
        zf.writestr("META-INF/container.xml", CONTAINER_XML)
        zf.writestr("OEBPS/content.opf", opf_xml)
        zf.writestr("OEBPS/ch1.xhtml", body.encode("gb2312"))

    adapter = EPUBAdapter()
    await adapter.extract_manifest(path)
    blocks: list[IRBlock] = []
    async for chapter in adapter.parse_stream(path):
        blocks.extend(chapter.blocks)
    assert any("中文字符串测试" in b.source_text for b in blocks), [b.source_text for b in blocks]
