from __future__ import annotations

import html
import zipfile
from pathlib import Path

import pytest

from ubt.core.ir.models import BlockStatus, IRBlock
from ubt.core.qe.fast_pass import REHEARSAL_MARKER
from ubt.model.ast import ElementT, Heading, Paragraph
from ubt.model.span import Span
from ubt.render.epub_view import compose_epub
from ubt.render.html_view import compose_html, element_text

pytestmark = pytest.mark.fast

_ELEMENTS: tuple[ElementT, ...] = (
    Heading(id="h1", spine_index=0, span=Span(page=1), text="Chapter One"),
    Paragraph(id="p1", spine_index=1, span=Span(page=1), text="The body of the matter."),
    Paragraph(id="p2", spine_index=2, span=Span(page=1), text="A second paragraph."),
)


def _blocks() -> tuple[list[IRBlock], dict[str, str]]:
    blocks = [
        IRBlock(element=element, target_text=f"{REHEARSAL_MARKER} {element.text}")
        for element in _ELEMENTS
        if isinstance(element, (Heading, Paragraph))
    ]
    translations = {block.id: block.target_text or "" for block in blocks}
    return blocks, translations


def test_the_html_view_carries_every_placed_text(tmp_path: Path) -> None:
    blocks, translations = _blocks()
    out = tmp_path / "view.html"
    composition = compose_html(blocks, translations, out)
    text = out.read_text(encoding="utf-8")

    assert text.startswith("<!DOCTYPE html")
    assert len(composition.placements) == len(blocks)
    for block in blocks:
        expected = html.escape(element_text(block, translations))
        assert expected and expected in text, block.id


def test_ordered_list_items_keep_their_numbering_in_the_companion_view(
    tmp_path: Path,
) -> None:
    from ubt.model.ast import ListItem

    elements: tuple[Paragraph | ListItem, ...] = (
        Paragraph(id="p0", spine_index=0, span=Span(page=1), text="Steps:"),
        ListItem(id="li1", spine_index=1, span=Span(page=1), text="First step.", marker="1."),
        ListItem(id="li2", spine_index=2, span=Span(page=1), text="Second step.", marker="2."),
        ListItem(id="li3", spine_index=3, span=Span(page=1), text="A bullet aside.", marker="•"),
    )
    blocks = [IRBlock(element=el, target_text=el.text) for el in elements]
    translations = {block.id: block.target_text or "" for block in blocks}
    out = tmp_path / "view.html"
    compose_html(blocks, translations, out)
    text = out.read_text(encoding="utf-8")
    assert "<ol>\n<li>First step.</li>" in text
    assert "<li>Second step.</li>" in text
    assert "</ol>\n<ul>" in text
    assert "<ul>\n<li>A bullet aside.</li>" in text


def test_the_epub_view_is_a_valid_ocf_container(tmp_path: Path) -> None:
    blocks, translations = _blocks()
    out = tmp_path / "view.epub"
    compose_epub(blocks, translations, out, title="doc")

    with zipfile.ZipFile(out) as archive:
        names = archive.namelist()
        infos = {info.filename: info for info in archive.infolist()}
        assert names and names[0] == "mimetype"
        assert infos["mimetype"].compress_type == zipfile.ZIP_STORED
        for required in (
            "META-INF/container.xml",
            "OEBPS/content.opf",
            "OEBPS/nav.xhtml",
            "OEBPS/text.xhtml",
        ):
            assert required in names
        text_xhtml = archive.read("OEBPS/text.xhtml").decode("utf-8")
        assert "The body of the matter." in text_xhtml


def test_a_block_kept_in_source_shows_its_source_text(tmp_path: Path) -> None:
    blocks, translations = _blocks()
    blocks[1].error_flags = ["render_skip:space_failure"]
    out = tmp_path / "view.html"
    compose_html(blocks, translations, out)
    text = out.read_text(encoding="utf-8")
    assert "<p>The body of the matter.</p>" in text
    assert REHEARSAL_MARKER + " The body of the matter." not in text


def test_a_quarantined_block_shows_source_not_the_marker(tmp_path: Path) -> None:
    blocks, translations = _blocks()
    blocks[1].status = BlockStatus.BLOCKED_HUMAN
    blocks[1].target_text = (
        '<mark class="ubt-blocked-human">【待人工审校 | Human review required】'
        "The body of the matter.</mark>"
    )
    translations.pop(blocks[1].id, None)
    out = tmp_path / "view.html"
    compose_html(blocks, translations, out)
    text = out.read_text(encoding="utf-8")
    assert "<p>The body of the matter.</p>" in text
    assert "待人工审校" not in text
    assert "ubt-blocked-human" not in text


def _styled_block(block_id: str, text: str, **run_kwargs: object) -> IRBlock:
    from ubt.core.ir.models import InlineRun, StyleMeta

    run = InlineRun(text=text, **run_kwargs)  # type: ignore[arg-type]
    return IRBlock(
        element=Paragraph(id=block_id, spine_index=1, span=Span(page=1), text="src"),
        target_text=f"The {text} stays styled.",
        style=StyleMeta(inline_runs=(run,)),
    )


def test_a_bold_source_run_ships_as_strong_in_the_html_view(tmp_path: Path) -> None:
    blocks = [_styled_block("b1", "key phrase", bold=True)]
    translations = {b.id: b.target_text or "" for b in blocks}
    out = tmp_path / "view.html"
    compose_html(blocks, translations, out)
    text = out.read_text(encoding="utf-8")

    assert "<strong>key phrase</strong>" in text
    assert "The <strong>key phrase</strong> stays styled." in text


def test_colour_and_superscript_runs_ship_as_html_styles(tmp_path: Path) -> None:
    blocks = [
        _styled_block("c1", "link text", color_hex="#1a4fd6"),
        _styled_block("s1", "†", superscript=True),
    ]
    translations = {b.id: b.target_text or "" for b in blocks}
    out = tmp_path / "view.html"
    compose_html(blocks, translations, out)
    text = out.read_text(encoding="utf-8")

    assert '<span style="color:#1a4fd6">link text</span>' in text
    assert "<sup>†</sup>" in text


def test_an_ital_run_ships_as_em_and_a_rewritten_run_stays_plain(tmp_path: Path) -> None:
    from ubt.core.ir.models import InlineRun, StyleMeta

    blocks = [
        IRBlock(
            element=Paragraph(id="i1", spine_index=1, span=Span(page=1), text="src"),
            target_text="The term stays styled.",
            style=StyleMeta(inline_runs=(InlineRun(text="term", italic=True),)),
        ),
        IRBlock(
            element=Paragraph(id="gone", spine_index=2, span=Span(page=1), text="src"),
            target_text="The rewritten wording.",
            style=StyleMeta(inline_runs=(InlineRun(text="vanished span", bold=True),)),
        ),
    ]
    translations = {b.id: b.target_text or "" for b in blocks}
    out = tmp_path / "view.html"
    compose_html(blocks, translations, out)
    text = out.read_text(encoding="utf-8")

    assert "<em>term</em>" in text
    assert "<strong>" not in text.split("</p>")[1]


def test_the_epub_view_carries_the_same_strong_markup(tmp_path: Path) -> None:
    blocks = [_styled_block("b1", "key phrase", bold=True)]
    translations = {b.id: b.target_text or "" for b in blocks}
    out = tmp_path / "view.epub"
    compose_epub(blocks, translations, out, title="doc")

    with zipfile.ZipFile(out) as archive:
        xhtml = archive.read("OEBPS/text.xhtml").decode("utf-8")
    assert "<strong>key phrase</strong>" in xhtml


def test_a_run_surviving_only_after_cjk_normalization_still_locates(tmp_path: Path) -> None:
    from ubt.core.ir.models import InlineRun, StyleMeta

    block = IRBlock(
        element=Paragraph(id="cjk", spine_index=1, span=Span(page=1), text="src"),
        target_text="关键 短语 位于此处。",
        style=StyleMeta(inline_runs=(InlineRun(text="关键短语", bold=True),)),
    )
    translations = {block.id: block.target_text or ""}
    out = tmp_path / "view.html"
    compose_html([block], translations, out)
    text = out.read_text(encoding="utf-8")

    assert "<strong>关键 短语</strong>" in text


def test_a_table_grid_lowers_to_a_real_table(tmp_path: Path) -> None:
    from ubt.model.ast import Table

    block = IRBlock(
        element=Table(
            id="t1",
            spine_index=1,
            span=Span(page=1),
            markup="| Model | Score |\n|---|---|\n| A | 1 |\n| B | 2 |",
        )
    )
    out = tmp_path / "view.html"
    compose_html([block], {}, out)
    text = out.read_text(encoding="utf-8")

    assert "<table>" in text
    assert "<th>Model</th><th>Score</th>" in text
    assert "<td>A</td><td>1</td>" in text
    assert "<td>B</td><td>2</td>" in text
    assert '<pre class="table">' not in text


def test_an_unparseable_table_grid_stays_a_pre_not_dropped(tmp_path: Path) -> None:
    from ubt.model.ast import Table

    block = IRBlock(
        element=Table(id="t2", spine_index=1, span=Span(page=1), markup="not a table at all")
    )
    out = tmp_path / "view.html"
    compose_html([block], {}, out)
    text = out.read_text(encoding="utf-8")

    assert '<pre class="table">not a table at all</pre>' in text
    assert "<table>" not in text


def test_a_formula_block_carries_latex_and_a_readable_fallback(tmp_path: Path) -> None:
    from ubt.model.ast import Formula

    block = IRBlock(
        element=Formula(
            id="f1", spine_index=1, span=Span(page=1), source=r"\frac{a}{b} \le \sum_i x_i"
        )
    )
    out = tmp_path / "view.html"
    compose_html([block], {}, out)
    text = out.read_text(encoding="utf-8")

    assert '<span class="math-tex"' in text
    assert "\\frac{a}{b}" in text
    assert "a/b ≤ ∑_i x_i" in text


def test_a_figure_with_a_resolvable_asset_ships_an_img(tmp_path: Path) -> None:
    from ubt.model.ast import Figure

    asset = tmp_path / "fig.png"
    asset.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    block = IRBlock(element=Figure(id="g1", spine_index=1, span=Span(page=1), asset_id=str(asset)))
    out = tmp_path / "view.html"
    compose_html([block], {}, out)
    text = out.read_text(encoding="utf-8")

    assert '<img src="data:image/png;base64,' in text
    assert str(tmp_path) not in text
    assert 'data-asset="fig.png"' in text


def test_a_figure_with_a_missing_asset_degrades_to_a_labelled_placeholder(
    tmp_path: Path,
) -> None:
    from ubt.model.ast import Figure

    block = IRBlock(
        element=Figure(
            id="g2", spine_index=1, span=Span(page=1), asset_id="/nonexistent/pic_p1_1.png"
        )
    )
    out = tmp_path / "view.html"
    compose_html([block], {}, out)
    text = out.read_text(encoding="utf-8")

    assert "figure-missing" in text
    assert "[figure: pic_p1_1.png]" in text
    assert "/nonexistent" not in text


def test_the_epub_packages_figures_with_manifest_entries(tmp_path: Path) -> None:
    from ubt.model.ast import Figure

    first = tmp_path / "a.png"
    first.write_bytes(b"AAA")
    second = tmp_path / "b.png"
    second.write_bytes(b"BBB")
    blocks = [
        IRBlock(element=Figure(id="g1", spine_index=1, span=Span(page=1), asset_id=str(first))),
        IRBlock(element=Figure(id="g2", spine_index=2, span=Span(page=1), asset_id=str(second))),
    ]
    out = tmp_path / "view.epub"
    compose_epub(blocks, {}, out, title="doc")

    with zipfile.ZipFile(out) as archive:
        names = archive.namelist()
        assert "OEBPS/images/0001.png" in names
        assert "OEBPS/images/0002.png" in names
        assert archive.read("OEBPS/images/0001.png") == b"AAA"
        opf = archive.read("OEBPS/content.opf").decode("utf-8")
        assert 'href="images/0001.png" media-type="image/png"' in opf
        assert 'href="images/0002.png" media-type="image/png"' in opf
        xhtml = archive.read("OEBPS/text.xhtml").decode("utf-8")
        assert '<img src="images/0001.png"' in xhtml
        assert '<img src="images/0002.png"' in xhtml


def test_the_epub_deduplicates_identical_figure_bytes(tmp_path: Path) -> None:
    from ubt.model.ast import Figure

    first = tmp_path / "a.png"
    first.write_bytes(b"SAME")
    second = tmp_path / "b.png"
    second.write_bytes(b"SAME")
    blocks = [
        IRBlock(element=Figure(id="g1", spine_index=1, span=Span(page=1), asset_id=str(first))),
        IRBlock(element=Figure(id="g2", spine_index=2, span=Span(page=1), asset_id=str(second))),
    ]
    out = tmp_path / "view.epub"
    compose_epub(blocks, {}, out, title="doc")

    with zipfile.ZipFile(out) as archive:
        images = [n for n in archive.namelist() if n.startswith("OEBPS/images/")]
        assert images == ["OEBPS/images/0001.png"]
        xhtml = archive.read("OEBPS/text.xhtml").decode("utf-8")
        assert xhtml.count('src="images/0001.png"') == 2


def test_a_failed_draft_wrapper_never_reaches_the_companion_views(tmp_path: Path) -> None:
    blocks, translations = _blocks()
    wrapped = (
        '<mark class="ubt-failed-draft" title="REPAIR ERROR RuntimeError: boom">你好世界。</mark>'
    )
    blocks[1].target_text = wrapped
    translations[blocks[1].id] = wrapped

    assert element_text(blocks[1], {}) == "你好世界。"
    assert element_text(blocks[1], translations) == "你好世界。"

    out = tmp_path / "view.html"
    compose_html(blocks, translations, out)
    text = out.read_text(encoding="utf-8")
    assert "你好世界。" in text
    assert "ubt-failed-draft" not in text, "the wrapper class must not reach the page"
    assert "RuntimeError" not in text, "the exception text must not reach the page"

    epub = tmp_path / "view.epub"
    compose_epub(blocks, translations, epub, title="doc")
    with zipfile.ZipFile(epub) as archive:
        body = archive.read("OEBPS/text.xhtml").decode("utf-8")
    assert "ubt-failed-draft" not in body
    assert "RuntimeError" not in body


def test_toc_rows_and_running_heads_are_not_promoted_to_section_headings(tmp_path: Path) -> None:
    from ubt.core.ir.models import BlockType, FlowID, make_element

    stamp = IRBlock(
        element=make_element(
            id="stamp",
            spine_index=0,
            block_type=BlockType.HEADING,
            flow_id=FlowID.MAIN_STORY,
            source_text="arXiv:2101.03961v3 [cs.CL] 20 Jan 2021",
            level=1,
        ),
        target_text="arXiv:2101.03961v3 [cs.CL] 20 Jan 2021",
    )
    toc_row = IRBlock(
        element=make_element(
            id="toc",
            spine_index=2,
            block_type=BlockType.HEADING,
            flow_id=FlowID.MAIN_STORY,
            source_text="1 Introduction",
            level=1,
        ),
        target_text="1 引言",
    )
    toc_row.provenance.toc_entry = True

    out = tmp_path / "view.html"
    compose_html([stamp, *_blocks()[0], toc_row], {}, out)
    text = out.read_text(encoding="utf-8")

    assert "arXiv:2101.03961" not in text, "a running head must not open the HTML body"
    assert '<ol class="ubt-toc">' in text, "TOC rows belong in a contents list"
    assert '<li class="ubt-toc-entry">1 引言</li>' in text
    assert "<h1>1 引言</h1>" not in text, "a TOC row is not a section heading"
    assert text.count("<ol") == text.count("</ol>"), "the TOC list must be closed"
    assert text.count("<ul") == text.count("</ul>"), "list tags must balance"


def _figure_block(tmp_path: Path) -> tuple[list[IRBlock], dict[str, str]]:
    from ubt.core.ir.models import BlockType, FlowID, make_element

    image = tmp_path / "figure.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
    block = IRBlock(
        element=make_element(
            id="pdf_main#img_0001",
            spine_index=1,
            block_type=BlockType.IMAGE,
            flow_id=FlowID.CAPTION,
            source_text=str(image),
            skip_translate=True,
        ),
        target_text=str(image),
    )
    return [block], {}


def test_an_epub_with_a_figure_is_well_formed_xml(tmp_path: Path) -> None:
    import xml.etree.ElementTree as ET

    blocks, translations = _figure_block(tmp_path)
    out = tmp_path / "figure.epub"
    compose_epub(blocks, translations, out, title="doc")

    with zipfile.ZipFile(out) as archive:
        text_xhtml = archive.read("OEBPS/text.xhtml").decode("utf-8")
    assert "<img" in text_xhtml
    ET.fromstring(text_xhtml)
