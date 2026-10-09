"""The semantic HTML/EPUB views.

The view lowers the delivered blocks: it must carry each block's placed text
(the delivered translation where one was placed, the source slice where it was
kept) and place every block. The EPUB view is the same body wrapped in a valid
OCF container (mimetype first and stored, container/OPF/nav/text present).
Fixtures are synthetic text elements; the real-PDF path stays with the corpus
harness in the slow tier.
"""

from __future__ import annotations

import html
import zipfile
from pathlib import Path

import pytest

from ubt.core.ir.models import IRBlock
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
    # Total placement: every block got one, and nothing was kept in source.
    assert len(composition.placements) == len(blocks)
    assert not composition.kept_source_ids
    for block in blocks:
        expected = html.escape(element_text(block, translations))
        assert expected and expected in text, block.id


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
    # A render-skip means the source shipped; the view must show the source, not
    # the (unplaced) translation.
    blocks[1].error_flags = ["render_skip:space_failure"]
    out = tmp_path / "view.html"
    compose_html(blocks, translations, out)
    text = out.read_text(encoding="utf-8")
    assert "<p>The body of the matter.</p>" in text
    assert REHEARSAL_MARKER + " The body of the matter." not in text
