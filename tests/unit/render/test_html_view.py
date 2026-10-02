"""The semantic HTML/EPUB views.

The L6 view lowers a *realized* document: it must carry each element's attested
text (the delivered translation where reconstructed), place every element, and
**refuse a coverage gap rather than silently dropping it**. The EPUB view is
the same body wrapped in a valid OCF container (mimetype first and stored,
container/OPF/nav/text present). Fixtures are synthetic text elements; the
real-PDF path stays with the corpus harness in the slow tier.
"""

from __future__ import annotations

import html
import zipfile
from pathlib import Path

import pytest

from ubt.analyze.bridge import document_from_blocks
from ubt.core.ir.models import IRBlock
from ubt.core.qe.fast_pass import REHEARSAL_MARKER, FastPassFilter
from ubt.layout.theme import resolve_theme
from ubt.model.ast import Document, ElementT, Heading, Paragraph
from ubt.model.fidelity import Attestation
from ubt.model.span import Span
from ubt.pipeline.steps import realize
from ubt.render.epub_view import compose_epub
from ubt.render.html_view import _element_text, compose_html
from ubt.render.outputs import LoweringUnsupported
from ubt.render.typst_backend import REFLOW_CLASSES, TypstBackend
from ubt.verify.verifier import build_verifiers

_ELEMENTS: tuple[ElementT, ...] = (
    Heading(id="h1", spine_index=0, span=Span(page=1), text="Chapter One"),
    Paragraph(id="p1", spine_index=1, span=Span(page=1), text="The body of the matter."),
    Paragraph(id="p2", spine_index=2, span=Span(page=1), text="A second paragraph."),
)


def _document() -> tuple[Document, dict[str, str], list[Attestation]]:
    blocks = [IRBlock(element=element) for element in _ELEMENTS]
    document = document_from_blocks(blocks, doc_id="doc", path="book.md")
    translations = {
        element.id: f"{REHEARSAL_MARKER} {element.text}"
        for element in _ELEMENTS
        if isinstance(element, REFLOW_CLASSES) and not element.skip_translate
    }
    verifiers = build_verifiers(FastPassFilter(source_lang="en", target_lang="zh"))
    backend = TypstBackend(translations, theme=resolve_theme("en", "zh"))
    attestations = [
        realize(element, backend, verifiers, document.source) for element in document.elements
    ]
    return document, translations, attestations


def test_the_html_view_carries_every_attested_text(tmp_path: Path) -> None:
    document, translations, attestations = _document()
    out = tmp_path / "view.html"
    composition = compose_html(document, attestations, translations, out)
    text = out.read_text(encoding="utf-8")

    assert text.startswith("<!DOCTYPE html")
    # Total placement: every element got one, and nothing was descended.
    assert len(composition.placements) == len(document.elements)
    assert not composition.descended_ids
    by_id = {attestation.element_id: attestation for attestation in attestations}
    for element in document.elements:
        expected = html.escape(
            _element_text(element, by_id[element.id].fidelity, document.source, translations)
        )
        assert expected and expected in text, element.id


def test_the_epub_view_is_a_valid_ocf_container(tmp_path: Path) -> None:
    document, translations, attestations = _document()
    out = tmp_path / "view.epub"
    compose_epub(document, attestations, translations, out, title="doc")

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


def test_a_missing_attestation_is_refused_not_dropped(tmp_path: Path) -> None:
    document, translations, attestations = _document()
    with pytest.raises(LoweringUnsupported):
        compose_html(document, attestations[:-1], translations, tmp_path / "gap.html")


def test_the_elements_all_realize_above_the_floor() -> None:
    # The fixture's promise: every synthetic text element reconstructs, so the
    # view tests exercise the delivered-translation path, not the source slice.
    _, _, attestations = _document()
    assert len(attestations) == len(_ELEMENTS)
    for element, attestation in zip(_ELEMENTS, attestations, strict=True):
        assert attestation.element_id == element.id
        assert attestation.fidelity.name == "RECONSTRUCTED_ADAPTED"
