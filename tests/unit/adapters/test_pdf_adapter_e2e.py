"""Synthetic-PDF end-to-end: the real adapter extracts, bridges, and lowers.

The PDF adapter family is the largest subsystem in the tree and, until this
module, had no ``fast``-tier coverage: the harnesses that exercise it
(``scripts/shadow_*.py`` via ``tests/integration/test_corpus_acceptance.py``)
live in the ``slow`` tier and need the gitignored corpus, so CI -- which runs
``pytest -m fast`` only -- never reached them. A PDF *render* cannot be a
``fast`` test either: every render path shells out to the external ``typst``
binary, which CI's ``uv sync --extra dev`` does not install.

So this pins what can be proven with the pure-runtime dependency set and no
toolchain, over a synthetic PDF built by an independent writer
(:mod:`tests.pdf_builders`):

- the adapter extracts the source text without loss and keeps the page number;
- the IR <-> AST bridge round-trips the extraction unchanged;
- the source preservation floor (``OverlayBackend``) lowers every element
  losslessly, the Axiom-A/Axiom-B construction-time guarantee.

It is deliberately *not* a translation-quality or renderer test -- those stay
with the corpus gate. It is the regression net under the PDF *ingest* half.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from pathlib import Path

import pytest
from pdf_builders import write_text_pdf, write_two_column_pdf

from ubt.adapters.factory import get_adapter_for_path
from ubt.adapters.pdf.pdfium_adapter import PDFiumAdapter
from ubt.analyze.bridge import blocks_from_document, document_from_blocks
from ubt.core.ir.models import IRBlock
from ubt.core.qe.fast_pass import FastPassFilter
from ubt.model.fidelity import Fidelity, ProofKind
from ubt.pipeline.steps import realize
from ubt.render.overlay_backend import OverlayBackend, source_slice
from ubt.verify.verifier import build_verifiers

pytestmark = pytest.mark.fast

#: A single-column page exercising prose, a citation, and an inline math token.
_PAGE_ONE = (
    "The Attention Machine",
    "The machine relies on attention. Each layer runs a forward pass over the batch.",
    "Euler wrote e^{i pi} + 1 = 0 linking five constants of analysis.",
    "Following Smith et al. (2020) the score reached 98.45 percent on every benchmark.",
)
_PAGE_TWO = (
    "The Descendants",
    "The second chapter widens the lens to every descendant of the first model.",
)

_SOURCE_TEXT = "\n".join((*_PAGE_ONE, *_PAGE_TWO))


def _chars(text: str) -> Counter[str]:
    """Whitespace-insensitive character histogram (segmentation may differ)."""
    return Counter(ch for ch in text if not ch.isspace())


def _blocks(path: Path, *, pdf_engine: str) -> list[IRBlock]:
    adapter = get_adapter_for_path(path, pdf_engine=pdf_engine)

    async def collect() -> list[IRBlock]:
        blocks: list[IRBlock] = []
        async for chapter in adapter.parse_stream(path):
            blocks.extend(chapter.blocks)
        return blocks

    return asyncio.run(collect())


def test_auto_routes_a_single_column_born_digital_pdf_to_the_pdfium_engine(
    tmp_path: Path,
) -> None:
    """Pin the engine probe: a plain single-column PDF must not need the OCR extra."""
    pdf = write_text_pdf(tmp_path / "single.pdf", [_PAGE_ONE])
    assert isinstance(get_adapter_for_path(pdf, pdf_engine="auto"), PDFiumAdapter)


def test_extraction_keeps_every_source_character_and_the_page_number(tmp_path: Path) -> None:
    """No lossy extraction: every non-space source char survives, page numbers carry."""
    pdf = write_text_pdf(tmp_path / "book.pdf", [_PAGE_ONE, _PAGE_TWO])
    blocks = _blocks(pdf, pdf_engine="pdfium")

    assert blocks, "the adapter extracted no blocks from a born-digital page"
    assert all(block.source_text.strip() for block in blocks)

    extracted: Counter[str] = Counter()
    for block in blocks:
        extracted.update(_chars(block.source_text))
    missing = _chars(_SOURCE_TEXT) - extracted
    assert sum(missing.values()) == 0, f"extraction dropped characters: {dict(missing)}"

    pages = {block.bbox.page for block in blocks if block.bbox is not None}
    assert pages == {1, 2}, f"expected both pages represented, got {sorted(pages)}"


def test_extraction_round_trips_through_the_bridge(tmp_path: Path) -> None:
    """IR -> AST -> IR preserves reading order, identity, and element kind."""
    pdf = write_text_pdf(tmp_path / "book.pdf", [_PAGE_ONE, _PAGE_TWO])
    blocks = _blocks(pdf, pdf_engine="pdfium")

    document = document_from_blocks(blocks, doc_id="synth")
    rebuilt = blocks_from_document(document)

    assert [block.id for block in rebuilt] == [block.id for block in blocks]
    assert [block.block_type for block in rebuilt] == [block.block_type for block in blocks]
    assert [block.source_text for block in rebuilt] == [block.source_text for block in blocks]


def test_a_cross_page_sentence_becomes_one_continuation_run(tmp_path: Path) -> None:
    # A sentence split by a page break is fused by the adapter into a single
    # block carrying a CompositeSpan, so translation receives the full sentence.
    from ubt.model.span import CompositeSpan

    pdf = write_text_pdf(
        tmp_path / "book.pdf",
        [["The machine relies on attention and"], ["runs a forward pass over the batch."]],
    )
    blocks = _blocks(pdf, pdf_engine="pdfium")

    assert len(blocks) == 1
    (block,) = blocks
    span = block.element.span
    assert isinstance(span, CompositeSpan)
    assert [box.page for box in span.boxes] == [1, 2]
    assert "The machine relies on attention and runs a forward pass" in block.source_text


def test_a_cross_column_sentence_becomes_one_continuation_run(tmp_path: Path) -> None:
    # A two-column page: the paragraph finishes at the bottom of the left column
    # and resumes at the top of the right; the adapter fuses them into one CompositeSpan.
    from ubt.model.span import CompositeSpan

    left = [f"left column filler line number {i} continues here" for i in range(1, 9)]
    right = ["and yet it runs in seconds on", "modern hardware today."] + [
        f"right column filler line {i}" for i in range(3, 9)
    ]
    pdf = write_two_column_pdf(tmp_path / "columns.pdf", [(left, right)])
    blocks = _blocks(pdf, pdf_engine="pdfium")

    composite_blocks = [b for b in blocks if isinstance(b.element.span, CompositeSpan)]
    assert len(composite_blocks) == 1
    (fused,) = composite_blocks
    fused_span = fused.element.span
    assert isinstance(fused_span, CompositeSpan)
    assert [box.page for box in fused_span.boxes] == [1, 1]
    assert (
        "left column filler line number 8 continues here and yet it runs in seconds"
        in fused.source_text
    )


def test_the_source_preservation_floor_lowers_every_element_losslessly(tmp_path: Path) -> None:
    """Overlay lowering is lossless by construction: preserved, delivered, carrying text."""
    pdf = write_text_pdf(tmp_path / "book.pdf", [_PAGE_ONE, _PAGE_TWO])
    document = document_from_blocks(_blocks(pdf, pdf_engine="pdfium"), doc_id="synth")
    backend = OverlayBackend()
    verifiers = build_verifiers(FastPassFilter(source_lang="en", target_lang="zh"))

    assert document.elements, "the extracted document has no elements to lower"
    for element in document.elements:
        attestation = realize(element, backend, verifiers, document.source)
        assert attestation.fidelity is Fidelity.PRESERVED_OPAQUE, element.id
        assert attestation.delivered, element.id
        assert attestation.proof.kind is ProofKind.PRESERVED, element.id
        assert source_slice(element, document.source), element.id
