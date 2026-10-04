"""The LayerCompositor prototype: three-layer absolute page composition.

Pins the invariants the composition design rests on, without a Typst toolchain
(the real fragment typesetter is exercised in the ``slow`` tier):

- Layer 0 carries every source page over verbatim (geometry and content);
- Layer 2's fragment is masked into and stamped at the region, and the mask is
  painted only after the fragment is in hand -- a failed fragment leaves the
  source untouched (never a mask with nothing under it);
- only above-the-floor, placed, delivered text elements become overlays.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import ClassVar

import pytest
from pdf_builders import write_text_pdf

from ubt.adapters.factory import get_adapter_for_path
from ubt.adapters.pdf import oxide_render, pdf_struct
from ubt.analyze.bridge import document_from_blocks
from ubt.core.ir.models import IRBlock
from ubt.model.ast import Document
from ubt.model.fidelity import Attestation, Fidelity, Proof, ProofKind
from ubt.model.span import PhysicalBox
from ubt.render.outputs import (
    LayerCompositor,
    Overlay,
    _dedup_identical_streams,
    bilingual_request_text,
    overlays_from_document,
)

pytestmark = pytest.mark.fast

_PAGE = (
    "The Attention Machine",
    "The machine relies on attention and runs a forward pass.",
)
#: A generous region box (the extractor's own bboxes are tight line boxes; a
#: fake fragment needs room to hold text at a normal point size).
_REGION = (54.0, 700.0, 354.0, 730.0)


def _flat(text: str) -> str:
    return " ".join(text.split())


def _text(path: Path) -> str:
    return _flat("\n".join(oxide_render.extract_page_texts(path)))


def _document(path: Path) -> Document:
    adapter = get_adapter_for_path(path, pdf_engine="pdfium")

    async def collect() -> list[IRBlock]:
        blocks: list[IRBlock] = []
        async for chapter in adapter.parse_stream(path):
            blocks.extend(chapter.blocks)
        return blocks

    return document_from_blocks(asyncio.run(collect()), doc_id="synth")


class _FragmentSpy:
    """A fragment typesetter that authors one PDF page at the requested box size."""

    name: ClassVar[str] = "fragment-spy"

    def __init__(
        self,
        tmp: Path,
        *,
        fail: bool = False,
        measure: Callable[[str, float], float] | None = None,
    ) -> None:
        self._tmp = tmp
        self._fail = fail
        self._measure = measure
        self.calls: list[tuple[str, float, float]] = []
        self.math_calls: list[tuple[str, float, float]] = []
        self.bilingual_calls: list[tuple[str, str, float, float]] = []
        self.prefetched: list[tuple[str, str, float, float]] = []

    def prefetch(self, requests: Sequence[tuple[str, str, float, float]]) -> None:
        self.prefetched.extend(requests)

    def typeset(
        self,
        text: str,
        width_pt: float,
        height_pt: float,
        *,
        kind: str = "text",
        font_size: float | None = None,
        is_bold: bool = False,
    ) -> Path | None:
        self.calls.append((text, width_pt, height_pt))
        if self._fail:
            return None
        fragment = self._tmp / f"fragment-{len(self.calls)}.pdf"
        return write_text_pdf(fragment, [[text]], width=width_pt, height=height_pt, margin=10.0)

    def typeset_math(self, latex: str, width_pt: float, height_pt: float) -> Path | None:
        self.math_calls.append((latex, width_pt, height_pt))
        if self._fail:
            return None
        fragment = self._tmp / f"math-{len(self.math_calls)}.pdf"
        return write_text_pdf(fragment, [[latex]], width=width_pt, height=height_pt, margin=10.0)

    def typeset_bilingual(
        self, source: str, target: str, width_pt: float, height_pt: float
    ) -> Path | None:
        self.bilingual_calls.append((source, target, width_pt, height_pt))
        if self._fail:
            return None
        fragment = self._tmp / f"bilingual-{len(self.bilingual_calls)}.pdf"
        return write_text_pdf(
            fragment,
            [[f"{target} / {source}"]],
            width=width_pt,
            height=height_pt,
            margin=10.0,
        )

    def measure(self, text: str, width_pt: float) -> float:
        if self._measure is not None:
            return self._measure(text, width_pt)
        return float(len(text))


def test_layer0_is_a_verbatim_page_copy(tmp_path: Path) -> None:
    source = write_text_pdf(tmp_path / "source.pdf", [_PAGE])
    output = tmp_path / "out.pdf"

    composition = LayerCompositor(source).compose([], output)

    assert composition.placements == ()
    assert pdf_struct.page_sizes(output) == pdf_struct.page_sizes(source)
    assert _text(output) == _text(source)


def test_a_region_is_masked_and_its_fragment_stamped(tmp_path: Path) -> None:
    source = write_text_pdf(tmp_path / "source.pdf", [_PAGE])
    output = tmp_path / "out.pdf"
    spy = _FragmentSpy(tmp_path)
    overlay = Overlay("e1", 1, _REGION, "TRANSLATED REGION TEXT")

    composition = LayerCompositor(source, typesetter=spy).compose([overlay], output)

    (placement,) = composition.placements
    assert placement.placed_as is Fidelity.RECONSTRUCTED_ADAPTED
    assert not placement.descended
    # Layer 0 geometry is preserved; Layer 2 text is present in the artifact.
    assert pdf_struct.page_sizes(output) == pdf_struct.page_sizes(source)
    assert "TRANSLATED REGION TEXT" in _text(output)
    # The box handed to the typesetter is the region size, in points.
    assert spy.calls == [
        ("TRANSLATED REGION TEXT", _REGION[2] - _REGION[0], _REGION[3] - _REGION[1])
    ]


def test_a_failed_fragment_descends_and_leaves_the_source_untouched(tmp_path: Path) -> None:
    source = write_text_pdf(tmp_path / "source.pdf", [_PAGE])
    output = tmp_path / "out.pdf"
    overlay = Overlay("e1", 1, _REGION, "TRANSLATED REGION TEXT")

    composition = LayerCompositor(source, typesetter=_FragmentSpy(tmp_path, fail=True)).compose(
        [overlay], output
    )

    (placement,) = composition.placements
    assert placement.placed_as is Fidelity.PRESERVED_OPAQUE
    assert placement.descended
    # No fragment -> no mask: the source text must be exactly what it was.
    assert _text(output) == _text(source)
    assert "TRANSLATED REGION TEXT" not in _text(output)


def test_overlays_select_only_above_floor_placed_delivered_text(tmp_path: Path) -> None:
    source = write_text_pdf(tmp_path / "source.pdf", [_PAGE])
    document = _document(source)
    text_elements = [element for element in document.elements if element.is_text]
    assert text_elements
    delivered = {element.id: "target" for element in text_elements}

    above_floor = [
        Attestation(element.id, Fidelity.RECONSTRUCTED_ADAPTED, Proof.ok(ProofKind.STRUCTURAL))
        for element in text_elements
    ]
    overlays = overlays_from_document(document, above_floor, delivered)
    assert {overlay.element_id for overlay in overlays} == {element.id for element in text_elements}

    at_floor = [
        Attestation(element.id, Fidelity.PRESERVED_OPAQUE, Proof.preserved())
        for element in text_elements
    ]
    assert overlays_from_document(document, at_floor, delivered) == ()

    # No delivered text -> not an overlay either.
    assert overlays_from_document(document, above_floor, {}) == ()


def test_a_math_overlay_uses_the_math_typesetter(tmp_path: Path) -> None:
    source = write_text_pdf(tmp_path / "source.pdf", [_PAGE])
    output = tmp_path / "out.pdf"
    spy = _FragmentSpy(tmp_path)
    overlay = Overlay("f1", 1, _REGION, "e^{i\\pi}+1=0", kind="math")

    LayerCompositor(source, typesetter=spy).compose([overlay], output)

    assert spy.calls == []  # prose typesetter untouched
    assert spy.math_calls == [("e^{i\\pi}+1=0", _REGION[2] - _REGION[0], _REGION[3] - _REGION[1])]


def test_an_overlay_box_past_the_page_is_clamped(tmp_path: Path) -> None:
    # An extraction box can spill past the mediabox; the fragment must be sized
    # and drawn against the clamp, or the visual gate reports block_out_of_bounds.
    source = write_text_pdf(tmp_path / "source.pdf", [_PAGE])
    spy = _FragmentSpy(tmp_path)
    overlay = Overlay("b1", 1, (100.0, 700.0, 900.0, 812.0), "spilling text")

    LayerCompositor(source, typesetter=spy).compose([overlay], tmp_path / "out.pdf")

    _text, width, height = spy.calls[0]
    assert width == 612.0 - 100.0
    assert height == 792.0 - 700.0


def test_identical_streams_are_deduplicated() -> None:
    # Every fragment carries a copy of the shared font/CMap; the compositor must
    # collapse byte-identical streams or the artifact bloats to tens of MB.
    import pikepdf

    pdf = pikepdf.new()
    pdf.add_blank_page()
    data = b"shared-resource" * 1000
    first = pdf.make_indirect(pikepdf.Stream(pdf, data))
    second = pdf.make_indirect(pikepdf.Stream(pdf, data))
    holder_a = pdf.make_indirect(pikepdf.Dictionary())
    holder_b = pdf.make_indirect(pikepdf.Dictionary())
    holder_a[pikepdf.Name("/X")] = first
    holder_b[pikepdf.Name("/X")] = second

    _dedup_identical_streams(pdf)

    assert holder_a[pikepdf.Name("/X")].objgen == holder_b[pikepdf.Name("/X")].objgen


def test_the_compositor_prefetches_single_box_fragments(tmp_path: Path) -> None:
    # The parallel prefetch is the whole cost of a large document; the compositor
    # must warm it for every single-box overlay before drawing, at the clamped
    # size so the draw-time lookup hits the compiled file.
    source = write_text_pdf(tmp_path / "source.pdf", [_PAGE])
    spy = _FragmentSpy(tmp_path)
    overlays = [Overlay("a", 1, _REGION, "first"), Overlay("b", 1, _REGION, "second")]

    LayerCompositor(source, typesetter=spy).compose(overlays, tmp_path / "out.pdf")

    assert [(kind, text) for kind, text, _w, _h in spy.prefetched] == [
        ("text", "first"),
        ("text", "second"),
    ]
    (_kind, _text, width, height) = spy.prefetched[0]
    assert (width, height) == (_REGION[2] - _REGION[0], _REGION[3] - _REGION[1])


def test_an_in_place_bilingual_overlay_stamps_both_languages(tmp_path: Path) -> None:
    # An overlay carrying a source draws target over source in one fragment, so a
    # bilingual page reads both without a second engine.
    source = write_text_pdf(tmp_path / "source.pdf", [_PAGE])
    output = tmp_path / "out.pdf"
    spy = _FragmentSpy(tmp_path)
    overlay = Overlay("e1", 1, _REGION, "TRANSLATED TEXT", source="SOURCE TEXT")

    composition = LayerCompositor(source, typesetter=spy).compose([overlay], output)

    (placement,) = composition.placements
    assert placement.placed_as is Fidelity.RECONSTRUCTED_ADAPTED
    assert spy.bilingual_calls == [
        ("SOURCE TEXT", "TRANSLATED TEXT", _REGION[2] - _REGION[0], _REGION[3] - _REGION[1])
    ]
    assert spy.calls == []  # the monolingual path is not used for a bilingual overlay
    text = _text(output)
    assert "TRANSLATED TEXT" in text and "SOURCE TEXT" in text


def test_the_compositor_prefetches_a_bilingual_overlay_as_one_request(tmp_path: Path) -> None:
    source = write_text_pdf(tmp_path / "source.pdf", [_PAGE])
    spy = _FragmentSpy(tmp_path)
    overlay = Overlay("e1", 1, _REGION, "TARGET", source="SOURCE")

    LayerCompositor(source, typesetter=spy).compose([overlay], tmp_path / "out.pdf")

    (request,) = spy.prefetched
    kind, text, _w, _h = request
    assert kind == "bilingual"
    assert text == bilingual_request_text("TARGET", "SOURCE")


def test_text_flows_across_a_chain_of_boxes(tmp_path: Path) -> None:
    source = write_text_pdf(tmp_path / "source.pdf", [_PAGE])
    output = tmp_path / "out.pdf"
    first = (54.0, 720.0, 354.0, 740.0)
    second = (54.0, 680.0, 354.0, 710.0)
    spy = _FragmentSpy(tmp_path, measure=lambda text, _width: float(len(text)))
    overlay = Overlay(
        "e1",
        1,
        first,
        "Alpha beta gamma. Delta epsilon zeta. Eta theta.",
        boxes=(PhysicalBox.of(1, first), PhysicalBox.of(1, second)),
    )

    composition = LayerCompositor(source, typesetter=spy).compose([overlay], output)

    (placement,) = composition.placements
    assert placement.placed_as is Fidelity.RECONSTRUCTED_ADAPTED
    # One fragment per box, with the first broken at the punctuation boundary.
    assert len(spy.calls) == 2
    assert spy.calls[0][0] == "Alpha beta gamma."
    assert spy.calls[1][0] == "Delta epsilon zeta. Eta theta."
    delivered = _text(output)
    assert "Alpha beta gamma." in delivered
    assert "Delta epsilon zeta. Eta theta." in delivered
