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
from typing import Any, ClassVar

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
    StyledRun,
    TypstFragmentTypesetter,
    _dedup_identical_streams,
    _line_slack,
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
#: The compositor adds the line slack below a box before fitting and drawing, so
#: an overlay with no source size (the tests') draws into the region plus this.
_SLACK = _line_slack(None)


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
        self.indents: list[float | None] = []
        self.math_calls: list[tuple[str, float, float]] = []
        self.bilingual_calls: list[tuple[str, str, float, float]] = []
        self.prefetched: list[tuple[Any, ...]] = []

    def prefetch(self, requests: Sequence[tuple[Any, ...]]) -> None:
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
        indent_pt: float | None = None,
        align_center: bool = False,
        runs: tuple[Any, ...] = (),
    ) -> Path | None:
        self.calls.append((text, width_pt, height_pt))
        self.indents.append(indent_pt)
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
        ("TRANSLATED REGION TEXT", _REGION[2] - _REGION[0], _REGION[3] - _REGION[1] + _SLACK)
    ]


def test_a_reflowed_overlay_draws_at_its_box_and_masks_the_source(tmp_path: Path) -> None:
    # A reflowed overlay draws at its new box with no line slack, but the source
    # text is erased at the *original* box, so the mask follows ``mask_boxes``.
    source = write_text_pdf(tmp_path / "source.pdf", [_PAGE])
    output = tmp_path / "out.pdf"
    spy = _FragmentSpy(tmp_path)
    draw = (54.0, 640.0, 354.0, 670.0)
    overlay = Overlay(
        "e1",
        1,
        draw,
        "TRANSLATED REFLOW",
        fixed_box=True,
        mask_boxes=(PhysicalBox.of(1, _REGION),),
    )

    LayerCompositor(source, typesetter=spy).compose([overlay], output)

    # No slack: the drawn height is exactly the box height.
    assert spy.calls == [("TRANSLATED REFLOW", draw[2] - draw[0], draw[3] - draw[1])]
    text = _text(output)
    assert "TRANSLATED REFLOW" in text
    # The source line that sat in the original box was stripped, not left behind.
    assert _flat(_PAGE[1]) not in text


def test_a_continuation_box_the_target_does_not_reach_is_still_masked(tmp_path: Path) -> None:
    # A paragraph crossing a page break owns two boxes: most of it on page 1 and
    # the spilled tail on page 2. When the (shorter) translation fits the first
    # box alone, the flow leaves the second empty -- but its source tail still
    # sits there, so it must be erased or a stray source word survives next to
    # the translation.
    source = write_text_pdf(tmp_path / "source.pdf", [["The paragraph begins here"], ["agent."]])
    output = tmp_path / "out.pdf"
    spy = _FragmentSpy(tmp_path)
    first = (54.0, 700.0, 354.0, 730.0)
    second = (54.0, 725.0, 120.0, 745.0)
    overlay = Overlay(
        "e1",
        1,
        first,
        "TRANSLATED",  # 10 chars: fits the first box under the spy's length measure
        boxes=(PhysicalBox.of(1, first), PhysicalBox.of(2, second)),
    )

    LayerCompositor(source, typesetter=spy).compose([overlay], output)

    pages = oxide_render.extract_page_texts(output)
    assert "TRANSLATED" in _flat(pages[0])
    # The page-2 tail was erased, not left as stray source.
    assert "agent." not in _flat(pages[1])


def test_overlapping_overlays_do_not_erase_each_other(tmp_path: Path) -> None:
    # The source strip runs once per page, before any overlay is stamped. A
    # per-overlay strip recursed into the Form of an overlay already drawn on
    # the same page and deleted its glyphs when the two boxes overlapped.
    source = write_text_pdf(tmp_path / "source.pdf", [_PAGE])
    output = tmp_path / "out.pdf"
    spy = _FragmentSpy(tmp_path)
    first = Overlay("e1", 1, (54.0, 640.0, 300.0, 700.0), "TRANSLATED ONE")
    second = Overlay("e2", 1, (100.0, 660.0, 360.0, 720.0), "TRANSLATED TWO")

    composition = LayerCompositor(source, typesetter=spy).compose([first, second], output)

    assert all(p.placed_as is Fidelity.RECONSTRUCTED_ADAPTED for p in composition.placements)
    text = _text(output)
    assert "TRANSLATED ONE" in text
    assert "TRANSLATED TWO" in text


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
    assert spy.math_calls == [
        ("e^{i\\pi}+1=0", _REGION[2] - _REGION[0], _REGION[3] - _REGION[1] + _SLACK)
    ]


def test_an_overlay_box_past_the_page_is_clamped(tmp_path: Path) -> None:
    # An extraction box can spill past the mediabox; the fragment must be sized
    # and drawn against the clamp, or the visual gate reports block_out_of_bounds.
    source = write_text_pdf(tmp_path / "source.pdf", [_PAGE])
    spy = _FragmentSpy(tmp_path)
    overlay = Overlay("b1", 1, (100.0, 700.0, 900.0, 812.0), "spilling text")

    LayerCompositor(source, typesetter=spy).compose([overlay], tmp_path / "out.pdf")

    _text, width, height = spy.calls[0]
    assert width == 612.0 - 100.0
    assert height == 792.0 - 700.0 + _SLACK


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

    assert [(request[0], request[1]) for request in spy.prefetched] == [
        ("text", "first"),
        ("text", "second"),
    ]
    width, height = spy.prefetched[0][2], spy.prefetched[0][3]
    assert (width, height) == (_REGION[2] - _REGION[0], _REGION[3] - _REGION[1] + _SLACK)


def test_a_prefetch_request_carries_the_draw_time_style(tmp_path: Path) -> None:
    # Regression: the request omitted is_bold/align_center/runs -- is_bold even
    # received draw_size_pt -- so every styled-run or centred fragment missed the
    # warmed cache and spawned one Typst process per box at draw time.
    source = write_text_pdf(tmp_path / "source.pdf", [_PAGE])
    spy = _FragmentSpy(tmp_path)
    runs = (StyledRun("first"),)
    overlay = Overlay("a", 1, _REGION, "first", is_bold=True, align_center=True, runs=runs)

    LayerCompositor(source, typesetter=spy).compose([overlay], tmp_path / "out.pdf")

    (request,) = spy.prefetched
    kind, text, _w, _h, _fs, indent, fixed, is_bold, draw_size, align_center, got_runs = request
    assert (kind, text) == ("text", "first")
    assert indent is None
    assert fixed is False
    assert is_bold is True
    assert draw_size is None
    assert align_center is True
    assert got_runs == runs


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
        (
            "SOURCE TEXT",
            "TRANSLATED TEXT",
            _REGION[2] - _REGION[0],
            _REGION[3] - _REGION[1] + _SLACK,
        )
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
    kind, text, _w, _h, _fs = request
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


# --------------------------------------------------------------------------- #
# Typst fragment source: every section names its weight explicitly
# --------------------------------------------------------------------------- #


def _typesetter(tmp_path: Path) -> TypstFragmentTypesetter:
    return TypstFragmentTypesetter(
        font=("Noto Serif CJK SC",), size_pt=10.0, cache_dir=tmp_path, target_lang="zh"
    )


def test_an_indented_overlay_reaches_the_typesetter_with_its_indent(tmp_path: Path) -> None:
    # A paragraph the band reflow left alone (its target does not fit the source
    # box, so it keeps the fitted path) must still draw with the source's
    # first-line indent: dropping it dropped the indent of every body paragraph
    # and of every list item's marker column.
    source = write_text_pdf(tmp_path / "source.pdf", [_PAGE])
    output = tmp_path / "out.pdf"
    spy = _FragmentSpy(tmp_path)
    overlay = Overlay(
        "e1", 1, _REGION, "(1) TRANSLATED REGION TEXT", font_size=10.0, indent_pt=18.0
    )

    LayerCompositor(source, typesetter=spy).compose([overlay], output)

    assert spy.indents == [18.0]
    assert "TRANSLATED REGION TEXT" in _text(output)


def test_an_indented_fragment_source_carries_its_first_line_indent(tmp_path: Path) -> None:
    # The indent is an inline non-weak hspace, not ``par.first-line-indent``:
    # inside a fixed-height box the paragraph rule makes the content a hair
    # taller than the box, which spills to a second page (the compositor copies
    # page 1 and draws nothing). The empty block after the hspace and the space
    # that follows it are both load-bearing: without the space a body opening
    # with "(" ("(1) Rollout ...") parses as a call and fails to compile.
    ts = _typesetter(tmp_path)
    src = ts._text_source("(1) 正文", 100.0, 20.0, 10.0, indent_pt=22.0)
    assert "#h(22.0pt, weak: false)#[] " in src
    assert "#h(22.0pt, weak: false)#[](1)" not in src


def test_a_plain_fragment_source_names_weight_regular(tmp_path: Path) -> None:
    ts = _typesetter(tmp_path)
    src = ts._text_source("正文", 100.0, 20.0, 10.0, kind="text", is_bold=False)
    assert 'weight: "regular"' in src
    assert 'weight: "bold"' not in src


def test_a_heading_fragment_source_names_weight_bold(tmp_path: Path) -> None:
    ts = _typesetter(tmp_path)
    src = ts._text_source("标题", 100.0, 20.0, 10.0, kind="heading")
    assert 'weight: "bold"' in src


def test_measure_bilingual_and_math_sources_all_name_a_weight(tmp_path: Path) -> None:
    ts = _typesetter(tmp_path)
    assert 'weight: "regular"' in ts._measure_source("正文", 100.0, 10.0)
    assert 'weight: "bold"' in ts._measure_source("标题", 100.0, 10.0, kind="heading")
    key = bilingual_request_text("目标", "源文")
    assert 'weight: "regular"' in ts._bilingual_measure_source(key, 100.0, 10.0)
    assert 'weight: "regular"' in ts._bilingual_text_source("目标", "源文", 100.0, 20.0, 10.0)
    math_src = ts._math_source("$x$", 100.0, 20.0)
    assert math_src is not None and 'weight: "regular"' in math_src


def test_a_batched_document_does_not_leak_bold_into_later_sections(tmp_path: Path) -> None:
    # Typst ``#set`` rules apply to the end of the enclosing content: a bold
    # section (a heading) once leaked its weight into every following section
    # whose ``#set text`` omitted the parameter, bolding whole pages of body
    # text. Every section must therefore carry an explicit weight.
    ts = _typesetter(tmp_path)
    heading = ts._text_source("标题", 100.0, 20.0, 10.0, kind="heading")
    body = ts._text_source("正文", 100.0, 20.0, 10.0, kind="text")
    batch = f"{heading}\n#pagebreak()\n{body}"
    sections = batch.split("#pagebreak()")
    for section in sections:
        set_line = next(ln for ln in section.splitlines() if ln.startswith("#set text"))
        assert 'weight: "' in set_line, section
    assert 'weight: "regular"' in sections[1]


def test_multiple_overlays_on_same_page_coalesce_into_single_page_form_xobject(
    tmp_path: Path,
) -> None:
    # When multiple overlays exist on the same page, LayerCompositor consolidates
    # them into a single page-level Form XObject, avoiding Form XObject proliferation
    # and quadratic stream rewriting.
    import pikepdf

    source = write_text_pdf(tmp_path / "source.pdf", [_PAGE])
    output = tmp_path / "out.pdf"
    spy = _FragmentSpy(tmp_path)
    first = Overlay("e1", 1, (54.0, 640.0, 300.0, 700.0), "FRAGMENT ONE")
    second = Overlay("e2", 1, (54.0, 560.0, 300.0, 620.0), "FRAGMENT TWO")
    third = Overlay("e3", 1, (54.0, 480.0, 300.0, 540.0), "FRAGMENT THREE")

    composition = LayerCompositor(source, typesetter=spy).compose([first, second, third], output)

    assert all(p.placed_as is Fidelity.RECONSTRUCTED_ADAPTED for p in composition.placements)
    with pikepdf.open(output) as pdf:
        page = pdf.pages[0]
        # The page's own Resources /XObject must contain exactly 1 top-level Form XObject
        assert len(page.Resources.XObject) == 1


def test_multiple_overlays_coalesce_micro_masks_into_single_stream(
    tmp_path: Path,
) -> None:
    # When multiple overlays are stamped without stripping, their bounding
    # box masks are batched into a single graphics stream rather than O(N) streams.
    import pikepdf

    source = write_text_pdf(tmp_path / "source.pdf", [_PAGE])
    output = tmp_path / "out.pdf"
    spy = _FragmentSpy(tmp_path)
    first = Overlay("e1", 1, (54.0, 640.0, 300.0, 700.0), "FRAGMENT ONE")
    second = Overlay("e2", 1, (54.0, 560.0, 300.0, 620.0), "FRAGMENT TWO")

    # Initial stream count of source page
    with pikepdf.open(source) as pdf:
        initial_contents_len = (
            len(pdf.pages[0].Contents) if isinstance(pdf.pages[0].Contents, pikepdf.Array) else 1
        )

    LayerCompositor(source, typesetter=spy, strip=False).compose([first, second], output)

    with pikepdf.open(output) as pdf:
        page = pdf.pages[0]
        final_contents_len = len(page.Contents) if isinstance(page.Contents, pikepdf.Array) else 1
        # Exactly 1 consolidated mask stream was appended, plus overlay stamping
        # Total streams increased by at most 2, never O(N) separate mask streams
        assert final_contents_len <= initial_contents_len + 2
