"""The LayerCompositor's real fragment typesetter (Typst), in the ``slow`` tier.

The fast composer tests use a fake fragment writer; this exercises the actual
micro-core: Typst laying out one bounded box, with markup characters escaped so
they render literally, compiled and stamped onto a source page.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from pdf_builders import write_text_pdf

from ubt.adapters.pdf import oxide_render, pdf_struct
from ubt.adapters.pdf.visual_gate import (
    artifact_text_boxes,
    page_bounds,
    render_pages_to_png,
    text_occlusion_findings,
)
from ubt.model.fidelity import Fidelity
from ubt.render.outputs import LayerCompositor, Overlay, TypstFragmentTypesetter

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        shutil.which("typst") is None,
        reason="the typst compiler is not installed; fragments cannot be typeset",
    ),
]

_PAGE = ("The Attention Machine", "The machine relies on attention and runs a forward pass.")
_REGION = (54.0, 700.0, 354.0, 730.0)


def _flat(text: str) -> str:
    return " ".join(text.split())


def _text(path: Path) -> str:
    return _flat("\n".join(oxide_render.extract_page_texts(path)))


def test_an_over_long_fragment_is_shrunk_rather_than_occluded(tmp_path: Path) -> None:
    # Regression: target text taller than its box was clipped, leaving text in
    # the layer with no ink -- exactly the visual gate's ``text_occluded``. The
    # fit search shrinks the font until the wrapped text fits, so the artifact
    # line carries ink.
    source = write_text_pdf(tmp_path / "source.pdf", [_PAGE])
    box = (54.0, 640.0, 300.0, 652.0)  # a one-line box, far shorter than the text
    long_text = "the quick brown fox jumps over the lazy dog " * 12
    output = tmp_path / "out.pdf"
    typesetter = TypstFragmentTypesetter()
    try:
        LayerCompositor(source, typesetter=typesetter).compose(
            [Overlay("long", 1, box, long_text)], output
        )
    finally:
        typesetter.close()

    pngs = render_pages_to_png(output, [1])
    boxes = artifact_text_boxes(output, [1])
    findings = text_occlusion_findings(pngs, boxes, page_bounds(output))
    assert [finding.code for finding in findings] == []


def test_prefetch_compiles_many_fragments_in_batches(tmp_path: Path) -> None:
    # The whole point of batching: every fragment resolves to its own PDF without
    # one Typst process per fragment.
    requests = [
        ("text", f"fragment number {index} with a few words here", 200.0, 12.0)
        for index in range(30)
    ]
    typesetter = TypstFragmentTypesetter()
    try:
        typesetter.prefetch(requests)
        for _kind, text, width_pt, height_pt in requests:
            fragment = typesetter.typeset(text, width_pt, height_pt)
            assert fragment is not None and fragment.exists()
    finally:
        typesetter.close()


def test_an_in_place_bilingual_fragment_carries_both_languages(tmp_path: Path) -> None:
    # In-place bilingual: the target is the fitted primary, the source a smaller
    # muted secondary; both live in one box, so one page serves both languages.
    typesetter = TypstFragmentTypesetter()
    try:
        fragment = typesetter.typeset_bilingual(
            "机器依赖注意力。", "The machine relies on attention.", 240.0, 80.0
        )
        assert fragment is not None, "Typst failed to typeset a bilingual fragment"
        text = _text(fragment)
        assert "机器" in text
        assert "machine" in text.lower()
    finally:
        typesetter.close()


def test_a_box_below_the_font_floor_descends(tmp_path: Path) -> None:
    typesetter = TypstFragmentTypesetter()
    try:
        # A box shorter than the 2pt floor (cap = height*0.82) cannot hold text.
        assert typesetter.typeset("hello", 100.0, 2.0) is None
    finally:
        typesetter.close()


def test_the_typst_fragment_typesetter_renders_literal_text(tmp_path: Path) -> None:
    typesetter = TypstFragmentTypesetter()
    try:
        fragment = typesetter.typeset("Hello #world [and] 100% done", 240.0, 24.0)
        assert fragment is not None, "Typst failed to typeset a plain fragment"
        assert fragment.exists()
        width, height = pdf_struct.page_sizes(fragment)[1]
        assert width == pytest.approx(240.0)
        assert height == pytest.approx(24.0)
        # Markup characters survive as literal text, not as commands.
        assert "Hello #world [and] 100% done" in _text(fragment)
    finally:
        typesetter.close()


def test_the_typst_typesetter_measures_wrapped_height(tmp_path: Path) -> None:
    typesetter = TypstFragmentTypesetter()
    try:
        short = typesetter.measure("A short line", 200.0)
        long = typesetter.measure("A short line " * 40, 200.0)
        assert 0.0 < short < long, (short, long)
    finally:
        typesetter.close()


def test_the_typst_typesetter_renders_a_math_fragment(tmp_path: Path) -> None:
    typesetter = TypstFragmentTypesetter()
    try:
        fragment = typesetter.typeset_math("e^{i\\pi} + 1 = 0", 160.0, 24.0)
        assert fragment is not None, "typstify_math failed on a simple formula"
        assert fragment.exists()
    finally:
        typesetter.close()


def test_the_compositor_stamps_a_real_typst_fragment(tmp_path: Path) -> None:
    source = write_text_pdf(tmp_path / "source.pdf", [_PAGE])
    output = tmp_path / "out.pdf"
    typesetter = TypstFragmentTypesetter()
    try:
        composition = LayerCompositor(source, typesetter=typesetter).compose(
            [Overlay("e1", 1, _REGION, "TRANSLATED REGION TEXT")], output
        )
    finally:
        typesetter.close()

    (placement,) = composition.placements
    assert placement.placed_as is Fidelity.RECONSTRUCTED_ADAPTED
    assert pdf_struct.page_sizes(output) == pdf_struct.page_sizes(source)
    assert "TRANSLATED REGION TEXT" in _text(output)
