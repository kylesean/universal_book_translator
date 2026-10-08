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
from ubt.model.span import PhysicalBox
from ubt.render.outputs import (
    LayerCompositor,
    Overlay,
    StyledRun,
    TypstFragmentTypesetter,
    _line_slack,
)

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


def test_an_indented_fragment_typesets_at_its_marker_column(tmp_path: Path) -> None:
    # Two regressions in one: the non-reflowed fit path used to drop the
    # source's first-line indent entirely, and the inline indent prefix used to
    # make a body opening with "(" (a list marker, "(1) Rollout ...") fail to
    # compile -- so the item silently kept its source language.
    typesetter = TypstFragmentTypesetter()
    try:
        fragment = typesetter.typeset(
            "(1) 展开与评估任务以突发方式创建沙盒。",
            300.0,
            30.0,
            kind="text",
            font_size=10.0,
            indent_pt=22.0,
        )
    finally:
        typesetter.close()

    assert fragment is not None
    assert "(1)" in _flat(_text(fragment))
    # The first line starts in the marker column; the wrapped lines would not.
    boxes = artifact_text_boxes(fragment, [1])
    first = max(boxes, key=lambda box: box.bbox.y1)
    assert first.bbox.x0 > 15.0


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
        ("text", f"fragment number {index} with a few words here", 200.0, 12.0, None)
        for index in range(30)
    ]
    typesetter = TypstFragmentTypesetter()
    try:
        typesetter.prefetch(requests)
        for _kind, text, width_pt, height_pt, _font_size in requests:
            fragment = typesetter.typeset(text, width_pt, height_pt)
            assert fragment is not None and fragment.exists()
    finally:
        typesetter.close()


def test_a_styled_run_fragment_is_warmed_so_the_draw_does_not_recompile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression: the prefetch request used to omit runs/align_center/is_bold, so
    # the warmed source differed from the draw-time one and every styled fragment
    # was recompiled, one Typst process at a time, during the draw loop.
    from ubt.render import outputs as outputs_mod

    real_compile = outputs_mod.typst_compile
    calls = {"n": 0}

    def counting(typ_path: str, pdf_path: str, binary: str) -> tuple[bool, str]:
        calls["n"] += 1
        return real_compile(typ_path, pdf_path, binary)

    monkeypatch.setattr(outputs_mod, "typst_compile", counting)
    runs = (StyledRun("attention"),)
    text = "The machine relies on attention and runs a forward pass."
    typesetter = TypstFragmentTypesetter(cache_dir=tmp_path / "fragments")
    try:
        typesetter.prefetch(
            [("text", text, 240.0, 20.0, None, None, False, False, None, False, runs)]
        )
        warmed_calls = calls["n"]
        fragment = typesetter.typeset(text, 240.0, 20.0, runs=runs)
        assert fragment is not None and fragment.exists()
        assert calls["n"] == warmed_calls, "the draw recompiled a warmed fragment"
    finally:
        typesetter.close()


def test_an_in_place_bilingual_fragment_carries_both_languages(tmp_path: Path) -> None:
    # In-place bilingual: the target is the fitted primary, the source a smaller
    # muted secondary; both live in one box, so one page serves both languages.
    typesetter = TypstFragmentTypesetter()
    try:
        fragment = typesetter.typeset_bilingual(
            "La machine repose sur l'attention.", "The machine relies on attention.", 240.0, 80.0
        )
        assert fragment is not None, "Typst failed to typeset a bilingual fragment"
        text = _text(fragment)
        assert "repose" in text.lower()
        assert "machine" in text.lower()
    finally:
        typesetter.close()


def test_a_box_below_the_font_floor_descends(tmp_path: Path) -> None:
    typesetter = TypstFragmentTypesetter()
    try:
        # A box shorter than the readable floor cannot hold text at a
        # reading-grade size, so typeset declines and the element descends.
        assert typesetter.typeset("hello", 100.0, 2.0) is None
        # The old 2pt floor would have drawn this at ~2pt; 6pt is the floor.
        assert typesetter.typeset("hello world this line will not fit", 40.0, 5.0) is None
    finally:
        typesetter.close()


def test_a_bilingual_box_too_short_for_the_echo_keeps_the_target_alone(
    tmp_path: Path,
) -> None:
    # A one-line-tall box cannot stack a readable target *and* its source echo
    # (the pair needs ~1.8x the box). The target is the deliverable and the echo
    # is best-effort, so the source is dropped and the target drawn alone -- a
    # heading comes out monolingual rather than descending to source. This is
    # the bug the 6pt floor surfaced: the old 2pt floor fit the pair by shrinking
    # below readability, and dropping the echo silently broke the delivery ratio.
    typesetter = TypstFragmentTypesetter()
    try:
        fragment = typesetter.typeset_bilingual(
            "La machine repose sur l'attention et la passe avant.",
            "The machine relies on attention.",
            380.0,
            14.2,
        )
        assert fragment is not None, "a target-only fallback should still draw the target"
        text = _text(fragment)
        assert "machine" in text.lower(), "the target must be drawn"
        assert "repose" not in text.lower(), "the source echo must be dropped, not shrunk"
    finally:
        typesetter.close()


def test_a_bilingual_paragraph_across_two_boxes_draws_at_one_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The pair (target + muted echo) needs ~1.8x the height of the target alone,
    # so a per-box re-fit drew one box at ~6pt and its neighbour at ~8pt for the
    # same paragraph. The compositor now sizes the run at the smallest pair fit
    # and hands every box that one size; the real typesetter must honor it.
    source = write_text_pdf(
        tmp_path / "source.pdf",
        [["The machine relies on attention and runs a forward pass.", "It stays accurate."]],
    )
    output = tmp_path / "out.pdf"
    first = (54.0, 690.0, 354.0, 750.0)
    second = (54.0, 620.0, 230.0, 680.0)  # narrower: its pair fits smaller
    target = (
        "The machine relies on attention and runs a forward pass over the batch "
        "of tokens that arrive from the encoder stack, and it stays accurate on "
        "every benchmark we have tested so far."
    )
    echo = (
        "La machine repose sur l'attention et effectue une passe avant sur le "
        "lot de jetons qui arrivent de la pile d'encodeurs, et elle reste "
        "precise sur chaque banc d'essai."
    )
    overlay = Overlay(
        "e1",
        1,
        first,
        target,
        source=echo,
        boxes=(PhysicalBox.of(1, first), PhysicalBox.of(1, second)),
        font_size=11.0,
    )
    typesetter = TypstFragmentTypesetter()
    sizes: list[float | None] = []
    real = typesetter.typeset_bilingual

    def record(
        source: str,
        target: str,
        width_pt: float,
        height_pt: float,
        *,
        size_pt: float | None = None,
    ) -> Path | None:
        sizes.append(size_pt)
        return real(source, target, width_pt, height_pt, size_pt=size_pt)

    monkeypatch.setattr(typesetter, "typeset_bilingual", record)
    try:
        LayerCompositor(source, typesetter=typesetter).compose([overlay], output)
    finally:
        typesetter.close()

    # Every pair-drawing box got the run's one size, below the target-alone cap.
    assert sizes, "no bilingual fragment was drawn"
    drawn = set(sizes)
    assert None not in drawn, "a box re-fit instead of taking the run size"
    assert len(drawn) == 1, f"one paragraph drew at two sizes: {sizes}"
    (run_size,) = drawn
    assert run_size is not None and 6.0 <= run_size < 11.0 * 1.05


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


def test_each_box_draws_at_the_size_that_fits_it() -> None:
    # A box the target does not overflow keeps its source size; only a cramped one
    # shrinks. (A document-wide shared size was tried and rejected: it dragged the
    # boxes that did fit down to the low percentile of the cramped ones, so a page
    # that failed to reflow read as uniformly shrunken next to its neighbours.)
    typesetter = TypstFragmentTypesetter()
    try:
        cap = 12.0 * 1.05
        roomy = typesetter._fit_size("short caption", 120.0, 40.0, max_size_pt=cap)
        cramped = typesetter._fit_size(
            "a much longer paragraph that has to wrap many times " * 4,
            120.0,
            40.0,
            max_size_pt=cap,
        )
        assert roomy == pytest.approx(cap), roomy
        assert cramped is not None and cramped < cap
    finally:
        typesetter.close()


def test_the_line_slack_lets_a_box_hold_the_source_size() -> None:
    # The extracted box is the glyph ink, which is shorter than a drawn line box,
    # so a fragment at the source size needs the line leading below it. Without
    # the slack the fit would shrink it well below the source size.
    typesetter = TypstFragmentTypesetter()
    try:
        text = "一段中文正文，用来测量行框余量"
        ink_height = 9.5  # a single extracted line at ~10.9pt
        without = typesetter._fit_size(text, 200.0, ink_height)
        with_slack = typesetter._fit_size(text, 200.0, ink_height + _line_slack(10.9))
        assert without is not None and with_slack is not None
        assert with_slack > without, (without, with_slack)
        assert with_slack > 0.9 * 10.9, with_slack
    finally:
        typesetter.close()


def test_the_typst_typesetter_renders_inline_math_not_literal_latex(tmp_path: Path) -> None:
    # Regression: the unified typesetter escaped the whole body, so an inline
    # ``$...$`` span the model emitted was printed verbatim as ``\Gamma``. The
    # body now goes through the overlay renderer, which emits strings the probe
    # compiles in math mode; the control sequences must not survive literally.
    typesetter = TypstFragmentTypesetter()
    try:
        fragment = typesetter.typeset(r"类型判断 $\Gamma \vdash t : T$ 表示上下文", 320.0, 40.0)
        assert fragment is not None, "Typst failed to typeset inline math"
        text = _text(fragment)
        assert "\\Gamma" not in text and "\\vdash" not in text, text
        assert "Γ" in text, text
    finally:
        typesetter.close()


def test_the_compositor_draws_an_inline_math_target_onto_the_region(tmp_path: Path) -> None:
    source = write_text_pdf(tmp_path / "source.pdf", [_PAGE])
    output = tmp_path / "out.pdf"
    typesetter = TypstFragmentTypesetter()
    try:
        composition = LayerCompositor(source, typesetter=typesetter).compose(
            [Overlay("e1", 1, _REGION, r"类型判断 $\Gamma \vdash t : T$ 表示上下文")], output
        )
    finally:
        typesetter.close()

    (placement,) = composition.placements
    assert placement.placed_as is Fidelity.RECONSTRUCTED_ADAPTED
    text = _text(output)
    assert "\\Gamma" not in text, text
    assert "Γ" in text, text


def test_a_toc_overlay_renders_title_leaders_and_page_number(tmp_path: Path) -> None:
    # A translated TOC row must carry its dot leaders and page number: the reader
    # drops the source's leader/number lines, so a plain text overlay would sit
    # in a title-width slot with no leader. The TOC kind redraws the whole row.
    source = write_text_pdf(tmp_path / "source.pdf", [_PAGE])
    output = tmp_path / "out.pdf"
    typesetter = TypstFragmentTypesetter()
    try:
        LayerCompositor(source, typesetter=typesetter).compose(
            [
                Overlay(
                    "toc1",
                    1,
                    (54.0, 640.0, 500.0, 654.0),
                    "3. 可逆效应与反应式余效应",
                    kind="toc",
                    toc_page="9",
                )
            ],
            output,
        )
    finally:
        typesetter.close()

    text = _text(output)
    assert "可逆效应与反应式余效应" in text, text
    assert "9" in text, text
    # The dot leaders were regenerated, not lost.
    assert "." in text, text
