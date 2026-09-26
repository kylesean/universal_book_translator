"""Unit tests for 2D stream stripping via pikepdf (Zero-AGPL)."""

from collections.abc import Sequence
from typing import Any, cast

import pikepdf

from tests.corpus_markers import requires_synthetic_mono
from ubt.adapters.pdf.stream_strip import (
    IDENTITY_MATRIX,
    FontAdvance,
    Rect2DIndex,
    _advance_from_widths,
    _font_advance,
    _show_glyph_bytes,
    mul_matrix,
    shared_form_objgens,
    strip_page_text_pikepdf,
    transform_point,
)


def test_matrix_operations() -> None:
    # Identity transform
    pt = transform_point(10.0, 20.0, IDENTITY_MATRIX)
    assert pt == (10.0, 20.0)

    # Translation (tx=5, ty=10)
    m_trans = (1.0, 0.0, 0.0, 1.0, 5.0, 10.0)
    assert transform_point(10.0, 20.0, m_trans) == (15.0, 30.0)

    # Multiplication: Trans1 x Trans2
    m_trans2 = (1.0, 0.0, 0.0, 1.0, 2.0, 3.0)
    combined = mul_matrix(m_trans, m_trans2)
    assert transform_point(0.0, 0.0, combined) == (7.0, 13.0)


def test_rect_2d_index() -> None:
    rects = [(50.0, 100.0, 150.0, 200.0), (250.0, 100.0, 350.0, 200.0)]
    idx = Rect2DIndex.build(rects)

    # Left column hit
    assert idx.contains_point(100.0, 150.0) is True
    # Column gutter (empty space between columns)
    assert idx.contains_point(200.0, 150.0) is False
    # Right column hit
    assert idx.contains_point(300.0, 150.0) is True

    # 2D text rect matching
    text_rect_left = (60.0, 110.0, 140.0, 130.0)
    assert idx.matches_text_for_removal(60.0, 110.0, text_rect_left) is True

    text_rect_middle = (180.0, 110.0, 220.0, 130.0)
    assert idx.matches_text_for_removal(180.0, 110.0, text_rect_middle) is False


def test_two_column_isolation() -> None:
    """Verify that 2D stripping strips left column text without affecting right column at the same y!"""
    pdf = pikepdf.new()
    page = pdf.add_blank_page(page_size=(600, 800))
    # Two text lines at the exact same Y baseline (y=500):
    # Left column: x=50..200, y=500
    # Right column: x=350..500, y=500
    stream_content = (
        b"BT /F1 12 Tf 50 500 Td (Left Column Text) Tj ET\n"
        b"BT /F1 12 Tf 350 500 Td (Right Column Text) Tj ET\n"
    )
    page.Contents = pdf.make_stream(stream_content)

    # Strip ONLY the left column: x in [40, 250], y in [490, 520]
    strip_rects = [(40.0, 490.0, 250.0, 520.0)]
    stats = strip_page_text_pikepdf(page, strip_rects, page_no=1)

    assert stats.aborted is None
    assert stats.dropped_ops == 1
    assert stats.kept_ops == 1

    remaining_bytes = page.Contents.read_bytes()
    assert b"Left Column Text" not in remaining_bytes
    assert b"Right Column Text" in remaining_bytes


def test_protected_formula_preservation() -> None:
    """Verify that text inside a protected formula region is never deleted."""
    pdf = pikepdf.new()
    page = pdf.add_blank_page(page_size=(600, 800))
    # Two lines in the same horizontal region:
    # 1. Prose line: x=50, y=500
    # 2. Formula symbol/math text: x=50, y=450
    stream_content = (
        b"BT /F1 12 Tf 50 500 Td (Prose to remove) Tj ET\nBT /F1 12 Tf 50 450 Td (E = mc^2) Tj ET\n"
    )
    page.Contents = pdf.make_stream(stream_content)

    # Suppose strip_rect covers both lines: y from 440 to 520
    strip_rects = [(40.0, 440.0, 300.0, 520.0)]
    # But protected_rects guards the formula at y from 440 to 465
    protected_rects = [(40.0, 440.0, 300.0, 465.0)]

    stats = strip_page_text_pikepdf(page, strip_rects, protected_rects=protected_rects, page_no=1)

    assert stats.aborted is None
    assert stats.dropped_ops == 1  # only prose dropped
    assert stats.kept_ops == 1

    remaining_bytes = page.Contents.read_bytes()
    assert b"Prose to remove" not in remaining_bytes
    assert b"E = mc^2" in remaining_bytes


def test_form_xobject_recursive_stripping() -> None:
    """Verify that text inside a Form XObject is recursively stripped."""
    pdf = pikepdf.new()
    page = pdf.add_blank_page(page_size=(600, 800))

    # Create Form XObject
    form = pdf.make_stream(b"BT /F1 12 Tf 100 500 Td (Inside Form XObject) Tj ET\n")
    form[pikepdf.Name("/Type")] = pikepdf.Name("/XObject")
    form[pikepdf.Name("/Subtype")] = pikepdf.Name("/Form")
    form[pikepdf.Name("/BBox")] = pikepdf.Array([0, 0, 600, 800])
    form[pikepdf.Name("/Matrix")] = pikepdf.Array([1, 0, 0, 1, 0, 0])

    page.Resources = pikepdf.Dictionary(
        {
            "/XObject": pikepdf.Dictionary(
                {
                    "/Fm1": form,
                }
            )
        }
    )
    page.Contents = pdf.make_stream(b"/Fm1 Do\n")

    strip_rects = [(80.0, 480.0, 300.0, 530.0)]
    stats = strip_page_text_pikepdf(page, strip_rects, page_no=1, recurse_forms=True)

    assert stats.aborted is None
    assert stats.dropped_ops == 1
    assert stats.forms_changed == 1

    form_bytes = form.read_bytes()
    assert b"Inside Form XObject" not in form_bytes


def _make_form(pdf: pikepdf.Pdf, content: bytes) -> Any:
    form = pdf.make_stream(content)
    form[pikepdf.Name("/Type")] = pikepdf.Name("/XObject")
    form[pikepdf.Name("/Subtype")] = pikepdf.Name("/Form")
    form[pikepdf.Name("/BBox")] = pikepdf.Array([0, 0, 600, 800])
    form[pikepdf.Name("/Matrix")] = pikepdf.Array([1, 0, 0, 1, 0, 0])
    return form


def test_shared_form_xobject_is_not_rewritten() -> None:
    """A Form drawn by two pages must not be stripped for one page's rect.

    Rewriting the shared indirect object used to erase the source text on both
    pages; the guard keeps it everywhere (fail-closed) instead.
    """
    pdf = pikepdf.new()
    p1 = pdf.add_blank_page(page_size=(600, 800))
    p2 = pdf.add_blank_page(page_size=(600, 800))
    form = _make_form(pdf, b"BT /F1 12 Tf 100 500 Td (Shared Form Text) Tj ET\n")
    for page in (p1, p2):
        page.Resources = pikepdf.Dictionary({"/XObject": pikepdf.Dictionary({"/Fm1": form})})
        page.Contents = pdf.make_stream(b"/Fm1 Do\n")

    shared = shared_form_objgens(pdf)
    assert form.objgen in shared

    stats = strip_page_text_pikepdf(
        p1, [(80.0, 480.0, 300.0, 530.0)], page_no=1, shared_forms=shared
    )
    assert stats.aborted is None
    assert stats.forms_changed == 0
    assert stats.shared_forms_skipped == 1
    assert b"Shared Form Text" in form.read_bytes()


def test_page_private_form_is_still_stripped() -> None:
    """A form only one page draws is not 'shared' and is stripped as before."""
    pdf = pikepdf.new()
    page = pdf.add_blank_page(page_size=(600, 800))
    form = _make_form(pdf, b"BT /F1 12 Tf 100 500 Td (Private Form) Tj ET\n")
    page.Resources = pikepdf.Dictionary({"/XObject": pikepdf.Dictionary({"/Fm1": form})})
    page.Contents = pdf.make_stream(b"/Fm1 Do\n")

    shared = shared_form_objgens(pdf)
    assert shared == set()

    stats = strip_page_text_pikepdf(
        page, [(80.0, 480.0, 300.0, 530.0)], page_no=1, shared_forms=shared
    )
    assert stats.forms_changed == 1
    assert stats.shared_forms_skipped == 0
    assert b"Private Form" not in form.read_bytes()


def test_form_drawn_twice_on_one_page_is_not_shared() -> None:
    """Two draws on the *same* page are page-local; stripping one is safe."""
    pdf = pikepdf.new()
    page = pdf.add_blank_page(page_size=(600, 800))
    form = _make_form(pdf, b"BT /F1 12 Tf 100 500 Td (Local Twice) Tj ET\n")
    page.Resources = pikepdf.Dictionary(
        {"/XObject": pikepdf.Dictionary({"/Fm1": form, "/Fm2": form})}
    )
    page.Contents = pdf.make_stream(b"/Fm1 Do\n/Fm2 Do\n")

    assert shared_form_objgens(pdf) == set()


def test_cm_transformed_text_stripping() -> None:
    """Verify that cm transforms (e.g. translation, scale) are correctly tracked."""
    pdf = pikepdf.new()
    page = pdf.add_blank_page(page_size=(600, 800))
    # Outer cm translates by dx=100, dy=200
    # Inside: text is at local 50, 50 -> user space (150, 250)
    stream_content = (
        b"q\n"
        b"1 0 0 1 100 200 cm\n"
        b"BT /F1 12 Tf 50 50 Td (Transformed Text) Tj ET\n"
        b"Q\n"
        b"BT /F1 12 Tf 50 50 Td (Untransformed Text) Tj ET\n"
    )
    page.Contents = pdf.make_stream(stream_content)

    # Strip rect targets user space (140..250, 240..270) -> only Transformed Text
    strip_rects = [(140.0, 240.0, 250.0, 270.0)]
    stats = strip_page_text_pikepdf(page, strip_rects, page_no=1)

    assert stats.aborted is None
    assert stats.dropped_ops == 1
    assert stats.kept_ops == 1

    remaining_bytes = page.Contents.read_bytes()
    assert b"Transformed Text" not in remaining_bytes
    assert b"Untransformed Text" in remaining_bytes


def test_tj_array_with_kerning() -> None:
    """Verify that TJ array text shows are correctly measured and stripped."""
    pdf = pikepdf.new()
    page = pdf.add_blank_page(page_size=(600, 800))
    stream_content = b"BT /F1 12 Tf 100 500 Td [(Kerned) -20 (Text)] TJ ET\n"
    page.Contents = pdf.make_stream(stream_content)

    strip_rects = [(90.0, 490.0, 250.0, 520.0)]
    stats = strip_page_text_pikepdf(page, strip_rects, page_no=1)

    assert stats.aborted is None
    assert stats.dropped_ops == 1
    remaining_bytes = page.Contents.read_bytes()
    assert b"Kerned" not in remaining_bytes


def test_font_advance_simple_and_guards() -> None:
    pdf = pikepdf.new()
    simple = pdf.make_indirect(
        pikepdf.Dictionary(
            Type="/Font",
            Subtype="/TrueType",
            FirstChar=65,  # 'A'
            Widths=pikepdf.Array([800, 200, 200]),  # A=800, B=200, C=200
        )
    )
    fa = _font_advance(simple)
    assert fa is not None and fa.code_bytes == 1
    assert fa.widths[65] == 800.0 and fa.widths[66] == 200.0
    # No /Widths → None → caller keeps the estimate.
    bare = pdf.make_indirect(pikepdf.Dictionary(Type="/Font", Subtype="/Type1"))
    assert _font_advance(bare) is None


def test_font_advance_type0_cid() -> None:
    """Type0/CIDFont widths come from the descendant /W array with /DW default,
    and Identity-H codes are two bytes."""
    pdf = pikepdf.new()
    cid_font = pdf.make_indirect(
        pikepdf.Dictionary(
            Type="/Font",
            Subtype="/CIDFontType0",
            DW=1000,
            W=pikepdf.Array([1, [500, 600], [10, 20, 700]]),  # cids 1,2; cids 10..20
        )
    )
    type0 = pdf.make_indirect(
        pikepdf.Dictionary(
            Type="/Font",
            Subtype="/Type0",
            Encoding="/Identity-H",
            DescendantFonts=pikepdf.Array([cid_font]),
        )
    )
    fa = _font_advance(type0)
    assert fa is not None and fa.code_bytes == 2 and fa.default == 1000.0
    assert fa.widths[1] == 500.0 and fa.widths[2] == 600.0 and fa.widths[15] == 700.0
    # Identity-H: two bytes per code, big-endian. cid 1 = b"\x00\x01".
    assert _advance_from_widths(b"\x00\x01\x00\x02", fa) == (500 + 600) / 1000.0
    # Unknown cid falls back to /DW.
    assert _advance_from_widths(b"\x00\xff", fa) == 1.0


def test_type0_non_identity_cmap_is_two_bytes() -> None:
    """Named 2-byte CJK CMaps (UniGB-UCS2-H, …) must not be measured as one
    byte per glyph; only a one-byte simple-font encoding is one byte."""
    pdf = pikepdf.new()
    cid_font = pdf.make_indirect(
        pikepdf.Dictionary(
            Type="/Font",
            Subtype="/CIDFontType0",
            DW=1000,
            W=pikepdf.Array([1, [500, 600]]),
        )
    )
    for cmap in ("/Identity-H", "/Identity-V", "/UniGB-UCS2-H", "/UniJIS-UCS2-H"):
        type0 = pdf.make_indirect(
            pikepdf.Dictionary(
                Type="/Font",
                Subtype="/Type0",
                Encoding=cmap,
                DescendantFonts=pikepdf.Array([cid_font]),
            )
        )
        fa = _font_advance(type0)
        assert fa is not None and fa.code_bytes == 2, cmap
        # Two bytes per code: cid 1 = b"\x00\x01", cid 2 = b"\x00\x02".
        assert _advance_from_widths(b"\x00\x01\x00\x02", fa) == (500 + 600) / 1000.0
    # A plain simple font stays one byte per code.
    simple = pdf.make_indirect(
        pikepdf.Dictionary(
            Type="/Font",
            Subtype="/TrueType",
            Encoding="/WinAnsiEncoding",
            FirstChar=65,
            Widths=pikepdf.Array([500]),
        )
    )
    assert _font_advance(simple).code_bytes == 1  # type: ignore[union-attr]


def test_advance_from_widths_and_show_bytes() -> None:
    fa = FontAdvance(code_bytes=1, widths={65: 500.0, 66: 250.0}, default=300.0)
    # "AAB" = 500+500+250 = 1250 → 1.25 em.
    assert _advance_from_widths(b"AAB", fa) == 1.25
    # Unknown code 90 ('Z') falls back to the default 300.
    assert _advance_from_widths(b"Z", fa) == 0.30
    # TJ array: only the string parts are glyphs; the number is kerning.
    assert _show_glyph_bytes([pikepdf.Array([b"Ta", -30, b"ble"])]) == b"Table"


def test_accurate_metrics_reposition_trailing_run() -> None:
    """A trailing run positioned by real /Widths lands inside the strip rect,
    where the fixed 0.5em/glyph estimate would have placed it outside — the
    colored-citation ghost (arXiv 2609.20519). 'A' is 1000/1000 em (1em), so
    eight A's at 10pt advance 80pt; the next run starts at x=180, inside the
    170..200 strip rect. The old 0.5em estimate would have put it at x=140,
    outside, and left it as a ghost."""
    pdf = pikepdf.new()
    page = pdf.add_blank_page(page_size=(600, 800))
    font = pdf.make_indirect(
        pikepdf.Dictionary(
            Type="/Font", Subtype="/TrueType", FirstChar=65, Widths=pikepdf.Array([1000])
        )
    )
    page.Resources = pikepdf.Dictionary(Font=pikepdf.Dictionary(F1=font))
    page.Contents = pdf.make_stream(b"BT /F1 10 Tf 100 500 Td (AAAAAAAA) Tj (X) Tj ET\n")

    stats = strip_page_text_pikepdf(page, [(170.0, 495.0, 200.0, 512.0)], page_no=1)
    assert stats.aborted is None
    remaining = page.Contents.read_bytes()
    assert b"(X)" not in remaining  # trailing run erased by accurate metrics


def test_q_restores_outer_font_advance() -> None:
    """q/Q must restore the text state's font, not only the CTM.

    F1 is a narrow font (0.1em/glyph); F2 is 1em/glyph. The outer run after Q
    relies on F1's widths. If Q leaves ``font_adv`` pointing at F2, the run is
    measured ten times too wide and the strip rect at x=110..150 erases it.
    """
    pdf = pikepdf.new()
    page = pdf.add_blank_page(page_size=(600, 800))
    f1 = pdf.make_indirect(
        pikepdf.Dictionary(
            Type="/Font", Subtype="/TrueType", FirstChar=65, Widths=pikepdf.Array([100])
        )
    )
    f2 = pdf.make_indirect(
        pikepdf.Dictionary(
            Type="/Font", Subtype="/TrueType", FirstChar=65, Widths=pikepdf.Array([1000])
        )
    )
    page.Resources = pikepdf.Dictionary(Font=pikepdf.Dictionary(F1=f1, F2=f2))
    page.Contents = pdf.make_stream(
        b"BT /F1 10 Tf 100 500 Td (AAAA) Tj ET\n"
        b"q\nBT /F2 10 Tf 100 400 Td (AAAA) Tj ET\nQ\n"
        b"BT 100 300 Td (AAAA) Tj ET\n"
    )

    stats = strip_page_text_pikepdf(page, [(110.0, 295.0, 150.0, 315.0)], page_no=1)
    assert stats.aborted is None
    assert stats.dropped_ops == 0  # outer-font run is narrow, outside the rect
    assert page.Contents.read_bytes().count(b"AAAA") == 3


def test_q_restores_outer_line_width() -> None:
    """q/Q must restore the line width — it is graphics state (PDF 32000-1 §8.4.1).

    A thick border drawn inside a ``q`` left ``line_width`` at 8 after ``Q``, so
    a later thin stroked underline measured thickness = bbox_h + 8, blew past
    the decoration cap and survived the strip — leaving the source underline
    floating under the translated overlay.
    """
    pdf = pikepdf.new()
    page = pdf.add_blank_page(page_size=(600, 800))
    stream = (
        b"q 8 w 40 200 m 200 200 l S Q\n"  # thick border inside a q, then Q
        b"BT /F1 12 Tf 50 500 Td (caption with underlined word) Tj ET\n"
        b"100 496.5 m 180 496.5 l S\n"  # thin underline, no explicit w
    )
    page.Contents = pdf.make_stream(stream)

    stats = strip_page_text_pikepdf(page, [(60.0, 490.0, 300.0, 520.0)], page_no=1)
    assert stats.aborted is None
    assert stats.dropped_paths == 1  # the underline, not the far-away border

    ops = [
        str(op)
        for _, op in cast("list[tuple[Sequence[Any], Any]]", pikepdf.parse_content_stream(page))
    ]
    assert ops.count("S") == 1  # only the thick border survives
    assert b"caption with underlined word" not in page.Contents.read_bytes()


class _BoomPage:
    def __contains__(self, item: object) -> bool:
        return True

    def contents_coalesce(self) -> None:
        raise RuntimeError("coalesce exploded")

    def get(self, item: object) -> None:
        return None


def test_strip_abort_reason_is_actionable() -> None:
    """Aborts must be surfaced with a clear, actionable reason, not swallowed."""
    stats = strip_page_text_pikepdf(
        _BoomPage(),  # type: ignore[arg-type]
        [(0.0, 0.0, 10.0, 10.0)],
        page_no=7,
    )
    assert stats.aborted is not None
    assert stats.aborted.startswith("error:RuntimeError:")
    assert "coalesce exploded" in stats.aborted


@requires_synthetic_mono
def test_strip_real_chapter_page_pikepdf() -> None:
    """Verify 2D stream stripping on real book chapter page."""
    # tests/fixtures/synthetic-mono.pdf is the live 13-page sample (generated, gitignored).
    pdf = pikepdf.open("tests/fixtures/synthetic-mono.pdf")
    page = pdf.pages[1]
    stats = strip_page_text_pikepdf(page, [(50.0, 300.0, 500.0, 600.0)], page_no=2)
    assert stats.aborted is None
    assert stats.dropped_ops > 0


def test_underline_decoration_rule_dropped_with_text() -> None:
    """Regression (arXiv 2609.20519 p9): the source underline of "…are
    underlined." is a vector rule, not a text operator, so the text strip
    left it floating alone under the re-typeset caption. A thin horizontal
    path FULLY covered by the erase rect must go with the text; geometry
    that is not covered, not thin, or not painted-as-decoration must stay."""
    pdf = pikepdf.new()
    page = pdf.add_blank_page(page_size=(600, 800))
    stream = (
        b"BT /F1 12 Tf 50 500 Td (caption with underlined word) Tj ET\n"
        b"1 0 0 rg 100 496.5 80 0.4 re f\n"  # underline: covered + thin -> drop
        b"0 0 0 rg 100 300 200 12 re f\n"  # far below the rect -> keep
        b"1 w 40 505 m 200 505 l S\n"  # crosses the rect's left border -> keep
        b"0.5 w 150 490 m 150 500 l S\n"  # vertical hairline -> keep
    )
    page.Contents = pdf.make_stream(stream)

    stats = strip_page_text_pikepdf(page, [(60.0, 490.0, 300.0, 520.0)], page_no=1)
    assert stats.aborted is None
    assert stats.dropped_paths == 1

    ops = [
        str(op)
        for _, op in cast("list[tuple[Sequence[Any], Any]]", pikepdf.parse_content_stream(page))
    ]
    assert ops.count("f") == 1  # only the uncovered thick rule survives
    assert ops.count("S") == 2  # border-crossing stroke + vertical survive
    assert b"caption with underlined word" not in page.Contents.read_bytes()


def test_decoration_inside_protected_rect_survives() -> None:
    """A covered thin rule that also intersects a protected region (formula,
    table, figure guard) must NOT be dropped — protection always wins."""
    pdf = pikepdf.new()
    page = pdf.add_blank_page(page_size=(600, 800))
    stream = b"100 496.5 80 0.4 re f\n"
    page.Contents = pdf.make_stream(stream)

    stats = strip_page_text_pikepdf(
        page,
        [(60.0, 490.0, 300.0, 520.0)],
        protected_rects=[(90.0, 480.0, 300.0, 510.0)],
        page_no=1,
    )
    assert stats.aborted is None
    assert stats.dropped_paths == 0
    ops = [
        str(op)
        for _, op in cast("list[tuple[Sequence[Any], Any]]", pikepdf.parse_content_stream(page))
    ]
    assert ops.count("f") == 1


def test_clip_path_inside_strip_rect_is_never_dropped() -> None:
    """W/n ends a path without painting; dropping it would silently change
    every later glyph's clip region. Covered + thin must still keep."""
    pdf = pikepdf.new()
    page = pdf.add_blank_page(page_size=(600, 800))
    stream = b"100 496.5 80 0.4 re W n\n100 496.5 80 0.4 re f\n"
    page.Contents = pdf.make_stream(stream)

    stats = strip_page_text_pikepdf(page, [(60.0, 490.0, 300.0, 520.0)], page_no=1)
    assert stats.aborted is None
    assert stats.dropped_paths == 1  # only the painted copy; the clip stays
    ops = [
        str(op)
        for _, op in cast("list[tuple[Sequence[Any], Any]]", pikepdf.parse_content_stream(page))
    ]
    assert "W" in ops and "n" in ops
    assert ops.count("f") == 0  # the painted decoration was dropped
