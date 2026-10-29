"""Unit tests for the 1:1 facsimile delivery channel (Track A).

Tests:
1. strip_pdf_text: text-line granularity, keep-wins, blanking (not deletion),
   fail-open on a malformed stream, and the page-scoped rect filter.
2. flow: solve_flow, punctuation preference, CJK 避头尾 rules.
3. layout_solver: solve_page_2d binary search, dynamic leading stretch, overlap
   resolution.
4. facsimile_renderer: coordinate transformation, markup generation, and full
   compilation.

The stripping tests pin the properties that were *broken* in the first
implementation, not the behaviour it happened to have:

* **Text-line granularity.** A ``BT…ET`` block is not a layout unit — measured on
  the corpus, one block carries up to 313 text lines. A stripper that decides
  per block collapses the page to all-or-nothing, which took every formula,
  folio and figure label with it.
* **Keep wins over erase.** A translated block's box and a protected formula
  box can overlap; the protection must survive.
* **Blanking preserves positions.** Deleting a show operator also deletes its
  text-matrix advance, so every following run on the page shifts.
* **Fail open.** A stream that cannot be parsed keeps its original bytes; the
  old ``return b""`` blanked the page's vector art as well as its text.
* **Absent page means keep.** A page with no entry in the erase map keeps all
  its text; the old default was "strip everything".
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pikepdf
import pytest

from ubt.adapters.pdf.docling_render import _block_font_size, _facsimile_strip_plan
from ubt.adapters.pdf.strip_pdf_text import (
    BoxRect,
    strip_form_xobjects_recursive,
    strip_stream_text,
    strip_text_streams,
)
from ubt.core.ir.models import BlockType, BoundingBox
from ubt.model.span import BBox, PhysicalBox
from ubt.render.flow import (
    approximate_measure,
    solve_flow,
)
from ubt.render.layout_solver import (
    FLOOR_PT,
    check_source_overlap,
    resolve_local_overlaps,
    solve_page_2d,
)
from ubt.render.typst.facsimile_renderer import (
    FacsimileBlockPlacement,
    FacsimilePageSpec,
    FacsimileRenderer,
)

pytestmark = pytest.mark.fast


def _instructions_in(page: pikepdf.Page) -> list[tuple[Any, Any]]:
    """The ``(operands, operator)`` pairs of a page's content stream.

    ``parse_content_stream`` is typed as an iterator over
    ``ContentStreamInstruction | ContentStreamInlineImage``. Only the
    instruction carries operands; an inline image is a bare object and has no
    tuple to unpack, so the two cases are separated here rather than at each
    use. (``_ObjectList`` is not a ``list`` subclass, which is why the operand
    test is duck-typed instead of ``isinstance``.)
    """
    pairs: list[tuple[Any, Any]] = []
    for item in pikepdf.parse_content_stream(page):
        operands = getattr(item, "operands", None)
        operator = getattr(item, "operator", None)
        if operands is None or operator is None:
            continue  # inline image
        if not hasattr(operands, "__iter__"):
            continue
        pairs.append((operands, operator))
    return pairs


def _texts_in(stream_bytes: bytes) -> list[str]:
    """Literal glyph runs left in a content stream (empty runs dropped).

    ``parse_content_stream`` yields a union: an instruction whose operands are
    an ``_ObjectList``, or an inline image whose operands are not iterable. The
    image case carries no text, so it is skipped explicitly. The check is for
    ``list`` *and* the list-like object pikepdf uses -- ``_ObjectList`` is not a
    ``list`` subclass, and testing only for ``list`` silently dropped every
    operator.
    """
    pdf = pikepdf.Pdf.new()
    page = pdf.add_blank_page(page_size=(300, 300))
    page.Contents = pdf.make_stream(stream_bytes)
    out: list[str] = []
    for operands, operator in _instructions_in(page):
        if str(operator) not in ("Tj", "TJ", "'", '"'):
            continue
        for operand in operands:
            if isinstance(operand, pikepdf.String):
                out.append(str(operand))
            elif isinstance(operand, pikepdf.Array):
                out.extend(str(item) for item in operand if isinstance(item, pikepdf.String))
    return [text for text in out if text]


def _operators_in(stream_bytes: bytes) -> list[str]:
    pdf = pikepdf.Pdf.new()
    page = pdf.add_blank_page(page_size=(300, 300))
    page.Contents = pdf.make_stream(stream_bytes)
    return [str(operator) for _operands, operator in _instructions_in(page)]


def _source_pdf(tmp_path: Path) -> Path:
    """A real one-page PDF at the geometry the strip plans are measured on.

    ``_facsimile_strip_plan`` opens the source file to detect vector-diagram
    regions, so it cannot take ``None``: a page with no diagram simply yields
    no extra keep rects, which is what these cases want.
    """
    path = tmp_path / "source_666.pdf"
    pdf = pikepdf.Pdf.new()
    pdf.add_blank_page(page_size=(540, 666))
    pdf.save(path)
    return path


def _fake_block(
    block_id: str,
    *,
    page: int,
    bbox: tuple[float, float, float, float],
    block_type: BlockType = BlockType.NARRATIVE,
    skip_translate: bool = False,
) -> SimpleNamespace:
    """The subset of an ``IRBlock`` that :func:`_facsimile_strip_plan` reads.

    A real ``IRBlock`` would need an element, a style and provenance; the plan
    touches none of them, and building them here would obscure what the test is
    actually pinning.
    """
    return SimpleNamespace(
        id=block_id,
        block_type=block_type,
        bbox=BoundingBox(page=page, x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3]),
        skip_translate=skip_translate,
        element=None,
        style=None,
        provenance=None,
        source_text="",
    )


# --------------------------------------------------------------------------- #
# strip_pdf_text
# --------------------------------------------------------------------------- #


def test_strip_stream_text_no_rects_keeps_everything() -> None:
    """The fail-safe default: no erase rects means *strip nothing*.

    The previous default stripped every block, and the renderer called it with
    no rects at all — which is how a whole book lost its formulas and folios.
    """
    pdf = pikepdf.Pdf.new()
    page = pdf.add_blank_page(page_size=(300, 300))
    page.Contents = pdf.make_stream(
        b"10 20 m BT /F1 12 Tf 1 0 0 1 50 200 Tm (Hello World) Tj ET 30 40 l S"
    )

    stripped = strip_stream_text(page)
    assert _texts_in(stripped) == ["Hello World"]
    assert b"10 20 m" in stripped
    assert b"30 40 l" in stripped


def test_strip_stream_text_erases_only_lines_in_the_box() -> None:
    """Two text lines, one erase rect: only the enclosed line is blanked."""
    pdf = pikepdf.Pdf.new()
    page = pdf.add_blank_page(page_size=(300, 300))
    page.Contents = pdf.make_stream(
        b"10 10 m "
        b"BT 1 0 0 1 50 50 Tm (Keep This Math Formula) Tj ET "
        b"BT 1 0 0 1 200 200 Tm (Strip This Prose) Tj ET "
        b"S"
    )

    erase = [BoxRect(page_index=1, x0=190.0, y0=190.0, x1=290.0, y1=210.0)]
    stripped = strip_stream_text(page, erase_rects=erase)
    assert _texts_in(stripped) == ["Keep This Math Formula"]
    assert b"10 10 m" in stripped


def test_pen_left_of_erase_box_still_erases() -> None:
    """A run whose pen sits just left of its erase box must still be erased.

    A glyph's ink starts a fraction of an em to the *right* of the pen that
    paints it (the left side bearing), but block boxes bound ink. So a run can
    have its pen outside the box while all of its ink is inside it, and a
    pen-position test alone lets it escape — the source text then survives
    underneath its translation, which is the worst failure the background layer
    can produce: the reader sees both languages superimposed.

    Measured: ``2606.23344v1.pdf`` p1's 15.9pt title has its pen at ``x = 92.2``
    against an erase box at ``x0 = 92.9``. It leaked on all three arXiv papers
    in the corpus until erase boxes were grown leftward by the bearing.
    """
    pdf = pikepdf.Pdf.new()
    page = pdf.add_blank_page(page_size=(300, 300))
    # Pen at x = 60; the box that owns this line starts at 60.6, so the pen is
    # 0.6pt outside it — exactly the measured shortfall.
    page.Contents = pdf.make_stream(b"BT /F1 16 Tf 1 0 0 1 60 100 Tm (Leaked Title) Tj ET")

    erase = [BoxRect(page_index=1, x0=60.6, y0=95.0, x1=250.0, y1=115.0)]
    stripped = strip_stream_text(page, erase_rects=erase)
    assert _texts_in(stripped) == []


def test_pen_pad_does_not_erase_text_left_of_the_box() -> None:
    """The leftward pad is a bearing, not a licence to erase the margin.

    A run that ends well before the box begins is a different block — the
    neighbouring column, a marginal note, a folio. Padding must stay small
    enough that it does not reach it, or erasing one block would silently blank
    the one beside it.
    """
    pdf = pikepdf.Pdf.new()
    page = pdf.add_blank_page(page_size=(300, 300))
    page.Contents = pdf.make_stream(b"BT /F1 9 Tf 1 0 0 1 40 100 Tm (Neighbour) Tj ET")

    # Pen at x = 40, box starts at 60.6: a 20pt gap, far wider than any bearing.
    erase = [BoxRect(page_index=1, x0=60.6, y0=95.0, x1=250.0, y1=115.0)]
    stripped = strip_stream_text(page, erase_rects=erase)
    assert _texts_in(stripped) == ["Neighbour"]


def test_pen_pad_covers_the_measured_toc_entry_bearing() -> None:
    """A body-size line's bearing is wider in ems than a display title's.

    The coefficient was first set from one 15.9pt title, whose 0.6pt shortfall
    is 0.038 em — comfortably inside 0.05 em. That coefficient then failed on
    ``2608.25512v1.pdf`` p2, where a 11pt table-of-contents entry's pen sits
    0.66pt left of its erase box: 0.05 · 11 = 0.55pt, a 0.11pt shortfall that
    left all 15 entry numbers on the page underneath their translations.

    So the pad must be sized for the *widest bearing in ems*, not the widest in
    points. Pinned at the measured value rather than at the coefficient, so a
    later "simplification" back to a smaller pad fails here instead of on a
    92-page document.
    """
    from ubt.adapters.pdf.strip_pdf_text import _keep_point

    erase = [BoxRect(page_index=1, x0=87.89, y0=669.21, x1=524.10, y1=680.44)]
    # Pen at 87.23: the measured TOC shortfall of 0.66pt at 11pt.
    assert _keep_point(87.23, 675.0, erase, None, font_size=11.0) is False


def test_pen_pad_scales_with_the_font_size() -> None:
    """The pad is an em share, so a display title gets a wider pad than body text.

    A fixed point pad would have to be sized for the largest face on the page
    and would then reach into the margin from small body text. Scaling it by the
    ``Tf`` size in force keeps the pad proportional to the bearing it compensates
    for.
    """
    from ubt.adapters.pdf.strip_pdf_text import PEN_PAD_EM, _keep_point

    erase = [BoxRect(page_index=1, x0=100.0, y0=0.0, x1=200.0, y1=20.0)]

    # A 40pt display line's pen 1.9pt left of the box is inside the pad.
    assert _keep_point(98.1, 10.0, erase, None, font_size=40.0) is False
    # The same 1.9pt shortfall at 9pt body size is outside it: 9 * PEN_PAD_EM.
    assert 9.0 * PEN_PAD_EM < 1.9
    assert _keep_point(98.1, 10.0, erase, None, font_size=9.0) is True


def test_keep_rect_wins_over_erase_rect() -> None:
    """An explicit protection is never overridden by an overlapping block box."""
    pdf = pikepdf.Pdf.new()
    page = pdf.add_blank_page(page_size=(300, 300))
    page.Contents = pdf.make_stream(b"BT 1 0 0 1 60 60 Tm (Protected Formula) Tj ET")

    erase = [BoxRect(page_index=1, x0=40.0, y0=40.0, x1=200.0, y1=80.0)]
    keep = [BoxRect(page_index=1, x0=50.0, y0=50.0, x1=150.0, y1=70.0)]
    stripped = strip_stream_text(page, erase_rects=erase, keep_rects=keep)
    assert _texts_in(stripped) == ["Protected Formula"]


def test_strip_blanking_preserves_the_operator_sequence() -> None:
    """Erasing must not delete operators: they advance the text matrix.

    Deleting a ``Tj`` also deletes its advance, so every later run on the page
    shifts. Measured on the corpus, blanking keeps survivors at 0.0 pt drift.
    """
    pdf = pikepdf.Pdf.new()
    page = pdf.add_blank_page(page_size=(300, 300))
    page.Contents = pdf.make_stream(
        b"BT /F1 12 Tf 14 TL 1 0 0 1 50 200 Tm (erase me) Tj "
        b"T* (keep me) Tj "
        b"0 -20 Td (erase me too) Tj "
        b"[(a) -120 (b) 50 (c)] TJ "
        b"ET"
    )

    # ``T*`` moves to y=200-14=186; ``Td 0 -20`` moves to 186-20=166.
    # Erase the two upper lines, keep the ``Td`` line and the ``TJ`` run.
    erase = [BoxRect(page_index=1, x0=0.0, y0=180.0, x1=300.0, y1=205.0)]
    stripped = strip_stream_text(page, erase_rects=erase)
    assert _operators_in(stripped) == [
        "BT",
        "Tf",
        "TL",
        "Tm",
        "Tj",
        "T*",
        "Tj",
        "Td",
        "Tj",
        "TJ",
        "ET",
    ]
    assert _texts_in(stripped) == ["erase me too", "a", "b", "c"]


def test_strip_splits_a_bt_block_by_line() -> None:
    """One ``BT`` block holding many lines is split per line, not all-or-nothing.

    This is the defect that mattered: on ``2609.22978v1.pdf`` p14 a single block
    carried 72 text lines, so a block-granularity decision could only strip the
    page clean or not at all.
    """
    pdf = pikepdf.Pdf.new()
    page = pdf.add_blank_page(page_size=(300, 300))
    page.Contents = pdf.make_stream(
        b"BT /F1 10 Tf 12 TL 1 0 0 1 40 250 Tm "
        b"(top line) Tj T* "
        b"(middle line) Tj T* "
        b"(bottom line) Tj "
        b"ET"
    )

    # Erase only the middle line: its baseline is 250 - 12 = 238.
    erase = [BoxRect(page_index=1, x0=0.0, y0=232.0, x1=300.0, y1=244.0)]
    stripped = strip_stream_text(page, erase_rects=erase)
    assert _texts_in(stripped) == ["top line", "bottom line"]


def test_strip_respects_the_ctm_when_placing_a_line() -> None:
    """A scaled ``cm`` moves the page position, so the rect must be in page space."""
    pdf = pikepdf.Pdf.new()
    page = pdf.add_blank_page(page_size=(300, 300))
    # A 10x scale then a 5,5 origin puts the line's page position at (50, 50).
    page.Contents = pdf.make_stream(b"q 10 0 0 10 0 0 cm BT 1 0 0 1 5 5 Tm (scaled) Tj ET Q")

    erase = [BoxRect(page_index=1, x0=40.0, y0=40.0, x1=60.0, y1=60.0)]
    stripped = strip_stream_text(page, erase_rects=erase)
    assert _texts_in(stripped) == []
    assert b"q" in stripped
    assert b"Q" in stripped


def test_strip_leaves_an_unparseable_stream_intact() -> None:
    """Fail open: a malformed stream keeps its bytes, so vector art survives.

    The previous implementation returned ``b""``, which blanked the page's
    paths, rules and colours along with the text it could not read.
    """
    pdf = pikepdf.Pdf.new()

    # A content object whose bytes are not a parseable content stream at all.
    broken = pdf.make_stream(b"garbage")
    broken.write(b"\x00\xff not a content stream \x00")
    stripped = strip_stream_text(broken)
    assert stripped != b""


def test_strip_form_xobjects_keeps_figure_labels_without_erase_rects() -> None:
    """Figure labels inside an XObject survive unless explicitly erased.

    Measured: on ``2606.23344v1.pdf`` p2 the XObjects hold chart axis ticks
    (``68 70 75 80 85 90``) and a legend — in-figure text that Phase 1–4 keeps.
    The facsimile path therefore does not descend into them at all, and even a
    caller that does must pass no erase rects to leave them untouched.
    """
    pdf = pikepdf.Pdf.new()
    page = pdf.add_blank_page(page_size=(300, 300))
    form = pdf.make_stream(b"5 5 m BT /F1 10 Tf 1 0 0 1 5 5 Tm (Axis 68) Tj ET 25 25 l S")
    form[pikepdf.Name("/Type")] = pikepdf.Name("/XObject")
    form[pikepdf.Name("/Subtype")] = pikepdf.Name("/Form")
    form[pikepdf.Name("/BBox")] = pikepdf.Array([0, 0, 100, 100])
    page[pikepdf.Name("/Resources")] = pikepdf.Dictionary(XObject=pikepdf.Dictionary(Fm0=form))

    # ``page.Resources`` is typed as ``Object`` (a PDF value can be a
    # dictionary, an array or a name). This page's resources were just assigned
    # as a dictionary, so asserting that states the precondition the function
    # relies on instead of casting it away.
    resources = page.Resources
    assert isinstance(resources, pikepdf.Dictionary)
    visited = strip_form_xobjects_recursive(resources)
    assert visited == 1
    assert b"Axis 68" in form.read_bytes()
    assert b"25 25 l" in form.read_bytes()


def test_strip_text_streams_absent_page_keeps_its_text(tmp_path: Path) -> None:
    """A page with no erase entry keeps all its text (the fail-safe direction)."""
    src_pdf = tmp_path / "source.pdf"
    dst_pdf = tmp_path / "stripped.pdf"

    pdf = pikepdf.Pdf.new()
    p1 = pdf.add_blank_page(page_size=(400, 500))
    p1.Contents = pdf.make_stream(b"10 20 m BT /F1 10 Tf 1 0 0 1 50 300 Tm (Page 1 Text) Tj ET S")
    p2 = pdf.add_blank_page(page_size=(400, 500))
    p2.Contents = pdf.make_stream(b"30 40 m BT /F1 10 Tf 1 0 0 1 50 300 Tm (Page 2 Text) Tj ET S")
    pdf.save(src_pdf)

    # Erase page 1's text only.
    erase = {1: [BoxRect(page_index=1, x0=40.0, y0=290.0, x1=200.0, y1=310.0)]}
    strip_text_streams(src_pdf, dst_pdf, per_page_erase_rects=erase)

    with pikepdf.open(dst_pdf) as out:
        assert len(out.pages) == 2
        p1_bytes = out.pages[0].Contents.read_bytes()
        p2_bytes = out.pages[1].Contents.read_bytes()
        assert _texts_in(p1_bytes) == []
        assert _texts_in(p2_bytes) == ["Page 2 Text"]
        assert b"10 20 m" in p1_bytes
        assert b"30 40 m" in p2_bytes


def test_strip_text_streams_rects_are_page_scoped(tmp_path: Path) -> None:
    """A rect carrying another page's index must not erase this page's text."""
    src_pdf = tmp_path / "source.pdf"
    dst_pdf = tmp_path / "stripped.pdf"

    pdf = pikepdf.Pdf.new()
    page = pdf.add_blank_page(page_size=(400, 500))
    page.Contents = pdf.make_stream(b"BT /F1 10 Tf 1 0 0 1 50 300 Tm (Body Text) Tj ET")
    pdf.save(src_pdf)

    # The rect covers the line, but claims to belong to page 2.
    erase = {1: [BoxRect(page_index=2, x0=40.0, y0=290.0, x1=200.0, y1=310.0)]}
    strip_text_streams(src_pdf, dst_pdf, per_page_erase_rects=erase)

    with pikepdf.open(dst_pdf) as out:
        assert _texts_in(out.pages[0].Contents.read_bytes()) == ["Body Text"]


# --------------------------------------------------------------------------- #
# flow
# --------------------------------------------------------------------------- #


def test_flow_solver_punctuation_preference() -> None:
    text = "这是第一句话。这是第二句话！这是第三句话？"
    bbox1: BBox = (0.0, 0.0, 100.0, 30.0)
    bbox2: BBox = (0.0, 0.0, 100.0, 30.0)
    box1 = PhysicalBox.of(1, bbox1)
    box2 = PhysicalBox.of(1, bbox2)

    measure = approximate_measure(font_size_pt=10.0, leading_pt=12.0)
    placements = solve_flow(text, [box1, box2], measure)
    assert len(placements) == 2
    assert placements[0].text.endswith(("。", "！", "？"))
    assert placements[0].text + placements[1].text == text


def test_flow_solver_cjk_kinsoku() -> None:
    text = "一个没有任何标点符号的长长中文段落用来测试避头尾规则是否正确生效"
    bbox1: BBox = (0.0, 0.0, 50.0, 20.0)
    bbox2: BBox = (0.0, 0.0, 100.0, 50.0)
    box1 = PhysicalBox.of(1, bbox1)
    box2 = PhysicalBox.of(1, bbox2)

    measure = approximate_measure(font_size_pt=10.0, leading_pt=12.0)
    placements = solve_flow(text, [box1, box2], measure)
    assert len(placements) == 2
    assert len(placements[0].text) > 0
    assert len(placements[1].text) > 0


# --------------------------------------------------------------------------- #
# layout_solver
# --------------------------------------------------------------------------- #


def test_layout_solver_binary_search_convergence() -> None:
    records = [
        {"id": "p1", "top": 500.0, "bot": 420.0, "w": 300.0, "glyphs": 50.0, "n": 3},
        {"id": "p2", "top": 400.0, "bot": 320.0, "w": 300.0, "glyphs": 60.0, "n": 4},
    ]
    sol = solve_page_2d([records], source_size_pt=10.0, source_leading_pt=13.0)
    assert sol.fits is True
    assert 7.5 <= sol.uniform_size_pt <= 11.0  # 1.10 * 10 = 11.0
    assert sol.leading_em >= 0.15
    assert sol.delivery_tier == "L1"


def test_layout_solver_dynamic_leading_stretch() -> None:
    records = [
        {"id": "p1", "top": 600.0, "bot": 500.0, "w": 400.0, "glyphs": 20.0, "n": 2},
        {"id": "p2", "top": 450.0, "bot": 350.0, "w": 400.0, "glyphs": 20.0, "n": 2},
    ]
    sol = solve_page_2d([records], source_size_pt=10.0, source_leading_pt=13.0)
    assert sol.fits is True
    assert sol.uniform_size_pt == 11.0  # Capped at ceiling
    assert sol.leading_em >= 0.30  # Leading stretched


def test_layout_solver_uses_the_page_source_size() -> None:
    """An 11.3pt page must not be capped at 11.0pt by a hardcoded 10pt source.

    The ceiling is ``min(MAX_BODY_PT, source * 1.10)``, so reading the source
    size as 10.0 forced every 11.3pt page down to 11.0pt — smaller than the
    book it was supposed to reproduce.
    """
    records = [
        {"id": "p1", "top": 600.0, "bot": 500.0, "w": 400.0, "glyphs": 20.0, "n": 2},
    ]
    sol = solve_page_2d([records], source_size_pt=11.3, source_leading_pt=15.0)
    assert sol.uniform_size_pt == pytest.approx(12.0)  # min(12.0, 11.3*1.10)


def test_layout_solver_never_returns_a_size_below_the_floor() -> None:
    """A mis-measured source size must not produce a too-small L1 solution.

    ``ceiling = min(MAX_BODY_PT, source * 1.10)``. When the measured source size
    is itself too small the ceiling drops *under* the floor: measured on
    ``chapter-3.pdf`` p22, a formula-dense page whose modal reported size is a
    math font, ``source_body_size`` returns 6.2pt for an 8.9pt book and the
    ceiling becomes 6.82pt. The solver returned that as ``fits=True,
    delivery_tier="L1"`` — text smaller than the source it reproduces, which is
    the one outcome the channel exists to avoid.

    The floor bounds the *output*, not merely the search interval, so this case
    must route to L2 and report the floor rather than ship 6.82pt.
    """
    records = [
        {"id": "p1", "top": 600.0, "bot": 300.0, "w": 400.0, "glyphs": 400.0, "n": 20},
    ]
    sol = solve_page_2d([records], source_size_pt=6.2, source_leading_pt=16.2)
    assert sol.uniform_size_pt >= FLOOR_PT
    assert sol.delivery_tier == "L2"
    assert sol.fits is False


def test_layout_solver_floor_is_not_a_target() -> None:
    """The floor is a lower bound, so a page that fits above it must use that.

    A guard on the mis-measured ceiling must not turn into "clamp everything to
    7.5pt": a healthy page still solves to its ceiling.
    """
    records = [
        {"id": "p1", "top": 600.0, "bot": 500.0, "w": 400.0, "glyphs": 20.0, "n": 2},
    ]
    sol = solve_page_2d([records], source_size_pt=9.0, source_leading_pt=12.0)
    assert sol.uniform_size_pt == pytest.approx(9.9)  # min(12.0, 9.0*1.10)
    assert sol.uniform_size_pt > FLOOR_PT


def test_layout_solver_overlap_detection_and_resolution() -> None:
    records = [
        {"id": "p1", "top": 500.0, "bot": 400.0, "w": 300.0, "glyphs": 30.0, "n": 2},
        {"id": "p2", "top": 401.5, "bot": 300.0, "w": 300.0, "glyphs": 30.0, "n": 2},
    ]
    assert check_source_overlap([records]) is True

    resolved = resolve_local_overlaps(records, max_merge_overlap_pt=2.0)
    assert check_source_overlap([resolved]) is False


# --------------------------------------------------------------------------- #
# facsimile_renderer
# --------------------------------------------------------------------------- #


def test_facsimile_renderer_markup_and_coordinates() -> None:
    placement1 = FacsimileBlockPlacement(
        block_id="b1",
        page=1,
        x0=50.0,
        y0=600.0,
        x1=350.0,
        y1=650.0,
        text="Chapter 1: Facsimile Architecture",
        is_heading=True,
        text_color_hex="#0078ae",
        # What the adapter actually sends for a heading: it classifies weight
        # from ``is_heading or is_bold``, so a heading arrives bold rather than
        # relying on this dataclass's ``regular`` default.
        font_weight="bold",
    )
    placement2 = FacsimileBlockPlacement(
        block_id="b2",
        page=1,
        x0=50.0,
        y0=500.0,
        x1=350.0,
        y1=580.0,
        text="This is 1:1 visual isomorphism prose.",
    )
    page_spec = FacsimilePageSpec(
        page_no=1,
        page_w=500.0,
        page_h=700.0,
        uniform_size_pt=10.5,
        leading_em=0.38,
        placements=[placement1, placement2],
    )

    renderer = FacsimileRenderer()
    markup = renderer.build_markup([page_spec], "bg_stripped.pdf", target_lang="zh")

    assert "#page(width: 500.0pt, height: 700.0pt" in markup
    assert 'image("bg_stripped.pdf", page: 1' in markup
    assert "dx: 50.0pt, dy: 50.0pt" in markup
    assert "dx: 50.0pt, dy: 120.0pt" in markup
    assert '#text(weight: "bold", fill: rgb("#0078ae"))' in markup


def test_facsimile_renderer_clones_the_heading_size() -> None:
    """A 16.2pt chapter title must not be drawn at the 10pt body size."""
    placement = FacsimileBlockPlacement(
        block_id="h1",
        page=1,
        x0=50.0,
        y0=600.0,
        x1=350.0,
        y1=650.0,
        text="Chapter One",
        is_heading=True,
        font_size_pt=16.2,
    )
    page_spec = FacsimilePageSpec(
        page_no=1,
        page_w=500.0,
        page_h=700.0,
        uniform_size_pt=10.0,
        leading_em=0.35,
        placements=[placement],
    )
    markup = FacsimileRenderer().build_markup([page_spec], "bg.pdf", target_lang="zh")
    assert "size: 16.2pt" in markup


def test_block_font_size_clones_a_small_caption_below_the_body_floor() -> None:
    """A 6.45pt figure caption must be drawn at 6.45pt, not raised to 7.5pt.

    ``FIG. A.1``…``FIG. A.6`` in ``chapter-3.pdf`` are genuinely 6.45pt in the
    source. Reproducing them is the channel's purpose; the 7.5pt floor bounds
    **body prose**, not a caption the original set small. A future "floor
    violation" report on these five blocks is a false positive — raising them
    would break the visual fidelity that is Track A's whole reason to exist.
    """
    block = SimpleNamespace(
        style=SimpleNamespace(font_size=6.45),
        provenance=SimpleNamespace(font_size=6.45),
    )
    size = _block_font_size(block, is_hdr=True, body_size=8.0)
    assert size is not None, "a heading with a plausible measurement must not inherit"
    assert size == pytest.approx(6.45)
    assert size < FLOOR_PT, "the caption is below the body floor by design"


def test_block_font_size_ignores_an_implausible_measurement() -> None:
    """A producer reporting ``1.0`` must not pin every heading to 1pt.

    ``chapter-3.pdf`` is one of the producers that reports ``1.0`` for every
    glyph, so the plausibility floor exists. Below it, the heading falls back to
    the page body size scaled by the heading ratio.
    """
    block = SimpleNamespace(
        style=SimpleNamespace(font_size=1.0),
        provenance=SimpleNamespace(font_size=1.0),
    )
    size = _block_font_size(block, is_hdr=True, body_size=10.0)
    assert size == pytest.approx(16.0)  # body_size * 1.6


def test_block_font_size_lets_body_prose_inherit_the_page_size() -> None:
    """Body prose must inherit the solved page size, not its own measurement.

    The page solves one uniform body size; a body block carrying its own size
    would fragment that, which is the opposite of the channel's contract.
    """
    block = SimpleNamespace(
        style=SimpleNamespace(font_size=11.0),
        provenance=SimpleNamespace(font_size=11.0),
    )
    assert _block_font_size(block, is_hdr=False, body_size=9.0) is None


def test_facsimile_renderer_heading_weight_is_not_inverted() -> None:
    """A regular-weight heading stays regular; only a bold one is forced bold.

    ``"bold" if weight in ("regular", "")`` flipped the sense: every heading
    whose measured weight was regular was emitted bold.
    """
    placement = FacsimileBlockPlacement(
        block_id="h1",
        page=1,
        x0=50.0,
        y0=600.0,
        x1=350.0,
        y1=650.0,
        text="A Regular Heading",
        is_heading=True,
        font_weight="regular",
    )
    page_spec = FacsimilePageSpec(
        page_no=1,
        page_w=500.0,
        page_h=700.0,
        uniform_size_pt=10.0,
        leading_em=0.35,
        placements=[placement],
    )
    markup = FacsimileRenderer().build_markup([page_spec], "bg.pdf", target_lang="zh")
    assert 'weight: "bold"' not in markup


def test_facsimile_renderer_tables_are_tables() -> None:
    """A markdown table block must become ``#table``, not escaped pipe text."""
    placement = FacsimileBlockPlacement(
        block_id="t1",
        page=1,
        x0=50.0,
        y0=400.0,
        x1=400.0,
        y1=520.0,
        text="| Metric | Value |\n|---|---|\n| Acc | 91.2 |",
        is_table=True,
    )
    page_spec = FacsimilePageSpec(
        page_no=1,
        page_w=500.0,
        page_h=700.0,
        uniform_size_pt=10.0,
        leading_em=0.35,
        placements=[placement],
    )
    markup = FacsimileRenderer().build_markup([page_spec], "bg.pdf", target_lang="zh")
    assert "#table(" in markup
    assert "|" not in markup


def test_facsimile_strip_plan_does_not_band_the_furniture_margins(tmp_path: Path) -> None:
    """The plan must not add a full-width keep band at the page's top or bottom.

    The first implementation kept ``[0, 0.08·H]`` and ``[0.92·H, H]`` on every
    page, so that a folio or running head — which has no IR block, the parser
    discards repeating chrome — would not be erased with nothing to redraw it.

    Measurement refuted both halves of that reasoning:

    * Text that no block covers has no erase rect over it either, so it survives
      on its own. On ``chapter-3.pdf`` p1–p3 the folio, running head and formula
      survived identically with the bands and without them.
    * Body prose routinely reaches into the band (p2's last paragraph line is at
      ``y0 = 31.3`` on a 666pt page, whose band top is 53.3). Those lines *are*
      covered by a translated block, so the band vetoed their erase rect and left
      the source line's ink under its translation: 2 leaked English lines per
      page on p1–p3, 0 once the bands were removed.

    So the assertion is on the *absence* of the band, and on a keep rect that is
    exactly the untranslated block rather than the whole page width.
    """
    body = _fake_block(
        "b1",
        page=1,
        bbox=(80.0, 31.3, 456.0, 95.2),  # reaches into the bottom 8% of 666pt
        block_type=BlockType.NARRATIVE,
        skip_translate=False,
    )
    running_head = _fake_block(
        "b2",
        page=1,
        bbox=(83.7, 624.0, 294.6, 633.9),
        block_type=BlockType.HEADING,
        skip_translate=True,
    )

    erase, keep = _facsimile_strip_plan({1: [body, running_head]}, 1, _source_pdf(tmp_path))

    assert len(erase[1]) == 1, "the translated body block must be erased"
    assert len(keep[1]) == 1, (
        "only the untranslated block may be protected; a furniture band would "
        "add a second full-width rect"
    )
    assert (keep[1][0].x0, keep[1][0].x1) == (83.7, 294.6), (
        "the keep rect must be the running head's own box, not the full page width"
    )


def test_facsimile_strip_plan_keeps_body_prose_that_reaches_into_the_margin(tmp_path: Path) -> None:
    """End-to-end: a paragraph line below ``0.08·H`` must still be erased.

    This is the defect the band caused, reproduced at the strip level so the
    regression cannot come back through a different code path.
    """
    page_h = 666.0
    pdf = pikepdf.Pdf.new()
    page = pdf.add_blank_page(page_size=(540, page_h))
    page.Contents = pdf.make_stream(
        b"BT 1 0 0 1 121 626 Tm (70 CHAPTER 3 Core model) Tj ET "
        b"BT 1 0 0 1 121 37 Tm (last line of the paragraph) Tj ET"
    )

    body = _fake_block(
        "b1",
        page=1,
        bbox=(121.0, 31.3, 456.0, 95.2),
        block_type=BlockType.NARRATIVE,
        skip_translate=False,
    )
    # ``_facsimile_strip_plan`` reads the source PDF to detect vector-diagram
    # regions, so it needs a real path; ``None`` would have raised inside it.
    erase, keep = _facsimile_strip_plan({1: [body]}, 1, _source_pdf(tmp_path))
    stripped = strip_stream_text(page, erase_rects=erase[1], keep_rects=keep.get(1, []))

    texts = _texts_in(stripped)
    assert "70 CHAPTER 3 Core model" in texts, "the running head must survive"
    assert "last line of the paragraph" not in texts, (
        "the body line at y=37 is inside a translated block and must be erased, "
        "even though it sits below 0.08·H"
    )


def test_facsimile_renderer_full_e2e_compile(tmp_path: Path) -> None:
    src_pdf = tmp_path / "book_source.pdf"
    pdf = pikepdf.Pdf.new()
    p1 = pdf.add_blank_page(page_size=(400, 600))
    p1.Contents = pdf.make_stream(b"20 20 m 200 200 l S BT /F1 12 Tf (Source text p1) Tj ET")
    p2 = pdf.add_blank_page(page_size=(400, 600))
    p2.Contents = pdf.make_stream(b"40 40 m 300 300 l S BT /F1 12 Tf (Source text p2) Tj ET")
    pdf.save(src_pdf)

    bg_stripped = tmp_path / "bg_stripped.pdf"
    # Blank the source text so the delivered page shows only the translation.
    erase = {
        page_no: [BoxRect(page_index=page_no, x0=0.0, y0=0.0, x1=400.0, y1=600.0)]
        for page_no in (1, 2)
    }
    strip_text_streams(src_pdf, bg_stripped, per_page_erase_rects=erase)

    p1_spec = FacsimilePageSpec(
        page_no=1,
        page_w=400.0,
        page_h=600.0,
        uniform_size_pt=10.0,
        leading_em=0.35,
        placements=[
            FacsimileBlockPlacement(
                block_id="p1_h",
                page=1,
                x0=40.0,
                y0=520.0,
                x1=360.0,
                y1=560.0,
                text="第一章：原位同构系统",
                is_heading=True,
            ),
            FacsimileBlockPlacement(
                block_id="p1_b",
                page=1,
                x0=40.0,
                y0=440.0,
                x1=360.0,
                y1=500.0,
                text="第一页翻译正文，原位填充在矩形内。",
            ),
        ],
    )
    p2_spec = FacsimilePageSpec(
        page_no=2,
        page_w=400.0,
        page_h=600.0,
        uniform_size_pt=10.0,
        leading_em=0.35,
        placements=[
            FacsimileBlockPlacement(
                block_id="p2_b",
                page=2,
                x0=40.0,
                y0=480.0,
                x1=360.0,
                y1=540.0,
                text="第二页翻译正文，矢量背景完全保留。",
            )
        ],
    )

    out_pdf = tmp_path / "delivered_facsimile.pdf"
    build_dir = tmp_path / ".build_facsimile"

    renderer = FacsimileRenderer()
    renderer.render_document(
        [p1_spec, p2_spec],
        bg_stripped,
        out_pdf,
        build_dir,
        target_lang="zh",
    )

    assert out_pdf.exists()
    with pikepdf.open(out_pdf) as res:
        assert len(res.pages) == 2
        assert "/XObject" in res.pages[0].Resources
        assert "/XObject" in res.pages[1].Resources
