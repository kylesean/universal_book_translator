"""Delivery invariants: the three assertions the PRD's §9 contract rests on.

These are deliberately *red first*. Each one pins a property the current
renderer does not have, and each failure names the exact page shape that
produced it in the 26-page BSIM-CMG job:

1. **Column detection.** A single-column page whose formulas are centred must
   not be read as two columns. Page 12 of ``chapter-3.pdf`` has two short
   left-edge labels and six centred display formulas; the ``x0 >= mid - 15``
   heuristic counted the formulas as a right column and reflowed the page into
   two columns, which is the "single column became two" defect.

2. **L1 placement bounds.** A canvas placement must declare the source box's
   height and clip overflow. The current ``#place(... block(width: ...))``
   declares only width, so a heading measured at the wrong width paints over
   the paragraph below (page 2) and a long paragraph is cut by the page edge
   with no continuation (page 6).

3. **Leading semantics.** ``solve_page_2d`` returns ``leading_em = em_used -
   1.0`` and the renderer passes it to Typst's ``par(leading:)``, which is a
   *gap* on top of the font's own line height, not a ratio. On Noto Serif CJK
   SC the base line height is ~0.733em, so a 1.35 line-height intent ships as
   ~1.083em: captions overlap and the solver believes it has more room than it
   has.

The tests read the compiled artifact or the emitted markup, not the solver's
own estimate. A model that grades itself is how the current job shipped with
``fits=1`` on pages that visibly overlap.
"""

from __future__ import annotations

import shutil
import statistics
from pathlib import Path

import pikepdf
import pytest

from ubt.core.ir.models import BlockType, BoundingBox, IRBlock, make_element
from ubt.render.layout_solver import solve_page_2d
from ubt.render.typst.facsimile_renderer import (
    FacsimileBlockPlacement,
    FacsimilePageSpec,
    FacsimileRenderer,
)
from ubt.render.typst.page_locked_renderer import extract_page_geometry

pytestmark = pytest.mark.fast


#: Page-12 boxes of the 26-page BSIM-CMG chapter, in PDF points. Two short
#: left-edge labels, six centred display formulas, and the body paragraphs that
#: span the full measure. Reproduced here rather than read from ``.ubt`` so the
#: regression runs on a clean checkout.
_P12_BOXES: tuple[tuple[float, float, float, float], ...] = (
    (121.5496, 576.8129528888888, 372.18135767999973, 586.3199916),
    (251.6598, 549.3276003555557, 456.3269808000001, 570.0092558),
    (242.5883, 520.0175003555556, 456.3269808000001, 540.9170492),
    (264.5291, 476.70400035555554, 456.3269808000001, 497.95255579999997),
    (121.5496, 366.71942595555555, 456.31936275999976, 470.8364916),
    (121.5501, 342.79522595555557, 456.26352845999975, 363.0550648),
    (214.1858, 309.95246394400004, 456.3269808000001, 339.94074919999997),
    (261.4109, 266.20330035555554, 456.3269808000001, 286.78016240000005),
    (121.5496, 251.7453259555556, 298.69658311999984, 260.0430648),
    (261.4109, 222.26650035555554, 456.3269808000001, 244.41314920000002),
    (121.5496, 100.2629259555556, 456.37365705999935, 216.1627648),
    (191.622, 68.79261722909087, 456.3270808000001, 98.48564920000001),
    (121.5496, 504.58615288888893, 178.62391122000002, 514.1498915999999),
    (121.5496, 293.5185528888889, 207.30470730000005, 303.0255916),
)


def _blocks(boxes: tuple[tuple[float, float, float, float], ...], page: int) -> list[IRBlock]:
    """Real ``IRBlock``s, not stand-ins.

    ``extract_page_geometry`` is typed against ``Sequence[IRBlock]`` and reads the
    block's derived ``bbox``/``block_type``/``style`` properties, which only a
    real element provides. A duck-typed namespace silently keeps the production
    signature unchecked, so the geometry helper's contract is tested against an
    object no caller ever passes.
    """
    out: list[IRBlock] = []
    for index, (x0, y0, x1, y1) in enumerate(boxes, 1):
        element = make_element(
            id=f"p{page:02d}#b{index:04d}",
            spine_index=index,
            block_type=BlockType.NARRATIVE,
            source_text="",
            bbox=BoundingBox(page=page, x0=x0, y0=y0, x1=x1, y1=y1),
        )
        out.append(IRBlock(element=element))
    return out


def test_p12_style_single_column_with_centred_formulas_is_not_two_columns() -> None:
    """A centred formula is not a right column.

    Page 12 has two short left labels (``b0105``/``b0106``) and six display
    formulas whose boxes start near the page midpoint. Counting ``x0 >= mid``
    as "right column" makes the page two-column, so the L2 renderer sorts the
    body into a left track and the formulas into a right track. The source page
    is single-column; the delivered page is not.
    """
    geometry = extract_page_geometry(540.0, 665.97, _blocks(_P12_BOXES, 12))
    assert geometry.columns == 1, (
        "page 12 is single-column; centred display formulas must not be counted "
        f"as a right column (got columns={geometry.columns})"
    )


def test_a_true_two_column_page_is_still_two_columns() -> None:
    """The guard must not be bought by making column detection always 1."""
    boxes: list[tuple[float, float, float, float]] = []
    for index in range(4):
        top = 600.0 - index * 120.0
        boxes.append((80.0, top - 90.0, 250.0, top))
        boxes.append((300.0, top - 90.0, 470.0, top))

    geometry = extract_page_geometry(540.0, 665.97, _blocks(tuple(boxes), 1))
    assert geometry.columns == 2


def test_l1_placement_markup_declares_height_and_clips_overflow() -> None:
    """A canvas placement must be bounded, not just positioned.

    The source box has a height; the translation may be taller. Declaring only
    ``width`` lets the extra lines paint over the next block (page 2) or run off
    the page edge (page 6). The placement must carry the box height and clip
    what does not fit, so the overflow is visible to QA and can be routed to
    another strategy instead of silently colliding.
    """
    placement = FacsimileBlockPlacement(
        block_id="b0010",
        page=2,
        x0=121.55,
        y0=263.97,
        x1=344.84,
        y1=276.27,
        text="3.1 Core model for double-gate FinFETs",
        is_heading=True,
        font_size_pt=12.67,
    )
    page = FacsimilePageSpec(
        page_no=2,
        page_w=540.0,
        page_h=665.97,
        uniform_size_pt=9.0,
        leading_em=0.348,
        placements=[placement],
    )

    markup = FacsimileRenderer().build_markup([page], "bg_stripped.pdf", target_lang="zh")

    # Scope the assertion to the placement itself. ``height:`` also appears on
    # the enclosing ``#page(...)``, and a test that accepts the page's height
    # would pass while the placement stayed unbounded — the same "assert on the
    # shape of markup" trap this file exists to avoid.
    placement_markup = markup.split("#place(", 1)[1]
    assert "width:" in placement_markup
    assert "height:" in placement_markup, (
        "the placement must declare the source box height; width alone lets a "
        "12.67pt heading measured at 334.85pt paint over the body at 223.3pt"
    )
    assert "clip: true" in placement_markup, (
        "the placement must clip overflow so a too-tall block cannot silently "
        "overlap the next block or run past the page edge"
    )


@pytest.mark.skipif(shutil.which("typst") is None, reason="typst is not installed")
def test_solver_leading_is_a_pitch_ratio_not_a_gap_multiplier(tmp_path: Path) -> None:
    """The leading the solver chooses must be the pitch the artifact delivers.

    ``solve_page_2d`` documents ``leading_em`` as ``em_used - 1.0``, i.e. a
    ratio. The renderer passes it to Typst's ``par(leading:)``, which is a gap
    *added to the font's line height*. On Noto Serif CJK SC the measured base is
    ~0.733em, so a 1.35 line-height intent arrives as ~1.083em. This test
    compiles the solver's own output and measures the baseline pitch in the
    artifact; it must equal the source's 1.35 ratio.
    """
    records = [
        # At 11pt this needs 8 lines and does not fit; at 10pt it needs 7 and
        # does. The solver therefore stops below the ceiling and keeps the
        # source 1.35 ratio instead of stretching it.
        {"id": "p1", "top": 600.0, "bot": 500.0, "w": 300.0, "glyphs": 200.0, "n": 3}
    ]
    solution = solve_page_2d([records], source_size_pt=10.0, source_leading_pt=13.5)
    assert solution.fits
    assert solution.uniform_size_pt < 11.0 - 0.05, (
        "the fixture must stay below the ceiling so the source ratio is not "
        "stretched by the solver's contraction step"
    )

    bg = tmp_path / "bg.pdf"
    pdf = pikepdf.Pdf.new()
    pdf.add_blank_page(page_size=(540, 666))
    pdf.save(bg)

    text = "在异步网络环境中任何容错分布式系统都必须在安全性与活性之间做出严格权衡" * 6
    page = FacsimilePageSpec(
        page_no=1,
        page_w=540.0,
        page_h=666.0,
        uniform_size_pt=solution.uniform_size_pt,
        leading_em=solution.leading_em,
        placements=[
            FacsimileBlockPlacement(
                block_id="p1",
                page=1,
                x0=120.0,
                y0=500.0,
                x1=420.0,
                y1=600.0,
                text=text,
            )
        ],
    )
    out = tmp_path / "delivered.pdf"
    FacsimileRenderer().render_document([page], bg, out, tmp_path / ".build", target_lang="zh")

    pitch_ratio = _measured_pitch_ratio(out, solution.uniform_size_pt)
    assert pitch_ratio == pytest.approx(1.35, abs=0.03), (
        "the solver's 1.35 line-height intent must arrive as a 1.35 pitch; "
        f"the artifact measures {pitch_ratio:.3f}"
    )


def _measured_pitch_ratio(pdf: Path, size_pt: float) -> float:
    """Median baseline pitch / font size, read from the compiled PDF."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(pdf))
    try:
        textpage = doc[0].get_textpage()
        text = "".join(textpage.get_text_range(i, 1) for i in range(textpage.count_chars()))
        starts = [0] + [i + 1 for i, char in enumerate(text) if char == "\n"]
        bottoms: list[float] = []
        for start in starts:
            index = start
            while index < len(text) and text[index] in "\r\n":
                index += 1
            if index >= len(text):
                continue
            x0, bottom, x1, _top = textpage.get_charbox(index)
            if x1 - x0 > 0.1:
                bottoms.append(float(bottom))
        assert len(bottoms) >= 2, "the compiled page must contain at least two lines"
        gaps = [bottoms[i] - bottoms[i + 1] for i in range(len(bottoms) - 1)]
        return statistics.median(gaps) / size_pt
    finally:
        doc.close()
