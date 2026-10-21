"""A delivered crop must not splice a neighbouring source line into the image.

The L4/repair path crops a block once to *ask a model about it*; a small bleed
around the box is harmless there — the model is told which region it is looking
at. The **delivered** crop is different: the renderer rebuilds the page from the
IR and places this PNG inline, so any source ink inside the crop is drawn onto
the reader's page as part of the picture. On the reference chapter a display
equation's bbox sits as close as 0.15pt to the next text line, and the old
blanket ``DEFAULT_BLEED_PT = 4.0`` sliced that line in: the delivered PDF showed
each formula welded to a half-cut English clause, while the reflowed translation
of the same sentence sat directly below it, reading like a mistranslated
fragment.

The fix is not "bleed zero" — 91 of 96 crops on that chapter have ink touching
the bbox edge, so a hard zero would cut glyphs. It is "no margin beyond the
clearance to the nearest neighbouring text run", applied per side.

Pinned here:

1. a delivered crop stops short of a neighbouring text line's rect;
2. it still contains the block's own region (no glyph is cut);
3. the prompt-side path keeps its margin, so L4 repair is unaffected;
4. a side with nothing near keeps the cap, so ordinary figures look unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pdf_builders import write_text_pdf
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from ubt.adapters.pdf import visual_scalpel
from ubt.adapters.pdf.textgeom import extract_text_rects
from ubt.adapters.pdf.visual_scalpel import (
    DEFAULT_BLEED_PT,
    DELIVERED_BLEED_PT,
    crop_block_pil,
    delivered_crop_insets,
    delivered_crop_pil,
)

pytestmark = pytest.mark.fast

#: Page box used by the tight fixture (US Letter, like the corpus papers).
_PAGE_W = 612.0
_PAGE_H = 792.0


def _tight_pdf(tmp_path: Path, *, leading: float = 8.0) -> Path:
    """Three Helvetica lines set at ``leading`` — a tight academic pitch.

    ``write_text_pdf`` fixes 16pt leading, which leaves 7pt of clearance and so
    can never exercise a 1-4pt margin. Real two-column chapters set body text at
    ~9-10pt with sub-point clearance, which is the layout this defect came from,
    so the fixture reproduces that pitch directly.
    """
    pdf = tmp_path / "tight.pdf"
    writer = PdfWriter()
    page = writer.add_blank_page(width=_PAGE_W, height=_PAGE_H)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_ref = writer._add_object(font)  # noqa: SLF001 - pypdf has no public page-resource API
    page[NameObject("/Resources")] = DictionaryObject(  # resources are a plain dict
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})}
    )
    ops = ["BT", "/F1 9 Tf", "1 0 0 1 72 600 Tm", f"{leading} TL"]
    ops.extend(f"(Prose line {A}) Tj T*" for A in ("A", "B", "C"))
    ops.append("ET")
    stream = DecodedStreamObject()
    stream.set_data("\n".join(ops).encode("latin-1"))
    page[NameObject("/Contents")] = writer._add_object(stream)  # noqa: SLF001
    with pdf.open("wb") as handle:
        writer.write(handle)
    return pdf


def _lines(pdf: Path) -> list[tuple[float, float, float, float]]:
    return sorted(extract_text_rects(pdf, 1), key=lambda r: -r[3])


def _at_cap(insets: dict[str, float], side: str) -> bool:
    return insets[side] >= DELIVERED_BLEED_PT - 1e-9


def test_the_prompt_bleed_is_not_reused_for_delivery() -> None:
    """The two crops answer different questions and must not share a margin."""
    assert DEFAULT_BLEED_PT > DELIVERED_BLEED_PT


def test_a_crowded_side_is_shrunk_and_an_open_side_is_not(tmp_path: Path) -> None:
    """Per-side clamping: only the side with a neighbour loses its margin."""
    pdf = _tight_pdf(tmp_path)
    middle = _lines(pdf)[1]
    insets = delivered_crop_insets(pdf, 1, middle)

    # A line sits above and below this one, so both vertical sides shrink.
    assert not _at_cap(insets, "top")
    assert not _at_cap(insets, "bottom")
    # Nothing shares this line's vertical band horizontally, so the sides keep
    # the full cap — the "ordinary figure looks unchanged" guarantee.
    assert _at_cap(insets, "left")
    assert _at_cap(insets, "right")


def test_the_delivered_crop_stops_short_of_the_neighbour(tmp_path: Path) -> None:
    """The delivered image is no taller than the gap the block actually has.

    The upper bound is what matters: with the old 4pt bleed the crop was taller
    than the clearance, which is exactly the neighbour's line arriving on the
    page.
    """
    pdf = _tight_pdf(tmp_path)
    lines = _lines(pdf)
    middle, above, below = lines[1], lines[0], lines[2]
    clearance_above = above[1] - middle[3]
    clearance_below = middle[1] - below[3]

    visual_scalpel._PAGE_CACHE = None
    try:
        img = delivered_crop_pil(pdf, 1, middle, dpi=300)
    finally:
        visual_scalpel._PAGE_CACHE = None

    scale = 300 / 72.0
    height_pt = img.height / scale
    block_pt = middle[3] - middle[1]
    grown_pt = height_pt - block_pt
    # Total growth is split across the two sides; each side must stay inside its
    # own clearance, which bounds the total by their sum.
    assert grown_pt <= clearance_above + clearance_below + 0.05
    assert height_pt < block_pt + DEFAULT_BLEED_PT * 2


def test_the_delivered_crop_still_contains_the_block(tmp_path: Path) -> None:
    """Shrinking margins must never clip the block's own rect."""
    pdf = _tight_pdf(tmp_path)
    middle = _lines(pdf)[1]
    visual_scalpel._PAGE_CACHE = None
    try:
        img = delivered_crop_pil(pdf, 1, middle, dpi=300)
    finally:
        visual_scalpel._PAGE_CACHE = None

    scale = 300 / 72.0
    assert img.width >= int((middle[2] - middle[0]) * scale) - 2
    assert img.height >= int((middle[3] - middle[1]) * scale) - 2


def test_an_isolated_block_keeps_the_full_cap(tmp_path: Path) -> None:
    """A block with clear space around it is cropped as before."""
    pdf = tmp_path / "single.pdf"
    write_text_pdf(pdf, [("MMMMM MMMMM MMMMM",)])
    only = extract_text_rects(pdf, 1)[0]
    insets = delivered_crop_insets(pdf, 1, only)
    assert all(v == pytest.approx(DELIVERED_BLEED_PT) for v in insets.values())


def test_missing_geometry_fails_closed_to_no_margin(tmp_path: Path) -> None:
    """No text geometry must not mean "grow the maximum"."""
    pdf = tmp_path / "empty.pdf"
    write_text_pdf(pdf, [("MMMMM MMMMM MMMMM",)])
    # Page 2 does not exist: the probe raises and the insets collapse to zero.
    insets = delivered_crop_insets(pdf, 2, (10.0, 10.0, 50.0, 40.0))
    assert all(v == 0.0 for v in insets.values())


def test_the_prompt_crop_is_unaffected(tmp_path: Path) -> None:
    """L4 repair's crop still carries its configured margin."""
    pdf = _tight_pdf(tmp_path)
    middle = _lines(pdf)[1]
    visual_scalpel._PAGE_CACHE = None
    try:
        prompt_crop = crop_block_pil(pdf, 1, middle, dpi=300)
        delivered = delivered_crop_pil(pdf, 1, middle, dpi=300)
    finally:
        visual_scalpel._PAGE_CACHE = None
    assert prompt_crop.height > delivered.height


def test_a_bbox_that_reaches_into_the_next_line_drops_that_sliver(tmp_path: Path) -> None:
    """Withholding the margin is not enough when the *box* already overlaps.

    ``delivered_crop_insets`` can only refuse to grow a crop; it cannot move a
    boundary the extractor placed. On Eq. (3.5) the assigned box ran 1.2pt into
    the line below, so ``bottom: 0`` still baked that line's glyph tops into the
    image — the delivered page showed a ghost line welded under the equation,
    with the reflowed translation of that line directly below it.

    Geometry alone cannot separate the two cases: the intruding line and a
    formula's own glyphs have the same height and both overshoot the box, so the
    cut has to be decided from the pixels, where a neighbour is a *detached*
    band of ink and the block's own glyphs are continuous.
    """
    pdf = _tight_pdf(tmp_path)
    lines = _lines(pdf)
    lower, upper = lines[-1], lines[-2]

    # The box for the upper line, extended down to swallow part of the line
    # below it — what the reference chapter's extractor produced.
    overlap_pt = 1.5
    greedy = (upper[0], lower[3] - overlap_pt, upper[2], upper[3])

    visual_scalpel._PAGE_CACHE = None
    try:
        trimmed = delivered_crop_pil(pdf, 1, greedy, dpi=300)
        # The same box with the intruding sliver already excluded.
        clean = delivered_crop_pil(pdf, 1, upper, dpi=300)
    finally:
        visual_scalpel._PAGE_CACHE = None

    assert trimmed.height <= clean.height + 2, (
        "the sliver of the line below is still in the delivered crop "
        f"({trimmed.height}px vs {clean.height}px)"
    )


def test_a_formula_whose_own_glyphs_overshoot_keeps_them(tmp_path: Path) -> None:
    """The trimming must not eat the block's own ink.

    A first attempt tested the text-rect geometry and cut any run crossing an
    edge. Eq. (A.15a)'s own glyphs overshoot their bbox by up to 1.5pt, so that
    rule cropped the equation out of its own asset — the delivered image came
    out blank. Ink that is continuous with the body is the block's; only a band
    separated by whitespace is a neighbour.
    """
    pdf = _tight_pdf(tmp_path)
    upper = _lines(pdf)[-2]

    visual_scalpel._PAGE_CACHE = None
    try:
        crop = delivered_crop_pil(pdf, 1, upper, dpi=300)
    finally:
        visual_scalpel._PAGE_CACHE = None

    grey = crop.convert("L")
    width, height = grey.size
    raw = grey.tobytes()
    inked = [(value < 200) for value in raw]
    assert sum(inked) > 100, "the crop lost the text it exists to show"
    # Ink in the middle band, not just a stray edge: the glyphs survived.
    middle = inked[(height // 3) * width : (2 * height // 3) * width]
    assert sum(middle) > 50
