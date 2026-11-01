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

_PAGE_W = 612.0
_PAGE_H = 792.0


def _tight_pdf(tmp_path: Path, *, leading: float = 8.0) -> Path:
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
    page[NameObject("/Resources")] = DictionaryObject(
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
    assert DEFAULT_BLEED_PT > DELIVERED_BLEED_PT


def test_a_crowded_side_is_shrunk_and_an_open_side_is_not(tmp_path: Path) -> None:
    pdf = _tight_pdf(tmp_path)
    middle = _lines(pdf)[1]
    insets = delivered_crop_insets(pdf, 1, middle)

    assert not _at_cap(insets, "top")
    assert not _at_cap(insets, "bottom")
    assert _at_cap(insets, "left")
    assert _at_cap(insets, "right")


def test_the_delivered_crop_stops_short_of_the_neighbour(tmp_path: Path) -> None:
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
    assert grown_pt <= clearance_above + clearance_below + 0.05
    assert height_pt < block_pt + DEFAULT_BLEED_PT * 2


def test_the_delivered_crop_still_contains_the_block(tmp_path: Path) -> None:
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
    pdf = tmp_path / "single.pdf"
    write_text_pdf(pdf, [("MMMMM MMMMM MMMMM",)])
    only = extract_text_rects(pdf, 1)[0]
    insets = delivered_crop_insets(pdf, 1, only)
    assert all(v == pytest.approx(DELIVERED_BLEED_PT) for v in insets.values())


def test_missing_geometry_fails_closed_to_no_margin(tmp_path: Path) -> None:
    pdf = tmp_path / "empty.pdf"
    write_text_pdf(pdf, [("MMMMM MMMMM MMMMM",)])
    insets = delivered_crop_insets(pdf, 2, (10.0, 10.0, 50.0, 40.0))
    assert all(v == 0.0 for v in insets.values())


def test_the_prompt_crop_is_unaffected(tmp_path: Path) -> None:
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
    pdf = _tight_pdf(tmp_path)
    lines = _lines(pdf)
    lower, upper = lines[-1], lines[-2]

    overlap_pt = 1.5
    greedy = (upper[0], lower[3] - overlap_pt, upper[2], upper[3])

    visual_scalpel._PAGE_CACHE = None
    try:
        trimmed = delivered_crop_pil(pdf, 1, greedy, dpi=300)
        clean = delivered_crop_pil(pdf, 1, upper, dpi=300)
    finally:
        visual_scalpel._PAGE_CACHE = None

    assert trimmed.height <= clean.height + 2, (
        "the sliver of the line below is still in the delivered crop "
        f"({trimmed.height}px vs {clean.height}px)"
    )


def test_a_formula_whose_own_glyphs_overshoot_keeps_them(tmp_path: Path) -> None:
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
    middle = inked[(height // 3) * width : (2 * height // 3) * width]
    assert sum(middle) > 50
