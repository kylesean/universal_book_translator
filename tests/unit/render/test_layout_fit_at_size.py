from __future__ import annotations

import pytest

from ubt.render.layout_solver import (
    apply_width_borrow,
    estimate_needed_height,
    fit_at_size,
)

pytestmark = pytest.mark.fast


def _record(
    *,
    top: float = 700.0,
    bot: float = 660.0,
    width: float = 400.0,
    glyphs: int = 40,
    right_budget: float | None = None,
    block_id: str = "b1",
) -> dict[str, float | int | str]:
    record: dict[str, float | int | str] = {
        "id": block_id,
        "top": top,
        "bot": bot,
        "w": width,
        "x0": 40.0,
        "x1": 40.0 + width,
        "n": 2,
        "glyphs": glyphs,
        "glyph_aspect": 1.0,
    }
    if right_budget is not None:
        record["right_budget"] = right_budget
    return record


def test_a_page_that_holds_the_size_reports_fits_and_no_deficit() -> None:
    report = fit_at_size([[_record(top=700.0, bot=600.0)]], size_pt=8.4, pitch=1.09)
    assert report.fits
    assert report.deficit_pt == 0.0
    assert report.block_ids == ()


def test_a_shortfall_is_quantified_with_the_blocks_that_missed() -> None:
    report = fit_at_size([[_record(top=700.0, bot=694.0)]], size_pt=8.4, pitch=1.09)
    assert not report.fits
    assert report.block_ids == ("b1",)
    assert report.deficit_pt > 0.0


def test_a_line_holds_whole_glyphs_not_a_fraction_of_one() -> None:
    # 344pt / 8.9pt = 38.65 glyphs. Typst wraps at whole-glyph boundaries, so
    # the line holds 38, and 77 glyphs need 3 lines (ceil(77/38) = 3), not the
    # 2 lines the old fractional model (ceil(77/38.65) = 2) promised. That
    # missing line was paid for by the in-box ladder shrinking the font.
    rec = _record(width=344.0, glyphs=77, top=700.0, bot=670.0)
    needed = estimate_needed_height(rec, size=8.9, em_ratio=1.35, tracking_em=0.0)
    assert needed == pytest.approx(3 * 1.35 * 8.9)
    # The fit verdict must agree with the whole-glyph model: a 30pt box holds
    # two 12pt lines but not a third. The fractional model called it 2 lines
    # and passed, leaving the in-box ladder to shrink the font to pay for the
    # line the plan said did not exist.
    report = fit_at_size([[rec]], size_pt=8.9, pitch=1.35)
    assert not report.fits


def test_width_borrowing_never_passes_the_columns_own_right_edge() -> None:
    records = [_record(width=340.0, right_budget=392.0)]
    widened, used = apply_width_borrow([records], 6.0)
    assert widened[0][0]["w"] == pytest.approx(346.0)
    assert used == pytest.approx(6.0)

    widened, used = apply_width_borrow([records], 20.0)
    assert widened[0][0]["w"] == pytest.approx(352.0)
    assert used == pytest.approx(12.0)


def test_a_record_without_a_budget_is_left_alone() -> None:
    records = [_record(width=300.0)]
    widened, used = apply_width_borrow([records], 4.0)
    assert widened[0][0]["w"] == pytest.approx(300.0)
    assert used == 0.0
