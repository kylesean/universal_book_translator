from __future__ import annotations

from pathlib import Path

import pytest

from ubt.adapters.pdf.visual_gate import (
    CANVAS_PAIR_IDENTITY_MIN,
    canvas_page_pair_identity,
    canvas_page_pairs,
)

pytestmark = pytest.mark.fast

_CANVAS_PAGE = (
    "#page(width: 540.0pt, height: 666.0pt, margin: 0pt, "
    'background: image("bg_stripped.pdf", page: {bg}, width: 100%, height: 100%))['
)
_REFLOW_PAGE = "#page(\n  width: 540.0pt,\n  height: 666.0pt,\n  margin: (top: 36pt),"


def test_a_canvas_book_reports_the_pages_it_drew_on_the_background() -> None:
    source = "\n".join(
        [
            _CANVAS_PAGE.format(bg=1),
            "]",
            _CANVAS_PAGE.format(bg=2),
            "]",
            _CANVAS_PAGE.format(bg=3),
            "]",
        ]
    )

    assert canvas_page_pairs(source) == {1: 1, 2: 2, 3: 3}


def test_reflowed_pages_are_absent_rather_than_assumed_to_match() -> None:
    source = "\n".join(
        [
            _CANVAS_PAGE.format(bg=1),
            "]",
            _REFLOW_PAGE,
            "]",
            _CANVAS_PAGE.format(bg=3),
            "]",
        ]
    )

    assert canvas_page_pairs(source) == {1: 1, 3: 3}


def test_a_page_drawn_on_a_different_background_page_is_recorded_verbatim() -> None:
    source = "\n".join(
        [
            _CANVAS_PAGE.format(bg=41),
            "]",
            _CANVAS_PAGE.format(bg=42),
            "]",
        ]
    )

    assert canvas_page_pairs(source) == {1: 41, 2: 42}


def test_a_purely_reflowed_document_reports_no_map_at_all() -> None:
    source = "\n".join([_REFLOW_PAGE, "]", _REFLOW_PAGE, "]"])

    assert canvas_page_pairs(source) is None


def test_a_source_with_no_page_markers_is_not_a_canvas_document() -> None:
    assert canvas_page_pairs("#set text(size: 10pt)\nHello.") is None
    assert canvas_page_pairs("") is None


def test_the_identity_threshold_is_a_real_gap_not_a_tuned_constant() -> None:
    l1_floor, l2_ceiling = 0.9777, 0.1318

    assert l2_ceiling < CANVAS_PAIR_IDENTITY_MIN < l1_floor
    assert (CANVAS_PAIR_IDENTITY_MIN - l2_ceiling) > 3 * (l1_floor - CANVAS_PAIR_IDENTITY_MIN)


def test_a_blank_background_does_not_disqualify_a_blank_page(tmp_path: Path) -> None:
    from PIL import Image

    blank = tmp_path / "blank_bg.png"
    also_blank = tmp_path / "blank_final.png"
    Image.new("L", (50, 50), 255).save(blank)
    Image.new("L", (50, 50), 255).save(also_blank)

    assert canvas_page_pair_identity(blank, also_blank) == 1.0


def test_an_unreadable_pair_is_unmeasurable_not_unpaired(tmp_path: Path) -> None:
    from PIL import Image

    good = tmp_path / "good.png"
    Image.new("L", (50, 50), 0).save(good)
    junk = tmp_path / "junk.png"
    junk.write_bytes(b"not a png")

    assert canvas_page_pair_identity(junk, good) is None
    assert canvas_page_pair_identity(good, junk) is None


def test_a_size_mismatch_is_unmeasurable(tmp_path: Path) -> None:
    from PIL import Image

    small = tmp_path / "small.png"
    large = tmp_path / "large.png"
    Image.new("L", (50, 50), 0).save(small)
    Image.new("L", (100, 100), 0).save(large)

    assert canvas_page_pair_identity(small, large) is None
