"""The canvas page map: which artifact pages were drawn on the background.

The facsimile channel is a **fusion**, and this is the seam where that matters.
An L1 page embeds its own stripped background as a Typst ``image(...)``; an L2
page is reflowed from the IR and embeds nothing. The gate cannot measure an L2
page against a background it never drew on, and before this map existed it did
exactly that: on a real 26-page book (19 L1, 7 L2) five pages were reported at
36–59% "ink collision" purely because a reflowed page of Chinese prose was being
compared with a page of the source's formulas.

Every test here is about the *map*, not the ratio — ``test_visual_gate_canvas.py``
pins the ratio and ``test_visual_gate_canvas_pdf.py`` pins the end-to-end wiring.
"""

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
    """The ordinary L1 facsimile: three pages, each over its own background page."""
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
    """An L2 page must not appear in the map, and must not shift the others.

    This is the exact fusion case: pages 1 and 3 are on the canvas, page 2 was
    reflowed. A naive implementation that pairs by position would map artifact
    page 3 to background page 3 by index rather than by its own ``page:`` field,
    which happens to agree here — so the assertion that matters is that page 2
    is *missing*, not that 3 is present.
    """
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
    """The map is artifact page -> background page, not an identity function.

    A resumable job may render a range whose first page is not background page 1.
    Reading the ``page:`` argument rather than assuming ``artifact == background``
    is what keeps the gate correct there, and it costs nothing: the renderer
    already wrote the number down.
    """
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
    """Not ``{}``: an empty map would read as "a canvas book with no canvas".

    ``None`` is the signal that the channel does not apply, which the gate turns
    into a printed skip note. An empty dict would make a reflow delivery
    indistinguishable from a facsimile run whose every page happened to solve to
    L2 — and the second is a real regression in the renderer while the first is
    simply a different product.
    """
    source = "\n".join([_REFLOW_PAGE, "]", _REFLOW_PAGE, "]"])

    assert canvas_page_pairs(source) is None


def test_a_source_with_no_page_markers_is_not_a_canvas_document() -> None:
    """A truncated or unrelated ``.typ`` must not be read as a canvas map."""
    assert canvas_page_pairs("#set text(size: 10pt)\nHello.") is None
    assert canvas_page_pairs("") is None


def test_the_identity_threshold_is_a_real_gap_not_a_tuned_constant() -> None:
    """Measured on a real fusion book; the guard must sit inside the gap.

    ``chapter-3.pdf``: 19 L1 pages scored 0.9777–0.9999, 7 L2 pages scored
    0.0000–0.1318. Any threshold in (0.14, 0.97) separates them; 0.85 is chosen
    because it is several times further from the L1 floor than from the L2
    ceiling, and the L1 side is the one whose false negatives would silently
    disable the tier.
    """
    l1_floor, l2_ceiling = 0.9777, 0.1318

    assert l2_ceiling < CANVAS_PAIR_IDENTITY_MIN < l1_floor
    assert (CANVAS_PAIR_IDENTITY_MIN - l2_ceiling) > 3 * (l1_floor - CANVAS_PAIR_IDENTITY_MIN)


def test_a_blank_background_does_not_disqualify_a_blank_page(tmp_path: Path) -> None:
    """A page with nothing on it is vacuously the same page, and cannot collide.

    0/0 is the one case the identity metric cannot answer, and answering it the
    wrong way is expensive in both directions: ``0.0`` would mark every genuinely
    blank page as unpaired ("the renderer dropped this page"), while raising would
    lose the measurement for the rest of the book.
    """
    from PIL import Image

    blank = tmp_path / "blank_bg.png"
    also_blank = tmp_path / "blank_final.png"
    Image.new("L", (50, 50), 255).save(blank)
    Image.new("L", (50, 50), 255).save(also_blank)

    assert canvas_page_pair_identity(blank, also_blank) == 1.0


def test_an_unreadable_pair_is_unmeasurable_not_unpaired(tmp_path: Path) -> None:
    """``None`` and ``0.0`` mean different things and the gate treats them alike."""
    from PIL import Image

    good = tmp_path / "good.png"
    Image.new("L", (50, 50), 0).save(good)
    junk = tmp_path / "junk.png"
    junk.write_bytes(b"not a png")

    assert canvas_page_pair_identity(junk, good) is None
    assert canvas_page_pair_identity(good, junk) is None


def test_a_size_mismatch_is_unmeasurable(tmp_path: Path) -> None:
    """Same rule as the ratio: never resample one page onto the other."""
    from PIL import Image

    small = tmp_path / "small.png"
    large = tmp_path / "large.png"
    Image.new("L", (50, 50), 0).save(small)
    Image.new("L", (100, 100), 0).save(large)

    assert canvas_page_pair_identity(small, large) is None
