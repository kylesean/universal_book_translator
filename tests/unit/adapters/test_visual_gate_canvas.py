"""Canvas ink collision: translated text landing on ink the stripper kept.

The facsimile channel draws translated text over a background that is the
source page with its body text removed. Everything the stripper kept — a table
rule, a figure, a grid — is still visible ink under the new text. No other tier
sees this: the geometry tier reads *text* rects, the type tier measures glyph
heights, ``blank_page``/``text_occluded`` ask whether text is MISSING, and the
VLM tier is sampled, opt-in and costs tokens.

These tests exist for one reason: **the gate must be provably able to fail.** A
detector nobody has ever seen fire is not a gate. The old pen-pad coefficient
(0.05) was proven inert the same way — a synthetic ruled page is drawn here
purely so the assertion "this page collides" can be made against a page that
really does collide.

Measured on the synthetic pages below (300x300, five 9px text bars):

    blank background          ratio 0.000  -> no finding
    one rule through text     ratio 0.033  -> finding (barely over the 3% budget)
    a table row (two rules)   ratio 0.067  -> finding
    a ruled grid              ratio 0.167  -> finding
    flat grey tint            ratio 0.000  -> no finding
    horizontal gradient       ratio 0.000  -> no finding

The tint and gradient rows are the important ones: they are why the detector
tests EDGES rather than "is this pixel dark". A darkness test would call every
figure fill a collision and fire on healthy books.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from ubt.adapters.pdf.visual_gate import (
    CANVAS_COLLISION_RATIO,
    CANVAS_MIN_TEXT_PIXELS,
    InkCollisionPage,
    canvas_ink_collision_findings,
    canvas_ink_collision_pages,
    canvas_ink_collision_ratio,
)

pytestmark = pytest.mark.fast

_SIZE = (300, 300)
#: Five text bars, 9px tall and 260px wide. Together they clear
#: ``CANVAS_MIN_TEXT_PIXELS`` by an order of magnitude, so a ``None`` return in
#: these tests always means "the check refused", never "the page was too empty".
_TEXT_BARS: tuple[tuple[int, int, int, int], ...] = tuple(
    (20, 40 + row * 20, 280, 49 + row * 20) for row in range(5)
)


def _draw_text(draw: ImageDraw.ImageDraw) -> None:
    for box in _TEXT_BARS:
        draw.rectangle(box, fill=0)


def _page_pair(
    tmp_path: Path,
    name: str,
    *,
    background: Callable[[ImageDraw.ImageDraw], None] | None = None,
    text: bool = True,
) -> tuple[Path, Path]:
    """Write a ``(background, final)`` PNG pair; the final is background + text.

    ``background`` is a *drawing callback*, not a path: each test paints the
    hazard it needs (a rule through the text, a filled bar, an edge) onto the
    background layer before the glyphs go on top.

    The final page is a copy of the background with the text drawn on top, which
    is structurally what the facsimile renderer produces: the same page, one
    layer of new glyphs over whatever the stripper left behind.
    """
    bg = Image.new("L", _SIZE, 255)
    if background is not None:
        background(ImageDraw.Draw(bg))
    final = bg.copy()
    if text:
        _draw_text(ImageDraw.Draw(final))
    bg_path = tmp_path / f"{name}_bg.png"
    final_path = tmp_path / f"{name}_final.png"
    bg.save(bg_path)
    final.save(final_path)
    return bg_path, final_path


# --------------------------------------------------------------------------- #
# The measurement
# --------------------------------------------------------------------------- #


def test_text_over_a_blank_background_does_not_collide(tmp_path: Path) -> None:
    """The negative control: nothing to collide with, so the ratio is exactly 0.

    Exactly, not "small": the final page is a copy of the background with glyphs
    added, so every changed pixel is a glyph pixel and no background edge exists
    to fall under one.
    """
    pair = _page_pair(tmp_path, "blank")

    measured = canvas_ink_collision_ratio(*pair)

    assert measured is not None
    ratio, text_pixels, collided = measured
    assert ratio == 0.0
    assert collided == 0
    assert text_pixels >= CANVAS_MIN_TEXT_PIXELS


def test_text_over_a_rule_the_stripper_kept_collides(tmp_path: Path) -> None:
    """**The test that proves the gate can fire.**

    Two horizontal rules — one table row's top and bottom boundary — drawn
    through two of the five text bars: the ordinary facsimile defect, where a
    table's rules were deliberately kept and a translated line's baseline drifted
    onto them. The text layer is perfect, the page count is 1:1, every other tier
    passes, and the reader gets lines struck through. Only this ratio sees it.

    Two rules rather than one on purpose: a single rule measures 0.033, barely
    over the 3% budget, and an assertion that close to its own threshold proves
    the detector *can* fire without proving it fires with margin on the defect
    the budget was chosen for.
    """

    def rule(draw: ImageDraw.ImageDraw) -> None:
        draw.line([(10, 45), (290, 45)], fill=0, width=2)
        draw.line([(10, 65), (290, 65)], fill=0, width=2)

    pair = _page_pair(tmp_path, "rule", background=rule)

    measured = canvas_ink_collision_ratio(*pair)

    assert measured is not None
    ratio, _text_pixels, collided = measured
    assert ratio > 2 * CANVAS_COLLISION_RATIO, (
        f"two table rules through two of five text lines measured {ratio:.3%}; the "
        "gate cannot see the defect it exists for"
    )
    assert collided > 0


def test_a_ruled_grid_collides_far_above_a_single_rule(tmp_path: Path) -> None:
    """Severity is monotonic in how much of the text sits on ink.

    A page where every line sits on a grid rule is worse than one line on one
    rule, and the number has to say so — otherwise a threshold can only ever be
    a binary alarm and the corpus baseline cannot rank pages.
    """

    def rule(draw: ImageDraw.ImageDraw) -> None:
        draw.line([(10, 45), (290, 45)], fill=0, width=2)
        draw.line([(10, 65), (290, 65)], fill=0, width=2)

    def grid(draw: ImageDraw.ImageDraw) -> None:
        for box in _TEXT_BARS:
            draw.line([(10, box[1] + 4), (290, box[1] + 4)], fill=0, width=1)

    one_rule = canvas_ink_collision_ratio(*_page_pair(tmp_path, "one", background=rule))
    ruled_grid = canvas_ink_collision_ratio(*_page_pair(tmp_path, "grid", background=grid))

    assert one_rule is not None and ruled_grid is not None
    assert ruled_grid[0] > one_rule[0]


@pytest.mark.parametrize("name", ["tint", "gradient"])
def test_flat_and_smooth_backgrounds_do_not_collide(tmp_path: Path, name: str) -> None:
    """Why the detector tests edges and not darkness.

    A full-page grey tint and a horizontal grey wash both read as "lots of
    non-white pixels" to a darkness test, and both are ordinary on a real page
    (a shaded table header, a figure's panel background). ``FIND_EDGES``
    responds to the gradient, not the level, so neither contributes a single
    collision. Measured: 0.000 for both.
    """

    def tint(draw: ImageDraw.ImageDraw) -> None:
        draw.rectangle([0, 0, _SIZE[0] - 1, _SIZE[1] - 1], fill=200)

    def gradient(draw: ImageDraw.ImageDraw) -> None:
        for x in range(_SIZE[0]):
            draw.line([(x, 0), (x, _SIZE[1] - 1)], fill=int(120 + 100 * x / (_SIZE[0] - 1)))

    painter = tint if name == "tint" else gradient
    measured = canvas_ink_collision_ratio(*_page_pair(tmp_path, name, background=painter))

    assert measured is not None
    assert measured[0] == 0.0, (
        f"a {name} background contributed collisions; the detector is measuring darkness, not edges"
    )


def test_a_size_mismatch_is_not_measured_rather_than_resampled(tmp_path: Path) -> None:
    """A 144-dpi artifact against a 72-dpi background must not be compared.

    Resampling one onto the other attributes every interpolation artefact to the
    text layer, which turns a healthy page into a reported collision. ``None``
    says "not measured"; 0.0 would claim the page was checked and clean.
    """
    bg_path, final_path = _page_pair(tmp_path, "mismatch")
    with Image.open(final_path) as img:
        img.resize((600, 600)).save(final_path)

    assert canvas_ink_collision_ratio(bg_path, final_path) is None


def test_a_page_with_too_little_text_is_not_measured(tmp_path: Path) -> None:
    """A near-blank figure page must not produce a ratio over 40 pixels.

    Same reasoning as ``TYPE_AUDIT_MIN_GLYPHS``. A single caption word on an
    otherwise empty page would otherwise report 0% or 100% depending on one
    stray pixel.
    """

    def small(draw: ImageDraw.ImageDraw) -> None:
        draw.rectangle([20, 20, 34, 26], fill=0)  # 15x7 px, well under the floor

    pair = _page_pair(tmp_path, "tiny", text=False)
    bg_path, final_path = pair
    with Image.open(bg_path) as img:
        bg = img.copy()
    final = bg.copy()
    small(ImageDraw.Draw(final))
    final.save(final_path)

    assert canvas_ink_collision_ratio(bg_path, final_path) is None


def test_an_unreadable_pair_is_not_measured(tmp_path: Path) -> None:
    """Never raise, never report 0.0 for a page that was not looked at."""
    junk = tmp_path / "junk.png"
    junk.write_bytes(b"not a png")
    bg_path, _ = _page_pair(tmp_path, "real")

    assert canvas_ink_collision_ratio(junk, bg_path) is None
    assert canvas_ink_collision_ratio(bg_path, junk) is None


# --------------------------------------------------------------------------- #
# The findings
# --------------------------------------------------------------------------- #


def test_a_colliding_page_produces_a_major_finding_naming_the_page(tmp_path: Path) -> None:
    """The message has to be actionable from the report alone.

    It carries the ratio and both counts, so a reader can tell "1.1% over the
    line" from "half the page is unreadable" without re-running the gate.
    """

    def rule(draw: ImageDraw.ImageDraw) -> None:
        draw.line([(10, 45), (290, 45)], fill=0, width=2)
        draw.line([(10, 65), (290, 65)], fill=0, width=2)

    bg_path, final_path = _page_pair(tmp_path, "fire", background=rule)

    findings = canvas_ink_collision_findings({7: bg_path}, {7: final_path})

    assert len(findings) == 1
    finding = findings[0]
    assert finding.code == "canvas_ink_collision"
    # MAJOR, not CRITICAL: a legibility defect in a shipping artefact, exactly
    # like ``text_occluded`` and ``block_overlap``, and like both of those it is
    # warn-only unless the opt-in blocking gate is on.
    assert finding.severity == "major"
    assert finding.page == 7
    assert "7" in finding.message
    assert "px" in finding.message


def test_a_clean_page_produces_no_finding(tmp_path: Path) -> None:
    bg_path, final_path = _page_pair(tmp_path, "clean")

    assert canvas_ink_collision_findings({1: bg_path}, {1: final_path}) == []


def test_pages_missing_from_one_side_are_skipped_not_invented(tmp_path: Path) -> None:
    """A background that rendered fewer pages must not raise on the gap."""
    bg_path, final_path = _page_pair(tmp_path, "partial")

    measured = canvas_ink_collision_pages({1: bg_path}, {1: final_path, 2: final_path})
    assert [page.page for page in measured] == [1]

    assert canvas_ink_collision_findings({1: bg_path}, {2: final_path}) == []


def test_every_measured_page_is_kept_even_when_it_is_under_the_threshold(
    tmp_path: Path,
) -> None:
    """The warn-to-fail decision needs the numbers under the line too.

    A report that keeps only the pages over the threshold cannot answer "would a
    different threshold have fired", which is the entire question behind
    promoting this code to fail-closed. The 5-document corpus baseline is read
    out of ``stats``, so a sub-threshold page that measured 0.029 must survive.
    """
    bg_path, final_path = _page_pair(tmp_path, "under")

    measured = canvas_ink_collision_pages({3: bg_path}, {3: final_path})

    assert measured == [
        InkCollisionPage(page=3, ratio=0.0, text_pixels=measured[0].text_pixels, collided_pixels=0)
    ]
    assert isinstance(measured[0], InkCollisionPage)


def test_a_lower_threshold_fires_on_a_page_the_default_allows(tmp_path: Path) -> None:
    """The threshold is a parameter, not a law of nature.

    Promotion to fail-closed is a threshold change plus a corpus run, so the
    threshold must be provably effective below the shipped default — otherwise
    the promotion is untestable until it is already shipping.
    """

    def faint_rule(draw: ImageDraw.ImageDraw) -> None:
        draw.line([(10, 45), (290, 45)], fill=0, width=1)

    bg_path, final_path = _page_pair(tmp_path, "thresh", background=faint_rule)
    measured = canvas_ink_collision_ratio(bg_path, final_path)
    assert measured is not None
    ratio = measured[0]

    strict = canvas_ink_collision_findings(
        {1: bg_path}, {1: final_path}, ratio_threshold=ratio - 0.001
    )
    assert len(strict) == 1
