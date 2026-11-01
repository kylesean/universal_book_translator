from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from ubt.adapters.pdf.visual_gate import (
    CANVAS_COLLISION_RATIO,
    CANVAS_MIN_TEXT_PIXELS,
    InkCollisionPage,
    _collision_findings,
    canvas_ink_collision_pages,
    canvas_ink_collision_ratio,
)

pytestmark = pytest.mark.fast

_SIZE = (300, 300)
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


def test_text_over_a_blank_background_does_not_collide(tmp_path: Path) -> None:
    pair = _page_pair(tmp_path, "blank")

    measured = canvas_ink_collision_ratio(*pair)

    assert measured is not None
    ratio, text_pixels, collided = measured
    assert ratio == 0.0
    assert collided == 0
    assert text_pixels >= CANVAS_MIN_TEXT_PIXELS


def test_text_over_a_rule_the_stripper_kept_collides(tmp_path: Path) -> None:

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
    bg_path, final_path = _page_pair(tmp_path, "mismatch")
    with Image.open(final_path) as img:
        img.resize((600, 600)).save(final_path)

    assert canvas_ink_collision_ratio(bg_path, final_path) is None


def test_a_page_with_too_little_text_is_not_measured(tmp_path: Path) -> None:

    def small(draw: ImageDraw.ImageDraw) -> None:
        draw.rectangle([20, 20, 34, 26], fill=0)

    pair = _page_pair(tmp_path, "tiny", text=False)
    bg_path, final_path = pair
    with Image.open(bg_path) as img:
        bg = img.copy()
    final = bg.copy()
    small(ImageDraw.Draw(final))
    final.save(final_path)

    assert canvas_ink_collision_ratio(bg_path, final_path) is None


def test_an_unreadable_pair_is_not_measured(tmp_path: Path) -> None:
    junk = tmp_path / "junk.png"
    junk.write_bytes(b"not a png")
    bg_path, _ = _page_pair(tmp_path, "real")

    assert canvas_ink_collision_ratio(junk, bg_path) is None
    assert canvas_ink_collision_ratio(bg_path, junk) is None


def test_a_colliding_page_produces_a_major_finding_naming_the_page(tmp_path: Path) -> None:

    def rule(draw: ImageDraw.ImageDraw) -> None:
        draw.line([(10, 45), (290, 45)], fill=0, width=2)
        draw.line([(10, 65), (290, 65)], fill=0, width=2)

    bg_path, final_path = _page_pair(tmp_path, "fire", background=rule)

    findings = _collision_findings(canvas_ink_collision_pages({7: bg_path}, {7: final_path}))

    assert len(findings) == 1
    finding = findings[0]
    assert finding.code == "canvas_ink_collision"
    assert finding.severity == "major"
    assert finding.page == 7
    assert "7" in finding.message
    assert "px" in finding.message


def test_a_clean_page_produces_no_finding(tmp_path: Path) -> None:
    bg_path, final_path = _page_pair(tmp_path, "clean")

    assert _collision_findings(canvas_ink_collision_pages({1: bg_path}, {1: final_path})) == []


def test_pages_missing_from_one_side_are_skipped_not_invented(tmp_path: Path) -> None:
    bg_path, final_path = _page_pair(tmp_path, "partial")

    measured = canvas_ink_collision_pages({1: bg_path}, {1: final_path, 2: final_path})
    assert [page.page for page in measured] == [1]

    assert _collision_findings(canvas_ink_collision_pages({1: bg_path}, {2: final_path})) == []


def test_every_measured_page_is_kept_even_when_it_is_under_the_threshold(
    tmp_path: Path,
) -> None:
    bg_path, final_path = _page_pair(tmp_path, "under")

    measured = canvas_ink_collision_pages({3: bg_path}, {3: final_path})

    assert measured == [
        InkCollisionPage(page=3, ratio=0.0, text_pixels=measured[0].text_pixels, collided_pixels=0)
    ]
    assert isinstance(measured[0], InkCollisionPage)


def test_a_lower_threshold_fires_on_a_page_the_default_allows(tmp_path: Path) -> None:

    def faint_rule(draw: ImageDraw.ImageDraw) -> None:
        draw.line([(10, 45), (290, 45)], fill=0, width=1)

    bg_path, final_path = _page_pair(tmp_path, "thresh", background=faint_rule)
    measured = canvas_ink_collision_ratio(bg_path, final_path)
    assert measured is not None
    ratio = measured[0]

    strict = _collision_findings(
        canvas_ink_collision_pages({1: bg_path}, {1: final_path}),
        ratio_threshold=ratio - 0.001,
    )
    assert len(strict) == 1


def test_a_region_collision_produces_a_finding_naming_the_block() -> None:
    page = InkCollisionPage(
        page=2,
        ratio=0.005,
        text_pixels=150_000,
        collided_pixels=750,
        worst_region_ratio=0.465,
        worst_region=(317.0, 308.0, 244.0, 39.0),
    )
    findings = _collision_findings([page])
    assert [f.code for f in findings] == ["canvas_ink_collision"]
    assert findings[0].severity == "major"
    assert "Page 2" in findings[0].message
    assert "(317, 308)" in findings[0].message
    assert "page-global" in findings[0].message


def test_a_page_over_the_global_budget_fires_on_the_page_message() -> None:
    page = InkCollisionPage(
        page=16,
        ratio=0.489,
        text_pixels=21_174,
        collided_pixels=10_346,
        worst_region_ratio=0.0,
        worst_region=None,
    )
    findings = _collision_findings([page])
    assert [f.code for f in findings] == ["canvas_ink_collision"]
    assert "48.9% of the translated text" in findings[0].message


def test_a_clean_region_never_fires() -> None:
    page = InkCollisionPage(
        page=5,
        ratio=0.001,
        text_pixels=100_000,
        collided_pixels=100,
        worst_region_ratio=0.054,
        worst_region=(59.0, 92.0, 230.0, 82.0),
    )
    assert _collision_findings([page]) == []


def test_keep_verbatim_pages_are_excluded_from_the_canvas_measurement() -> None:
    from types import SimpleNamespace

    from ubt.adapters.pdf.visual_gate import translated_canvas_pages

    def _block(page: int | None, *, target: str, source: str, skip: bool = False) -> object:
        return SimpleNamespace(
            bbox=SimpleNamespace(page=page),
            target_text=target,
            source_text=source,
            skip_translate=skip,
        )

    blocks = [
        _block(2, target="译文", source="source text"),
        _block(15, target="source text", source="source text"),
        _block(16, target="", source="source text"),
        _block(17, target="source text", source="source text", skip=True),
        _block(17, target="第 17 页的译文", source="page 17 source"),
        _block(None, target="没有页码的译文", source="x"),
    ]
    assert translated_canvas_pages(blocks) == {2, 17}
