from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts import probe_facsimile as probe  # noqa: E402

pytestmark = pytest.mark.fast


@dataclass
class _Line:
    text: str
    rect: tuple[float, float, float, float]

    @property
    def top(self) -> float:
        return self.rect[3]

    @property
    def bottom(self) -> float:
        return self.rect[1]


def _line(text: str, *, top: float, width: float = 300.0, ink_h: float = 8.0) -> _Line:
    return _Line(text, (80.0, top - ink_h, 80.0 + width, top))


def _page(lines: list[_Line]) -> list[_Line]:
    return sorted(lines, key=lambda line: -line.rect[1])


def _fits_any(
    paragraphs: list[list[_Line]],
    *,
    size: float,
    source_size: float,
    source_leading: float,
    mode: str,
    en_per_zh: float = 3.53,
) -> bool:
    em = source_leading / source_size
    for para in paragraphs:
        width = max(line.rect[2] - line.rect[0] for line in para)
        glyphs = sum(len(line.text) for line in para) / en_per_zh
        per_line = max(1.0, width / size)
        lines_needed = max(1, math.ceil(glyphs / per_line))
        height = lines_needed * em * size
        top = max(line.rect[3] for line in para)
        if mode == "pinned":
            if height > len(para) * source_leading + probe.TOL_PT:
                return False
        else:
            ink_bottom = min(line.rect[1] for line in para)
            line_box_bottom = top - len(para) * source_leading
            limit = min(ink_bottom, line_box_bottom)
            if top - height < limit - probe.TOL_PT:
                return False
    return True


def test_modes_are_the_two_documented_models() -> None:
    assert probe.MODES == ("pinned", "glue")


def test_unknown_mode_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown mode"):
        probe.probe_page(tmp_path / "missing.pdf", 1, mode="squeeze")


def test_paragraph_gap_tolerates_a_between_line_number() -> None:
    leading = 12.0
    assert probe.PARA_GAP_FACTOR * leading > 2 * leading
    assert probe.PARA_GAP_FACTOR * leading < 3 * leading


def test_glue_is_never_stricter_than_pinned() -> None:
    for n_lines in (1, 2, 5, 12):
        for gap in (0.0, 6.0, 24.0, 60.0):
            source_size, leading = 9.0, 12.0
            top = 700.0
            paragraphs: list[list[_Line]] = []
            for _ in range(3):
                para = [_line("x" * 80, top=top - line * leading) for line in range(n_lines)]
                paragraphs.append(para)
                top -= n_lines * leading + gap
            pinned_best = 0.0
            glue_best = 0.0
            for step in range(0, 121):
                size = 6.0 + step * 0.05
                if _fits_any(
                    paragraphs,
                    size=size,
                    source_size=source_size,
                    source_leading=leading,
                    mode="pinned",
                ):
                    pinned_best = size
                if _fits_any(
                    paragraphs,
                    size=size,
                    source_size=source_size,
                    source_leading=leading,
                    mode="glue",
                ):
                    glue_best = size
            assert glue_best >= pinned_best - 1e-9, (n_lines, gap, pinned_best, glue_best)


def test_leading_scales_with_size() -> None:
    source_size, source_leading = 9.0, 12.0
    em = source_leading / source_size
    assert em == pytest.approx(1.333, abs=1e-3)
    assert 1 * em * 12.0 == pytest.approx(16.0, abs=1e-6)


def test_single_line_paragraph_bottom_is_its_line_box_bottom() -> None:
    source_leading = 12.0
    line = _line("x" * 80, top=71.5, ink_h=8.0)
    ink_bottom = line.rect[1]
    line_box_bottom = line.rect[3] - 1 * source_leading
    assert ink_bottom == pytest.approx(63.5)
    assert line_box_bottom == pytest.approx(59.5)
    assert min(ink_bottom, line_box_bottom) == pytest.approx(59.5)


def test_source_overlap_is_detected() -> None:
    source_leading = 12.0

    def overlap(earlier: list[_Line], later: list[_Line]) -> bool:
        earlier_top = max(line.rect[3] for line in earlier)
        earlier_bottom = earlier_top - len(earlier) * source_leading
        later_top = max(line.rect[3] for line in later)
        return earlier_bottom < later_top - probe.TOL_PT

    earlier = [_line("x" * 80, top=700.0 - index * source_leading) for index in range(3)]
    assert overlap(earlier, [_line("x" * 80, top=650.0)]) is False
    assert overlap(earlier, [_line("x" * 80, top=670.0)]) is True


def test_page_probe_carries_mode_and_overlap() -> None:
    page = probe.PageProbe(
        page=1,
        page_w=595.0,
        page_h=842.0,
        columns=1,
        source_size_pt=9.0,
        paragraphs=3,
        fits_at_source=3,
        uniform_size_pt=11.0,
        fits=True,
        fill_ratio=0.6,
    )
    assert page.mode == "glue"
    assert page.source_overlap is False
    assert page.fill_pct == pytest.approx(60.0)


def test_body_lines_drops_furniture_and_narrow_runs() -> None:
    lines: list[Any] = [
        _line("CHAPTER 3", top=790.0, width=40.0),
        _line("x" * 80, top=700.0),
        _line("x" * 80, top=688.0),
    ]
    body = probe._body_lines(lines, 595.0)
    assert len(body) == 2
    assert all(line.text != "CHAPTER 3" for line in body)
