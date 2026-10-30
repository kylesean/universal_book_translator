"""Body-size measurement must reject lines whose ink disagrees with their size.

``solve_page_2d``'s ceiling is ``source_size * 1.10`` and its floor is
:data:`~ubt.render.layout_solver.FLOOR_PT` (7.5pt). So a page measured below
6.82pt has ``ceiling < floor``: no size satisfies its own budget, and
``docling_render`` routes it to L2. Measured on the reference chapter
(``chapter-3.pdf`` page 22) that misrouting was caused by the *measurement*,
not by the page: pdfium reported 6.2pt for equation lines that ink an 8.0–9.3pt
box, and those lines outnumbered the body lines, so the mode landed on 6.2pt on
a page whose text is unmistakably 8.9–9pt.

A body face is self-consistent by construction — its ink box is its em — so a
line whose reported size and drawn height disagree is a math glyph, a fraction,
or a rotated label, and must not vote in the body's mode.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ubt.adapters.pdf.source_metrics import (
    MAX_INK_TO_SIZE_RATIO,
    MIN_INK_TO_SIZE_RATIO,
    _body_size_from_reported,
    source_body_size,
)

pytestmark = pytest.mark.fast


def _line(text: str, *, size: float, height: float) -> SimpleNamespace:
    """A line in the shape ``textgeom`` emits: ``rect`` is (x0, y0, x1, y1)."""
    return SimpleNamespace(text=text, font_size=size, rect=(36.0, 500.0, 500.0, 500.0 + height))


class TestAgreementFilter:
    def test_self_consistent_lines_win_the_mode(self) -> None:
        """The measured page-22 shape: 5 math lines at 6.2, 3 body lines at 9.0.

        Plain mode picks 6.2 (the math) and the page falls out of the facsimile
        channel for a capacity ceiling that nothing can meet.
        """
        lines = [
            _line("secg0 = cosg0", size=6.2, height=8.0),
            _line("secg0sq = secg0", size=6.2, height=8.0),
            _line("T3 = ...", size=6.2, height=9.3),
            _line("T4 = ...", size=6.2, height=9.0),
            _line("T5 = ...", size=6.2, height=9.0),
            _line("1. Compute:", size=9.0, height=9.0),
            _line("2. Compute:", size=9.0, height=9.0),
            _line("4. Compute derivatives:", size=8.9, height=9.4),
        ]
        assert source_body_size(lines) == pytest.approx(9.0, abs=0.11)

    def test_the_reported_size_is_preferred_when_it_agrees(self) -> None:
        lines = [_line(f"正文 {i}", size=8.9, height=8.9) for i in range(20)]
        assert source_body_size(lines) == pytest.approx(8.9)

    def test_a_headline_does_not_become_the_body(self) -> None:
        lines = [_line("正文", size=8.9, height=9.0) for _ in range(10)]
        lines.append(_line("标题", size=16.2, height=17.3))
        assert source_body_size(lines) == pytest.approx(8.9)

    def test_all_lines_disagreeing_falls_back_to_ink_height(self) -> None:
        """No consistent size on the page -> measure the ink, scaled by the ratio."""
        lines = [_line("只有公式", size=5.0, height=9.0) for _ in range(4)]
        assert _body_size_from_reported(lines) is None
        got = source_body_size(lines)
        assert 8.0 <= got <= 10.5, got

    def test_a_page_with_no_lines_keeps_the_default(self) -> None:
        assert source_body_size([]) == pytest.approx(9.0)

    @pytest.mark.parametrize("size,height", [(9.0, 9.0), (9.0, 8.0), (9.0, 10.2)])
    def test_ratio_band_is_the_documented_one(self, size: float, height: float) -> None:
        """The band's interior defines which lines get a vote; pin them together.

        Values are kept off the exact edges: ``10.35 / 9.0`` is
        ``1.1500000000000001`` in binary floating point, so an edge case here
        tests the platform's rounding rather than the band.
        """
        ratio = height / size
        in_band = MIN_INK_TO_SIZE_RATIO <= ratio <= MAX_INK_TO_SIZE_RATIO
        lines = [_line("x", size=size, height=height) for _ in range(6)]
        voted = _body_size_from_reported(lines) is not None
        assert voted == in_band, (size, height, ratio)

    def test_a_placeholder_size_still_excluded(self) -> None:
        """A producer reporting ``font_size=1.0`` for every glyph is unusable."""
        lines = [_line("任意", size=1.0, height=8.9) for _ in range(20)]
        assert _body_size_from_reported(lines) is None
        # The ink fallback is what ships the size, and it is plausible.
        assert source_body_size(lines) > 7.5
