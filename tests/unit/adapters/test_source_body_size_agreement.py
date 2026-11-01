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
    return SimpleNamespace(text=text, font_size=size, rect=(36.0, 500.0, 500.0, 500.0 + height))


class TestAgreementFilter:
    def test_self_consistent_lines_win_the_mode(self) -> None:
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
        lines = [_line("只有公式", size=5.0, height=9.0) for _ in range(4)]
        assert _body_size_from_reported(lines) is None
        got = source_body_size(lines)
        assert 8.0 <= got <= 10.5, got

    def test_a_page_with_no_lines_keeps_the_default(self) -> None:
        assert source_body_size([]) == pytest.approx(9.0)

    @pytest.mark.parametrize("size,height", [(9.0, 9.0), (9.0, 8.0), (9.0, 10.2)])
    def test_ratio_band_is_the_documented_one(self, size: float, height: float) -> None:
        ratio = height / size
        in_band = MIN_INK_TO_SIZE_RATIO <= ratio <= MAX_INK_TO_SIZE_RATIO
        lines = [_line("x", size=size, height=height) for _ in range(6)]
        voted = _body_size_from_reported(lines) is not None
        assert voted == in_band, (size, height, ratio)

    def test_a_placeholder_size_still_excluded(self) -> None:
        lines = [_line("任意", size=1.0, height=8.9) for _ in range(20)]
        assert _body_size_from_reported(lines) is None
        assert source_body_size(lines) > 7.5
