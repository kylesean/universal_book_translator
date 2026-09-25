"""Unit tests for CJK width metrics (fontTools advance sums)."""

from typing import Any

from ubt.adapters.pdf.font_metrics import text_width_pt


class _Head:
    unitsPerEm = 1000


class _StubFont:
    def __init__(self) -> None:
        self.getbest_calls = 0
        self._head = _Head()
        self._hmtx: dict[str, tuple[int, int]] = {".notdef": (500, 0), "a": (600, 0)}

    def getBestCmap(self) -> dict[int, str]:
        self.getbest_calls += 1
        return {ord("a"): "a"}

    def __getitem__(self, key: str) -> Any:
        return self._head if key == "head" else self._hmtx


def test_text_width_pt_caches_cmap_per_font() -> None:
    font = _StubFont()
    assert text_width_pt(font, "aa", 10.0) == 12.0
    assert text_width_pt(font, "aaa", 10.0) == 18.0
    # Rebuilding the cmap per call dominated the anchored fitter; build once.
    assert font.getbest_calls == 1


def test_text_width_pt_falls_back_to_notdef() -> None:
    font = _StubFont()
    assert text_width_pt(font, "z", 10.0) == 5.0
