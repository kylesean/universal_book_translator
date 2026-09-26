"""Unit tests for CJK width metrics (fontTools advance sums)."""

from types import SimpleNamespace
from typing import Any

import pytest

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


def _r0918b_face(font: Any) -> str:
    return str(font["name"].getDebugName(4))


def test_width_metrics_face_follows_the_render_font(monkeypatch: pytest.MonkeyPatch) -> None:
    """``load_width_font`` used to raise unless the TTC held one exact face.

    ``UBT_CJK_FONT`` pointing at a Serif collection -- a path
    ``resolve_cjk_ttc`` itself recommends and ``ubt doctor`` reports as OK --
    aborted the whole anchored render over a *measurement* font, and a
    configured ``font_family`` was measured with Sans widths anyway.
    """
    import fontTools.ttLib as ttlib

    from ubt.adapters.pdf.font_metrics import load_width_font

    def install(*names: str) -> None:
        class _Collection:
            def __init__(self, _path: str) -> None:
                # fontTools fonts are dict-like: the loader reads font["name"].
                self.fonts = [
                    {"name": SimpleNamespace(getDebugName=lambda _i, n=n: n)} for n in names
                ]

        monkeypatch.setattr(ttlib, "TTCollection", _Collection)

        # The requested family is the one the page renders in, so measure it.

    install("Noto Serif CJK SC", "Noto Sans CJK TC")
    assert _r0918b_face(load_width_font("x.ttc", "Noto Serif CJK SC")) == "Noto Serif CJK SC"
    # Nothing resembles the request: measure the first face instead of failing.
    assert _r0918b_face(load_width_font("x.ttc", "Kaiti SC")) == "Noto Serif CJK SC"
    assert _r0918b_face(load_width_font("x.ttc")) == "Noto Serif CJK SC"
    # Packaging renames still resolve ("NotoSerifCJKsc-Regular" is a Serif face).
    install("NotoSerifCJKsc-Regular")
    assert _r0918b_face(load_width_font("x.ttc", "Noto Serif CJK SC")) == "NotoSerifCJKsc-Regular"
    install()
    with pytest.raises(Exception, match="No font face inside"):
        load_width_font("x.ttc")
