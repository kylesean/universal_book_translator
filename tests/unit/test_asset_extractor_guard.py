"""Guard tests: one malformed page must not lose the whole book's figures.

extract_pdf_figures walks every page in one function; a raise from the
caption-anchor search or page.get_objects() used to propagate out to the
caller, which logged at debug and discarded figures already extracted from
*every other* page — and skipped closing the document and the in-flight page
handles (a native pdfium leak). The page body is now individually guarded and
the document close is in a finally.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from ubt.adapters.pdf import asset_extractor


class _FakeBitmap:
    def to_pil(self) -> Any:
        class _Img:
            size = (10, 10)

            def save(self, path: Path) -> None:
                Path(path).write_bytes(b"png")

        return _Img()

    def close(self) -> None:  # noqa: D401 - native-handle stand-in
        return None


class _FakeSearch:
    def __init__(self, hits: list[tuple[int, int]]) -> None:
        self._hits = hits
        self._i = 0

    def get_next(self) -> tuple[int, int] | None:
        if self._i < len(self._hits):
            hit = self._hits[self._i]
            self._i += 1
            return hit
        return None


class _FakeTextPage:
    def __init__(self, text: str, *, raise_on_search: bool = False) -> None:
        self._text = text
        self.closed = False
        self._raise = raise_on_search

    def get_text_range(self) -> str:
        return self._text

    def search(self, _q: str) -> _FakeSearch:
        if self._raise:
            raise RuntimeError("corrupt textpage")
        return _FakeSearch([(0, 5)])

    def count_rects(self, _start: int, _count: int) -> int:
        return 1

    def get_rect(self, _i: int) -> tuple[float, float, float, float]:
        return (0.0, 700.0, 100.0, 720.0)

    def close(self) -> None:
        self.closed = True


class _FakeObj:
    def __init__(self, kind: int) -> None:
        self.type = kind

    def get_bounds(self) -> tuple[float, float, float, float]:
        return (50.0, 300.0, 200.0, 400.0)


class _FakePage:
    def __init__(
        self,
        text: str,
        *,
        raw: Any,
        raise_on_search: bool = False,
        raise_on_objects: bool = False,
    ) -> None:
        self._tp = _FakeTextPage(text, raise_on_search=raise_on_search)
        self._raise_obj = raise_on_objects
        self._raw = raw
        self.closed = False

    def get_textpage(self) -> _FakeTextPage:
        return self._tp

    def get_size(self) -> tuple[float, float]:
        return (612.0, 792.0)

    def get_objects(self) -> list[_FakeObj]:
        if self._raise_obj:
            raise RuntimeError("corrupt page tree")
        return [_FakeObj(self._raw.FPDF_PAGEOBJ_IMAGE)]

    def render(self, scale: float = 1.0, crop: Any = None) -> _FakeBitmap:  # noqa: ARG002
        return _FakeBitmap()

    def close(self) -> None:
        self.closed = True


class _FakeDoc:
    def __init__(self, pages: list[_FakePage]) -> None:
        self._pages = pages
        self.closed = False

    def __iter__(self) -> Any:
        return iter(self._pages)

    def close(self) -> None:
        self.closed = True


def _install_fake_pdfium(
    monkeypatch: pytest.MonkeyPatch, pages: list[_FakePage], tmp_path: Path
) -> _FakeDoc:
    doc = _FakeDoc(pages)

    fake_raw = types.SimpleNamespace(FPDF_PAGEOBJ_IMAGE=3, FPDF_PAGEOBJ_PATH=4)
    fake_mod = types.ModuleType("pypdfium2")

    def _PdfDocument(_path: str) -> _FakeDoc:
        return doc

    fake_mod.PdfDocument = _PdfDocument  # type: ignore[attr-defined]
    fake_mod.raw = fake_raw  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pypdfium2", fake_mod)
    (tmp_path / "assets").mkdir(exist_ok=True)
    return doc


def _good_page(text: str, raw: Any) -> _FakePage:
    return _FakePage(text, raw=raw)


def test_malformed_page_skipped_others_survive_and_doc_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = types.SimpleNamespace(FPDF_PAGEOBJ_IMAGE=3, FPDF_PAGEOBJ_PATH=4)
    pages = [
        _good_page("FIG. 1.1 First diagram", raw),
        _FakePage("FIG. 2.1 Second diagram", raw=raw, raise_on_objects=True),
        _good_page("FIG. 3.1 Third diagram", raw=raw),
    ]
    doc = _install_fake_pdfium(monkeypatch, pages, tmp_path)
    book = tmp_path / "book.pdf"
    book.write_bytes(b"%PDF-1.4\n")

    result = asset_extractor.extract_pdf_figures(book, tmp_path / "assets")

    # The bad page is skipped; the two good pages still yield figures.
    assert {"1.1_p1", "3.1_p3"} <= set(result), result
    assert "2.1_p2" not in result
    # And the document is closed even though one page raised mid-loop.
    assert doc.closed is True


def test_search_error_page_also_survives(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = types.SimpleNamespace(FPDF_PAGEOBJ_IMAGE=3, FPDF_PAGEOBJ_PATH=4)
    pages = [
        _FakePage("FIG. 1.1 boom page", raw=raw, raise_on_search=True),
        _good_page("FIG. 2.1 good page", raw=raw),
    ]
    doc = _install_fake_pdfium(monkeypatch, pages, tmp_path)
    book = tmp_path / "book.pdf"
    book.write_bytes(b"%PDF-1.4\n")

    result = asset_extractor.extract_pdf_figures(book, tmp_path / "assets")

    assert "1.1_p1" not in result
    assert "2.1_p2" in result
    assert doc.closed is True


@pytest.mark.fast
def test_asset_extractor_boundary_overlap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import MagicMock

    import pypdfium2 as pdfium

    from ubt.adapters.pdf.asset_extractor import extract_pdf_figures

    dummy_pdf = tmp_path / "dummy.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4 dummy")

    mock_doc = MagicMock()
    mock_page = MagicMock()
    mock_doc.__len__.return_value = 1
    mock_doc.__getitem__.return_value = mock_page
    mock_doc.__iter__.return_value = [mock_page]
    mock_page.get_size.return_value = (600.0, 800.0)

    mock_textpage = MagicMock()
    mock_page.get_textpage.return_value = mock_textpage
    mock_textpage.get_text_range.return_value = "Figure 1.1: Test diagram\n"

    # Mock caption search hit
    mock_search = MagicMock()
    mock_search.get_next.side_effect = [(0, 10), None, (0, 10), None]
    mock_textpage.search.return_value = mock_search
    # Caption rect at y_bottom=100.0, y_top=115.0
    mock_textpage.get_rect.return_value = (50.0, 100.0, 200.0, 115.0)

    # Image object whose bottom starts at 75.0 (< c["y_bottom"] - 20 = 80.0),
    # but top extends into figure area at 250.0.
    # It crosses the boundary and should be captured in relevant_boxes!
    mock_img = MagicMock()
    mock_img.type = pdfium.raw.FPDF_PAGEOBJ_IMAGE
    mock_img.get_bounds.return_value = (60.0, 75.0, 250.0, 250.0)

    mock_page.get_objects.return_value = [mock_img]
    mock_page.render.return_value.to_pil.return_value.save = MagicMock()

    monkeypatch.setattr(pdfium, "PdfDocument", lambda _p: mock_doc)

    figs = extract_pdf_figures(dummy_pdf, tmp_path / "assets")
    assert len(figs) == 1
    fig = list(figs.values())[0]
    # If the image object was captured in relevant_boxes, bx0 is derived from mock_img (60 - 45 = 15.0),
    # NOT the fallback 45.0!
    assert fig.bbox is not None, "figure bbox should be resolved"
    assert fig.bbox[0] == 15.0
