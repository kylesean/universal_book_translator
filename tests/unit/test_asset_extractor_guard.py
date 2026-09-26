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
from types import SimpleNamespace
from typing import Any, cast

import pytest

from ubt.adapters.pdf import asset_extractor
from ubt.adapters.pdf.asset_extractor import extract_pdf_figures
from ubt.core.ir.models import BlockType, FlowID, IRBlock


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


def test_caption_driven_figure_crops_do_not_ship_the_same_figure_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A caption rect that contains another crop printed one figure twice.

    On chapter-3-zh.pdf page 4, ``fig_3_3_p4`` fully contained ``fig_3_2_p4``
    (overlap/min-area 1.00) and overlapped ``fig_3_4_p4`` by 57%, so the reader
    got figure 3.2 again — with figure 3.2's caption inside figure 3.3 — while
    both were also emitted separately.
    """
    import ubt.adapters.pdf.asset_extractor as asset_extractor
    import ubt.adapters.pdf.svg_diagram as svg_diagram
    from ubt.adapters.pdf.docling_render import DoclingRenderStrategy
    from ubt.core.ir.models import BoundingBox

    source_pdf = Path(__file__).resolve().parents[2] / "docs" / "synthetic-mono.pdf"
    if not source_pdf.is_file():
        pytest.skip(f"{source_pdf.name} fixture missing")

    monkeypatch.setattr(svg_diagram, "is_svg_backend_available", lambda: False)
    monkeypatch.setattr(svg_diagram, "is_svg_rendering_supported", lambda: False)
    monkeypatch.setattr(svg_diagram, "detect_diagram_regions", lambda *a, **k: [])

    fig_png = tmp_path / "fig.png"
    fig_png.write_bytes(b"\x89PNG\r\n\x1a\n")

    def figure(fig_id: str, bbox: tuple[float, float, float, float]) -> object:
        return asset_extractor.ExtractedFigure(
            fig_id=fig_id,
            caption_en=f"FIG. {fig_id}: demo",
            page=1,
            image_path=fig_png,
            relative_path="assets/fig.png",
            bbox=bbox,
        )

    monkeypatch.setattr(
        asset_extractor,
        "extract_pdf_figures",
        lambda *a, **k: {
            "big": figure("big", (0.0, 0.0, 300.0, 300.0)),
            "inside": figure("inside", (100.0, 100.0, 200.0, 200.0)),
            "apart": figure("apart", (500.0, 500.0, 600.0, 600.0)),
        },
    )

    strategy = DoclingRenderStrategy(
        reconstructor=SimpleNamespace(),  # type: ignore[arg-type]
        alternator=SimpleNamespace(),  # type: ignore[arg-type]
        diagram_localizer=cast("Any", SimpleNamespace(get_page_height=lambda _src, _page: 800.0)),
    )
    blocks = [
        IRBlock(
            id="pdf_main#b001",
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            flow_id=FlowID.MAIN_STORY,
            source_text="Body text.",
            bbox=BoundingBox(page=1, x0=0.0, y0=0.0, x1=100.0, y1=100.0),
        )
    ]

    out, _ = strategy._vectorize_diagrams_sync(source_pdf, blocks, tmp_path / "assets", "zh")
    ids = sorted(b.id for b in out if b.id.startswith("pdf_main#fig_"))
    assert ids == ["pdf_main#fig_apart", "pdf_main#fig_big"], ids


def _r0921p_caption_pdf(path: Path, pages: int = 2) -> Path:
    """Minimal born-digital PDF: every page shows the same "FIG. 3.2" caption.

    Hand-written bytes (the shape ``tests/pdf_builders.text_pdf`` uses) so the
    extraction test does not also test a writer library.
    """
    kids = " ".join(f"{3 + i * 2} 0 R" for i in range(pages))
    objs: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: f"<< /Type /Pages /Kids [{kids}] /Count {pages} >>".encode(),
    }
    for i in range(pages):
        text = "FIG. 3.2 Shared caption body text here."
        content = f"BT /F1 12 Tf 72 100 Td ({text}) Tj ET".encode("latin-1")
        objs[3 + i * 2] = (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
            + f"/Contents {4 + i * 2} 0 R /Resources << /Font << /F1 99 0 R >> >> >>".encode()
        )
        objs[4 + i * 2] = (
            f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream"
        )
    objs[99] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for num in sorted(objs):
        offsets[num] = len(out)
        out += f"{num} 0 obj\n".encode() + objs[num] + b"\nendobj\n"
    size = max(offsets) + 1
    xref_pos = len(out)
    out += f"xref\n0 {size}\n".encode() + b"0000000000 65535 f \n"
    for num in range(1, size):
        out += f"{offsets.get(num, 0):010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF".encode()
    path.write_bytes(bytes(out))
    return path


def test_figures_with_the_same_number_on_different_pages_stay_distinct(
    tmp_path: Path,
) -> None:
    """A book repeats "Figure 3.2" in more than one chapter.

    Keying by figure number alone made the later page overwrite the earlier
    PNG and dict entry, so the first figure's translation was lost and both
    blocks pointed at one image.
    """
    pdf = _r0921p_caption_pdf(tmp_path / "caps.pdf", pages=2)

    figures = extract_pdf_figures(pdf, tmp_path / "assets", dpi=72)

    assert len(figures) == 2, "each page's figure must survive as its own entry"
    assert {fig.page for fig in figures.values()} == {1, 2}
    paths = {fig.image_path for fig in figures.values()}
    assert len(paths) == 2 and all(p.exists() for p in paths)
