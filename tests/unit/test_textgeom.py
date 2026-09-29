"""Tests for pdfium geometric text extraction helpers (``ubt/adapters/pdf/textgeom.py``)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ubt.adapters.pdf.textgeom import LineBox, column_order

pytestmark = pytest.mark.fast


def _order(lines: list[LineBox], page_width: float = 600.0) -> list[str]:
    return [ln.text for ln in column_order(lines, page_width)]


def test_centered_banner_does_not_bridge_two_columns() -> None:
    """A centered line spanning the gutter must not collapse the page to top-down.

    Regression: connected-component clustering merged every overlapping interval,
    so one centered banner (width in (0.15, 0.7]·span) chained the left and right
    columns into a single cluster and the page read ``TITLE,L0,R0,L1,R1`` —
    interleaved, wrong reading order.
    """
    lines = [
        LineBox(text="TITLE", rect=(150, 700, 450, 720)),
        LineBox(text="L0", rect=(50, 650, 290, 665)),
        LineBox(text="R0", rect=(310, 650, 550, 665)),
        LineBox(text="L1", rect=(50, 630, 290, 645)),
        LineBox(text="R1", rect=(310, 630, 550, 645)),
    ]
    order = _order(lines)
    assert [t for t in order if t in {"L0", "L1"}] == ["L0", "L1"], order
    assert [t for t in order if t in {"R0", "R1"}] == ["R0", "R1"], order
    # No interleaving: every left-column line precedes every right-column line.
    assert order.index("L1") < order.index("R0"), order
    # Banner placed above both columns must be read before the columns.
    assert order.index("TITLE") < order.index("L0"), order


def test_two_columns_read_left_before_right() -> None:
    """Without any spanning line the two columns must still read left-first."""
    lines = [
        LineBox(text="L0", rect=(50, 700, 290, 715)),
        LineBox(text="R0", rect=(310, 700, 550, 715)),
        LineBox(text="L1", rect=(50, 680, 290, 695)),
        LineBox(text="R1", rect=(310, 680, 550, 695)),
    ]
    assert _order(lines) == ["L0", "L1", "R0", "R1"]


def test_single_column_ragged_lines_stay_top_down() -> None:
    """A ragged single column must not be split into fake columns."""
    lines = [
        LineBox(text="a long line spanning most of the column", rect=(50, 700, 550, 715)),
        LineBox(text="short", rect=(50, 680, 150, 695)),
        LineBox(text="a medium-length line here", rect=(50, 660, 300, 675)),
        LineBox(text="another long line spanning the column", rect=(50, 640, 545, 655)),
    ]
    assert _order(lines) == [
        "a long line spanning most of the column",
        "short",
        "a medium-length line here",
        "another long line spanning the column",
    ]


def test_full_width_lines_do_not_form_columns() -> None:
    """Only full-width lines: there is no gutter, so order is strict top-down."""
    lines = [
        LineBox(text="a", rect=(0, 700, 600, 715)),
        LineBox(text="b", rect=(0, 680, 600, 695)),
    ]
    assert _order(lines) == ["a", "b"]


def _install_fake_pdfium(monkeypatch: pytest.MonkeyPatch, closes: list[str]) -> None:
    import sys
    import types

    class _FakeTextPage:
        def count_rects(self, start: int, count: int) -> int:
            return 2

        def get_rect(self, idx: int) -> tuple[float, float, float, float]:
            return (0.0, 0.0, 10.0, 10.0)

        def close(self) -> None:
            closes.append("textpage")

    class _FakePage:
        def get_textpage(self) -> _FakeTextPage:
            return _FakeTextPage()

        def close(self) -> None:
            closes.append("page")

    class _FakeDoc:
        def __init__(self, _path: str) -> None:
            pass

        def __len__(self) -> int:
            return 1

        def __getitem__(self, idx: int) -> _FakePage:
            return _FakePage()

        def close(self) -> None:
            closes.append("doc")

    monkeypatch.setitem(sys.modules, "pypdfium2", types.SimpleNamespace(PdfDocument=_FakeDoc))


def test_extract_text_rects_closes_pdfium_handles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every native handle opened by ``extract_text_rects`` must be closed.

    The function opened the document, page and textpage and returned without
    closing any of them; measured with the sibling ``extract_lines`` pattern as
    control, that leaks ~0.6 MB of native memory per call (+197 MB RSS over 300
    calls vs +3 MB with explicit closes), and the export-time visual gate calls
    it once per page.
    """
    from ubt.adapters.pdf import textgeom

    closes: list[str] = []
    _install_fake_pdfium(monkeypatch, closes)

    rects = textgeom.extract_text_rects(tmp_path / "x.pdf", 1)
    assert len(rects) == 2
    assert closes == ["textpage", "page", "doc"]


def test_extract_text_rects_closes_handles_on_out_of_range_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The out-of-range raise path must not leak the document handle."""
    from ubt.adapters.pdf import textgeom
    from ubt.core.exceptions import DocumentParseError

    closes: list[str] = []
    _install_fake_pdfium(monkeypatch, closes)

    with pytest.raises(DocumentParseError):
        textgeom.extract_text_rects(tmp_path / "x.pdf", 2)
    assert closes == ["doc"]
