"""The vision table reader: parse a real reply, and refuse to merge a bad one.

Every case here comes from an actual run against PaddleOCR-VL-1.6 on arXiv
2609.20519, including the shapes that exposed bugs — a reply whose first cell
was shifted by one column, and Docling grids that must not be overwritten.
"""

from __future__ import annotations

import pytest

from ubt.adapters.pdf import table_reader
from ubt.adapters.pdf.table_reader import merge_with_docling, parse_table_markup

pytestmark = pytest.mark.fast

#: Verbatim from the run, including the span markers.
_TABLE_MARKUP = (
    "<fcel>Configuration<fcel>Recorded Token Traffic (B)<lcel><lcel><lcel><lcel>"
    "<fcel>Token Cost<fcel>Avg. Score<fcel>Token Eff.<nl>"
    "<ucel><fcel>Input<fcel>Cache R.<fcel>Cache W.<fcel>Output<fcel>Total<ucel><ucel><ucel><nl>"
    "<fcel>Pi Baseline [35]<fcel>0.0011<fcel>2.1326<fcel>0.0141<fcel>0.0059"
    "<fcel>2.1538<fcel>1,339<fcel>44.833<fcel>0.5855<nl>"
)


def test_a_real_reply_parses_to_the_right_width() -> None:
    """The first version produced 10 columns here and shifted every value.

    It appended the empty text before the first ``<fcel>`` as a cell of its own,
    so a nine-column table came back ten wide with each value one column off —
    exactly the corruption this module exists to prevent.
    """
    grid = parse_table_markup(_TABLE_MARKUP)
    assert grid is not None
    assert grid.columns == 9
    assert len(grid.rows) == 3
    assert grid.rows[2][0] == "Pi Baseline [35]"
    assert grid.rows[2][5] == "2.1538"


def test_span_markers_hold_their_position_without_text() -> None:
    """``<lcel>``/``<ucel>`` mean "this position is covered", not "a cell of blank"."""
    grid = parse_table_markup(_TABLE_MARKUP)
    assert grid is not None
    # Row 0: the group header spans five columns; rows 1..4 of that span are empty.
    assert grid.rows[0][2:6] == ("", "", "", "")
    # Row 1 continues the first column from above, then opens new cells.
    assert grid.rows[1][0] == ""
    assert grid.rows[1][1] == "Input"


def test_a_reply_without_cell_markers_is_not_a_table() -> None:
    assert parse_table_markup("just some prose") is None
    assert parse_table_markup("") is None


def test_merged_results_keep_docling_header_and_take_model_body() -> None:
    """The header is Docling's; the fused data cell is the model's."""
    grid = parse_table_markup(_TABLE_MARKUP)
    assert grid is not None
    docling = [
        [
            "Configuration",
            "Recorded Token Traffic (B)",
            "x",
            "x",
            "x",
            "x",
            "Token Cost",
            "Score",
            "Eff",
        ],
        ["", "Input", "Cache R.", "Cache W.", "Output", "Total", "", "", ""],
        # Docling fused two rows into one cell: the defect the reader fixes.
        [
            "Pi Baseline [35] + Action Fusion",
            "0.0011",
            "2.1326",
            "0.0141",
            "0.0059",
            "2.1538",
            "1,339",
            "",
            "",
        ],
    ]
    merged = merge_with_docling(docling, grid)
    assert merged is not None
    assert merged[0][1] == "Recorded Token Traffic (B)"  # header untouched
    assert merged[2][0] == "Pi Baseline [35]"  # body replaced


def test_a_column_mismatch_refuses_the_merge() -> None:
    """A foreign grid would corrupt every column, so Docling's table stands."""
    grid = parse_table_markup(_TABLE_MARKUP)
    assert grid is not None
    docling = [["a", "b", "c"], ["1", "2", "3"]]
    assert merge_with_docling(docling, grid) is None


def test_fewer_body_rows_refuses_the_merge() -> None:
    """Collapsing rows is the failure being fixed, not a reason to accept it."""
    grid = parse_table_markup(_TABLE_MARKUP)
    assert grid is not None
    docling = [[str(i) for i in range(9)] for _ in range(10)]
    assert merge_with_docling(docling, grid) is None


def test_the_reader_is_off_unless_opted_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default pipeline must work with no service and no network."""
    monkeypatch.delenv("UBT_TABLE_READER", raising=False)
    assert table_reader.enabled() is False
    monkeypatch.setenv("UBT_TABLE_READER", "1")
    monkeypatch.delenv("UBT_OCR_ENDPOINT", raising=False)
    assert table_reader.enabled() is False  # switched on but nowhere to send
    monkeypatch.setenv("UBT_OCR_ENDPOINT", "http://127.0.0.1:8185/v1")
    assert table_reader.enabled() is True


def test_an_unreachable_reader_keeps_docling(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed upgrade returns None; the caller then keeps Docling's table."""
    monkeypatch.setenv("UBT_TABLE_READER", "1")
    monkeypatch.setenv("UBT_OCR_ENDPOINT", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("UBT_OCR_MODEL", "test-model")
    monkeypatch.setenv("UBT_OCR_TIMEOUT", "3")

    class _Image:
        def convert(self, mode: str) -> _Image:
            return self

        def save(self, buf: object, format: str) -> None:  # noqa: A002 - PIL signature
            assert hasattr(buf, "write")
            buf.write(b"png")

    assert table_reader.read_table_image_sync(_Image()) is None
