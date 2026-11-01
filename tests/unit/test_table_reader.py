from __future__ import annotations

import pytest

from ubt.adapters.pdf import table_reader
from ubt.adapters.pdf.table_reader import merge_with_docling, parse_table_markup

pytestmark = pytest.mark.fast

_TABLE_MARKUP = (
    "<fcel>Configuration<fcel>Recorded Token Traffic (B)<lcel><lcel><lcel><lcel>"
    "<fcel>Token Cost<fcel>Avg. Score<fcel>Token Eff.<nl>"
    "<ucel><fcel>Input<fcel>Cache R.<fcel>Cache W.<fcel>Output<fcel>Total<ucel><ucel><ucel><nl>"
    "<fcel>Pi Baseline [35]<fcel>0.0011<fcel>2.1326<fcel>0.0141<fcel>0.0059"
    "<fcel>2.1538<fcel>1,339<fcel>44.833<fcel>0.5855<nl>"
)


def test_a_real_reply_parses_to_the_right_width() -> None:
    grid = parse_table_markup(_TABLE_MARKUP)
    assert grid is not None
    assert grid.columns == 9
    assert len(grid.rows) == 3
    assert grid.rows[2][0] == "Pi Baseline [35]"
    assert grid.rows[2][5] == "2.1538"


def test_span_markers_hold_their_position_without_text() -> None:
    grid = parse_table_markup(_TABLE_MARKUP)
    assert grid is not None
    assert grid.rows[0][2:6] == ("", "", "", "")
    assert grid.rows[1][0] == ""
    assert grid.rows[1][1] == "Input"


def test_a_reply_without_cell_markers_is_not_a_table() -> None:
    assert parse_table_markup("just some prose") is None
    assert parse_table_markup("") is None


def test_merged_results_keep_docling_header_and_take_model_body() -> None:
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
    assert merged[0][1] == "Recorded Token Traffic (B)"
    assert merged[2][0] == "Pi Baseline [35]"


def test_a_column_mismatch_refuses_the_merge() -> None:
    grid = parse_table_markup(_TABLE_MARKUP)
    assert grid is not None
    docling = [["a", "b", "c"], ["1", "2", "3"]]
    assert merge_with_docling(docling, grid) is None


def test_fewer_body_rows_refuses_the_merge() -> None:
    grid = parse_table_markup(_TABLE_MARKUP)
    assert grid is not None
    docling = [[str(i) for i in range(9)] for _ in range(10)]
    assert merge_with_docling(docling, grid) is None


def test_the_reader_is_off_unless_opted_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UBT_TABLE_READER", raising=False)
    assert table_reader.enabled() is False
    monkeypatch.setenv("UBT_TABLE_READER", "1")
    monkeypatch.delenv("UBT_OCR_ENDPOINT", raising=False)
    assert table_reader.enabled() is False
    monkeypatch.setenv("UBT_OCR_ENDPOINT", "http://127.0.0.1:8185/v1")
    assert table_reader.enabled() is True


def test_an_unreachable_reader_keeps_docling(monkeypatch: pytest.MonkeyPatch) -> None:
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
