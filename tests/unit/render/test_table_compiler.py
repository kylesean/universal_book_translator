"""Unit tests for TableComplexity probe and Typst table compiler."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ubt.render.table_compiler import (
    TableComplexity,
    parse_markdown_table,
    probe_table_complexity,
)

pytestmark = pytest.mark.fast


def test_parse_markdown_table_valid() -> None:
    md = """
| Header 1 | Header 2 | Header 3 |
| :--- | :---: | ---: |
| Value A | Value B | Value C |
| Value D | Value E | Value F |
"""
    parsed = parse_markdown_table(md)
    assert parsed is not None
    headers, rows = parsed
    assert headers == ["Header 1", "Header 2", "Header 3"]
    assert len(rows) == 2
    assert rows[0] == ["Value A", "Value B", "Value C"]
    assert rows[1] == ["Value D", "Value E", "Value F"]


def test_parse_markdown_table_invalid() -> None:
    assert parse_markdown_table("Not a table at all") is None
    assert parse_markdown_table("| Lone line |") is None


def test_probe_table_complexity_simple() -> None:
    md = """
| Method | Accuracy | F1 |
| --- | --- | --- |
| Baseline | 82.5% | 81.2 |
| Ours | 89.1% | 88.5 |
"""
    bbox = SimpleNamespace(x0=50.0, y0=100.0, x1=400.0, y1=200.0, height=100.0, width=350.0)
    comp = probe_table_complexity(md, bbox)
    assert isinstance(comp, TableComplexity)
    assert not comp.is_complex
    assert comp.reason == "simple_grid"
    assert comp.rows == 3
    assert comp.cols == 3


def test_probe_table_complexity_merged_cells() -> None:
    md = """
| Model | <td colspan="2">Results</td> |
| --- | --- |
| A | 10 | 20 |
"""
    comp = probe_table_complexity(md)
    assert comp.is_complex
    assert comp.reason == "merged_cells"


def test_probe_table_complexity_too_many_columns() -> None:
    md = """
| C1 | C2 | C3 | C4 | C5 | C6 | C7 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 3 | 4 | 5 | 6 | 7 |
"""
    comp = probe_table_complexity(md)
    assert comp.is_complex
    assert "too_many_columns" in comp.reason


def test_probe_table_complexity_dense_row_height() -> None:
    md = """
| A | B |
| --- | --- |
| 1 | 2 |
| 3 | 4 |
| 5 | 6 |
| 7 | 8 |
"""
    # 5 rows total, height 40pt -> 8pt per row < 12pt
    bbox = SimpleNamespace(x0=50.0, y0=100.0, x1=300.0, y1=140.0, height=40.0, width=250.0)
    comp = probe_table_complexity(md, bbox)
    assert comp.is_complex
    assert "dense_row_height" in comp.reason


def test_probe_table_complexity_prose_cells() -> None:
    long_prose = (
        "This is an extremely long narrative explanation of why this specific model failed on edge case scenarios. "
        * 3
    )
    md = f"""
| Scenario | Explanation |
| --- | --- |
| Edge Case 1 | {long_prose} |
"""
    comp = probe_table_complexity(md)
    assert comp.is_complex
    assert comp.reason == "prose_paragraphs_in_cells"


def test_probe_table_complexity_dense_single_col_suggests_landscape() -> None:
    # 8 columns in single column layout
    header = "| " + " | ".join(f"Col{i}" for i in range(8)) + " |"
    sep = "| " + " | ".join("---" for _ in range(8)) + " |"
    row = "| " + " | ".join(f"Val{i}" for i in range(8)) + " |"
    md = f"{header}\n{sep}\n{row}\n"

    comp = probe_table_complexity(md, page_columns=1)
    assert comp.is_complex is True
    assert "too_many_columns_8" in comp.reason
    assert comp.suggests_landscape is True
    assert comp.contingency == "landscape"


def test_probe_table_complexity_dense_dual_col_no_landscape() -> None:
    # 8 columns in dual column layout
    header = "| " + " | ".join(f"Col{i}" for i in range(8)) + " |"
    sep = "| " + " | ".join("---" for _ in range(8)) + " |"
    row = "| " + " | ".join(f"Val{i}" for i in range(8)) + " |"
    md = f"{header}\n{sep}\n{row}\n"

    comp = probe_table_complexity(md, page_columns=2)
    assert comp.is_complex is True
    assert comp.suggests_landscape is False
    assert comp.contingency == "appendix"

    # Even with narrow column content width (e.g. 219.5pt in academic dual-col),
    # dual-column tables must NOT suggest landscape
    comp_narrow = probe_table_complexity(md, page_columns=2, content_width_pt=219.5)
    assert comp_narrow.suggests_landscape is False
    assert comp_narrow.contingency == "appendix"


def test_probe_table_complexity_narrow_width_suggests_landscape() -> None:
    # 6 columns in narrow single-column layout (content_width_pt=300)
    header = "| " + " | ".join(f"Col{i}" for i in range(6)) + " |"
    sep = "| " + " | ".join("---" for _ in range(6)) + " |"
    row = "| " + " | ".join(f"Val{i}" for i in range(6)) + " |"
    md = f"{header}\n{sep}\n{row}\n"

    comp = probe_table_complexity(md, page_columns=1, content_width_pt=300.0)
    assert comp.is_complex is True
    assert comp.suggests_landscape is True
    assert comp.contingency == "landscape"
