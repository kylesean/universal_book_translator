"""Unit tests for TableComplexity probe and Typst table compiler."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ubt.render.table_compiler import (
    TableComplexity,
    markdown_table_to_typst,
    parse_markdown_table,
    probe_table_complexity,
    render_table_appendix_typst,
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


def test_markdown_table_to_typst() -> None:
    md = """
| 特征 | 基准模型 | 优化模型 |
| :--- | :--- | :--- |
| 吞吐量 | 120 ops/s | 350 ops/s |
| 显存占用 | 16 GB | 8 GB |
"""
    typst_code = markdown_table_to_typst(md, width_pt=300.0, height_pt=100.0, font_size_pt=8.0)
    assert typst_code is not None
    assert "#table(" in typst_code
    assert "columns: (3 * (1fr,))" in typst_code
    assert "吞吐量" in typst_code
    assert "优化模型" in typst_code
    assert "width: 300.0pt" in typst_code


def test_render_table_appendix_typst() -> None:
    tables = [
        (
            3,
            "Table 1: 消融实验",
            "| 组件 | BLEU |\n| --- | --- |\n| 完整 | 34.2 |\n| 无记忆 | 31.0 |",
        ),
        (
            15,
            "Table 4: 性能测试",
            "| 批大小 | 延迟 |\n| --- | --- |\n| 1 | 12ms |\n| 8 | 45ms |",
        ),
    ]
    doc = render_table_appendix_typst(tables)
    assert "附录：核心图表与数据矩阵对照" in doc
    assert "原书第 3 页: Table 1: 消融实验" in doc
    assert "原书第 15 页: Table 4: 性能测试" in doc
    assert "BLEU" in doc
    assert "延迟" in doc
