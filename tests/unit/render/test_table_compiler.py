"""Unit tests for TableComplexity probe and Typst table compiler."""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from ubt.render.table_compiler import (
    TableComplexity,
    compile_typst_booktabs,
    normalize_table_to_source,
    parse_markdown_table,
    probe_table_complexity,
    verify_table_reconstruction,
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


def test_normalize_table_to_source_merges_a_shattered_row() -> None:
    """A model re-chunking one source cell into several target cells is repaired.

    Measured on the ATC'25 replay: the API table carried each row's signatures
    in one cell and the translation came back with the descriptions split into
    their own cells (a 2-column row delivered as 6). The renderer then lays a
    ragged grid under a 2-column spec and the reconstruction check reports a
    shattered table. The repair keeps every character.
    """
    source = (
        "| APIs | Descriptions |\n|---|---|\n"
        "| Scheduler::enqueue(&self, task) | 放入任务并访问队列 |\n"
        "| FrameAlloc::alloc(&self, layout) | 分配帧 |"
    )
    target = (
        "| APIs | 描述 |\n|---|---|\n"
        "| Scheduler::enqueue(&self, task) | 放入任务 | 访问本地运行队列 |\n"
        "| FrameAlloc::alloc(&self, layout) | 分配帧 |"
    )
    normalized = normalize_table_to_source(source, target)

    ok = verify_table_reconstruction(source, normalized)
    assert ok.ok, ok.reason
    # Nothing was dropped: both fragments of the shattered cell survive, joined
    # in the row's last cell.
    assert "放入任务" in normalized and "访问本地运行队列" in normalized


def test_normalize_table_to_source_leaves_a_clean_grid_alone() -> None:
    source = "| A | B |\n|---|---|\n| 一 | 二 |"
    assert normalize_table_to_source(source, source) == source


def test_normalize_table_to_source_never_pads_a_dropped_column() -> None:
    """A short row lost a column at translation time -- real data loss.

    Padding it with empty cells would bless the loss as ``verified``; the row
    ships unchanged and the shape check reports the shortfall.
    """
    source = "| A | B | C |\n|---|---|---|\n| 一 | 二 | 三 |"
    target = "| A | B | C |\n|---|---|---|\n| 一 | 二 |"
    normalized = normalize_table_to_source(source, target)
    assert normalized == target
    ok = verify_table_reconstruction(source, normalized)
    assert not ok.ok and "row_1_cell_count_mismatch" in ok.reason


def test_compile_typst_booktabs_structure() -> None:
    """Standard table emits Typst booktabs with 3-line rules and no vertical borders."""
    md = """
| 方法 | 准确率 | 延迟 (ms) |
| --- | --- | --- |
| 基线模型 | 82.5% | 15.2 |
| 拓扑增强 | 91.8% | 16.0 |
"""
    result = compile_typst_booktabs(md, content_width_pt=350.0)
    assert result, "expected compiled booktabs markup"
    assert "stroke: none" in result
    assert "table.hline(stroke: 1.2pt)" in result  # top & bottom rules
    assert "table.hline(stroke: 0.5pt)" in result  # mid header rule
    assert "[*方法*]" in result
    assert "[82.5%]" in result


def test_compile_typst_booktabs_adaptive_widths() -> None:
    """Wide CJK descriptive columns receive proportionally wider allocation than narrow numeric ones."""
    md = """
| ID | 详细架构描述与注意力特征提取分析 | 评分 |
| --- | --- | --- |
| 1 | 这是一个非常宽且详尽的多头自注意力机制说明文本 | 9.5 |
"""
    result = compile_typst_booktabs(md, content_width_pt=400.0)
    # Extract columns: (w1, w2, w3)
    match = re.search(r"columns:\s*\(([^)]+)\)", result)
    assert match is not None
    widths = [float(w.strip().removesuffix("pt")) for w in match.group(1).split(",")]
    assert len(widths) == 3
    # Middle column (wide CJK) must be significantly wider than col 0 (ID) and col 2 (Score)
    assert widths[1] > widths[0] * 3.0
    assert widths[1] > widths[2] * 3.0


def test_compile_typst_booktabs_stretch_compact() -> None:
    """When stretch_pct is provided, font stretch is injected for variable font compaction."""
    md = """
| Col1 | Col2 |
| --- | --- |
| A | B |
"""
    result = compile_typst_booktabs(md, stretch_pct=90.0)
    assert "stretch: 90%" in result


def test_compile_typst_booktabs_cell_clipping_height_fit() -> None:
    """When content_height_pt is constrained, font scales down and insets compact to prevent cell clipping."""
    md = "\n".join(
        ["| Index | Measurement |", "|---|---|"] + [f"| item_{i} | value_{i} |" for i in range(10)]
    )
    # 10 rows in a 50pt box -> must shrink scale and compact inset
    result = compile_typst_booktabs(md, content_height_pt=50.0, base_font_size_pt=10.0)
    assert result
    # Emitted scale in #set text(size: Xem) should be scaled down
    match = re.search(r"#set text\(size:\s*([\d\.]+)em", result)
    assert match is not None
    scale = float(match.group(1))
    assert scale < 1.0


def test_compile_typst_booktabs_math_and_breaks() -> None:
    """Math formulas and <br> line breaks in cells are preserved cleanly."""
    md = """
| Parameter | Value |
| --- | --- |
| Loss $L_2$ | Line 1<br>Line 2 |
"""
    result = compile_typst_booktabs(md)
    assert "$L_2$" in result or "L_2" in result
    assert "\\\n" in result  # Typst line break for <br>


def test_compile_typst_booktabs_column_alignments() -> None:
    """Markdown column alignments (:---, :---:, ---:) are transpiled to Typst alignment arrays."""
    md = """
| Left Align | Center Align | Right Align |
| :--- | :---: | ---: |
| Alpha | Beta | 123.45 |
"""
    result = compile_typst_booktabs(md)
    assert "left + horizon, center + horizon, right + horizon" in result


def test_compile_typst_booktabs_slash_breaks_in_urls() -> None:
    """Long URLs with slashes are treated as break opportunities so they do not over-inflate run minimums."""
    md = """
| Name | Endpoint URL |
| --- | --- |
| API | https://example.com/api/v1/projects/technical/evaluations/run |
"""
    result = compile_typst_booktabs(md, content_width_pt=350.0)
    assert result
    # Columns shouldn't be shrunk down to minimum font size because of slash breaks
    assert "#set text(size: 0.75em" not in result


def test_compile_typst_booktabs_multiline_br_height_budget() -> None:
    """Rows with multiple <br> tags are accurately accounted for in vertical height estimation."""
    md = """
| Module | Responsibilities |
| --- | --- |
| Core | Task 1<br>Task 2<br>Task 3<br>Task 4 |
"""
    # 1 row with 4 lines in a tight 30pt height box -> must scale down
    result = compile_typst_booktabs(md, content_height_pt=30.0, base_font_size_pt=10.0)
    assert result
    match = re.search(r"#set text\(size:\s*([\d\.]+)em", result)
    assert match is not None
    scale = float(match.group(1))
    assert scale < 1.0


def test_compile_typst_booktabs_empty_rows() -> None:
    """Table with only headers and no data rows compiles cleanly without redundant 0.5pt mid rule."""
    md = """
| Empty Header 1 | Empty Header 2 |
| --- | --- |
"""
    result = compile_typst_booktabs(md)
    assert result
    assert "table.hline(stroke: 1.2pt)" in result
    assert "table.hline(stroke: 0.5pt)" not in result


def test_compile_typst_booktabs_dense_cjk_proportional_allocation_and_height_solve() -> None:
    """Academic tables with multi-char CJK headers and tight vertical boxes dynamically scale and fit."""
    md = """
| 基于 Rust 的 OS | Linux | Tock | RedLeaf | Theseus |
|---|---|---|---|---|
| 使用不安全关键字的包 | 6 / 11 (55%) | 91 / 98 (93%) | 36 / 58 (62%) | 54 / 171 (32%) |
"""
    # 38.7pt height budget (as on ATC'25 page 2)
    result = compile_typst_booktabs(
        md,
        content_width_pt=244.49,
        content_height_pt=38.71,
        base_font_size_pt=9.75,
        min_font_size_pt=6.5,
    )
    assert result
    # Column 0 (10-char CJK text) must receive wider width than numeric columns
    match = re.search(r"columns:\s*\(([^)]+)\)", result)
    assert match is not None
    widths = [float(w.strip().removesuffix("pt")) for w in match.group(1).split(",")]
    assert len(widths) == 5
    assert widths[0] > widths[1]
    assert widths[0] > widths[2]
    # Font scale must be scaled down to fit within the 38.71pt box
    scale_match = re.search(r"#set text\(size:\s*([\d\.]+)em", result)
    assert scale_match is not None
    scale = float(scale_match.group(1))
    assert scale < 1.0
