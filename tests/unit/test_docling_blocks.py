"""Unit tests for docling table conversion and block transformations."""

import pytest

from ubt.adapters.pdf.docling_blocks import table_to_markdown


class MockTableCell:
    def __init__(
        self,
        text: str,
        *,
        row_span: int = 1,
        col_span: int = 1,
        start_row_offset_idx: int = 0,
        end_row_offset_idx: int = 1,
        start_col_offset_idx: int = 0,
        end_col_offset_idx: int = 1,
    ) -> None:
        self.text = text
        self.row_span = row_span
        self.col_span = col_span
        self.start_row_offset_idx = start_row_offset_idx
        self.end_row_offset_idx = end_row_offset_idx
        self.start_col_offset_idx = start_col_offset_idx
        self.end_col_offset_idx = end_col_offset_idx

    def _get_text(self, doc: object = None, **kwargs: object) -> str:
        return self.text


class MockTableData:
    def __init__(self, grid: list[list[MockTableCell]]) -> None:
        self.grid = grid
        self.num_rows = len(grid)
        self.num_cols = len(grid[0]) if grid else 0


class MockTableItem:
    def __init__(self, data: MockTableData) -> None:
        self.data = data


@pytest.mark.fast
def test_table_to_markdown_merged_colspan_does_not_duplicate_text() -> None:
    """A cell spanning multiple columns must only emit text in its primary cell,
    leaving continuation cells empty rather than duplicating text across columns.
    """
    c_merged = MockTableCell(
        text="Benchmark Results",
        row_span=1,
        col_span=2,
        start_row_offset_idx=0,
        end_row_offset_idx=1,
        start_col_offset_idx=0,
        end_col_offset_idx=2,
    )
    c_col3 = MockTableCell(
        text="Notes",
        row_span=1,
        col_span=1,
        start_row_offset_idx=0,
        end_row_offset_idx=1,
        start_col_offset_idx=2,
        end_col_offset_idx=3,
    )
    r2_1 = MockTableCell(
        text="Task A",
        start_row_offset_idx=1,
        end_row_offset_idx=2,
        start_col_offset_idx=0,
        end_col_offset_idx=1,
    )
    r2_2 = MockTableCell(
        text="95.5",
        start_row_offset_idx=1,
        end_row_offset_idx=2,
        start_col_offset_idx=1,
        end_col_offset_idx=2,
    )
    r2_3 = MockTableCell(
        text="Passed",
        start_row_offset_idx=1,
        end_row_offset_idx=2,
        start_col_offset_idx=2,
        end_col_offset_idx=3,
    )

    # Grid references the same merged cell object for col 0 and col 1 in row 0
    grid = [
        [c_merged, c_merged, c_col3],
        [r2_1, r2_2, r2_3],
    ]
    item = MockTableItem(MockTableData(grid))

    md = table_to_markdown(item, None)
    assert md.count("Benchmark Results") == 1, (
        f"Merged cell text was duplicated in markdown output:\n{md}"
    )
    lines = md.strip().splitlines()
    assert len(lines) >= 3  # header, separator, row 2
    # Ensure 3 columns are maintained
    header_cells = [c.strip() for c in lines[0].strip("|").split("|")]
    assert len(header_cells) == 3
    assert header_cells[0] == "Benchmark Results"
    assert header_cells[1] == ""  # continuation must be empty
    assert header_cells[2] == "Notes"


@pytest.mark.fast
def test_table_to_markdown_merged_rowspan_and_colspan_distinct_cell_objects() -> None:
    """When a 2x2 merged cell is represented with distinct cell objects in the grid
    and offset attributes are unpopulated, table_to_markdown must track row_span/col_span
    coordinates and emit the text only once, leaving continuation cells empty.
    """

    class SimpleCell:
        def __init__(self, text: str, row_span: int = 1, col_span: int = 1) -> None:
            self.text = text
            self.row_span = row_span
            self.col_span = col_span

    c00 = SimpleCell("TopLeftMerged", row_span=2, col_span=2)
    c01 = SimpleCell("TopLeftMerged")
    c10 = SimpleCell("TopLeftMerged")
    c11 = SimpleCell("TopLeftMerged")
    c02 = SimpleCell("Header3")
    c12 = SimpleCell("Data3")

    grid = [
        [c00, c01, c02],
        [c10, c11, c12],
    ]
    item = MockTableItem(MockTableData(grid))  # type: ignore[arg-type]
    md = table_to_markdown(item, None)
    assert md.count("TopLeftMerged") == 1, f"Expected 1 occurrence, got:\n{md}"
    lines = md.strip().splitlines()
    assert len(lines) >= 3
    row0 = [c.strip() for c in lines[0].strip("|").split("|")]
    row1 = [c.strip() for c in lines[2].strip("|").split("|")]
    assert row0 == ["TopLeftMerged", "", "Header3"]
    assert row1 == ["", "", "Data3"]
