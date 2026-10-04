"""Native PDF reader structure: list boundaries and table-of-contents rows.

Two parsing-layer regressions the Docling path handled but the pdfium reader
did not:

- consecutive list items were grouped into one paragraph (the reader's own
  docstring rule 4 -- "a line beginning with a bullet/number starts a list
  item" -- was unimplemented), so a numbered list collapsed into one block;
- a table of contents was read as translated titles plus a run of bare
  page-number blocks, because ``textgeom`` returns the title and its
  right-margin page number as two lines.

Both are pinned here over synthetic PDFs written by the independent
``pdf_builders`` writer.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pdf_builders import write_text_pdf, write_toc_pdf

from ubt.analyze.bridge import blocks_from_document
from ubt.analyze.reader_pdf import read_pdf
from ubt.analyze.structure import pdf_list_marker
from ubt.core.ir.models import BlockType

pytestmark = pytest.mark.fast


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("1. Introduction", ("1.", "Introduction")),
        ("1.1. Background", ("1.1.", "Background")),
        ("1.2.3. Deep", ("1.2.3.", "Deep")),
        ("2) x", ("2)", "x")),
        ("(a) y", ("(a)", "y")),
        ("一、 中文", ("一、", "中文")),
        ("- item", ("-", "item")),
        ("• bullet", ("•", "bullet")),
        # Ambiguous forms stay prose: a decimal, a signed number, a year, a deref.
        ("1.5 ratio", None),
        ("-3 is negative", None),
        ("*ptr", None),
        ("2020. was a year", None),
        ("plain prose.", None),
    ],
)
def test_pdf_list_marker(line: str, expected: tuple[str, str] | None) -> None:
    assert pdf_list_marker(line) == expected


def _blocks(path: Path):
    return blocks_from_document(read_pdf(path))


def test_consecutive_ordered_items_are_separate_list_items(tmp_path: Path) -> None:
    source = write_text_pdf(
        tmp_path / "list.pdf",
        [
            [
                "Introduction",
                "1. First item that is short",
                "2. Second item",
                "3. Third item",
                "Closing prose.",
            ]
        ],
    )
    blocks = _blocks(source)
    items = [block for block in blocks if block.block_type is BlockType.LIST_ITEM]
    assert [block.source_text for block in items] == [
        "First item that is short",
        "Second item",
        "Third item",
    ]
    assert [getattr(block.element, "marker", "") for block in items] == ["1.", "2.", "3."]


def test_a_wrapped_continuation_stays_with_its_item(tmp_path: Path) -> None:
    source = write_text_pdf(
        tmp_path / "wrapped.pdf",
        [
            [
                "1. First item that is long enough",
                "to wrap onto a second line",
                "2. Second item",
            ]
        ],
    )
    items = [block for block in _blocks(source) if block.block_type is BlockType.LIST_ITEM]
    assert [getattr(block.element, "marker", "") for block in items] == ["1.", "2."]
    assert "wrap onto a second line" in items[0].source_text


def test_a_toc_is_kept_as_preserved_rows_not_scattered_page_numbers(tmp_path: Path) -> None:
    source = write_toc_pdf(
        tmp_path / "toc.pdf",
        [
            ("1. Introduction", "1"),
            ("1.1. Background", "2"),
            ("1.2. Method", "3"),
            ("2. Results", "7"),
        ],
    )
    blocks = _blocks(source)
    rows = [block.source_text for block in blocks]
    assert rows == [
        "1. Introduction 1",
        "1.1. Background 2",
        "1.2. Method 3",
        "2. Results 7",
    ]
    # The page numbers are part of the row, not separate translatable blocks.
    assert all(block.skip_translate for block in blocks)
    assert not any(block.source_text.strip().isdigit() for block in blocks)
