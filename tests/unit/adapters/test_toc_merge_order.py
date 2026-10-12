"""The merged table of contents must read in destination-page order.

Docling's TOC pass and the PDF's bookmark tree are two partial views of the same
list: Docling reads the sub-section rows with their geometry, the bookmarks carry
the top-level chapters it drops. Merging them additively put every chapter after
every section, so the delivered contents opened on "2.1" and printed chapter 1
below its own sub-sections.
"""

from __future__ import annotations

import pytest

from ubt.adapters.pdf.docling_parser import _toc_sort_key
from ubt.core.ir.models import BlockProvenance, BlockType, IRBlock, make_element

pytestmark = pytest.mark.fast


def _row(title: str, toc_page: str, index: int) -> IRBlock:
    return IRBlock(
        element=make_element(
            id=f"pdf_main#b{index:04d}",
            spine_index=index,
            block_type=BlockType.HEADING,
            source_text=title,
            level=1,
            skip_translate=False,
        ),
        provenance=BlockProvenance(toc_entry=True, toc_page=toc_page),
    )


def test_a_chapter_sorts_before_the_sections_it_contains() -> None:
    # The shape the merge produced: sections first, chapters appended.
    rows = [
        _row("2.1 Simplifying Sparse Routing", "5", 0),
        _row("2.2 Efficient Sparse Routing", "6", 1),
        _row("1 Introduction", "3", 2),
        _row("2 Switch Transformer", "4", 3),
    ]
    ordered = [b.source_text for b in sorted(rows, key=_toc_sort_key)]

    assert ordered == [
        "1 Introduction",
        "2 Switch Transformer",
        "2.1 Simplifying Sparse Routing",
        "2.2 Efficient Sparse Routing",
    ]


def test_rows_sharing_a_page_keep_their_source_order() -> None:
    # ``sorted`` is stable, so a sub-section cannot jump above its chapter when
    # both name the same destination page.
    rows = [
        _row("2.3 Putting It All Together", "8", 0),
        _row("2.4 Improved Training", "8", 1),
    ]
    ordered = [b.source_text for b in sorted(rows, key=_toc_sort_key)]
    assert ordered == ["2.3 Putting It All Together", "2.4 Improved Training"]


def test_an_unparseable_page_sorts_last() -> None:
    # A malformed entry must not be reordered to the front on a guess.
    rows = [_row("Broken", "", 0), _row("1 Introduction", "3", 1)]
    ordered = [b.source_text for b in sorted(rows, key=_toc_sort_key)]
    assert ordered == ["1 Introduction", "Broken"]
