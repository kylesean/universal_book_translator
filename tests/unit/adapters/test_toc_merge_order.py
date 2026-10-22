"""The merged table of contents must not overwrite the blocks it follows.

Two regressions, one collision shape:

* ``cb2286a`` renumbered the TOC rows from spine 1 so every downstream
  ``spine_index`` sort would keep the destination-page order. But the front
  matter (title, authors, abstract, footnotes) had already been emitted with
  those very ids, and the ledger's ``ON CONFLICT(job_id, block_id) DO UPDATE``
  let the TOC rows *overwrite* them block-for-block: the delivered PDF opened
  on the contents page with no title page at all.
* Renaming the ids is not enough — the rows must also keep their position in
  the reading order (the printed TOC is page 2, after the front matter), so
  the renumber starts at ``len(blocks) + 1``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ubt.adapters.pdf.docling_outline import _toc_sort_key
from ubt.adapters.pdf.docling_parser import map_iterated_items
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
    # both name the same destination page — unless the label order itself says
    # otherwise (next test): two *sibling* sections share the page and neither
    # contains the other.
    rows = [
        _row("2.3 Putting It All Together", "8", 0),
        _row("2.4 Improved Training", "8", 1),
    ]
    ordered = [b.source_text for b in sorted(rows, key=_toc_sort_key)]
    assert ordered == ["2.3 Putting It All Together", "2.4 Improved Training"]


def test_a_chapter_sorts_above_its_section_when_both_share_a_page() -> None:
    # The arXiv 2101.03961 shape: "4 Downstream Results" (outline row) and
    # "4.1 Fine-Tuning" (Docling geometry row) both point at page 14, Docling
    # reads 4.1 first, and the pure stable sort delivered a contents page that
    # opened on the section above the chapter containing it. The label breaks
    # the tie: the shorter (parent) number sorts first.
    rows = [
        _row("4.1 Fine-Tuning", "14", 0),
        _row("4 Downstream Results", "14", 1),
    ]
    ordered = [b.source_text for b in sorted(rows, key=_toc_sort_key)]
    assert ordered == ["4 Downstream Results", "4.1 Fine-Tuning"]


def test_appendix_letter_rows_sort_after_numbered_rows_on_the_same_page() -> None:
    # Appendix letters ("A Switch for Attention") follow numbered sections of
    # the same destination page, and stay ordered among themselves.
    rows = [
        _row("A Switch for Attention", "27", 0),
        _row("9 Conclusion", "27", 1),
        _row("C Encouraging Exploration Across Experts", "29", 2),
        _row("B Preventing Token Dropping", "29", 3),
    ]
    ordered = [b.source_text for b in sorted(rows, key=_toc_sort_key)]
    assert ordered == [
        "9 Conclusion",
        "A Switch for Attention",
        "B Preventing Token Dropping",
        "C Encouraging Exploration Across Experts",
    ]


def test_a_deep_child_sorts_after_its_parent_and_siblings() -> None:
    # "2.3.1" > "2.3" > "2.29": tuple comparison keeps numeric parts ordered
    # numerically, not lexicographically.
    rows = [
        _row("2.3.1 Details", "9", 0),
        _row("2.29 Later", "9", 1),
        _row("2.3 Middle", "9", 2),
    ]
    ordered = [b.source_text for b in sorted(rows, key=_toc_sort_key)]
    assert ordered == ["2.3 Middle", "2.3.1 Details", "2.29 Later"]


def test_an_unparseable_page_sorts_last() -> None:
    # A malformed entry must not be reordered to the front on a guess.
    rows = [_row("Broken", "", 0), _row("1 Introduction", "3", 1)]
    ordered = [b.source_text for b in sorted(rows, key=_toc_sort_key)]
    assert ordered == ["1 Introduction", "Broken"]


# --------------------------------------------------------------------------
# The id collision: renumbered TOC rows must never reuse a taken id.
# --------------------------------------------------------------------------


class _FakeProv:
    def __init__(self, page_no: int) -> None:
        self.page_no = page_no
        self.bbox = None


class _FakeItem:
    """A minimal DOCUMENT_INDEX stand-in (the mapping only reads these attrs)."""

    label = "document_index"

    def __init__(self) -> None:
        self.prov = [_FakeProv(2)]


class _FakeDocItem:
    def __init__(self, label: str, text: str, page_no: int) -> None:
        self.label = label
        self.text = text
        self.prov = [_FakeProv(page_no)]


def _items_with_front_matter_then_toc() -> tuple[list[tuple[object, int]], object]:
    """A document whose first page (title/abstract) precedes the TOC item."""

    class _Doc:
        pass

    doc = _Doc()
    items: list[tuple[object, int]] = [
        (_FakeDocItem("text", "The Greatest Paper", 1), 0),
        (_FakeDocItem("text", "In deep learning, models reuse parameters.", 1), 0),
        (_FakeItem(), 0),
    ]
    return items, doc


def test_toc_renumbering_never_collides_with_the_blocks_it_follows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole regression in one assertion: unique ids with front matter first."""
    items, doc = _items_with_front_matter_then_toc()

    # _extract_document_index_blocks needs a real PDF; stub it to return rows
    # in the (destination, insertion) shape the real merge produces: the
    # section read first, then the chapter that contains it.
    def fake_index(
        item: object, pdf_path: object, cache: dict[int, list[object]], start_index: int
    ) -> list[IRBlock]:
        return [
            _row("4.1 Fine-Tuning", "14", start_index),
            _row("4 Downstream Results", "14", start_index + 1),
        ]

    import ubt.adapters.pdf.docling_parser as dp

    monkeypatch.setattr(dp, "_extract_document_index_blocks", fake_index)
    monkeypatch.setattr(dp, "_extract_outline_only_blocks", lambda *a, **k: [])

    blocks = map_iterated_items(items, doc, tmp_path / "toc-assets", pdf_path=None)

    ids = [b.id for b in blocks]
    assert len(ids) == len(set(ids)), f"colliding ids: {ids}"
    # The front matter keeps ids 1..2 and the TOC rows are renumbered after it.
    assert blocks[0].id == "pdf_main#b0001"
    assert blocks[0].source_text == "The Greatest Paper"
    assert blocks[1].id == "pdf_main#b0002"
    # The chapter still sorts above its own section (the destination-page order
    # cb2286a was after), now at the ids that follow the front matter.
    toc = [b for b in blocks if b.provenance.toc_entry]
    assert [b.source_text for b in toc] == ["4 Downstream Results", "4.1 Fine-Tuning"]
    assert toc[0].id == "pdf_main#b0003"
    assert toc[1].id == "pdf_main#b0004"


# --------------------------------------------------------------------------
# A contents section spanning several pages arrives as several items.
# --------------------------------------------------------------------------


def test_a_multi_page_contents_merges_the_outline_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every outline-only row is added once, not once per DOCUMENT_INDEX item.

    arXiv 2608.25512's contents runs over pages 2-3, which Docling reports as
    two items. The outline merge ran inside the per-item branch, so each item
    re-added the whole outline: 128 rows for 64 entries, every chapter printed
    twice in the delivered contents.
    """
    items: list[tuple[object, int]] = [
        (_FakeDocItem("text", "The Greatest Paper", 1), 0),
        (_FakeItem(), 0),
        (_FakeItem(), 0),
    ]

    class _Doc:
        pass

    calls: list[int] = []

    def fake_index(
        item: object, pdf_path: object, cache: dict[int, list[object]], start_index: int
    ) -> list[IRBlock]:
        # Each page of the contents supplies its own half of the rows.
        page = len(calls)
        calls.append(page)
        return [_row(f"{page + 1} Page {page + 1}", str(page + 10), start_index)]

    def fake_outline(pdf_path: object, already: list[IRBlock], start_index: int) -> list[IRBlock]:
        # Two outline-only chapters the Docling pass dropped.
        return [
            _row("A Appendix", "50", start_index),
            _row("B Appendix", "51", start_index + 1),
        ]

    import ubt.adapters.pdf.docling_parser as dp

    monkeypatch.setattr(dp, "_extract_document_index_blocks", fake_index)
    monkeypatch.setattr(dp, "_extract_outline_only_blocks", fake_outline)

    blocks = map_iterated_items(items, _Doc(), tmp_path / "toc-assets", pdf_path=None)

    titles = [b.source_text for b in blocks if b.provenance.toc_entry]
    assert titles == ["1 Page 1", "2 Page 2", "A Appendix", "B Appendix"], titles


def test_outline_rows_with_the_same_title_but_different_pages_both_survive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dedup is on (title, destination), not title alone.

    arXiv 2608.25512 lists "4.3.2 Temporal Composability" (p. 44) and
    "7.3 Temporal Composability" (p. 79). Folding on the title alone deleted a
    real section from the delivered contents.
    """
    from ubt.adapters.pdf.docling_outline import _extract_outline_only_blocks

    already = [_row("Temporal Composability", "44", 1)]

    monkeypatch.setattr(
        "ubt.adapters.pdf.docling_outline.read_pdf_outline_entries",
        lambda _p: [(1, "Temporal Composability", 44), (1, "Temporal Composability", 79)],
    )

    added = _extract_outline_only_blocks(Path("unused.pdf"), already, start_index=2)

    assert [b.provenance.toc_page for b in added] == ["79"]


# --------------------------------------------------------------------------
# Merging the rows must not move them out of the reading order.
# --------------------------------------------------------------------------


def test_the_contents_stays_where_the_source_printed_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Collecting the rows must not append them at the end of the book.

    Deferring the outline merge to after the loop fixed a real duplicate-rows
    bug (a two-page contents re-added the outline once per page), but the first
    version also *placed* the rows there. The contents then read after the body
    it describes, and the "Contents" heading was left orphaned above the first
    paragraph — measured in both the Typst render and the HTML companion, which
    order by ``spine_index``.

    The source prints the contents on page 2: front matter, then the rows, then
    the body.
    """
    items: list[tuple[object, int]] = [
        (_FakeDocItem("text", "The Greatest Paper", 1), 0),
        (_FakeDocItem("title", "Contents", 2), 0),
        (_FakeItem(), 0),
        (_FakeDocItem("text", "Body paragraph one.", 5), 0),
        (_FakeDocItem("text", "Body paragraph two.", 5), 0),
    ]

    def fake_index(
        item: object, pdf_path: object, cache: dict[int, list[object]], start_index: int
    ) -> list[IRBlock]:
        return [_row("1 Introduction", "5", start_index), _row("2 Methods", "7", start_index + 1)]

    import ubt.adapters.pdf.docling_parser as dp

    monkeypatch.setattr(dp, "_extract_document_index_blocks", fake_index)
    monkeypatch.setattr(dp, "_extract_outline_only_blocks", lambda *a, **k: [])

    blocks = map_iterated_items(items, type("_Doc", (), {})(), tmp_path / "a", pdf_path=None)

    ordered = [b.source_text for b in sorted(blocks, key=lambda b: b.spine_index)]
    assert ordered == [
        "The Greatest Paper",
        "Contents",
        "1 Introduction",
        "2 Methods",
        "Body paragraph one.",
        "Body paragraph two.",
    ]
    # Ids stay unique and encode the position they are read at.
    assert [b.id for b in sorted(blocks, key=lambda b: b.spine_index)] == [
        f"pdf_main#b{index:04d}" for index in range(1, 7)
    ]


def test_inserting_the_rows_keeps_asset_ids_and_leaves_no_collision() -> None:
    """The shift renumbers positions, never asset ids, and never collides.

    Two things must hold when rows are inserted before blocks that were already
    emitted:

    * the block that used to own ``#b{k+1}`` must move out of the way. A row
      inserted at ``k + 1`` takes that id, and the ledger's
      ``ON CONFLICT(job_id, block_id) DO UPDATE`` resolves the clash silently by
      overwriting one block with the other — which is exactly how the TOC once
      replaced the title page.
    * an asset block keeps its ``img_``/``code_``/``formula_`` prefix. Those are
      not positions; the quality gate and the renderer key on them, so a blanket
      renumber to ``#bNNNN`` would erase the distinction.
    """
    from ubt.adapters.pdf.docling_parser import _place_toc_rows

    front = [
        IRBlock(
            element=make_element(
                id="pdf_main#b0001",
                spine_index=1,
                block_type=BlockType.NARRATIVE,
                source_text="Title",
            )
        ),
        IRBlock(
            element=make_element(
                id="pdf_main#b0002",
                spine_index=2,
                block_type=BlockType.NARRATIVE,
                source_text="Abstract",
            )
        ),
    ]
    body = [
        IRBlock(
            element=make_element(
                id="pdf_main#b0003",
                spine_index=3,
                block_type=BlockType.NARRATIVE,
                source_text="Body",
            )
        ),
        IRBlock(
            element=make_element(
                id="pdf_main#img_0004",
                spine_index=4,
                block_type=BlockType.IMAGE,
                source_text="pic.png",
                skip_translate=True,
            )
        ),
        IRBlock(
            element=make_element(
                id="pdf_main#b0005",
                spine_index=5,
                block_type=BlockType.NARRATIVE,
                source_text="Tail",
            )
        ),
    ]
    rows = [
        _row("1 Introduction", "5", 0),
        _row("2 Methods", "7", 0),
    ]

    blocks = front + body
    _place_toc_rows(blocks, rows, insert_at=2)

    ids = [b.id for b in blocks]
    assert ids == [
        "pdf_main#b0001",  # front matter untouched
        "pdf_main#b0002",
        "pdf_main#b0003",  # the two inserted rows
        "pdf_main#b0004",
        "pdf_main#b0005",  # "Body", shifted out of the way
        "pdf_main#img_0004",  # asset id left exactly as emitted
        "pdf_main#b0007",  # "Tail"
    ], ids
    assert len(ids) == len(set(ids))
    assert [b.spine_index for b in blocks] == list(range(1, len(blocks) + 1))
