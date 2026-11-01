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
    rows = [
        _row("2.3 Putting It All Together", "8", 0),
        _row("2.4 Improved Training", "8", 1),
    ]
    ordered = [b.source_text for b in sorted(rows, key=_toc_sort_key)]
    assert ordered == ["2.3 Putting It All Together", "2.4 Improved Training"]


def test_a_chapter_sorts_above_its_section_when_both_share_a_page() -> None:
    rows = [
        _row("4.1 Fine-Tuning", "14", 0),
        _row("4 Downstream Results", "14", 1),
    ]
    ordered = [b.source_text for b in sorted(rows, key=_toc_sort_key)]
    assert ordered == ["4 Downstream Results", "4.1 Fine-Tuning"]


def test_appendix_letter_rows_sort_after_numbered_rows_on_the_same_page() -> None:
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
    rows = [
        _row("2.3.1 Details", "9", 0),
        _row("2.29 Later", "9", 1),
        _row("2.3 Middle", "9", 2),
    ]
    ordered = [b.source_text for b in sorted(rows, key=_toc_sort_key)]
    assert ordered == ["2.3 Middle", "2.3.1 Details", "2.29 Later"]


def test_an_unparseable_page_sorts_last() -> None:
    rows = [_row("Broken", "", 0), _row("1 Introduction", "3", 1)]
    ordered = [b.source_text for b in sorted(rows, key=_toc_sort_key)]
    assert ordered == ["1 Introduction", "Broken"]


class _FakeProv:
    def __init__(self, page_no: int) -> None:
        self.page_no = page_no
        self.bbox = None


class _FakeItem:
    label = "document_index"

    def __init__(self) -> None:
        self.prov = [_FakeProv(2)]


class _FakeDocItem:
    def __init__(self, label: str, text: str, page_no: int) -> None:
        self.label = label
        self.text = text
        self.prov = [_FakeProv(page_no)]


def _items_with_front_matter_then_toc() -> tuple[list[tuple[object, int]], object]:

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
    items, doc = _items_with_front_matter_then_toc()

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
    assert blocks[0].id == "pdf_main#b0001"
    assert blocks[0].source_text == "The Greatest Paper"
    assert blocks[1].id == "pdf_main#b0002"
    toc = [b for b in blocks if b.provenance.toc_entry]
    assert [b.source_text for b in toc] == ["4 Downstream Results", "4.1 Fine-Tuning"]
    assert toc[0].id == "pdf_main#b0003"
    assert toc[1].id == "pdf_main#b0004"


def test_a_multi_page_contents_merges_the_outline_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        page = len(calls)
        calls.append(page)
        return [_row(f"{page + 1} Page {page + 1}", str(page + 10), start_index)]

    def fake_outline(pdf_path: object, already: list[IRBlock], start_index: int) -> list[IRBlock]:
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
    from ubt.adapters.pdf.docling_outline import _extract_outline_only_blocks

    already = [_row("Temporal Composability", "44", 1)]

    monkeypatch.setattr(
        "ubt.adapters.pdf.docling_outline.read_pdf_outline_entries",
        lambda _p: [(1, "Temporal Composability", 44), (1, "Temporal Composability", 79)],
    )

    added = _extract_outline_only_blocks(Path("unused.pdf"), already, start_index=2)

    assert [b.provenance.toc_page for b in added] == ["79"]


def test_the_contents_stays_where_the_source_printed_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    assert [b.id for b in sorted(blocks, key=lambda b: b.spine_index)] == [
        f"pdf_main#b{index:04d}" for index in range(1, 7)
    ]


def test_inserting_the_rows_keeps_asset_ids_and_leaves_no_collision() -> None:
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
        "pdf_main#b0001",
        "pdf_main#b0002",
        "pdf_main#b0003",
        "pdf_main#b0004",
        "pdf_main#b0005",
        "pdf_main#img_0004",
        "pdf_main#b0007",
    ], ids
    assert len(ids) == len(set(ids))
    assert [b.spine_index for b in blocks] == list(range(1, len(blocks) + 1))
