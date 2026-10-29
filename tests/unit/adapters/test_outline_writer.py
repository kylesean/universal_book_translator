"""PDF bookmark write-back: entry construction and the page invariant.

The gate these tests defend is not "an outline was written" but "the outline
names the pages the reader will actually turn to". Those are different claims:
a bookmark tree of translated titles pointing at the wrong pages is worse than
no tree at all, because the reader trusts it and is taken somewhere else. So the
tests split in two — the entry builder (pure, no PDF) and the writer (a real
``pikepdf`` round trip on a generated PDF), with the page-count check exercised
from both sides.
"""

from __future__ import annotations

from pathlib import Path

import pikepdf
import pytest

from ubt.adapters.pdf.outline_writer import (
    MAX_OUTLINE_ENTRIES,
    MAX_OUTLINE_LEVEL,
    OutlineEntry,
    build_outline_entries,
    write_outline,
    write_outline_from_blocks,
)
from ubt.core.ir.models import BlockType, BoundingBox, FlowID, IRBlock, make_element

pytestmark = pytest.mark.fast


def _heading(
    block_id: str,
    text: str,
    page: int,
    *,
    level: int = 1,
    source: str | None = None,
) -> IRBlock:
    """A heading block with a measured bbox and a translated title."""
    return IRBlock(
        element=make_element(
            id=block_id,
            spine_index=page,
            block_type=BlockType.HEADING,
            flow_id=FlowID.MAIN_STORY,
            source_text=source if source is not None else text,
            bbox=BoundingBox(page=page, x0=72.0, y0=700.0, x1=500.0, y1=720.0),
            level=level,
        ),
        target_text=text,
    )


def _paragraph(block_id: str, page: int) -> IRBlock:
    return IRBlock(
        element=make_element(
            id=block_id,
            spine_index=page,
            block_type=BlockType.NARRATIVE,
            flow_id=FlowID.MAIN_STORY,
            source_text="body text",
            bbox=BoundingBox(page=page, x0=72.0, y0=600.0, x1=500.0, y1=690.0),
        ),
        target_text="正文",
    )


def _blank_pdf(path: Path, pages: int) -> Path:
    pdf = pikepdf.new()
    for _ in range(pages):
        pdf.add_blank_page(page_size=(595, 842))
    pdf.save(path)
    pdf.close()
    return path


def _read_outline(path: Path) -> list[tuple[str, int, int]]:
    """``(title, page, level)`` for every bookmark, depth-first.

    ``1 + pdf.pages.index(...)`` is the only reliable way back from a
    destination to a page number: pikepdf exposes ``Object.index`` as a *bound
    method* on the stub, so reading ``dest[0].index`` yields a callable rather
    than an integer and a naive read silently reports page 0 for every entry.
    """
    result: list[tuple[str, int, int]] = []

    def page_of(pdf: pikepdf.Pdf, item: object) -> int:
        dest = item.destination  # type: ignore[attr-defined]
        if dest is None:
            return 0
        try:
            return pdf.pages.index(dest[0]) + 1
        except Exception:
            return 0

    def walk(pdf: pikepdf.Pdf, items: object, level: int) -> None:
        for item in items:  # type: ignore[attr-defined]
            result.append((str(item.title), page_of(pdf, item), level))
            walk(pdf, item.children, level + 1)

    with pikepdf.open(path) as pdf, pdf.open_outline() as outline:
        walk(pdf, outline.root, 1)
    return result


# --------------------------------------------------------------------------- #
# Entry construction (pure)
# --------------------------------------------------------------------------- #
def test_entries_use_the_translated_title_not_the_source() -> None:
    """The sidebar must speak the reader's language, like the rest of the book."""
    block = _heading("h1", "第1章 引言", 1, source="1 Introduction")
    entries = build_outline_entries([block], max_page=10)
    assert [e.title for e in entries] == ["第1章 引言"]


def test_entries_skip_non_headings_and_untitled_headings() -> None:
    blocks = [
        _heading("h1", "第1章", 1),
        _paragraph("p1", 2),
        _heading("h2", "   ", 3),
    ]
    entries = build_outline_entries(blocks, max_page=10)
    assert [e.title for e in entries] == ["第1章"]


def test_entries_drop_headings_past_the_artifact_end() -> None:
    """A stale block from a resumed or page-limited run must not create a
    bookmark to a page that does not exist — readers error on open."""
    blocks = [_heading("h1", "第1章", 3), _heading("h2", "第2章", 99)]
    entries = build_outline_entries(blocks, max_page=10)
    assert [e.title for e in entries] == ["第1章"]


def test_entries_drop_headings_without_a_measured_page() -> None:
    block = IRBlock(
        element=make_element(
            id="h1",
            spine_index=1,
            block_type=BlockType.HEADING,
            flow_id=FlowID.MAIN_STORY,
            source_text="Chapter",
            level=1,
        ),
        target_text="第1章",
    )
    assert build_outline_entries([block], max_page=10) == []


def test_entries_skip_levels_deeper_than_the_cap() -> None:
    blocks = [
        _heading("h1", "一", 1, level=1),
        _heading("h2", "一.一", 2, level=2),
        _heading("h4", "太深", 3, level=MAX_OUTLINE_LEVEL + 1),
    ]
    entries = build_outline_entries(blocks, max_page=10)
    assert [e.title for e in entries] == ["一", "一.一"]


def test_entries_deduplicate_a_heading_split_across_a_page_break() -> None:
    """The same title on the same page is one bookmark, not two."""
    blocks = [_heading("h1", "第3章 方法", 5), _heading("h2", "第3章 方法", 5)]
    entries = build_outline_entries(blocks, max_page=10)
    assert len(entries) == 1


def test_entries_are_capped() -> None:
    blocks = [_heading(f"h{i}", f"第{i}章", 1, level=1) for i in range(MAX_OUTLINE_ENTRIES + 50)]
    assert len(build_outline_entries(blocks, max_page=10)) == MAX_OUTLINE_ENTRIES


def test_entries_unwrap_quarantine_markers_from_the_title() -> None:
    """A quarantined heading ships its SOURCE under a ``<mark>``; the literal
    markup must not become a bookmark label."""
    block = IRBlock(
        element=make_element(
            id="h1",
            spine_index=1,
            block_type=BlockType.HEADING,
            flow_id=FlowID.MAIN_STORY,
            source_text="2 Methods",
            bbox=BoundingBox(page=2, x0=72.0, y0=700.0, x1=500.0, y1=720.0),
            level=1,
        ),
        target_text='<mark class="ubt-blocked-human" title="boom">2 Methods</mark>',
    )
    entries = build_outline_entries([block], max_page=10)
    assert entries and "<mark" not in entries[0].title


# --------------------------------------------------------------------------- #
# Writing (real pikepdf round trip)
# --------------------------------------------------------------------------- #
def test_write_outline_nests_by_level_and_targets_the_right_pages(tmp_path: Path) -> None:
    pdf = _blank_pdf(tmp_path / "out.pdf", 6)
    written = write_outline(
        pdf,
        [
            OutlineEntry("第1章 引言", 1, 1),
            OutlineEntry("1.1 背景", 2, 2),
            OutlineEntry("1.2 目标", 3, 2),
            OutlineEntry("第2章 方法", 5, 1),
        ],
    )
    assert written == 4
    assert _read_outline(pdf) == [
        ("第1章 引言", 1, 1),
        ("1.1 背景", 2, 2),
        ("1.2 目标", 3, 2),
        ("第2章 方法", 5, 1),
    ]


def test_write_outline_sets_page_mode_so_readers_show_the_sidebar(tmp_path: Path) -> None:
    """Bookmarks nobody can see read as unimplemented to the person holding
    the file."""
    pdf = _blank_pdf(tmp_path / "out.pdf", 2)
    write_outline(pdf, [OutlineEntry("第1章", 1, 1)])
    with pikepdf.open(pdf) as opened:
        assert str(opened.Root.get("/PageMode")) == "/UseOutlines"


def test_write_outline_drops_entries_past_the_page_count(tmp_path: Path) -> None:
    pdf = _blank_pdf(tmp_path / "out.pdf", 3)
    written = write_outline(
        pdf,
        [OutlineEntry("第1章", 1, 1), OutlineEntry("幽灵", 99, 1)],
    )
    assert written == 1
    assert [t for t, _, _ in _read_outline(pdf)] == ["第1章"]


def test_write_outline_replaces_a_preexisting_tree(tmp_path: Path) -> None:
    """The delivered PDF is a fresh compile; a surviving source tree would put
    untranslated titles in the sidebar."""
    pdf = _blank_pdf(tmp_path / "out.pdf", 3)
    write_outline(pdf, [OutlineEntry("旧标题", 1, 1)])
    write_outline(pdf, [OutlineEntry("第1章", 2, 1)])
    assert _read_outline(pdf) == [("第1章", 2, 1)]


def test_write_outline_with_no_entries_changes_nothing(tmp_path: Path) -> None:
    pdf = _blank_pdf(tmp_path / "out.pdf", 2)
    before = pdf.read_bytes()
    assert write_outline(pdf, []) == 0
    assert pdf.read_bytes() == before


def test_write_outline_never_raises_on_a_corrupt_file(tmp_path: Path) -> None:
    """An outline is navigation, not content: it must not fail a render that
    already produced a good artifact."""
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf")
    assert write_outline(bad, [OutlineEntry("第1章", 1, 1)]) == 0


# --------------------------------------------------------------------------- #
# The page invariant
# --------------------------------------------------------------------------- #
def test_from_blocks_writes_the_translated_tree(tmp_path: Path) -> None:
    pdf = _blank_pdf(tmp_path / "out.pdf", 5)
    blocks = [
        _heading("h1", "第1章 引言", 1, level=1),
        _heading("h2", "1.1 背景", 2, level=2),
        _paragraph("p1", 2),
    ]
    assert write_outline_from_blocks(pdf, blocks, expected_pages=5) == 2
    assert _read_outline(pdf) == [("第1章 引言", 1, 1), ("1.1 背景", 2, 2)]


def test_from_blocks_refuses_when_the_artifact_page_count_moved(tmp_path: Path) -> None:
    """The reflow channel changes the page count, so ``bbox.page`` stops naming
    a delivered page. Writing the tree anyway would mislabel every bookmark;
    refusing keeps the reader on real pages."""
    pdf = _blank_pdf(tmp_path / "out.pdf", 7)
    blocks = [_heading("h1", "第1章", 1, level=1)]
    assert write_outline_from_blocks(pdf, blocks, expected_pages=5) == 0
    assert _read_outline(pdf) == []


def test_from_blocks_without_an_expectation_trusts_the_artifact(tmp_path: Path) -> None:
    pdf = _blank_pdf(tmp_path / "out.pdf", 4)
    blocks = [_heading("h1", "第1章", 3, level=1)]
    assert write_outline_from_blocks(pdf, blocks) == 1
    assert _read_outline(pdf) == [("第1章", 3, 1)]


@pytest.mark.parametrize("pages", [1, 3, 12])
def test_roundtrip_page_numbers_are_one_based(tmp_path: Path, pages: int) -> None:
    """``OutlineItem`` takes a 0-based page index; every other layer in this
    project speaks 1-based pages. Off-by-one here sends the reader to the
    previous chapter, which looks like working navigation in a spot check."""
    pdf = _blank_pdf(tmp_path / "out.pdf", pages)
    write_outline(pdf, [OutlineEntry("末章", pages, 1)])
    assert _read_outline(pdf) == [("末章", pages, 1)]
