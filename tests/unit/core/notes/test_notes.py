"""Unit tests for Footnote & Endnote Flow Manager."""

from __future__ import annotations

import pytest

from ubt.core.ir.models import BlockType, FlowID, IRBlock, make_element
from ubt.core.notes.extractor import extract_notes_topology
from ubt.core.notes.manager import (
    FootnoteFlowManager,
    render_endnote_section_html,
    render_endnote_section_markdown,
    render_footnote_inline_typst,
)
from ubt.core.notes.models import (
    FootnoteItem,
    NoteNumberingScope,
    NoteNumberingStyle,
    NotePlacement,
    NotesConfig,
    NotesTopology,
)


def _mk_block(
    b_id: str,
    b_type: BlockType,
    target: str = "",
    source: str = "",
    level: int = 1,
    flow_id: FlowID = FlowID.MAIN_STORY,
) -> IRBlock:
    el = make_element(
        id=b_id,
        spine_index=1,
        block_type=b_type,
        source_text=source or target,
        level=level,
        flow_id=flow_id,
    )
    return IRBlock(
        id=b_id,
        block_type=b_type,
        source_text=source or target,
        target_text=target or source,
        element=el,
        flow_id=flow_id,
    )


@pytest.mark.fast
def test_notes_models_formatting_and_dict() -> None:
    item = FootnoteItem(
        note_id="fn-ch1-1",
        index_num=1,
        callout_marker="[^1]",
        content="This is an explanatory footnote.",
        chapter_index=1,
    )
    assert item.formatted_marker(NoteNumberingStyle.ARABIC) == "1"
    assert item.formatted_marker(NoteNumberingStyle.CIRCLED) == "①"
    assert item.formatted_marker(NoteNumberingStyle.ROMAN) == "i"
    assert item.formatted_marker(NoteNumberingStyle.SYMBOL) == "*"

    d = item.to_dict()
    assert d["note_id"] == "fn-ch1-1"
    assert d["index_num"] == 1
    assert d["callout_marker"] == "[^1]"

    cfg = NotesConfig()
    cfg_d = cfg.to_dict()
    assert cfg_d["placement"] == "footnote"
    assert cfg_d["numbering_scope"] == "per_chapter"

    top = NotesTopology(notes=[item])
    assert top.total_notes == 1
    assert top.resolution_rate == 1.0


@pytest.mark.fast
def test_extract_notes_topology_markdown() -> None:
    blocks = [
        _mk_block(
            "h1",
            BlockType.HEADING,
            source="Chapter 1: Storage",
            target="第 1 章：存储系统",
        ),
        _mk_block(
            "p1",
            BlockType.NARRATIVE,
            source="LSM-tree is used widely[^1].",
            target="LSM-tree 被广泛应用[^1]。",
        ),
        _mk_block(
            "fn1",
            BlockType.NARRATIVE,
            source="[^1]: See Bigtable paper.",
            target="[^1]: 参见 Bigtable 论文。",
            flow_id=FlowID.FOOTNOTE,
        ),
    ]

    top = extract_notes_topology(blocks)
    assert top.total_notes == 1
    assert top.resolution_rate == 1.0
    note = top.notes[0]
    assert note.note_id == "fn-ch1-1"
    assert note.index_num == 1
    assert note.callout_marker == "[^1]"
    assert "Bigtable 论文" in note.content
    assert note.source_block_id == "p1"
    assert note.def_block_id == "fn1"
    assert "fn1" in top.def_block_ids


@pytest.mark.fast
def test_extract_notes_topology_per_chapter_and_continuous() -> None:
    blocks = [
        _mk_block("h1", BlockType.HEADING, target="第 1 章"),
        _mk_block("p1", BlockType.NARRATIVE, target="概念一[^1]"),
        _mk_block("fn1", BlockType.NARRATIVE, target="[^1]: 解释一", flow_id=FlowID.FOOTNOTE),
        _mk_block("h2", BlockType.HEADING, target="第 2 章"),
        _mk_block("p2", BlockType.NARRATIVE, target="概念二[^1]"),
        _mk_block("fn2", BlockType.NARRATIVE, target="[^1]: 解释二", flow_id=FlowID.FOOTNOTE),
    ]

    # Per chapter
    top_ch = extract_notes_topology(
        blocks, NotesConfig(numbering_scope=NoteNumberingScope.PER_CHAPTER)
    )
    assert len(top_ch.notes) == 2
    assert top_ch.notes[0].note_id == "fn-ch1-1"
    assert top_ch.notes[0].index_num == 1
    assert top_ch.notes[1].note_id == "fn-ch2-1"
    assert top_ch.notes[1].index_num == 1

    # Continuous
    top_cont = extract_notes_topology(
        blocks, NotesConfig(numbering_scope=NoteNumberingScope.CONTINUOUS)
    )
    assert len(top_cont.notes) == 2
    assert top_cont.notes[0].note_id == "fn-1"
    assert top_cont.notes[0].index_num == 1
    assert top_cont.notes[1].note_id == "fn-2"
    assert top_cont.notes[1].index_num == 2


@pytest.mark.fast
def test_extract_notes_topology_orphans() -> None:
    blocks = [
        _mk_block("p1", BlockType.NARRATIVE, target="未定义的引用[^missing]"),
        _mk_block(
            "fn1", BlockType.NARRATIVE, target="[^orphan]: 没有调用的注释", flow_id=FlowID.FOOTNOTE
        ),
    ]
    top = extract_notes_topology(blocks)
    assert top.total_notes == 1
    assert len(top.orphan_definitions) == 1
    assert len(top.orphan_callouts) == 1
    assert top.resolution_rate < 1.0


@pytest.mark.fast
def test_manager_transform_blocks_footnote_inline() -> None:
    blocks = [
        _mk_block(
            "p1",
            BlockType.NARRATIVE,
            source="Consistent hashing[^1] distributes keys.",
            target="一致性哈希[^1]分散键空间。",
        ),
        _mk_block(
            "fn1",
            BlockType.NARRATIVE,
            source="[^1]: Karger et al., 1997.",
            target="[^1]: Karger 等人，1997年。",
            flow_id=FlowID.FOOTNOTE,
        ),
    ]

    top = extract_notes_topology(blocks)
    mgr = FootnoteFlowManager(NotesConfig(placement=NotePlacement.FOOTNOTE))
    res = mgr.transform_blocks(blocks, top)

    # Footnote definition block is consumed
    assert len(res) == 1
    assert res[0].id == "p1"
    assert "#footnote[Karger 等人，1997年。]" in res[0].target_text


@pytest.mark.fast
def test_manager_transform_blocks_chapter_endnote() -> None:
    blocks = [
        _mk_block("h1", BlockType.HEADING, target="第 1 章：共识"),
        _mk_block("p1", BlockType.NARRATIVE, target="Paxos 协议[^1]是经典算法。"),
        _mk_block(
            "fn1", BlockType.NARRATIVE, target="[^1]: Lamport, 1998.", flow_id=FlowID.FOOTNOTE
        ),
    ]

    cfg = NotesConfig(placement=NotePlacement.CHAPTER_ENDNOTE)
    top = extract_notes_topology(blocks, cfg)
    mgr = FootnoteFlowManager(cfg)
    res = mgr.transform_blocks(blocks, top)

    # Contains heading, transformed p1 with link, and endnote section block
    assert len(res) == 3
    assert "#link(<fn-ch1-1>)[#super[1]]" in res[1].target_text
    assert res[2].id == "endnotes-ch1"
    assert "Lamport, 1998." in res[2].target_text


@pytest.mark.fast
def test_render_endnotes_html_and_markdown() -> None:
    item = FootnoteItem(
        note_id="fn-1",
        index_num=1,
        callout_marker="[^1]",
        content="Detailed explanation.",
    )
    html_out = render_endnote_section_html([item])
    assert '<section class="ubt-endnotes"' in html_out
    assert '<li id="fn-1" value="1">' in html_out
    assert 'class="ubt-backlink"' in html_out

    md_out = render_endnote_section_markdown([item])
    assert md_out.strip() == "[^1]: Detailed explanation."

    typst_out = render_footnote_inline_typst(item)
    assert typst_out == "#footnote[Detailed explanation.]"
