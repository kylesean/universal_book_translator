"""Unit tests for bilingual concordance index extraction, phonetics, and rendering."""

from __future__ import annotations

import pytest

from ubt.core.index.extractor import extract_index_from_blocks
from ubt.core.index.generator import (
    render_index_html,
    render_index_markdown,
    render_index_typst,
)
from ubt.core.index.models import ConcordanceIndex, IndexEntry, IndexEntryLocation
from ubt.core.index.phonetics import get_latin_initial, get_pinyin_initial
from ubt.core.ir.models import BlockType, IRBlock, make_element

pytestmark = pytest.mark.fast


def test_phonetics_initials() -> None:
    # Latin initial calculation
    assert get_latin_initial("Consensus") == "C"
    assert get_latin_initial("  algorithm  ") == "A"
    assert get_latin_initial("123 numbers") == "0-9"
    assert get_latin_initial("Éléphant") == "E"
    assert get_latin_initial("$$math$$") == "#"

    # Pinyin initial calculation
    assert get_pinyin_initial("共识算法") == "G"
    assert get_pinyin_initial("拜占庭") == "B"
    assert get_pinyin_initial("网络") == "W"
    assert get_pinyin_initial("系统") == "X"
    assert get_pinyin_initial("状态机") == "Z"
    assert get_pinyin_initial("Distributed") == "D"
    assert get_pinyin_initial("123") == "0-9"
    assert get_pinyin_initial("???") == "#"


def test_index_entry_model_properties() -> None:
    entry = IndexEntry(
        source_term="Consensus Algorithm",
        target_term="共识算法",
        category="Distributed Systems",
        occurrences_count=5,
    )
    assert entry.source_initial == "C"
    assert entry.target_initial == "G"
    assert entry.display_en_zh == "Consensus Algorithm (共识算法)"
    assert entry.display_zh_en == "共识算法 (Consensus Algorithm)"


def test_concordance_index_grouping_and_filtering() -> None:
    entries = [
        IndexEntry(source_term="Algorithm", target_term="算法", occurrences_count=10),
        IndexEntry(source_term="Byzantine", target_term="拜占庭", occurrences_count=4),
        IndexEntry(source_term="Consensus", target_term="共识", occurrences_count=12),
        IndexEntry(source_term="Data", target_term="数据", occurrences_count=1),
    ]
    idx = ConcordanceIndex(entries=entries)
    assert idx.total_terms == 4
    assert idx.total_occurrences == 27

    # Grouped by EN initial
    en_groups = idx.grouped_by_source_initial()
    assert list(en_groups.keys()) == ["A", "B", "C", "D"]
    assert len(en_groups["C"]) == 1
    assert en_groups["C"][0].source_term == "Consensus"

    # Grouped by ZH (Pinyin) initial: "算法"->S, "拜占庭"->B, "共识"->G, "数据"->S
    zh_groups = idx.grouped_by_target_initial()
    assert set(zh_groups.keys()) == {"B", "G", "S"}
    assert len(zh_groups["S"]) == 2  # 算法, 数据

    # Filtering by min_occurrences
    filtered = idx.filter(min_occurrences=5)
    assert filtered.total_terms == 2
    assert {e.source_term for e in filtered.entries} == {"Algorithm", "Consensus"}

    # Top terms
    top = idx.top_terms(2)
    assert len(top) == 2
    assert top[0].source_term == "Consensus"
    assert top[1].source_term == "Algorithm"


def test_extract_index_from_blocks_with_parentheses() -> None:
    el_h1 = make_element(
        id="h1",
        spine_index=0,
        block_type=BlockType.HEADING,
        source_text="Chapter 1: Foundations",
        level=1,
    )
    el_p1 = make_element(
        id="p1",
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        source_text="This chapter explains Consensus Algorithm and Byzantine Fault Tolerance.",
    )
    el_p2 = make_element(
        id="p2",
        spine_index=2,
        block_type=BlockType.NARRATIVE,
        source_text="State machine replication is also discussed.",
    )
    blocks = [
        IRBlock(element=el_h1, target_text="第 1 章：基础理论"),
        IRBlock(
            element=el_p1,
            target_text="本章主要讨论共识算法（Consensus Algorithm）以及拜占庭容错（Byzantine Fault Tolerance）。",
        ),
        IRBlock(
            element=el_p2,
            target_text="另外还涉及状态机复制（State Machine Replication）和共识算法（Consensus Algorithm）的应用。",
        ),
    ]

    idx = extract_index_from_blocks(blocks)
    assert idx.total_terms == 3
    terms_dict = {e.source_term.lower(): e for e in idx.entries}
    assert "consensus algorithm" in terms_dict
    assert terms_dict["consensus algorithm"].target_term == "共识算法"
    assert terms_dict["consensus algorithm"].occurrences_count == 2
    assert len(terms_dict["consensus algorithm"].locations) == 1
    assert terms_dict["consensus algorithm"].locations[0].section_title == "第 1 章：基础理论"


def test_extract_index_from_blocks_with_supplied_glossary() -> None:
    glossary = [
        {"source": "Raft", "target": "Raft协议", "category": "Consensus"},
        {"source": "Paxos", "target": "Paxos算法", "category": "Consensus"},
        {"source": "Vector Clock", "target": "向量时钟", "category": "Time"},
    ]
    el_h1 = make_element(
        id="h1",
        spine_index=0,
        block_type=BlockType.HEADING,
        source_text="Distributed Time",
        level=1,
    )
    el_p1 = make_element(
        id="p1",
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        source_text="We use Vector Clock for ordering events. Raft is another topic.",
    )
    blocks = [
        IRBlock(element=el_h1, target_text="分布式时钟"),
        IRBlock(element=el_p1, target_text="我们使用向量时钟来定序事件。"),
    ]

    idx = extract_index_from_blocks(blocks, glossary=glossary)
    # Vector Clock matched both in source ('Vector Clock') and target ('向量时钟') -> count = 2
    # Raft matched in source ('Raft') -> count = 1
    # Paxos not mentioned -> count = 0
    assert idx.total_terms == 2
    term_names = {e.source_term for e in idx.entries}
    assert term_names == {"Vector Clock", "Raft"}


def test_render_index_typst_html_markdown() -> None:
    entries = [
        IndexEntry(
            source_term="Consensus",
            target_term="共识",
            locations=[
                IndexEntryLocation(
                    block_id="b1", section_title="第一章", section_anchor="sec-1", page=12
                ),
                IndexEntryLocation(
                    block_id="b2", section_title="第三章", section_anchor="sec-3", page=45
                ),
            ],
            occurrences_count=5,
        ),
        IndexEntry(
            source_term="Byzantine",
            target_term="拜占庭",
            locations=[
                IndexEntryLocation(
                    block_id="b3", section_title="第二章", section_anchor="sec-2", page=28
                ),
            ],
            occurrences_count=3,
        ),
    ]
    idx = ConcordanceIndex(entries=entries)

    # Typst
    typst_code = render_index_typst(idx, language_first="en", columns=3)
    assert "#heading(level: 1, numbering: none)[索引 (Index)]" in typst_code
    assert "#columns(3" in typst_code
    assert "*Consensus* (共识)" in typst_code
    assert "#link(<sec-1>)[12]" in typst_code

    # HTML
    html_code = render_index_html(idx, language_first="en", columns=3)
    assert '<section class="ubt-concordance-index">' in html_code
    assert '<nav class="index-quick-jump">' in html_code
    assert 'href="#idx-C"' in html_code
    assert "<strong>Consensus</strong>" in html_code
    assert '<a href="#sec-1" class="index-link">12</a>' in html_code

    # Markdown
    md_code = render_index_markdown(idx, language_first="zh")
    assert "## 📑 术语索引 (Index)" in md_code
    assert "**拜占庭** (Byzantine)" in md_code
    assert "**共识** (Consensus)" in md_code
