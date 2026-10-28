"""Unit tests for chapter epigraph and abstract parsing, detection, and rendering."""

from __future__ import annotations

import pytest

from ubt.core.epigraph.detector import (
    detect_chapter_epigraphs_and_abstracts,
    parse_abstract_text,
    parse_epigraph_text,
)
from ubt.core.epigraph.generator import (
    render_abstract_html,
    render_abstract_typst,
    render_epigraph_html,
    render_epigraph_typst,
)
from ubt.core.epigraph.models import (
    ChapterAbstract,
    Epigraph,
    EpigraphConfig,
    EpigraphStyle,
)
from ubt.core.ir.models import BlockType, IRBlock, make_element

pytestmark = pytest.mark.fast


def _make_block(btype: BlockType, text: str, block_id: str = "b1", level: int = 1) -> IRBlock:
    elem = make_element(
        id=block_id,
        spine_index=1,
        block_type=btype,
        source_text=text,
        level=level,
    )
    return IRBlock(
        id=block_id,
        type=btype,
        source_text=text,
        target_text=text,
        element=elem,
    )


def test_parse_epigraph_multiline() -> None:
    text = (
        '> "Premature optimization is the root of all evil."\n'
        "> —— Donald E. Knuth, The Art of Computer Programming"
    )
    res = parse_epigraph_text(text)
    assert res is not None
    quote, author = res
    assert quote == "Premature optimization is the root of all evil."
    assert author == "Donald E. Knuth, The Art of Computer Programming"


def test_parse_epigraph_cjk_inline() -> None:
    text = "知之为知之，不知为不知，是知也。 —— 《论语·为政》"
    res = parse_epigraph_text(text)
    assert res is not None
    quote, author = res
    assert quote == "知之为知之，不知为不知，是知也。"
    assert author == "《论语·为政》"


def test_parse_abstract_text_chinese() -> None:
    text = "**本章导读**：本章系统性介绍分布式共识协议的基础理论与 Raft 实现细节。"
    res = parse_abstract_text(text)
    assert res is not None
    title, body = res
    assert title == "本章导读"
    assert "分布式共识协议的基础理论" in body


def test_parse_abstract_text_english() -> None:
    text = "> **Chapter Summary**: This chapter covers memory hierarchies and cache line coherence."
    res = parse_abstract_text(text)
    assert res is not None
    title, body = res
    assert title == "Chapter Summary"
    assert "memory hierarchies" in body


def test_detect_chapter_epigraphs_and_abstracts() -> None:
    blocks = [
        _make_block(BlockType.HEADING, "第 1 章 系统基础", block_id="h1", level=1),
        _make_block(
            BlockType.NARRATIVE,
            '> "Simplicity is prerequisite for reliability."\n> —— Edsger W. Dijkstra',
            block_id="p_epi",
        ),
        _make_block(
            BlockType.NARRATIVE,
            "**本章导读**：本章介绍系统可靠性设计的核心准则与评估指标。",
            block_id="p_abs",
        ),
        _make_block(
            BlockType.NARRATIVE, "在现代软件工程中，复杂性是最大的敌人。", block_id="p_body"
        ),
    ]

    epigraphs, abstracts = detect_chapter_epigraphs_and_abstracts(blocks)
    assert "p_epi" in epigraphs
    assert epigraphs["p_epi"].text == "Simplicity is prerequisite for reliability."
    assert epigraphs["p_epi"].source == "Edsger W. Dijkstra"
    assert epigraphs["p_epi"].chapter_index == 1

    assert "p_abs" in abstracts
    assert abstracts["p_abs"].title == "本章导读"
    assert "核心准则与评估指标" in abstracts["p_abs"].text


def test_render_epigraph_typst_styles() -> None:
    epi = Epigraph(
        text="Everything should be made as simple as possible, but not simpler.",
        source="Albert Einstein",
    )

    # Classic right
    cfg_classic = EpigraphConfig(style=EpigraphStyle.CLASSIC_RIGHT)
    out_classic = render_epigraph_typst(epi, config=cfg_classic)
    assert "#align(right)[" in out_classic
    assert "width: 75%" in out_classic
    assert "Albert Einstein" in out_classic

    # Border left
    cfg_border = EpigraphConfig(style=EpigraphStyle.BORDER_LEFT)
    out_border = render_epigraph_typst(epi, config=cfg_border)
    assert "stroke: (left: 2.5pt + luma(160))" in out_border

    # Callout box
    cfg_box = EpigraphConfig(style=EpigraphStyle.CALLOUT_BOX)
    out_box = render_epigraph_typst(epi, config=cfg_box)
    assert "fill: luma(250)" in out_box


def test_render_abstract_typst() -> None:
    ab = ChapterAbstract(
        title="本章提要",
        text="本章探讨现代操作系统的虚拟内存管理与缺页异常处理机制。",
    )
    out = render_abstract_typst(ab)
    assert "#block(" in out
    assert "fill: luma(248)" in out
    assert "本章提要" in out
    assert "虚拟内存管理" in out


def test_render_epigraph_and_abstract_html() -> None:
    epi = Epigraph(text="Stay hungry, stay foolish.", source="Steve Jobs")
    html_epi = render_epigraph_html(epi)
    assert '<blockquote class="ubt-epigraph' in html_epi
    assert "Stay hungry, stay foolish." in html_epi
    assert "Steve Jobs" in html_epi

    ab = ChapterAbstract(title="Summary", text="Chapter overview.")
    html_ab = render_abstract_html(ab)
    assert '<aside class="ubt-chapter-abstract">' in html_ab
    assert "Summary" in html_ab
    assert "Chapter overview." in html_ab
