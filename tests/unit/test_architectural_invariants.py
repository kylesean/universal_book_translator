"""Tests for the 6 Deterministic Typographic Invariants, Skeleton Bible Extraction,
and Language-Aware Token/Cost Estimation."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from ubt.adapters.pdf.rigid.typesetter import UPSCALE_MAX, RigidTypesetter
from ubt.adapters.pdf.rigid.zones import PageFacts, Zone, build_zones
from ubt.adapters.pdf.textgeom import LineBox
from ubt.core.engine.cost_estimate import (
    count_text_tokens,
    resolve_output_token_ratio,
)
from ubt.core.ir.models import BlockType, BoundingBox, FlowID, IRBlock, LayoutRole


@pytest.mark.fast
def test_invariant1_linebox_and_zones_preserve_ground_truth_font_size_and_style() -> None:
    """Zones must use the PDF ground-truth font_size, bold, and italic from LineBox
    instead of guessing from ink bounding-box height (y1 - y0) * 0.95."""
    # Simulate '1. Introduction' (13pt Bold, ink height only 9.0pt due to no descenders)
    # and '1.2.3. The Coarse-Grained Workaround' (11pt BoldItalic, ink height only 8.2pt)
    # and body paragraph (11pt Regular, ink height 11.2pt).
    h1_line = LineBox(
        "1. Introduction",
        (71.0, 732.0, 165.0, 741.0),
        font_size=13.0,
        bold=True,
        italic=False,
    )
    h3_line = LineBox(
        "1.2.3. The Coarse-Grained Workaround",
        (71.0, 172.0, 266.0, 180.2),
        font_size=11.0,
        bold=True,
        italic=True,
    )
    body_line1 = LineBox(
        "Composition assembling complex systems from simpler parts is a foundational",
        (71.0, 701.0, 527.0, 712.2),
        font_size=11.0,
        bold=False,
        italic=False,
    )
    body_line2 = LineBox(
        "principle of software engineering and modern programming languages.",
        (71.0, 688.0, 527.0, 699.2),
        font_size=11.0,
        bold=False,
        italic=False,
    )
    facts = PageFacts(
        page=4,
        width=595.3,
        height=841.9,
        lines=(h1_line, body_line1, body_line2, h3_line),
    )
    b_h1 = IRBlock(
        id="h1",
        flow_id=FlowID.MAIN_STORY,
        spine_index=1,
        block_type=BlockType.HEADING,
        source_text="1. Introduction",
        target_text="1 引言",
        bbox=BoundingBox(page=4, x0=71.0, y0=732.0, x1=165.0, y1=741.0),
    )
    b_body = IRBlock(
        id="body",
        flow_id=FlowID.MAIN_STORY,
        spine_index=2,
        block_type=BlockType.NARRATIVE,
        source_text=(
            "Composition assembling complex systems from simpler parts is a foundational "
            "principle of software engineering and modern programming languages."
        ),
        target_text="组合是由简单部件组装复杂系统的一项软件工程基本原则。",
        bbox=BoundingBox(page=4, x0=71.0, y0=688.0, x1=527.0, y1=712.2),
    )
    b_h3 = IRBlock(
        id="h3",
        flow_id=FlowID.MAIN_STORY,
        spine_index=3,
        block_type=BlockType.HEADING,
        source_text="1.2.3. The Coarse-Grained Workaround",
        target_text="1.2.3 粗粒度权宜方案",
        bbox=BoundingBox(page=4, x0=71.0, y0=172.0, x1=266.0, y1=180.2),
    )

    zone_map = build_zones({4: facts}, [b_h1, b_body, b_h3])
    z_h1 = zone_map["h1"][0]
    z_body = zone_map["body"][0]
    z_h3 = zone_map["h3"][0]

    assert z_h1.base_size == pytest.approx(13.0)
    assert z_h1.bold is True
    assert z_body.base_size == pytest.approx(11.0)
    assert z_body.bold is False
    assert z_h3.base_size == pytest.approx(11.0)
    assert z_h3.bold is True
    assert z_h3.italic is True


@pytest.mark.fast
def test_invariant2_no_upscale_above_source_font_size_and_preserves_bold_italic() -> None:
    """UPSCALE_MAX must be 1.0 so compact translations never inflate body text above
    headings, and _zone_typst must emit bold/italic styling."""
    assert pytest.approx(1.0) == UPSCALE_MAX

    ts = RigidTypesetter(target_lang="zh")
    z_heading = Zone(
        block_id="h1",
        page=1,
        x0=70.0,
        y0=728.0,
        x1=527.0,
        y1=746.0,
        base_size=13.0,
        rows=("1. Introduction",),
        bold=True,
        italic=False,
        align="left",
    )
    z_body = Zone(
        block_id="body",
        page=1,
        x0=70.0,
        y0=580.0,
        x1=527.0,
        y1=712.0,
        base_size=11.0,
        rows=("Long English paragraph...",),
        bold=False,
        italic=False,
        align="justify",
    )
    b_h1 = IRBlock(
        id="h1",
        flow_id=FlowID.MAIN_STORY,
        spine_index=1,
        block_type=BlockType.HEADING,
        source_text="1. Introduction",
        target_text="1 引言",
        bbox=BoundingBox(page=1, x0=70.0, y0=728.0, x1=165.0, y1=746.0),
    )
    b_body = IRBlock(
        id="body",
        flow_id=FlowID.MAIN_STORY,
        spine_index=2,
        block_type=BlockType.NARRATIVE,
        source_text="Long English paragraph...",
        target_text="简短中文段落。",
        bbox=BoundingBox(page=1, x0=70.0, y0=580.0, x1=527.0, y1=712.0),
    )
    paints, _report = ts._plan_blocks(
        [b_h1, b_body],
        {"h1": (z_heading,), "body": (z_body,)},
        {1: 841.9},
    )
    by_id = {entry.zone.block_id: entry for entry in paints[1]}
    assert by_id["h1"].size == pytest.approx(13.0)
    assert by_id["body"].size == pytest.approx(11.0)
    assert by_id["h1"].size > by_id["body"].size

    h1_typst = ts._zone_typst(by_id["h1"], 841.9)
    assert 'weight: "bold"' in h1_typst
    body_typst = ts._zone_typst(by_id["body"], 841.9)
    assert "justify: true" in body_typst


@pytest.mark.fast
def test_invariant3_and_4_center_alignment_and_margin_expansion_prevent_clipping() -> None:
    """Centered blocks (title, author, abstract) must detect align='center' and expand
    symmetrically across the page width; short single-line footnotes must expand rightwards
    into blank margin so trailing words ('Marketplace。') never wrap or clip."""
    title_line = LineBox(
        "A Programming Paradigm for Spatiotemporal Composability",
        (75.7, 709.7, 519.8, 725.7),
        font_size=16.0,
        bold=True,
    )
    author_line = LineBox(
        "Yifan Shi 1,2 , Wei Zhang 1 , Tianyi Cui 2",
        (209.4, 667.8, 385.7, 680.6),
        font_size=11.0,
        bold=False,
    )
    abstract_line = LineBox(
        "Abstract",
        (269.3, 594.9, 326.2, 606.1),
        font_size=15.0,
        bold=True,
    )
    body_line = LineBox(
        "Modern software from plugin systems to self-evolving agent harnesses increasingly",
        (70.9, 540.0, 529.4, 554.0),
        font_size=11.0,
        bold=False,
    )
    footnote_line = LineBox(
        "1 Data retrieved from the Visual Studio Code Marketplace on June 9, 2026.",
        (79.9, 82.6, 371.0, 93.1),
        font_size=9.0,
        bold=False,
    )
    facts = PageFacts(
        page=1,
        width=595.3,
        height=841.9,
        lines=(title_line, author_line, abstract_line, body_line, footnote_line),
    )
    blocks = [
        IRBlock(
            id="title",
            flow_id=FlowID.MAIN_STORY,
            spine_index=1,
            block_type=BlockType.HEADING,
            source_text=title_line.text,
            target_text="时空可组合性的编程范式",
            bbox=BoundingBox(page=1, x0=75.7, y0=709.7, x1=519.8, y1=725.7),
        ),
        IRBlock(
            id="author",
            flow_id=FlowID.MAIN_STORY,
            spine_index=2,
            block_type=BlockType.NARRATIVE,
            source_text=author_line.text,
            target_text="Yifan Shi 1,2，Wei Zhang 1，Tianyi Cui 2",
            bbox=BoundingBox(page=1, x0=209.4, y0=667.8, x1=385.7, y1=680.6),
        ),
        IRBlock(
            id="abstract",
            flow_id=FlowID.MAIN_STORY,
            spine_index=3,
            block_type=BlockType.HEADING,
            source_text="Abstract",
            target_text="摘要",
            bbox=BoundingBox(page=1, x0=269.3, y0=594.9, x1=326.2, y1=606.1),
        ),
        IRBlock(
            id="footnote",
            flow_id=FlowID.MAIN_STORY,
            spine_index=4,
            block_type=BlockType.NARRATIVE,
            layout_role=LayoutRole.FOOTNOTE,
            source_text=footnote_line.text,
            target_text="1 数据取自 2026 年 6 月 9 日的 Visual Studio Code Marketplace。",
            bbox=BoundingBox(page=1, x0=79.9, y0=82.6, x1=371.0, y1=93.1),
        ),
    ]
    zones = build_zones({1: facts}, blocks)
    assert zones["title"][0].align == "center"
    assert zones["author"][0].align == "center"
    # Author zone width expanded well beyond the tight 176pt box so full-width commas fit at 11pt
    assert zones["author"][0].width >= 400.0
    assert zones["abstract"][0].align == "center"
    assert zones["abstract"][0].base_size == pytest.approx(15.0)
    # Footnote expanded rightwards into unoccupied margin so 'Marketplace。' fits on 1 line at 9pt
    assert zones["footnote"][0].width >= 430.0
    assert zones["footnote"][0].base_size == pytest.approx(9.0)


@pytest.mark.fast
def test_skeleton_terminology_extractor_extracts_llm_glossary_across_languages() -> None:
    """Tier-2 LLM skeleton extractor must sample document skeleton and extract
    domain terms (e.g., coeffect -> 余效应) without relying on regex miners."""
    from ubt.core.memory.skeleton_extractor import extract_skeleton_terms_llm

    blocks = [
        IRBlock(
            id="b1",
            flow_id=FlowID.MAIN_STORY,
            spine_index=1,
            block_type=BlockType.HEADING,
            source_text="A Programming Paradigm for Spatiotemporal Composability",
        ),
        IRBlock(
            id="b2",
            flow_id=FlowID.MAIN_STORY,
            spine_index=2,
            block_type=BlockType.NARRATIVE,
            source_text=(
                "We formalize revertible effects and reactive coeffects inside a unified "
                "context paradigm for self-evolving agent harnesses."
            ),
        ),
    ]

    async def fake_complete_raw(system_prompt: str, user_prompt: str) -> str:
        assert "A Programming Paradigm" in user_prompt
        assert "reactive coeffects" in user_prompt
        return (
            '{"domain": "Programming Language Theory", "terms": ['
            '{"source": "coeffect", "translation": "余效应", "kind": "term"},'
            '{"source": "reactive coeffects", "translation": "反应式余效应", "kind": "term"},'
            '{"source": "agent harness", "translation": "智能体运行框架", "kind": "term"}'
            "]}"
        )

    entries = asyncio.run(
        extract_skeleton_terms_llm(
            blocks,
            complete_raw_fn=fake_complete_raw,
            source_lang="en",
            target_lang="zh",
        )
    )
    by_src = {e.source: e.translation for e in entries}
    assert by_src["coeffect"] == "余效应"
    assert by_src["reactive coeffects"] == "反应式余效应"
    assert by_src["agent harness"] == "智能体运行框架"


@pytest.mark.fast
def test_language_aware_token_and_fertility_estimation() -> None:
    """Token estimator must accurately count CJK/non-Latin scripts (not len//4)
    and apply directional language-pair output token ratios instead of a hardcoded 1.5."""
    zh_text = (
        "从插件系统到自演进智能体框架，现代软件日益需要动态组合，而其形式化基础仍不完善。" * 10
    )
    # 400 Chinese characters are ~250-450 BPE tokens, NEVER 400 // 4 = 100 tokens!
    zh_tokens = count_text_tokens(zh_text)
    assert zh_tokens >= 200

    # Directional output token ratio: CJK -> EN shrinks in tokens (< 1.05),
    # whereas EN -> DE/RU or EN -> ZH has language-pair specific ratio.
    ratio_zh_to_en = resolve_output_token_ratio("zh", "en")
    ratio_en_to_zh = resolve_output_token_ratio("en", "zh")
    ratio_en_to_ru = resolve_output_token_ratio("en", "ru")
    assert ratio_zh_to_en < 1.05
    assert 1.05 <= ratio_en_to_zh <= 1.40
    assert ratio_en_to_ru > ratio_en_to_zh


@pytest.mark.fast
def test_invariant5_geometric_superscript_reconstruction_and_typst_super() -> None:
    """Raised smaller-font PDF fragments (author affiliations '1,2', '1 Peking University',
    footnote '1 Data retrieved...') must be reconstructed as geometric superscripts and
    rendered as Typst native #super[...] instead of flat baseline numbers."""
    from ubt.adapters.pdf.textgeom import dehyph, merge_row_fragments

    # Author row fragments from Page 1 of 2608.25512v1.pdf
    author_frags = [
        LineBox("Yifan Shi", (210.9, 670.5, 254.4, 678.9), font_size=11.0),
        LineBox("1,2", (255.0, 675.0, 262.6, 680.7), font_size=6.6),
        LineBox(", Wei Zhang", (263.0, 667.6, 321.4, 678.8), font_size=11.0),
        LineBox("1", (322.0, 676.0, 324.3, 680.7), font_size=6.6),
        LineBox(", Tianyi Cui", (325.0, 667.6, 381.0, 678.7), font_size=11.0),
        LineBox("2", (381.2, 676.0, 384.2, 680.6), font_size=6.6),
    ]
    # Affiliation row fragments from Page 1 of 2608.25512v1.pdf
    affil_frags = [
        LineBox("1", (210.1, 655.7, 212.9, 660.2), font_size=6.6),
        LineBox("Peking University", (213.6, 647.5, 303.7, 658.4), font_size=11.0),
        LineBox("2", (315.0, 655.7, 317.9, 660.1), font_size=6.6),
        LineBox("DeepSeek-AI", (318.5, 647.6, 385.0, 658.4), font_size=11.0),
    ]
    merged = merge_row_fragments(author_frags + affil_frags)
    assert len(merged) == 2
    assert "¹˒²" in merged[0].text
    assert "¹Peking University" in merged[1].text
    assert "²DeepSeek-AI" in merged[1].text
    # dehyph must fold superscripts back to plain digits for cross-engine matching
    assert dehyph(merged[0].text) == dehyph("Yifan Shi 1,2 , Wei Zhang 1 , Tianyi Cui 2")

    # Even when target_text in the ledger has flat ASCII digits ("1 北京大学 2 DeepSeek-AI"),
    # _zone_typst must restore superscripts from zone.rows and emit Typst #super[...]
    ts = RigidTypesetter(font_family="Noto Serif CJK SC", target_lang="zh")
    ts.probe = MagicMock()
    ts.probe.check.return_value = True

    affil_zone = Zone(
        block_id="affil",
        page=1,
        x0=75.7,
        y0=646.0,
        x1=519.6,
        y1=662.0,
        base_size=11.0,
        rows=(merged[1].text,),
        bold=True,
        align="center",
    )
    affil_block = IRBlock(
        id="affil",
        flow_id=FlowID.MAIN_STORY,
        spine_index=3,
        block_type=BlockType.NARRATIVE,
        source_text="1 Peking University 2 DeepSeek-AI",
        target_text="1 北京大学 2 DeepSeek-AI",
        bbox=BoundingBox(page=1, x0=210.1, y0=647.5, x1=385.0, y1=660.2),
    )
    paints, _ = ts._plan_blocks([affil_block], {"affil": (affil_zone,)}, {1: 841.9})
    typst_out = ts._zone_typst(paints[1][0], 841.9)
    assert "#super(" in typst_out
    assert "[1]北京大学" in typst_out
    assert "#h(1.2em)" in typst_out
    assert "[2]DeepSeek-AI" in typst_out


@pytest.mark.fast
def test_invariant6_toc_document_index_extraction_and_leader_dots() -> None:
    """Table of Contents (DOCUMENT_INDEX) lines with dot leaders must be extracted
    with leaders stripped for LLM translation and rendered with Typst 1fr dot-leader
    fill + flush-right page numbers."""
    from ubt.adapters.pdf.docling_blocks import parse_toc_entry_line

    parsed = parse_toc_entry_line(
        "1.1. Dimensions of Composability . . . . . . . . . . . . . . . . . . . . . . . . 4"
    )
    assert parsed == ("1.1. Dimensions of Composability", "4")

    ts = RigidTypesetter(font_family="Noto Serif CJK SC", target_lang="zh")
    ts.probe = MagicMock()
    ts.probe.check.return_value = True

    toc_zone = Zone(
        block_id="toc_1_1",
        page=2,
        x0=87.9,
        y0=669.2,
        x1=524.1,
        y1=680.4,
        base_size=11.0,
        rows=(
            "1.1. Dimensions of Composability . . . . . . . . . . . . . . . . . . . . . . . . 4",
        ),
        bold=False,
        align="left",
    )
    toc_block = IRBlock(
        id="toc_1_1",
        flow_id=FlowID.MAIN_STORY,
        spine_index=10,
        block_type=BlockType.NARRATIVE,
        source_text="1.1. Dimensions of Composability",
        target_text="1.1 可组合性的维度",
        bbox=BoundingBox(page=2, x0=87.9, y0=669.2, x1=524.1, y1=680.4),
        provenance={"toc_entry": True, "toc_page": "4"},
    )
    paints, _ = ts._plan_blocks([toc_block], {"toc_1_1": (toc_zone,)}, {2: 841.9})
    typst_out = ts._zone_typst(paints[2][0], 841.9)
    assert "1.1 可组合性的维度" in typst_out
    assert "box(width: 1fr, repeat(" in typst_out
    assert "4" in typst_out
