"""Unit tests for cross-chapter citation topology, anchor extraction, and Typst hyperlink injection."""

from __future__ import annotations

from pathlib import Path

import pytest

from ubt.core.ir.models import BlockStatus, BoundingBox, IRBlock
from ubt.core.references.extractor import (
    extract_document_anchors,
)
from ubt.core.references.linker import (
    build_citation_topology,
    inject_hyperlinks,
)
from ubt.core.references.models import ReferenceKind
from ubt.model.ast import Caption, Figure, Formula, Heading, Paragraph, Table
from ubt.model.span import Span
from ubt.render.typst.document_renderer import TypstDocumentRenderer

pytestmark = pytest.mark.fast


def _make_block(
    block_id: str,
    element: any,
    page: int = 1,
) -> IRBlock:
    return IRBlock(
        element=element,
        style=None,
        status=BlockStatus.PENDING,
        bbox=BoundingBox(page=page, x0=50.0, y0=100.0, x1=500.0, y1=200.0),
    )


def test_extract_heading_anchors() -> None:
    blocks = [
        _make_block(
            "h1",
            Heading(id="h1", spine_index=0, span=Span(page=1), level=1, text="1. Introduction"),
        ),
        _make_block(
            "h2",
            Heading(
                id="h2",
                spine_index=1,
                span=Span(page=1),
                level=2,
                text="2.1 Background & Prior Work",
            ),
        ),
        _make_block(
            "h3",
            Heading(
                id="h3",
                spine_index=2,
                span=Span(page=2),
                level=1,
                text="Chapter 3: System Architecture",
            ),
        ),
        _make_block(
            "h4",
            Heading(id="h4", spine_index=3, span=Span(page=3), level=1, text="第 4 章：实验评估"),
        ),
    ]

    topology = extract_document_anchors(blocks)
    assert "sec-1" in topology.anchors
    assert "sec-2-1" in topology.anchors
    assert "sec-3" in topology.anchors
    assert "sec-4" in topology.anchors

    assert topology.key_to_anchor[(ReferenceKind.HEADING, "1")] == "sec-1"
    assert topology.key_to_anchor[(ReferenceKind.HEADING, "2.1")] == "sec-2-1"
    assert topology.key_to_anchor[(ReferenceKind.HEADING, "3")] == "sec-3"
    assert topology.key_to_anchor[(ReferenceKind.HEADING, "4")] == "sec-4"


def test_extract_figure_table_and_equation_anchors() -> None:
    blocks = [
        _make_block(
            "fig1", Figure(id="fig1", spine_index=0, span=Span(page=1), asset_id="fig1.png")
        ),
        _make_block(
            "cap1",
            Caption(
                id="cap1",
                spine_index=1,
                span=Span(page=1),
                text="Figure 1: Overall pipeline architecture.",
            ),
        ),
        _make_block(
            "cap2", Caption(id="cap2", spine_index=2, span=Span(page=1), text="图 2：消歧图谱")
        ),
        _make_block(
            "tbl1",
            Table(
                id="tbl1",
                spine_index=3,
                span=Span(page=2),
                markup="| A | B |\n| --- | --- |\n| 1 | 2 |",
            ),
        ),
        _make_block(
            "cap3",
            Caption(
                id="cap3", spine_index=4, span=Span(page=2), text="Table 3: Hyperparameter grid"
            ),
        ),
        _make_block(
            "cap4", Caption(id="cap4", spine_index=5, span=Span(page=2), text="表 4：定量测试对比")
        ),
        _make_block(
            "eq1",
            Formula(id="eq1", spine_index=6, span=Span(page=3), source="y = W x + b \\quad (1)"),
        ),
    ]

    topology = extract_document_anchors(blocks)
    assert "fig-1" in topology.anchors
    assert "fig-2" in topology.anchors
    assert "tbl-3" in topology.anchors
    assert "tbl-4" in topology.anchors
    assert "eq-1" in topology.anchors


def test_extract_bibliography_anchors() -> None:
    blocks = [
        _make_block(
            "h_bib",
            Heading(id="h_bib", spine_index=0, span=Span(page=10), level=1, text="References"),
        ),
        _make_block(
            "b1",
            Paragraph(
                id="b1",
                spine_index=1,
                span=Span(page=10),
                text="[1] Vaswani, A., et al. Attention Is All You Need. NeurIPS 2017.",
            ),
        ),
        _make_block(
            "b2",
            Paragraph(
                id="b2",
                spine_index=2,
                span=Span(page=10),
                text="[12] Devlin, J., et al. BERT: Pre-training of Deep Bidirectional Transformers. NAACL 2019.",
            ),
        ),
        _make_block(
            "b3",
            Paragraph(
                id="b3",
                spine_index=3,
                span=Span(page=11),
                text="[Vaswani17] Vaswani, A., et al. Attention Is All You Need.",
            ),
        ),
    ]

    topology = extract_document_anchors(blocks)
    assert "bib-1" in topology.anchors
    assert "bib-12" in topology.anchors
    assert "bib-vaswani17" in topology.anchors


def test_find_prose_citations_and_resolution() -> None:
    blocks = [
        _make_block(
            "h_bib",
            Heading(id="h_bib", spine_index=0, span=Span(page=5), level=1, text="References"),
        ),
        _make_block(
            "b1", Paragraph(id="b1", spine_index=1, span=Span(page=5), text="[1] Reference 1")
        ),
        _make_block(
            "b2", Paragraph(id="b2", spine_index=2, span=Span(page=5), text="[2] Reference 2")
        ),
        _make_block(
            "h1",
            Heading(id="h1", spine_index=3, span=Span(page=1), level=1, text="2.1 Architecture"),
        ),
        _make_block(
            "cap1",
            Caption(id="cap1", spine_index=4, span=Span(page=1), text="Figure 3: Main layout"),
        ),
        _make_block(
            "p1",
            Paragraph(
                id="p1",
                spine_index=5,
                span=Span(page=1),
                text="The model follows [1] and [2], as discussed in Section 2.1 and shown in Figure 3.",
            ),
        ),
    ]

    topology = build_citation_topology(blocks)
    assert len(topology.links) >= 4
    assert topology.total_citations_found >= 4
    assert topology.resolved_citations_count >= 4
    assert topology.resolution_rate == 1.0


def test_inject_hyperlinks_typst() -> None:
    blocks = [
        _make_block(
            "h_bib",
            Heading(id="h_bib", spine_index=0, span=Span(page=5), level=1, text="References"),
        ),
        _make_block("b1", Paragraph(id="b1", spine_index=1, span=Span(page=5), text="[1] Ref 1")),
        _make_block(
            "h1", Heading(id="h1", spine_index=1, span=Span(page=1), level=2, text="2.1 Methods")
        ),
        _make_block(
            "cap1",
            Caption(id="cap1", spine_index=2, span=Span(page=2), text="Figure 2: Architecture"),
        ),
        _make_block(
            "cap2", Caption(id="cap2", spine_index=3, span=Span(page=2), text="Table 4: Scores")
        ),
    ]
    topology = build_citation_topology(blocks)

    # 1. English text
    en_text = "As shown in [1], Section 2.1 details the framework. Refer to Figure 2 and Table 4."
    en_linked = inject_hyperlinks(en_text, topology, target_format="typst")
    assert "#link(<bib-1>)[[1]]" in en_linked
    assert "#link(<sec-2-1>)[Section 2.1]" in en_linked
    assert "#link(<fig-2>)[Figure 2]" in en_linked
    assert "#link(<tbl-4>)[Table 4]" in en_linked

    # 2. Chinese text
    zh_text = "根据文献 [1] 的定义，详见第 2.1 节，结构如图 2 所示，性能见表 4。"
    zh_linked = inject_hyperlinks(zh_text, topology, target_format="typst")
    assert "#link(<bib-1>)[[1]]" in zh_linked
    assert "#link(<sec-2-1>)[第 2.1 节]" in zh_linked
    assert "#link(<fig-2>)[如图 2]" in zh_linked
    assert "#link(<tbl-4>)[表 4]" in zh_linked


def test_inject_hyperlinks_protects_code_and_math() -> None:
    blocks = [
        _make_block(
            "h_bib",
            Heading(id="h_bib", spine_index=0, span=Span(page=5), level=1, text="References"),
        ),
        _make_block("b1", Paragraph(id="b1", spine_index=1, span=Span(page=5), text="[1] Ref 1")),
    ]
    topology = build_citation_topology(blocks)

    # In math mode and code block, [1] must NOT be replaced!
    text = "Citation [1] is valid, but $x \\in [1, 2]$ and `code [1]` must stay intact."
    linked = inject_hyperlinks(text, topology, target_format="typst")

    assert "#link(<bib-1>)[[1]]" in linked
    assert "$x \\in [1, 2]$" in linked
    assert "`code [1]`" in linked


def test_typst_document_renderer_emits_anchors_and_links(tmp_path: Path) -> None:
    blocks = [
        _make_block(
            "h1",
            Heading(id="h1", spine_index=0, span=Span(page=1), level=1, text="1. Introduction"),
        ),
        _make_block(
            "p1",
            Paragraph(
                id="p1",
                spine_index=1,
                span=Span(page=1),
                text="In this paper we extend Section 1, referencing [1].",
            ),
        ),
        _make_block(
            "h_bib",
            Heading(id="h_bib", spine_index=2, span=Span(page=2), level=1, text="References"),
        ),
        _make_block(
            "b1",
            Paragraph(
                id="b1",
                spine_index=3,
                span=Span(page=2),
                text="[1] Vaswani et al. 2017.",
            ),
        ),
    ]
    blocks[3].policy_translate = False  # keeps source for bibliography

    renderer = TypstDocumentRenderer()
    source, stats = renderer.build_source(blocks, tmp_path)

    # Heading should carry anchor label <sec-1>
    assert "#heading(level: 1)[1\\. Introduction] <sec-1>" in source
    # Prose should carry clickable #link
    assert "#link(<sec-1>)[Section 1]" in source
    assert "#link(<bib-1>)[\\[1\\]]" in source
    # Bibliography entry should carry anchor point
    assert "<bib-1>" in source
