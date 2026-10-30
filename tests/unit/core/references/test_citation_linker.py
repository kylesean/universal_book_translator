"""Unit tests for cross-chapter citation topology, anchor extraction, and Typst hyperlink injection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ubt.core.ir.models import BlockStatus, IRBlock
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
    element: Any,
    page: int = 1,
) -> IRBlock:
    # ``IRBlock`` has no ``bbox`` field: the bounding box is the element's
    # ``span`` and is exposed as a read-only property. Building the element with
    # that span is the supported route, and it keeps ``block.bbox`` populated.
    return IRBlock(element=element, style=None, status=BlockStatus.PENDING)


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


def test_inject_hyperlinks_does_not_steal_a_style_wrapper_body() -> None:
    """A citation inside a styled run must not lose the run's content block.

    The renderer wraps a coloured citation run as
    ``#text(fill: rgb("#00699d"))[1]`` before links are injected. The citation
    regex used to see the body's ``[1]`` as prose, replace it with
    ``#link(<bib-1>)[[1]]`` and consume the body's brackets, leaving
    ``#text(fill: …)#link(…)`` — a Typst call with no body
    (``missing argument: body``) that aborted the whole page-locked render on
    26-page replay block ``pdf_main#b0004``.
    """
    blocks = [
        _make_block(
            "h_bib",
            Heading(id="h_bib", spine_index=0, span=Span(page=5), level=1, text="References"),
        ),
        _make_block("b1", Paragraph(id="b1", spine_index=1, span=Span(page=5), text="[1] Ref 1")),
    ]
    topology = build_citation_topology(blocks)
    text = r'数值方法 \[#text(fill: rgb("#00699d"))[1]/**/\]。'

    linked = inject_hyperlinks(text, topology, target_format="typst")

    assert '#text(fill: rgb("#00699d"))[1]' in linked
    assert "#text(fill: rgb(" + '"#00699d"))#link' not in linked
    assert "\\[#text(fill:" in linked, "the escaped bracket that opens the span survives"


def test_inject_hyperlinks_still_links_plain_prose_citations() -> None:
    blocks = [
        _make_block(
            "h_bib",
            Heading(id="h_bib", spine_index=0, span=Span(page=5), level=1, text="References"),
        ),
        _make_block("b1", Paragraph(id="b1", spine_index=1, span=Span(page=5), text="[1] Ref 1")),
    ]
    topology = build_citation_topology(blocks)

    linked = inject_hyperlinks("见文献 [1]。", topology, target_format="typst")

    assert "#link(<bib-1>)[[1]]" in linked


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


def test_extract_page_listing_and_algorithm_anchors() -> None:
    blocks = [
        _make_block(
            "p1",
            Paragraph(id="p1", spine_index=0, span=Span(page=1), text="Text on page 1"),
            page=1,
        ),
        _make_block(
            "p2",
            Paragraph(id="p2", spine_index=1, span=Span(page=3), text="Text on page 3"),
            page=3,
        ),
        _make_block(
            "lst1",
            Caption(id="lst1", spine_index=2, span=Span(page=3), text="Listing 1: Server setup"),
            page=3,
        ),
        _make_block(
            "alg1",
            Caption(id="alg1", spine_index=3, span=Span(page=4), text="Algorithm 2: Fast search"),
            page=4,
        ),
    ]

    topology = extract_document_anchors(blocks)
    assert "page-1" in topology.anchors
    assert "page-3" in topology.anchors
    assert "page-4" in topology.anchors
    assert "lst-1" in topology.anchors
    assert "algo-2" in topology.anchors

    assert topology.key_to_anchor[(ReferenceKind.PAGE, "1")] == "page-1"
    assert topology.key_to_anchor[(ReferenceKind.PAGE, "3")] == "page-3"
    assert topology.key_to_anchor[(ReferenceKind.LISTING, "1")] == "lst-1"
    assert topology.key_to_anchor[(ReferenceKind.ALGORITHM, "2")] == "algo-2"


def test_format_mention_text_schemes() -> None:
    from ubt.core.references.linker import format_mention_text
    from ubt.core.references.models import CrossRefPrefixScheme

    # Chinese standard
    assert (
        format_mention_text(
            "Figure 2",
            ReferenceKind.FIGURE,
            "2",
            CrossRefPrefixScheme.CHINESE_STANDARD,
            lang="zh",
        )
        == "图 2"
    )
    assert (
        format_mention_text(
            "Table 3",
            ReferenceKind.TABLE,
            "3",
            CrossRefPrefixScheme.CHINESE_STANDARD,
            lang="zh",
        )
        == "表 3"
    )
    assert (
        format_mention_text(
            "Section 4.1",
            ReferenceKind.HEADING,
            "4.1",
            CrossRefPrefixScheme.CHINESE_STANDARD,
            lang="zh",
        )
        == "第 4.1 节"
    )
    assert (
        format_mention_text(
            "Chapter 5",
            ReferenceKind.HEADING,
            "5",
            CrossRefPrefixScheme.CHINESE_STANDARD,
            lang="zh",
        )
        == "第 5 章"
    )
    assert (
        format_mention_text(
            "page 15",
            ReferenceKind.PAGE,
            "15",
            CrossRefPrefixScheme.CHINESE_STANDARD,
            lang="zh",
        )
        == "第 15 页"
    )

    # Academic formal
    assert (
        format_mention_text(
            "图 2",
            ReferenceKind.FIGURE,
            "2",
            CrossRefPrefixScheme.ACADEMIC_FORMAL,
            lang="en",
        )
        == "Figure 2"
    )
    assert (
        format_mention_text(
            "表 3",
            ReferenceKind.TABLE,
            "3",
            CrossRefPrefixScheme.ACADEMIC_FORMAL,
            lang="en",
        )
        == "Table 3"
    )

    # Compact
    assert (
        format_mention_text(
            "Figure 2",
            ReferenceKind.FIGURE,
            "2",
            CrossRefPrefixScheme.COMPACT,
            lang="en",
        )
        == "Fig. 2"
    )
    assert (
        format_mention_text(
            "Section 4.1",
            ReferenceKind.HEADING,
            "4.1",
            CrossRefPrefixScheme.COMPACT,
            lang="en",
        )
        == "Sec. 4.1"
    )


def test_inject_hyperlinks_with_page_and_native_ref() -> None:
    from ubt.core.references.models import CrossRefConfig, CrossRefStyle

    blocks = [
        _make_block(
            "cap1", Caption(id="cap1", spine_index=0, span=Span(page=2), text="Figure 2: Layout")
        ),
        _make_block(
            "p1",
            Paragraph(id="p1", spine_index=1, span=Span(page=5), text="Text on page 5"),
            page=5,
        ),
    ]
    topology = build_citation_topology(blocks)

    text = "Refer to Figure 2, see page 5 for details."

    # 1. With page number dynamic lookup
    cfg_page = CrossRefConfig(style=CrossRefStyle.WITH_PAGE, target_lang="zh")
    linked_page = inject_hyperlinks(text, topology, target_format="typst", config=cfg_page)
    assert "#link(<fig-2>)[Figure 2]#context" in linked_page
    assert "loc.first().location().page() != here().page()" in linked_page
    assert "#link(<page-5>)[see page 5]" in linked_page

    # 2. Native ref
    cfg_native = CrossRefConfig(style=CrossRefStyle.NATIVE_REF)
    linked_native = inject_hyperlinks(text, topology, target_format="typst", config=cfg_native)
    assert "#ref(<fig-2>)" in linked_native


def test_inject_hyperlinks_html_and_markdown() -> None:
    from ubt.core.references.models import CrossRefConfig

    blocks = [
        _make_block(
            "cap1",
            Caption(id="cap1", spine_index=0, span=Span(page=1), text="Table 3: Benchmarks"),
        ),
    ]
    topology = build_citation_topology(blocks)

    text = "Detailed metrics are shown in Table 3."

    # HTML
    cfg_html = CrossRefConfig(link_css_class="doc-link", open_in_new_tab=True)
    linked_html = inject_hyperlinks(text, topology, target_format="html", config=cfg_html)
    assert (
        '<a href="#tbl-3" class="doc-link doc-link-table" data-ref-kind="table" target="_blank" rel="noopener noreferrer">Table 3</a>'
        in linked_html
    )

    # Markdown
    linked_md = inject_hyperlinks(text, topology, target_format="markdown")
    assert "[Table 3](#tbl-3)" in linked_md


def test_typst_document_renderer_with_xref_enabled(tmp_path: Path) -> None:
    from ubt.core.references.models import CrossRefConfig, CrossRefStyle

    blocks = [
        _make_block(
            "h1",
            Heading(id="h1", spine_index=0, span=Span(page=1), level=1, text="1. Overview"),
            page=1,
        ),
        _make_block(
            "eq1",
            Formula(id="eq1", spine_index=1, span=Span(page=1), source="E = m c^2 \\quad (1)"),
            page=1,
        ),
        _make_block(
            "p1",
            Paragraph(
                id="p1",
                spine_index=2,
                span=Span(page=2),
                text="In Section 1, we defined Equation (1). See page 1.",
            ),
            page=2,
        ),
    ]

    cfg = CrossRefConfig(style=CrossRefStyle.WITH_PAGE, target_lang="zh")
    renderer = TypstDocumentRenderer(enable_xref=True, xref_config=cfg)
    source, stats = renderer.build_source(blocks, tmp_path)

    # Page start metadata emitted
    assert '[#metadata("page-start") <page-1>]' in source
    assert '[#metadata("page-start") <page-2>]' in source

    # Equation label emitted
    assert "<eq-1>" in source

    # Page reference linked
    assert "#link(<page-1>)[See page 1]" in source
