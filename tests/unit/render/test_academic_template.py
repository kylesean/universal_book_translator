"""Unit tests for the academic Typst template (single and dual column)."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from ubt.adapters.pdf.typst_compile import resolve_typst_binary
from ubt.core.ir.models import BlockType, BookManifest, IRBlock
from ubt.render.typst.academic import (
    academic_preamble,
    academic_title_banner,
    is_academic_document,
    resolve_document_columns,
)
from ubt.render.typst.document_renderer import TypstDocumentRenderer

pytestmark = pytest.mark.fast


def _compile_typst(doc_content: str) -> tuple[bool, str]:
    """Helper to verify that Typst actually compiles the document."""
    try:
        binary = resolve_typst_binary(os.environ.get("UBT_TYPST_BINARY", "typst"))
        if not binary:
            pytest.skip("typst compiler not found on system")
    except Exception:
        pytest.skip("typst compiler not found on system")

    with tempfile.NamedTemporaryFile("w", suffix=".typ") as f:
        f.write(doc_content)
        f.flush()
        out_pdf = Path(f.name).with_suffix(".pdf")
        try:
            res = subprocess.run(
                [binary, "compile", f.name, str(out_pdf)],
                capture_output=True,
                text=True,
                check=False,
            )
            return res.returncode == 0, res.stderr
        finally:
            if out_pdf.exists():
                out_pdf.unlink(missing_ok=True)


def test_is_academic_document() -> None:
    manifest_paper = BookManifest(
        doc_id="d1",
        source_path="sample.pdf",
        title="Attention",
        metadata={"doc_type": "paper"},
    )
    assert is_academic_document(manifest_paper)

    manifest_profile = BookManifest(
        doc_id="d4",
        source_path="paper.pdf",
        title="A Paper",
        metadata={"genre_profile": "academic"},
    )
    assert is_academic_document(manifest_profile)

    manifest_thesis = BookManifest(
        doc_id="d5",
        source_path="thesis.pdf",
        title="Ph.D. Thesis",
        metadata={"doc_type": "thesis"},
    )
    assert is_academic_document(manifest_thesis)

    manifest_book = BookManifest(
        doc_id="d3",
        source_path="book.epub",
        title="Novel",
        metadata={"genre_profile": "novel", "page_kinds": {1: "editable_text"}},
    )
    assert not is_academic_document(manifest_book)
    assert not is_academic_document(None)


def test_resolve_document_columns_fidelity_first() -> None:
    """Document columns mirror the source: multi-column evidence -> 2, else 1.

    The signal is the route decision's physical multi-column finding (where the
    profiler actually writes it), and the only thing that outranks it is an
    explicit ``--columns`` override on the run.
    """
    # 1. Single-column paper/report/thesis defaults to 1 (never forced to 2 just
    #    because the genre profile is academic — the profile styles text, not layout).
    manifest_single = BookManifest(
        doc_id="d1",
        source_path="neurips_paper.pdf",
        title="NeurIPS Single Column",
        metadata={"genre_profile": "academic"},
    )
    assert resolve_document_columns(manifest_single) == 1

    # 2. Physical multi-column evidence in the source -> 2. This is where the
    #    profiler actually records it: run.route_decision, not metadata. A real
    #    two-column document flags most body pages, so its share is well past half.
    manifest_multi_share = BookManifest(
        doc_id="d3",
        source_path="acm_paper.pdf",
        title="ACM Dual Column",
        metadata={"genre_profile": "paper"},
    )
    manifest_multi_share.run.route_decision = {
        "mode": "long",
        "multicolumn_page_share": 0.72,
    }
    assert resolve_document_columns(manifest_multi_share) == 2

    # A pre-reduced boolean is trusted directly.
    manifest_multi_flag = BookManifest(
        doc_id="d2",
        source_path="ieee_paper.pdf",
        title="IEEE Dual Column",
        metadata={"genre_profile": "academic"},
    )
    manifest_multi_flag.run.route_decision = {"has_multicolumn": True}
    assert resolve_document_columns(manifest_multi_flag) == 2

    # A minority of multi-column pages (front matter, references, a couple of
    # full-width figure pages) is NOT a two-column document. This is the shape a
    # single-column paper takes once the profiler's margin-stamp false positive
    # is gone — measured 0.225 before, 0.0 after.
    manifest_noise = BookManifest(
        doc_id="d6",
        source_path="book.pdf",
        title="Book",
        metadata={"genre_profile": "general"},
    )
    manifest_noise.run.route_decision = {"multicolumn_page_share": 0.225}
    assert resolve_document_columns(manifest_noise) == 1

    manifest_mostly_single = BookManifest(
        doc_id="d7",
        source_path="single.pdf",
        title="Single Column",
        metadata={"genre_profile": "paper"},
    )
    manifest_mostly_single.run.route_decision = {"multicolumn_page_share": 0.05}
    assert resolve_document_columns(manifest_mostly_single) == 1

    # 3. Explicit user override (--columns) takes top priority, on the run.
    manifest_override_dual = BookManifest(
        doc_id="d4",
        source_path="single_paper.pdf",
        title="Paper",
        metadata={"genre_profile": "academic"},
    )
    manifest_override_dual.run.layout_columns = 2
    assert resolve_document_columns(manifest_override_dual) == 2

    manifest_override_single = BookManifest(
        doc_id="d5",
        source_path="dual_paper.pdf",
        title="Paper",
        metadata={"genre_profile": "paper"},
    )
    manifest_override_single.run.route_decision = {"has_multicolumn": True}
    manifest_override_single.run.layout_columns = 1
    assert resolve_document_columns(manifest_override_single) == 1


def test_resolve_document_columns_reads_the_real_manifest_shape() -> None:
    """Regression: the real manifest records multi-column under ``run.route_decision``.

    Two defects are pinned here together, because they masked each other:

    1. The first cut read only ``manifest.metadata``, where the profiler's
       ``multicolumn_page_share`` is never written (it lives under
       ``run.route_decision``), so the wiring never saw the signal at all.
    2. The profiler itself mis-measured single-column papers: a rotated margin
       stamp read as a right column and scored 0.225 on the arXiv Switch
       Transformer paper, whose body is one column.

    With (1) fixed the signal is read; with (2) fixed the real paper measures 0.0
    and correctly stays single-column. This uses the shape the pipeline produces.
    """
    manifest = BookManifest(
        doc_id="real",
        source_path="2101.03961v3.pdf",
        title="Switch Transformers",
        # Only the genre profile is stamped onto metadata — no column keys here.
        metadata={"genre_profile": "paper"},
    )
    manifest.run.route_decision = {
        "mode": "long",
        "pages": 40,
        # Measured 0.0 on the real single-column paper after the stamp fix
        # (was a false 0.225 before it).
        "multicolumn_page_share": 0.0,
        "structural_page_share": 0.3,
    }
    assert resolve_document_columns(manifest) == 1, (
        "a single-column paper must stay single-column — --profile never forces two"
    )

    # The same shape for a genuinely two-column source flips to two.
    manifest.run.route_decision = {"mode": "long", "multicolumn_page_share": 0.83}
    assert resolve_document_columns(manifest) == 2


def _block(bid: str, block_type: BlockType, source: str, target: str) -> IRBlock:
    from ubt.core.ir.models import make_element

    return IRBlock(
        element=make_element(id=bid, spine_index=1, block_type=block_type, source_text=source),
        target_text=target,
    )


def test_single_column_academic_paper_renders_clean_single_column(tmp_path: Path) -> None:
    """A single-column academic paper (arXiv, NeurIPS, thesis) MUST render as single-column."""
    import asyncio

    from ubt.adapters.pdf.docling_render import DoclingRenderStrategy

    manifest = BookManifest(
        doc_id="d",
        title="Attention Is All You Need",
        source_path="paper.pdf",
        metadata={"genre_profile": "academic"},  # single-column by default
    )
    blocks = [
        _block("b1", BlockType.NARRATIVE, "We present the Transformer.", "我们提出 Transformer。"),
    ]
    out = tmp_path / "paper.pdf"
    strategy = DoclingRenderStrategy()
    asyncio.run(
        strategy.render_blocks(manifest=manifest, blocks=blocks, target_lang="zh", output_path=out)
    )
    assert out.exists() and out.stat().st_size > 0
    typ = (tmp_path / f".{out.stem}_typst" / "main.typ").read_text(encoding="utf-8")
    assert "columns: 1" in typ, "single-column paper was unexpectedly forced to two columns"
    assert "columns: 2" not in typ
    assert "size: 10.5pt" in typ
    assert "Attention Is All You Need" in typ


def test_dual_column_academic_paper_renders_dual_column(tmp_path: Path) -> None:
    """A paper with multi-column source (or user override) renders dual-column IEEE style."""
    import asyncio

    from ubt.adapters.pdf.docling_render import DoclingRenderStrategy

    manifest = BookManifest(
        doc_id="d_ieee",
        title="IEEE Dual Column Paper",
        source_path="ieee.pdf",
        metadata={"genre_profile": "paper"},
    )
    # The real multi-column signal: the profiler's finding on the route decision.
    manifest.run.route_decision = {"mode": "long", "multicolumn_page_share": 0.6}
    blocks = [
        _block("b1", BlockType.NARRATIVE, "Measurements were taken.", "进行了测量。"),
    ]
    out = tmp_path / "ieee.pdf"
    strategy = DoclingRenderStrategy()
    asyncio.run(
        strategy.render_blocks(manifest=manifest, blocks=blocks, target_lang="zh", output_path=out)
    )
    assert out.exists() and out.stat().st_size > 0
    typ = (tmp_path / f".{out.stem}_typst" / "main.typ").read_text(encoding="utf-8")
    assert "columns: 2" in typ
    assert "size: 9.5pt" in typ
    assert 'scope: "parent"' in typ


def test_academic_preamble_structure() -> None:
    # Single column mode
    preamble_single = academic_preamble(target_lang="zh", paper_title="Test Paper", columns=1)
    assert "columns: 1" in preamble_single
    assert "margin: (x: 2.5cm, y: 2.5cm)" in preamble_single
    assert "size: 10.5pt" in preamble_single
    assert "Test Paper" in preamble_single

    # Dual column mode
    preamble_dual = academic_preamble(target_lang="zh", paper_title="Test Paper", columns=2)
    assert "columns: 2" in preamble_dual
    assert "margin: (x: 1.8cm, y: 2.0cm)" in preamble_dual
    assert "size: 9.5pt" in preamble_dual
    assert "Test Paper" in preamble_dual


def test_academic_title_banner() -> None:
    banner_single = academic_title_banner(
        title="Single Column Title",
        abstract="Single column abstract.",
        authors=["Author One"],
        columns=1,
    )
    assert "block(width: 100%" in banner_single
    assert "Single Column Title" in banner_single

    banner_dual = academic_title_banner(
        title="Dual Column Title",
        abstract="Dual column abstract.",
        authors=["Author One"],
        columns=2,
    )
    assert 'scope: "parent"' in banner_dual
    assert "Dual Column Title" in banner_dual


def test_typst_renderer_compiles_both_single_and_dual_columns(tmp_path: Path) -> None:
    from ubt.core.ir.models import make_element

    for cols in (1, 2):
        renderer = TypstDocumentRenderer(
            layout_template="academic",
            columns=cols,
            paper_title="Attention Is All You Need",
            abstract="The dominant sequence transduction models are based on complex neural networks.",
        )

        el1 = make_element(
            id="b1",
            spine_index=0,
            block_type=BlockType.HEADING,
            source_text="1. Introduction",
            level=1,
        )
        el2 = make_element(
            id="b2",
            spine_index=0,
            block_type=BlockType.NARRATIVE,
            source_text="Recurrent models compute factorizations.",
        )
        el3 = make_element(
            id="b3",
            spine_index=0,
            block_type=BlockType.FORMULA,
            source_text=r"\begin{pmatrix} a & b \\ c & d \end{pmatrix}",
        )

        blocks = [
            IRBlock(element=el1, target_text="1. 引言"),
            IRBlock(element=el2, target_text="循环模型通常沿着序列进行计算。"),
            IRBlock(
                element=el3,
                target_text=r"\begin{pmatrix} a & b \\ c & d \end{pmatrix}",
            ),
        ]

        build_dir = tmp_path / f"academic_test_col_{cols}"
        source, _ = renderer.build_source(blocks, build_dir, target_lang="zh")
        if cols == 1:
            assert "columns: 1" in source
            assert "size: 10.5pt" in source
        else:
            assert "columns: 2" in source
            assert "size: 9.5pt" in source

        ok, stderr = _compile_typst(source)
        assert ok, f"Typst compile failed for columns={cols}: {stderr}"


# --------------------------------------------------------------------------- #
# Table of contents: TOC rows must render as a TOC (title + page number), never
# as orphaned section headings. Regression for the near-empty page a paper's
# TOC produced when every toc_entry block took the #heading path.
# --------------------------------------------------------------------------- #


def _toc_block(bid: str, title: str, page: str) -> IRBlock:
    from ubt.core.ir.models import BlockProvenance, make_element

    element = make_element(id=bid, spine_index=1, block_type=BlockType.HEADING, source_text=title)
    return IRBlock(
        element=element,
        target_text=title,
        provenance=BlockProvenance(toc_entry=True, toc_page=page),
    )


def _heading_block(bid: str, text: str) -> IRBlock:
    from ubt.core.ir.models import make_element

    element = make_element(id=bid, spine_index=1, block_type=BlockType.HEADING, source_text=text)
    return IRBlock(element=element, target_text=text)


def _para_block(bid: str, text: str) -> IRBlock:
    from ubt.core.ir.models import make_element

    element = make_element(id=bid, spine_index=1, block_type=BlockType.NARRATIVE, source_text=text)
    return IRBlock(element=element, target_text=text)


def test_toc_entries_group_into_runs() -> None:
    from ubt.render.typst.document_renderer import _group_toc_runs

    blocks = [
        _toc_block("t1", "1 引言", "3"),
        _toc_block("t2", "2 切换 Transformer", "4"),
        _heading_block("h1", "1 引言"),
        _toc_block("t3", "3 扩展特性", "11"),
    ]
    runs = _group_toc_runs(blocks)
    assert [[b.id for b in run] for run in runs] == [["t1", "t2"], ["t3"]]


def test_toc_rows_do_not_become_section_headings(tmp_path: Path) -> None:
    """The defect: TOC rows rendered as ``#heading`` (orphaned, empty page)."""
    renderer = TypstDocumentRenderer(layout_template="academic", columns=1, paper_title="Paper")
    blocks = [
        _toc_block("t1", "6 相关工作", "24"),
        _toc_block("t2", "7 讨论", "25"),
        _toc_block("t3", "8 未来工作", "26"),
        _heading_block("h1", "1 引言"),
    ]
    source, _ = renderer.build_source(blocks, tmp_path / "b", target_lang="zh")
    # The TOC is one block carrying the entries and their page numbers ...
    assert "目录" in source
    assert "6 相关工作" in source and "24" in source
    # ... and the three TOC rows did NOT take the heading path. Only the one
    # genuine section heading remains.
    assert source.count("#heading(level: 1)") == 1


def test_toc_renders_and_compiles(tmp_path: Path) -> None:
    renderer = TypstDocumentRenderer(layout_template="academic", columns=1, paper_title="Paper")
    blocks = [
        _toc_block("t1", "1 引言", "3"),
        _toc_block("t2", "2 切换 Transformer", "4"),
        _toc_block("t3", "2.1 稀疏路由的简化", "5"),
        _para_block("p1", "这是正文。"),
    ]
    source, _ = renderer.build_source(blocks, tmp_path / "b", target_lang="zh")
    ok, stderr = _compile_typst(source)
    assert ok, f"Typst compile failed for a TOC document: {stderr}"
