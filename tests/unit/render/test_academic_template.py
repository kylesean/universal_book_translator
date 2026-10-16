"""Unit tests for the academic two-column Typst template."""

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

    manifest_cols = BookManifest(
        doc_id="d2",
        source_path="survey.pdf",
        title="Survey",
        metadata={"columns": 2},
    )
    assert is_academic_document(manifest_cols)

    # The two signals the runtime actually produces: the genre profile stamped
    # at ingest and the page profiler's per-page layout classification.
    manifest_profile = BookManifest(
        doc_id="d4",
        source_path="paper.pdf",
        title="A Paper",
        metadata={"genre_profile": "academic"},
    )
    assert is_academic_document(manifest_profile)

    manifest_pages = BookManifest(
        doc_id="d5",
        source_path="paper.pdf",
        title="A Paper",
        metadata={
            "page_kinds": {
                1: "mixed_complex",
                2: "mixed_complex",
                3: "mixed_complex",
                4: "editable_text",
            }
        },
    )
    assert is_academic_document(manifest_pages)

    manifest_book = BookManifest(
        doc_id="d3",
        source_path="book.epub",
        title="Novel",
        metadata={"genre_profile": "novel", "page_kinds": {1: "editable_text"}},
    )
    assert not is_academic_document(manifest_book)

    assert not is_academic_document(None)


def _block(bid: str, block_type: BlockType, source: str, target: str) -> IRBlock:
    from ubt.core.ir.models import make_element

    return IRBlock(
        element=make_element(id=bid, spine_index=1, block_type=block_type, source_text=source),
        target_text=target,
    )


def test_the_publish_path_dispatches_the_academic_template(tmp_path: Path) -> None:
    """A manifest carrying an academic genre profile must render two-column.

    This drives the *real* dispatch (``DoclingRenderStrategy._render_publish_sync``
    -> ``is_academic_document`` -> ``academic_preamble``), not a hand-injected
    ``layout_template``. The earlier version of this file passed ``layout_template``
    straight to the constructor, so it passed while the runtime path was
    unreachable (nothing produced the metadata key the dispatch read).
    """
    import asyncio

    from ubt.adapters.pdf.docling_render import DoclingRenderStrategy

    manifest = BookManifest(
        doc_id="d",
        title="Attention Is All You Need",
        source_path="paper.pdf",
        metadata={"genre_profile": "academic"},
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
    assert "columns: 2" in typ, "academic profile did not select the two-column template"
    assert "place(top" in typ, "academic title banner missing"


def test_a_novel_stays_single_column(tmp_path: Path) -> None:
    """The academic branch must not capture a plain book."""
    import asyncio

    from ubt.adapters.pdf.docling_render import DoclingRenderStrategy

    manifest = BookManifest(
        doc_id="d",
        title="A Novel",
        source_path="book.pdf",
        metadata={"genre_profile": "novel", "page_kinds": {1: "editable_text"}},
    )
    blocks = [_block("b1", BlockType.NARRATIVE, "Once upon a time.", "从前。")]
    out = tmp_path / "book.pdf"
    strategy = DoclingRenderStrategy()
    asyncio.run(
        strategy.render_blocks(manifest=manifest, blocks=blocks, target_lang="zh", output_path=out)
    )
    typ = (tmp_path / f".{out.stem}_typst" / "main.typ").read_text(encoding="utf-8")
    assert "columns: 2" not in typ


def test_academic_preamble_structure() -> None:
    preamble = academic_preamble(target_lang="zh", paper_title="Test Paper")
    assert "columns: 2" in preamble
    assert "margin: (x: 1.8cm, y: 2.0cm)" in preamble
    assert "size: 9.5pt" in preamble
    assert "header: context" in preamble
    assert "Test Paper" in preamble


def test_academic_title_banner() -> None:
    banner = academic_title_banner(
        title="Attention Is All You Need",
        abstract="We present the Transformer architecture.",
        authors=["Ashish Vaswani", "Noam Shazeer"],
    )
    assert 'scope: "parent"' in banner
    assert "Attention Is All You Need" in banner
    assert "Ashish Vaswani" in banner
    assert "We present the Transformer architecture." in banner


def test_typst_renderer_with_academic_template(tmp_path: Path) -> None:
    from ubt.core.ir.models import make_element

    renderer = TypstDocumentRenderer(
        layout_template="academic",
        paper_title="Attention Is All You Need",
        abstract="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
    )

    el1 = make_element(
        id="b1", spine_index=0, block_type=BlockType.HEADING, source_text="1. Introduction", level=1
    )
    el2 = make_element(
        id="b2",
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        source_text="Recurrent models typically factor computation along the symbol positions.",
    )
    el3 = make_element(
        id="b3",
        spine_index=2,
        block_type=BlockType.FORMULA,
        source_text=r"p_{i}(x) = \frac{e^{h(x)_i}}{\sum_j^N e^{h(x)_j}}",
    )

    blocks = [
        IRBlock(element=el1, target_text="1. 引言"),
        IRBlock(
            element=el2, target_text="循环模型通常沿着输入和输出序列的符号位置进行计算因子分解。"
        ),
        IRBlock(element=el3, target_text=r"p_{i}(x) = \frac{e^{h(x)_i}}{\sum_j^N e^{h(x)_j}}"),
    ]

    source, stats = renderer.build_source(blocks, tmp_path, target_lang="zh")
    assert "columns: 2" in source
    assert "Attention Is All You Need" in source
    assert "引言" in source
    assert stats.blocks_rendered == 3

    ok, err = _compile_typst(source)
    assert ok, f"Typst failed to compile academic document: {err}"
