"""Unit tests for Track 2 modern PDF adapters (Docling, Typst, BilingualAlternator)."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pypdf
import pytest

from tests.pdf_builders import blank_pdf, text_pdf
from ubt.adapters.factory import get_adapter_for_path
from ubt.adapters.pdf.alternator import BilingualAlternator
from ubt.adapters.pdf.docling_adapter import DoclingPDFAdapter
from ubt.adapters.pdf.docling_blocks import resolve_overlapping_formula_blocks
from ubt.adapters.pdf.typst_reconstructor import TypstReconstructor
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.exceptions import DocumentParseError, UnsupportedDocumentFormatError
from ubt.core.ir.models import (
    BlockType,
    BookManifest,
    BoundingBox,
    ChapterIR,
    ChapterMeta,
    FlowID,
    IRBlock,
)
from ubt.core.ir.run_metadata import RunMetadata


def test_factory_defaults_to_docling_for_pdf() -> None:
    """Factory should default to DoclingPDFAdapter for .pdf files."""
    adapter = get_adapter_for_path("paper.pdf")
    assert isinstance(adapter, DoclingPDFAdapter)
    assert adapter.engine_name == "docling"

    # Explicit canonical docling
    assert isinstance(get_adapter_for_path("paper.pdf", pdf_engine="docling"), DoclingPDFAdapter)

    # The old typst/modern aliases pointed at the same class and made one engine
    # look like three; they are removed so they now resolve as unknown.
    for alias in ("typst", "modern"):
        with pytest.raises(UnsupportedDocumentFormatError, match="unregistered PDF engine"):
            get_adapter_for_path("paper.pdf", pdf_engine=alias)

    # Deprecated babeldoc engine
    with pytest.raises(UnsupportedDocumentFormatError, match="BabelDOC engine is deprecated"):
        get_adapter_for_path("paper.pdf", pdf_engine="babeldoc")


def test_typst_reconstructor_markup_generation() -> None:
    """TypstReconstructor should generate correct Typst markup for headers, math, code, and narrative."""
    reconstructor = TypstReconstructor(font_size_pt=10.5)

    blocks = [
        IRBlock(
            id="b1",
            spine_index=1,
            block_type=BlockType.HEADING,
            flow_id=FlowID.MAIN_STORY,
            source_text="Introduction to Attention",
            target_text="注意力机制导论",
        ),
        IRBlock(
            id="b2",
            spine_index=2,
            block_type=BlockType.NARRATIVE,
            flow_id=FlowID.MAIN_STORY,
            source_text="The Transformer is based solely on attention mechanisms.",
            target_text="Transformer 模型完全基于注意力机制构建。",
        ),
        IRBlock(
            id="b3",
            spine_index=3,
            block_type=BlockType.FORMULA,
            flow_id=FlowID.MAIN_STORY,
            source_text="Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V",
            target_text="Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V",
            skip_translate=True,
        ),
        IRBlock(
            id="b4",
            spine_index=4,
            block_type=BlockType.CODE,
            flow_id=FlowID.MAIN_STORY,
            source_text="def attention(): pass",
            skip_translate=True,
        ),
        IRBlock(
            id="b5",
            spine_index=5,
            block_type=BlockType.LIST_ITEM,
            flow_id=FlowID.MAIN_STORY,
            source_text="First bullet item",
            target_text="首个列表项",
        ),
    ]

    # Monolingual target generation
    typ_source = reconstructor.generate_typst_source(blocks, title="Paper Title")
    assert "= Paper Title" in typ_source
    assert "== 注意力机制导论" in typ_source
    assert "Transformer 模型完全基于注意力机制构建。" in typ_source
    # Fail-closed math: bare multi-letter identifiers (Attention, softmax)
    # are unknown variables in Typst math mode, so the probe degrades the
    # line to a verbatim span instead of emitting document-breaking `$...$`.
    assert "`Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V`" in typ_source
    assert "```\ndef attention(): pass\n```" in typ_source
    assert "- 首个列表项" in typ_source

    # Facing bilingual generation
    bilingual_typ = reconstructor.generate_typst_source(blocks, bilingual=True)
    assert "The Transformer is based solely on attention mechanisms." in bilingual_typ
    assert "Transformer 模型完全基于注意力机制构建。" in bilingual_typ
    assert "First bullet item" in bilingual_typ
    assert "首个列表项" in bilingual_typ


def test_typst_reconstructor_interior_page_list_items() -> None:
    """TypstReconstructor should render all list items (even with mixed punctuation) on interior pages."""
    from ubt.core.ir.models import BoundingBox

    reconstructor = TypstReconstructor()
    # Simulate Page 19 with eyebrow, title, intro, 5 list items, and concluding paragraph
    blocks = [
        IRBlock(
            id="h1",
            spine_index=1,
            block_type=BlockType.HEADING,
            flow_id=FlowID.MAIN_STORY,
            source_text="17 - FROM ONE REQUEST TO A SERVER",
            target_text="17 - 从单请求到高并发服务器",
            bbox=BoundingBox(page=2, x0=50, y0=650, x1=200, y1=660),
        ),
        IRBlock(
            id="h2",
            spine_index=2,
            block_type=BlockType.HEADING,
            flow_id=FlowID.MAIN_STORY,
            source_text="Why cache management becomes difficult",
            target_text="缓存管理变得困难的原因是什么？",
            bbox=BoundingBox(page=2, x0=50, y0=630, x1=200, y1=640),
        ),
        IRBlock(
            id="p1",
            spine_index=3,
            block_type=BlockType.NARRATIVE,
            flow_id=FlowID.MAIN_STORY,
            source_text="At any moment, the server may need to manage:",
            target_text="在任何时刻，服务器可能需要处理：",
            bbox=BoundingBox(page=2, x0=50, y0=580, x1=200, y1=590),
        ),
        IRBlock(
            id="l1",
            spine_index=4,
            block_type=BlockType.LIST_ITEM,
            flow_id=FlowID.MAIN_STORY,
            source_text="one sequence that has just started;",
            target_text="刚刚开始的一段序列；",
            bbox=BoundingBox(page=2, x0=50, y0=550, x1=200, y1=560),
        ),
        IRBlock(
            id="l2",
            spine_index=5,
            block_type=BlockType.LIST_ITEM,
            flow_id=FlowID.MAIN_STORY,
            source_text="another already holding tens of thousands of cached tokens;",
            target_text="还有一些已经存储了数万个缓存的token；",
            bbox=BoundingBox(page=2, x0=50, y0=530, x1=200, y1=540),
        ),
        IRBlock(
            id="l3",
            spine_index=6,
            block_type=BlockType.LIST_ITEM,
            flow_id=FlowID.MAIN_STORY,
            source_text="several requests that finish and free memory;",
            target_text="多个请求完成并释放内存；",
            bbox=BoundingBox(page=2, x0=50, y0=510, x1=200, y1=520),
        ),
        IRBlock(
            id="l4",
            spine_index=7,
            block_type=BlockType.LIST_ITEM,
            flow_id=FlowID.MAIN_STORY,
            source_text="requests that share prefixes;",
            target_text="具有相同前缀的请求",
            bbox=BoundingBox(page=2, x0=50, y0=490, x1=200, y1=500),
        ),
        IRBlock(
            id="l5",
            spine_index=8,
            block_type=BlockType.LIST_ITEM,
            flow_id=FlowID.MAIN_STORY,
            source_text="requests that need different attention-window patterns.",
            target_text="需要不同注意力窗口模式的请求。",
            bbox=BoundingBox(page=2, x0=50, y0=470, x1=200, y1=480),
        ),
        IRBlock(
            id="p2",
            spine_index=9,
            block_type=BlockType.NARRATIVE,
            flow_id=FlowID.MAIN_STORY,
            source_text="This turns KV cache into a dynamic memory-management problem.",
            target_text="这使得 KV 缓存转变为动态内存管理问题。",
            bbox=BoundingBox(page=2, x0=50, y0=440, x1=200, y1=450),
        ),
    ]

    typ_page = reconstructor.generate_typst_source(blocks, page_strict=True)
    # All 5 list items must be rendered with - prefix
    assert "- 刚刚开始的一段序列；" in typ_page
    assert "- 还有一些已经存储了数万个缓存的token；" in typ_page
    assert "- 多个请求完成并释放内存；" in typ_page
    assert "- 具有相同前缀的请求" in typ_page
    assert "- 需要不同注意力窗口模式的请求。" in typ_page
    # Normal paragraphs must NOT have - prefix
    assert "- 在任何时刻" not in typ_page
    assert "- 这使得" not in typ_page


def test_page_strict_honours_a_recorded_page_when_the_bbox_is_missing() -> None:
    """A block whose provenance carried no bbox still knows its page.

    docling_parser records ``source_page`` for exactly that case; ignoring it
    dropped the block onto the nearest *preceding* page's group, breaking the
    1:1 source/target page alignment page-strict mode exists to hold.
    """
    reconstructor = TypstReconstructor()
    page_four = IRBlock(
        id="p4",
        spine_index=1,
        source_text="Fourth page paragraph.",
        target_text="第四页段落。",
        bbox=BoundingBox(page=4, x0=50, y0=700, x1=200, y1=710),
    )
    recorded = IRBlock(
        id="p5",
        spine_index=2,
        source_text="Fifth page paragraph.",
        target_text="第五页段落。",
        bbox=None,
        provenance={"source_page": 5},
    )
    typ = reconstructor.generate_typst_source([page_four, recorded], page_strict=True)
    assert "第四页段落。" in typ
    assert "第五页段落。" in typ
    # The page-5 block is not folded into page 4: one pagebreak separates them.
    assert typ.count("#pagebreak()") == 1


def test_bilingual_alternator_interleaving(tmp_path: Path) -> None:
    """BilingualAlternator should correctly interleave pages in zipper sequence."""
    src_pdf = blank_pdf(tmp_path / "src.pdf", pages=2)
    trans_pdf = blank_pdf(tmp_path / "trans.pdf", pages=2)
    out_pdf = tmp_path / "bilingual_facing.pdf"

    alternator = BilingualAlternator()

    # Dimension inspection
    dims = alternator.inspect_dimensions(src_pdf)
    assert len(dims) == 2
    assert pytest.approx(dims[0].width_pt, 0.1) == 595.276

    # Page interleaving
    result = alternator.interleave_pages(src_pdf, trans_pdf, out_pdf)
    assert result.output_path.exists()
    assert result.padding_pages == ()  # equal page counts need no padding

    # Verify interleaved page count: 2 + 2 = 4 pages
    reader = pypdf.PdfReader(str(result.output_path))
    assert len(reader.pages) == 4


def test_bilingual_alternator_asymmetric_pages(tmp_path: Path) -> None:
    """BilingualAlternator should pad unequal page counts with blank pages to preserve facing symmetry."""
    src_pdf = blank_pdf(tmp_path / "src_3p.pdf", pages=3)
    trans_pdf = blank_pdf(tmp_path / "trans_1p.pdf", pages=1)
    out_pdf = tmp_path / "bilingual_asymmetric.pdf"

    alternator = BilingualAlternator()
    result = alternator.interleave_pages(src_pdf, trans_pdf, out_pdf)
    assert result.output_path.exists()
    # src 3p vs trans 1p, no facing flyleaf: pads the short target at pages 4, 6.
    assert result.padding_pages == (4, 6)

    reader = pypdf.PdfReader(str(result.output_path))
    # 3 max pages * 2 = 6 pages (facing pairs preserved with blank padding)
    assert len(reader.pages) == 6


def test_bilingual_alternator_reports_padding_pages(tmp_path: Path) -> None:
    """The alternator must declare every blank page it inserts.

    Facing/alternating interleaving fills the shorter side with intentional
    blanks. The visual gate can only exempt them if it is told which pages they
    are; the page-1 flyleaf used to be the only one it knew about, so real
    padding pages were reported as CRITICAL blank_page defects. The interleave
    result now carries the exact 1-based page numbers it filled.
    """
    src_pdf = blank_pdf(tmp_path / "src_3p.pdf", pages=3)
    trans_pdf = blank_pdf(tmp_path / "trans_1p.pdf", pages=1)
    out_pdf = tmp_path / "bilingual_facing.pdf"

    result = BilingualAlternator().interleave_pages(src_pdf, trans_pdf, out_pdf, facing_spread=True)

    assert result.output_path == out_pdf
    # Page 1 flyleaf, then src[1]→2 / trans[0]→3, src[2]→4 / pad→5,
    # src[3]→6 / pad→7.
    assert result.padding_pages == (1, 5, 7)
    assert len(pypdf.PdfReader(str(result.output_path)).pages) == 7


@pytest.mark.network
@pytest.mark.asyncio
@pytest.mark.slow  # real Docling model load (~9s)
async def test_docling_adapter_manifest_and_pypdf_stream(tmp_path: Path) -> None:
    """DoclingPDFAdapter should extract manifest and parse stream using built-in pypdf fallback."""
    pdf_path = text_pdf(tmp_path / "test_doc.pdf", pages=1)
    adapter = DoclingPDFAdapter()

    manifest = await adapter.extract_manifest(pdf_path)
    assert manifest.title == "test_doc"
    assert len(manifest.chapters) == 1
    assert manifest.chapters[0].chapter_id == "pdf_main"

    chapters = []
    async for ch in adapter.parse_stream(pdf_path):
        chapters.append(ch)

    assert len(chapters) == 1
    assert chapters[0].chapter_id == "pdf_main"


@pytest.mark.asyncio
async def test_docling_adapter_render_output_markdown_and_typst(tmp_path: Path) -> None:
    """DoclingPDFAdapter should render both Markdown and Typst outputs from ledger."""
    db_path = tmp_path / "ledger.db"
    ledger = SQLiteJobLedger(db_path)

    manifest = BookManifest(
        doc_id="doc_test",
        title="Test Doc",
        source_path=str(tmp_path / "test.pdf"),
        chapters=[
            ChapterMeta(
                chapter_id="pdf_main",
                title="Test Doc",
                spine_index=1,
                source_file="test.pdf",
            )
        ],
    )
    ledger.init_job_from_manifest("doc_test", manifest)

    block = IRBlock(
        id="blk_001",
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        flow_id=FlowID.MAIN_STORY,
        source_text="Hello world.",
        target_text="你好，世界。",
    )
    chapter_ir = ChapterIR(
        doc_id="doc_test",
        chapter_id="pdf_main",
        title="Test Doc",
        spine_index=1,
        blocks=[block],
    )
    ledger.append_chapter("doc_test", chapter_ir)

    adapter = DoclingPDFAdapter()

    # Render Markdown
    md_out = tmp_path / "output.md"
    await adapter.render_output(manifest, ledger, "zh", md_out)
    assert md_out.exists()
    assert "你好，世界。" in md_out.read_text(encoding="utf-8")

    # Render Typst markup
    typ_out = tmp_path / "output.typ"
    await adapter.render_output(manifest, ledger, "zh", typ_out)
    assert typ_out.exists()
    assert "你好，世界。" in typ_out.read_text(encoding="utf-8")


def test_docling_mock_conversion(tmp_path: Path) -> None:
    """Verify Docling parsing mapping logic when docling is present."""
    adapter = DoclingPDFAdapter()
    dummy_pdf = blank_pdf(tmp_path / "sample.pdf", pages=1)

    mock_doc = MagicMock()
    mock_doc.export_to_dict.return_value = {
        "texts": [
            {"text": "Paper Heading", "label": "section_header"},
            {"text": "Paragraph body text.", "label": "paragraph"},
            {"text": "Bullet item 1.", "label": "list_item"},
            {"text": "E = mc^2", "label": "formula"},
            {"text": "print(1)", "label": "code"},
        ],
        "tables": [{"data": {"markdown": "| A | B |\n|---|---|\n| 1 | 2 |"}}],
    }
    mock_result = MagicMock()
    mock_result.document = mock_doc

    mock_input_format = MagicMock()
    mock_pipeline_options_cls = MagicMock()
    mock_converter_cls = MagicMock()
    mock_converter_cls.return_value.convert.return_value = mock_result
    mock_format_option_cls = MagicMock()

    # Stub the import seam instead of swapping `sys.modules` entries for the
    # real torch-backed `docling` packages: module swapping has historically
    # corrupted interpreter teardown (GC segfault, exit 139, after green runs).
    with (
        patch(
            "ubt.adapters.pdf.docling_adapter._docling_symbols",
            return_value=(
                mock_input_format,
                mock_pipeline_options_cls,
                mock_converter_cls,
                mock_format_option_cls,
            ),
        ),
        patch.object(adapter, "is_docling_installed", return_value=True),
    ):
        blocks = adapter._extract_with_docling(dummy_pdf)

        assert len(blocks) == 6
        # Heading
        assert blocks[0].block_type == BlockType.HEADING
        assert blocks[0].flow_id == FlowID.MAIN_STORY
        # Paragraph
        assert blocks[1].block_type == BlockType.NARRATIVE
        assert blocks[1].flow_id == FlowID.MAIN_STORY
        # List Item
        assert blocks[2].block_type == BlockType.LIST_ITEM
        assert blocks[2].flow_id == FlowID.MAIN_STORY
        # Formula (skip_translate)
        assert blocks[3].block_type == BlockType.FORMULA
        assert blocks[3].skip_translate is True
        # Code (skip_translate)
        assert blocks[4].block_type == BlockType.CODE
        assert blocks[4].flow_id == FlowID.MAIN_STORY
        assert blocks[4].skip_translate is True
        # Table (FlowID.TABLE_GRID)
        assert blocks[5].block_type == BlockType.TABLE
        assert blocks[5].flow_id == FlowID.TABLE_GRID


@pytest.mark.network
@pytest.mark.asyncio
@pytest.mark.slow  # real Docling model load (~4s)
async def test_docling_adapter_manifest_metadata_and_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """DoclingPDFAdapter should record engine in manifest.metadata and log the fallback."""
    pdf_path = text_pdf(tmp_path / "metadata_test.pdf", pages=1)
    adapter = DoclingPDFAdapter()

    import logging

    # The fallback is logged at INFO (docling is an optional extra, so a warning
    # on every PDF for installs without it would be noise). Capture INFO and
    # assert the message the product actually emits; the previous WARNING-level
    # capture matched nothing and reddened the main CI job, which deliberately
    # installs dev-only (no docling).
    with caplog.at_level(logging.INFO):
        manifest = await adapter.extract_manifest(pdf_path)
        assert "pdf_parser_engine" in manifest.metadata

        # Trigger fallback parse
        async for _ in adapter.parse_stream(pdf_path):
            pass

    if not adapter.is_docling_installed():
        assert manifest.metadata["pdf_parser_engine"] == "oxide_fallback"
        assert any("Docling not installed" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_docling_adapter_render_output_alternating_bilingual_pdf(tmp_path: Path) -> None:
    """DoclingPDFAdapter should wire up BilingualAlternator to deliver facing bilingual PDFs."""
    src_pdf = blank_pdf(tmp_path / "original_source.pdf", pages=2)
    db_path = tmp_path / "bilingual_ledger.db"
    ledger = SQLiteJobLedger(db_path)

    manifest = BookManifest(
        doc_id="doc_bilingual",
        title="Bilingual Book",
        source_path=str(src_pdf),
        chapters=[
            ChapterMeta(
                chapter_id="pdf_main",
                title="Bilingual Book",
                spine_index=1,
                source_file=src_pdf.name,
            )
        ],
        run=RunMetadata(bilingual_mode="alternating"),
    )
    ledger.init_job_from_manifest("doc_bilingual", manifest)

    block = IRBlock(
        id="blk_001",
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        flow_id=FlowID.MAIN_STORY,
        source_text="Source paragraph text.",
        target_text="中文翻译段落文字。",
    )
    chapter_ir = ChapterIR(
        doc_id="doc_bilingual",
        chapter_id="pdf_main",
        title="Bilingual Book",
        spine_index=1,
        blocks=[block],
    )
    ledger.append_chapter("doc_bilingual", chapter_ir)

    adapter = DoclingPDFAdapter()

    # Mock Typst compilation to produce a valid 2-page dummy translated PDF
    async def mock_compile_async(typ_source: str, out_path: Path | str) -> Path:
        target_path = Path(out_path)
        compiled: Path = blank_pdf(target_path, pages=2)
        return compiled

    with (
        patch.object(adapter.reconstructor, "is_compiler_available", return_value=True),
        patch.object(adapter.reconstructor, "compile_pdf_async", side_effect=mock_compile_async),
    ):
        out_pdf = tmp_path / "final_bilingual_edition.pdf"
        await adapter.render_output(manifest, ledger, "zh", out_pdf)

        assert out_pdf.exists()
        # Verify alternator interleaved 2 original pages + 2 translated pages = 4 pages
        reader = pypdf.PdfReader(str(out_pdf))
        assert len(reader.pages) == 4


@pytest.mark.asyncio
async def test_missing_typst_compiler_fails_instead_of_reporting_missing_pdf(
    tmp_path: Path,
) -> None:
    """Regression: without the compiler the render must not claim a PDF it never wrote."""
    src_pdf = blank_pdf(tmp_path / "src.pdf", pages=1)
    manifest = BookManifest(
        doc_id="doc_nocompiler",
        title="No Compiler",
        source_path=str(src_pdf),
        chapters=[
            ChapterMeta(
                chapter_id="pdf_main",
                title="No Compiler",
                spine_index=1,
                source_file=src_pdf.name,
            )
        ],
        run=RunMetadata(bilingual_mode="monolingual", render_engine="publication"),
    )
    block = IRBlock(
        id="blk_001",
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        flow_id=FlowID.MAIN_STORY,
        source_text="Source paragraph text.",
        target_text="中文翻译段落文字。",
    )

    adapter = DoclingPDFAdapter()
    out_pdf = tmp_path / "final.pdf"
    with (
        patch.object(adapter.reconstructor, "is_compiler_available", return_value=False),
        pytest.raises(DocumentParseError, match="Typst compiler not found"),
    ):
        await adapter.render_blocks(manifest, [block], "zh", out_pdf)

    assert not out_pdf.exists()
    assert out_pdf.with_suffix(".typ").exists()
    assert out_pdf.with_suffix(".md").exists()


def test_docling_formula_enrichment_resolution() -> None:
    """Verify on-demand formula enrichment resolution and graceful defaults."""
    adapter_off = DoclingPDFAdapter(formula_enrichment="off")
    assert adapter_off._resolve_formula_enrichment() is False

    adapter_on = DoclingPDFAdapter(formula_enrichment="on")
    assert adapter_on._resolve_formula_enrichment() is True

    # In auto mode, probes GPU presence
    adapter_auto = DoclingPDFAdapter(formula_enrichment="auto")
    with patch("ubt.adapters.pdf.docling_adapter._has_accelerator", return_value=False):
        assert adapter_auto._resolve_formula_enrichment() is False
    with patch("ubt.adapters.pdf.docling_adapter._has_accelerator", return_value=True):
        assert adapter_auto._resolve_formula_enrichment() is True


def test_typst_reconstructor_drops_chrome_headers_and_footers() -> None:
    """Verify that TypstReconstructor drops chrome footer and header blocks from flowing prose."""
    from ubt.adapters.pdf.typst_reconstructor import TypstReconstructor
    from ubt.core.ir.models import BlockType, IRBlock, LayoutRole

    blocks = [
        IRBlock(
            id="b1",
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            source_text="First paragraph of chapter.",
            target_text="第一段内容。",
        ),
        IRBlock(
            id="b2",
            spine_index=2,
            block_type=BlockType.NARRATIVE,
            layout_role=LayoutRole.FOOTER,
            skip_translate=True,
            policy_reason="verdict:chrome",
            source_text="Copyright 2024 Elsevier Inc. All rights reserved.",
            target_text="Copyright 2024 Elsevier Inc. All rights reserved.",
        ),
        IRBlock(
            id="b3",
            spine_index=3,
            block_type=BlockType.NARRATIVE,
            source_text="Second paragraph of chapter.",
            target_text="第二段内容。",
        ),
    ]

    recon = TypstReconstructor()
    typ_source = recon.generate_typst_source(
        blocks, title="Test", bilingual=False, page_strict=False
    )
    assert "第一段内容。" in typ_source
    assert "第二段内容。" in typ_source
    assert "Elsevier Inc" not in typ_source


def test_bilingual_alternator_facing_spread_flyleaf() -> None:
    """Verify that facing_spread=True inserts a Recto flyleaf on page 1 for proper 2-page spreads."""
    tmp = Path(tempfile.mkdtemp())
    src_pdf = tmp / "src.pdf"
    trans_pdf = tmp / "trans.pdf"
    out_pdf = tmp / "out.pdf"

    # Create 2-page dummy PDFs
    w_src = pypdf.PdfWriter()
    w_src.add_blank_page(width=100, height=100)
    w_src.add_blank_page(width=100, height=100)
    with src_pdf.open("wb") as f:
        w_src.write(f)

    w_trans = pypdf.PdfWriter()
    w_trans.add_blank_page(width=100, height=100)
    w_trans.add_blank_page(width=100, height=100)
    with trans_pdf.open("wb") as f:
        w_trans.write(f)

    alternator = BilingualAlternator()

    # Normal mode: 2 + 2 = 4 pages
    alternator.interleave_pages(src_pdf, trans_pdf, out_pdf, facing_spread=False)
    r_norm = pypdf.PdfReader(str(out_pdf))
    assert len(r_norm.pages) == 4

    # Facing spread mode: 1 (flyleaf) + 2 + 2 = 5 pages
    out_facing = tmp / "out_facing.pdf"
    alternator.interleave_pages(src_pdf, trans_pdf, out_facing, facing_spread=True)
    r_facing = pypdf.PdfReader(str(out_facing))
    assert len(r_facing.pages) == 5


def test_eyebrow_fusion_keeps_the_fused_headings_counter_updates() -> None:
    """Fusing an eyebrow with its title consumed the title without rendering it.

    The branch advances ``i += 2``, so the second heading's own chapter /
    equation-counter lines were never emitted: the heading counter stayed on the
    eyebrow's number and every equation in the chapter was numbered under it.
    """
    from ubt.core.ir.models import BoundingBox

    def heading(idx: int, text: str, y: float) -> IRBlock:
        return IRBlock(
            id=f"h{idx}",
            spine_index=idx,
            block_type=BlockType.HEADING,
            flow_id=FlowID.MAIN_STORY,
            source_text=text,
            target_text=text,
            bbox=BoundingBox(page=1, x0=50, y0=y, x1=300, y1=y + 10),
        )

    body = IRBlock(
        id="p1",
        spine_index=3,
        block_type=BlockType.NARRATIVE,
        flow_id=FlowID.MAIN_STORY,
        source_text="Body prose about decoding cost.",
        target_text="Body prose about decoding cost.",
        bbox=BoundingBox(page=1, x0=50, y0=640, x1=300, y1=660),
    )
    blocks = [
        heading(1, "5 - PREFILL AND DECODE", 700),
        heading(2, "Chapter 6: Decoding costs", 680),
        body,
    ]
    source = TypstReconstructor().generate_typst_source(blocks, target_lang="en", page_strict=True)

    assert "#counter(heading).update(5)" in source
    assert "#counter(heading).update(6)" in source, "the fused chapter lost its counter"
    assert "Chapter 6: Decoding costs" in source


@pytest.mark.fast
def test_sample_pdf_pages_closes_handles(tmp_path: Path) -> None:
    import pypdfium2 as pdfium

    from ubt.adapters.pdf.plain_text_extractor import sample_pdf_pages

    pdf = pdfium.PdfDocument.new()
    pdf.new_page(width=100, height=100)
    pdf_path = tmp_path / "test_sample.pdf"
    pdf.save(str(pdf_path))
    pdf.close()

    closed_pages = []
    closed_textpages = []
    orig_page_close = pdfium.PdfPage.close
    orig_textpage_close = pdfium.PdfTextPage.close

    def mock_page_close(self: object) -> None:
        closed_pages.append(self)
        orig_page_close(self)

    def mock_textpage_close(self: object) -> None:
        closed_textpages.append(self)
        orig_textpage_close(self)

    with (
        patch.object(pdfium.PdfPage, "close", mock_page_close),
        patch.object(pdfium.PdfTextPage, "close", mock_textpage_close),
    ):
        page_count, is_scanned, sample = sample_pdf_pages(pdf_path)
        assert len(closed_pages) >= 1, "PdfPage.close must be called in sample_pdf_pages"
        assert len(closed_textpages) >= 1, "PdfTextPage.close must be called in sample_pdf_pages"


@pytest.mark.fast
def test_docling_adapter_extract_sync_not_serialized() -> None:
    from ubt.adapters.pdf.docling_adapter import DoclingPDFAdapter

    adapter = DoclingPDFAdapter()
    fn = adapter._extract_blocks_sync
    # Verify that the function is NOT wrapped by pdfium_serialized (which wraps with PDFIUM_LOCK)
    # Functions wrapped with @pdfium_serialized have '__wrapped__' or closure holding PDFIUM_LOCK
    import inspect

    closure_vars = inspect.getclosurevars(fn)
    assert "PDFIUM_LOCK" not in closure_vars.nonlocals, (
        "_extract_blocks_sync must not be decorated with @pdfium_serialized holding global PDFIUM_LOCK"
    )


@pytest.mark.fast
def test_resolve_overlapping_formula_blocks_merges_vertical_overlap() -> None:
    """Two consecutive FORMULA blocks on the same page whose bounding boxes overlap
    vertically (e.g. equation + commutative diagram) must merge into a single union block."""
    f1 = IRBlock(
        id="pdf_main#b0148",
        spine_index=148,
        block_type=BlockType.FORMULA,
        flow_id=FlowID.MAIN_STORY,
        source_text=r"\Pr_1 \circ f' = f \circ \Pr_1",
        skip_translate=True,
        bbox=BoundingBox(page=10, x0=200.0, y0=410.0, x1=430.0, y1=490.0),
    )
    f2 = IRBlock(
        id="pdf_main#b0149",
        spine_index=149,
        block_type=BlockType.FORMULA,
        flow_id=FlowID.MAIN_STORY,
        source_text=r"\begin{array}{ccc} \Gamma \xrightarrow{f} \Gamma \end{array}",
        skip_translate=True,
        bbox=BoundingBox(page=10, x0=210.0, y0=360.0, x1=390.0, y1=455.0),
    )
    merged = resolve_overlapping_formula_blocks([f1, f2])
    assert len(merged) == 1
    assert merged[0].bbox == BoundingBox(page=10, x0=200.0, y0=360.0, x1=430.0, y1=490.0)
