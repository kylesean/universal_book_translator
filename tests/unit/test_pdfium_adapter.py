"""Unit tests for the PDF multi-engine layer: pypdfium2 fast path,
first-page heuristic engine selection, and factory auto routing."""

import io
from pathlib import Path
from typing import Any

import pytest

from tests.pdf_builders import blank_pdf
from ubt.adapters.factory import (
    _PDF_ENGINE_REGISTRY,
    get_adapter_for_path,
)
from ubt.adapters.pdf.docling_adapter import DoclingPDFAdapter
from ubt.adapters.pdf.engine_selector import select_pdf_engine
from ubt.adapters.pdf.pdfium_adapter import PDFiumAdapter
from ubt.core.ir.models import BlockType, FlowID

# ---------------------------------------------------------------------------
# Minimal hand-written text PDF (no external generation dependency)
# ---------------------------------------------------------------------------


def build_text_pdf(
    path: Path,
    lines: list[str],
    col2_lines: list[str] | None = None,
    line_gap: int = 22,
    pages: int = 1,
) -> Path:
    """Write a minimal valid PDF containing Helvetica text lines.

    ``line_gap`` controls the vertical leading so paragraph-grouping
    behaviour can be exercised (gap > 1.8x line height starts a new
    paragraph). ``col2_lines`` renders a second column at x=320.
    """

    def _tj(x: float, y: float, ln: str) -> str:
        esc = ln.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        return f"1 0 0 1 {x} {y} Tm ({esc}) Tj"

    content_parts: list[str] = []
    for _ in range(pages):
        content = ["BT /F1 11 Tf"]
        y = 780.0
        for ln in lines:
            content.append(_tj(50, y, ln))
            y -= line_gap
        if col2_lines:
            y = 780.0
            for ln in col2_lines:
                content.append(_tj(320, y, ln))
                y -= line_gap
        content.append("ET")
        content_parts.append("\n".join(content))

    streams = [part.encode("latin-1", "replace") for part in content_parts]
    # Object layout: 1=Catalog, 2=Pages, 3=Font, then (Page, Content) pairs
    page_ids = [4 + 2 * i for i in range(len(streams))]
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {len(streams)} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    ]
    for pid, stream in zip(page_ids, streams, strict=True):
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
            f"/Contents {pid + 1} 0 R /Resources << /Font << /F1 3 0 R >> >> >>".encode()
        )
        objects.append(
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
        )

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets: list[int] = []
    for i, body in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(f"{i} 0 obj\n".encode() + body + b"\nendobj\n")
    xref_pos = out.tell()
    out.write(f"xref\n0 {len(objects) + 1}\n".encode())
    out.write(b"0000000000 65535 f \n")
    for off in offsets:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF".encode()
    )
    path.write_bytes(out.getvalue())
    return path


SINGLE_COL_LINES = [f"Line number {i} of the single column body text." for i in range(1, 11)]


# ---------------------------------------------------------------------------
# PDFiumAdapter extraction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pdfium_manifest_tags_engine(tmp_path: Path) -> None:
    pdf_path = build_text_pdf(tmp_path / "fast.pdf", SINGLE_COL_LINES)
    adapter = PDFiumAdapter()

    manifest = await adapter.extract_manifest(pdf_path)
    assert manifest.metadata["pdf_parser_engine"] == "pdfium"
    assert adapter.engine_name == "pdfium"


@pytest.mark.asyncio
async def test_pdfium_extracts_born_digital_paragraphs(tmp_path: Path) -> None:
    """Wide leading → one block per line; reading order top-down."""
    pdf_path = build_text_pdf(tmp_path / "single.pdf", SINGLE_COL_LINES)
    adapter = PDFiumAdapter()

    chapters: list[Any] = []
    async for chapter in adapter.parse_stream(pdf_path):
        chapters.append(chapter)

    assert len(chapters) == 1
    blocks = chapters[0].blocks
    assert len(blocks) == len(SINGLE_COL_LINES)
    assert blocks[0].source_text == SINGLE_COL_LINES[0]
    assert blocks[-1].source_text == SINGLE_COL_LINES[-1]
    # Narrative typing and block id / bbox conventions match the pypdf path
    assert blocks[0].block_type == BlockType.NARRATIVE
    assert blocks[0].flow_id == FlowID.MAIN_STORY
    assert blocks[0].id == "pdf_main#b0001"
    assert blocks[0].bbox.page == 1
    assert blocks[-1].id == f"pdf_main#b{len(SINGLE_COL_LINES):04d}"


@pytest.mark.asyncio
async def test_pdfium_groups_tight_leading_into_one_paragraph(tmp_path: Path) -> None:
    """Narrow leading (gap ≤ 1.8x line height) stays a single paragraph block."""
    tight_lines = [
        "First sentence lives inside the paragraph.",
        "Second sentence follows on the next visual line.",
        "Third sentence concludes the paragraph here.",
    ]
    pdf_path = build_text_pdf(tmp_path / "tight.pdf", tight_lines, line_gap=13)
    adapter = PDFiumAdapter()

    blocks: list[Any] = []
    async for chapter in adapter.parse_stream(pdf_path):
        blocks.extend(chapter.blocks)

    assert len(blocks) == 1
    assert blocks[0].source_text == " ".join(tight_lines)


@pytest.mark.asyncio
async def test_pdfium_heading_and_list_typing(tmp_path: Path) -> None:
    """Short non-sentence lines type as HEADings; bullets as LIST_ITEMs."""
    lines = ["Chapter Overview", "- First bullet item", "- Second bullet item"]
    pdf_path = build_text_pdf(tmp_path / "mixed.pdf", lines)
    adapter = PDFiumAdapter()

    blocks: list[Any] = []
    async for chapter in adapter.parse_stream(pdf_path):
        blocks.extend(chapter.blocks)

    assert [b.block_type for b in blocks] == [
        BlockType.HEADING,
        BlockType.LIST_ITEM,
        BlockType.LIST_ITEM,
    ]


@pytest.mark.asyncio
async def test_pdfium_corrupt_file_falls_back_to_pdf_oxide(tmp_path: Path) -> None:
    """pdfium extraction failure degrades to pdf_oxide; complete corruption raises DocumentParseError."""
    from ubt.core.exceptions import DocumentParseError

    bad = tmp_path / "corrupt.pdf"
    bad.write_bytes(b"%PDF-1.4 this is not a real pdf body \xff\xfe")
    adapter = PDFiumAdapter()

    with pytest.raises(DocumentParseError):
        async for _ in adapter.parse_stream(bad):
            pass


@pytest.mark.asyncio
async def test_pdfium_inherits_render_stack(tmp_path: Path) -> None:
    """PDFiumAdapter reuses the Docling mainline render output (Markdown leg)."""
    from ubt.core.engine.ledger import SQLiteJobLedger
    from ubt.core.ir.models import (
        BookManifest,
        ChapterIR,
        ChapterMeta,
        IRBlock,
    )

    ledger = SQLiteJobLedger(tmp_path / "ledger.db")
    manifest = BookManifest(
        doc_id="doc_pdfium",
        title="Fast Doc",
        source_path=str(tmp_path / "src.pdf"),
        chapters=[ChapterMeta(chapter_id="pdf_main", title="Fast Doc", spine_index=1)],
    )
    ledger.init_job_from_manifest("doc_pdfium", manifest)
    ledger.append_chapter(
        "doc_pdfium",
        ChapterIR(
            doc_id="doc_pdfium",
            chapter_id="pdf_main",
            title="Fast Doc",
            spine_index=1,
            blocks=[
                IRBlock(
                    id="blk_001",
                    spine_index=1,
                    block_type=BlockType.NARRATIVE,
                    flow_id=FlowID.MAIN_STORY,
                    source_text="Hello world.",
                    target_text="你好，世界。",
                )
            ],
        ),
    )

    adapter = PDFiumAdapter()
    md_out = tmp_path / "output.md"
    await adapter.render_output(manifest, ledger, "zh", md_out)
    assert md_out.exists()
    assert "你好，世界。" in md_out.read_text(encoding="utf-8")
    ledger.close()


# ---------------------------------------------------------------------------
# First-page heuristic engine selection
# ---------------------------------------------------------------------------


def test_selector_single_column_born_digital_uses_pdfium(tmp_path: Path) -> None:
    pdf_path = build_text_pdf(tmp_path / "single.pdf", SINGLE_COL_LINES)
    assert select_pdf_engine(pdf_path) == "pdfium"


def test_inspect_pdf_route_plan_single_column(tmp_path: Path) -> None:
    from ubt.adapters.pdf.engine_selector import inspect_pdf_route_plan

    pdf_path = build_text_pdf(tmp_path / "single_plan.pdf", SINGLE_COL_LINES)
    plan = inspect_pdf_route_plan(pdf_path)
    assert plan.primary_engine == "pdfium"
    assert not plan.has_vector_diagrams
    assert not plan.has_formulas
    assert not plan.has_multicolumn
    assert plan.diagram_strategy == "none"


def test_selector_two_column_layout_stays_on_docling(tmp_path: Path) -> None:
    left = ["Left column body text line that is long enough to wrap nicely."] * 4
    right = ["Right column body text line that is long enough to wrap too."] * 4
    pdf_path = build_text_pdf(tmp_path / "two_col.pdf", left, col2_lines=right)
    assert select_pdf_engine(pdf_path) == "docling"


def test_selector_blank_pdf_routes_to_docling(tmp_path: Path) -> None:
    pdf_path = blank_pdf(tmp_path / "blank.pdf")
    assert select_pdf_engine(pdf_path) == "docling"


def test_selector_corrupt_pdf_defaults_to_docling(tmp_path: Path) -> None:
    bad = tmp_path / "corrupt.pdf"
    bad.write_bytes(b"not a pdf at all")
    assert select_pdf_engine(bad) == "docling"


def test_selector_missing_file_defaults_to_docling(tmp_path: Path) -> None:
    assert select_pdf_engine(tmp_path / "missing.pdf") == "docling"


def test_selector_formula_dense_single_column_routes_to_docling(tmp_path: Path) -> None:
    """Shattered-equation debris (isolated K/V/x tokens) needs Docling."""
    formula_lines = [
        "Consider one self attention layer with learned projections:",
        "h , k h h WQ WK WV q , v = = = WQ WK WV i i i i i i .",
        "T K q t 1 : t softmax V1 V1 : t . d h",
        "= [ ] = [ ] k K K , V1 V1 v ; ; 1 : 1 : 1 V1 : V1 : 1 t t t",
        "K , V R B H T D , B HKV HQ free B0 B1 B2 B3",
        "M KV = 2 L T H KV D h b MK L T HK D = 2 h MKV HKV",
    ]
    pdf_path = build_text_pdf(tmp_path / "formula.pdf", formula_lines)
    assert select_pdf_engine(pdf_path) == "docling"


def test_selector_clean_cover_does_not_hide_formula_body(tmp_path: Path) -> None:
    """Sampling must look past a clean cover page into the body."""
    import io as _io

    cover = ["Understanding KV Cache", "A short technical handbook", "@techNmak"]
    body = [
        "h , k h h WQ WK WV q , v = = = WQ WK WV i i i i i i .",
        "= [ ] = [ ] k K K , V1 V1 v ; ; 1 : 1 : 1 V1 : V1 : 1 t t t",
        "K , V R B H T D , B HKV HQ free B0 B1 B2 B3",
    ]
    pages_content = [cover, body, body]
    streams = []
    for lines in pages_content:
        part = ["BT /F1 11 Tf"]
        y = 780.0
        for ln in lines:
            esc = ln.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
            part.append(f"1 0 0 1 50 {y} Tm ({esc}) Tj")
            y -= 22
        part.append("ET")
        streams.append("\n".join(part).encode("latin-1", "replace"))
    page_ids = [4 + 2 * i for i in range(len(streams))]
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {len(streams)} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for pid, stream in zip(page_ids, streams, strict=True):
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
            f"/Contents {pid + 1} 0 R /Resources << /Font << /F1 3 0 R >> >> >>".encode()
        )
        objects.append(
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
        )
    out = _io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets: list[int] = []
    for i, body_obj in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(f"{i} 0 obj\n".encode() + body_obj + b"\nendobj\n")
    xref_pos = out.tell()
    out.write(f"xref\n0 {len(objects) + 1}\n".encode())
    out.write(b"0000000000 65535 f \n")
    for off in offsets:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF".encode()
    )
    pdf_path = tmp_path / "cover_formula.pdf"
    pdf_path.write_bytes(out.getvalue())
    assert select_pdf_engine(pdf_path) == "docling"


# ---------------------------------------------------------------------------
# Factory wiring
# ---------------------------------------------------------------------------


def test_factory_registers_pdfium_and_routes_explicitly() -> None:
    assert _PDF_ENGINE_REGISTRY["pdfium"] is PDFiumAdapter
    adapter = get_adapter_for_path("paper.pdf", pdf_engine="pdfium")
    assert isinstance(adapter, PDFiumAdapter)
    assert adapter.engine_name == "pdfium"


def test_factory_auto_routes_single_column_file_to_pdfium(tmp_path: Path) -> None:
    pdf_path = build_text_pdf(tmp_path / "born_digital.pdf", SINGLE_COL_LINES)
    adapter = get_adapter_for_path(pdf_path, pdf_engine="auto")
    assert isinstance(adapter, PDFiumAdapter)


def test_factory_auto_routes_two_column_file_to_docling(tmp_path: Path) -> None:
    left = ["Left column body text line that is long enough to wrap nicely."] * 4
    right = ["Right column body text line that is long enough to wrap too."] * 4
    pdf_path = build_text_pdf(tmp_path / "paper.pdf", left, col2_lines=right)
    adapter = get_adapter_for_path(pdf_path, pdf_engine="auto")
    assert isinstance(adapter, DoclingPDFAdapter)
    assert adapter.engine_name == "docling"


def test_factory_explicit_docling_unchanged() -> None:
    adapter = get_adapter_for_path("paper.pdf", pdf_engine="docling")
    assert isinstance(adapter, DoclingPDFAdapter)
    assert adapter.engine_name == "docling"


@pytest.mark.fast
def test_pdfium_clustering_preserves_horizontal_word_order() -> None:
    from ubt.adapters.pdf.pdfium_adapter import _cluster_rects_into_lines

    # Two words on the same line with slight subpixel vertical baseline jitter
    # Word 1 (left=10, right=50, bottom=500, top=512.0)
    # Word 2 (left=55, right=100, bottom=500.1, top=512.3)
    rects = [
        (55.0, 500.1, 100.0, 512.3),  # Word 2 has higher top (512.3)
        (10.0, 500.0, 50.0, 512.0),  # Word 1 has lower top (512.0)
    ]
    median_h = 12.0
    lines = _cluster_rects_into_lines(rects, median_h)
    assert len(lines) == 1, f"Expected 1 clustered line, got {len(lines)}"
    # First rect in the line must be Word 1 (left=10.0), not Word 2 (left=55.0)
    assert lines[0][0][0] == 10.0
    assert lines[0][1][0] == 55.0
