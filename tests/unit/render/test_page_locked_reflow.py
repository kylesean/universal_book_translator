"""Unit tests for Tier L2 Page-Locked Reflow Renderer (P2-15)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ubt.core.ir.models import BlockType, BoundingBox, IRBlock, StyleMeta, make_element
from ubt.render.typst.facsimile_renderer import (
    FacsimileBlockPlacement,
    FacsimilePageSpec,
    FacsimileRenderer,
)
from ubt.render.typst.page_locked_renderer import (
    PageGeometry,
    PageLockedPageSpec,
    PageLockedReflowRenderer,
    extract_page_geometry,
    solve_page_lock,
)

pytestmark = pytest.mark.fast


def _make_block(
    block_id: str,
    text: str,
    *,
    page: int = 1,
    bbox: tuple[float, float, float, float] | None = None,
    btype: BlockType = BlockType.NARRATIVE,
    color_hex: str = "#000000",
) -> IRBlock:
    bb = BoundingBox(
        page=page,
        x0=bbox[0] if bbox else 36.0,
        y0=bbox[1] if bbox else 700.0,
        x1=bbox[2] if bbox else 550.0,
        y1=bbox[3] if bbox else 720.0,
    )
    el = make_element(
        id=block_id,
        spine_index=1,
        block_type=btype,
        source_text=text,
        bbox=bb,
    )
    return IRBlock(
        element=el,
        target_text=text,
        style=StyleMeta(color_hex=color_hex),
    )


def test_extract_page_geometry_single_column() -> None:
    blocks = [
        _make_block(
            "b1",
            "Title",
            bbox=(50.0, 750.0, 500.0, 780.0),
            btype=BlockType.HEADING,
            color_hex="#1a56db",
        ),
        _make_block("b2", "Paragraph 1", bbox=(50.0, 680.0, 500.0, 740.0)),
        _make_block("b3", "Paragraph 2", bbox=(50.0, 600.0, 500.0, 670.0)),
    ]
    geom = extract_page_geometry(page_w=595.0, page_h=842.0, blocks=blocks)

    assert geom.page_w == 595.0
    assert geom.page_h == 842.0
    assert geom.margin_left == 50.0
    assert geom.margin_right == 95.0  # 595 - 500
    assert geom.margin_top == 62.0  # 842 - 780
    assert geom.margin_bottom == 108.0  # clamped to max 108pt
    assert geom.columns == 1
    assert geom.theme_color_hex == "#1a56db"


def test_extract_page_geometry_two_column() -> None:
    # 2-column layout on A4 (595pt width): midpoint is 297.5pt
    blocks = [
        _make_block("b1", "Left 1", bbox=(40.0, 700.0, 270.0, 750.0)),
        _make_block("b2", "Left 2", bbox=(40.0, 620.0, 270.0, 680.0)),
        _make_block("b3", "Right 1", bbox=(320.0, 700.0, 550.0, 750.0)),
        _make_block("b4", "Right 2", bbox=(320.0, 620.0, 550.0, 680.0)),
    ]
    geom = extract_page_geometry(page_w=595.0, page_h=842.0, blocks=blocks)

    assert geom.columns == 2
    assert geom.margin_left == 40.0
    assert geom.margin_right == 45.0  # 595 - 550
    assert geom.column_gutter_pt == 14.0


def test_extract_page_geometry_empty_blocks() -> None:
    geom = extract_page_geometry(page_w=500.0, page_h=700.0, blocks=[])
    assert geom.page_w == 500.0
    assert geom.page_h == 700.0
    assert geom.margin_left == 36.0
    assert geom.margin_right == 36.0
    assert geom.columns == 1
    assert geom.theme_color_hex == "#0078ae"


def test_solve_page_lock_roomy() -> None:
    geom = PageGeometry(
        page_w=595.0,
        page_h=842.0,
        margin_top=50.0,
        margin_bottom=50.0,
        margin_left=50.0,
        margin_right=50.0,
        columns=1,
    )
    blocks = [
        _make_block("h1", "Short Heading", btype=BlockType.HEADING),
        _make_block("p1", "Short text paragraph with only a few characters."),
    ]
    size, leading, spacing, tracking = solve_page_lock(geom, blocks, target_lang="zh")
    assert size == 11.5
    assert leading >= 0.35
    assert spacing >= 0.65
    assert tracking == 0.0


def test_solve_page_lock_tight() -> None:
    geom = PageGeometry(
        page_w=400.0,
        page_h=500.0,
        margin_top=40.0,
        margin_bottom=40.0,
        margin_left=40.0,
        margin_right=40.0,
        columns=1,
    )
    # Long text that requires compression to fit
    heavy_text = "这是一段非常非常长的文字内容，用于模拟高密度排版和边界收敛场景。" * 25
    blocks = [_make_block(f"p{i}", heavy_text) for i in range(5)]
    size, leading, spacing, tracking = solve_page_lock(geom, blocks, target_lang="zh")
    assert size < 10.0
    assert leading <= 0.35


def test_build_page_markup() -> None:
    geom = PageGeometry(
        page_w=595.0,
        page_h=842.0,
        margin_top=36.0,
        margin_bottom=36.0,
        margin_left=36.0,
        margin_right=36.0,
        columns=2,
        theme_color_hex="#0078ae",
    )
    blocks = [
        _make_block("h1", "Quantum Field Theory", btype=BlockType.HEADING),
        _make_block("p1", "Introduction to particle physics."),
    ]
    spec = PageLockedPageSpec(
        page_no=1,
        geometry=geom,
        uniform_size_pt=9.5,
        leading_em=0.33,
        par_spacing_em=0.60,
        blocks=blocks,
    )
    renderer = PageLockedReflowRenderer()
    markup = renderer.build_page_markup(spec, target_lang="zh")

    assert "#page(" in markup
    assert "width: 595.0pt" in markup
    assert "height: 842.0pt" in markup
    assert "columns: 2" in markup
    assert "size: 9.5pt" in markup
    assert "leading: 0.33em" in markup
    assert "Quantum Field Theory" in markup
    assert "Introduction to particle physics." in markup
    assert 'fill: rgb("#0078ae")' in markup


def test_page_locked_render_document_mock(tmp_path: Path) -> None:
    geom = PageGeometry(
        page_w=595.0,
        page_h=842.0,
        margin_top=36.0,
        margin_bottom=36.0,
        margin_left=36.0,
        margin_right=36.0,
    )
    spec = PageLockedPageSpec(
        page_no=1,
        geometry=geom,
        uniform_size_pt=10.0,
        leading_em=0.35,
        par_spacing_em=0.65,
        blocks=[_make_block("p1", "Hello Page Lock")],
    )
    out_pdf = tmp_path / "output.pdf"
    build_dir = tmp_path / ".build"

    called_args: list[tuple[str, str, str]] = []

    def mock_compile(src: str, dst: str, lang: str) -> tuple[bool, str]:
        called_args.append((src, dst, lang))
        Path(dst).write_bytes(b"%PDF-1.4 mock pdf")
        return True, ""

    renderer = PageLockedReflowRenderer()
    res = renderer.render_document(
        [spec],
        out_pdf,
        build_dir,
        target_lang="zh",
        compile_fn=mock_compile,
    )
    assert res == out_pdf
    assert len(called_args) >= 1
    assert (build_dir / "page_locked.typ").exists()


def test_facsimile_renderer_hybrid_l1_l2() -> None:
    geom = PageGeometry(
        page_w=595.0,
        page_h=842.0,
        margin_top=36.0,
        margin_bottom=36.0,
        margin_left=36.0,
        margin_right=36.0,
    )
    l2_page = PageLockedPageSpec(
        page_no=2,
        geometry=geom,
        uniform_size_pt=9.0,
        leading_em=0.30,
        par_spacing_em=0.50,
        blocks=[_make_block("p2", "Collided page reflowed")],
    )

    page1 = FacsimilePageSpec(
        page_no=1,
        page_w=595.0,
        page_h=842.0,
        uniform_size_pt=10.0,
        leading_em=0.35,
        tier="L1",
        placements=[
            FacsimileBlockPlacement(
                block_id="b1",
                page=1,
                x0=50.0,
                y0=700.0,
                x1=500.0,
                y1=740.0,
                text="Clean L1 Canvas Text",
            )
        ],
    )
    page2 = FacsimilePageSpec(
        page_no=2,
        page_w=595.0,
        page_h=842.0,
        uniform_size_pt=9.0,
        leading_em=0.30,
        tier="L2",
        l2_spec=l2_page,
    )

    renderer = FacsimileRenderer()
    markup = renderer.build_markup([page1, page2], "bg.pdf", target_lang="zh")

    # Page 1 has background image and #place
    assert 'background: image("bg.pdf", page: 1' in markup
    assert "#place(top + left" in markup
    assert "Clean L1 Canvas Text" in markup

    # Page 2 is L2 page-locked without background image
    assert "Collided page reflowed" in markup
    assert 'background: image("bg.pdf", page: 2' not in markup


@pytest.mark.asyncio
async def test_docling_render_page_locked_routing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pikepdf

    from ubt.adapters.pdf.docling_render import DoclingRenderStrategy
    from ubt.core.ir.models import BookManifest

    strategy = DoclingRenderStrategy(delivery_mode="page_locked")
    assert strategy.delivery_mode == "page_locked"

    # Create dummy source PDF
    src_pdf = tmp_path / "src.pdf"
    pdf = pikepdf.Pdf.new()
    pdf.add_blank_page(page_size=(595, 842))
    pdf.save(src_pdf)

    manifest = BookManifest(
        doc_id="d1",
        title="Test Book",
        source_path=str(src_pdf),
        metadata={"source_path": str(src_pdf)},
    )
    blocks = [
        _make_block("b1", "Section 1", btype=BlockType.HEADING),
        _make_block("b2", "Content paragraph"),
    ]
    out_pdf = tmp_path / "out.pdf"

    def mock_render_doc(self: Any, pages: Any, out: Path, build_dir: Path, **kwargs: Any) -> Path:
        out.write_bytes(b"%PDF-1.4 mock")
        return out

    monkeypatch.setattr(PageLockedReflowRenderer, "render_document", mock_render_doc)

    res = await strategy.render_blocks(manifest, blocks, "zh", out_pdf)
    assert res == out_pdf
    assert manifest.metadata["delivery_mode"] == "page_locked"
