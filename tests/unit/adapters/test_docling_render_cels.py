"""Unit tests for CELS coordinator & distortion engine assembly into DoclingRenderStrategy."""

from __future__ import annotations

from pathlib import Path

import pikepdf
import pytest

from ubt.adapters.pdf.docling_render import DoclingRenderStrategy
from ubt.core.ir.models import BlockType, BookManifest, BoundingBox, IRBlock, make_element
from ubt.render.typst.facsimile_renderer import FacsimilePageSpec, FacsimileRenderer

pytestmark = pytest.mark.fast


@pytest.mark.asyncio
async def test_cels_clean_page_routes_l1_and_records_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clean single-column page without overlaps or deficits routes to L1 with CELS metrics."""
    src_pdf = tmp_path / "clean_cels_src.pdf"
    with pikepdf.Pdf.new() as pdf:
        pdf.add_blank_page(page_size=(500, 700))
        pdf.save(src_pdf)

    blocks = [
        IRBlock(
            element=make_element(
                id="b_hdr",
                spine_index=1,
                block_type=BlockType.HEADING,
                source_text="Introduction",
                bbox=BoundingBox(page=1, x0=50.0, y0=620.0, x1=450.0, y1=650.0),
            ),
            target_text="第一章 引言",
        ),
        IRBlock(
            element=make_element(
                id="b_p1",
                spine_index=2,
                block_type=BlockType.NARRATIVE,
                source_text="This is clean narrative paragraph one.",
                bbox=BoundingBox(page=1, x0=50.0, y0=540.0, x1=450.0, y1=600.0),
            ),
            target_text="这是第一段干净的正文内容，没有任何几何冲突。",
        ),
        IRBlock(
            element=make_element(
                id="b_p2",
                spine_index=3,
                block_type=BlockType.NARRATIVE,
                source_text="This is clean narrative paragraph two.",
                bbox=BoundingBox(page=1, x0=50.0, y0=460.0, x1=450.0, y1=520.0),
            ),
            target_text="这是第二段干净的正文内容，与上一段保持充裕的段间距。",
        ),
    ]

    rendered_pages: list[FacsimilePageSpec] = []

    def fake_render_doc(
        self: FacsimileRenderer,
        pages: list[FacsimilePageSpec],
        bg_pdf_path: Path,
        output_path: Path,
        build_dir: Path,
        **kwargs: object,
    ) -> Path:
        rendered_pages.extend(pages)
        output_path.write_bytes(src_pdf.read_bytes())
        return output_path

    monkeypatch.setattr(FacsimileRenderer, "render_document", fake_render_doc)

    strategy = DoclingRenderStrategy(delivery_mode="facsimile", enable_pdf_optimization=False)
    manifest = BookManifest(
        doc_id="doc_cels_clean",
        title="CELS Clean Page",
        source_path=str(src_pdf),
        metadata={"source_path": str(src_pdf)},
    )
    out_pdf = tmp_path / "out_cels_clean.pdf"
    await strategy.render_blocks(manifest, blocks, "zh", out_pdf)

    assert len(rendered_pages) == 1
    assert rendered_pages[0].tier == "L1"

    # Verify CELS metrics recorded in page_geometry
    geom = strategy.last_page_geometry
    assert len(geom) == 1
    row = geom[0]
    assert row["tier"] == "L1"
    assert "cels_distortion_score" in row
    assert row["cels_distortion_score"] < 0.15
    assert row["cels_distortion_verdict"] == "faithful_deliverable"
    assert row["cels_deficit_pt"] == 0.0
    assert row["cels_is_multicolumn"] is False
    assert row["cels_track_count"] == 1


@pytest.mark.asyncio
async def test_cels_high_distortion_auto_degrades_to_l2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When CELS distortion is critical (e.g. source blocks heavily overlap), the page routes to L2."""
    src_pdf = tmp_path / "corrupt_cels_src.pdf"
    with pikepdf.Pdf.new() as pdf:
        pdf.add_blank_page(page_size=(500, 700))
        pdf.save(src_pdf)

    # 3 paragraphs occupying the EXACT same bounding box (100% collision)
    blocks = [
        IRBlock(
            element=make_element(
                id=f"b_collide_{i}",
                spine_index=i,
                block_type=BlockType.NARRATIVE,
                source_text=f"Colliding block {i}",
                bbox=BoundingBox(page=1, x0=50.0, y0=400.0, x1=450.0, y1=550.0),
            ),
            target_text=f"重叠碰撞正文内容 {i}，严重占用同一物理区域导致几何崩溃。",
        )
        for i in range(1, 4)
    ]

    rendered_pages: list[FacsimilePageSpec] = []

    def fake_render_doc(
        self: FacsimileRenderer,
        pages: list[FacsimilePageSpec],
        bg_pdf_path: Path,
        output_path: Path,
        build_dir: Path,
        **kwargs: object,
    ) -> Path:
        rendered_pages.extend(pages)
        output_path.write_bytes(src_pdf.read_bytes())
        return output_path

    monkeypatch.setattr(FacsimileRenderer, "render_document", fake_render_doc)

    strategy = DoclingRenderStrategy(delivery_mode="facsimile", enable_pdf_optimization=False)
    manifest = BookManifest(
        doc_id="doc_cels_corrupt",
        title="CELS Corrupt Page",
        source_path=str(src_pdf),
        metadata={"source_path": str(src_pdf)},
    )
    out_pdf = tmp_path / "out_cels_corrupt.pdf"
    await strategy.render_blocks(manifest, blocks, "zh", out_pdf)

    assert len(rendered_pages) == 1
    # Degraded to L2 page-locked reflow
    assert rendered_pages[0].tier == "L2"

    geom = strategy.last_page_geometry
    assert len(geom) == 1
    row = geom[0]
    assert row["tier"] == "L2"
    assert (
        row["cels_distortion_score"] >= 0.15
        or row["cels_distortion_verdict"] != "faithful_deliverable"
    )


@pytest.mark.asyncio
async def test_cels_two_column_page_partitions_tracks_in_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A two-column document is recognized by CELS column detection during render."""
    src_pdf = tmp_path / "twocol_cels_src.pdf"
    with pikepdf.Pdf.new() as pdf:
        pdf.add_blank_page(page_size=(612, 792))
        pdf.save(src_pdf)

    # Left column: 54..280, Right column: 332..558
    blocks = [
        IRBlock(
            element=make_element(
                id=f"left_{i}",
                spine_index=i,
                block_type=BlockType.NARRATIVE,
                source_text=f"Left text {i}",
                bbox=BoundingBox(
                    page=1, x0=54.0, y0=100.0 + i * 120.0, x1=280.0, y1=200.0 + i * 120.0
                ),
            ),
            target_text=f"左栏段落文本 {i}",
        )
        for i in range(4)
    ] + [
        IRBlock(
            element=make_element(
                id=f"right_{i}",
                spine_index=10 + i,
                block_type=BlockType.NARRATIVE,
                source_text=f"Right text {i}",
                bbox=BoundingBox(
                    page=1, x0=332.0, y0=100.0 + i * 120.0, x1=558.0, y1=200.0 + i * 120.0
                ),
            ),
            target_text=f"右栏段落文本 {i}",
        )
        for i in range(4)
    ]

    rendered_pages: list[FacsimilePageSpec] = []

    def fake_render_doc(
        self: FacsimileRenderer,
        pages: list[FacsimilePageSpec],
        bg_pdf_path: Path,
        output_path: Path,
        build_dir: Path,
        **kwargs: object,
    ) -> Path:
        rendered_pages.extend(pages)
        output_path.write_bytes(src_pdf.read_bytes())
        return output_path

    monkeypatch.setattr(FacsimileRenderer, "render_document", fake_render_doc)

    strategy = DoclingRenderStrategy(delivery_mode="facsimile", enable_pdf_optimization=False)
    manifest = BookManifest(
        doc_id="doc_cels_twocol",
        title="CELS Two-Column Page",
        source_path=str(src_pdf),
        metadata={"source_path": str(src_pdf)},
    )
    out_pdf = tmp_path / "out_cels_twocol.pdf"
    await strategy.render_blocks(manifest, blocks, "zh", out_pdf)

    assert len(rendered_pages) == 1
    assert rendered_pages[0].tier == "L1"

    row = strategy.last_page_geometry[0]
    assert row["cels_is_multicolumn"] is True
    assert row["cels_track_count"] == 2
    assert row["cels_distortion_score"] < 0.15


@pytest.mark.asyncio
async def test_cels_rollback_switch_falls_back_to_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When enable_cels=False, strategy bypasses CELS coordinator and falls back to legacy packing."""
    src_pdf = tmp_path / "rollback_src.pdf"
    with pikepdf.Pdf.new() as pdf:
        pdf.add_blank_page(page_size=(500, 700))
        pdf.save(src_pdf)

    blocks = [
        IRBlock(
            element=make_element(
                id="b_p1",
                spine_index=1,
                block_type=BlockType.NARRATIVE,
                source_text="This is narrative paragraph one.",
                bbox=BoundingBox(page=1, x0=50.0, y0=540.0, x1=450.0, y1=600.0),
            ),
            target_text="这是第一段内容。",
        ),
    ]

    rendered_pages: list[FacsimilePageSpec] = []

    def fake_render_doc(
        self: FacsimileRenderer,
        pages: list[FacsimilePageSpec],
        bg_pdf_path: Path,
        output_path: Path,
        build_dir: Path,
        **kwargs: object,
    ) -> Path:
        rendered_pages.extend(pages)
        output_path.write_bytes(src_pdf.read_bytes())
        return output_path

    monkeypatch.setattr(FacsimileRenderer, "render_document", fake_render_doc)

    # Instantiate strategy with rollback switch enable_cels=False
    strategy = DoclingRenderStrategy(
        delivery_mode="facsimile",
        enable_pdf_optimization=False,
        enable_cels=False,
    )
    assert strategy.enable_cels is False

    manifest = BookManifest(
        doc_id="doc_cels_rollback",
        title="CELS Rollback Page",
        source_path=str(src_pdf),
        metadata={"source_path": str(src_pdf)},
    )
    out_pdf = tmp_path / "out_cels_rollback.pdf"
    await strategy.render_blocks(manifest, blocks, "zh", out_pdf)

    assert len(rendered_pages) == 1
    assert rendered_pages[0].tier == "L1"

    # Verify that CELS metrics are disabled in page_geometry
    row = strategy.last_page_geometry[0]
    assert row["cels_enabled"] is False
    assert "cels_distortion_score" not in row
