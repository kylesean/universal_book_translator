"""Unit tests for PageBBoxResolver and scanned page coordinate normalization."""

from pathlib import Path

from ubt.adapters.pdf.coordinate_resolver import (
    PageBBoxResolver,
    synthesize_line_boxes_for_blocks,
)
from ubt.core.ir.models import BlockType, BoundingBox, IRBlock


def test_resolver_normalized_0_1() -> None:
    # Page size: 600 x 800
    resolver = PageBBoxResolver(page_width=600.0, page_height=800.0)

    # Box at top-left: x from 0.1 to 0.5, y from 0.1 to 0.2 (top 10% to 20%)
    raw_box = [0.1, 0.1, 0.5, 0.2]
    rect = resolver.resolve_bbox(raw_box)
    assert rect is not None
    # In PDF points (bottom-left origin):
    # x0 = 0.1 * 600 = 60
    # x1 = 0.5 * 600 = 300
    # y0 = (1 - 0.2) * 800 = 640
    # y1 = (1 - 0.1) * 800 = 720
    assert rect == (60.0, 640.0, 300.0, 720.0)


def test_resolver_normalized_1000() -> None:
    # Page size: 600 x 800
    resolver = PageBBoxResolver(page_width=600.0, page_height=800.0)

    # Box in [0, 1000] space:
    raw_box = [100.0, 100.0, 500.0, 200.0]
    rect = resolver.resolve_bbox(raw_box, coord_system="normalized_1000")
    assert rect is not None
    assert rect == (60.0, 640.0, 300.0, 720.0)

    # Also test auto detection when coordinates exceed page dimensions (>600 or >800)
    raw_box_auto = [100.0, 100.0, 900.0, 200.0]
    rect_auto = resolver.resolve_bbox(raw_box_auto)
    assert rect_auto is not None
    assert rect_auto == (60.0, 640.0, 540.0, 720.0)


def test_resolver_image_pixel_space() -> None:
    # Page size: 600 x 800 pt, rendered image: 1200 x 1600 px (2x scale)
    resolver = PageBBoxResolver(
        page_width=600.0,
        page_height=800.0,
        image_width=1200.0,
        image_height=1600.0,
    )

    # Box in pixel coordinates:
    raw_box = [200.0, 160.0, 600.0, 320.0]
    rect = resolver.resolve_bbox(raw_box, coord_system="image_pixel")
    assert rect is not None
    # Scale is 0.5
    # x0 = 100.0, x1 = 300.0
    # y0 = 800 - 160 = 640.0, y1 = 800 - 80 = 720.0
    assert rect == (100.0, 640.0, 300.0, 720.0)


def test_resolver_rotation_90() -> None:
    # 600 x 800 page rotated 90 degrees
    resolver = PageBBoxResolver(page_width=600.0, page_height=800.0, rotation=90)
    raw_box = [100.0, 200.0, 300.0, 400.0]
    rect = resolver.resolve_bbox(raw_box)
    assert rect is not None
    # Check that rotation is applied and dimensions are valid
    assert rect[0] < rect[2]
    assert rect[1] < rect[3]


def test_degenerate_boxes_are_rejected_in_every_space() -> None:
    """A zero-extent box has no placement, in any coordinate system.

    Only the explicit-PDF branch used to screen this, so a degenerate
    normalized/pixel box came back as a zero-width "resolved" rect that line
    synthesis then mis-handled.
    """
    resolver = PageBBoxResolver(
        page_width=600.0, page_height=800.0, image_width=1200.0, image_height=1600.0
    )

    # Zero width / zero height in normalized [0, 1] space (auto-detected).
    assert resolver.resolve_bbox([0.2, 0.1, 0.2, 0.4]) is None
    assert resolver.resolve_bbox([0.1, 0.3, 0.5, 0.3]) is None
    # Degenerate after the ordering swap (x0 > x1 with equal values).
    assert resolver.resolve_bbox([0.5, 0.1, 0.5, 0.2]) is None
    # Image-pixel space.
    assert resolver.resolve_bbox([100.0, 100.0, 100.0, 400.0], coord_system="image_pixel") is None
    # Native PDF points.
    assert resolver.resolve_bbox([10.0, 10.0, 10.0, 50.0], coord_system="pdf_points") is None
    # A real box in the same spaces still resolves.
    assert resolver.resolve_bbox([0.1, 0.1, 0.5, 0.2]) is not None
    assert (
        resolver.resolve_bbox([100.0, 100.0, 300.0, 400.0], coord_system="image_pixel") is not None
    )


def test_degenerate_block_bbox_yields_no_synthesized_lines() -> None:
    block = IRBlock(
        id="scan_degenerate",
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        source_text="text",
        target_text="文本",
        bbox=BoundingBox(page=1, x0=100.0, y0=500.0, x1=100.0, y1=540.0),
    )
    assert synthesize_line_boxes_for_blocks([block], page_size=(600.0, 800.0)) == []


def test_synthesize_line_boxes_for_blocks() -> None:
    # Simulate a scanned page block with no vlm_lines but a multiline text
    block = IRBlock(
        id="scan_b01",
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        source_text="First line of scanned text.\nSecond line of scanned text.",
        target_text="第一行扫描文本。\n第二行扫描文本。",
        bbox=BoundingBox(page=1, x0=100.0, y0=500.0, x1=400.0, y1=540.0),
    )

    lines = synthesize_line_boxes_for_blocks([block], page_size=(600.0, 800.0))
    assert len(lines) == 2
    # Check text
    assert lines[0].text == "First line of scanned text."
    assert lines[1].text == "Second line of scanned text."
    # Check geometry: line 0 is top line (higher y)
    assert lines[0].rect[1] >= lines[1].rect[3]
    # Check height slice: each is 20pt high (540 - 500 = 40; 40 / 2 = 20)
    assert lines[0].rect[3] - lines[0].rect[1] == 20.0
    assert lines[1].rect[3] - lines[1].rect[1] == 20.0


async def test_scanned_pdf_anchored_rendering(tmp_path: Path) -> None:
    """Pure scanned/textless PDFs render via synthetic OCR rows plus a bg cover."""
    from pathlib import Path

    import pikepdf

    from ubt.adapters.pdf.rigid import RigidTypesetter
    from ubt.core.ir.models import BookManifest

    # 1. Create a pure textless PDF (simulating scan image)
    pdf = pikepdf.new()
    pdf.add_blank_page(page_size=(600, 800))
    scan_pdf_path = tmp_path / "scanned_doc.pdf"
    pdf.save(scan_pdf_path)

    # 2. Block from OCR with bbox but NO embedded PDF text stream
    block = IRBlock(
        id="scan_p01_b01",
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        source_text="Scanned book heading.\nScanned paragraph body continues.",
        target_text="扫描书籍标题。\n扫描段落正文继续。",
        bbox=BoundingBox(page=1, x0=100.0, y0=500.0, x1=450.0, y1=560.0),
    )

    manifest = BookManifest(
        doc_id="test_scan",
        title="Scanned Book",
        source_path=str(scan_pdf_path),
        chapters=[],
    )

    out_pdf = tmp_path / "translated_scan.pdf"
    engine = RigidTypesetter()
    result_path, report = await engine.render(
        manifest=manifest,
        blocks=[block],
        target_lang="zh",
        output_path=out_pdf,
    )

    assert Path(result_path).exists()
    assert "scan_p01_b01" in report.rendered_blocks
