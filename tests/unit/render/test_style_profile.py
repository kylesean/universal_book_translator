"""Unit tests for DocumentStyleProfile and visual style cloning (P3 / G3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ubt.adapters.pdf.docling_render import DoclingRenderStrategy
from ubt.core.ir.models import (
    BlockType,
    BookManifest,
    BoundingBox,
    IRBlock,
    StyleMeta,
    make_element,
)
from ubt.render.typst.academic import academic_preamble, academic_title_banner
from ubt.render.typst.document_renderer import TypstDocumentRenderer
from ubt.render.typst.style_profile import (
    DEFAULT_PRIMARY_HEADING_HEX,
    extract_document_style_profile,
)

pytestmark = pytest.mark.fast


def test_extract_document_style_profile_dominant_color() -> None:
    blocks = [
        IRBlock(
            element=make_element(
                id="h1",
                spine_index=1,
                block_type=BlockType.HEADING,
                source_text="Introduction",
                bbox=BoundingBox(page=1, x0=50.0, y0=700.0, x1=200.0, y1=720.0),
            ),
            style=StyleMeta(color_hex="#8b0000"),
            target_text="引言",
        ),
        IRBlock(
            element=make_element(
                id="p1",
                spine_index=2,
                block_type=BlockType.NARRATIVE,
                source_text="Prose paragraph here.",
                bbox=BoundingBox(page=1, x0=50.0, y0=650.0, x1=350.0, y1=690.0),
            ),
            style=StyleMeta(font_size=10.5),
            target_text="正文段落。",
        ),
        IRBlock(
            element=make_element(
                id="h2",
                spine_index=3,
                block_type=BlockType.HEADING,
                source_text="Methodology",
                bbox=BoundingBox(page=2, x0=50.0, y0=700.0, x1=200.0, y1=720.0),
            ),
            style=StyleMeta(color_hex="#8b0000"),
            target_text="方法论",
        ),
    ]

    profile = extract_document_style_profile(blocks, page_dims=[(540.0, 720.0), (540.0, 720.0)])
    assert profile.has_colored_headings is True
    assert profile.primary_heading_hex == "#8b0000"
    assert profile.body_size_pt == 10.5
    assert profile.page_width_pt == 540.0
    assert profile.page_height_pt == 720.0
    assert 1 in profile.per_page_profiles
    assert 2 in profile.per_page_profiles
    assert profile.per_page_profiles[1].primary_heading_hex == "#8b0000"


def test_extract_document_style_profile_monochrome() -> None:
    blocks = [
        IRBlock(
            element=make_element(
                id="h1",
                spine_index=1,
                block_type=BlockType.HEADING,
                source_text="Introduction",
            ),
            style=StyleMeta(color_hex="#000000"),
            target_text="引言",
        ),
        IRBlock(
            element=make_element(
                id="p1",
                spine_index=2,
                block_type=BlockType.NARRATIVE,
                source_text="Prose paragraph here.",
            ),
            target_text="正文段落。",
        ),
    ]

    profile = extract_document_style_profile(blocks)
    assert profile.has_colored_headings is False
    assert profile.primary_heading_hex == DEFAULT_PRIMARY_HEADING_HEX


def test_academic_preamble_clones_heading_color() -> None:
    preamble_uncolored = academic_preamble(target_lang="zh", columns=1)
    assert "fill: rgb(" not in preamble_uncolored

    preamble_colored = academic_preamble(target_lang="zh", columns=1, heading_color_hex="#8b0000")
    assert 'fill: rgb("#8b0000")' in preamble_colored
    assert "#show heading.where(level: 1)" in preamble_colored
    assert "#show heading.where(level: 2)" in preamble_colored


def test_academic_title_banner_clones_title_color() -> None:
    banner_uncolored = academic_title_banner(
        title="Test Title",
        authors=["Alice"],
    )
    assert "fill: rgb(" not in banner_uncolored

    banner_colored = academic_title_banner(
        title="Test Title",
        authors=["Alice"],
        title_color_hex="#1a5276",
    )
    assert 'fill: rgb("#1a5276")' in banner_colored
    assert "Test Title" in banner_colored


def test_document_renderer_academic_applies_style_profile(tmp_path: Path) -> None:
    blocks = [
        IRBlock(
            element=make_element(
                id="h1",
                spine_index=1,
                block_type=BlockType.HEADING,
                source_text="Chapter 1",
            ),
            style=StyleMeta(color_hex="#8b0000"),
            target_text="第一章",
        ),
        IRBlock(
            element=make_element(
                id="p1",
                spine_index=2,
                block_type=BlockType.NARRATIVE,
                source_text="Prose paragraph.",
            ),
            target_text="正文。",
        ),
    ]

    renderer = TypstDocumentRenderer(layout_template="academic", paper_title="Sample Paper")
    source, stats = renderer.build_source(blocks, tmp_path)
    assert stats.blocks_rendered == 2
    assert 'fill: rgb("#8b0000")' in source


@pytest.mark.asyncio
async def test_docling_render_manifest_metadata_style_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DoclingRenderStrategy records the document_style_profile in manifest.metadata."""
    import ubt.adapters.pdf.typst_compile as tc

    monkeypatch.setattr(tc, "typst_compile", lambda *_a, **_kw: (True, ""))
    strategy = DoclingRenderStrategy(delivery_mode="reflow")

    blocks = [
        IRBlock(
            element=make_element(
                id="h1",
                spine_index=1,
                block_type=BlockType.HEADING,
                source_text="Introduction",
            ),
            style=StyleMeta(color_hex="#8b0000"),
            target_text="引言",
        ),
        IRBlock(
            element=make_element(
                id="b1",
                spine_index=2,
                block_type=BlockType.NARRATIVE,
                source_text="Source paragraph.",
            ),
            style=StyleMeta(font_size=11.0),
            target_text="目标段落。",
        ),
    ]
    manifest = BookManifest(
        doc_id="doc_style",
        title="Style Test",
        source_path=str(tmp_path / "src.pdf"),
        metadata={"source_path": str(tmp_path / "src.pdf")},
    )
    out_pdf = tmp_path / "out_style.pdf"

    await strategy.render_blocks(
        manifest=manifest,
        blocks=blocks,
        target_lang="zh",
        output_path=out_pdf,
    )

    prof_meta = manifest.metadata.get("document_style_profile")
    assert prof_meta is not None
    assert prof_meta["primary_heading_hex"] == "#8b0000"
    assert prof_meta["has_colored_headings"] is True
    assert prof_meta["body_size_pt"] == 11.0
