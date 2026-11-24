from __future__ import annotations

from pathlib import Path

import pytest

from ubt.adapters.codec_adapter import DocumentCodecAdapter
from ubt.adapters.factory import get_adapter_for_path

pytestmark = pytest.mark.fast


@pytest.mark.asyncio
async def test_a_markdown_document_round_trips_through_the_codec_adapter(tmp_path: Path) -> None:
    source = tmp_path / "book.md"
    source.write_text("# Title\n\nHello world.\n\nSecond paragraph.\n", encoding="utf-8")

    adapter = DocumentCodecAdapter(source)
    manifest = await adapter.extract_manifest(source)
    assert manifest.chapters

    blocks = []
    async for chapter in adapter.parse_stream(source):
        blocks.extend(chapter.blocks)
    assert len(blocks) == 3
    for block in blocks:
        block.target_text = f"译 {block.source_text}"

    output = tmp_path / "book_out.md"
    await adapter.render_blocks(manifest, blocks, "zh", output)
    assert output.exists()
    assert "译" in output.read_text(encoding="utf-8")


def test_the_factory_defaults_to_the_codec_and_keeps_an_inspection_adapter(
    tmp_path: Path,
) -> None:
    from ubt.adapters.factory import get_inspection_adapter
    from ubt.adapters.markdown.adapter import MarkdownInspectionAdapter

    source = tmp_path / "book.md"
    source.write_text("# Title\n", encoding="utf-8")

    codec = get_adapter_for_path(source)
    inspection = get_inspection_adapter(source)
    assert isinstance(codec, DocumentCodecAdapter)
    assert isinstance(inspection, MarkdownInspectionAdapter)


@pytest.mark.asyncio
async def test_markdown_heading_levels_survive_the_codec_bridge(tmp_path: Path) -> None:
    source = tmp_path / "book.md"
    source.write_text("# Top\n\n### Deep\n", encoding="utf-8")

    adapter = DocumentCodecAdapter(source)
    blocks = []
    async for chapter in adapter.parse_stream(source):
        blocks.extend(chapter.blocks)

    assert [getattr(block.element, "level", 0) for block in blocks] == [1, 3]


@pytest.mark.asyncio
async def test_codec_adapter_unwraps_marks_and_preserves_blocked_source(tmp_path: Path) -> None:
    from ubt.core.ir.models import BlockStatus

    source = tmp_path / "book.md"
    source.write_text("# Title\n\nFirst line.\n\nSecond line.\n", encoding="utf-8")

    adapter = DocumentCodecAdapter(source)
    manifest = await adapter.extract_manifest(source)
    blocks = []
    async for chapter in adapter.parse_stream(source):
        blocks.extend(chapter.blocks)

    blocks[0].target_text = "标题"
    blocks[1].status = BlockStatus.FAILED
    blocks[1].target_text = '<mark class="ubt-failed-draft" title="err">第一行译文</mark>'
    blocks[2].status = BlockStatus.BLOCKED_HUMAN
    blocks[
        2
    ].target_text = '<mark class="ubt-blocked-human" title="mqm">【待人工审校 | Human review required】Second line.</mark>'

    output = tmp_path / "out.md"
    await adapter.render_blocks(manifest, blocks, "zh", output)
    rendered = output.read_text(encoding="utf-8")
    assert "<mark" not in rendered
    assert "第一行译文" in rendered
    assert "Second line." in rendered
    assert "[UBT]" in rendered


def test_native_pdf_adapter_auto_mode_is_mono_and_unwraps_marks() -> None:
    from ubt.adapters.pdf.native_adapter import (
        NativePDFAdapter,
        _native_translations,
        _resolve_delivery_mode,
    )
    from ubt.codecs.protocol import DeliveryMode
    from ubt.core.ir.models import BlockStatus, BlockType, IRBlock, make_element

    assert _resolve_delivery_mode("auto") is DeliveryMode.MONO
    assert _resolve_delivery_mode("dual") is DeliveryMode.INTERLEAVED

    b_failed = IRBlock(
        element=make_element(
            id="b1", spine_index=0, block_type=BlockType.NARRATIVE, source_text="Src1"
        ),
        status=BlockStatus.FAILED,
        target_text='<mark class="ubt-failed-draft" title="boom">失败译文</mark>',
    )
    b_blocked = IRBlock(
        element=make_element(
            id="b2", spine_index=1, block_type=BlockType.NARRATIVE, source_text="Src2"
        ),
        status=BlockStatus.BLOCKED_HUMAN,
        target_text='<mark class="ubt-blocked-human" title="mqm">【待人工审校】Src2</mark>',
    )
    b_skip = IRBlock(
        element=make_element(
            id="b3",
            spine_index=2,
            block_type=BlockType.NARRATIVE,
            source_text="Src3",
            skip_translate=True,
        ),
        status=BlockStatus.MTQE_PASSED,
        target_text="不应出现",
    )
    assert _native_translations([b_failed, b_blocked, b_skip]) == {"b1": "失败译文"}
    assert NativePDFAdapter().engine_name == "native"
