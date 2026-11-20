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


def test_the_factory_only_uses_the_codec_when_asked(tmp_path: Path) -> None:
    source = tmp_path / "book.md"
    source.write_text("# Title\n", encoding="utf-8")

    legacy = get_adapter_for_path(source)
    codec = get_adapter_for_path(source, prefer_codec=True)
    assert not isinstance(legacy, DocumentCodecAdapter)
    assert isinstance(codec, DocumentCodecAdapter)


@pytest.mark.asyncio
async def test_markdown_heading_levels_survive_the_codec_bridge(tmp_path: Path) -> None:
    source = tmp_path / "book.md"
    source.write_text("# Top\n\n### Deep\n", encoding="utf-8")

    adapter = DocumentCodecAdapter(source)
    blocks = []
    async for chapter in adapter.parse_stream(source):
        blocks.extend(chapter.blocks)

    assert [getattr(block.element, "level", 0) for block in blocks] == [1, 3]
