from __future__ import annotations

from pathlib import Path
from typing import Any

import pikepdf
import pytest

from ubt.adapters.pdf.docling_render import DoclingRenderStrategy
from ubt.core.ir.models import (
    BlockType,
    BookManifest,
    BoundingBox,
    FlowID,
    IRBlock,
    make_element,
)
from ubt.render.typst.facsimile_renderer import FacsimileRenderer

pytestmark = pytest.mark.fast


def _heading(block_id: str, text: str, page: int, *, level: int = 1) -> IRBlock:
    return IRBlock(
        element=make_element(
            id=block_id,
            spine_index=page - 1,
            block_type=BlockType.HEADING,
            flow_id=FlowID.MAIN_STORY,
            source_text=f"Source heading {page}",
            bbox=BoundingBox(page=page, x0=40.0, y0=500.0, x1=360.0, y1=540.0),
            level=level,
        ),
        target_text=text,
    )


def _source_pdf(path: Path, pages: int) -> Path:
    pdf = pikepdf.Pdf.new()
    for _ in range(pages):
        pdf.add_blank_page(page_size=(400, 600))
    pdf.save(path)
    pdf.close()
    return path


@pytest.fixture(name="stub_compile")
def _stub_compile(monkeypatch: pytest.MonkeyPatch) -> None:

    def _render(self: FacsimileRenderer, pages, bg, out, build_dir, **kwargs):  # type: ignore[no-untyped-def]
        pdf = pikepdf.Pdf.new()
        for _ in pages:
            pdf.add_blank_page(page_size=(400, 600))
        pdf.save(out)
        pdf.close()
        return out

    monkeypatch.setattr(FacsimileRenderer, "render_document", _render)


def _outline(path: Path) -> list[str]:
    out: list[str] = []

    def walk(pdf: pikepdf.Pdf, items: object, depth: int) -> None:
        for item in items:  # type: ignore[attr-defined]
            page = pdf.pages.index(item.destination[0]) + 1
            out.append(f"{'  ' * (depth - 1)}{item.title}@{page}")
            walk(pdf, item.children, depth + 1)

    with pikepdf.open(path) as pdf, pdf.open_outline() as ol:
        walk(pdf, ol.root, 1)
    return out


async def _render(strategy: DoclingRenderStrategy, manifest: Any, blocks: Any, out: Path) -> None:
    await strategy.render_blocks(manifest, blocks, "zh", out)


def test_facsimile_render_writes_translated_bookmarks(tmp_path: Path, stub_compile: None) -> None:
    import asyncio

    src = _source_pdf(tmp_path / "book.pdf", 3)
    manifest = BookManifest(doc_id="d", title="t", source_path=str(src), target_lang="zh")
    blocks = [
        _heading("h1", "第一章 绪论", 1, level=1),
        _heading("h2", "1.1 背景", 2, level=2),
        _heading("h3", "第二章 方法", 3, level=1),
    ]
    out = tmp_path / "out.pdf"
    strategy = DoclingRenderStrategy(delivery_mode="facsimile", enable_pdf_optimization=False)
    asyncio.run(_render(strategy, manifest, blocks, out))

    assert _outline(out) == [
        "第一章 绪论@1",
        "  1.1 背景@2",
        "第二章 方法@3",
    ]
    assert manifest.metadata["pdf_outline_entries"] == 3


def test_facsimile_render_opens_the_reader_with_the_sidebar(
    tmp_path: Path, stub_compile: None
) -> None:
    import asyncio

    src = _source_pdf(tmp_path / "book.pdf", 2)
    manifest = BookManifest(doc_id="d", title="t", source_path=str(src), target_lang="zh")
    out = tmp_path / "out.pdf"
    strategy = DoclingRenderStrategy(delivery_mode="facsimile", enable_pdf_optimization=False)
    asyncio.run(_render(strategy, manifest, [_heading("h1", "第一章", 1)], out))

    with pikepdf.open(out) as pdf:
        assert str(pdf.Root.get("/PageMode")) == "/UseOutlines"


def test_facsimile_render_records_no_outline_when_there_is_no_heading(
    tmp_path: Path, stub_compile: None
) -> None:
    import asyncio

    src = _source_pdf(tmp_path / "book.pdf", 2)
    manifest = BookManifest(doc_id="d", title="t", source_path=str(src), target_lang="zh")
    out = tmp_path / "out.pdf"
    strategy = DoclingRenderStrategy(delivery_mode="facsimile", enable_pdf_optimization=False)
    asyncio.run(_render(strategy, manifest, [], out))

    assert "pdf_outline_entries" not in manifest.metadata
    assert _outline(out) == []


def test_reflow_channel_never_reaches_the_outline_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    import ubt.adapters.pdf.outline_writer as ow

    calls: list[object] = []
    real = ow.write_outline_from_blocks

    def spy(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(ow, "write_outline_from_blocks", spy)

    async def _fake_publish(self, manifest, blocks, target_lang, output_path, bilingual=False):  # type: ignore[no-untyped-def]
        pdf = pikepdf.Pdf.new()
        for _ in range(2):
            pdf.add_blank_page(page_size=(400, 600))
        pdf.save(output_path)
        pdf.close()
        manifest.metadata["delivery_mode"] = "reflow"
        return output_path

    monkeypatch.setattr(DoclingRenderStrategy, "_render_publish", _fake_publish)

    src = _source_pdf(tmp_path / "book.pdf", 5)
    manifest = BookManifest(doc_id="d", title="t", source_path=str(src), target_lang="zh")
    out = tmp_path / "out.pdf"
    strategy = DoclingRenderStrategy(delivery_mode="reflow", enable_pdf_optimization=False)
    asyncio.run(strategy.render_blocks(manifest, [_heading("h1", "第一章", 4, level=1)], "zh", out))

    assert calls == [], "the reflow channel must not write bookmarks"
    assert "pdf_outline_entries" not in manifest.metadata
    assert _outline(out) == []
