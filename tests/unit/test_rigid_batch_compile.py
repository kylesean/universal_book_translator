"""Tests for single-pass multi-page Typst batch compilation in RigidTypesetter."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pikepdf
import pytest

from ubt.adapters.pdf.rigid.typesetter import RigidTypesetter
from ubt.adapters.pdf.textgeom import extract_lines
from ubt.core.ir.models import BlockType, BookManifest, BoundingBox, FlowID, IRBlock


def _deps_available() -> bool:
    if shutil.which("typst") is None:
        return False
    try:
        import pikepdf  # noqa: F401
        import pypdfium2  # noqa: F401

        from ubt.adapters.pdf.font_metrics import resolve_cjk_ttc

        resolve_cjk_ttc()
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _deps_available(), reason="typst + pikepdf + pypdfium2 + CJK font required"
)

MULTI_PAGE_TYP = """
#set page(width: 400pt, height: 300pt, margin: 30pt)
#set text(size: 11pt, font: "Liberation Serif")

Page one text: This is paragraph one on the first page of the test document with enough words to wrap properly.

#pagebreak()

Page two text: This is paragraph two on the second page of the test document with enough words to wrap properly.

#pagebreak()

Page three text: This is paragraph three on the third page of the test document with enough words to wrap properly.
"""


def _build_multipage_pdf(tmp_path: Path) -> Path:
    typ = tmp_path / "src.typ"
    pdf = tmp_path / "src.pdf"
    typ.write_text(MULTI_PAGE_TYP, encoding="utf-8")
    subprocess.run(["typst", "compile", str(typ), str(pdf)], check=True)
    return pdf


def _blocks_for_multipage(pdf: Path) -> list[IRBlock]:
    blocks: list[IRBlock] = []
    for p in (1, 2, 3):
        lines, _size = extract_lines(pdf, p)
        assert lines, f"Page {p} should have extracted lines"
        blocks.append(
            IRBlock(
                id=f"b_p{p}",
                flow_id=FlowID.MAIN_STORY,
                spine_index=p,
                block_type=BlockType.NARRATIVE,
                source_text=" ".join(line.text for line in lines),
                target_text=f"这是第 {p} 页的中文测试译文段落，用于验证多页批量编译Overlay是否完整排版渲染并保留原本画布。",
                bbox=BoundingBox(
                    page=p,
                    x0=min(line.rect[0] for line in lines),
                    y0=min(line.rect[1] for line in lines),
                    x1=max(line.rect[2] for line in lines),
                    y1=max(line.rect[3] for line in lines),
                ),
            )
        )
    return blocks


def test_rigid_batch_compile_multipage_pdf(tmp_path: Path) -> None:
    """Multi-page documents should compile overlays and merge cleanly."""
    pdf = _build_multipage_pdf(tmp_path)
    with pikepdf.open(str(pdf)) as probe:
        assert len(probe.pages) == 3

    blocks = _blocks_for_multipage(pdf)
    assert len(blocks) == 3

    manifest = BookManifest(
        doc_id="test_multipage_doc",
        title="multipage_test",
        source_path=str(pdf),
        chapters=[],
    )

    out_path = tmp_path / "rendered_output.pdf"
    typesetter = RigidTypesetter()

    rendered, report = asyncio.run(
        typesetter.render(
            manifest=manifest,
            blocks=blocks,
            target_lang="zh",
            output_path=out_path,
        )
    )

    assert rendered.exists()
    assert len(report.rendered_blocks) == 3
    assert len(report.skipped) == 0

    with pikepdf.open(str(rendered)) as res:
        assert len(res.pages) == 3


def test_rigid_batch_compile_fallback_on_page_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If batch compilation fails, per-page fallback must still succeed for valid pages."""
    pdf = _build_multipage_pdf(tmp_path)
    blocks = _blocks_for_multipage(pdf)

    manifest = BookManifest(
        doc_id="test_fallback_doc",
        title="fallback_test",
        source_path=str(pdf),
        chapters=[],
    )

    out_path = tmp_path / "fallback_output.pdf"
    typesetter = RigidTypesetter()

    # Poison batch overlay generation to force fallback
    original_batch = typesetter._batch_page_overlay

    def poisoned_batch(pages: Any, paints: Any) -> tuple[str, list[int]]:
        typ, page_list = original_batch(pages, paints)
        # Inject invalid Typst syntax that causes batch compile failure
        return "#invalid syntax @@@\n" + typ, page_list

    monkeypatch.setattr(typesetter, "_batch_page_overlay", poisoned_batch)

    rendered, report = asyncio.run(
        typesetter.render(
            manifest=manifest,
            blocks=blocks,
            target_lang="zh",
            output_path=out_path,
        )
    )

    assert rendered.exists()
    assert len(report.rendered_blocks) == 3
    assert len(report.skipped) == 0


@pytest.mark.fast
def test_rigid_batch_overlay_no_extra_pagebreak() -> None:
    typesetter = RigidTypesetter()
    facts_p1 = MagicMock(width=500.0, height=700.0, bg=False)
    facts_p2 = MagicMock(width=500.0, height=700.0, bg=False)
    pages: dict[int, Any] = {1: facts_p1, 2: facts_p2}
    zone1 = MagicMock(x0=10.0, y1=50.0, width=100.0, height=40.0)
    zone2 = MagicMock(x0=10.0, y1=50.0, width=100.0, height=40.0)
    paints: dict[int, list[Any]] = {
        1: [MagicMock(zone=zone1, height=700.0, text="hello", size=10.0)],
        2: [MagicMock(zone=zone2, height=700.0, text="world", size=10.0)],
    }

    overlay_src, compiled = typesetter._batch_page_overlay(pages, paints)
    # Typst #set page(...) automatically creates a pagebreak when preceded by content.
    # An explicit #pagebreak() directly before #set page(...) causes an extra blank page.
    assert not (
        "#pagebreak()\n#set page" in overlay_src or "#pagebreak()\r\n#set page" in overlay_src
    ), "Overlay Typst source must not have #pagebreak() immediately preceding #set page"
