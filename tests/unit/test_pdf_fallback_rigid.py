"""Test that DoclingPDFAdapter fallback produces paintable blocks for RigidTypesetter."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path

import pytest

from ubt.adapters.pdf.docling_adapter import DoclingPDFAdapter
from ubt.adapters.pdf.rigid.typesetter import RigidTypesetter
from ubt.core.ir.models import BookManifest


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

SAMPLE_TYP = """
#set page(width: 400pt, height: 300pt, margin: 30pt)
#set text(size: 11pt, font: "Liberation Serif")

Sample paragraph to verify that Docling fallback path produces paintable bounding boxes.
"""


def _build_pdf(tmp_path: Path) -> Path:
    typ = tmp_path / "sample.typ"
    pdf = tmp_path / "sample.pdf"
    typ.write_text(SAMPLE_TYP, encoding="utf-8")
    subprocess.run(["typst", "compile", str(typ), str(pdf)], check=True)
    return pdf


def test_docling_fallback_without_docling_renders_rigid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = _build_pdf(tmp_path)
    adapter = DoclingPDFAdapter()

    # Simulate Docling not installed
    monkeypatch.setattr(adapter, "is_docling_installed", lambda: False)

    blocks = adapter._extract_blocks_sync(pdf)
    assert blocks, "Should extract at least one block"

    # Crucial assertion: blocks must carry non-zero geometry, not all-zero dummy boxes
    for b in blocks:
        assert b.bbox is not None
        assert b.bbox.x1 > b.bbox.x0
        assert b.bbox.y1 > b.bbox.y0
        b.target_text = "测试译文：验证非 Docling 环境下依然能正常进行 Rigid 排版。"

    manifest = BookManifest(
        doc_id="test_fallback_rigid",
        title="sample",
        source_path=str(pdf),
        chapters=[],
    )

    out_path = tmp_path / "out_rigid.pdf"
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
    assert len(report.rendered_blocks) > 0
    assert len(report.skipped) == 0
