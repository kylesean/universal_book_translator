from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ubt.core.ir.models import BlockStatus, BlockType, BoundingBox, IRBlock, make_element
from ubt.render import page_preview

pytestmark = pytest.mark.fast


def _block(page: int = 1) -> IRBlock:
    return IRBlock(
        element=make_element(
            id="b1",
            spine_index=0,
            block_type=BlockType.FORMULA,
            source_text=r"$$ \frac{a}{} $$",
            bbox=BoundingBox(page=page, x0=0.0, y0=0.0, x1=100.0, y1=10.0),
        ),
        status=BlockStatus.MTQE_PASSED,
    )


def test_preview_renderer_receives_the_math_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}

    class _SpyRenderer:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def build_source(
            self, blocks: list[IRBlock], build_dir: Path, target_lang: str
        ) -> tuple[str, Any]:
            raise page_preview.PagePreviewUnavailable("spy: constructed")

    monkeypatch.setattr("ubt.render.typst.document_renderer.TypstDocumentRenderer", _SpyRenderer)
    source_pdf = tmp_path / "src.pdf"
    source_pdf.write_bytes(b"%PDF-1.4\n")

    with pytest.raises(page_preview.PagePreviewUnavailable):
        page_preview.render_page_preview(
            source_pdf=source_pdf,
            blocks=[_block()],
            target_lang="zh",
            page=1,
            workdir=tmp_path / "work",
        )

    assert "math_gate" in captured, "the preview renderer must be built with a math gate"
    assert callable(captured["math_gate"])
