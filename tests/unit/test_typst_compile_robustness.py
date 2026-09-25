"""Typst subprocess robustness: missing binary and timeouts must not raise."""

from __future__ import annotations

from pathlib import Path

import pytest

from ubt.adapters.pdf.rigid.typesetter import RigidTypesetter
from ubt.adapters.pdf.typst_compile import typst_available, typst_compile
from ubt.core.exceptions import DocumentParseError
from ubt.core.ir.models import BookManifest, ChapterMeta

_MISSING = "ubt-typst-binary-that-does-not-exist"


def test_typst_available_false_for_missing_binary() -> None:
    assert typst_available(_MISSING) is False


def test_typst_compile_returns_verdict_not_exception() -> None:
    ok, err = typst_compile("/nonexistent/input.typ", "/nonexistent/out.pdf", _MISSING)
    assert ok is False
    assert "not found" in err


@pytest.mark.asyncio
async def test_anchored_render_fails_closed_without_compiler(tmp_path: Path) -> None:
    manifest = BookManifest(
        doc_id="d",
        title="t",
        source_path=str(tmp_path / "missing.pdf"),
        source_lang="en",
        target_lang="zh",
        chapters=[ChapterMeta(chapter_id="c", title="c", spine_index=1)],
        metadata={},
    )
    engine = RigidTypesetter(typst_binary=_MISSING)
    with pytest.raises(DocumentParseError, match="not found on system"):
        await engine.render(
            manifest=manifest,
            blocks=[],
            target_lang="zh",
            output_path=tmp_path / "out.pdf",
        )


def test_typst_reconstructor_close_releases_math_renderer() -> None:
    from ubt.adapters.pdf.typst_reconstructor import TypstReconstructor

    class _Renderer:
        closed = False

        def close(self) -> None:
            self.closed = True

    recon = TypstReconstructor()
    renderer = _Renderer()
    recon._math_renderer = renderer
    recon.close()
    assert renderer.closed
    recon.close()  # idempotent


def test_docling_adapter_close_delegates_to_reconstructor() -> None:
    from ubt.adapters.pdf.docling_adapter import DoclingPDFAdapter
    from ubt.adapters.pdf.typst_reconstructor import TypstReconstructor

    class _Recon(TypstReconstructor):
        closed = False

        def close(self) -> None:
            self.closed = True

    recon = _Recon()
    adapter = DoclingPDFAdapter(reconstructor=recon)
    adapter.close()
    assert recon.closed
