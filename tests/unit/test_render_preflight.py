from __future__ import annotations

import asyncio
import importlib.util
import re
from pathlib import Path
from typing import Any

import pikepdf
import pytest
from pdf_builders import write_text_pdf

from ubt.adapters.factory import get_adapter_for_path
from ubt.core.engine.render_preflight import run_render_preflight
from ubt.core.exceptions import DocumentParseError
from ubt.core.ir.models import (
    BlockType,
    BookManifest,
    BoundingBox,
    FlowID,
    IRBlock,
    make_element,
)

pytestmark = pytest.mark.fast


class _FakeTypst:
    def __init__(self, *, ok: bool) -> None:
        self.ok = ok
        self.calls = 0

    def __call__(
        self,
        typ_path: str,
        pdf_path: str,
        binary: str = "typst",
        timeout: float = 120.0,
        target_lang: str | None = None,
    ) -> tuple[bool, str]:
        self.calls += 1
        if not self.ok:
            return False, "fake compiler failure"
        source = Path(typ_path).read_text(encoding="utf-8")
        width = re.search(r"width:\s*([\d.]+)pt", source)
        height = re.search(r"height:\s*([\d.]+)pt", source)
        page_size = (
            float(width.group(1)) if width else 200.0,
            float(height.group(1)) if height else 12.0,
        )
        with pikepdf.new() as pdf:
            pdf.add_blank_page(page_size=page_size)
            pdf.save(pdf_path)
        return True, ""


def _setup(tmp_path: Path) -> tuple[Any, BookManifest, list[IRBlock]]:
    source = write_text_pdf(
        tmp_path / "src.pdf", [("Title", "The machine relies on attention and runs a pass.")]
    )
    adapter = get_adapter_for_path(source, pdf_engine="pdfium")
    manifest = BookManifest(doc_id="d", title="t", source_path=str(source))
    element = make_element(
        id="b1",
        spine_index=0,
        block_type=BlockType.NARRATIVE,
        flow_id=FlowID.MAIN_STORY,
        source_text="The machine relies on attention and runs a pass.",
        bbox=BoundingBox(page=1, x0=54.0, y0=700.0, x1=354.0, y1=730.0),
    )
    return adapter, manifest, [IRBlock(element=element)]


def _patch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake: _FakeTypst) -> None:
    monkeypatch.setattr("ubt.adapters.pdf.typst_compile.typst_compile", fake)


def test_preflight_really_invokes_the_typesetter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeTypst(ok=True)
    _patch(monkeypatch, tmp_path, fake)
    adapter, manifest, blocks = _setup(tmp_path)

    asyncio.run(
        run_render_preflight(adapter=adapter, manifest=manifest, blocks=blocks, target_lang="zh")
    )

    assert fake.calls >= 1, "the preflight passed without ever compiling a fragment"


def test_preflight_fails_when_no_fragment_can_be_typeset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeTypst(ok=False)
    _patch(monkeypatch, tmp_path, fake)
    adapter, manifest, blocks = _setup(tmp_path)

    with pytest.raises(DocumentParseError, match="compile failed|no usable PDF"):
        asyncio.run(
            run_render_preflight(
                adapter=adapter, manifest=manifest, blocks=blocks, target_lang="zh"
            )
        )


def test_preflight_wraps_a_raw_renderer_error_and_keeps_scratch_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A renderer error must reach the operator with its type and scratch path.

    The reported failure surfaced as ``PdfiumError (see server logs; job_id=...)``
    because the raw pdfium exception escaped the pre-flight's
    ``except DocumentParseError`` and the scratch .typ/.pdf were deleted in
    ``finally``. Both together left nothing to reproduce with.
    """
    adapter, manifest, blocks = _setup(tmp_path)
    scratch_root = tmp_path / "scratch"
    scratch_root.mkdir()
    created: list[Path] = []

    def _mkdtemp(*, prefix: str) -> str:
        directory = scratch_root / f"{prefix}{len(created)}"
        directory.mkdir()
        created.append(directory)
        return str(directory)

    monkeypatch.setattr("ubt.core.engine.render_preflight.tempfile.mkdtemp", _mkdtemp)

    class _BrokenRenderer:
        engine_name = "pdf"

        async def render_blocks(self, **kwargs: Any) -> None:
            Path(kwargs["output_path"]).write_bytes(b"%PDF-1.4\n% truncated just now")
            raise RuntimeError("Failed to load document (PDFium: Data format error).")

    with pytest.raises(DocumentParseError) as exc_info:
        asyncio.run(
            run_render_preflight(
                adapter=_BrokenRenderer(),  # type: ignore[arg-type]
                manifest=manifest,
                blocks=blocks,
                target_lang="zh",
            )
        )

    message = str(exc_info.value)
    assert "RuntimeError" in message
    assert "PDFium: Data format error" in message
    assert created, "the pre-flight did not create a scratch directory"
    assert created[0].is_dir(), "the failing scratch sample was deleted before inspection"
    assert (created[0] / "preflight.pdf").is_file()
    assert f"kept at {created[0]}" in message


def test_preflight_removes_scratch_artifacts_on_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeTypst(ok=True)
    _patch(monkeypatch, tmp_path, fake)
    adapter, manifest, blocks = _setup(tmp_path)
    scratch_root = tmp_path / "scratch"
    scratch_root.mkdir()
    created: list[Path] = []

    def _mkdtemp(*, prefix: str) -> str:
        directory = scratch_root / f"{prefix}{len(created)}"
        directory.mkdir()
        created.append(directory)
        return str(directory)

    monkeypatch.setattr("ubt.core.engine.render_preflight.tempfile.mkdtemp", _mkdtemp)

    asyncio.run(
        run_render_preflight(adapter=adapter, manifest=manifest, blocks=blocks, target_lang="zh")
    )

    assert fake.calls >= 1
    assert created and not created[0].exists(), "a successful scratch render leaked its temp dir"


def test_preflight_sample_reaches_the_typesetter_for_a_formula_heavy_book(
    tmp_path: Path,
) -> None:
    from ubt.core.engine.render_preflight import (
        _placeholder_blocks,
        select_preflight_sample,
    )
    from ubt.render.typst.document_renderer import TypstDocumentRenderer

    blocks = [
        IRBlock(
            element=make_element(
                id=f"b{i}",
                spine_index=i,
                block_type=BlockType.FORMULA,
                flow_id=FlowID.MAIN_STORY,
                source_text=f"x_{i} = y_{i} + 1",
                bbox=BoundingBox(page=1, x0=54.0, y0=700.0 - i * 20, x1=354.0, y1=712.0 - i * 20),
                skip_translate=True,
            )
        )
        for i in range(8)
    ]

    sample = select_preflight_sample(blocks)
    assert sample
    renderer = TypstDocumentRenderer()
    source, stats = renderer.build_source(_placeholder_blocks(sample), tmp_path / "build")
    assert stats.blocks_rendered >= 1, "the sample emitted nothing to compile"
    assert "$" in source, "the formula sample produced no math markup"


def test_placeholder_text_replaces_an_empty_formula_body() -> None:
    from ubt.core.engine.render_preflight import _placeholder_text

    def formula(text: str) -> IRBlock:
        return IRBlock(
            element=make_element(
                id="f",
                spine_index=0,
                block_type=BlockType.FORMULA,
                flow_id=FlowID.MAIN_STORY,
                source_text=text,
                bbox=BoundingBox(page=1, x0=54.0, y0=700.0, x1=200.0, y1=712.0),
                skip_translate=True,
            )
        )

    assert _placeholder_text(formula("$$")) == "preflight"
    assert _placeholder_text(formula("")) == "preflight"
    assert _placeholder_text(formula("x_{i} = y_{i} + 1")) == "x_{i} = y_{i} + 1"


def test_preflight_survives_a_book_of_empty_formula_placeholders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeTypst(ok=True)
    _patch(monkeypatch, tmp_path, fake)
    source = write_text_pdf(tmp_path / "src.pdf", [("Title", "The body of the page.")])
    if importlib.util.find_spec("docling") is None:
        pytest.skip("docling is not installed; the refusal path is pinned separately")
    adapter = get_adapter_for_path(source, pdf_engine="pdfium")
    manifest = BookManifest(doc_id="d", title="t", source_path=str(source))
    blocks = [
        IRBlock(
            element=make_element(
                id=f"f{i}",
                spine_index=i,
                block_type=BlockType.FORMULA,
                flow_id=FlowID.MAIN_STORY,
                source_text="$$",
                bbox=BoundingBox(page=1, x0=54.0, y0=700.0 - i * 20, x1=200.0, y1=712.0 - i * 20),
                skip_translate=True,
            )
        )
        for i in range(5)
    ]

    asyncio.run(
        run_render_preflight(adapter=adapter, manifest=manifest, blocks=blocks, target_lang="zh")
    )

    assert fake.calls >= 1, "the preflight passed without ever compiling a fragment"
