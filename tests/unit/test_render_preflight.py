"""The pre-draft render preflight must really compile a sample.

It exists to fail before any token is billed when the PDF renderer (Typst) is
dead. LayerCompositor descends a fragment it cannot typeset to the source page
instead of raising, so the gate has to detect that *nothing* was drawn; a
source-only sample would otherwise pass without ever touching the typesetter.
"""

from __future__ import annotations

import asyncio
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
    """Stand-in for ``typst_compile``; writes a page of the requested size."""

    def __init__(self, *, ok: bool) -> None:
        self.ok = ok
        self.calls = 0

    def __call__(self, typ_path: str, pdf_path: str, binary: str) -> tuple[bool, str]:
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
    monkeypatch.setattr("ubt.render.outputs.typst_compile", fake)
    monkeypatch.setattr("ubt.render.outputs.cache_root", lambda: tmp_path / "cache")


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

    with pytest.raises(DocumentParseError, match="could not typeset any of the"):
        asyncio.run(
            run_render_preflight(
                adapter=adapter, manifest=manifest, blocks=blocks, target_lang="zh"
            )
        )


def test_preflight_sample_reaches_the_typesetter_for_a_formula_heavy_book(
    tmp_path: Path,
) -> None:
    # A formula/table-heavy book's sample is all static-skip blocks; the old
    # placeholder kept skip_translate=True, so overlays_from_blocks returned
    # nothing and the "nothing was drawn" guard was inert (a dead Typst passed).
    from ubt.core.engine.render_preflight import (
        _placeholder_blocks,
        select_preflight_sample,
    )
    from ubt.render.outputs import overlays_from_blocks

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
    assert sample  # the sample is not empty
    assert len(overlays_from_blocks(_placeholder_blocks(sample), None)) >= 1


def test_placeholder_text_replaces_an_empty_formula_body() -> None:
    # ``docling_parser`` writes a bare ``$$`` for a formula it could not read.
    # ``typeset_math`` rejects that empty body, so the rehearsal must substitute
    # a synthetic one or the gate mistakes the placeholder for a dead toolchain.
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
    # Regression: a formula-dense book whose formulas all extracted as bare
    # ``$$`` placeholders used to abort before translation, because the sample
    # is deliberately formula-first and every placeholder failed to typeset.
    fake = _FakeTypst(ok=True)
    _patch(monkeypatch, tmp_path, fake)
    source = write_text_pdf(tmp_path / "src.pdf", [("Title", "The body of the page.")])
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
