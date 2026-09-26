"""Tests for the zero-token render pre-flight (D6').

The gate exists so a broken PDF toolchain fails before Stage 3 bills the book.
It must (a) skip non-PDF adapters and mock runs, (b) pick a markup-sensitive
sample, and (c) convert a render failure into a message that says nothing has
been billed yet.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from ubt.core.engine.render_preflight import (
    PREFLIGHT_MAX_BLOCKS,
    run_render_preflight,
    select_preflight_sample,
)
from ubt.core.exceptions import DocumentParseError
from ubt.core.ir.models import BlockType, IRBlock


def _block(
    i: int, text: str = "plain prose", block_type: BlockType = BlockType.NARRATIVE
) -> IRBlock:
    return IRBlock(
        id=f"b{i}", spine_index=i, block_type=block_type, source_text=text, target_text=None
    )


class _FakePdfAdapter:
    engine_name = "fake"

    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.error = error

    async def render_blocks(self, **kwargs: Any) -> Path:
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return Path(kwargs["output_path"])


class _FakeTextAdapter:
    """No engine_name attribute: text formats never reach Typst."""

    async def render_blocks(self, **kwargs: Any) -> Path:  # pragma: no cover
        raise AssertionError("text adapters must not be pre-flighted")


def test_non_pdf_adapter_is_skipped() -> None:
    asyncio.run(
        run_render_preflight(
            adapter=_FakeTextAdapter(),  # type: ignore[arg-type]
            manifest=Any,
            blocks=[_block(1)],
            target_lang="zh",
        )
    )


def test_sample_prefers_math_and_keeps_reading_order() -> None:
    blocks = [_block(i) for i in range(12)]
    blocks[7].source_text = "where $V_{th} = 0.5$ holds"
    blocks[3].block_type = BlockType.FORMULA
    blocks[9].block_type = BlockType.TABLE
    sample = select_preflight_sample(blocks)
    assert len(sample) == PREFLIGHT_MAX_BLOCKS
    ids = [b.id for b in sample]
    assert "b7" in ids and "b3" in ids and "b9" in ids
    assert ids == sorted(ids, key=lambda x: int(x[1:]))  # spine order preserved


def test_preflight_renders_sample_with_metadata(tmp_path: Path) -> None:
    class _Manifest:
        metadata = {"render_engine": "reflow", "bilingual_mode": "monolingual"}

    adapter = _FakePdfAdapter()
    blocks = [_block(i, "text with $x$ math") for i in range(30)]
    asyncio.run(
        run_render_preflight(
            adapter=adapter,  # type: ignore[arg-type]
            manifest=_Manifest(),
            blocks=blocks,
            target_lang="zh",
        )
    )
    assert len(adapter.calls) == 1
    call = adapter.calls[0]
    assert call["render_engine"] == "reflow"
    assert call["bilingual_mode"] == "monolingual"
    assert len(call["blocks"]) == PREFLIGHT_MAX_BLOCKS
    # The scratch render must not linger.
    assert not Path(call["output_path"]).exists()


def test_preflight_does_not_mutate_the_live_manifest() -> None:
    """The scratch render must not record its route on the real run.

    ``render_blocks`` writes the rigid monolingual downgrade onto the manifest
    it is handed. The 5-block sample can route to rigid while the full document
    routes to reflow, so a pre-flight render on the live manifest permanently
    downgraded a requested bilingual run to monolingual (the short-chain
    advisory then preserved it and export read it back).
    """
    from ubt.core.ir.models import BookManifest

    class _MutatingAdapter(_FakePdfAdapter):
        async def render_blocks(self, **kwargs: Any) -> Path:
            run = kwargs["manifest"].run
            run.dual_mode_downgraded = "inline"
            run.bilingual_mode = "monolingual"
            run.effective_dual_mode = "monolingual"
            run.render_engine_effective = "rigid"
            return await super().render_blocks(**kwargs)

    manifest = BookManifest(doc_id="d", title="t", source_path="/tmp/b.pdf")
    before = (
        manifest.run.bilingual_mode,
        manifest.run.effective_dual_mode,
        manifest.run.render_engine_effective,
    )
    adapter = _MutatingAdapter()
    asyncio.run(
        run_render_preflight(
            adapter=adapter,  # type: ignore[arg-type]
            manifest=manifest,
            blocks=[_block(i, "text with $x$ math") for i in range(5)],
            target_lang="zh",
        )
    )
    # The adapter rendered into a throwaway copy, never the live manifest.
    assert adapter.calls[0]["manifest"] is not manifest
    assert (
        manifest.run.bilingual_mode,
        manifest.run.effective_dual_mode,
        manifest.run.render_engine_effective,
    ) == before


def test_render_failure_wraps_with_zero_billed_message() -> None:
    adapter = _FakePdfAdapter(error=DocumentParseError("Typst compilation failed (exit code 1)"))
    with pytest.raises(DocumentParseError) as exc:
        asyncio.run(
            run_render_preflight(
                adapter=adapter,  # type: ignore[arg-type]
                manifest=Any,
                blocks=[_block(1)],
                target_lang="zh",
            )
        )
    msg = str(exc.value)
    assert "before any translation was billed" in msg
    assert "doctor" in msg


def test_unrelated_exception_propagates_uncaught() -> None:
    """Only DocumentParseError is reframed; a bug stays a bug."""
    adapter = _FakePdfAdapter(error=RuntimeError("bug"))
    with pytest.raises(RuntimeError):
        asyncio.run(
            run_render_preflight(
                adapter=adapter,  # type: ignore[arg-type]
                manifest=Any,
                blocks=[_block(1)],
                target_lang="zh",
            )
        )


@pytest.mark.asyncio
async def test_internal_not_implemented_error_is_not_a_legacy_fallback(tmp_path: Path) -> None:
    """Only the base stub's signal means "legacy adapter" (review-2 X16).

    The export stage fell back to the legacy ``render_output`` contract on any
    ``NotImplementedError``, so one raised *inside* a working adapter's render
    path was silently re-rendered through the other contract instead of
    surfacing. The base stubs now raise ``RenderBlocksNotImplementedError``.
    """
    from collections.abc import AsyncIterator

    from ubt.adapters.base import BaseDocumentAdapter
    from ubt.core.engine.ledger import SQLiteJobLedger
    from ubt.core.engine.stages.export import _render_adapter_output
    from ubt.core.ir.models import BookManifest, ChapterIR

    class _BrokenAdapter(BaseDocumentAdapter):
        async def extract_manifest(self, input_path: Path) -> BookManifest:
            raise NotImplementedError

        def parse_stream(
            self, input_path: Path, pages: set[int] | None = None
        ) -> AsyncIterator[ChapterIR]:
            raise NotImplementedError

        async def render_blocks(
            self,
            manifest: BookManifest,
            blocks: list[IRBlock],
            target_lang: str,
            output_path: Path,
            bilingual_mode: str | None = None,
            render_engine: str | None = None,
            **kwargs: Any,
        ) -> Path:
            raise NotImplementedError("unsupported nested form XObject")

    ledger = SQLiteJobLedger(tmp_path / "ledger.sqlite")
    try:
        with pytest.raises(NotImplementedError, match="nested form XObject"):
            await _render_adapter_output(
                adapter=_BrokenAdapter(),
                manifest=BookManifest(doc_id="d", title="t", source_path="b.md", chapters=[]),
                ledger=ledger,
                blocks=[_block(1)],
                target_lang="zh",
                output_path=tmp_path / "out.md",
            )
    finally:
        ledger.close()


def test_config_render_engine_supports_rigid_and_auto() -> None:
    """Verify UBTConfig accepts the 'rigid' and 'auto' render engines."""
    from ubt.core.config import UBTConfig

    cfg_rigid = UBTConfig(render_engine="rigid")
    assert cfg_rigid.render_engine == "rigid"

    cfg_auto = UBTConfig(render_engine="auto")
    assert cfg_auto.render_engine == "auto"


def test_config_render_engine_migrates_retired_engines() -> None:
    """Stored configs naming the deleted engines degrade instead of failing."""
    from ubt.core.config import UBTConfig

    assert UBTConfig.model_validate({"render_engine": "inplace"}).render_engine == "rigid"
    assert UBTConfig.model_validate({"render_engine": "hybrid"}).render_engine == "auto"
