from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from ubt.adapters.pdf.docling_adapter import DoclingPDFAdapter
from ubt.adapters.pdf.docling_render import DoclingRenderStrategy
from ubt.core.content.nodes import ReconstructionRecord
from ubt.core.ir.models import BookManifest
from ubt.core.ports import get_last_render_reconstructions

pytestmark = pytest.mark.fast


class _FakeRenderer:
    def __init__(self) -> None:
        self.last_render_skips: list[tuple[str, str]] = [("t1", "table_unparseable")]
        self.last_render_flags: list[tuple[str, str]] = [("f1", "formula_degraded")]
        self.last_render_reconstructions: list[ReconstructionRecord] = [
            ReconstructionRecord(block_id="tb1", verified=True, corrupt=False, reason="ok"),
        ]
        self.last_outcome: Any = None

    async def render_blocks(self, *args: Any, **kwargs: Any) -> Path:
        return Path("/tmp/opencode/out.pdf")


def _stub_manifest() -> BookManifest:
    return BookManifest(
        doc_id="test-doc",
        title="t",
        source_path="t.pdf",
        chapters=[],
    )


def test_the_adapter_carries_reconstruction_verdicts(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = DoclingPDFAdapter.__new__(DoclingPDFAdapter)
    adapter._renderer = cast("DoclingRenderStrategy", _FakeRenderer())

    import asyncio

    out = asyncio.run(
        adapter.render_blocks(
            manifest=_stub_manifest(),
            blocks=[],
            target_lang="zh",
            output_path=Path("/tmp/opencode/out.pdf"),
        )
    )
    assert out == Path("/tmp/opencode/out.pdf")
    assert adapter.last_render_skips == [("t1", "table_unparseable")]
    assert adapter.last_render_flags == [("f1", "formula_degraded")]
    fake = cast("_FakeRenderer", adapter._renderer)
    assert adapter.last_render_reconstructions == fake.last_render_reconstructions
    assert get_last_render_reconstructions(adapter) == fake.last_render_reconstructions


def test_a_second_render_does_not_report_the_first_verdicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = DoclingPDFAdapter.__new__(DoclingPDFAdapter)
    renderer = _FakeRenderer()
    adapter._renderer = cast("DoclingRenderStrategy", renderer)
    import asyncio

    async def _one() -> None:
        await adapter.render_blocks(
            manifest=_stub_manifest(), blocks=[], target_lang="zh", output_path=Path("o.pdf")
        )

    asyncio.run(_one())
    first = list(adapter.last_render_reconstructions)
    assert first
    renderer.last_render_reconstructions = [
        ReconstructionRecord(block_id="tb2", verified=False, corrupt=True, reason="broken"),
    ]
    asyncio.run(_one())
    assert [r.block_id for r in adapter.last_render_reconstructions] == ["tb2"]
