"""The adapter side channel must carry reconstruction verdicts to the contract.

The renderer records per-table ``ReconstructionRecord`` verdicts; the strategy
copies them onto ``last_render_reconstructions``; the export stage reads the
ADAPTER's property and feeds the delivery contract. The adapter hop was the
broken link: it copied skips and flags but never the reconstructions, so the
base property's lazy ``[]`` answered every export and the contract's
RECONSTRUCTED (verified/corrupt) branch was unreachable again — P0-6 regressed
in the assembly layer while every unit test of the renderer and the contract
stayed green (2026-10-14 review, W1-3).
"""

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
    """Stands in for DoclingRenderStrategy: the strategy-side channel shape."""

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
    # The fake renderer ignores it; mypy still checks the declared contract.
    return BookManifest(
        doc_id="test-doc",
        title="t",
        source_path="t.pdf",
        chapters=[],
    )


def test_the_adapter_carries_reconstruction_verdicts(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = DoclingPDFAdapter.__new__(DoclingPDFAdapter)  # skip heavy __init__
    adapter._renderer = cast(
        "DoclingRenderStrategy", _FakeRenderer()
    )  # the fake satisfies the channel shape the test drives

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
    # The two channels that always worked…
    assert adapter.last_render_skips == [("t1", "table_unparseable")]
    assert adapter.last_render_flags == [("f1", "formula_degraded")]
    # …and the one this regression pins:
    fake = cast("_FakeRenderer", adapter._renderer)
    assert adapter.last_render_reconstructions == fake.last_render_reconstructions
    # The consumer path (export reads through the ports helper):
    assert get_last_render_reconstructions(adapter) == fake.last_render_reconstructions


def test_a_second_render_does_not_report_the_first_verdicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The strategy resets all three channels at the start of each render."""
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
    # The next render starts empty before recording its own verdicts.
    renderer.last_render_reconstructions = [
        ReconstructionRecord(block_id="tb2", verified=False, corrupt=True, reason="broken"),
    ]
    asyncio.run(_one())
    assert [r.block_id for r in adapter.last_render_reconstructions] == ["tb2"]
