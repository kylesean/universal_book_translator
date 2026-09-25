"""Reflow quarantine must target the output page, not the source page.

The rigid overlay and the alternating zipper keep the source page as the
canvas, so an IR block's ``bbox.page`` *is* its output page. A publication
reflow rebuilds every page and re-flows blocks across them, so ``bbox.page``
(source) says nothing about where the text landed -- while the visual gate
reports findings against *output* pages. The historical code matched the two
directly and quarantined arbitrary blocks.

These tests pin the recovery: map each block to the output page its rendered
text appears on, and quarantine against that. The geometry-preserving path must
keep using ``bbox.page``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pytest

import ubt.core.ports as ports
from ubt.adapters.pdf.output_page_map import map_blocks_to_output_pages
from ubt.core.engine.reflow_loop import ReflowControlLoop
from ubt.core.ir.models import BlockStatus, BookManifest, BoundingBox, FlowID, IRBlock
from ubt.core.ports import reset_ports, set_visual_gate_runner

pytestmark = pytest.mark.fast

_LONG_A = "The quick brown fox jumps over the lazy dog repeatedly today"
_LONG_B = "A completely different sentence about quantum mechanics research"


def _block(bid: str, page: int, target: str) -> IRBlock:
    return IRBlock(
        id=bid,
        flow_id=FlowID.MAIN_STORY,
        spine_index=1,
        source_text="src",
        target_text=target,
        bbox=BoundingBox(page=page, x0=10, y0=10, x1=100, y1=100),
    )


# ---------------------------------------------------------------------------
# Pure mapping
# ---------------------------------------------------------------------------


def test_map_blocks_by_rendered_text() -> None:
    blocks = [_block("b1", 1, _LONG_A), _block("b2", 1, _LONG_B)]
    pages = ["unrelated page one text", f"prefix {_LONG_A} suffix"]
    assert map_blocks_to_output_pages(blocks, pages) == {"b1": 2}


def test_short_text_is_not_guessed() -> None:
    blocks = [_block("b1", 1, "Yes.")]
    assert map_blocks_to_output_pages(blocks, ["Yes.", "Yes."]) == {}


def test_missing_target_is_skipped() -> None:
    block = _block("b1", 1, _LONG_A)
    block.target_text = None
    assert map_blocks_to_output_pages([block], [f"x {_LONG_A}"]) == {}


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


@dataclass
class _Finding:
    severity: str
    code: str
    message: str
    page: int | None = None


@dataclass
class _GateResult:
    passed: bool
    findings: tuple[_Finding, ...] = ()
    sampled_pages: tuple[int, ...] = (1,)
    vlm_pages: tuple[int, ...] = ()
    skipped_reason: str | None = None
    stats: dict[str, Any] = field(default_factory=dict)


class _Adapter:
    engine_name = "docling"

    def __init__(self) -> None:
        self.last_render_skips: list[tuple[str, str]] = []


class _Ledger:
    def __init__(self) -> None:
        self.saved: list[list[dict[str, Any]]] = []

    def save_checkpoints_batch(self, checkpoints: list[dict[str, Any]]) -> None:
        self.saved.append(checkpoints)

    def record_visual_report(self, job_id: str, report: dict[str, Any]) -> None: ...


@pytest.fixture(autouse=True)
def _cleanup_ports() -> Any:
    reset_ports()
    yield
    reset_ports()


async def _run_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    engine: str,
    blocks: list[IRBlock],
    output_pages: list[str],
) -> tuple[Any, list[IRBlock]]:
    pdf_path = tmp_path / "out.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 mock")

    async def mock_gate_runner(*args: Any, **kwargs: Any) -> _GateResult:
        return _GateResult(
            passed=False,
            findings=(_Finding("major", "blank_candidate", "page 2 blank", page=2),),
        )

    set_visual_gate_runner(mock_gate_runner)
    # The loop resolves this through the ports module at call time, so patching
    # the module attribute stands in for the artifact's text layer.
    monkeypatch.setattr(
        ports, "extract_output_page_texts", lambda _path: output_pages, raising=False
    )

    manifest = BookManifest(doc_id="doc1", title="Book", source_path=str(tmp_path / "none.pdf"))
    manifest.run.render_engine_effective = engine
    loop = ReflowControlLoop(
        adapter=_Adapter(),
        manifest=manifest,
        ledger=cast(Any, _Ledger()),
        job_id="job1",
        target_lang="zh",
    )
    _, _, gate = await loop.run(pdf_path, blocks)
    return gate, blocks


@pytest.mark.asyncio
async def test_reflow_quarantines_by_output_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocks = [_block("b1", 1, _LONG_A), _block("b2", 1, _LONG_B)]
    # Output page 2 carries b1's text; page 1 carries b2's. Both source bboxes
    # say page 1, so a bbox-based match would quarantine the wrong block.
    pages = [f"intro {_LONG_B} tail", f"intro {_LONG_A} tail"]
    await _run_loop(tmp_path, monkeypatch, engine="publication", blocks=blocks, output_pages=pages)
    statuses = {b.id: b.status for b in blocks}
    assert statuses["b1"] == BlockStatus.NEEDS_HUMAN
    assert statuses["b2"] != BlockStatus.NEEDS_HUMAN


@pytest.mark.asyncio
async def test_geometry_preserving_still_uses_bbox_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocks = [_block("b1", 1, _LONG_A), _block("b2", 2, _LONG_B)]
    # Under rigid, bbox.page is the output page: the finding on page 2 hits b2.
    await _run_loop(tmp_path, monkeypatch, engine="rigid", blocks=blocks, output_pages=["", ""])
    statuses = {b.id: b.status for b in blocks}
    assert statuses["b2"] == BlockStatus.NEEDS_HUMAN
    assert statuses["b1"] != BlockStatus.NEEDS_HUMAN
