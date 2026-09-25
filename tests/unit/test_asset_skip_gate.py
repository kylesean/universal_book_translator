"""Reflow asset-inclusion gate: a dropped content figure must block delivery.

A reflow render that fails to stage a content figure drops it from the
delivered PDF while still reporting full render coverage -- the only other
signal is a ``//`` comment in the ``.typ`` source. These tests pin the
classifier (content figure -> major, decorative/cover -> info) and the wiring
that injects its findings into the visual gate, and prove the rigid route is
NOT affected (its skip reasons are text-placement, not asset loss).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pytest

from ubt.adapters.pdf.artifact_parity import asset_skip_findings
from ubt.adapters.pdf.visual_gate import blocking_gate_tripped
from ubt.core.engine.reflow_loop import ReflowControlLoop
from ubt.core.ir.models import BlockType, BookManifest, FlowID, IRBlock
from ubt.core.ports import reset_ports, set_visual_gate_runner

pytestmark = pytest.mark.fast


def _block(bid: str, block_type: BlockType) -> IRBlock:
    return IRBlock(
        id=bid, flow_id=FlowID.MAIN_STORY, spine_index=1, block_type=block_type, source_text=""
    )


# ---------------------------------------------------------------------------
# Pure classifier
# ---------------------------------------------------------------------------


def test_content_figure_skip_is_major() -> None:
    findings = asset_skip_findings([_block("img1", BlockType.IMAGE)], [("img1", "missing_asset")])
    assert len(findings) == 1
    assert findings[0].severity == "major"
    assert findings[0].code == "content_asset_missing"


def test_decorative_and_cover_skips_are_info() -> None:
    blocks = [_block("dec1", BlockType.IMAGE), _block("cov1", BlockType.IMAGE)]
    findings = asset_skip_findings(
        blocks, [("dec1", "decorative_banner"), ("cov1", "cover_asset_missing")]
    )
    assert [f.severity for f in findings] == ["info", "info"]
    assert {f.code for f in findings} == {"asset_skip_decorative"}


def test_content_asset_missing_fails_closed_even_when_gate_disabled() -> None:
    findings = asset_skip_findings([_block("img1", BlockType.IMAGE)], [("img1", "missing_asset")])
    tripped = blocking_gate_tripped(cast(Any, findings), total_pages=100, enabled=False)
    assert [f.code for f in tripped] == ["content_asset_missing"]


# ---------------------------------------------------------------------------
# Wiring: ReflowControlLoop injects the findings, but only on the reflow route
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

    def __init__(self, skips: list[tuple[str, str]]) -> None:
        self.last_render_skips = skips


class _Ledger:
    def save_checkpoints_batch(self, checkpoints: list[dict[str, Any]]) -> None: ...

    def record_visual_report(self, job_id: str, report: dict[str, Any]) -> None: ...


@pytest.fixture(autouse=True)
def _cleanup_ports() -> Any:
    reset_ports()
    yield
    reset_ports()


async def _run_loop(
    tmp_path: Path, *, engine: str, skips: list[tuple[str, str]], blocks: list[IRBlock]
) -> Any:
    pdf_path = tmp_path / "out.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 mock")

    async def mock_gate_runner(*args: Any, **kwargs: Any) -> _GateResult:
        return _GateResult(passed=True, findings=())

    set_visual_gate_runner(mock_gate_runner)
    # A nonexistent source skips the (subprocess-backed) artifact-parity probe so
    # this test isolates the asset-skip gate.
    manifest = BookManifest(doc_id="doc1", title="Book", source_path=str(tmp_path / "none.pdf"))
    manifest.run.render_engine_effective = engine
    loop = ReflowControlLoop(
        adapter=_Adapter(skips),
        manifest=manifest,
        ledger=cast(Any, _Ledger()),
        job_id="job1",
        target_lang="zh",
    )
    _, _, gate = await loop.run(pdf_path, blocks)
    return gate


@pytest.mark.asyncio
async def test_reflow_content_figure_skip_blocks(tmp_path: Path) -> None:
    blocks = [_block("img1", BlockType.IMAGE)]
    gate = await _run_loop(
        tmp_path, engine="publication", skips=[("img1", "missing_asset")], blocks=blocks
    )
    codes = {getattr(f, "code", "") for f in gate.findings}
    assert "content_asset_missing" in codes
    assert gate.passed is False


@pytest.mark.asyncio
async def test_rigid_text_skip_is_not_an_asset_defect(tmp_path: Path) -> None:
    # Rigid skips non-prose blocks on purpose (the source canvas keeps them);
    # that must never be read as a lost asset.
    blocks = [_block("p1", BlockType.NARRATIVE)]
    gate = await _run_loop(tmp_path, engine="rigid", skips=[("p1", "non_prose")], blocks=blocks)
    codes = {getattr(f, "code", "") for f in gate.findings}
    assert "content_asset_missing" not in codes
