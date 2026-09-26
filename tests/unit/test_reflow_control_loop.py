"""Unit tests for ReflowControlLoop self-healing and quarantine behavior."""

import json
import logging
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.reflow_loop import ReflowControlLoop
from ubt.core.ir.models import BlockStatus, BookManifest, BoundingBox, FlowID, IRBlock
from ubt.core.log_aggregate import install_noise_aggregators, noise_aggregators
from ubt.core.ports import reset_ports, set_visual_gate_runner


@dataclass
class FakeFinding:
    severity: str
    code: str
    message: str
    page: int | None = None


@dataclass
class FakeGateResult:
    passed: bool
    findings: tuple[FakeFinding, ...] = ()
    sampled_pages: tuple[int, ...] = (1,)
    vlm_pages: tuple[int, ...] = ()
    skipped_reason: str | None = None
    stats: dict[str, Any] = field(default_factory=dict)


class FakeReconstructor:
    def __init__(self) -> None:
        self.font_size_pt = 10.5
        self.leading_em = 0.85


class FakeAdapter:
    def __init__(self) -> None:
        self.engine_name = "docling"
        self.reconstructor = FakeReconstructor()


class FakeLedger:
    def __init__(self) -> None:
        self.saved_checkpoints: list[list[dict[str, Any]]] = []
        self.recorded_reports: list[dict[str, Any]] = []

    def save_checkpoints_batch(self, checkpoints: list[dict[str, Any]]) -> None:
        self.saved_checkpoints.append(checkpoints)

    def record_visual_report(self, job_id: str, report: dict[str, Any]) -> None:
        self.recorded_reports.append(report)


@pytest.fixture(autouse=True)
def cleanup_ports() -> Generator[None, None, None]:
    reset_ports()
    yield
    reset_ports()


@pytest.mark.asyncio
async def test_reflow_control_loop_clean_pass(tmp_path: Path) -> None:
    pdf_path = tmp_path / "out.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 mock")

    async def mock_gate_runner(*args: Any, **kwargs: Any) -> FakeGateResult:
        return FakeGateResult(passed=True, findings=())

    set_visual_gate_runner(mock_gate_runner)

    adapter = FakeAdapter()
    ledger = FakeLedger()
    manifest = BookManifest(doc_id="doc1", title="Book", source_path=str(pdf_path))
    blocks = [IRBlock(id="b1", flow_id=FlowID.MAIN_STORY, spine_index=1, source_text="text")]

    loop = ReflowControlLoop(
        adapter=adapter,
        manifest=manifest,
        ledger=cast(SQLiteJobLedger, ledger),
        job_id="job1",
        target_lang="zh",
    )

    out_path, report_path, gate = await loop.run(pdf_path, blocks)
    assert gate.passed is True
    assert report_path.exists()
    assert len(ledger.recorded_reports) == 1
    assert ledger.recorded_reports[0]["passed"] is True
    assert ledger.recorded_reports[0]["self_healed"] is False


@pytest.mark.asyncio
async def test_reflow_forwards_declared_padding_pages_to_the_gate(tmp_path: Path) -> None:
    """Render-declared padding pages must reach the visual gate.

    The alternator records the pages it filled with intentional blanks on
    ``manifest.run``; unless the reflow loop threads them into the gate call the
    exemption never applies and a padded facing artifact still reports
    ``passed=False``.
    """
    pdf_path = tmp_path / "out.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 mock")

    captured: dict[str, Any] = {}

    async def mock_gate_runner(*args: Any, **kwargs: Any) -> FakeGateResult:
        captured.update(kwargs)
        return FakeGateResult(passed=True, findings=())

    set_visual_gate_runner(mock_gate_runner)

    manifest = BookManifest(doc_id="doc1", title="Book", source_path=str(pdf_path))
    manifest.run.render_padding_pages = [5, 7]
    blocks = [IRBlock(id="b1", flow_id=FlowID.MAIN_STORY, spine_index=1, source_text="text")]
    loop = ReflowControlLoop(
        adapter=FakeAdapter(),
        manifest=manifest,
        ledger=cast(SQLiteJobLedger, FakeLedger()),
        job_id="job1",
        target_lang="zh",
    )

    await loop.run(pdf_path, blocks)
    assert captured.get("padding_pages") == (5, 7)


@pytest.mark.asyncio
async def test_reflow_control_loop_typography_self_healing(tmp_path: Path) -> None:
    pdf_path = tmp_path / "out.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 mock")

    calls = 0

    async def mock_gate_runner(*args: Any, **kwargs: Any) -> FakeGateResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            return FakeGateResult(
                passed=False,
                findings=(
                    FakeFinding(severity="major", code="block_overlap", message="overlap", page=1),
                ),
            )
        return FakeGateResult(passed=True, findings=())

    set_visual_gate_runner(mock_gate_runner)

    adapter = FakeAdapter()
    ledger = FakeLedger()
    manifest = BookManifest(doc_id="doc1", title="Book", source_path=str(pdf_path))
    blocks = [
        IRBlock(
            id="b1",
            flow_id=FlowID.MAIN_STORY,
            spine_index=1,
            bbox=BoundingBox(page=1, x0=10, y0=10, x1=100, y1=100),
            source_text="text",
        )
    ]

    async def mock_render_fn(**kwargs: Any) -> Path:
        return pdf_path

    loop = ReflowControlLoop(
        adapter=adapter,
        manifest=manifest,
        ledger=cast(SQLiteJobLedger, ledger),
        job_id="job1",
        target_lang="zh",
        render_fn=mock_render_fn,
    )

    out_path, report_path, gate = await loop.run(pdf_path, blocks)
    assert gate.passed is True
    assert calls == 2
    assert ledger.recorded_reports[0]["self_healed"] is True
    assert ledger.recorded_reports[0]["healing_strategy"] == "typography_tuning"
    # Reconstructor restored
    assert adapter.reconstructor.font_size_pt == 10.5
    assert adapter.reconstructor.leading_em == 0.85


@pytest.mark.asyncio
async def test_reflow_quarantine_only_escalates_and_skips_verbatim(tmp_path: Path) -> None:
    """FAILED/BLOCKED_HUMAN verdicts are stronger and must not be downgraded
    into NEEDS_HUMAN; skip_translate passthrough text is never quarantined."""
    pdf_path = tmp_path / "out.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 mock")

    async def mock_gate_runner(*args: Any, **kwargs: Any) -> FakeGateResult:
        return FakeGateResult(
            passed=False,
            findings=(
                FakeFinding(severity="critical", code="blank_page", message="blank", page=2),
            ),
        )

    set_visual_gate_runner(mock_gate_runner)

    bbox = BoundingBox(page=2, x0=10, y0=10, x1=100, y1=100)
    blocks = [
        IRBlock(
            id="blocked",
            flow_id=FlowID.MAIN_STORY,
            spine_index=1,
            status=BlockStatus.BLOCKED_HUMAN,
            bbox=bbox,
            source_text="blocked",
        ),
        IRBlock(
            id="failed",
            flow_id=FlowID.MAIN_STORY,
            spine_index=2,
            status=BlockStatus.FAILED,
            bbox=bbox,
            source_text="failed",
        ),
        IRBlock(
            id="verbatim",
            flow_id=FlowID.MAIN_STORY,
            spine_index=3,
            status=BlockStatus.MTQE_PASSED,
            skip_translate=True,
            bbox=bbox,
            source_text="verbatim",
        ),
        IRBlock(
            id="escalate",
            flow_id=FlowID.MAIN_STORY,
            spine_index=4,
            status=BlockStatus.MTQE_PASSED,
            bbox=bbox,
            source_text="needs review",
        ),
    ]

    ledger = FakeLedger()
    manifest = BookManifest(doc_id="doc1", title="Book", source_path=str(pdf_path))
    loop = ReflowControlLoop(
        adapter=FakeAdapter(),
        manifest=manifest,
        ledger=cast(SQLiteJobLedger, ledger),
        job_id="job1",
        target_lang="zh",
    )

    await loop.run(pdf_path, blocks)

    assert blocks[0].status == BlockStatus.BLOCKED_HUMAN
    assert blocks[1].status == BlockStatus.FAILED
    assert blocks[2].status == BlockStatus.MTQE_PASSED
    assert blocks[3].status == BlockStatus.NEEDS_HUMAN
    assert len(ledger.saved_checkpoints) == 1
    assert [c["block_id"] for c in ledger.saved_checkpoints[0]] == ["escalate"]


@pytest.mark.asyncio
async def test_reflow_control_loop_quarantine_persistent_failures(tmp_path: Path) -> None:
    pdf_path = tmp_path / "out.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 mock")

    async def mock_gate_runner(*args: Any, **kwargs: Any) -> FakeGateResult:
        return FakeGateResult(
            passed=False,
            findings=(
                FakeFinding(severity="major", code="block_overlap", message="overlap", page=2),
            ),
        )

    set_visual_gate_runner(mock_gate_runner)

    adapter = FakeAdapter()
    ledger = FakeLedger()
    manifest = BookManifest(doc_id="doc1", title="Book", source_path=str(pdf_path))
    blocks = [
        IRBlock(
            id="b1",
            flow_id=FlowID.MAIN_STORY,
            spine_index=1,
            bbox=BoundingBox(page=1, x0=10, y0=10, x1=100, y1=100),
            source_text="p1 ok",
        ),
        IRBlock(
            id="b2",
            flow_id=FlowID.MAIN_STORY,
            spine_index=2,
            bbox=BoundingBox(page=2, x0=10, y0=10, x1=100, y1=100),
            source_text="p2 failing",
        ),
    ]

    async def mock_render_fn(**kwargs: Any) -> Path:
        return pdf_path

    loop = ReflowControlLoop(
        adapter=adapter,
        manifest=manifest,
        ledger=cast(SQLiteJobLedger, ledger),
        job_id="job1",
        target_lang="zh",
        render_fn=mock_render_fn,
    )

    out_path, report_path, gate = await loop.run(pdf_path, blocks)
    assert gate.passed is False
    assert blocks[0].status != BlockStatus.NEEDS_HUMAN
    assert blocks[1].status == BlockStatus.NEEDS_HUMAN
    assert any("visual_gate_failed:block_overlap" in flag for flag in blocks[1].error_flags)
    assert len(ledger.saved_checkpoints) == 1
    assert ledger.saved_checkpoints[0][0]["block_id"] == "b2"


@pytest.mark.asyncio
async def test_reflow_control_loop_no_futile_heal_for_rigid_engine(tmp_path: Path) -> None:
    """B2: the rigid engine must not be "self-healed" by a retune it ignores.

    The rigid typesetter sizes text from the source zone's median line
    height and docling_adapter returns before the Typst reconstructor, so
    mutating reconstructor.font_size_pt / leading_em re-renders the identical
    PDF (pdfium re-extract + per-page Typst compile) while claiming a healing
    attempt that cannot happen. The gate verdict and page quarantine must
    survive the skip.
    """
    pdf_path = tmp_path / "out.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 mock")

    async def mock_gate_runner(*args: Any, **kwargs: Any) -> FakeGateResult:
        return FakeGateResult(
            passed=False,
            findings=(
                FakeFinding(severity="critical", code="blank_page", message="blank", page=2),
            ),
        )

    set_visual_gate_runner(mock_gate_runner)

    render_calls = 0

    async def mock_render_fn(**kwargs: Any) -> Path:
        nonlocal render_calls
        render_calls += 1
        return pdf_path

    adapter = FakeAdapter()
    ledger = FakeLedger()
    manifest = BookManifest(
        doc_id="doc1",
        title="Book",
        source_path=str(pdf_path),
        metadata={"render_engine_effective": "rigid"},
    )
    blocks = [
        IRBlock(
            id="b2",
            flow_id=FlowID.MAIN_STORY,
            spine_index=1,
            bbox=BoundingBox(page=2, x0=10, y0=10, x1=100, y1=100),
            source_text="failing page",
        )
    ]

    loop = ReflowControlLoop(
        adapter=adapter,
        manifest=manifest,
        ledger=cast(SQLiteJobLedger, ledger),
        job_id="job1",
        target_lang="zh",
        render_fn=mock_render_fn,
    )

    out_path, report_path, gate = await loop.run(pdf_path, blocks)

    assert render_calls == 0  # no futile re-render
    assert out_path == pdf_path
    assert gate.passed is False  # the failing gate still stands
    report = ledger.recorded_reports[0]
    assert report["self_healed"] is False
    assert report["healing_strategy"] is None
    assert report["healing_skipped_reason"] == "rigid_engine_not_retunable"
    # Retune knobs untouched, and the quarantine behaviour is preserved.
    assert adapter.reconstructor.font_size_pt == 10.5
    assert adapter.reconstructor.leading_em == 0.85
    assert blocks[0].status == BlockStatus.NEEDS_HUMAN
    assert any("visual_gate_failed:blank_page" in flag for flag in blocks[0].error_flags)
    assert len(ledger.saved_checkpoints) == 1


def _fidelity_probe_spy(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    from ubt.core import ports

    calls: list[int] = []

    def _probe(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        calls.append(1)
        return {"pages_measured": 0}

    monkeypatch.setattr(ports, "render_fidelity_stats", _probe)
    return calls


async def _run_clean_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, render_fidelity_enabled: bool
) -> tuple[list[int], FakeLedger]:
    pdf_path = tmp_path / "fidelity.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 mock")

    async def mock_gate_runner(*args: Any, **kwargs: Any) -> FakeGateResult:
        return FakeGateResult(passed=True, findings=())

    set_visual_gate_runner(mock_gate_runner)
    calls = _fidelity_probe_spy(monkeypatch)
    ledger = FakeLedger()
    manifest = BookManifest(
        doc_id="doc1",
        title="Book",
        source_path=str(pdf_path),
        metadata={"render_engine_effective": "rigid"},
    )
    loop = ReflowControlLoop(
        adapter=FakeAdapter(),
        manifest=manifest,
        ledger=cast(SQLiteJobLedger, ledger),
        job_id="job1",
        target_lang="zh",
        render_fidelity_enabled=render_fidelity_enabled,
    )
    await loop.run(pdf_path, [])
    return calls, ledger


@pytest.mark.asyncio
async def test_render_fidelity_probe_is_off_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two rasterizations for an `info`-only ruler nobody consumes: opt-in."""
    calls, _ = await _run_clean_loop(tmp_path, monkeypatch, render_fidelity_enabled=False)
    assert calls == []


@pytest.mark.asyncio
async def test_render_fidelity_probe_runs_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls, _ = await _run_clean_loop(tmp_path, monkeypatch, render_fidelity_enabled=True)
    assert calls == [1]


@pytest.mark.asyncio
async def test_visual_report_persists_aggregated_table_structure_noise(tmp_path: Path) -> None:
    """E2E: parse-time Docling orphan warnings must survive into the artifact.

    The log aggregator trades 1492 terminal WARNING lines for a few summary
    lines, which is only honest if the exact distribution lands somewhere
    durable. Ground truth here is the ``visual_report.json`` written beside the
    deliverable (and mirrored into the SQLite ledger) — a reader comparing a
    suspect table against this file has to be able to see that most cells were
    placed by geometry, not by the model.
    """
    pdf_path = tmp_path / "out.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 mock")
    noise_aggregators().clear()
    install_noise_aggregators()

    # Reproduce the measured shape of the 92-page run: 821 high-risk and
    # 274 mid-risk cells, so the artifact must not flatten that to a total.
    log = logging.getLogger("docling_ibm_models.tableformer.data_management")
    for cid in range(821):
        log.warning(
            f"Orphan pdf_cell {cid} recovered to col={cid % 4} by nearest-column "
            f"fallback (row={cid % 40}, x=306.0, dist=241.{cid % 10})"
        )
    for cid in range(274):
        log.warning(
            f"Orphan pdf_cell {cid} recovered to col={cid % 3} by nearest-column "
            f"fallback (row={cid % 30}, x=200.0, dist=64.{cid % 10})"
        )

    async def mock_gate_runner(*args: Any, **kwargs: Any) -> FakeGateResult:
        return FakeGateResult(passed=True, findings=())

    set_visual_gate_runner(mock_gate_runner)
    ledger = FakeLedger()
    manifest = BookManifest(doc_id="doc1", title="Book", source_path=str(pdf_path))
    blocks = [IRBlock(id="b1", flow_id=FlowID.MAIN_STORY, spine_index=1, source_text="t")]

    loop = ReflowControlLoop(
        adapter=FakeAdapter(),
        manifest=manifest,
        ledger=cast(SQLiteJobLedger, ledger),
        job_id="job1",
        target_lang="zh",
    )
    _out, report_path, _gate = await loop.run(pdf_path, blocks)

    # The repeatable artifact, read back off disk rather than from the return.
    report = json.loads(report_path.read_text(encoding="utf-8"))
    noise = report["parse_noise"]["table_structure_guess"]
    assert noise["total"] == 1095
    assert noise["buckets"][">200pt"] == 821
    assert noise["buckets"]["30-100pt"] == 274
    assert noise["worst"] == 241.9
    # The ledger mirror must agree — it is the copy a job query reads.
    assert ledger.recorded_reports[0]["parse_noise"] == report["parse_noise"]
    noise_aggregators().clear()


@pytest.mark.fast
def test_cli_preflight_guardrail_warns_and_interrupts_on_forced_reflow(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When --render-engine reflow is passed on a PDF where DocumentAdvisor recommends rigid,
    the CLI must print the Pre-Flight warning panel and interrupt for confirmation when interactive."""
    from typer.testing import CliRunner

    from ubt.cli.main import app
    from ubt.core.advisor import DocumentAdvisor

    pdf_file = tmp_path / "dense_paper.pdf"
    pdf_file.write_bytes(b"%PDF-1.4\n%fake\n")

    fake_adv = MagicMock()
    fake_adv.recommended_render_engine = "rigid"
    fake_adv.math_density = "high"
    fake_adv.category = "academic_paper"
    fake_adv.check_conflict.return_value = ["Forced reflow on rigid-recommended document"]

    monkeypatch.setattr(DocumentAdvisor, "analyze", staticmethod(lambda _p: fake_adv))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    runner = CliRunner()
    # Simulate user typing '3' (Abort) at the interactive interruption prompt
    result = runner.invoke(
        app,
        [
            "translate",
            str(pdf_file),
            "--profile",
            "paper",
            "--render-engine",
            "reflow",
            "--dual-mode",
            "inline",
        ],
        input="3\n",
    )
    assert result.exit_code != 0
    assert "排版风险预警" in result.output or "Pre-Flight Layout Tradeoff" in result.output
