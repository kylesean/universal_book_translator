"""``ReflowControlLoop``: the post-render visual gate's quarantine decisions.

The loop is the only place a rendered artifact is judged against its own
source, and its output decides whether a block ships or is held for a human.
Its end-to-end path is exercised by the ``slow`` render e2e, which drives the
happy path only -- a clean render produces no findings, so the quarantine
branch never runs there. These tests pin the decisions themselves with a fake
gate runner, so the branch that holds a defective page cannot silently rot:

- a ``major``/``critical`` finding quarantines the blocks **on that page** and
  leaves blocks on other pages alone;
- a finding with no page is not a page-level defect and quarantines nothing;
- blocks already ``FAILED``/``BLOCKED_HUMAN`` keep their terminal status (a
  quarantine must not overwrite a worse verdict), and ``skip_translate``
  blocks are never touched;
- quarantine is idempotent: a second run does not duplicate the flag;
- the report sidecar and ledger record are written on every run, and a gate
  that *crashes* is recorded as a failed gate rather than as "no gate".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from ubt.adapters.pdf.visual_gate import VisualFinding, VisualGateResult
from ubt.core.engine.reflow_loop import ReflowControlLoop
from ubt.core.ir.models import BlockStatus, BlockType, BookManifest, IRBlock, make_element

pytestmark = pytest.mark.fast


def _block(
    block_id: str,
    page: int | None,
    *,
    status: BlockStatus = BlockStatus.MTQE_PASSED,
    skip_translate: bool = False,
) -> IRBlock:
    element = make_element(
        id=block_id,
        spine_index=0,
        block_type=BlockType.NARRATIVE,
        source_text=f"source of {block_id}",
        bbox=None if page is None else _bbox(page),
        skip_translate=skip_translate,
    )
    return IRBlock(element=element, status=status, target_text=f"target of {block_id}")


def _bbox(page: int) -> Any:
    from ubt.core.ir.models import BoundingBox

    return BoundingBox(page=page, x0=10.0, y0=10.0, x1=100.0, y1=40.0)


@dataclass
class _RecordingLedger:
    """Minimal ledger: records the checkpoint batches the loop persists."""

    batches: list[list[dict[str, Any]]] = field(default_factory=list)
    visual_reports: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def save_checkpoints_batch(self, checkpoints: Sequence[dict[str, Any]]) -> None:
        self.batches.append(list(checkpoints))

    def record_visual_report(self, job_id: str, payload: dict[str, Any]) -> None:
        self.visual_reports.append((job_id, payload))

    def get_job_metadata_value(self, job_id: str, key: str) -> Any:
        return None


def _manifest(tmp_path: Path) -> BookManifest:
    return BookManifest(
        doc_id="doc",
        title="Book",
        source_path=str(tmp_path / "source.pdf"),
        chapters=[],
    )


def _manifest_with_source(tmp_path: Path) -> BookManifest:
    """A manifest whose source file exists, so the T0.5/T0.6 probes run."""
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-1.4\n%%EOF\n")
    return BookManifest(
        doc_id="doc",
        title="Book",
        source_path=str(source),
        chapters=[],
    )


def _loop(
    tmp_path: Path,
    ledger: _RecordingLedger,
    gate: VisualGateResult | BaseException,
    *,
    blocks: list[IRBlock] | None = None,
    manifest: BookManifest | None = None,
) -> ReflowControlLoop:
    loop = ReflowControlLoop(
        adapter=Any,  # type: ignore[arg-type]
        manifest=manifest or _manifest(tmp_path),
        ledger=ledger,  # type: ignore[arg-type]
        job_id="job-1",
        target_lang="zh",
    )

    async def _fake_gate(*_args: Any, **_kwargs: Any) -> VisualGateResult:
        if isinstance(gate, BaseException):
            raise gate
        return gate

    loop._evaluate_gate = _fake_gate  # type: ignore[method-assign]
    return loop


def _run(loop: ReflowControlLoop, tmp_path: Path, blocks: list[IRBlock]) -> Any:
    import asyncio

    return asyncio.run(loop.run(tmp_path / "artifact.pdf", blocks))


# --------------------------------------------------------------------------- #
# Quarantine
# --------------------------------------------------------------------------- #


def test_a_major_finding_quarantines_only_its_page(tmp_path: Path) -> None:
    ledger = _RecordingLedger()
    on_bad = _block("bad", page=3)
    on_good = _block("good", page=4)
    gate = VisualGateResult(
        passed=False,
        findings=(VisualFinding(severity="major", code="text_overflow", message="x", page=3),),
    )
    _run(_loop(tmp_path, ledger, gate), tmp_path, [on_bad, on_good])

    assert on_bad.status is BlockStatus.NEEDS_HUMAN
    assert on_bad.error_flags == ["visual_gate_failed:text_overflow"]
    assert on_good.status is BlockStatus.MTQE_PASSED
    assert on_good.error_flags == []
    assert ledger.batches == [
        [
            {
                "block_id": "bad",
                "status": BlockStatus.NEEDS_HUMAN,
                "error_flags": ["visual_gate_failed:text_overflow"],
            }
        ]
    ]


def test_every_finding_code_on_a_page_lands_in_the_flag(tmp_path: Path) -> None:
    ledger = _RecordingLedger()
    block = _block("b", page=2)
    gate = VisualGateResult(
        passed=False,
        findings=(
            VisualFinding(severity="critical", code="blank_page", message="x", page=2),
            VisualFinding(severity="major", code="text_overflow", message="y", page=2),
            VisualFinding(severity="info", code="ignored", message="z", page=2),
        ),
    )
    _run(_loop(tmp_path, ledger, gate), tmp_path, [block])

    # The info finding is not a quarantine trigger, so its code is not named.
    assert block.error_flags == ["visual_gate_failed:blank_page,text_overflow"]


def test_a_finding_without_a_page_quarantines_nothing(tmp_path: Path) -> None:
    """A document-level finding has no page to quarantine."""
    ledger = _RecordingLedger()
    block = _block("b", page=1)
    gate = VisualGateResult(
        passed=False,
        findings=(
            VisualFinding(severity="critical", code="global_defect", message="x", page=None),
        ),
    )
    _run(_loop(tmp_path, ledger, gate), tmp_path, [block])

    assert block.status is BlockStatus.MTQE_PASSED
    assert ledger.batches == []


def test_a_clean_gate_quarantines_nothing(tmp_path: Path) -> None:
    ledger = _RecordingLedger()
    block = _block("b", page=1)
    _run(
        _loop(
            tmp_path,
            ledger,
            VisualGateResult(passed=True),
        ),
        tmp_path,
        [block],
    )

    assert block.status is BlockStatus.MTQE_PASSED
    assert ledger.batches == []


def test_terminal_and_skipped_blocks_keep_their_status(tmp_path: Path) -> None:
    """A quarantine must not overwrite a worse verdict, nor touch a kept block."""
    ledger = _RecordingLedger()
    failed = _block("failed", page=1, status=BlockStatus.FAILED)
    blocked = _block("blocked", page=1, status=BlockStatus.BLOCKED_HUMAN)
    kept = _block("kept", page=1, skip_translate=True)
    translated = _block("translated", page=1)
    gate = VisualGateResult(
        passed=False,
        findings=(VisualFinding(severity="major", code="text_overflow", message="x", page=1),),
    )
    _run(_loop(tmp_path, ledger, gate), tmp_path, [failed, blocked, kept, translated])

    assert failed.status is BlockStatus.FAILED
    assert blocked.status is BlockStatus.BLOCKED_HUMAN
    assert kept.status is BlockStatus.MTQE_PASSED
    assert kept.error_flags == []
    assert [c["block_id"] for c in ledger.batches[0]] == ["translated"]


def test_a_block_without_geometry_is_never_quarantined(tmp_path: Path) -> None:
    ledger = _RecordingLedger()
    no_box = _block("no-box", page=None)
    gate = VisualGateResult(
        passed=False,
        findings=(VisualFinding(severity="major", code="text_overflow", message="x", page=1),),
    )
    _run(_loop(tmp_path, ledger, gate), tmp_path, [no_box])

    assert no_box.status is BlockStatus.MTQE_PASSED
    assert ledger.batches == []


def test_quarantine_is_idempotent_across_runs(tmp_path: Path) -> None:
    """A re-run must not append the same flag twice."""
    ledger = _RecordingLedger()
    block = _block("b", page=1)
    gate = VisualGateResult(
        passed=False,
        findings=(VisualFinding(severity="major", code="text_overflow", message="x", page=1),),
    )
    loop = _loop(tmp_path, ledger, gate)
    _run(loop, tmp_path, [block])
    _run(loop, tmp_path, [block])

    assert block.error_flags == ["visual_gate_failed:text_overflow"]


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def test_the_report_sidecar_and_ledger_record_are_always_written(tmp_path: Path) -> None:
    ledger = _RecordingLedger()
    _run(_loop(tmp_path, ledger, VisualGateResult(passed=True)), tmp_path, [])

    from ubt.core.job_options import sidecar_path

    report = sidecar_path(tmp_path / "artifact.pdf", "visual_report.json")
    assert report.exists()
    assert ledger.visual_reports[0][0] == "job-1"
    assert ledger.visual_reports[0][1]["passed"] is True
    # The loop owns this key; the gate's own half does not carry it.
    assert "parse_noise" in ledger.visual_reports[0][1]


def test_cancellation_before_the_gate_raises_and_touches_nothing(tmp_path: Path) -> None:
    """A cancelled job must not be recorded as a gate outcome."""
    import asyncio

    from ubt.core.exceptions import JobInterruptedError

    ledger = _RecordingLedger()
    token = asyncio.Event()
    token.set()
    loop = _loop(tmp_path, ledger, VisualGateResult(passed=True))
    loop.cancel_token = token

    with pytest.raises(JobInterruptedError):
        asyncio.run(loop.run(tmp_path / "artifact.pdf", []))
    assert ledger.visual_reports == []


def test_a_crashing_gate_is_recorded_as_a_failure_not_an_absence(tmp_path: Path) -> None:
    """The loop propagates the crash; the caller turns it into a failed gate.

    ``run_export_stage`` catches the exception and builds a crash result, so
    the loop's job is only to surface it rather than swallow it -- a swallowed
    crash would read downstream as "gate absent", skipping blocking
    enforcement and reporting a perfect pass.
    """
    ledger = _RecordingLedger()
    with pytest.raises(RuntimeError, match="gate exploded"):
        _run(_loop(tmp_path, ledger, RuntimeError("gate exploded")), tmp_path, [])
    assert ledger.visual_reports == []


# --------------------------------------------------------------------------- #
# Probe degradation
# --------------------------------------------------------------------------- #


def _finding_codes(gate: Any) -> list[str]:
    return [getattr(f, "code", "") for f in gate.findings]


def test_a_crashed_parity_probe_is_recorded_not_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crashed T0.5 probe must appear in the report as unmeasured.

    The probe is allowed to fail without breaking delivery, but recording
    nothing made the crash indistinguishable from a clean measurement.
    """
    import ubt.core.ports as ports

    ledger = _RecordingLedger()

    def _boom(**_kwargs: Any) -> list[Any]:
        raise RuntimeError("parity exploded")

    monkeypatch.setattr(ports, "artifact_parity_findings", _boom)
    loop = _loop(
        tmp_path,
        ledger,
        VisualGateResult(passed=True),
        manifest=_manifest_with_source(tmp_path),
    )
    _, _, gate = _run(loop, tmp_path, [])

    assert "parity_probe_crashed" in _finding_codes(gate)
    assert gate.passed is True, "an unmeasured probe is advisory, not a delivery block"


def test_a_crashed_fidelity_probe_is_recorded_not_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ubt.core.ports as ports

    ledger = _RecordingLedger()

    def _boom(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("fidelity exploded")

    monkeypatch.setattr(ports, "render_fidelity_stats", _boom)
    loop = _loop(
        tmp_path,
        ledger,
        VisualGateResult(passed=True),
        manifest=_manifest_with_source(tmp_path),
    )
    _, _, gate = _run(loop, tmp_path, [])

    assert "fidelity_probe_crashed" in _finding_codes(gate)
    assert gate.passed is True


def test_a_fidelity_probe_that_measures_nothing_records_its_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`pages_measured == 0` is 'could not measure', not 'measured and clean'."""
    import ubt.core.ports as ports

    ledger = _RecordingLedger()

    def _empty(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"pages_measured": 0, "skipped_reason": "pypdfium2_unavailable"}

    monkeypatch.setattr(ports, "render_fidelity_stats", _empty)
    loop = _loop(
        tmp_path,
        ledger,
        VisualGateResult(passed=True),
        manifest=_manifest_with_source(tmp_path),
    )
    _, _, gate = _run(loop, tmp_path, [])

    assert "fidelity_not_measured" in _finding_codes(gate)
    assert any("pypdfium2_unavailable" in getattr(f, "message", "") for f in gate.findings)
