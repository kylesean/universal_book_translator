from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from ubt.core.engine.stages.export import _terminology_and_structure_pass
from ubt.core.ir.models import BlockStatus, BlockType, FlowID, IRBlock
from ubt.core.ports import DocumentAdapter
from ubt.core.validators.consistency import GlossaryConsistencyValidator
from ubt.core.validators.glossary_enforcer import DeterministicGlossaryEnforcer
from ubt.core.validators.html_delta import HTMLDeltaValidator


@pytest.mark.fast
def test_export_stage_enforces_glossary_before_cjk_normalization() -> None:
    """Glossary enforcement must precede publishing CJK normalization.

    When an alias (e.g. '转换器') is replaced with an English term ('Transformer'),
    subsequent CJK normalization must insert spacing around the Latin term.
    If normalization runs before glossary substitution, no spacing is inserted
    and the final export text has unspaced CJK/Latin boundaries.
    """
    block = IRBlock(
        id="b1",
        spine_index=0,
        flow_id=FlowID.MAIN_STORY,
        block_type=BlockType.NARRATIVE,
        source_text="We use attention architecture.",
        target_text="我们使用 attention 架构进行训练。",
        status=BlockStatus.MTQE_PASSED,
    )
    glossary = [{"source": "attention", "translation": "注意力"}]
    enforcer = DeterministicGlossaryEnforcer(glossary=glossary)
    validator = GlossaryConsistencyValidator(glossary=glossary)
    html_val = HTMLDeltaValidator()

    _terminology_and_structure_pass(
        [block],
        glossary_enforcer=enforcer,
        glossary_validator=validator,
        html_validator=html_val,
        target_lang="zh",
    )

    # After glossary enforcement replaces 'attention' with '注意力',
    # CJK normalization must clean up the spaces between Chinese characters.
    assert block.target_text == "我们使用注意力架构进行训练。"


@pytest.mark.fast
def test_completion_floor_counts_fail_closed_render_skips_as_untranslated() -> None:
    from ubt.core.engine.stages.export import _check_completion_ratio
    from ubt.core.exceptions import IntegrityViolationError

    blocks = [
        IRBlock(
            id="a",
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            source_text="One.",
            target_text="一。",
            status=BlockStatus.MTQE_PASSED,
        ),
        IRBlock(
            id="b",
            spine_index=2,
            block_type=BlockType.NARRATIVE,
            source_text="Two.",
            target_text="二。",
            status=BlockStatus.MTQE_PASSED,
            error_flags=["render_skip:overflow"],
        ),
    ]
    # Block b has a fail-closed render skip, so only 1 of 2 is actually rendered translated (50%)
    with pytest.raises(IntegrityViolationError):
        _check_completion_ratio("job", blocks, 0.9)


@pytest.mark.fast
def test_completion_floor_ignores_intentional_preserved_skips() -> None:
    from ubt.core.engine.stages.export import _check_completion_ratio

    blocks = [
        IRBlock(
            id="a",
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            source_text="One.",
            target_text="一。",
            status=BlockStatus.MTQE_PASSED,
        ),
        IRBlock(
            id="b",
            spine_index=2,
            block_type=BlockType.NARRATIVE,
            source_text="Two.",
            target_text="二。",
            status=BlockStatus.MTQE_PASSED,
            error_flags=["render_skip:chrome"],  # intentional preserved skip
        ),
    ]
    # Preserved skip is intentional and does not count as a lost translation
    _check_completion_ratio("job", blocks, 0.9)


@pytest.mark.fast
async def test_crashed_visual_gate_reports_failure_and_leaves_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A gate that crashes must read as failed, not absent.

    With the crash swallowed and the gate left None, blocking enforcement was
    skipped entirely and the KPI collector recorded the outage as a perfect
    pass (visual_critical_rate=0.0) in metrics.json.
    """
    import json

    import ubt.core.engine.reflow_loop as reflow_module
    from tests.stage_ctx_factory import build_stage_ctx
    from ubt.core.config import UBTConfig
    from ubt.core.engine.stages.export import _run_visual_gate

    class _ExplodingLoop:
        def __init__(self, **kwargs: object) -> None:
            return

        async def run(self, **kwargs: object) -> tuple[object, object, object]:
            raise RuntimeError("reflow exploded")

    monkeypatch.setattr(reflow_module, "ReflowControlLoop", _ExplodingLoop)
    ctx = build_stage_ctx(tmp_path, config=UBTConfig(db_dir=tmp_path, visual_gate_enabled=True))
    recorded: list[dict[str, object]] = []
    monkeypatch.setattr(
        ctx.ledger,
        "record_visual_report",
        lambda job_id, report: recorded.append(report),
    )

    rendered = tmp_path / "book_bilingual.pdf"
    rendered.touch()
    _out, report_path, gate = await _run_visual_gate(ctx, cast(DocumentAdapter, None), [], rendered)

    assert gate is not None
    assert gate.passed is False
    assert any(f.code == "visual_gate_crashed" for f in gate.findings)
    assert report_path is not None and report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["findings"][0]["code"] == "visual_gate_crashed"
    assert recorded and recorded[0]["passed"] is False


@pytest.mark.fast
def test_blocking_enforcement_refuses_on_a_crashed_gate() -> None:
    from ubt.adapters.pdf.visual_gate import VisualFinding, VisualGateResult
    from ubt.core.engine.stages.export import _enforce_blocking_gate
    from ubt.core.exceptions import UBTError

    gate = VisualGateResult(
        passed=False,
        findings=(
            VisualFinding(
                severity="critical",
                code="visual_gate_crashed",
                message="visual gate did not complete: RuntimeError: boom",
            ),
        ),
        stats={"total_pages": 0},
    )
    with pytest.raises(UBTError, match="visual_gate_crashed"):
        _enforce_blocking_gate("job1", gate, None, False)


@pytest.mark.fast
def test_stale_report_sweep_also_removes_the_markdown_audit(tmp_path: Path) -> None:
    """Turning kdp_audit_markdown off for a re-run must leave no stale audit.

    The sweep knew about the JSON/metrics/visual sidecars but not the
    quality_report.md the reporter writes alongside them, so a re-run without
    the flag shipped an outdated audit document beside the fresh reports.
    """
    from ubt.core.engine.stages.export import _drop_stale_run_reports

    rendered = tmp_path / "book_bilingual.pdf"
    stale_md = tmp_path / "book_bilingual_pdf_quality_report.md"
    stale_md.write_text("# outdated audit", encoding="utf-8")

    _drop_stale_run_reports(rendered)

    assert not stale_md.exists()
