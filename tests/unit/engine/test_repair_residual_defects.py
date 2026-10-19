"""Track 1 (uncalibrated repair) must not promote a still-defective candidate.

``RepairLoop`` has two tracks. The *calibrated* track re-scores with a neural
runner and re-runs the numeric/glossary validators before it will call a block
REPAIRED. The *uncalibrated* track (heuristic runner) decides on structural
validity alone: a tier-1 in-place repair only has to pass ``_structural_check``
(the 0-Token invariants — no HTML drift, no length blow-up), which says nothing
about whether the defect that *triggered* the repair is gone.

The hole: a draft alters a source number, FastPass flags it, repair is asked to
restore it, and the model returns a splice that changes the number to a *third*
value. The splice is structurally sound and preserves all out-of-span content,
so ``t1_chosen_flags`` is ``[]`` and the old code set ``status = REPAIRED`` —
shipping a translation whose defect the repair was specifically invoked to fix,
and (because REPAIRED is terminal) never re-checking it. These tests pin the
residual-defect re-check that closes it.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from ubt.core.engine.repair_loop import RepairLoop
from ubt.core.ir.models import BlockStatus, BlockType, BoundingBox, IRBlock, make_element
from ubt.core.qe.base import BaseQERunner
from ubt.core.router.router import ModelRouter

pytestmark = pytest.mark.fast


class _UncalibratedRunner(BaseQERunner):
    """A runner whose scores are defect classes, not quality (the heuristic)."""

    async def score_pairs(self, pairs: list[dict[str, str]]) -> list[float]:
        return [0.9 for _ in pairs]

    def is_calibrated(self) -> bool:
        return False


class _FakeRouter:
    """Returns a fixed repair payload; no provider is contacted."""

    repair_reasoning_effort = "low"

    def __init__(self, reply: str) -> None:
        self._reply = reply

    async def repair(self, **_kw: Any) -> str:
        return self._reply


def _loop(reply: str) -> RepairLoop:
    return RepairLoop(router=cast(ModelRouter, _FakeRouter(reply)), qe_runner=_UncalibratedRunner())


def _block(*, source: str, draft: str, flags: list[str]) -> IRBlock:
    return IRBlock(
        element=make_element(
            id="b1",
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            source_text=source,
            bbox=BoundingBox(x0=0, y0=0, x1=100, y1=100, page=1),
        ),
        target_text=draft,
        draft_text=draft,
        status=BlockStatus.REPAIR_PENDING,
        error_flags=flags,
    )


# The draft states "43 days"; the source says 42. The annotator marks the "43"
# as span id "1", so a `<correction id="1">…</correction>` reply is spliced in
# place — the tier-1 path that never re-runs the numeric validator itself.
_SOURCE = "The trial ran for 42 days."
_DRAFT = "该试验持续了43天。"
_FLAGS = ["Numeric fidelity: mismatched number '43', expected '42'"]


@pytest.mark.asyncio
async def test_in_place_splice_to_a_wrong_number_is_not_promoted() -> None:
    """A splice that swaps one wrong number for another keeps the block REPAIR_PENDING."""
    block = _block(source=_SOURCE, draft=_DRAFT, flags=_FLAGS)
    # The model "fixes" 43 -> 41: still not the source's 42.
    loop = _loop('<correction id="1">41</correction>')

    result = await loop.repair_single_block(block)

    assert result.status is not BlockStatus.REPAIRED, (
        "a candidate that still does not state the source number must not ship as REPAIRED"
    )
    assert any("repair_structural_failure" in f for f in result.error_flags)


@pytest.mark.asyncio
async def test_in_place_splice_restoring_the_number_is_promoted() -> None:
    """The control: the same path promotes normally once the number is correct."""
    block = _block(source=_SOURCE, draft=_DRAFT, flags=_FLAGS)
    loop = _loop('<correction id="1">42</correction>')

    result = await loop.repair_single_block(block)

    assert result.status is BlockStatus.REPAIRED
    assert result.error_flags == []


@pytest.mark.asyncio
async def test_residual_check_adds_no_flag_on_a_clean_candidate() -> None:
    """A candidate the numeric validator accepts gains no spurious flag."""
    block = _block(
        source="Chapter 3 lists 12 species.",
        draft="第3章列出了11个物种。",
        flags=["Numeric fidelity: mismatched number '11', expected '12'"],
    )
    loop = _loop('<correction id="1">12</correction>')

    result = await loop.repair_single_block(block)

    assert result.error_flags == []
    assert result.status is BlockStatus.REPAIRED


# --------------------------------------------------------------------------- #
# Glossary drift is repaired deterministically, before the paid model
#
# The enforcer substitutes an untranslated source term for its canonical
# rendering at zero token cost, and the same validator that flagged the drift
# verifies the result -- the "one rule both rejects and accepts" property that
# makes a repair checkable. On the ATC'25 run all 13 glossary candidates went to
# the paid model instead, every call 500'd, and all 13 were quarantined as
# NEEDS_HUMAN while the export stage fixed the same drift deterministically.
# --------------------------------------------------------------------------- #

_GLOSSARY: list[dict[str, Any]] = [
    {"source": "framekernel", "translation": "帧内核", "aliases": [], "inflected_variants": []}
]
_DRIFT = "Glossary term violation: Glossary terms drifted: ['framekernel -> 帧内核']"


def _pending(draft: str, flags: list[str]) -> IRBlock:
    block = IRBlock(
        element=make_element(
            id="b1",
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            source_text="We propose framekernel.",
        )
    )
    block.target_text = draft
    block.draft_text = draft
    block.error_flags = list(flags)
    block.status = BlockStatus.REPAIR_PENDING
    return block


def _bare_loop() -> RepairLoop:
    """A loop carrying only the deterministic QE runner the pre-repair needs."""
    from ubt.core.qe.heuristic import HeuristicQERunner

    loop = RepairLoop.__new__(RepairLoop)
    loop.qe_runner = HeuristicQERunner()
    return loop


def _try(block: IRBlock) -> IRBlock | None:
    import asyncio

    return asyncio.run(
        _bare_loop()._try_deterministic_glossary(block, block.draft_text or "", _GLOSSARY, "zh")
    )


def test_glossary_drift_is_fixed_without_calling_the_model() -> None:
    block = _pending("我们提出了 framekernel。", [_DRIFT])
    result = _try(block)
    assert result is not None
    # The enforcer applies the CJK/Latin spacing rule, so the substituted term
    # may be padded: 我们提出了 帧内核。 — the drift is gone either way.
    assert "framekernel" not in (result.target_text or "")
    assert "帧内核" in (result.target_text or "")
    assert result.status is BlockStatus.REPAIRED
    assert result.error_flags == []


def test_a_deterministic_repair_is_re_scored_not_inherited() -> None:
    """The stored score belongs to the DEFECT, not to the repaired text.

    Leaving it produced 14 blocks reading `repaired` at score 0.25, which made
    `min_qe` look like a quality failure on the ATC'25 delivery.
    """
    block = _pending("我们提出了 framekernel。", [_DRIFT])
    block.mtqe_score = 0.25  # the score the drift earned
    result = _try(block)
    assert result is not None
    assert result.mtqe_score is not None
    assert result.mtqe_score > 0.25, "the repaired text kept the defect's score"


def test_a_drift_that_leaves_another_defect_still_reaches_the_model() -> None:
    """Only a full resolution may skip the paid repair."""
    block = _pending("我们提出了 framekernel。", [_DRIFT, "Omission suspected: verbatim residue"])
    assert _try(block) is None


def test_a_triage_verdict_is_never_cleared_by_the_enforcer() -> None:
    """`needs_human_review` is a permanent verdict, not a repairable flag."""
    block = _pending("我们提出了 framekernel。", [_DRIFT, "needs_human_review"])
    assert _try(block) is None


def test_an_already_canonical_draft_is_not_a_repair() -> None:
    block = _pending("我们提出了帧内核。", [_DRIFT])
    assert _try(block) is None


def test_the_export_stage_writes_the_canonical_terminology_marker() -> None:
    """Export-side drift must use the string the workbench groups on.

    The export stage wrote its own ``"glossary_inconsistency: …"`` while every
    QE producer wrote ``GLOSSARY_VIOLATION_MARKER`` (``"Glossary term
    violation"``). The workbench groups issues by matching that marker, so a
    term violation discovered at export reached no terminology ribbon, could not
    be filtered, and never entered the PE queue — the reader's copy disagreed
    with the ledger and nothing surfaced it.
    """
    from unittest.mock import MagicMock

    from ubt.core.qe.heuristic import GLOSSARY_VIOLATION_MARKER, glossary_violation_flag

    # Defined in validators.base; consistency only imports it, so reaching
    # it through that module is what mypy --strict flags.
    from ubt.core.validators.base import ValidationResult

    validator = MagicMock()
    validator.validate.return_value = ValidationResult.failure(
        error_code="GLOSSARY_DRIFT",
        message="Glossary terms drifted: [Transformer -> 变换器]",
        suggested_action="RETRY",
    )

    flag = glossary_violation_flag("The Transformer model", "这个模型", validator)

    assert flag is not None
    assert flag.startswith(GLOSSARY_VIOLATION_MARKER), flag
    # The workbench's own grouping rule (``ubt.api.review._ISSUE_MARKERS``).
    assert any(marker in flag for marker in ("Glossary term violation",))
    # The retired private string must not come back.
    assert "glossary_inconsistency" not in flag


def test_the_export_stage_does_not_mint_its_own_glossary_string() -> None:
    """A structural pin: the export stage must not reintroduce a private marker."""
    import inspect

    from ubt.core.engine.stages import export as export_stage

    # Check the code, not the comments: the remediation note above mentions the
    # retired string on purpose.
    source = inspect.getsource(export_stage)
    code_lines = [line for line in source.splitlines() if not line.lstrip().startswith("#")]
    code = "\n".join(code_lines)
    assert "glossary_inconsistency" not in code, (
        "export must emit the canonical marker via glossary_violation_flag"
    )
    assert "glossary_violation_flag" in code
