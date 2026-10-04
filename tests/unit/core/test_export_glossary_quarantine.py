"""A quarantined (blocked-human) block is not run through the glossary.

Regression: the export terminology pass validated a blocked block's quarantine
placeholder -- which is the *source* text wrapped in a ``<mark>`` -- against the
target-language glossary, so every blocked block with a glossary term reported a
spurious ``glossary_inconsistency`` (44 of 50 warnings on a real run). A
quarantined block ships no machine output, so enforcement and validation must
both skip it, while an ordinary block is still validated.
"""

from __future__ import annotations

import pytest

from ubt.core.engine.stages.export import _terminology_and_structure_pass
from ubt.core.ir.models import BlockStatus, BlockType, BoundingBox, IRBlock, make_element
from ubt.core.validators.consistency import GlossaryConsistencyValidator
from ubt.core.validators.html_delta import HTMLDeltaValidator

pytestmark = pytest.mark.fast

_GLOSSARY = [{"source": "effect", "translation": "效应"}]
_QUARANTINE = (
    '<mark class="ubt-blocked-human" title="x">'
    "【待人工审校 | Human review required】The effect is large.</mark>"
)


def _block(*, source: str, target: str, status: BlockStatus) -> IRBlock:
    block = IRBlock(
        element=make_element(
            id="b1",
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            source_text=source,
            bbox=BoundingBox(x0=54.0, y0=700.0, x1=354.0, y1=712.0, page=1),
        )
    )
    block.target_text = target
    block.status = status
    return block


def _run(blocks: list[IRBlock]) -> list[IRBlock]:
    _terminology_and_structure_pass(
        blocks,
        glossary_enforcer=None,
        glossary_validator=GlossaryConsistencyValidator(glossary=_GLOSSARY),
        html_validator=HTMLDeltaValidator(),
        target_lang="zh",
    )
    return blocks


def test_a_quarantined_block_is_not_flagged_for_glossary_drift() -> None:
    block = _block(
        source="The effect is large.", target=_QUARANTINE, status=BlockStatus.BLOCKED_HUMAN
    )
    (result,) = _run([block])
    assert not any("glossary_inconsistency" in flag for flag in result.error_flags)


def test_an_ordinary_block_is_still_flagged_for_glossary_drift() -> None:
    block = _block(
        source="The effect is large.", target="效果很大。", status=BlockStatus.MTQE_PASSED
    )
    (result,) = _run([block])
    assert any("glossary_inconsistency" in flag for flag in result.error_flags)
