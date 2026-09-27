from __future__ import annotations

import pytest

from ubt.core.engine.stages.export import _terminology_and_structure_pass
from ubt.core.ir.models import BlockStatus, BlockType, FlowID, IRBlock
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
