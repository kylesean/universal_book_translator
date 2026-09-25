"""Dedicated-MT tier admission gate: precision-first suitability filter.

STATUS: not wired into the draft stage yet. It backs the MT-tier
calibration/experiment harness (``tests/integration/test_mt_tier_live.py``,
``test_mt_vs_llm_compare.py``, ``test_qe_calibration.py``); no production
pipeline code calls :func:`is_mt_suitable`. Do not describe it as an active
stage of a TM → MT → LLM funnel until the draft stage actually invokes it.

Admission is precision-first (every condition a hard AND) because:

- a false rejection only costs one extra LLM call (cheap, safe);
- a false admission burns a wrong translation into the draft with **no**
  glossary, few-shot, or terminology machinery to protect it.
"""

from __future__ import annotations

from ubt.core.ir.models import BlockType
from ubt.core.qe.term_shape import count_sentences as count_sentences

# Only plain prose-ish blocks are admitted; CODE / FORMULA / IMAGE / TABLE
# blocks carry structure NMT models cannot preserve.
# ``count_sentences`` is imported from term_shape to share the abbreviation-aware
# sentence counter with the omission gate.
_MT_BLOCK_TYPES: frozenset[BlockType] = frozenset(
    {BlockType.NARRATIVE, BlockType.HEADING, BlockType.LIST_ITEM}
)


def is_mt_suitable(
    source_text: str,
    block_type: BlockType,
    *,
    has_terms: bool,
    has_few_shot: bool,
    has_masked_spans: bool,
    max_chars: int = 240,
) -> bool:
    """Decide whether a block may be drafted by the dedicated-MT tier.

    Args:
        source_text: Raw (unmasked) source text of the block.
        block_type: IR block type.
        has_terms: Glossary/abbreviation terms matched in the block —
            terminology consistency requires the LLM prompt machinery.
        has_few_shot: A fuzzy TM hit injected a reference translation;
            such near-misses benefit from LLM context, not NMT.
        has_masked_spans: Code or citation placeholders were masked out;
            reinsertion semantics must stay under LLM control.
        max_chars: Length ceiling for admission.
    """
    if block_type not in _MT_BLOCK_TYPES:
        return False
    if not source_text or len(source_text) > max_chars:
        return False
    if has_terms or has_few_shot or has_masked_spans:
        return False
    return not count_sentences(source_text) > 1
