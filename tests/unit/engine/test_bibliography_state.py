"""The bibliography section state machine must close on post-list body content.

``in_bibliography`` opens at a References-like heading and used to close only
on another heading. Body content that follows the entries before any heading —
a table caption between the reference list and ``Appendix A`` (arXiv
2609.32391 b0116) — inherited ``in_bibliography=True`` and shipped
untranslated. A narrative block the per-block rules find translatable now
closes the section; blocks that classify as entries, debris or bylines keep
it open.
"""

from __future__ import annotations

import pytest

from ubt.core.engine.stages.ingest import update_bibliography_section_state
from ubt.core.ir.models import BlockType, FlowID, IRBlock, make_element
from ubt.model.ast import RegionKind

pytestmark = pytest.mark.fast


def _block(bid: str, block_type: BlockType, text: str) -> IRBlock:
    return IRBlock(
        element=make_element(
            id=bid,
            spine_index=0,
            block_type=block_type,
            flow_id=FlowID.MAIN_STORY,
            region=RegionKind.BODY,
            source_text=text,
            bbox=None,
        )
    )


def test_translatable_narrative_after_entries_closes_section() -> None:
    caption = _block(
        "b_cap",
        BlockType.NARRATIVE,
        "Table3 Requirements for evaluating and training a continual-learning agent "
        "as deployed, ordered so that each depends on those to its left.",
    )
    assert update_bibliography_section_state([caption], 0, True) is False


def test_entry_shaped_block_keeps_section_open() -> None:
    entry = _block(
        "b_entry",
        BlockType.LIST_ITEM,
        "John Yang, Carlos Jimenez. Springer, pages 45-60, 2019.",
    )
    assert update_bibliography_section_state([entry], 0, True) is True


def test_debris_keeps_section_open() -> None:
    debris = _block("b_deb", BlockType.NARRATIVE, "78")
    assert update_bibliography_section_state([debris], 0, True) is True


def test_heading_still_resets_only_when_upcoming_blocks_are_prose() -> None:
    heading = _block("b_h", BlockType.HEADING, "Appendix A")
    entry = _block("b_e", BlockType.LIST_ITEM, "J. Smith. IEEE, 245-247, 2019.")
    prose = _block("b_p", BlockType.NARRATIVE, "We present the appendix material.")
    # Heading followed by an entry-shaped block: section stays open.
    assert update_bibliography_section_state([heading, entry], 0, True) is True
    # Heading followed by real prose: closes.
    assert update_bibliography_section_state([heading, prose], 0, True) is False


def test_section_stays_closed_once_a_narrative_block_closes_it() -> None:
    caption = _block("b_cap", BlockType.NARRATIVE, "Table3 Requirements for the agent.")
    heading = _block("b_h", BlockType.HEADING, "Appendix A")
    blocks = [caption, heading]
    state = update_bibliography_section_state(blocks, 0, True)
    assert state is False
    assert update_bibliography_section_state(blocks, 1, state) is False


def test_narrative_block_surrounded_by_bib_entries_keeps_section_open() -> None:
    # A narrative entry that does not meet the standalone bib regex (like Kimi Team)
    # must not prematurely close the section if followed by more bibliography entries.
    kimi = _block(
        "b_kimi", BlockType.NARRATIVE, "Kimi Team. Kimi K2.5: Visual agentic intelligence, 2026."
    )
    next_bib = _block("b_next", BlockType.LIST_ITEM, "J. Smith. IEEE, 245-247, 2019.")
    blocks = [kimi, next_bib]
    assert update_bibliography_section_state(blocks, 0, True) is True
