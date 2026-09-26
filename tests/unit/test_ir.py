"""Unit tests for Semantic Flow-Isolated IR models."""

import pytest

from ubt.core.ir.models import (
    DocumentIR,
    FlowID,
    IRBlock,
)

pytestmark = pytest.mark.fast


def test_flow_isolation_filtering() -> None:
    """Validate that DocumentIR strictly isolates different semantic flows."""
    main_block = IRBlock(
        id="ch01#p001",
        flow_id=FlowID.MAIN_STORY,
        spine_index=1,
        source_text="Main story narrative text.",
    )
    sidebar_block = IRBlock(
        id="ch01#aside001",
        flow_id=FlowID.SIDEBAR_ASIDE,
        spine_index=2,
        source_text="Key Concept: Cognitive dissonance is the mental discomfort...",
    )
    footnote_block = IRBlock(
        id="ch01#fn001",
        flow_id=FlowID.FOOTNOTE,
        spine_index=3,
        source_text="See Festinger (1957) for details.",
    )

    doc = DocumentIR(
        doc_id="test_hash_12345",
        source_path="/path/to/book.epub",
        format_type="epub",
        metadata={"title": "Psychology 101"},
        blocks=[main_block, sidebar_block, footnote_block],
    )

    assert doc.total_blocks == 3
    main_flows = doc.get_blocks_by_flow(FlowID.MAIN_STORY)
    assert len(main_flows) == 1
    assert main_flows[0].id == "ch01#p001"

    sidebars = doc.get_blocks_by_flow(FlowID.SIDEBAR_ASIDE)
    assert len(sidebars) == 1
    assert sidebars[0].id == "ch01#aside001"

    footnotes = doc.get_blocks_by_flow(FlowID.FOOTNOTE)
    assert len(footnotes) == 1
    assert footnotes[0].id == "ch01#fn001"
