from __future__ import annotations

import pytest

from ubt.core.index.extractor import extract_index_from_blocks
from ubt.core.ir.models import BlockType, IRBlock, make_element
from ubt.core.memory.bible import BibleEntry

pytestmark = pytest.mark.fast


def _block(block_id: str, text: str) -> IRBlock:
    return IRBlock(
        element=make_element(
            id=block_id,
            spine_index=0,
            block_type=BlockType.NARRATIVE,
            source_text=text,
        )
    )


def test_a_bible_entry_glossary_does_not_crash_the_index_extractor() -> None:
    entries = [BibleEntry(source="machine learning", translation="机器学习")]
    blocks = [_block("b1", "We trained the machine learning model today.")]

    index = extract_index_from_blocks(blocks, glossary=entries)

    assert [e.source_term for e in index.entries] == ["machine learning"]
    assert index.entries[0].target_term == "机器学习"


def test_a_bible_entry_uses_translation_as_the_approved_rendering() -> None:
    entries = [BibleEntry(source="entropy", translation="熵", kind="term")]
    blocks = [_block("b1", "Entropy always increases in a closed system.")]

    index = extract_index_from_blocks(blocks, glossary=entries)

    assert index.entries[0].target_term == "熵"


def test_mapping_glossary_entries_still_work() -> None:
    blocks = [_block("b1", "The transformer architecture changed everything.")]

    index = extract_index_from_blocks(
        blocks, glossary=[{"source": "transformer", "target": "变换器", "category": "arch"}]
    )

    assert [e.source_term for e in index.entries] == ["transformer"]
    assert index.entries[0].target_term == "变换器"


def test_a_glossary_entry_absent_from_the_text_is_not_indexed() -> None:
    entries = [BibleEntry(source="absent term", translation="缺席")]
    index = extract_index_from_blocks([_block("b1", "Nothing relevant here.")], glossary=entries)

    assert index.entries == []


def test_the_ledger_method_the_roundtrip_command_calls_exists() -> None:
    from ubt.core.engine.ledger import SQLiteJobLedger

    assert hasattr(SQLiteJobLedger, "get_all_blocks")
    assert not hasattr(SQLiteJobLedger, "fetch_all_blocks")


def test_the_progress_event_carries_the_fields_the_workbench_needs() -> None:
    from ubt.cli.main import _workbench_meta
    from ubt.core.engine.events import EventType, TranslationProgressEvent

    event = TranslationProgressEvent(
        event_type=EventType.DRAFT_BATCH_COMPLETED,
        job_id="job_1",
        total_blocks=10,
        completed_blocks=4,
        repaired_blocks=1,
        estimated_cost_usd=0.25,
    )
    meta = _workbench_meta(event)

    assert meta["blocks_total"] == 10
    assert meta["blocks_done"] == 4
    assert meta["repaired"] == 1
    assert meta["cost_usd"] == pytest.approx(0.25)
    assert 0.0 <= meta["pct"] <= 100.0


def test_no_cost_key_is_invented_when_spend_is_unknown() -> None:
    from ubt.cli.main import _workbench_meta
    from ubt.core.engine.events import EventType, TranslationProgressEvent

    event = TranslationProgressEvent(
        event_type=EventType.JOB_STARTED, job_id="job_1", total_blocks=10, completed_blocks=0
    )

    assert "cost_usd" not in _workbench_meta(event)


def test_the_progress_event_really_has_no_metadata_attribute() -> None:
    from ubt.core.engine.events import TranslationProgressEvent

    assert "metadata" not in TranslationProgressEvent.model_fields
