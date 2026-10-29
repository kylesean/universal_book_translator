"""Three defects the strict-mypy gate found in already-shipped code.

Each of these was a real crash on a documented code path, not a typing nit:
the annotation and the runtime disagreed, and the disagreement is only visible
when the path actually runs. They are pinned here so the typo cannot come back
behind a green type check.
"""

from __future__ import annotations

import pytest

from ubt.core.index.extractor import extract_index_from_blocks
from ubt.core.ir.models import IRBlock
from ubt.core.memory.bible import BibleEntry
from ubt.model.ast import Paragraph

pytestmark = pytest.mark.fast


def _block(block_id: str, text: str) -> IRBlock:
    return IRBlock(element=Paragraph(id=block_id, spine_index=0, span=None, text=text))


# --- 1. ubt index --glossary crashed on every entry ------------------------


def test_a_bible_entry_glossary_does_not_crash_the_index_extractor() -> None:
    """``BibleEntry`` has no ``.get``, and the extractor used ``item.get``.

    ``ubt index inspect --glossary terms.csv`` loads the file through
    ``load_external_glossary``, which returns ``list[BibleEntry]``. The
    extractor assumed mapping entries and raised
    ``AttributeError: 'BibleEntry' object has no attribute 'get'`` on the first
    one, so the flag could never have worked.
    """
    entries = [BibleEntry(source="machine learning", translation="机器学习")]
    blocks = [_block("b1", "We trained the machine learning model today.")]

    index = extract_index_from_blocks(blocks, glossary=entries)

    assert [e.source_term for e in index.entries] == ["machine learning"]
    assert index.entries[0].target_term == "机器学习"


def test_a_bible_entry_uses_translation_as_the_approved_rendering() -> None:
    """The field is ``translation``, not ``target``; reading ``target`` gives ""."""
    entries = [BibleEntry(source="entropy", translation="熵", kind="term")]
    blocks = [_block("b1", "Entropy always increases in a closed system.")]

    index = extract_index_from_blocks(blocks, glossary=entries)

    assert index.entries[0].target_term == "熵"


def test_mapping_glossary_entries_still_work() -> None:
    """Normalising for ``BibleEntry`` must not drop the mapping shape."""
    blocks = [_block("b1", "The transformer architecture changed everything.")]

    index = extract_index_from_blocks(
        blocks, glossary=[{"source": "transformer", "target": "变换器", "category": "arch"}]
    )

    assert [e.source_term for e in index.entries] == ["transformer"]
    assert index.entries[0].target_term == "变换器"


def test_a_glossary_entry_absent_from_the_text_is_not_indexed() -> None:
    """Pre-registration is not publication: a term still has to occur.

    ``min_occurrences`` defaults to 1, so a glossary term the book never uses
    must not appear -- otherwise a user's 500-term glossary would produce 500
    empty index entries.
    """
    entries = [BibleEntry(source="absent term", translation="缺席")]
    index = extract_index_from_blocks([_block("b1", "Nothing relevant here.")], glossary=entries)

    assert index.entries == []


# --- 2. ubt roundtrip --job called a method that does not exist ------------


def test_the_ledger_method_the_roundtrip_command_calls_exists() -> None:
    """The command called ``fetch_all_blocks``; the ledger spells it differently.

    mypy reported ``"SQLiteJobLedger" has no attribute "fetch_all_blocks"``.
    Nothing caught it at runtime because no test exercised ``roundtrip --job``,
    which is why the assertion is about the API surface rather than a call.
    """
    from ubt.core.engine.ledger import SQLiteJobLedger

    assert hasattr(SQLiteJobLedger, "get_all_blocks")
    assert not hasattr(SQLiteJobLedger, "fetch_all_blocks")


# --- 3. the interactive workbench died on its first progress event --------


def test_the_progress_event_carries_the_fields_the_workbench_needs() -> None:
    """The workbench read ``event.metadata``, which the model never defined.

    ``TranslationProgressEvent`` exposes named counters and no ``metadata``
    mapping, so ``ubt translate --interactive`` raised ``AttributeError`` on
    the first event and never rendered a frame. The projection that replaces
    that read is pinned here by its output keys, since the controller's
    contract is the flat mapping rather than the event model.
    """
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

    # Exactly the keys WorkbenchController.update_from_event consumes.
    assert meta["blocks_total"] == 10
    assert meta["blocks_done"] == 4
    assert meta["repaired"] == 1
    assert meta["cost_usd"] == pytest.approx(0.25)
    assert 0.0 <= meta["pct"] <= 100.0


def test_no_cost_key_is_invented_when_spend_is_unknown() -> None:
    """``None`` means "not reported", and must not become a fabricated 0.0.

    A panel showing 0.0 reads as "confirmed zero spend", which is a different
    statement from "no usage has been reported yet".
    """
    from ubt.cli.main import _workbench_meta
    from ubt.core.engine.events import EventType, TranslationProgressEvent

    event = TranslationProgressEvent(
        event_type=EventType.JOB_STARTED, job_id="job_1", total_blocks=10, completed_blocks=0
    )

    assert "cost_usd" not in _workbench_meta(event)


def test_the_progress_event_really_has_no_metadata_attribute() -> None:
    """Guard the guard: the field the old code read must stay absent.

    If someone adds ``metadata`` back to the model, this test should be
    *deleted deliberately* rather than silently passing for a different reason.
    """
    from ubt.core.engine.events import TranslationProgressEvent

    assert "metadata" not in TranslationProgressEvent.model_fields
