from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from ubt.core.engine.events import EventType
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.ledger_reading import (
    ledger_event_factory,
    manifest_from_ledger,
    require_output_suffix,
    terminology_from_ledger,
)
from ubt.core.exceptions import UBTError, UnsupportedDocumentFormatError
from ubt.core.ir.models import BookManifest

pytestmark = pytest.mark.fast


def _seed(ledger: SQLiteJobLedger) -> None:
    ledger.init_job_from_manifest(
        "job1",
        BookManifest(
            doc_id="job1",
            title="A Book",
            source_path="/tmp/book.md",
            source_lang="en",
            target_lang="zh",
            metadata={
                "delivery_mode": "reflow",
                "delivery_plan": {"contract": "faithful"},
                "author": "Someone",
            },
        ),
    )
    ledger.set_job_metadata_value("job1", "agent_config", {"profile": "general"})
    ledger.set_job_metadata_value(
        "job1",
        "bible_cache",
        {
            "glossary_dicts": [{"source": "ALU", "target": "算术逻辑单元"}],
            "abbreviation_entries": [{"source": "CPU", "target": "CPU"}],
        },
    )


def test_manifest_carries_delivery_metadata_and_agent_config(tmp_path: Path) -> None:
    with SQLiteJobLedger(tmp_path / "job1.sqlite") as ledger:
        _seed(ledger)
        manifest = manifest_from_ledger(ledger, "job1")

    assert manifest.doc_id == "job1"
    assert manifest.source_path == "/tmp/book.md"
    assert manifest.target_lang == "zh"
    assert manifest.metadata["delivery_plan"] == {"contract": "faithful"}
    assert manifest.metadata["delivery_mode"] == "reflow"
    assert "agent_config" not in manifest.metadata
    assert manifest.metadata["_agent_config"] == {"profile": "general"}


def test_manifest_of_an_unknown_job_is_an_error(tmp_path: Path) -> None:
    with (
        SQLiteJobLedger(tmp_path / "job1.sqlite") as ledger,
        pytest.raises(UBTError, match="no such job"),
    ):
        manifest_from_ledger(ledger, "missing")


def test_terminology_reads_the_bible_cache(tmp_path: Path) -> None:
    with SQLiteJobLedger(tmp_path / "job1.sqlite") as ledger:
        _seed(ledger)
        terminology = terminology_from_ledger(ledger, "job1")

    assert [t["source"] for t in terminology.glossary_dicts] == ["ALU"]
    assert [t["source"] for t in terminology.abbreviation_entries] == ["CPU"]


def test_terminology_without_a_cache_is_empty(tmp_path: Path) -> None:
    with SQLiteJobLedger(tmp_path / "job1.sqlite") as ledger:
        _seed(ledger)
        ledger.set_job_metadata_value("job1", "bible_cache", None)
        terminology = terminology_from_ledger(ledger, "job1")

    assert terminology.glossary_dicts == []


@pytest.mark.asyncio
async def test_event_factory_reports_the_ledger_counters() -> None:
    stats = {
        "total": 10,
        "completed": 7,
        "drafted": 1,
        "repair_pending": 1,
        "failed": 1,
    }

    class _Ledger:
        def get_job_stats(self, _job_id: str) -> dict[str, Any]:
            return stats

    factory = ledger_event_factory(cast("Any", _Ledger()), "job1")
    event = await factory(EventType.EXPORT_COMPLETED, "job1", cast("Any", _Ledger()))

    assert event.total_blocks == 10
    assert event.completed_blocks == 7
    assert event.failed_blocks == 1
    assert event.repair_pending_blocks == 1


class _PdfAdapter:
    output_suffixes = frozenset({".pdf"})


def test_output_suffix_must_be_one_the_adapter_writes() -> None:
    adapter = _PdfAdapter()

    with pytest.raises(UnsupportedDocumentFormatError, match="would not be a readable md"):
        require_output_suffix(adapter, Path("book.md"), subject="writing")

    require_output_suffix(adapter, Path("book"), subject="writing")
    require_output_suffix(adapter, Path("BOOK.PDF"), subject="writing")


def test_output_suffix_check_skips_an_adapter_that_declares_none() -> None:
    require_output_suffix(SimpleNamespace(), Path("book.md"), subject="writing")
