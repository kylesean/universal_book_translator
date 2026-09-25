"""Short-document fast lane (probe + seed-only bible)."""

import sys
import types
from pathlib import Path

import pytest

from tests.pdf_builders import blank_pdf, text_pdf
from tests.stage_ctx_factory import build_stage_ctx
from ubt.adapters.pdf.short_doc import is_fast_lane_eligible, probe_pdf_pages
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.stages.bible import run_bible_stage
from ubt.core.ir.models import (
    BlockType,
    BookManifest,
    ChapterIR,
    FlowID,
    IRBlock,
)
from ubt.core.policy.layout_policy import QA_FULL_GATE_MAX_PAGES


def test_probe_counts_pages_and_text(tmp_path: Path) -> None:
    pdf = text_pdf(tmp_path / "short.pdf", 2)
    pages, chars = probe_pdf_pages(pdf)
    assert pages == 2
    assert chars >= 400


def test_probe_closes_document_when_a_page_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A per-page exception must not leak the native PdfDocument handle."""
    closed = {"doc": False, "page": False}

    class _BoomPage:
        def get_textpage(self) -> object:
            raise RuntimeError("corrupt page")

        def close(self) -> None:
            closed["page"] = True

    class _Doc:
        def __len__(self) -> int:
            return 1

        def __getitem__(self, _i: int) -> _BoomPage:
            return _BoomPage()

        def close(self) -> None:
            closed["doc"] = True

    fake = types.ModuleType("pypdfium2")
    fake.PdfDocument = lambda _path: _Doc()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pypdfium2", fake)

    probe_pdf_pages(text_pdf(tmp_path / "boom.pdf", 1))

    assert closed["doc"] is True
    assert closed["page"] is True


def test_short_digital_pdf_is_eligible(tmp_path: Path) -> None:
    assert is_fast_lane_eligible(text_pdf(tmp_path / "a.pdf", 2)) is True
    assert is_fast_lane_eligible(text_pdf(tmp_path / "b.pdf", QA_FULL_GATE_MAX_PAGES)) is True


def test_long_or_scan_pdf_not_eligible(tmp_path: Path) -> None:
    assert is_fast_lane_eligible(text_pdf(tmp_path / "c.pdf", QA_FULL_GATE_MAX_PAGES + 1)) is False
    assert is_fast_lane_eligible(blank_pdf(tmp_path / "d.pdf", 2)) is False
    assert is_fast_lane_eligible(tmp_path / "missing.pdf") is False


def _ledger_with_block(tmp_path: Path, name: str, text: str) -> tuple[SQLiteJobLedger, str]:
    ledger = SQLiteJobLedger(tmp_path / f"{name}.sqlite")
    manifest = BookManifest(doc_id="d1", title="t", source_path="s.pdf", chapters=[], metadata={})
    ledger.init_job_from_manifest("j1", manifest)
    ledger.append_chapter(
        "j1",
        ChapterIR(
            doc_id="d1",
            chapter_id="c1",
            title="c1",
            spine_index=1,
            blocks=[
                IRBlock(
                    id="b1",
                    flow_id=FlowID.MAIN_STORY,
                    spine_index=1,
                    block_type=BlockType.NARRATIVE,
                    source_text=text,
                )
            ],
        ),
    )
    return ledger, "j1"


async def test_bible_fast_lane_is_seed_subset(tmp_path: Path) -> None:
    """Fast lane = same stage, subset policy: seeds only, no mining, no LLM backfill."""
    text = "The subthreshold swing degrades. Full Expansion (FE) observed here."
    full_calls: list[str] = []
    fast_calls: list[str] = []

    async def _full_complete(*args: object, **kwargs: object) -> str:
        full_calls.append("llm")
        return ""

    async def _fast_complete(*args: object, **kwargs: object) -> str:
        fast_calls.append("llm")
        return ""

    manifest = BookManifest(doc_id="d1", title="t", source_path="s.pdf", chapters=[], metadata={})
    ledger_full, _ = _ledger_with_block(tmp_path, "full", text)
    ctx_full = build_stage_ctx(
        tmp_path,
        complete_raw_fn=_full_complete,
        ledger=ledger_full,
        job_id="j1",
        manifest=manifest,
        profile_name="semiconductor",
        target_lang="zh",
        source_lang="en",
    )
    _events = [e async for e in run_bible_stage(ctx_full)]
    count_full = ctx_full.block_count
    assert ctx_full.bible is not None
    full_sources = {e.source for e in ctx_full.bible.glossary}

    ledger_fast, _ = _ledger_with_block(tmp_path, "fast", text)
    ctx_fast = build_stage_ctx(
        tmp_path,
        complete_raw_fn=_fast_complete,
        fast_lane=True,
        ledger=ledger_fast,
        job_id="j1",
        manifest=manifest,
        profile_name="semiconductor",
        target_lang="zh",
        source_lang="en",
    )
    _events = [e async for e in run_bible_stage(ctx_fast)]
    count_fast = ctx_fast.block_count
    assert ctx_fast.bible is not None
    fast_sources = {e.source for e in ctx_fast.bible.glossary}

    # Subset: every fast-lane entry also exists in the full run.
    assert fast_sources <= full_sources
    # Mining skipped: the planted acronym never surfaces on the fast lane.
    assert "Full Expansion" not in fast_sources
    assert any("Full Expansion" in s for s in full_sources)
    # Seeds survive on both paths (all carry translations, none untranslated).
    assert "subthreshold swing" in fast_sources
    # No LLM channel on the fast lane. The full lane drives two of them through
    # the stage's single completion fn — document-skeleton terminology
    # extraction and abbreviation backfill — so counting them together was
    # brittle once skeleton extraction was added; each lane counts separately.
    assert full_calls, "full lane made no LLM call"
    assert fast_calls == [], "fast lane must not invoke the LLM backfill channel"
    assert count_full == count_fast == 1


def test_fast_lane_ports_bridge(tmp_path: Path) -> None:
    """Core reaches the probe only through ports (license-guard pattern)."""
    from ubt.core import ports

    pdf = text_pdf(tmp_path / "e.pdf", 2)
    assert ports.is_fast_lane_eligible(pdf) is True
    assert ports.is_fast_lane_eligible(blank_pdf(tmp_path / "f.pdf", 2)) is False
    ports.reset_ports()


async def test_bible_stage_yields_its_event_both_paths(tmp_path: Path) -> None:
    """The bible stage must yield BIBLE_EXTRACTED on both the fresh-mine and the
    cache-reuse path. The generator's yield is the only progress channel, so a
    consumer that iterates the pipeline must see the event during the
    multi-minute mining + backfill phase (2026-09 review L13)."""
    from ubt.core.engine.events import EventType

    text = "The subthreshold swing degrades near the drain contact."
    manifest = BookManifest(
        doc_id="d1",
        title="t",
        source_path="s.pdf",
        chapters=[],
        metadata={},
    )
    ledger, _ = _ledger_with_block(tmp_path, "pub", text)

    def _ctx() -> object:
        return build_stage_ctx(
            tmp_path,
            ledger=ledger,
            job_id="j1",
            manifest=manifest,
            profile_name="general",
            target_lang="zh",
            source_lang="en",
        )

    fresh_events = [e async for e in run_bible_stage(_ctx())]  # type: ignore[arg-type]
    assert any(getattr(e, "event_type", None) == EventType.BIBLE_EXTRACTED for e in fresh_events)

    # Second run over the same ledger hits the cache path: it must yield too,
    # not fall silent just because the expensive mining work is skipped.
    cache_events = [e async for e in run_bible_stage(_ctx())]  # type: ignore[arg-type]
    assert any(getattr(e, "event_type", None) == EventType.BIBLE_EXTRACTED for e in cache_events)
