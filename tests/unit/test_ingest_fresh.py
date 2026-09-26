"""Unit tests for ingest resume guards: source fingerprint + --fresh re-ingest."""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from tests.stage_ctx_factory import build_stage_ctx
from ubt.core.config import UBTConfig
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.stages.ingest import run_ingest_stage
from ubt.core.exceptions import DocumentParseError, LedgerError
from ubt.core.ir.models import BookManifest, ChapterIR, IRBlock


def _manifest(doc_id: str = "jobtest", source_path: str = "/tmp/src.pdf") -> BookManifest:
    return BookManifest(doc_id=doc_id, title="t", source_path=source_path)


class _StubAdapter:
    """Minimal async parse_stream yielding canned chapters."""

    def __init__(self, texts: list[str]) -> None:
        self._texts = texts
        self.parse_calls = 0
        self.last_pages: set[int] | None = None

    async def parse_stream(self, path: Path, pages: set[int] | None = None) -> AsyncIterator[Any]:
        self.parse_calls += 1
        self.last_pages = pages
        blocks = [
            IRBlock(id=f"t#b{i:04d}", spine_index=i, source_text=text)
            for i, text in enumerate(self._texts, start=1)
        ]
        yield ChapterIR(doc_id="jobtest", chapter_id="ch1", title="c", spine_index=1, blocks=blocks)


async def _drain(coro: Any) -> None:
    async for _ in coro:
        pass


@pytest.mark.asyncio
async def test_ingest_stores_fingerprint_and_resumes(tmp_path: Path) -> None:
    src = tmp_path / "book.pdf"
    src.write_bytes(b"v1-bytes")
    ledger = SQLiteJobLedger(tmp_path / "ledger.db")
    adapter = _StubAdapter(["Hello world, this is a test paragraph."])
    manifest = _manifest(source_path=str(src))

    await _drain(
        run_ingest_stage(
            build_stage_ctx(
                tmp_path,
                adapter=adapter,
                input_path=src,
                ledger=ledger,
                job_id="job_1",
                manifest=manifest,
                source_lang="en",
            ),
        )
    )
    assert adapter.parse_calls == 1
    stored = ledger.get_job_fingerprint("job_1")
    assert stored and len(stored) == 64
    total = ledger.get_job_stats("job_1")["total"]

    # Unchanged source resumes without re-parsing.
    await _drain(
        run_ingest_stage(
            build_stage_ctx(
                tmp_path,
                adapter=adapter,
                input_path=src,
                ledger=ledger,
                job_id="job_1",
                manifest=manifest,
                source_lang="en",
            ),
        )
    )
    assert adapter.parse_calls == 1
    assert ledger.get_job_stats("job_1")["total"] == total


@pytest.mark.asyncio
async def test_ingest_changed_source_fails_fast_without_fresh(tmp_path: Path) -> None:
    src = tmp_path / "book.pdf"
    src.write_bytes(b"v1-bytes")
    ledger = SQLiteJobLedger(tmp_path / "ledger.db")
    manifest = _manifest(source_path=str(src))
    await _drain(
        run_ingest_stage(
            build_stage_ctx(
                tmp_path,
                adapter=_StubAdapter(["Hello world, this is a test paragraph."]),
                input_path=src,
                ledger=ledger,
                job_id="job_1",
                manifest=manifest,
                source_lang="en",
            ),
        )
    )
    src.write_bytes(b"v2-changed-bytes")
    with pytest.raises(DocumentParseError, match="--fresh"):
        await _drain(
            run_ingest_stage(
                build_stage_ctx(
                    tmp_path,
                    config=UBTConfig(fresh=False),
                    adapter=_StubAdapter(["other"]),
                    input_path=src,
                    ledger=ledger,
                    job_id="job_1",
                    manifest=manifest,
                    source_lang="en",
                ),
            )
        )


@pytest.mark.asyncio
async def test_ingest_fresh_clears_and_reparses(tmp_path: Path) -> None:
    src = tmp_path / "book.pdf"
    src.write_bytes(b"v1-bytes")
    ledger = SQLiteJobLedger(tmp_path / "ledger.db")
    manifest = _manifest(source_path=str(src))
    adapter = _StubAdapter(["Hello world, this is a test paragraph."])
    await _drain(
        run_ingest_stage(
            build_stage_ctx(
                tmp_path,
                adapter=adapter,
                input_path=src,
                ledger=ledger,
                job_id="job_1",
                manifest=manifest,
                source_lang="en",
            ),
        )
    )
    src.write_bytes(b"v2-changed-bytes")
    adapter2 = _StubAdapter(["Completely new text for the second edition here."])
    await _drain(
        run_ingest_stage(
            build_stage_ctx(
                tmp_path,
                config=UBTConfig(fresh=True),
                adapter=adapter2,
                input_path=src,
                ledger=ledger,
                job_id="job_1",
                manifest=manifest,
                source_lang="en",
            ),
        )
    )
    assert adapter2.parse_calls == 1
    blocks = ledger.get_all_blocks("job_1")
    assert len(blocks) == 1
    assert "second edition" in blocks[0].source_text


@pytest.mark.asyncio
async def test_ingest_forwards_selected_pages_to_adapter(tmp_path: Path) -> None:
    """A page-ranged job must hand the selection to the adapter, not only filter
    blocks after a full parse (docling then converts just that range)."""
    src = tmp_path / "book.pdf"
    src.write_bytes(b"v1-bytes")
    ledger = SQLiteJobLedger(tmp_path / "ledger.db")
    manifest = _manifest(source_path=str(src))
    adapter = _StubAdapter(["Hello world, this is a test paragraph."])
    await _drain(
        run_ingest_stage(
            build_stage_ctx(
                tmp_path,
                adapter=adapter,
                input_path=src,
                ledger=ledger,
                job_id="job_1",
                manifest=manifest,
                source_lang="en",
                config=UBTConfig(pages="2-4"),
            ),
        )
    )
    assert adapter.last_pages == {2, 3, 4}


@pytest.mark.asyncio
async def test_ingest_pages_on_non_paged_adapter_raises(tmp_path: Path) -> None:
    """Regression (P1-b): ``--pages`` against an adapter with no page geometry
    must refuse before spending, not silently translate (and bill) the whole
    document — the afterwards bbox filter keeps every non-PDF block."""
    src = tmp_path / "book.epub"
    src.write_bytes(b"v1-bytes")
    ledger = SQLiteJobLedger(tmp_path / "ledger.db")
    manifest = _manifest(source_path=str(src))

    class _NoPagesAdapter(_StubAdapter):
        supports_page_selection = False

    with pytest.raises(DocumentParseError, match="only supported for PDF"):
        await _drain(
            run_ingest_stage(
                build_stage_ctx(
                    tmp_path,
                    adapter=_NoPagesAdapter(["Hello world, this is a test paragraph."]),
                    input_path=src,
                    ledger=ledger,
                    job_id="job_1",
                    manifest=manifest,
                    source_lang="en",
                    config=UBTConfig(pages="2"),
                )
            )
        )


@pytest.mark.asyncio
async def test_resume_with_corrupt_metadata_json_aborts_and_keeps_blocks(
    tmp_path: Path,
) -> None:
    """Corrupt metadata_json must not read as "no fingerprint".

    The resume path clears every block when the fingerprint is *definitely*
    absent (crashed mid-ingest). A JSON parse failure must raise loudly
    instead of taking that branch with a translated book on the line.
    """
    src = tmp_path / "book.pdf"
    src.write_bytes(b"v1-bytes")
    ledger = SQLiteJobLedger(tmp_path / "ledger.db")
    manifest = _manifest(source_path=str(src))
    await _drain(
        run_ingest_stage(
            build_stage_ctx(
                tmp_path,
                adapter=_StubAdapter(["Hello world, this is a test paragraph."]),
                input_path=src,
                ledger=ledger,
                job_id="job_1",
                manifest=manifest,
                source_lang="en",
            ),
        )
    )
    with sqlite3.connect(ledger.db_path) as raw:
        raw.execute("UPDATE job_meta SET metadata_json = '{corrupt'")

    # The resume-time window guard rewrites metadata first; the ledger must
    # refuse to rebuild-from-{} (which would silently drop the fingerprint)
    # rather than proceed to the wipe branch.
    with pytest.raises(LedgerError, match="corrupt"):
        await _drain(
            run_ingest_stage(
                build_stage_ctx(
                    tmp_path,
                    adapter=_StubAdapter(["whatever"]),
                    input_path=src,
                    ledger=ledger,
                    job_id="job_1",
                    manifest=manifest,
                    source_lang="en",
                ),
            )
        )
    assert ledger.get_job_stats("job_1")["total"] == 1


@pytest.mark.asyncio
async def test_resume_with_non_object_metadata_raises_ledger_error(tmp_path: Path) -> None:
    """A valid-JSON-but-not-an-object metadata_json is corruption, not absence."""
    src = tmp_path / "book.pdf"
    src.write_bytes(b"v1-bytes")
    ledger = SQLiteJobLedger(tmp_path / "ledger.db")
    manifest = _manifest(source_path=str(src))
    await _drain(
        run_ingest_stage(
            build_stage_ctx(
                tmp_path,
                adapter=_StubAdapter(["Hello world, this is a test paragraph."]),
                input_path=src,
                ledger=ledger,
                job_id="job_1",
                manifest=manifest,
                source_lang="en",
            ),
        )
    )
    with sqlite3.connect(ledger.db_path) as raw:
        raw.execute("UPDATE job_meta SET metadata_json = '[]'")

    with pytest.raises(LedgerError, match="not a JSON object"):
        await _drain(
            run_ingest_stage(
                build_stage_ctx(
                    tmp_path,
                    adapter=_StubAdapter(["whatever"]),
                    input_path=src,
                    ledger=ledger,
                    job_id="job_1",
                    manifest=manifest,
                    source_lang="en",
                ),
            )
        )
    assert ledger.get_job_stats("job_1")["total"] == 1


@pytest.mark.asyncio
async def test_ingest_fresh_restarts_with_a_different_page_selection(tmp_path: Path) -> None:
    """--fresh must also lift the selected-pages resume guard (2026-09 review B3).

    The selection is part of the resume identity like the chapter window:
    changing it without --fresh refuses instead of silently re-exporting the
    old selection; with --fresh it restarts and the new selection flows
    through to the adapter.
    """
    src = tmp_path / "book.pdf"
    src.write_bytes(b"v1-bytes")
    ledger = SQLiteJobLedger(tmp_path / "ledger.db")
    manifest = _manifest(source_path=str(src))
    adapter = _StubAdapter(["Hello world, this is a test paragraph."])
    await _drain(
        run_ingest_stage(
            build_stage_ctx(
                tmp_path,
                adapter=adapter,
                input_path=src,
                ledger=ledger,
                job_id="job_1",
                manifest=manifest,
                source_lang="en",
                config=UBTConfig(pages="2,3"),
            ),
        )
    )
    # Changing the selection without --fresh refuses instead of silently
    # re-exporting the old selection.
    with pytest.raises(DocumentParseError, match="page selection"):
        await _drain(
            run_ingest_stage(
                build_stage_ctx(
                    tmp_path,
                    adapter=_StubAdapter(["other"]),
                    input_path=src,
                    ledger=ledger,
                    job_id="job_1",
                    manifest=manifest,
                    source_lang="en",
                    config=UBTConfig(pages="4"),
                ),
            )
        )
    # With --fresh the same change restarts and forwards the new selection.
    adapter2 = _StubAdapter(["Fresh pages for the restarted job."])
    await _drain(
        run_ingest_stage(
            build_stage_ctx(
                tmp_path,
                adapter=adapter2,
                input_path=src,
                ledger=ledger,
                job_id="job_1",
                manifest=manifest,
                source_lang="en",
                config=UBTConfig(fresh=True, pages="4"),
            ),
        )
    )
    assert adapter2.last_pages == {4}


@pytest.mark.asyncio
async def test_resume_after_hash_unavailable_does_not_clear_translated_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression (P0-2 / 2026-09 review): when hashing fails at ingest the
    job must record the ``fingerprint_unavailable`` sentinel, and a later
    resume whose hash *succeeds* must NOT be treated as "crashed mid-ingest"
    (which cleared every block and re-billed a fully translated book)."""
    import ubt.core.engine.stages.ingest as ingest_mod

    # Reference the real function from its source module (ingest binds it via
    # ``from ... import``), so we can restore it after the first phase without
    # reading a re-exported attribute off the ingest module.
    from ubt.core.ir.serializer import compute_file_sha256 as real_hash

    src = tmp_path / "book.pdf"
    src.write_bytes(b"v1-bytes")
    ledger = SQLiteJobLedger(tmp_path / "ledger.db")
    manifest = _manifest(source_path=str(src))

    def _boom(*_a: Any, **_k: Any) -> str:
        raise DocumentParseError("hash unavailable")

    monkeypatch.setattr(ingest_mod, "compute_file_sha256", _boom)
    await _drain(
        run_ingest_stage(
            build_stage_ctx(
                tmp_path,
                adapter=_StubAdapter(["Hello world, this is a test paragraph."]),
                input_path=src,
                ledger=ledger,
                job_id="job_1",
                manifest=manifest,
                source_lang="en",
            )
        )
    )
    assert ledger.get_job_fingerprint("job_1") == ingest_mod._FINGERPRINT_UNAVAILABLE
    assert ledger.get_job_stats("job_1")["total"] == 1

    # Resume with hashing working again: the sentinel must not be mistaken for
    # either a changed-source mismatch (raise) or a crash (clear).
    monkeypatch.setattr(ingest_mod, "compute_file_sha256", real_hash)
    await _drain(
        run_ingest_stage(
            build_stage_ctx(
                tmp_path,
                adapter=_StubAdapter(["ignored on resume"]),
                input_path=src,
                ledger=ledger,
                job_id="job_1",
                manifest=manifest,
                source_lang="en",
            )
        )
    )
    assert ledger.get_job_stats("job_1")["total"] == 1


@pytest.mark.fast
async def test_ingest_stage_raises_document_parse_error_on_zero_blocks(tmp_path: Path) -> None:
    """run_ingest_stage raises DocumentParseError when 0 blocks are parsed."""
    config = UBTConfig()
    input_file = tmp_path / "empty.txt"
    input_file.write_text("")
    manifest = BookManifest(doc_id="doc1", title="Title", source_path=str(input_file), chapters=[])
    ledger = SQLiteJobLedger(tmp_path / "test.sqlite")
    ledger.init_job_from_manifest("job_test", manifest)

    ctx = build_stage_ctx(
        tmp_path,
        job_id="job_test",
        input_path=input_file,
        config=config,
        manifest=manifest,
        ledger=ledger,
    )

    class DummyAdapter:
        async def parse_stream(self, path: Path, selected_pages: Any = None) -> Any:
            from ubt.core.ir.models import ChapterIR

            yield ChapterIR(
                doc_id="doc1", chapter_id="ch0", spine_index=0, title="Empty", blocks=[]
            )

    ctx.__dict__["require_adapter"] = lambda: DummyAdapter()

    with pytest.raises(DocumentParseError, match="0 content blocks"):
        async for _ in run_ingest_stage(ctx):
            pass
