"""Regression guards for the 2026-09-21 review round.

Each test reproduces a defect listed in the 2026-09-21 code review (plus the
security/correctness items it inherits from the 2026-09-19 review); both review
docs were later removed — recover them with
``git show 62fcd75^:docs/CODE_REVIEW_2026-09-21.md`` and
``git show 62fcd75^:docs/CODE_REVIEW_2026-09-19.md``.
Each guard fails if its fix is reverted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.stage_ctx_factory import build_stage_ctx
from ubt.core.config import UBTConfig
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.stages import advisory as advisory_stage
from ubt.core.ir.models import (
    BlockStatus,
    BlockType,
    BookManifest,
    BoundingBox,
    ChapterIR,
    ChapterMeta,
    IRBlock,
)
from ubt.core.memory.tm import TMPendingEntry, TranslationMemory
from ubt.core.policy.bilingual_advisor import Advisory, ModeScore

SRC = "The quick brown fox jumps over the lazy dog near the river bank."
ZH = "那只敏捷的棕色狐狸跃过河边懒狗。"


def _manifest(doc_id: str = "doc_review") -> BookManifest:
    return BookManifest(
        doc_id=doc_id,
        title="t",
        source_path="book.epub",
        chapters=[ChapterMeta(chapter_id="c1", title="c1", spine_index=1, source_file="c1.xhtml")],
    )


def _block(
    block_id: str,
    *,
    source: str = SRC,
    target: str | None = ZH,
    draft: str | None = None,
    status: BlockStatus = BlockStatus.MTQE_PASSED,
) -> IRBlock:
    return IRBlock(
        id=block_id,
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        source_text=source,
        target_text=target,
        draft_text=draft,
        status=status,
    )


def _seed_blocks(ledger: SQLiteJobLedger, job_id: str, *blocks: IRBlock) -> None:
    """Put blocks in the ledger with their source text (checkpoints cannot)."""
    ledger.append_chapter(
        job_id,
        ChapterIR(
            doc_id=job_id,
            chapter_id="c1",
            title="c1",
            spine_index=1,
            blocks=list(blocks),
        ),
    )


# ---------------------------------------------------------------------------
# P1-5: the advisory stages must not score the ingest-era block snapshot
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mode_advisory_forces_a_fresh_block_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P1-5: repair/consistency/triage mutate blocks before this stage reads them.

    The stage used to call ``ctx.current_blocks()`` with the default cache, so it
    scored the post-ingest snapshot. It must pass ``force_refresh=True``, and the
    proof is that the ledger's current rows — not the stale empty snapshot primed
    below — reach ``advise_layout``.
    """
    ctx = build_stage_ctx(tmp_path, job_id="job_advisory")
    ctx.ledger.init_job_from_manifest("job_advisory", _manifest("job_advisory"))
    _seed_blocks(ctx.ledger, "job_advisory", _block("b1"))

    seen: list[list[IRBlock]] = []

    def _record(blocks: list[IRBlock], requested: str = "inline", **_kwargs: Any) -> Advisory:
        seen.append(list(blocks))
        return Advisory(
            requested=requested,  # type: ignore[arg-type]
            tier="ok",
            ranking=(ModeScore(mode=requested, score=1.0),),  # type: ignore[arg-type]
            reasons=(),
        )

    monkeypatch.setattr(advisory_stage, "advise_layout", _record)

    refresh_flags: list[bool] = []
    real_current_blocks = ctx.current_blocks

    async def _spy(force_refresh: bool = False) -> list[IRBlock]:
        refresh_flags.append(force_refresh)
        return await real_current_blocks(force_refresh=force_refresh)

    ctx.current_blocks = _spy  # type: ignore[method-assign]
    # Stale prime tagged with an obsolete revision: the fix must ignore it.
    ctx._blocks = ([], ctx.ledger.blocks_seq - 1)

    events = [event async for event in advisory_stage.run_mode_advisory_stage(ctx)]

    assert refresh_flags == [True], "the advise read must force a refresh"
    assert events, "the stage still emits its MODE_ADVISED event"
    assert seen and {b.id for b in seen[0]} == {"b1"}, "the ledger's current rows must be used"


@pytest.mark.asyncio
async def test_extraction_witness_forces_a_fresh_block_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P1-5, witness half: this stage fills the cache and mutates flags on disk.

    Reading through the default cache handed the *next* stage the pre-witness
    snapshot, so the flags this stage just wrote were invisible to it.
    """
    ctx = build_stage_ctx(tmp_path, job_id="job_witness")
    ctx.ledger.init_job_from_manifest("job_witness", _manifest("job_witness"))
    _seed_blocks(ctx.ledger, "job_witness", _block("b1"))

    monkeypatch.setattr(advisory_stage, "inspect_font_encoding_damage", lambda _p: {})
    monkeypatch.setattr(
        advisory_stage,
        "summarize_font_encoding_damage",
        lambda _v: {"pages": 0, "confirmed_pages": 0, "residue_chars": 0},
    )
    seen: list[list[IRBlock]] = []

    def _flag(blocks: list[IRBlock], _verdicts: Any) -> list[IRBlock]:
        seen.append(list(blocks))
        return []

    monkeypatch.setattr(advisory_stage, "flag_font_encoding_damage", _flag)
    ctx.input_path = tmp_path / "book.pdf"  # the witness only runs for PDFs
    ctx.source_pdf_path  # noqa: B018 - documents that the guard reads input_path
    ctx._blocks = ([], ctx.ledger.blocks_seq - 1)  # stale prime

    await advisory_stage.run_extraction_witness_stage(ctx)

    assert seen and {b.id for b in seen[0]} == {"b1"}


# ---------------------------------------------------------------------------
# P1-6: withdrawn by review 2026-09-21 §6.1 — this test pins what the
# measurement showed, so a future "fix" can't quietly re-litigate it.
# ---------------------------------------------------------------------------


def test_tm_pool_revalidates_per_pair_not_whole_cache(tmp_path: Path) -> None:
    """Per-pair generations replaced the whole-cache clear, because of its cost.

    This test pinned the opposite choice (review 2026-09-21 §6.1): ``PRAGMA
    data_version`` moves on any connection's commit — including the ``use_count``
    bump both lookup paths write — and every cached pool was dropped on that
    signal. It was defended as churn because "real pools are small". Measured on
    this checkout a reload costs 0.4 ms at 500 rows, 4 ms at 5k and 17 ms at 19k,
    while reading one pair's generation costs ~1 µs; a shared ``tm.sqlite``
    crosses 5k rows after a couple of books, and every hit re-dirties the signal,
    so each fuzzy lookup was paying a rescan. What freshness still has to
    survive is asserted below: a foreign commit that adds a row to THIS pair
    reloads it, a use_count bump and a write to ANOTHER pair do not.
    """
    db = tmp_path / "tm.sqlite"
    writer = TranslationMemory(db)
    reader = TranslationMemory(db)
    try:
        writer.writeback([TMPendingEntry("en", "zh", SRC, ZH, "machine")])
        cached = reader._pool("en", "zh")

        assert writer.lookup_exact("en", "zh", SRC) is not None
        assert reader._pool("en", "zh") is cached, "a use_count bump changed no source"

        writer.writeback([TMPendingEntry("fr", "zh", SRC, ZH, "machine")])
        assert reader._pool("en", "zh") is cached, "another pair must not evict this one"

        writer.writeback([TMPendingEntry("en", "zh", "Second sentence.", "第二句。")])
        reloaded = reader._pool("en", "zh")
        assert reloaded is not cached, "a row added to this pair must reach a live reader"
        assert len(reloaded[1]) == 2

        # The pool and the count are filled on different calls, so reading the
        # count must not certify the pool as fresh.
        writer.writeback([TMPendingEntry("en", "zh", "Third sentence.", "第三句。")])
        assert reader.entry_count("en", "zh") == 3
        assert len(reader._pool("en", "zh")[1]) == 3
    finally:
        writer.close()
        reader.close()


# ---------------------------------------------------------------------------
# B2: ``--dry-run`` promises zero spend; the echo provider only covers the LLM
# hop, so OCR / VLM / visual-judge egress has to be shut as well.
# ---------------------------------------------------------------------------


def test_dry_run_closes_every_page_egress_path() -> None:
    """A rehearsal must not make a real cloud call off the operator's config.

    ``ocr_mode``/``allow_page_upload``/``visual_judge_enabled`` are read from
    the adapter config, not the router, so a "zero-spend" dry run with OCR on
    still called the cloud endpoint with the operator's key and uploaded
    manuscript pages.
    """
    from ubt.core.engine.dry_run import create_dry_run_orchestrator

    config = UBTConfig(ocr_mode="vlm", allow_page_upload=True, visual_judge_enabled=True)
    orchestrator = create_dry_run_orchestrator(config)

    assert orchestrator.config.ocr_mode == "off"
    assert orchestrator.config.allow_page_upload is False
    assert orchestrator.config.visual_judge_enabled is False
    # The caller's config is untouched — the rehearsal works on a copy.
    assert config.ocr_mode == "vlm"
    assert config.allow_page_upload is True
    assert config.visual_judge_enabled is True


# ---------------------------------------------------------------------------
# B3: the page-egress gate must have one source of truth (the resolved config),
# not a second ``os.environ`` read in the parser.
# ---------------------------------------------------------------------------


def _two_page_pdf(tmp_path: Path) -> Path:
    import pypdf

    pdf = tmp_path / "two_pages.pdf"
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_blank_page(width=612, height=792)
    with pdf.open("wb") as fh:
        writer.write(fh)
    return pdf


def _page_two_block() -> IRBlock:
    """One block on page 2, so page 1 is "missing" but the book isn't empty."""
    return IRBlock(
        id="b2",
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        source_text=SRC,
        bbox=BoundingBox(page=2, x0=0.0, y0=0.0, x1=1.0, y1=1.0),
    )


def _capture_probe(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    from ubt.adapters.pdf.vlm import registry

    seen: dict[str, object] = {}

    def _probe(
        mode: str = "auto",
        endpoint: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        allow_page_upload: bool = False,
    ) -> tuple[None, None]:
        seen["allow_page_upload"] = allow_page_upload
        seen["model"] = model
        return (None, None)

    monkeypatch.setattr(registry, "probe_effective_driver", _probe)
    return seen


def test_page_upload_gate_prefers_the_resolved_config_over_the_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ubt.adapters.pdf import docling_parser

    pdf = _two_page_pdf(tmp_path)
    seen = _capture_probe(monkeypatch)
    monkeypatch.setenv("UBT_VLM_SCAN_FALLBACK", "missing")
    monkeypatch.setenv("UBT_ALLOW_PAGE_UPLOAD", "true")

    docling_parser.vlm_fallback_missing_pages(
        pdf, [_page_two_block()], ocr_mode="vlm", allow_page_upload=False
    )

    assert seen["allow_page_upload"] is False, "env must not override the resolved config"
    assert seen["model"] is None, "no model configured must not invent one"


def test_page_upload_gate_falls_back_to_env_without_a_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``None`` is the direct-caller path (tests, library use): keep the env read."""
    from ubt.adapters.pdf import docling_parser

    pdf = _two_page_pdf(tmp_path)
    seen = _capture_probe(monkeypatch)
    monkeypatch.setenv("UBT_VLM_SCAN_FALLBACK", "missing")
    monkeypatch.setenv("UBT_ALLOW_PAGE_UPLOAD", "true")

    docling_parser.vlm_fallback_missing_pages(pdf, [_page_two_block()], ocr_mode="vlm")

    assert seen["allow_page_upload"] is True
