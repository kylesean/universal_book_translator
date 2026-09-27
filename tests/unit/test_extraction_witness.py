"""Tests for the extraction witness (D2').

Acceptance criteria (from the 2026-09-18 route comparison, since removed;
git history preserves the measurements):
chapter-3 (the damaged file) must hit >= 20 confirmed pages, chapter-1 (the
control) at most 2 — the witness must not be a detector that fires everywhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.stage_ctx_factory import build_stage_ctx
from ubt.adapters.pdf.extraction_witness import (
    PageVerdict,
    annotate_blocks,
    at_risk_pages,
    dirty_pages,
    inspect_pdf,
    summarize,
)
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

REPO = Path(__file__).resolve().parents[2]


def _block(bid: str, page: int | None) -> IRBlock:
    bbox = BoundingBox(page=page, x0=0, y0=0, x1=1, y1=1) if page is not None else None
    return IRBlock(id=bid, spine_index=1, source_text="x", bbox=bbox)


class TestPureLogic:
    def test_confirmed_and_risk_are_disjoint(self) -> None:
        verdicts = [
            PageVerdict(1, confirm_hits=3, risk_fonts=2, non_ascii=10),
            PageVerdict(2, confirm_hits=0, risk_fonts=1, non_ascii=4),
            PageVerdict(3, confirm_hits=0, risk_fonts=0, non_ascii=9),
            PageVerdict(4, confirm_hits=0, risk_fonts=2, non_ascii=0),
        ]
        assert dirty_pages(verdicts) == {1}
        assert at_risk_pages(verdicts) == {2}

    def test_summarize_totals(self) -> None:
        verdicts = [PageVerdict(1, 2, 1, 5), PageVerdict(2, 0, 1, 3)]
        assert summarize(verdicts) == {
            "pages": 2,
            "confirmed_pages": 1,
            "at_risk_pages": 1,
            "residue_chars": 2,
        }

    def test_annotate_marks_only_dirty_pages_once(self) -> None:
        verdicts = [PageVerdict(3, 5, 0, 12)]
        blocks = [_block("a", 3), _block("b", 4), _block("c", None)]
        flagged = annotate_blocks(blocks, verdicts)
        assert [b.id for b in flagged] == ["a"]
        assert blocks[0].error_flags == ["font_encoding_damage:page=3"]
        # Idempotent: a second pass adds no duplicate flag.
        assert annotate_blocks(blocks, verdicts) == []

    def test_annotate_skips_header_and_footer_blocks(self) -> None:
        """Header and footer blocks must not be flagged with font_encoding_damage."""
        from ubt.core.ir.models import LayoutRole

        verdicts = [PageVerdict(3, 5, 0, 12)]
        b_narrative = _block("narrative", 3)
        b_header = _block("header", 3)
        b_header.layout_role = LayoutRole.HEADER
        b_footer = _block("footer", 3)
        b_footer.layout_role = LayoutRole.FOOTER

        flagged = annotate_blocks([b_narrative, b_header, b_footer], verdicts)
        assert [b.id for b in flagged] == ["narrative"]
        assert "font_encoding_damage:page=3" not in b_header.error_flags
        assert "font_encoding_damage:page=3" not in b_footer.error_flags


class TestAcceptanceOnRealPdfs:
    @pytest.mark.skipif(
        not (REPO / "tests/fixtures/synthetic-duo-damaged.pdf").exists(), reason="fixture absent"
    )
    def test_damaged_fixture_confirms_most_of_its_damaged_pages(self) -> None:
        # synthetic-duo-damaged.pdf is generated (scripts/make_sample_corpus.py)
        # with a tampered ToUnicode map — the same damage mode the retired
        # Elsevier sample carried naturally (font claims the wrong unicode,
        # extraction bleeds ¼/ð/Þ).
        stats = summarize(inspect_pdf(REPO / "tests/fixtures/synthetic-duo-damaged.pdf"))
        assert stats["pages"] == 26
        assert stats["confirmed_pages"] >= 20
        assert stats["residue_chars"] >= 200

    @pytest.mark.skipif(
        not (REPO / "tests/fixtures/synthetic-mono.pdf").exists(), reason="fixture absent"
    )
    def test_control_file_stays_clean(self) -> None:
        stats = summarize(inspect_pdf(REPO / "tests/fixtures/synthetic-mono.pdf"))
        assert stats["confirmed_pages"] <= 2


class TestDocumentIsClosed:
    """The native ``PdfDocument`` handle must not outlive ``inspect_pdf``."""

    def _fake_pdfium(self, monkeypatch: pytest.MonkeyPatch) -> list[bool]:
        import pypdfium2

        closed: list[bool] = []

        class _FakeDoc:
            def __init__(self, _path: str) -> None:
                pass

            def __len__(self) -> int:
                return 0

            def close(self) -> None:
                closed.append(True)

        monkeypatch.setattr(pypdfium2, "PdfDocument", _FakeDoc)
        return closed

    def test_closes_on_the_happy_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import pikepdf

        closed = self._fake_pdfium(monkeypatch)
        pdf = tmp_path / "clean.pdf"
        pdf_obj = pikepdf.new()
        pdf_obj.add_blank_page(page_size=(100, 100))
        pdf_obj.save(pdf)
        pdf_obj.close()

        inspect_pdf(pdf)
        assert closed == [True]

    def test_closes_when_pikepdf_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A mid-parse failure must not leak the pdfium handle for the process life."""
        import pikepdf

        closed = self._fake_pdfium(monkeypatch)
        pdf = tmp_path / "broken.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        def _boom(*_args: object, **_kwargs: object) -> object:
            raise RuntimeError("pikepdf exploded")

        monkeypatch.setattr(pikepdf, "open", _boom)
        with pytest.raises(RuntimeError):
            inspect_pdf(pdf)
        assert closed == [True]


_r0921_SRC = "The quick brown fox jumps over the lazy dog near the river bank."

_r0921_ZH = "那只敏捷的棕色狐狸跃过河边懒狗。"


def _r0921_block(
    block_id: str,
    *,
    source: str = _r0921_SRC,
    target: str | None = _r0921_ZH,
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


def _r0921_manifest(doc_id: str = "doc_review") -> BookManifest:
    return BookManifest(
        doc_id=doc_id,
        title="t",
        source_path="book.epub",
        chapters=[ChapterMeta(chapter_id="c1", title="c1", spine_index=1, source_file="c1.xhtml")],
    )


def _r0921_seed_blocks(ledger: SQLiteJobLedger, job_id: str, *blocks: IRBlock) -> None:
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


@pytest.mark.asyncio
async def test_extraction_witness_forces_a_fresh_block_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Witness half: this stage fills the cache and mutates flags on disk.

    Reading through the default cache handed the *next* stage the pre-witness
    snapshot, so the flags this stage just wrote were invisible to it.
    """
    ctx = build_stage_ctx(tmp_path, job_id="job_witness")
    ctx.ledger.init_job_from_manifest("job_witness", _r0921_manifest("job_witness"))
    _r0921_seed_blocks(ctx.ledger, "job_witness", _r0921_block("b1"))

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
