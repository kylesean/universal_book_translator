"""Unit tests for Paper Mode vs Book Mode scenario decoupling (P1-6)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from ubt.core.archetype import Archetype, DocCategory, MathDensity
from ubt.core.config import UBTConfig
from ubt.core.engine.events import TranslationProgressEvent
from ubt.core.engine.facts import Terminology
from ubt.core.engine.stage_context import StageContext
from ubt.core.engine.stages.bible import run_bible_stage
from ubt.core.ir.models import BlockType, BoundingBox, IRBlock, make_element
from ubt.core.job_options import resolve_document_mode
from ubt.render.typst.academic import is_academic_document

pytestmark = pytest.mark.fast


def test_document_mode_validation() -> None:
    # Valid values
    assert UBTConfig(document_mode="auto").document_mode == "auto"
    assert UBTConfig(document_mode="paper").document_mode == "paper"
    assert UBTConfig(document_mode="book").document_mode == "book"
    # Case-insensitive
    assert UBTConfig(document_mode="PAPER").document_mode == "paper"
    assert UBTConfig(document_mode="Book").document_mode == "book"

    # Invalid values raise ValidationError
    with pytest.raises(ValidationError):
        UBTConfig(document_mode="invalid_mode")  # type: ignore[arg-type]


def test_resolve_document_mode_explicit() -> None:
    assert resolve_document_mode("paper", None) == "paper"
    assert resolve_document_mode("book", None) == "book"
    assert resolve_document_mode("PAPER", None) == "paper"
    assert resolve_document_mode("Book", None) == "book"


def test_resolve_document_mode_auto_inference(tmp_path: Path) -> None:
    dummy_pdf = tmp_path / "paper.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4 dummy")

    paper_arch = Archetype(
        format_ext="pdf",
        page_or_ch_count=10,
        is_scanned=False,
        math_density=MathDensity.HIGH,
        detected_domain="general",
        domain_confidence=0.8,
        category=DocCategory.ACADEMIC_PAPER,
        sample_chars=1000,
    )

    book_arch = Archetype(
        format_ext="pdf",
        page_or_ch_count=200,
        is_scanned=False,
        math_density=MathDensity.NONE,
        detected_domain="general",
        domain_confidence=0.8,
        category=DocCategory.LITERATURE,
        sample_chars=5000,
    )

    with patch("ubt.core.archetype.analyze_archetype", return_value=paper_arch):
        assert resolve_document_mode("auto", dummy_pdf) == "paper"

    with patch("ubt.core.archetype.analyze_archetype", return_value=book_arch):
        assert resolve_document_mode("auto", dummy_pdf) == "book"

    # None or missing path defaults to book
    assert resolve_document_mode("auto", None) == "book"
    assert resolve_document_mode("auto", tmp_path / "non_existent.pdf") == "book"


def test_academic_layout_respects_document_mode() -> None:
    # Explicit paper mode forces academic template
    assert is_academic_document({"document_mode": "paper"}) is True
    # Explicit book mode forces non-academic template even if profile or doc_type says paper
    assert is_academic_document({"document_mode": "book", "genre_profile": "paper"}) is False
    assert is_academic_document({"document_mode": "book", "doc_type": "academic"}) is False

    # Without document_mode, falls back to legacy heuristics
    assert is_academic_document({"genre_profile": "paper"}) is True
    assert is_academic_document({"genre_profile": "fiction"}) is False


@pytest.mark.asyncio
async def test_bible_stage_character_mining_gated_in_paper_mode(tmp_path: Path) -> None:
    class FakeLedger:
        def get_all_blocks(self, job_id: str) -> list[IRBlock]:
            return [
                IRBlock(
                    element=make_element(
                        id="b1",
                        spine_index=1,
                        block_type=BlockType.NARRATIVE,
                        source_text="Vaswani and Devlin proposed novel attention mechanisms.",
                        bbox=BoundingBox(x0=0, y0=0, x1=100, y1=100, page=1),
                    )
                )
            ]

        def get_job_metadata_value(self, job_id: str, key: str) -> Any:
            return None

        def set_job_metadata_value(self, job_id: str, key: str, value: Any) -> None:
            pass

    async def fake_create_event(*args: Any, **kwargs: Any) -> TranslationProgressEvent:
        return SimpleNamespace()  # type: ignore[return-value]

    async def mock_complete(sys: str, user: str) -> str:
        return ""

    # In paper mode: character mining is skipped
    paper_ctx = StageContext(
        config=UBTConfig(fresh=True, document_mode="paper"),
        router=cast("Any", SimpleNamespace(complete_raw=mock_complete)),
        ledger=cast("Any", FakeLedger()),
        manifest=cast(
            "Any", SimpleNamespace(doc_id="doc1", chapters=[], metadata={"document_mode": "paper"})
        ),
        job_id="paper-job",
        input_path=tmp_path / "test.pdf",
        source_lang="en",
        target_lang="zh",
        profile_name="paper",
        create_event=fake_create_event,
    )

    paper_terminology = Terminology()
    with patch("ubt.core.engine.stages.bible.mine_characters_stream") as mock_mine:
        async for _ in run_bible_stage(paper_ctx, paper_terminology):
            pass
        mock_mine.assert_not_called()

    # In book mode: character mining runs
    book_ctx = StageContext(
        config=UBTConfig(fresh=True, document_mode="book"),
        router=cast("Any", SimpleNamespace(complete_raw=mock_complete)),
        ledger=cast("Any", FakeLedger()),
        manifest=cast(
            "Any", SimpleNamespace(doc_id="doc2", chapters=[], metadata={"document_mode": "book"})
        ),
        job_id="book-job",
        input_path=tmp_path / "test.pdf",
        source_lang="en",
        target_lang="zh",
        profile_name="fiction",
        create_event=fake_create_event,
    )

    book_terminology = Terminology()
    with patch("ubt.core.engine.stages.bible.mine_characters_stream", return_value=[]) as mock_mine:
        async for _ in run_bible_stage(book_ctx, book_terminology):
            pass
        mock_mine.assert_called_once()


def test_cli_translate_help_includes_mode() -> None:
    from typer.testing import CliRunner

    from ubt.cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["translate", "--help"])
    assert result.exit_code == 0
    assert "--mode" in result.output
    assert "paper" in result.output
    assert "book" in result.output
