"""Driver registry + anchoring contract (repo tests use fakes only).

The rapidocr driver needs model downloads + a GPU/CPU OCR stack, so it is
exercised by smoke scripts (/tmp), never here.
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tests.corpus_markers import requires_synthetic_mono
from tests.stage_ctx_factory import build_stage_ctx, drain
from ubt.adapters.pdf.docling_parser import vlm_fallback_missing_pages
from ubt.adapters.pdf.vlm.anchor import anchor_transcript
from ubt.adapters.pdf.vlm.registry import get_driver, list_drivers, register_driver
from ubt.adapters.pdf.vlm.transcribe import transcribe_page_to_blocks
from ubt.adapters.pdf.vlm.types import PageTranscript, VlmLine
from ubt.core.ir.models import BlockType, BoundingBox, FlowID, IRBlock


class _FakeDriver:
    name = "fake"
    measured_boxes = False

    def __init__(self, lines: list[VlmLine] | None = None) -> None:
        self._lines = lines or []

    def recognize(
        self, image: object, page_size_pt: tuple[float, float], scale: float
    ) -> PageTranscript:
        return PageTranscript(lines=tuple(self._lines), engine=self.name)


def test_registry_lists_builtin_and_rejects_unknown() -> None:
    drivers = list_drivers()
    assert "rapidocr" in drivers
    assert "deepseek-ocr" in drivers
    ds_driver = get_driver("deepseek-ocr")
    assert ds_driver.name == "deepseek-ocr"
    assert ds_driver.measured_boxes is False
    with pytest.raises(KeyError):
        get_driver("no-such-driver")


def test_deepseek_driver_is_proofread_only() -> None:
    from ubt.adapters.pdf.vlm.drivers.deepseek_driver import (
        FREE_OCR_PROMPT,
        DeepSeekOcrDriver,
    )

    drv = DeepSeekOcrDriver()
    assert drv.measured_boxes is False
    assert FREE_OCR_PROMPT == "<image>\nFree OCR."
    # Recognition mode must refuse unmeasured geometry (contract level).
    fake = PageTranscript(lines=(VlmLine(text="guess", reading_index=0),), engine="deepseek-ocr")
    with pytest.raises(ValueError):
        anchor_transcript([], fake, (540.0, 720.0))


def test_registry_override_injects_fake() -> None:
    register_driver("fake", _FakeDriver)
    assert get_driver("fake").name == "fake"
    assert get_driver(None).name in list_drivers()


def test_proofread_keeps_pdfium_geometry_and_drops_insertions() -> None:
    pdfium = [
        ("The quick brown fox", (10.0, 500.0, 200.0, 512.0)),
        ("jumps over", (10.0, 484.0, 120.0, 496.0)),
    ]
    fake = PageTranscript(
        lines=(
            VlmLine(text="The quick brown fox", reading_index=1),
            VlmLine(text="jumps over", reading_index=0),
            VlmLine(text="hallucinated header", reading_index=2),
        ),
        engine="fake",
    )
    out, stats = anchor_transcript(pdfium, fake, (540.0, 720.0))
    assert stats.matched == 2
    assert [ln.text for ln in out] == ["The quick brown fox", "jumps over"]
    assert all(ln.provenance == "pdfium+proofread" for ln in out)
    assert out[0].box == (10.0, 500.0, 200.0, 512.0)


def test_proofread_keeps_unmatched_pdfium_lines() -> None:
    pdfium = [("visible text", (10.0, 500.0, 200.0, 512.0))]
    fake = PageTranscript(lines=(), engine="fake")
    out, stats = anchor_transcript(pdfium, fake, (540.0, 720.0))
    assert len(out) == 1 and out[0].provenance == "pdfium"
    assert stats.pdfium_only == 1


def test_recognition_uses_measured_boxes() -> None:
    fake = PageTranscript(
        lines=(
            VlmLine(
                text="scanned line one", reading_index=0, measured_box=(10.0, 500.0, 300.0, 514.0)
            ),
            VlmLine(
                text="scanned line two", reading_index=1, measured_box=(10.0, 482.0, 280.0, 496.0)
            ),
        ),
        engine="fake-measured",
        measured_boxes=True,
    )
    out, stats = anchor_transcript([], fake, (540.0, 720.0))
    assert [ln.text for ln in out] == ["scanned line one", "scanned line two"]
    assert all(ln.provenance == "vlm-measured" for ln in out)
    assert out[0].box == (10.0, 500.0, 300.0, 514.0)
    assert stats.vlm_only == 2


def test_recognition_refuses_hallucinated_geometry() -> None:
    fake = PageTranscript(lines=(VlmLine(text="guess", reading_index=0),), engine="llm-vlm")
    with pytest.raises(ValueError):
        anchor_transcript([], fake, (540.0, 720.0))


def test_grouping_joins_rhythm_breaks_on_gap_and_column() -> None:
    from ubt.adapters.pdf.vlm.transcribe import group_lines_to_paragraphs

    assert group_lines_to_paragraphs([]) == []
    lines = [
        ("l1", (100.0, 500.0, 300.0, 512.0)),
        ("l2", (100.0, 484.0, 300.0, 496.0)),
        ("far", (100.0, 300.0, 300.0, 312.0)),
        ("col2", (350.0, 500.0, 500.0, 512.0)),
    ]
    groups = group_lines_to_paragraphs(lines)
    assert len(groups) == 4
    assert groups[0] == [0]
    assert groups[1] == [3]


def test_synthetic_lines_only_from_vlm_provenance() -> None:
    from ubt.adapters.pdf.vlm.transcribe import VlmEvidence, synthetic_vlm_lines
    from ubt.core.ir.models import BlockType, BoundingBox, FlowID, IRBlock

    def _block(bid: str, prov: dict[str, object]) -> IRBlock:
        return IRBlock(
            id=bid,
            flow_id=FlowID.MAIN_STORY,
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            source_text="x",
            bbox=BoundingBox(page=1, x0=0.0, y0=0.0, x1=10.0, y1=10.0),
            provenance=prov,
        )

    assert synthetic_vlm_lines([_block("a", {})]) == []
    assert synthetic_vlm_lines([]) == []
    vlm = _block(
        "v",
        {
            "vlm_lines": [
                {"text": "hi", "box": [1.0, 2.0, 3.0, 4.0]},
                {"text": "", "box": [0, 0, 1, 1]},
                {"text": "bad"},
            ]
        },
    )
    synth = synthetic_vlm_lines([vlm])
    assert len(synth) == 1
    assert isinstance(synth[0], VlmEvidence)
    assert synth[0].text == "hi" and synth[0].box == (1.0, 2.0, 3.0, 4.0)


@requires_synthetic_mono
def test_transcribe_groups_paragraphs_with_fake_driver(tmp_path: Path) -> None:
    from ubt.adapters.pdf.vlm.transcribe import transcribe_page_to_blocks

    class _FakeOcr:
        name = "fake-ocr"
        measured_boxes = True

        def recognize(
            self, image: object, page_size_pt: tuple[float, float], scale: float
        ) -> PageTranscript:
            return PageTranscript(
                lines=tuple(
                    VlmLine(
                        text=f"row {i}",
                        reading_index=i,
                        measured_box=(10.0, 500.0 - i * 14.0, 300.0, 512.0 - i * 14.0),
                    )
                    for i in range(10)
                ),
                engine=self.name,
                measured_boxes=True,
            )

    register_driver("fake-ocr", _FakeOcr)
    # Textless page: rasterize docs/synthetic-mono.pdf p1 to an image-only PDF in tmp.
    # NOTE: tests/fixtures/book3-*.pdf was removed; any real PDF works as the raster source.
    import pypdfium2 as pdfium

    src = pdfium.PdfDocument("docs/synthetic-mono.pdf")
    try:
        img = src[0].render(scale=1.5).to_pil().convert("RGB")
    finally:
        src.close()
    textless = tmp_path / "textless.pdf"
    img.save(textless, "PDF")
    blocks, stats = transcribe_page_to_blocks(textless, 1, driver_name="fake-ocr")
    assert len(blocks) == 1  # 10 rhythm-joined rows -> single paragraph
    assert blocks[0].provenance["parser"] == "vlm:fake-ocr"
    assert len(blocks[0].provenance["vlm_lines"]) == 10
    assert stats.vlm_only == 10
    assert blocks[0].provenance["anchor_stats"]["vlm_only"] == 10


def test_vlm_fallback_modes_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    from ubt.adapters.pdf.vlm.transcribe import (
        FALLBACK_ENV_VAR,
        VlmFallbackMode,
        fallback_enabled,
        get_fallback_mode,
    )

    monkeypatch.delenv(FALLBACK_ENV_VAR, raising=False)
    assert get_fallback_mode() == VlmFallbackMode.OFF
    assert not fallback_enabled()

    monkeypatch.setenv(FALLBACK_ENV_VAR, "0")
    assert get_fallback_mode() == VlmFallbackMode.OFF
    assert not fallback_enabled()

    monkeypatch.setenv(FALLBACK_ENV_VAR, "1")
    assert get_fallback_mode() == VlmFallbackMode.MISSING
    assert fallback_enabled()

    monkeypatch.setenv(FALLBACK_ENV_VAR, "missing")
    assert get_fallback_mode() == VlmFallbackMode.MISSING
    assert fallback_enabled()

    monkeypatch.setenv(FALLBACK_ENV_VAR, "weak")
    assert get_fallback_mode() == VlmFallbackMode.WEAK
    assert fallback_enabled()

    monkeypatch.setenv(FALLBACK_ENV_VAR, "all")
    assert get_fallback_mode() == VlmFallbackMode.ALL
    assert fallback_enabled()


def test_docling_adapter_tiering_dispatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import pypdfium2 as pdfium

    from ubt.adapters.pdf.docling_adapter import DoclingPDFAdapter
    from ubt.adapters.pdf.vlm.transcribe import FALLBACK_ENV_VAR
    from ubt.core.ir.models import BlockType, BoundingBox, FlowID, IRBlock

    # Create a dummy 2-page PDF
    pdf = pdfium.PdfDocument.new()
    pdf.new_page(200, 200)
    pdf.new_page(200, 200)
    pdf_path = tmp_path / "test_doc.pdf"
    pdf.save(str(pdf_path))
    pdf.close()

    blk1 = IRBlock(
        id="b1",
        flow_id=FlowID.MAIN_STORY,
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        source_text="Normal prose text on page 1.",
        bbox=BoundingBox(page=1, x0=10.0, y0=10.0, x1=100.0, y1=50.0),
        provenance={"parser": "docling"},
    )

    # 1. Mode OFF: returns untouched
    monkeypatch.setenv(FALLBACK_ENV_VAR, "off")
    res_off = DoclingPDFAdapter._vlm_fallback_missing_pages(pdf_path, [blk1])
    assert len(res_off) == 1
    assert res_off[0].id == "b1"

    # Register a fake OCR driver for transcription
    class _FakeDispatchOcr:
        name = "fake-dispatch"
        measured_boxes = True

        def recognize(
            self, image: object, page_size_pt: tuple[float, float], scale: float
        ) -> PageTranscript:
            return PageTranscript(
                lines=(
                    VlmLine(
                        text="Recovered text",
                        reading_index=0,
                        measured_box=(10.0, 10.0, 80.0, 30.0),
                    ),
                ),
                engine=self.name,
                measured_boxes=True,
            )

    register_driver("fake-dispatch", _FakeDispatchOcr)
    monkeypatch.setenv("UBT_VLM_DRIVER", "fake-dispatch")

    # 2. Mode MISSING: transcribes page 2 (which has 0 blocks)
    monkeypatch.setenv(FALLBACK_ENV_VAR, "missing")
    res_missing = DoclingPDFAdapter._vlm_fallback_missing_pages(pdf_path, [blk1])
    assert len(res_missing) == 2
    assert res_missing[0].id == "b1"
    assert res_missing[1].bbox is not None and res_missing[1].bbox.page == 2
    assert res_missing[1].provenance["parser"] == "vlm:fake-dispatch"

    # 3. Mode WEAK: page 1 has formula debris -> upgrades page 1
    blk_weak = IRBlock(
        id="b_weak",
        flow_id=FlowID.MAIN_STORY,
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        source_text="K V x y z A B C D E F G H I J",  # high formula debris
        bbox=BoundingBox(page=1, x0=10.0, y0=10.0, x1=100.0, y1=50.0),
        provenance={"parser": "docling"},
    )
    monkeypatch.setenv(FALLBACK_ENV_VAR, "weak")
    res_weak = DoclingPDFAdapter._vlm_fallback_missing_pages(pdf_path, [blk_weak])
    # Both page 1 (weak upgraded) and page 2 (missing) transcribed
    assert len(res_weak) == 2
    assert all(b.provenance["parser"] == "vlm:fake-dispatch" for b in res_weak)


@pytest.mark.asyncio
async def test_quality_gate_anchor_stats_weighting(tmp_path: Path) -> None:

    from ubt.core.engine.events import EventType, TranslationProgressEvent
    from ubt.core.engine.ledger import SQLiteJobLedger
    from ubt.core.engine.stages.quality_gate import run_quality_gate_stage
    from ubt.core.ir.models import BlockStatus, BlockType, BoundingBox, DocumentIR, FlowID, IRBlock
    from ubt.core.qe.base import BaseQERunner
    from ubt.core.qe.fast_pass import FastPassFilter

    ledger = SQLiteJobLedger(tmp_path / "ledger.db")
    job_id = "test_job_anchor"

    # Block 1: Suspicious by FastPass, strong anchor agreement (matched 9 / 10)
    # MTQE base score = 0.72 (threshold is 0.75). With +0.05 boost -> 0.77 >= 0.75 -> MTQE_PASSED!
    b1 = IRBlock(
        id="blk_anchor_pass",
        flow_id=FlowID.MAIN_STORY,
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        status=BlockStatus.DRAFTED,
        source_text="Original text snippet with numbers 1 2 3.",
        target_text="Translated text snippet with numbers 1 2 3.",
        bbox=BoundingBox(page=1, x0=10.0, y0=10.0, x1=100.0, y1=50.0),
        provenance={
            "anchor_provenance": "pdfium+proofread",
            "needs_review": False,
            "anchor_stats": {"matched": 9, "vlm_only": 1, "pdfium_only": 0},
        },
    )

    # Block 2: High base score (0.90), but needs_review is True -> flagged "Visual witness discrepancy" -> REPAIR_PENDING
    b2 = IRBlock(
        id="blk_anchor_review",
        flow_id=FlowID.MAIN_STORY,
        spine_index=2,
        block_type=BlockType.NARRATIVE,
        status=BlockStatus.DRAFTED,
        source_text="Second block original text.",
        target_text="Second block translated text.",
        bbox=BoundingBox(page=1, x0=10.0, y0=60.0, x1=100.0, y1=100.0),
        provenance={
            "anchor_provenance": "pdfium+proofread",
            "needs_review": True,
            "anchor_stats": {"matched": 2, "vlm_only": 8, "pdfium_only": 0},
        },
    )

    doc = DocumentIR(
        doc_id=job_id,
        source_path="test.pdf",
        format_type="pdf",
        blocks=[b1, b2],
    )
    ledger.init_job(job_id, doc, target_lang="zh")

    # Mark both blocks as DRAFTED for quality gate
    ledger.save_checkpoints_batch(
        [
            {"block_id": b1.id, "target_text": b1.target_text, "status": BlockStatus.DRAFTED},
            {"block_id": b2.id, "target_text": b2.target_text, "status": BlockStatus.DRAFTED},
        ]
    )

    class _MockFastPass(FastPassFilter):
        def evaluate(
            self,
            source_text: str,
            target_text: str,
            *,
            block_type: Any = None,
            skip_translate: bool = False,
        ) -> Any:
            from ubt.core.qe.fast_pass import FastPassDecision

            return FastPassDecision(
                passed=False,
                reason="Length ratio suspect",
                target_ratio=0.5,
                length_ratio=0.5,
            )

    class _MockQERunner(BaseQERunner):
        async def score_pairs(self, pairs: list[dict[str, str]]) -> list[float]:
            return [0.72, 0.90]

    async def _mock_event(
        event_type: EventType, jid: str, ld: Any, **kwargs: Any
    ) -> TranslationProgressEvent:
        return TranslationProgressEvent(
            event_type=event_type,
            job_id=jid,
            total_blocks=2,
            completed_blocks=2,
        )

    await drain(
        run_quality_gate_stage(
            build_stage_ctx(
                tmp_path,
                ledger=ledger,
                job_id=job_id,
                fast_pass=_MockFastPass(),
                qe_runner=_MockQERunner(),
                create_event=_mock_event,
            ),
        )
    )

    updated_b1 = ledger.get_block("blk_anchor_pass")
    assert updated_b1 is not None
    assert updated_b1.status == BlockStatus.MTQE_PASSED
    assert updated_b1.mtqe_score == pytest.approx(0.77, abs=1e-3)

    updated_b2 = ledger.get_block("blk_anchor_review")
    assert updated_b2 is not None
    assert updated_b2.status == BlockStatus.REPAIR_PENDING
    assert "Visual witness discrepancy" in updated_b2.error_flags


def test_proofread_circuit_breaker_stops_billing_on_dead_endpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The VLM proofread loop must abort once a dead endpoint has failed the
    breaker window, instead of paying for every candidate page.

    Regression guard: ``attempts`` used to advance only on a *successful*
    transcription, so a wholly failing endpoint kept it at 0, never satisfied
    ``attempts >= VLM_CIRCUIT_MIN_TRIES``, and billed every page anyway. The
    sibling missing-pages loop already counted each attempt up front.
    """
    import pypdfium2 as pdfium

    import ubt.adapters.pdf.vlm.transcribe as transcribe_mod
    from ubt.adapters.pdf.docling_parser import vlm_fallback_missing_pages
    from ubt.adapters.pdf.vlm.registry import register_driver
    from ubt.adapters.pdf.vlm.transcribe import FALLBACK_ENV_VAR
    from ubt.core.ir.models import BlockType, BoundingBox, FlowID, IRBlock
    from ubt.core.policy.layout_policy import VLM_CIRCUIT_MIN_TRIES

    n_pages = VLM_CIRCUIT_MIN_TRIES + 3  # strictly more candidates than the window

    pdf = pdfium.PdfDocument.new()
    for _ in range(n_pages):
        pdf.new_page(200, 200)
    pdf_path = tmp_path / "weak.pdf"
    pdf.save(str(pdf_path))
    pdf.close()

    # One high-debris narrative block per page -> WEAK flags every page as a
    # proofread candidate and none as a zero-block "missing" page, isolating the
    # loop this test guards.
    blocks = [
        IRBlock(
            id=f"b{i}",
            flow_id=FlowID.MAIN_STORY,
            spine_index=i + 1,
            block_type=BlockType.NARRATIVE,
            source_text="K V x y z A B C D E F G H I J",
            bbox=BoundingBox(page=i + 1, x0=10.0, y0=10.0, x1=100.0, y1=50.0),
            provenance={"parser": "docling"},
        )
        for i in range(n_pages)
    ]

    class _DeadDriver:
        name = "dead"
        measured_boxes = True

        def recognize(
            self, image: object, page_size_pt: tuple[float, float], scale: float
        ) -> PageTranscript:
            raise RuntimeError("endpoint down")

    register_driver("dead", _DeadDriver)
    monkeypatch.setenv("UBT_VLM_DRIVER", "dead")
    monkeypatch.setenv(FALLBACK_ENV_VAR, "weak")

    calls = 0

    def _boom(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("paid transcription failed")

    # The parser imports transcribe_page_to_blocks from this module at call
    # time, so patching the source symbol is what the running code sees.
    monkeypatch.setattr(transcribe_mod, "transcribe_page_to_blocks", _boom)

    out = vlm_fallback_missing_pages(pdf_path, blocks)

    # Breaker aborts at the attempt window, not after every page.
    assert calls == VLM_CIRCUIT_MIN_TRIES
    assert calls < n_pages
    # Every original block survives un-transcribed.
    assert len(out) == n_pages
    assert all(b.provenance["parser"] == "docling" for b in out)


def test_fallback_keeps_column_order_of_untouched_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Transcribing one page must not re-sort the pages Docling already read.

    The stitching sort keyed on ``-y1`` as well as the page, so a scanned
    appendix page reordered *every* page of the book geometrically — on a
    two-column book that interleaves the columns — and bbox-less blocks slid to
    the front of the document.
    """
    import pypdfium2 as pdfium

    from ubt.adapters.pdf import vlm
    from ubt.adapters.pdf.docling_parser import vlm_fallback_missing_pages
    from ubt.adapters.pdf.vlm.transcribe import FALLBACK_ENV_VAR
    from ubt.core.ir.models import BlockType, BoundingBox, FlowID, IRBlock

    pdf_path = tmp_path / "mixed.pdf"
    doc = pdfium.PdfDocument.new()
    doc.new_page(500, 700)
    doc.new_page(500, 700)
    doc.save(str(pdf_path))
    doc.close()

    columns = [
        (1, 60.0, 640.0, "L1"),
        (2, 60.0, 560.0, "L2"),
        (3, 250.0, 640.0, "R1"),
        (4, 250.0, 560.0, "R2"),
    ]
    blocks = [
        IRBlock(
            id=f"b{i}",
            flow_id=FlowID.MAIN_STORY,
            spine_index=i,
            block_type=BlockType.NARRATIVE,
            source_text=text,
            bbox=BoundingBox(page=1, x0=x0, y0=y0, x1=x0 + 150.0, y1=y0 + 40.0),
            provenance={"parser": "docling"},
        )
        for i, x0, y0, text in columns
    ]

    def _transcribe(
        _path: object, page_no: int, **_kw: object
    ) -> tuple[list[IRBlock], dict[str, int]]:
        fresh = IRBlock(
            id=f"ocr_{page_no}",
            flow_id=FlowID.MAIN_STORY,
            spine_index=99,
            block_type=BlockType.NARRATIVE,
            source_text=f"scanned page {page_no}",
            bbox=BoundingBox(page=page_no, x0=10.0, y0=10.0, x1=400.0, y1=690.0),
            provenance={"parser": "vlm"},
        )
        return [fresh], {}

    from ubt.adapters.pdf.vlm.registry import register_driver

    class _OrderDriver:
        name = "fake-order"
        measured_boxes = True

        def recognize(
            self, image: object, page_size_pt: tuple[float, float], scale: float
        ) -> PageTranscript:
            return PageTranscript()

    register_driver("fake-order", _OrderDriver)
    monkeypatch.setenv("UBT_VLM_DRIVER", "fake-order")
    monkeypatch.setenv(FALLBACK_ENV_VAR, "missing")
    monkeypatch.setattr(vlm.transcribe, "transcribe_page_to_blocks", _transcribe)

    out = vlm_fallback_missing_pages(pdf_path, blocks)

    assert [b.source_text for b in out] == ["L1", "L2", "R1", "R2", "scanned page 2"]


@pytest.mark.fast
def test_transcribe_page_bounds_check(tmp_path: Path) -> None:
    # Create a minimal 1-page PDF using pypdfium2
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument.new()
    pdf.new_page(width=100, height=100)
    pdf_path = tmp_path / "test.pdf"
    pdf.save(str(pdf_path))
    pdf.close()

    # page_no = 0 must raise IndexError, not access pdf[-1]
    with pytest.raises(IndexError):
        transcribe_page_to_blocks(pdf_path, page_no=0)

    # page_no > total pages must raise IndexError
    with pytest.raises(IndexError):
        transcribe_page_to_blocks(pdf_path, page_no=5)


@pytest.mark.fast
def test_vlm_fallback_missing_pages_closes_driver_and_logs_correct_remaining_count(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """[HIGH-T2-1 & MEDIUM-T2-4] vlm_fallback_missing_pages must call driver.close() in finally
    and log len(missing) - idx (not len(missing) - page_no + 1) when circuit breaker trips."""
    pdf_path = tmp_path / "dummy.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    mock_driver = MagicMock()
    mock_driver.close = MagicMock()

    existing_block = IRBlock(
        id="b1",
        flow_id=FlowID.MAIN_STORY,
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        source_text="Existing page 1 text",
        bbox=BoundingBox(page=1, x0=10, y0=10, x1=100, y1=50),
    )

    import pypdfium2 as pdfium

    from ubt.adapters.pdf.vlm.transcribe import VlmFallbackMode

    pdf = pdfium.PdfDocument.new()
    for _ in range(53):
        pdf.new_page(width=200, height=200)
    pdf.save(str(pdf_path))
    pdf.close()

    with (
        patch(
            "ubt.adapters.pdf.vlm.transcribe.get_fallback_mode",
            return_value=VlmFallbackMode.MISSING,
        ),
        patch(
            "ubt.adapters.pdf.vlm.registry.probe_effective_driver",
            return_value=("deepseek", mock_driver),
        ),
        patch(
            "ubt.adapters.pdf.vlm.transcribe.transcribe_page_to_blocks",
            side_effect=RuntimeError("VLM worker crashed"),
        ),
    ):
        result = vlm_fallback_missing_pages(
            pdf_path,
            [existing_block],
            ocr_mode="deepseek",
            page_range=(50, 53),
        )

    assert len(result) == 1
    mock_driver.close.assert_called_once()
    # Circuit breaker trips at idx=3 (page_no=53), so remaining pages is 4 - 3 = 1 (NOT 4 - 53 + 1 = -48!)
    assert "remaining 1 page(s)" in caplog.text
