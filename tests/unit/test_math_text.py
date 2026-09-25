"""Unit tests for C-track math-text spans (calibrated on KV corpus)."""

import asyncio

from tests.stage_ctx_factory import build_stage_ctx
from ubt.core.cleaners.math_text import (
    extract_text_spans,
    is_translatable_text,
    reassemble,
    skeleton_holds,
    translate_math_text,
)
from ubt.core.engine.events import TranslationProgressEvent


def test_kv_corpus_calibration() -> None:
    # The whole KV handbook has exactly these 3 spans (18 formulas).
    assert not is_translatable_text("softmax")  # term
    assert not is_translatable_text("GiB")  # unit
    assert is_translatable_text("by terms")  # natural phrase


def test_classifier_boundaries() -> None:
    assert is_translatable_text("Hit Rate")  # BabelDOC's frozen example
    assert not is_translatable_text("ReLU")  # single token, however cased
    assert not is_translatable_text("Figure 1")  # digits are Gate-4 territory
    assert not is_translatable_text("x_{i}")  # nested math stays
    assert not is_translatable_text("")  # empty
    assert not is_translatable_text("a" * 121)  # pathological


def test_extract_reading_order_and_balance() -> None:
    spans = extract_text_spans("A \\text{by terms} and $\\mathrm{X}$ end")
    assert [(s.cmd, s.inner) for s in spans] == [
        ("text", "by terms"),
        ("mathrm", "X"),
    ]
    assert extract_text_spans("broken \\text{abc") == []


def test_skeleton_invariant() -> None:
    src = "K = [k_1; \\text{by terms}] + x"
    assert skeleton_holds(src, "K = [k_1; \\text{按项}] + x")
    assert not skeleton_holds(src, "K = [k_1; \\text{按项}] + y")  # math changed
    assert not skeleton_holds(src, "K = [k_1] + x")  # span deleted
    assert skeleton_holds("plain x", "plain x")  # span-free trivially holds


def test_reassemble_keeps_skeleton() -> None:
    src = "\\text{by terms} + \\text{softmax}"
    out = reassemble(src, {0: "按项"})
    assert out == "\\text{按项} + \\text{softmax}"


async def _fake_complete(inner: str, target_lang: str) -> str:
    assert target_lang == "zh"
    return {"by terms": "按项"}.get(inner, inner)


def test_translate_end_to_end() -> None:
    out = asyncio.run(translate_math_text("K = [k_1; \\text{by terms}]", "zh", _fake_complete))
    assert out == "K = [k_1; \\text{按项}]"


def test_translate_fail_closed() -> None:
    async def _boom(inner: str, target_lang: str) -> str:
        raise RuntimeError("llm down")

    assert asyncio.run(translate_math_text("\\text{by terms}", "zh", _boom)) is None

    async def _evil(inner: str, target_lang: str) -> str:
        return "\\textbf{按项}"

    assert asyncio.run(translate_math_text("\\text{by terms}", "zh", _evil)) is None
    # Nothing translatable at all.
    assert asyncio.run(translate_math_text("K_{t} + \\text{softmax}", "zh", _fake_complete)) is None


def test_c_text_stage_translates_spans(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Stage 3.5: ingest-finalized FORMULA blocks get span translation."""
    import asyncio
    from pathlib import Path

    from ubt.core.config import UBTConfig
    from ubt.core.engine.ledger import SQLiteJobLedger
    from ubt.core.engine.stages.ctext import run_c_text_stage
    from ubt.core.ir.models import (
        BlockStatus,
        BlockType,
        DocumentIR,
        FlowID,
        IRBlock,
    )
    from ubt.core.router.provider import MockModelProvider
    from ubt.core.router.router import ModelRouter

    assert UBTConfig().c_text_enabled is False  # flag defaults off

    formula_src = "K = [k_1; \\text{by terms}] + x"
    job = "job_ctrack"
    # Mimic ingest finalization: verbatim target, MTQE_PASSED, never drafted.
    doc = DocumentIR(
        doc_id="ctrack_doc",
        source_path="/tmp/ctrack.pdf",
        format_type="pdf",
        metadata={},
        blocks=[
            IRBlock(
                id="ch01#b001",
                flow_id=FlowID.MAIN_STORY,
                spine_index=1,
                block_type=BlockType.FORMULA,
                source_text=formula_src,
                target_text=formula_src,
                skip_translate=True,
                status=BlockStatus.MTQE_PASSED,
            ),
            IRBlock(
                id="ch01#b002",
                flow_id=FlowID.MAIN_STORY,
                spine_index=2,
                block_type=BlockType.FORMULA,
                source_text="K_{t} + \\text{softmax}",
                target_text="K_{t} + \\text{softmax}",
                skip_translate=True,
                status=BlockStatus.MTQE_PASSED,
            ),
        ],
    )
    ledger = SQLiteJobLedger(Path(str(tmp_path)) / "ctrack.sqlite")
    ledger.init_job(job, doc, target_lang="zh")
    router = ModelRouter(
        provider=MockModelProvider(default_response="按项"),
        draft_model="mock",
        max_retries=0,
    )
    ctx = build_stage_ctx(tmp_path, ledger=ledger, job_id=job, router=router, target_lang="zh")

    async def _go() -> list[TranslationProgressEvent]:
        return [e async for e in run_c_text_stage(ctx)]

    events = asyncio.run(_go())
    event = events[0] if events else None
    # The completion event's message is the stage's outward count channel: the
    # tally must be carried there, not in a context field no consumer reads.
    assert event is not None and event.message == (
        "C-track: 1 formula(s) span-translated, 1 failed closed, 0 already translated"
    )
    by_id = {b.id: b for b in ledger.get_all_blocks(job)}
    assert by_id["ch01#b001"].target_text == "K = [k_1; \\text{按项}] + x"
    assert by_id["ch01#b001"].status == BlockStatus.MTQE_PASSED  # untouched
    assert by_id["ch01#b002"].target_text == "K_{t} + \\text{softmax}"  # kept


def test_c_text_stage_skips_already_translated(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A resume/re-export must not re-translate or overwrite accepted output."""
    from pathlib import Path

    from ubt.core.engine.ledger import SQLiteJobLedger
    from ubt.core.engine.stages.ctext import run_c_text_stage
    from ubt.core.ir.models import BlockStatus, BlockType, DocumentIR, FlowID, IRBlock
    from ubt.core.router.provider import MockModelProvider
    from ubt.core.router.router import ModelRouter

    src = "K = [k_1; \\text{by terms}] + x"
    accepted = "K = [k_1; \\text{按项}] + x"
    job = "job_ctrack_resume"
    doc = DocumentIR(
        doc_id="ctrack_doc",
        source_path="/tmp/ctrack.pdf",
        format_type="pdf",
        metadata={},
        blocks=[
            IRBlock(
                id="ch01#b001",
                flow_id=FlowID.MAIN_STORY,
                spine_index=1,
                block_type=BlockType.FORMULA,
                source_text=src,
                target_text=accepted,
                skip_translate=True,
                status=BlockStatus.MTQE_PASSED,
            )
        ],
    )
    ledger = SQLiteJobLedger(Path(str(tmp_path)) / "ctrack.sqlite")
    ledger.init_job(job, doc, target_lang="zh")
    router = ModelRouter(
        provider=MockModelProvider(default_response="SHOULD-NOT-BE-USED"),
        draft_model="mock",
        max_retries=0,
    )
    ctx = build_stage_ctx(tmp_path, ledger=ledger, job_id=job, router=router, target_lang="zh")

    async def _go() -> list[TranslationProgressEvent]:
        return [e async for e in run_c_text_stage(ctx)]

    events = asyncio.run(_go())
    # Nothing to do: the stage returns before emitting its completion event,
    # and the accepted target stays character-identical (the mock's
    # "SHOULD-NOT-BE-USED" default would surface if it were re-translated).
    assert events == []
    assert {b.id: b for b in ledger.get_all_blocks(job)}["ch01#b001"].target_text == accepted
    ledger.close()
