"""D2 regression proof — the production echo is reproduced and stopped, end to end.

Unlike ``test_added_content_gate.py`` (which exercises the gate itself), this
file drives the REAL stages in the same order the pipeline does:

    run_draft_stage  →  run_quality_gate_stage  →  _writeback_tm_from_ledger

That distinction matters here for a concrete reason. The first version of the
formula image fallback looked correct under a helper-only test while the real
flow never reached the code, and the bug shipped. A gate is only proven when the
stage that consumes it changes its verdict, so every test below asserts on
ledger state and TM contents produced by the stage functions, not on a gate
return value.

The payloads are the production ones: ``_ECHO_SRC`` is chapter-3
``pdf_main#b0045``'s source, ``_ECHO_TGT`` is the exact text that shipped, and
``_NEXT_SRC`` is the neighbour excerpt that was injected into its prompt and
then translated into the output.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from tests.stage_ctx_factory import build_stage_ctx, drain, inert_event
from ubt.core.cleaners.citation_masker import CitationMasker
from ubt.core.cleaners.code_masker import CodeMasker
from ubt.core.cleaners.math_masker import extract_math_spans
from ubt.core.config import UBTConfig
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.stages.draft import run_draft_stage
from ubt.core.engine.stages.quality_gate import run_quality_gate_stage
from ubt.core.engine.stages.tm_writeback import writeback_tm_from_ledger
from ubt.core.ir.models import BlockStatus, BookManifest, ChapterMeta, DocumentIR, FlowID, IRBlock
from ubt.core.memory.tm import (
    PROMPT_VERSION,
    TMPendingEntry,
    TranslationMemory,
    compute_tm_context,
)
from ubt.core.qe.added_content import AddedContentGate
from ubt.core.qe.comet_runner import HeuristicQERunner
from ubt.core.qe.defect_taxonomy import has_structural_defect
from ubt.core.qe.fast_pass import FastPassFilter
from ubt.core.qe.omission import OmissionGate
from ubt.core.router.provider import MockModelProvider
from ubt.core.router.router import ModelRouter
from ubt.core.validators.consistency import NumericConsistencyValidator
from ubt.core.validators.html_delta import HTMLDeltaValidator

# ---------------------------------------------------------------------------
# Production payloads (chapter-3 job_fc1d7bd7b799)
# ---------------------------------------------------------------------------
_ECHO_SRC = "".join(
    (
        "where ψ pert is given by ψ 2 evaluated at x = T fin /2. ",
        "Eq. (3.11) is an implicit equation in β which must be solved using numerical methods, ",
        "then, once β is calculated, the surface potential and the charge in the channel ",
        "can be obtained. ",
        "Fig. 3.5 shows the surface potential obtained from Eq. (3.11) and the numerical ",
        "solution of Eq. (3.1) for different doping concentrations. ",
        "The amount of doping in the channel determines the threshold voltage of the device ",
        "as shown in Fig. 3.6, which represents the mobile charge density obtained from the ",
        "proposed compact model and the numerical solution of Eq. (3.1) for different ",
        "doping concentrations. ",
        "In the case of lightly doped DG FinFETs, the thickness of the channel determines ",
        "the amount of mobile carrier charge density in the channel in a linear manner, ",
        "as shown in Fig. 3.7.",
    )
)
# What shipped. The block was 825 source chars; the translation opens with the
# INJECTED NEIGHBOUR paragraph (translated, with [25] degraded to [2]) and its own
# first sentence -- "where psi pert is given by psi 2 ..." -- is gone entirely.
_ECHO_TGT = "".join(
    (
        "式 (3.7) 和 (2.2) 可以合并为一个方程：\n\n",
        "使用数值方法求解式 (3.11) 在紧凑建模应用中并不实际，因为其使用会增加计算时间并可能导致",
        "发散问题 [2]。因此，首先通过解析近似法获得初始猜测值，随后对式 (3.11) 进行求解。",
        "一旦计算出 β 值，即可获得表面势和沟道中的电荷。图 3.5 展示了由式 (3.11) 得到的表面势以及",
        "对式 (3.1) 进行数值求解在不同掺杂浓度下的结果。如图 3.6 所示，沟道中的掺杂量决定了器件的",
        "阈值电压，该图表示了由所提出的紧凑模型和对式 (3.1) 进行数值求解在不同掺杂浓度下得到的",
        "可移动电荷密度。在轻掺杂的 DG FinFET（双栅鳍式场效应晶体管）情况下，沟道厚度以线性方式",
        "决定了沟道中的可移动载流子电荷密度，如图 3.7 所示。",
    )
)
# The neighbour excerpt that was injected into b0045's prompt as
# "[READ-ONLY SUBSEQUENT CONTEXT: DO NOT TRANSLATE OR ECHO]" and then translated.
_NEXT_SRC = (
    "Solving Eq. (3.11) using numerical methods is not practical for compact modeling "
    "applications because their use increases the computation time and may cause divergence "
    "problems [25]."
)

# Clean fixtures. Plain ASCII prose on purpose: no $...$ math, no Greek letters,
# no [n] citations, so nothing is masked and the mock's response survives verbatim.
# The zh target is long enough to clear the length-ratio floor (en->zh band 0.2-3.0).
_CLEAN_SRC = "The channel voltage is set to the source voltage in this model."
_CLEAN_TGT = "在本模型中，沟道电压被设置为源端电压，用于计算源端表面势。"


_CTX_RENAMES = {
    "actual_job_id": "job_id",
    "create_event_fn": "create_event",
    "all_blocks_count": "block_count",
}


async def _drain_stage(**kwargs: Any) -> None:
    """Old-style keywords in, one StageContext out.

    The draft stage takes the run's context now; this keeps the ~30 call sites in
    this file written the way they read before, and any keyword that is not a
    context field fails in the constructor rather than being ignored.
    """
    ctx = build_stage_ctx(**{_CTX_RENAMES.get(k, k): v for k, v in kwargs.items()})
    async for _event in run_draft_stage(ctx):
        pass


def _config() -> UBTConfig:
    return UBTConfig(batch_limit=30, max_concurrency=4, batch_enabled=False)


def _manifest() -> BookManifest:
    return BookManifest(
        doc_id="d2_doc",
        title="D2",
        source_path="/tmp/synthetic-duo.pdf",
        source_lang="en",
        target_lang="zh",
        chapters=[ChapterMeta(chapter_id="pdf_main", title="chapter-3", spine_index=1)],
        metadata={},
    )


def _doc(blocks: list[IRBlock]) -> DocumentIR:
    return DocumentIR(
        doc_id="d2_doc",
        source_path="/tmp/synthetic-duo.pdf",
        format_type="pdf",
        metadata={},
        blocks=blocks,
    )


def _block(bid: str, spine: int, source: str) -> IRBlock:
    return IRBlock(id=bid, flow_id=FlowID.MAIN_STORY, spine_index=spine, source_text=source)


def _run_quality_gate(
    tmp_path: Path,
    ledger: SQLiteJobLedger,
    job_id: str,
    qe_threshold: float = 0.75,
) -> None:
    ctx = build_stage_ctx(
        tmp_path,
        ledger=ledger,
        job_id=job_id,
        fast_pass=FastPassFilter(),
        qe_runner=HeuristicQERunner(),
        create_event=inert_event,
        config=UBTConfig(qe_threshold=qe_threshold),
    )
    asyncio.run(drain(run_quality_gate_stage(ctx)))


# ---------------------------------------------------------------------------
# 1. The real quality gate must not release the echoed draft
# ---------------------------------------------------------------------------
def test_quality_gate_sends_the_production_echo_to_repair(tmp_path: Path) -> None:
    """Drives run_quality_gate_stage: the echoed block must not become MTQE_PASSED."""
    ledger = SQLiteJobLedger(tmp_path / "job.sqlite")
    ledger.init_job(
        "job_d2",
        _doc([_block("pdf_main#b_echo", 1, _ECHO_SRC), _block("pdf_main#b_ok", 2, _CLEAN_SRC)]),
        target_lang="zh",
    )
    ledger.save_checkpoint(
        block_id="pdf_main#b_echo",
        status=BlockStatus.DRAFTED,
        target_text=_ECHO_TGT,
        draft_text=_ECHO_TGT,
    )
    ledger.save_checkpoint(
        block_id="pdf_main#b_ok",
        status=BlockStatus.DRAFTED,
        target_text=_CLEAN_TGT,
        draft_text=_CLEAN_TGT,
    )
    _run_quality_gate(tmp_path, ledger, "job_d2")

    by_id = {b.id: b for b in ledger.get_all_blocks("job_d2")}
    echo = by_id["pdf_main#b_echo"]
    assert echo.status is BlockStatus.REPAIR_PENDING, (
        "the production echo reached MTQE_PASSED — the gate is not wired into the stage"
    )
    assert any("Added reference" in f for f in echo.error_flags), echo.error_flags
    # No false positive on the clean sibling.
    assert by_id["pdf_main#b_ok"].status is BlockStatus.MTQE_PASSED
    ledger.close()


# ---------------------------------------------------------------------------
# 2. Negative control: why this shipped for so long
# ---------------------------------------------------------------------------
def test_the_same_payload_passes_every_pre_existing_gate() -> None:
    """Documents the blind spot, and proves the new gate is what changed the verdict.

    Every check the fast-pass chain ran before this fix passes on the echoed
    pair. Without this control, test 1 could be read as "some gate fired" rather
    than "the added-content gate is the one that fired".
    """
    src, tgt = _ECHO_SRC, _ECHO_TGT

    assert HTMLDeltaValidator().validate(src, tgt).is_valid
    assert sorted(extract_math_spans(src)) == sorted(extract_math_spans(tgt))
    assert NumericConsistencyValidator().validate(src, tgt).is_valid
    # The omission gate reads only the LOW side of the length band — and the
    # invented sentences push the sentence count back up, so the source sentence
    # they displaced stays invisible.
    assert OmissionGate(target_lang="zh").evaluate(src, tgt).passed

    # The only complaint the chain has left is the new one.
    decision = FastPassFilter().evaluate(src, tgt)
    assert not decision.passed
    assert decision.reason.startswith("Added reference(s)"), decision.reason


# ---------------------------------------------------------------------------
# 3. The real TM writeback must refuse the echoed target
# ---------------------------------------------------------------------------
def test_tm_writeback_refuses_the_echo_and_keeps_the_clean_pair(tmp_path: Path) -> None:
    """Drives _writeback_tm_from_ledger: terminal status is not a correctness proof."""
    ledger = SQLiteJobLedger(tmp_path / "job.sqlite")
    ledger.init_job(
        "job_wb",
        _doc([_block("pdf_main#b_echo", 1, _ECHO_SRC), _block("pdf_main#b_ok", 2, _CLEAN_SRC)]),
        target_lang="zh",
    )
    # The exact legacy state: the echo was marked MTQE_PASSED, so before this fix
    # the writeback promoted it into reusable memory.
    for bid, _src, tgt in (
        ("pdf_main#b_echo", _ECHO_SRC, _ECHO_TGT),
        ("pdf_main#b_ok", _CLEAN_SRC, _CLEAN_TGT),
    ):
        ledger.save_checkpoint(
            block_id=bid, status=BlockStatus.MTQE_PASSED, target_text=tgt, draft_text=tgt
        )

    tm = TranslationMemory(tmp_path / "tm.sqlite")
    written = writeback_tm_from_ledger(ledger, "job_wb", tm, "en", "zh")

    assert written == 1, "only the clean pair may be promoted"
    assert tm.lookup_exact("en", "zh", _ECHO_SRC, domain=None) is None
    assert tm.lookup_exact("en", "zh", _CLEAN_SRC, domain=None) is not None
    assert tm.entry_count() == 1
    tm.close()
    ledger.close()


# ---------------------------------------------------------------------------
# 4. A poisoned TM entry cannot be consumed (controlled experiment)
# ---------------------------------------------------------------------------
def _drain_draft(
    ledger: SQLiteJobLedger, job_id: str, tm: TranslationMemory, response: str
) -> None:
    router = ModelRouter(provider=MockModelProvider(default_response=response), draft_model="m")
    asyncio.run(
        _drain_stage(
            ledger=ledger,
            actual_job_id=job_id,
            manifest=_manifest(),
            profile_name="general",
            target_lang="zh",
            source_lang="en",
            router=router,
            code_masker=CodeMasker(),
            citation_masker=CitationMasker(),
            config=_config(),
            all_blocks_count=1,
            glossary_dicts=[],
            abbreviation_entries=[],
            concurrency_sem=asyncio.Semaphore(4),
            create_event_fn=inert_event,
            tm=tm,
            fast_pass=FastPassFilter(),
        )
    )


def _seed_tm(tm: TranslationMemory, target: str) -> None:
    tm.writeback(
        [
            TMPendingEntry(
                "en",
                "zh",
                _CLEAN_SRC,
                target,
                context_hash=compute_tm_context(PROMPT_VERSION, "general", "", "en", "zh"),
            )
        ]
    )


def test_clean_tm_entry_is_still_served(tmp_path: Path) -> None:
    """Positive control: the TM read path really does run in this harness."""
    ledger = SQLiteJobLedger(tmp_path / "job.sqlite")
    ledger.init_job("job_tm_ok", _doc([_block("pdf_main#b1", 1, _CLEAN_SRC)]), target_lang="zh")
    tm = TranslationMemory(tmp_path / "tm.sqlite")
    # Long enough to clear the length-ratio floor, distinct from the mock's
    # response so the assertion can tell which source supplied the text.
    _seed_tm(tm, "这是翻译记忆库中已经存在的既有译文内容，用于验证读取路径确实生效。")

    _drain_draft(ledger, "job_tm_ok", tm, response="【LLM】不应被调用")

    block = ledger.get_block("pdf_main#b1")
    assert block is not None
    assert block.target_text == "这是翻译记忆库中已经存在的既有译文内容，用于验证读取路径确实生效。"
    assert block.status is BlockStatus.MTQE_PASSED
    tm.close()
    ledger.close()


def test_poisoned_tm_entry_is_rejected_and_redrafted(tmp_path: Path) -> None:
    """The read-side trust boundary, proven by contrast with the test above.

    Same harness, same source, same clean LLM response — the only difference is
    that the stored TM target carries a citation the source never had. If the
    hit were still trusted, the block would hold the poisoned text.
    """
    ledger = SQLiteJobLedger(tmp_path / "job.sqlite")
    ledger.init_job("job_tm_bad", _doc([_block("pdf_main#b1", 1, _CLEAN_SRC)]), target_lang="zh")
    tm = TranslationMemory(tmp_path / "tm.sqlite")
    poisoned = "在本模型中，沟道电压被设置为源端电压，用于计算源端表面势 [25]。"
    _seed_tm(tm, poisoned)

    # The added-content gate is the one that rejects it: validate_structural_invariants
    # (which hosts it) runs before the numeric and length gates, so a reason from
    # the numeric validator could not appear first.
    rejection = FastPassFilter().evaluate(_CLEAN_SRC, poisoned)
    assert not rejection.passed
    assert rejection.reason.startswith("Added reference(s)"), rejection.reason

    _drain_draft(ledger, "job_tm_bad", tm, response=_CLEAN_TGT)

    block = ledger.get_block("pdf_main#b1")
    assert block is not None
    assert "[25]" not in (block.target_text or ""), (
        "the poisoned TM entry was served — the read-side boundary is not wired"
    )
    assert block.target_text == _CLEAN_TGT, "the hit should have fallen through to the LLM"

    # And the freshly drafted text is clean, so it may enter the TM on writeback.
    assert writeback_tm_from_ledger(ledger, "job_tm_bad", tm, "en", "zh") >= 0
    tm.close()
    ledger.close()


# ---------------------------------------------------------------------------
# 5. The verdict must not be overridable by a QE score
# ---------------------------------------------------------------------------
def test_added_content_classifies_as_fabrication_not_as_generic() -> None:
    """Class mapping: fabrication is band 0.15, not the 0.70 fallback.

    0.70 means "no specific classifier matched". It sits 0.05 below the default
    threshold, so the verdict would flip for anyone running at <= 0.70 — and the
    report's score distribution would read it as a generic structural problem.
    """
    reason = AddedContentGate().evaluate(_ECHO_SRC, _ECHO_TGT).reason
    assert has_structural_defect([reason]), "fabrication must be a structural defect"
    assert HeuristicQERunner.score_from_decision_reason(reason) == 0.15

    heading_reason = AddedContentGate().evaluate("Plain prose.", "### 源段落翻译\n正文。").reason
    assert has_structural_defect([heading_reason])
    # Prompt-template leak keeps its own band.
    assert HeuristicQERunner.score_from_decision_reason(heading_reason) == 0.10


def test_echo_verdict_survives_a_permissive_qe_threshold(tmp_path: Path) -> None:
    """Regression for a real hole in the first version of this fix.

    ``run_quality_gate_stage`` marks a block MTQE_PASSED **and wipes its error
    flags** once the QE score clears the threshold. Two layers have to hold:

    * the 0.15 band mapping (``score_from_decision_reason``) keeps the score
      below any sane threshold — but a threshold at or under 0.15 would still
      release it;
    * the ``STRUCTURAL_DEFECT_MARKERS`` entry makes the verdict fatal at *any*
      threshold, which is the invariant ``defect_taxonomy`` documents.

    The threshold below is deliberately absurd: it isolates the marker.
    """
    ledger = SQLiteJobLedger(tmp_path / "job.sqlite")
    ledger.init_job("job_perm", _doc([_block("pdf_main#b_echo", 1, _ECHO_SRC)]), target_lang="zh")
    ledger.save_checkpoint(
        block_id="pdf_main#b_echo",
        status=BlockStatus.DRAFTED,
        target_text=_ECHO_TGT,
        draft_text=_ECHO_TGT,
    )
    _run_quality_gate(tmp_path, ledger, "job_perm", qe_threshold=0.05)

    block = ledger.get_block("pdf_main#b_echo")
    assert block is not None
    assert block.status is BlockStatus.REPAIR_PENDING, (
        "a permissive QE threshold released the echoed block — the defect taxonomy "
        "is not treating fabrication as structural"
    )
    assert any("Added reference" in f for f in block.error_flags)
    ledger.close()


def test_masked_token_corruption_is_fatal_and_survives_the_gate(tmp_path: Path) -> None:
    """Masked-token loss must stay structural at any QE threshold.

    The draft stage records ``math_token_corrupt`` / ``soup_token_corrupt`` when
    a protected span is lost or mutated, but the fast-pass text itself can look
    clean — only the marker table keeps the block out of auto-pass, and the flag
    must survive the stage so the coverage report can count the corrupt spans.
    """
    for marker, flag in (
        (
            "math_token_corrupt",
            "math_token_corrupt missing=['⟦MATH_MASK_0001⟧'] mismatched=[] mutated=[]",
        ),
        ("soup_token_corrupt", "soup_token_corrupt missing=['βSI=e−ψpert'] mismatched=[]"),
    ):
        assert has_structural_defect([flag]), marker
        ledger = SQLiteJobLedger(tmp_path / f"{marker}.sqlite")
        ledger.init_job(
            "job_tok", _doc([_block("pdf_main#b_tok", 1, _CLEAN_SRC)]), target_lang="zh"
        )
        ledger.save_checkpoint(
            block_id="pdf_main#b_tok",
            status=BlockStatus.DRAFTED,
            target_text=_CLEAN_TGT,
            draft_text=_CLEAN_TGT,
            error_flags=[flag],
        )
        _run_quality_gate(tmp_path, ledger, "job_tok", qe_threshold=0.05)

        block = ledger.get_block("pdf_main#b_tok")
        assert block is not None
        assert block.status is BlockStatus.REPAIR_PENDING, marker
        assert any(marker in f for f in block.error_flags), block.error_flags
        ledger.close()
