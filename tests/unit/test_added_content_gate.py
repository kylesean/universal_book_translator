"""Added-content gate — the complement of the omission gate.

The omission gate catches what a translation LOST. This one catches what it
GAINED, which nothing did until now. The regression proof is chapter-3
``pdf_main#b0045``: the drafter translated the read-only *neighbour* excerpt it
was shown and prepended it to its own output. The result passed the entire
fast-pass chain — the added sentences are fluent, on-topic and in the target
language, and because the omission gate only reads the low side of the length
band the invented sentences actively masked the source sentence they displaced.
It was then written into the shared Translation Memory and served verbatim on
every later run (``use_count`` 6-9 for the affected entries).
"""
import asyncio
from pathlib import Path

import pytest

from tests.stage_ctx_factory import build_stage_ctx, drain, inert_event
from tests.unit.ir_seed import SeedDoc, seed_job
from ubt.core.cleaners.math_masker import extract_math_spans
from ubt.core.config import UBTConfig
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.stages.quality_gate import run_quality_gate_stage
from ubt.core.engine.stages.tm_writeback import tm_writeback_eligible
from ubt.core.ir.models import BlockStatus, BlockType, FlowID, IRBlock
from ubt.core.memory.tm import (
    PROVENANCE_MACHINE,
    TMPendingEntry,
    TranslationMemory,
)
from ubt.core.qe.added_content import (
    AddedContentGate,
    markdown_headings,
    reference_tokens,
)
from ubt.core.qe.comet_runner import QE_SCORE_FABRICATED, HeuristicQERunner
from ubt.core.qe.defect_taxonomy import (
    CRITICAL_DEFECT_MARKERS,
    NEAR_ECHO_MARKER,
    STRUCTURAL_DEFECT_MARKERS,
    has_structural_defect,
    is_transient_failure,
)
from ubt.core.qe.fast_pass import FastPassFilter
from ubt.core.qe.omission import OmissionGate
from ubt.core.validators.consistency import NumericConsistencyValidator
from ubt.core.validators.html_delta import HTMLDeltaValidator

# ---------------------------------------------------------------------------
# Real chapter-3 payloads, trimmed to the parts that carry the signal.
# ---------------------------------------------------------------------------
# Source mentions 3.1/3.5/3.6/3.11 and no citations at all.
_B0045_SRC = (
    "where psi_pert is given by psi_2 evaluated at x = T_fin/2. Eq. (3.11) is an implicit "
    "equation in beta which must be solved using numerical methods. Fig. 3.5 shows the "
    "surface potential obtained from Eq. (3.11) and the numerical solution of Eq. (3.1) "
    "for different doping concentrations. Fig. 3.6 represents the mobile charge density."
)
# The shipped translation: a translated *neighbour* paragraph prepended, carrying
# the citation [2] and the equation number (2.2) that this source never had.
_B0045_TGT = (
    "式 (3.7) 和 (2.2) 可以合并为一个方程：\n\n"
    "使用数值方法求解式 (3.11) 在紧凑建模应用中并不实际，因为其使用会增加计算时间并可能导致"
    "发散问题 [2]。因此，首先通过解析近似法获得初始猜测值，随后对式 (3.11) 进行求解。"
)

# A clean pair: every reference in the target is present in the source.
_CLEAN_SRC = (
    "Eqs. (3.7), (3.8) can be written as a single equation. Solving Eq. (3.11) numerically "
    "is not practical [25]. See Figs. 3.14 and 3.15."
)
_CLEAN_TGT = "式 (3.7)、(3.8) 可以合并为一个方程。用数值方法求解式 (3.11) 并不实际 [25]。见图 3.14 和图 3.15。"


class TestReferenceExtraction:
    def test_coordinated_list_yields_every_number(self) -> None:
        """'Figs. 3.14 and 3.15' must yield both, or the correct 3.15 looks fabricated."""
        assert reference_tokens("as shown in Figs. 3.14 and 3.15, which compare") == {
            "3.14",
            "3.15",
        }
        assert reference_tokens("Eqs. (3.7), (3.8) can be written") == {"3.7", "3.8"}

    def test_plural_and_uppercase_keywords_match(self) -> None:
        assert reference_tokens("FIG. 3.9: a schematic") == {"3.9"}
        assert reference_tokens("Equations. 3.10 shows") == {"3.10"}
        assert reference_tokens("Tables 3.2") == {"3.2"}

    def test_docling_spaced_form_matches(self) -> None:
        """Docling emits 'E q s . \\, ( 3 . 7 )'; unspaced, it would parse as empty."""
        assert reference_tokens(r"E q s . \, ( 3 . 7 ) , ( 3 . 8 ) \, \text { x }") == {
            "3.7",
            "3.8",
        }

    def test_citation_lists_and_chinese_spellings(self) -> None:
        assert reference_tokens("reported in [27,28]") == {"27", "28"}
        assert reference_tokens("如式 (3.25) 所示") == {"3.25"}
        assert reference_tokens("见图 3.14") == {"3.14"}

    def test_plain_decimal_is_not_a_reference(self) -> None:
        """A bare value must not be mistaken for an equation number."""
        assert reference_tokens("the ratio is 1.5 for all samples") == set()
        assert reference_tokens("see (2.2)") == {"2.2"}


class TestAddedContentGate:
    def test_clean_pair_passes(self) -> None:
        d = AddedContentGate().evaluate(_CLEAN_SRC, _CLEAN_TGT)
        assert d.passed, d.reason

    def test_real_echo_payload_fails_with_actionable_reason(self) -> None:
        d = AddedContentGate().evaluate(_B0045_SRC, _B0045_TGT)
        assert not d.passed
        # 3.7 comes from the echoed neighbour sentence; 2.2 and [2] are invented.
        assert set(d.fabricated_refs) == {"2", "2.2", "3.7"}
        # The reason doubles as the repair instruction (surfaced verbatim).
        assert "Source Paragraph to Translate" in d.reason
        assert "delete the fabricated reference" in d.reason

    def test_echo_without_reference_numbers_is_not_claimed(self) -> None:
        """Honest about the limit: prose-only echo carries no deterministic signal."""
        src = "The keeper climbed the stairs. The lantern was brass."
        tgt = "老守卫每天爬上螺旋楼梯。他提着一盏黄铜灯笼。这是一段来自邻居段落的回声内容。"
        assert AddedContentGate().evaluate(src, tgt).passed

    def test_leaked_heading_fails_only_when_source_has_none(self) -> None:
        src = "The Unified FinFET Compact Model is extended to model GAA devices."
        leaked = "### 源段落翻译\n统一 FinFET 紧凑模型被扩展以模拟 GAA 器件。"
        d = AddedContentGate().evaluate(src, leaked)
        assert not d.passed
        assert d.leaked_headings == ("### 源段落翻译",)

        # With the source carrying a heading, the model is reproducing structure.
        src_with_heading = "### 3.2 Unified FinFET compact model\nBody text here."
        assert (
            AddedContentGate()
            .evaluate(src_with_heading, "### 3.2 统一的 FinFET 紧凑模型\n正文。")
            .passed
        )

    def test_verbatim_source_echo_never_fires(self) -> None:
        """skip_translate contract: target == source cannot add anything."""
        for text in (_B0045_SRC, _CLEAN_SRC, "\\beta = \\sqrt { \\frac { q } { 2 x } }"):
            assert AddedContentGate().evaluate(text, text).passed

    def test_markdown_headings_helper(self) -> None:
        assert markdown_headings("3.2 Unified FinFET") == ()
        assert markdown_headings("### 源段落翻译\nbody") == ("### 源段落翻译",)


class TestFastPassIntegration:
    def test_contaminated_draft_fails_fast_pass(self) -> None:
        """Routes to REPAIR_PENDING with the reason as the fix instruction."""
        decision = FastPassFilter().evaluate(_B0045_SRC, _B0045_TGT)
        assert not decision.passed
        assert "Added reference" in decision.reason

    def test_contaminated_target_rejected_on_tm_hit_path(self) -> None:
        """draft.py re-evaluates TM hits with fast pass; the gate must cover it."""
        decision = FastPassFilter().evaluate(_B0045_SRC, _B0045_TGT, block_type=BlockType.NARRATIVE)
        assert not decision.passed

    def test_clean_pair_still_passes_the_whole_chain(self) -> None:
        decision = FastPassFilter().evaluate(_CLEAN_SRC, _CLEAN_TGT, block_type=BlockType.NARRATIVE)
        assert decision.passed, decision.reason


class TestTmTrustBoundary:
    """A TM entry outlives its run, so write and read are both trust boundaries."""

    def _block(self, source: str, target: str, *, skip: bool = False) -> IRBlock:
        return IRBlock(
            id="b_echo",
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            source_text=source,
            target_text=target,
            skip_translate=skip,
        )

    def test_contaminated_block_is_refused_writeback(self) -> None:
        assert not tm_writeback_eligible(self._block(_B0045_SRC, _B0045_TGT))

    def test_clean_block_is_eligible(self) -> None:
        assert tm_writeback_eligible(self._block(_CLEAN_SRC, _CLEAN_TGT))

    def test_skip_translate_block_is_eligible(self) -> None:
        """Formula blocks echo their source by contract, so they cannot add."""
        assert tm_writeback_eligible(self._block(_B0045_SRC, _B0045_SRC, skip=True))

    def test_evict_ids_removes_entries_and_keeps_fts_consistent(self, tmp_path: Path) -> None:
        """Reusable memory needs a correction path, not just an append path."""
        tm = TranslationMemory(tmp_path / "tm.sqlite")
        tm.writeback(
            [
                TMPendingEntry(
                    src_lang="en",
                    tgt_lang="zh",
                    source_text=_B0045_SRC,
                    target_text=_B0045_TGT,
                    provenance=PROVENANCE_MACHINE,
                    context_hash="ctx",
                ),
                TMPendingEntry(
                    src_lang="en",
                    tgt_lang="zh",
                    source_text=_CLEAN_SRC,
                    target_text=_CLEAN_TGT,
                    provenance=PROVENANCE_MACHINE,
                    context_hash="ctx",
                ),
            ]
        )
        assert tm.entry_count() == 2
        poisoned = [
            e.id
            for e in tm.scan()
            if not AddedContentGate().evaluate(e.source_text, e.target_text).passed
        ]
        assert len(poisoned) == 1

        assert tm.evict_ids(poisoned) == 1
        assert tm.entry_count() == 1
        assert tm.lookup_exact("en", "zh", _B0045_SRC, context_hash="ctx") is None
        # The surviving entry is still reachable and the trigram index is intact.
        assert tm.lookup_exact("en", "zh", _CLEAN_SRC, context_hash="ctx") is not None
        tm.close()


class TestLedgerRequeue:
    def test_reset_blocks_to_pending_clears_translation_state(self, tmp_path: Path) -> None:
        """save_checkpoint can never clear target_text, so re-queueing needs its own path."""
        ledger = SQLiteJobLedger(tmp_path / "job.sqlite")
        block = IRBlock(
            id="pdf_main#b_echo",
            flow_id=FlowID.MAIN_STORY,
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            source_text=_B0045_SRC,
        )
        seed_job(ledger,
            "job_x",
            SeedDoc(
                doc_id="test_doc_sha256",
                source_path="/tmp/synthetic-duo.pdf",
                format_type="pdf",
                blocks=[block],
            ),
            target_lang="zh",
        )
        # The poisoned target lands exactly the way the pipeline wrote it.
        ledger.save_checkpoint(
            block_id="pdf_main#b_echo",
            status=BlockStatus.MTQE_PASSED,
            target_text=_B0045_TGT,
            draft_text=_B0045_TGT,
            mtqe_score=1.0,
            tm_hit=True,
        )
        stored = ledger.get_block("pdf_main#b_echo")
        assert stored is not None and stored.target_text == _B0045_TGT

        assert ledger.reset_blocks_to_pending(["pdf_main#b_echo"]) == 1
        requeued = ledger.get_block("pdf_main#b_echo")
        assert requeued is not None
        assert requeued.status is BlockStatus.PENDING
        assert not requeued.target_text
        # fetch_pending_blocks only ever returns PENDING blocks, so the next run
        # re-drafts this one instead of serving the stored text.
        pending = [b.id for b in ledger.fetch_pending_blocks("job_x")]
        assert "pdf_main#b_echo" in pending

    def test_empty_input_is_a_noop(self, tmp_path: Path) -> None:
        ledger = SQLiteJobLedger(tmp_path / "job.sqlite")
        assert ledger.reset_blocks_to_pending([]) == 0


def test_chinese_section_reference_is_fabricated_when_source_lacks_it() -> None:
    from ubt.core.qe.added_content import AddedContentGate, reference_tokens

    decision = AddedContentGate().evaluate(
        "The channel voltage is fixed.", "详见第 3.2 节，电压被固定。"
    )
    assert not decision.passed
    assert "3.2" in decision.fabricated_refs
    assert "3.2" in reference_tokens("见第 3.2 节")


def test_parenthesised_plain_quantity_is_not_a_fabricated_reference() -> None:
    from ubt.core.qe.added_content import AddedContentGate

    decision = AddedContentGate().evaluate("The speedup was 3.5 times.", "实现了 (3.5) 倍的加速。")
    assert decision.passed, decision.reason


def test_version_spacing_collapse_does_not_cross_sentence_edges() -> None:
    from ubt.core.qe.added_content import reference_tokens

    # "(2020. 5" must not fuse into the version-shaped "2020.5".
    assert "2020.5" not in reference_tokens("Published in 2020. 5 samples were used.")


def test_appendix_and_three_level_references_are_seen() -> None:
    from ubt.core.qe.added_content import reference_tokens

    assert reference_tokens("See Appendix A.10 for the derivation.") == {"A.10"}
    assert reference_tokens("见附录 A.10 中的推导。") == {"A.10"}
    assert reference_tokens("Section 3.4.1 generalises Eq. (3.4).") == {"3.4.1", "3.4"}


def test_fabricated_figure_reference_is_not_exempted_by_a_plain_source_number() -> None:
    from ubt.core.qe.added_content import AddedContentGate

    decision = AddedContentGate().evaluate(
        "The speedup was 3.5 times.", "见图 3.5，实现了 3.5 倍的加速。"
    )
    assert not decision.passed
    assert "3.5" in decision.fabricated_refs


def test_correct_figure_reference_still_passes_when_source_has_it() -> None:
    from ubt.core.qe.added_content import AddedContentGate

    decision = AddedContentGate().evaluate(
        "Fig. 3.5 shows the surface potential.", "图 3.5 展示了表面电势。"
    )
    assert decision.passed, decision.reason


@pytest.mark.fast
def test_spaced_three_level_reference_is_seen() -> None:
    from ubt.core.qe.added_content import AddedContentGate, reference_tokens

    assert reference_tokens("Section 3 . 4 . 1") == frozenset({"3.4.1"})
    decision = AddedContentGate().evaluate(
        "As described in Section 3 . 4 . 1 , the method generalises.",
        "如第 3.4.1 节所述，该方法得到了推广。",
    )
    assert decision.passed, decision.reason


@pytest.mark.fast
def test_added_content_fullwidth_parentheses() -> None:
    """Equation references with full-width parentheses （3.11） are recognized."""
    src = "As shown in Eq. (3.11), the rate is constant."
    tgt = "如式（3.11）所示，速率是恒定的。"

    gate = AddedContentGate()
    decision = gate.evaluate(src, tgt)
    assert decision.passed, f"Gate failed unexpectedly: {decision.reason}"
    assert "3.11" in decision.source_refs
    assert "3.11" in decision.target_refs

    # Conversely, a fabricated fullwidth reference should be caught
    tgt_bad = "如式（2.2）所示，速率是恒定的。"
    decision_bad = gate.evaluate(src, tgt_bad)
    assert not decision_bad.passed
    assert "2.2" in decision_bad.fabricated_refs


_a0920_SOURCE_PARAGRAPH = (
    "The device operates in inversion when the gate exceeds the threshold "
    "voltage across the oxide layer here."
)


def test_near_echo_is_the_same_defect_class_as_an_exact_echo() -> None:
    """An almost-verbatim target is an untranslated paragraph, not "other".

    The near-echo reason matched none of the score classifier's keywords, so it
    landed on ``QE_SCORE_STRUCTURAL_OTHER`` (0.70) — within 0.05 of the default
    threshold — and carried no structural marker, so the quality gate released
    a never-translated paragraph as ``MTQE_PASSED``.
    """
    near_echo = _a0920_SOURCE_PARAGRAPH.replace("layer", "layers")
    decision = FastPassFilter(target_lang="zh").evaluate(
        _a0920_SOURCE_PARAGRAPH, near_echo, block_type=BlockType.NARRATIVE
    )
    assert not decision.passed
    assert decision.reason.startswith(NEAR_ECHO_MARKER)
    assert HeuristicQERunner.score_from_decision_reason(decision.reason) == QE_SCORE_FABRICATED
    assert has_structural_defect([decision.reason])
    assert any(marker in decision.reason for marker in CRITICAL_DEFECT_MARKERS)


_d2echo_CLEAN_SRC = "The channel voltage is set to the source voltage in this model."

_d2echo_CLEAN_TGT = "在本模型中，沟道电压被设置为源端电压，用于计算源端表面势。"

_d2echo_ECHO_SRC = "".join(
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

_d2echo_ECHO_TGT = "".join(
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


def _d2echo_block(bid: str, spine: int, source: str) -> IRBlock:
    return IRBlock(id=bid, flow_id=FlowID.MAIN_STORY, spine_index=spine, source_text=source)


def _d2echo_doc(blocks: list[IRBlock]) -> SeedDoc:
    return SeedDoc(
        doc_id="d2_doc",
        source_path="/tmp/synthetic-duo.pdf",
        format_type="pdf",
        metadata={},
        blocks=blocks,
    )


def _d2echo_run_quality_gate(
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


def test_quality_gate_sends_the_production_echo_to_repair(tmp_path: Path) -> None:
    """Drives run_quality_gate_stage: the echoed block must not become MTQE_PASSED."""
    ledger = SQLiteJobLedger(tmp_path / "job.sqlite")
    seed_job(ledger,
        "job_d2",
        _d2echo_doc(
            [
                _d2echo_block("pdf_main#b_echo", 1, _d2echo_ECHO_SRC),
                _d2echo_block("pdf_main#b_ok", 2, _d2echo_CLEAN_SRC),
            ]
        ),
        target_lang="zh",
    )
    ledger.save_checkpoint(
        block_id="pdf_main#b_echo",
        status=BlockStatus.DRAFTED,
        target_text=_d2echo_ECHO_TGT,
        draft_text=_d2echo_ECHO_TGT,
    )
    ledger.save_checkpoint(
        block_id="pdf_main#b_ok",
        status=BlockStatus.DRAFTED,
        target_text=_d2echo_CLEAN_TGT,
        draft_text=_d2echo_CLEAN_TGT,
    )
    _d2echo_run_quality_gate(tmp_path, ledger, "job_d2")

    by_id = {b.id: b for b in ledger.get_all_blocks("job_d2")}
    echo = by_id["pdf_main#b_echo"]
    assert echo.status is BlockStatus.REPAIR_PENDING, (
        "the production echo reached MTQE_PASSED — the gate is not wired into the stage"
    )
    assert any("Added reference" in f for f in echo.error_flags), echo.error_flags
    # No false positive on the clean sibling.
    assert by_id["pdf_main#b_ok"].status is BlockStatus.MTQE_PASSED
    ledger.close()


def test_the_same_payload_passes_every_pre_existing_gate() -> None:
    """Documents the blind spot, and proves the new gate is what changed the verdict.

    Every check the fast-pass chain ran before this fix passes on the echoed
    pair. Without this control, test 1 could be read as "some gate fired" rather
    than "the added-content gate is the one that fired".
    """
    src, tgt = _d2echo_ECHO_SRC, _d2echo_ECHO_TGT

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


def test_added_content_classifies_as_fabrication_not_as_generic() -> None:
    """Class mapping: fabrication is band 0.15, not the 0.70 fallback.

    0.70 means "no specific classifier matched". It sits 0.05 below the default
    threshold, so the verdict would flip for anyone running at <= 0.70 — and the
    report's score distribution would read it as a generic structural problem.
    """
    reason = AddedContentGate().evaluate(_d2echo_ECHO_SRC, _d2echo_ECHO_TGT).reason
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
    seed_job(ledger,
        "job_perm",
        _d2echo_doc([_d2echo_block("pdf_main#b_echo", 1, _d2echo_ECHO_SRC)]),
        target_lang="zh",
    )
    ledger.save_checkpoint(
        block_id="pdf_main#b_echo",
        status=BlockStatus.DRAFTED,
        target_text=_d2echo_ECHO_TGT,
        draft_text=_d2echo_ECHO_TGT,
    )
    _d2echo_run_quality_gate(tmp_path, ledger, "job_perm", qe_threshold=0.05)

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
        seed_job(ledger,
            "job_tok",
            _d2echo_doc([_d2echo_block("pdf_main#b_tok", 1, _d2echo_CLEAN_SRC)]),
            target_lang="zh",
        )
        ledger.save_checkpoint(
            block_id="pdf_main#b_tok",
            status=BlockStatus.DRAFTED,
            target_text=_d2echo_CLEAN_TGT,
            draft_text=_d2echo_CLEAN_TGT,
            error_flags=[flag],
        )
        _d2echo_run_quality_gate(tmp_path, ledger, "job_tok", qe_threshold=0.05)

        block = ledger.get_block("pdf_main#b_tok")
        assert block is not None
        assert block.status is BlockStatus.REPAIR_PENDING, marker
        assert any(marker in f for f in block.error_flags), block.error_flags
        ledger.close()


_r0918b_ECHO_REASON = "Target identical to source"

_r0918b_SRC_EN = "The quick brown fox jumps over the lazy dog near the river bank."


def _r0918b_fp(source_lang: str, target_lang: str) -> FastPassFilter:
    return FastPassFilter(source_lang=source_lang, target_lang=target_lang)


def test_verbatim_echo_is_rejected_for_same_script_pairs() -> None:
    """Before: ``en->fr``/``de->en``/``ja->zh`` echoes returned passed=True
    with "Flawless", and the quality gate released them as MTQE_PASSED because
    the only identity check lived in a scorer that branch never calls."""
    for src_lang, tgt_lang in (("en", "fr"), ("de", "en"), ("en", "en"), ("ja", "zh")):
        decision = _r0918b_fp(src_lang, tgt_lang).evaluate(
            _r0918b_SRC_EN, _r0918b_SRC_EN, block_type=BlockType.NARRATIVE
        )
        assert not decision.passed, f"{src_lang}->{tgt_lang} shipped an echo: {decision.reason}"
        assert _r0918b_ECHO_REASON in decision.reason


def test_real_translation_still_passes_and_echo_class_is_fabricated() -> None:
    ok = "Le renard brun rapide saute par-dessus le chien paresseux pres de la riviere."
    decision = _r0918b_fp("en", "fr").evaluate(_r0918b_SRC_EN, ok, block_type=BlockType.NARRATIVE)
    assert decision.passed, decision.reason
    # One rule, one band: ``score_pairs`` used to carry a second copy of the
    # identity check with its own length/format exemptions.
    echo = _r0918b_fp("en", "fr").evaluate(
        _r0918b_SRC_EN, _r0918b_SRC_EN, block_type=BlockType.NARRATIVE
    )
    assert HeuristicQERunner.score_from_decision_reason(echo.reason) == pytest.approx(0.15)
    assert asyncio.run(
        HeuristicQERunner().score_pairs([{"src": _r0918b_SRC_EN, "mt": _r0918b_SRC_EN}])
    ) == [pytest.approx(0.15)]


def test_echo_gate_exempts_the_blocks_that_keep_origin_by_contract() -> None:
    """Verbatim ships must not be routed into repair (which ignores
    ``skip_translate`` and used to leave them stale-FAILED)."""
    assert (
        _r0918b_fp("en", "fr").evaluate(_r0918b_SRC_EN, _r0918b_SRC_EN, skip_translate=True).passed
    )
    assert (
        _r0918b_fp("en", "fr")
        .evaluate(_r0918b_SRC_EN, _r0918b_SRC_EN, block_type=BlockType.CODE)
        .passed
    )
    assert (
        _r0918b_fp("en", "fr")
        .evaluate(_r0918b_SRC_EN, _r0918b_SRC_EN, block_type=BlockType.FORMULA)
        .passed
    )
    # Short and wordless blocks legitimately survive the trip unchanged.
    assert (
        _r0918b_fp("en", "fr").evaluate("Fig. 3", "Fig. 3", block_type=BlockType.NARRATIVE).passed
    )
    assert _r0918b_fp("en", "fr").evaluate(
        "1234 5678 9", "1234 5678 9", block_type=BlockType.HEADING
    )


def test_echo_marker_is_fatal_but_never_a_transient_failure() -> None:
    """The marker must survive any QE threshold *and* must not be re-queued on
    resume: ``is_transient_failure`` matches the lowercase ``untranslated:``
    lifecycle prefix, so a near-miss in casing would silently change the
    resume semantics of every echoed block."""
    flag = f"{_r0918b_ECHO_REASON}: the passage was not translated"
    assert flag in STRUCTURAL_DEFECT_MARKERS or any(m in flag for m in STRUCTURAL_DEFECT_MARKERS), (
        "echo must be fatal"
    )
    assert any(m in flag for m in CRITICAL_DEFECT_MARKERS), "an unrepaired echo is Critical"
    assert not is_transient_failure([flag])
