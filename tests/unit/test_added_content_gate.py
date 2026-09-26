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

from pathlib import Path

from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.stages.tm_writeback import tm_writeback_eligible
from ubt.core.ir.models import BlockStatus, BlockType, DocumentIR, FlowID, IRBlock
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
from ubt.core.qe.fast_pass import FastPassFilter

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
        ledger.init_job(
            "job_x",
            DocumentIR(
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
