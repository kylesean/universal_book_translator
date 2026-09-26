"""Placeholder-integrity regressions: dropped, reordered, and echoed masked spans.

Verified defects pinned here, each with the user-visible failure it prevented
spelled out in the test docstring:

* the draft stage restored code/citation tokens with the weak ``unmask``
  variants, so a dropped span produced no flag and could ship;
* a corrupt restore was checkpointed as a finished draft, so the auto-pass path
  could release it;
* swapping two intact checksummed tokens left every report bucket empty, so a
  protected span could move position unnoticed;
* a model echo of the citation's original text alongside its intact token left
  the restore character-perfect and every bucket empty, so the reference
  shipped twice (review defect 10.4-3, fail-open restoration).
"""

import asyncio
from pathlib import Path

from tests.stage_ctx_factory import build_stage_ctx, inert_event
from tests.unit.ir_seed import SeedDoc, seed_job
from ubt.core.cleaners.citation_masker import CitationMasker
from ubt.core.cleaners.code_masker import CodeMasker
from ubt.core.cleaners.math_masker import MathMasker
from ubt.core.config import UBTConfig
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.stages.draft import run_draft_stage
from ubt.core.ir.models import (
    BlockStatus,
    BookManifest,
    ChapterMeta,
    FlowID,
    IRBlock,
)
from ubt.core.router.provider import MockModelProvider
from ubt.core.router.router import ModelRouter

_JOB_ID = "job_placeholder"
_BLOCK_ID = "ch01#b001"
_CITATION_SRC = "See [12] for the details."
_CODE_SRC = "Call `foo()` before the run."


def _draft(tmp_path: Path, source: str, response: str) -> SQLiteJobLedger:
    """Drive the real draft stage over one block with a canned model reply."""
    ledger = SQLiteJobLedger(tmp_path / "placeholder.sqlite")
    seed_job(
        ledger,
        _JOB_ID,
        SeedDoc(
            doc_id="placeholder_doc",
            source_path="/tmp/placeholder.epub",
            format_type="epub",
            metadata={},
            blocks=[
                IRBlock(
                    id=_BLOCK_ID,
                    flow_id=FlowID.MAIN_STORY,
                    spine_index=1,
                    source_text=source,
                )
            ],
        ),
        target_lang="zh",
    )
    router = ModelRouter(provider=MockModelProvider(default_response=response), draft_model="mock")
    config = UBTConfig(batch_limit=30, max_concurrency=2, batch_enabled=False, tm_enabled=False)

    async def _drain() -> None:
        async for _event in run_draft_stage(
            build_stage_ctx(
                tmp_path,
                ledger=ledger,
                job_id=_JOB_ID,
                manifest=BookManifest(
                    doc_id="placeholder_doc",
                    title="Placeholder",
                    source_path="/tmp/placeholder.epub",
                    source_lang="en",
                    target_lang="zh",
                    chapters=[ChapterMeta(chapter_id="ch01", title="One", spine_index=1)],
                    metadata={},
                ),
                profile_name="general",
                target_lang="zh",
                source_lang="en",
                router=router,
                code_masker=CodeMasker(),
                citation_masker=CitationMasker(),
                config=config,
                block_count=1,
                glossary_dicts=[],
                abbreviation_entries=[],
                concurrency_sem=asyncio.Semaphore(2),
                create_event=inert_event,
            ),
        ):
            pass

    asyncio.run(_drain())
    return ledger


def _swap_mask_tokens(masked: str, mapping: dict[str, str]) -> str:
    """Transpose the first two masked spans, checksums left intact."""
    tokens = list(mapping)
    assert len(tokens) >= 2, masked
    return (
        masked.replace(tokens[0], "\x00").replace(tokens[1], tokens[0]).replace("\x00", tokens[1])
    )


def test_dropped_code_token_is_flagged_and_cannot_auto_pass(tmp_path: Path) -> None:
    """A swallowed inline-code token must be flagged, never shipped as a draft.

    User-visible failure prevented: the model dropped an inline code span, the
    weak ``unmask`` path returned a bare string so no flag was recorded,
    FastPass saw short-but-plausible prose, and the released translation had
    silently lost the identifier.
    """
    masked, mapping = CodeMasker().mask(_CODE_SRC)
    token = next(iter(mapping))
    dropped = masked.replace(token, "")
    report = CodeMasker().unmask_checked(dropped, mapping)
    assert not report.clean and report.missing == [1], report

    ledger = _draft(tmp_path, _CODE_SRC, response=f"调用前{dropped}运行。")
    block = ledger.get_block(_BLOCK_ID)
    assert block is not None
    assert any("code_token_corrupt" in flag for flag in block.error_flags), block.error_flags
    assert block.status is BlockStatus.REPAIR_PENDING, (
        "a draft that lost a protected code span was checkpointed as a finished draft"
    )
    # The quality gate releases DRAFTED blocks only, so the corrupt draft is
    # unreachable by the auto-pass path.
    assert ledger.fetch_blocks_by_status(_JOB_ID, BlockStatus.DRAFTED) == []
    ledger.close()


def test_dropped_citation_token_is_flagged_and_cannot_auto_pass(tmp_path: Path) -> None:
    """A swallowed citation token must be flagged, never shipped as a draft.

    User-visible failure prevented: the model deleted a bracketed reference, the
    weak ``unmask`` path recorded nothing, and the shipped text dropped the
    citation instead of routing the block to repair.
    """
    masked, mapping = CitationMasker().mask(_CITATION_SRC)
    token = next(iter(mapping))
    dropped = masked.replace(token, "")
    report = CitationMasker().unmask_checked(dropped, mapping)
    assert not report.clean and report.missing == [1], report

    ledger = _draft(tmp_path, _CITATION_SRC, response=f"详见{dropped}的说明。")
    block = ledger.get_block(_BLOCK_ID)
    assert block is not None
    assert any("cite_token_corrupt" in flag for flag in block.error_flags), block.error_flags
    assert block.status is BlockStatus.REPAIR_PENDING, (
        "a draft that lost a protected citation was checkpointed as a finished draft"
    )
    assert ledger.fetch_blocks_by_status(_JOB_ID, BlockStatus.DRAFTED) == []
    ledger.close()


def test_swapped_masked_spans_report_reordered_and_fail_clean() -> None:
    """A transposed pair of intact spans must not read as a clean restore.

    User-visible failure prevented: a model that swaps two protected spans
    (two citations, two identifiers, two formulas) keeps every checksum valid,
    so ``missing``/``mismatched``/``mutated`` all stayed empty and the block
    auto-passed with its protected content in the wrong places.
    """
    for masker, source in (
        (MathMasker(), "A $x$ plus $y$."),
        (CodeMasker(), "Call `foo()` then `bar()`."),
        (CitationMasker(), "See [12] and [14]."),
    ):
        masked, mapping = masker.mask(source)
        report = masker.unmask_checked(_swap_mask_tokens(masked, mapping), mapping)
        name = type(masker).__name__
        assert report.reordered == [1, 2], (name, report)
        assert not report.clean, name
        assert report.missing == [] and report.mismatched == [] and report.mutated == [], report


def test_reordered_spans_reach_the_block_error_flags(tmp_path: Path) -> None:
    """The reorder verdict must reach the ledger and route the block to repair.

    User-visible failure prevented: two formulas traded places during drafting
    and the block was persisted as a clean DRAFTED block, so the swapped
    formulas shipped with no flag and no repair attempt.
    """
    source = "A $x$ plus $y$."
    masked, mapping = MathMasker().mask(source)
    ledger = _draft(tmp_path, source, response=f"甲 {_swap_mask_tokens(masked, mapping)} 乙。")
    block = ledger.get_block(_BLOCK_ID)
    assert block is not None
    assert any(
        "math_token_corrupt" in flag and "reordered=[1, 2]" in flag for flag in block.error_flags
    ), block.error_flags
    assert block.status is BlockStatus.REPAIR_PENDING
    assert ledger.fetch_blocks_by_status(_JOB_ID, BlockStatus.DRAFTED) == []
    ledger.close()


def test_find_reordered_tolerates_unknown_indices() -> None:
    """Indices the mapping never issued are skipped, not a KeyError.

    User-visible failure prevented: the inversion scan indexes ``rank[idx]``
    directly, so a future masker that passes an unfiltered observed list (one
    stray token-shaped reference) would crash the restore path instead of
    degrading to a partial order check.
    """
    from ubt.core.cleaners.mask_tokens import find_reordered

    assert find_reordered([1, 2], [1, 99, 2]) == []
    assert find_reordered([1, 2], [2, 99, 1]) == [1, 2]  # inversion still detected
    assert find_reordered([], [1, 2, 3]) == []


def test_citation_echoed_alongside_token_is_flagged_duplicated() -> None:
    """A literal '[12]' emitted next to its intact token must not pass as clean.

    User-visible failure prevented (review 10.4-3): masking replaces every
    bracketed citation, so a draft that contains ``[12]`` as free text *plus*
    its intact token restored the citation twice — the reference shipped
    duplicated in the published chapter. The restore is character-perfect, so
    ``missing``/``mismatched``/``mutated``/``reordered`` all stayed empty and
    ``clean`` was True (fail-open).
    """
    masked, mapping = CitationMasker().mask("See [12] for the details.")
    token = next(iter(mapping))
    echoed = masked.replace(token, f"{token}（即 [12]）")
    report = CitationMasker().unmask_checked(echoed, mapping)
    assert report.duplicated == [1], report
    assert not report.clean
    # The other buckets stay empty: only the duplication check sees this.
    assert (
        report.missing == []
        and report.mismatched == []
        and report.mutated == []
        and report.reordered == []
    ), report


def test_multi_token_citation_echo_reported_on_every_issuing_index() -> None:
    """Echoing both masked citations flags every index that issued them."""
    masked, mapping = CitationMasker().mask("See [12] and [14].")
    echoed = masked + "（参见 [12] 与 [14]）"  # stray copies of both citations
    report = CitationMasker().unmask_checked(echoed, mapping)
    assert report.duplicated == [1, 2], report
    assert not report.clean
    # The faithful single-occurrence draft stays clean — no false positive.
    faithful = CitationMasker().unmask_checked(masked, mapping)
    assert faithful.clean and faithful.duplicated == []


def test_duplicated_citation_flags_reach_the_block_error_flags(tmp_path: Path) -> None:
    """The duplicated verdict must reach the ledger and route the block to repair.

    User-visible failure prevented: the model echoed the citation as free text
    next to its token and the block was persisted as a clean DRAFTED block, so
    the duplicated reference shipped with no flag and no repair attempt.
    """
    source = "See [12] for the details."
    masked, mapping = CitationMasker().mask(source)
    token = next(iter(mapping))
    response = f"详见 {token}（即 [12]）的说明。"
    ledger = _draft(tmp_path, source, response=response)
    block = ledger.get_block(_BLOCK_ID)
    assert block is not None
    assert any(
        "cite_token_corrupt" in flag and "duplicated=[1]" in flag for flag in block.error_flags
    ), block.error_flags
    assert block.status is BlockStatus.REPAIR_PENDING
    assert ledger.fetch_blocks_by_status(_JOB_ID, BlockStatus.DRAFTED) == []
    ledger.close()
