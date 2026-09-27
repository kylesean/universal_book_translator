"""Guards for the shared QE-score population policy.

The report's ``score_metrics`` excludes skip/TM 1.0 placeholders while the
ledger's status-side ``AVG(mtqe_score)`` historically did not, so the two
surfaces printed different averages for the same job. Both sides must now go
through ubt.core.qe.score_policy; this test proves report, Python predicate
and SQL predicate agree on a placeholder-heavy sample — and that the naive
average (what the unfixed ledger computed) does not.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.reporter import build_quality_report
from ubt.core.ir.models import (
    BlockStatus,
    BlockType,
    BookManifest,
    ChapterIR,
    ChapterMeta,
    FlowID,
    IRBlock,
)
from ubt.core.qe.score_policy import (
    PLACEHOLDER_MTQE_SCORE,
    QE_SCORED_SQL,
    average_qe_scored,
    is_qe_scored,
    qe_scored_values,
)

pytestmark = pytest.mark.fast


def _placeholder_sample() -> list[IRBlock]:
    """One real pass, one real defect, two placeholders, one unscored block."""
    return [
        IRBlock(
            id="s01",
            flow_id=FlowID.MAIN_STORY,
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            source_text="Passing paragraph.",
            target_text="通过的段落。",
            status=BlockStatus.MTQE_PASSED,
            mtqe_score=0.92,
        ),
        IRBlock(
            id="s02",
            flow_id=FlowID.MAIN_STORY,
            spine_index=2,
            block_type=BlockType.NARRATIVE,
            source_text="Defective paragraph.",
            target_text="有缺陷的段落。",
            status=BlockStatus.FAILED,
            mtqe_score=0.30,
            error_flags=["html_tag_mismatch"],
        ),
        # Verbatim skip: 1.0 is a stamp, not a measurement.
        IRBlock(
            id="s03",
            flow_id=FlowID.MAIN_STORY,
            spine_index=3,
            block_type=BlockType.CODE,
            source_text="print(42)",
            target_text="print(42)",
            status=BlockStatus.MTQE_PASSED,
            mtqe_score=PLACEHOLDER_MTQE_SCORE,
            skip_translate=True,
        ),
        # TM exact hit: same stamp, but excluded by provenance (tm_hit), not by
        # the magic score value.
        IRBlock(
            id="s04",
            flow_id=FlowID.MAIN_STORY,
            spine_index=4,
            block_type=BlockType.NARRATIVE,
            source_text="Repeated sentence.",
            target_text="重复的句子。",
            status=BlockStatus.MTQE_PASSED,
            mtqe_score=PLACEHOLDER_MTQE_SCORE,
            tm_hit=True,
        ),
        # Never scored (FastPass-cleared style): NULL, excluded everywhere.
        IRBlock(
            id="s05",
            flow_id=FlowID.MAIN_STORY,
            spine_index=5,
            block_type=BlockType.NARRATIVE,
            source_text="Unscored paragraph.",
            target_text="未评分的段落。",
            status=BlockStatus.MTQE_PASSED,
            mtqe_score=None,
        ),
    ]


def test_predicate_excludes_placeholders_skips_and_unscored() -> None:
    blocks = _placeholder_sample()
    assert PLACEHOLDER_MTQE_SCORE == 1.0
    assert is_qe_scored(blocks[0]) is True
    assert is_qe_scored(blocks[1]) is True
    assert is_qe_scored(blocks[2]) is False  # skip_translate
    assert is_qe_scored(blocks[3]) is False  # 1.0 placeholder on a TM hit
    assert is_qe_scored(blocks[4]) is False  # mtqe_score is None
    assert qe_scored_values(blocks) == [0.30, 0.92]
    assert average_qe_scored(blocks) == 0.61


def test_status_and_report_averages_agree_on_a_placeholder_sample(tmp_path: Path) -> None:
    """The literal acceptance check: status口径 == report口径, placeholders in."""
    blocks = _placeholder_sample()
    manifest = BookManifest(
        doc_id="score_policy_doc",
        title="Score Policy Book",
        source_path="score_policy.md",
        chapters=[ChapterMeta(chapter_id="ch01", title="One", spine_index=1)],
    )
    ledger = SQLiteJobLedger(tmp_path / "score_policy.sqlite")
    ledger.init_job_from_manifest("job_score_policy", manifest)
    ledger.append_chapter(
        "job_score_policy",
        ChapterIR(
            doc_id=manifest.doc_id,
            chapter_id="ch01",
            title="One",
            spine_index=1,
            blocks=blocks,
        ),
    )
    for b in blocks:
        ledger.save_checkpoint(
            block_id=b.id,
            target_text=b.target_text or "",
            status=b.status,
            mtqe_score=b.mtqe_score,
            repair_rounds=b.repair_rounds,
            error_flags=b.error_flags,
            tm_hit=b.tm_hit,
        )

    report = build_quality_report(
        ledger=ledger,
        job_id="job_score_policy",
        manifest=manifest,
        output_path=tmp_path / "out.md",
    )

    # Report口径: only the two real scores (0.92 + 0.30) / 2.
    report_avg = report.score_metrics.avg_qe
    assert report_avg == 0.61
    # Status口径 through the shared policy, evaluated as SQL over the same
    # blocks table the ledger's AVG reads — must give the identical number.
    with ledger._get_conn() as conn:  # noqa: SLF001 — the SQL predicate is the API under test
        row = conn.execute(
            f"SELECT AVG(mtqe_score) AS avg FROM blocks "  # noqa: S608 — policy constant, not user input
            f"WHERE job_id = 'job_score_policy' AND {QE_SCORED_SQL}",
        ).fetchone()
    sql_avg = round(float(row["avg"]), 4)
    assert sql_avg == report_avg
    # Ledger's public stats and snapshot APIs must also agree.
    stats = ledger.get_job_stats("job_score_policy")
    assert stats["avg_qe_score"] == report_avg
    snap = ledger.get_job_snapshot("job_score_policy")
    assert snap is not None
    assert snap["avg_qe_score"] == report_avg
    # And the Python predicate over the same rows agrees as well.
    assert average_qe_scored(blocks) == report_avg
    assert report.avg_qe == report_avg  # top-level alias (issue 4) rides along

    # The unfiltered average the ledger used to ship is visibly wrong here:
    # two 1.0 placeholders drag a 0.61 book to 0.805.
    with ledger._get_conn() as conn:  # noqa: SLF001
        naive = conn.execute(
            "SELECT AVG(mtqe_score) AS avg FROM blocks WHERE job_id = 'job_score_policy'"
        ).fetchone()
    assert round(float(naive["avg"]), 4) == 0.805
    assert round(float(naive["avg"]), 4) != report_avg
    ledger.close()


def test_a_genuine_perfect_score_is_counted_not_mistaken_for_a_placeholder() -> None:
    """A real 1.0 (neural/LLM judge, or a repair boost) must not be dropped.

    The old value-based filter (``score != 1.0``) could not tell a genuine
    perfect score from the skip/TM stamp, so it silently removed every real
    1.0 from ``avg_qe`` / ``min_qe`` / the percentiles.
    """
    real_perfect = IRBlock(
        id="perfect",
        flow_id=FlowID.MAIN_STORY,
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        source_text="A flawless paragraph.",
        target_text="一段完美的译文。",
        status=BlockStatus.MTQE_PASSED,
        mtqe_score=1.0,
    )
    assert is_qe_scored(real_perfect) is True
    assert qe_scored_values([real_perfect]) == [1.0]
    assert average_qe_scored([real_perfect]) == 1.0


def test_empty_population_reports_zero_not_a_fabricated_score() -> None:
    blocks = _placeholder_sample()
    for b in blocks:
        b.mtqe_score = None
    assert qe_scored_values(blocks) == []
    assert average_qe_scored(blocks) == 0.0
