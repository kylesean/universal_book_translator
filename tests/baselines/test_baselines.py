"""Golden baseline end-to-end regression tests and quality gate verification.

Baselines evaluated:
1. cognitive-psychology-ch03 (Textbook layout: multi-flow, math, code, sidebars)
2. dual-column-paper (Academic paper: LaTeX equations, algorithm code, tables)
3. standard-alice (Standard Ebooks EPUB with 45 inline illustrations)
4. call-of-the-wild (Long-form prose novel with streaming chapter cursor)
"""

import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from tests.mock_providers import TokenEchoMockProvider
from ubt.core.config import UBTConfig
from ubt.core.engine.events import EventType, TranslationProgressEvent
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.pipeline import PipelineOrchestrator
from ubt.core.ir.models import BlockStatus
from ubt.core.job_options import sidecar_path
from ubt.core.metrics import check_thresholds, compare_kpi_sets, load_kpis, save_metrics_report
from ubt.core.qe.comet_runner import MockQERunner
from ubt.core.router.rate_limiter import AdaptiveTokenBucket
from ubt.core.router.router import ModelRouter

# Deliberately NOT `slow`-marked: measured 2026-09-23 the whole tier is 5 tests /
# 2.27 s (slowest case 1.27 s) and spawns no subprocess. It is the only tier that
# proves the pipeline worked end to end, so it belongs in the DEFAULT loop. It is
# not in the `-m fast` per-edit loop, which collects tests/unit only.

#: Set to "1" to rewrite the checked-in KPI goldens instead of gating on them.
GOLDEN_UPDATE_ENV = "UBT_UPDATE_GOLDENS"


def assert_no_kpi_regression(output_path: Path, golden_path: Path) -> None:
    """Gate one deterministic baseline run against its checked-in KPI golden.

    The baseline runs use :class:`TokenEchoMockProvider` + MockQERunner, so the
    KPI artifact is stable across machines; a regression here means the pipeline
    changed, not that a model got noisier. (The double carries a structural
    fidelity contract — see ``tests/mock_providers.py`` — which is what makes
    that reading safe.) Regenerate deliberately with ``UBT_UPDATE_GOLDENS=1`` and
    review the diff: the rewrite itself runs the absolute bounds on the candidate
    before writing, and ``test_goldens_satisfy_absolute_bounds`` re-checks the
    checked-in goldens in the default/CI loop, so a recorded defect goes red
    rather than shipping.
    """
    metrics_path = sidecar_path(output_path, "metrics.json")
    assert metrics_path.exists(), f"missing KPI artifact: {metrics_path}"
    candidate = load_kpis(metrics_path)
    if os.environ.get(GOLDEN_UPDATE_ENV) == "1":
        # A re-record must not launder a defect: run the absolute bounds on the
        # candidate before writing, or ``UBT_UPDATE_GOLDENS=1`` would fold a
        # leaked translation / dropped placeholder straight into the baseline.
        candidate_bounds = check_thresholds(candidate.kpis)
        assert not candidate_bounds, (
            f"refusing to record {golden_path.name}: the candidate violates its own "
            "absolute bounds (a defect cannot be recorded, only fixed):\n"
            + "\n".join(candidate_bounds)
        )
        # Drop the volatile job block (tmp path, timestamp): the gate compares
        # KPIs, and a checked-in golden should not carry machine-specific state.
        save_metrics_report(candidate.model_copy(update={"job": {}}), golden_path)
        return
    assert golden_path.exists(), (
        f"missing golden {golden_path}; regenerate with {GOLDEN_UPDATE_ENV}=1"
    )
    golden = load_kpis(golden_path)
    # The absolute bounds run against the golden first, and that ordering is the
    # point: ``UBT_UPDATE_GOLDENS=1`` rewrites history, so a relative gate alone
    # would let one re-record fold a leaked translation or a dropped placeholder
    # into the new baseline. A defect cannot be recorded, only fixed.
    golden_bounds = check_thresholds(golden.kpis)
    assert not golden_bounds, (
        f"{golden_path.name} violates its own absolute bounds — re-recording "
        "cannot launder a defect:\n" + "\n".join(golden_bounds)
    )
    report = compare_kpi_sets(golden, candidate, strict_names=True)
    # strict_names: a KPI this pipeline emits that the golden lacks gates
    # nothing at all, so adding one must force a deliberate re-record.
    violations = list(report.violations)
    violations += [f"candidate: {v}" for v in check_thresholds(candidate.kpis)]
    assert not violations, "KPI regression:\n" + "\n".join(violations)


BASELINES_DIR = Path(__file__).resolve().parent


def _unthrottled_limiter() -> AdaptiveTokenBucket:
    """A limiter that never throttles an offline baseline run.

    Both buckets must be raised: the router charges each request its
    estimated *token* count (prompt + completion), so a high RPM alone still
    throttles every block against the default 100k-TPM bucket (~1.7k tokens/s,
    i.e. seconds per prompt on a glossary-heavy book).
    """
    return AdaptiveTokenBucket(
        initial_rpm=1_000_000,
        max_rpm=1_000_000,
        initial_tpm=1_000_000_000,
        max_tpm=1_000_000_000,
    )


@dataclass(frozen=True, slots=True)
class BaselineRun:
    """One completed offline translation: what it wrote and what it emitted."""

    output: Path
    events: list[TranslationProgressEvent]

    def report(self) -> dict[str, Any]:
        path = sidecar_path(self.output, "quality_report.json")
        assert path.exists(), f"missing quality report: {path}"
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return payload


async def _run_baseline(
    *,
    tmp_path: Path,
    provider: TokenEchoMockProvider,
    src: Path,
    out: Path,
    db_name: str,
    job_id: str,
    default_score: float,
    profile_name: str = "general",
) -> BaselineRun:
    """Translate one corpus end to end with the offline double.

    Four baselines differ only in corpus, glossary and the score MockQERunner
    starts from. Folding the rest in here is what stops a pipeline knob being
    set in one baseline and quietly left different in another; the streaming
    events come back so a test can assert on them. ``profile_name`` repeats the
    engine's own default rather than passing ``None``, which it does not accept.
    """
    config = UBTConfig(db_dir=tmp_path / db_name, rate_limit_rpm=100_000)
    router = ModelRouter(
        provider=provider,
        draft_model="mock-draft",
        repair_model="mock-repair",
        rate_limiter=_unthrottled_limiter(),
    )
    orchestrator = PipelineOrchestrator(
        config=config,
        router=router,
        qe_runner=MockQERunner(default_score=default_score),
    )
    events: list[TranslationProgressEvent] = []
    async for event in orchestrator.run(
        input_path=src,
        output_path=out,
        target_lang="zh",
        profile_name=profile_name,
        job_id=job_id,
    ):
        events.append(event)
    return BaselineRun(output=out, events=events)


@pytest.mark.asyncio
async def test_baseline_cognitive_psychology_ch03_end_to_end(tmp_path: Path) -> None:
    """Validate textbook layout: terminology consistency, formula/code masking, sidebar isolation."""
    src_file = BASELINES_DIR / "cognitive-psychology-ch03" / "cognitive_psychology_ch03.md"
    assert src_file.exists()
    out_file = tmp_path / "cognitive_psychology_bilingual.md"

    custom_responses = {
        "Working Memory": "工作记忆",
        "Phonological Loop": "语音回路",
        "Visuospatial Sketchpad": "视空间画板",
        "Central Executive": "中央执行系统",
        "Architecture of Working Memory": "工作记忆的架构体系",
        "Working Memory (WM) refers to": "工作记忆（Working Memory, WM）是指能够暂时保持和操纵信息的认知系统。",
        "Alan Baddeley and Graham Hitch proposed": "艾伦·巴德利和格雷厄姆·希区提出了工作记忆的多元模型，包含中央执行系统、语音回路和视空间画板。",
        "Sidebar 3.1": "边栏 3.1：工作记忆缺陷的临床神经心理学案例",
        "Patient H.M.": "1953年切除双侧内侧颞叶的患者 H.M. 表现出严重顺行性遗忘症，但语音回路功能依然完好。",
    }

    mock_provider = TokenEchoMockProvider(
        default_response="认知心理学通用翻译内容。",
        custom_responses=custom_responses,
    )

    run = await _run_baseline(
        tmp_path=tmp_path,
        provider=mock_provider,
        src=src_file,
        out=out_file,
        db_name="db_cogpsy",
        job_id="job_baseline_cogpsy",
        default_score=0.93,
        profile_name="textbook",
    )

    # 1. Verify output file exists
    assert out_file.exists()
    rendered_text = out_file.read_text(encoding="utf-8")

    # 2. Terminology handling is NOT asserted here: TokenEchoMockProvider echoes
    # the fixture's custom_responses verbatim, so a literal match would only prove
    # the fixture wired itself up. Deterministic glossary enforcement and drift
    # detection are covered directly in tests/unit/test_glossary_enforcer.py.
    assert rendered_text.strip()

    # 3. Verify math equation and code block preservation (skip_translate)
    assert "$$ d' = Z(\\text{Hit Rate}) - Z(\\text{False Alarm Rate}) $$" in rendered_text
    assert "def compute_signal_detection_sensitivity" in rendered_text
    assert "stats.norm.ppf(hit_rate)" in rendered_text

    # 4. Verify quality report
    rep = run.report()
    assert rep["summary"]["total_blocks"] >= 10
    assert rep["summary"]["completed_blocks"] == rep["summary"]["total_blocks"]
    assert rep["summary"]["failed_blocks"] == 0
    assert rep["summary"]["pass_rate"] == 1.0
    assert rep["score_metrics"]["avg_qe"] >= 0.70
    assert_no_kpi_regression(
        out_file, BASELINES_DIR / "cognitive-psychology-ch03" / "metrics.golden.json"
    )


@pytest.mark.asyncio
async def test_baseline_dual_column_paper_end_to_end(tmp_path: Path) -> None:
    """Validate academic paper: LaTeX math preservation, code listing, and table retention."""
    src_file = BASELINES_DIR / "dual-column-paper" / "dual_column_paper.md"
    assert src_file.exists()
    out_file = tmp_path / "paper_bilingual.md"

    run = await _run_baseline(
        tmp_path=tmp_path,
        provider=TokenEchoMockProvider(default_response="学术论文中文翻译对照内容。"),
        src=src_file,
        out=out_file,
        db_name="db_paper",
        job_id="job_baseline_paper",
        default_score=0.90,
        profile_name="paper",
    )

    assert out_file.exists()
    rendered = out_file.read_text(encoding="utf-8")

    # Math preservation
    assert "\\mathcal{D}" in rendered
    assert "\\hat{y}_i" in rendered

    # Code preservation
    assert "def compute_translation_loss" in rendered
    assert "math.log(max(p, 1e-12))" in rendered

    # Report verification. The partition is the contract, not a positivity
    # check: `completed_blocks > 0` stayed green while one block in the corpus
    # went missing, because "some blocks finished" is true for almost any
    # broken run.
    rep = run.report()
    summary = rep["summary"]
    assert (
        summary["completed_blocks"] + summary["blocked_human_blocks"] == summary["total_blocks"]
    ), "a block that is neither completed nor quarantined was lost"
    assert summary["failed_blocks"] == 0
    assert_no_kpi_regression(out_file, BASELINES_DIR / "dual-column-paper" / "metrics.golden.json")


@pytest.mark.asyncio
async def test_paid_drafts_and_billing_survive_reopening_the_ledger(tmp_path: Path) -> None:
    """A finished run's money state must live in the file, not in the writer.

    Every baseline gates the exported artifact and the quality report. None of
    them re-opened the ledger *by path* from a fresh connection, which is exactly
    how resume and the billing read-back work. So a checkpoint that only reached
    the flusher's buffer, or landed in another job's db file, would keep all four
    baselines green.
    """
    out_file = tmp_path / "paper_bilingual.md"
    run = await _run_baseline(
        tmp_path=tmp_path,
        provider=TokenEchoMockProvider(default_response="学术论文中文翻译对照内容。"),
        src=BASELINES_DIR / "dual-column-paper" / "dual_column_paper.md",
        out=out_file,
        db_name="db_paper",
        job_id="job_baseline_paper",
        default_score=0.90,
        profile_name="paper",
    )

    db_path = tmp_path / "db_paper" / "job_baseline_paper.sqlite"
    assert db_path.exists(), f"run wrote no ledger where resume looks: {db_path}"

    reopened = SQLiteJobLedger(db_path)
    try:
        blocks = reopened.get_all_blocks("job_baseline_paper")
        assert blocks, "the reopened ledger holds no blocks"
        summary = run.report()["summary"]
        assert len(blocks) == summary["total_blocks"], "ledger and report disagree on the corpus"
        for block in blocks:
            if block.status in (BlockStatus.MTQE_PASSED, BlockStatus.REPAIRED):
                assert block.target_text, f"{block.id} passed without carrying its paid draft"
        # Per-model usage is deliberately NOT asserted here: `usage.py` returns
        # early when the run spent nothing, and the offline double spends nothing,
        # so an empty bill is correct for this tier. Billing persistence belongs to
        # the live tier, where a real provider actually charges.
    finally:
        reopened.close()


@pytest.mark.asyncio
async def test_baseline_standard_alice_epub_e2e(tmp_path: Path) -> None:
    """Validate standard-alice EPUB: 20 chapters, 45 illustrations preserved, valid bilingual container."""
    src_epub = BASELINES_DIR / "standard-alice" / "standard-alice.epub"
    assert src_epub.exists()
    out_epub = tmp_path / "alice_bilingual.epub"

    run = await _run_baseline(
        tmp_path=tmp_path,
        provider=TokenEchoMockProvider(default_response="这是《爱丽丝梦游仙境》中英双语对照译文。"),
        src=src_epub,
        out=out_epub,
        db_name="db_alice",
        job_id="job_baseline_alice",
        default_score=0.91,
    )

    # 1. Verify file generation
    assert out_epub.exists()
    assert out_epub.stat().st_size > 5_000_000  # Full EPUB with images > 5MB

    # 2. Verify internal EPUB zip integrity
    with zipfile.ZipFile(out_epub) as zf:
        namelist = zf.namelist()
        assert "mimetype" in namelist
        assert any("META-INF" in n for n in namelist)
        # All illustrations preserved
        images = [
            n for n in namelist if n.startswith("images/") or n.endswith((".jpg", ".png", ".svg"))
        ]
        assert len(images) >= 40
        # XHTML chapter files present
        ch_files = [n for n in namelist if n.endswith((".xhtml", ".html"))]
        assert len(ch_files) >= 15

    # 3. Quality report verification
    rep = run.report()
    assert rep["book_title"] == "Alice’s Adventures in Wonderland"
    assert rep["summary"]["total_blocks"] == 885
    assert (
        rep["summary"]["completed_blocks"] + rep["summary"]["blocked_human_blocks"]
        == rep["summary"]["total_blocks"]
    )
    assert rep["summary"]["completed_blocks"] == 885
    assert rep["summary"]["blocked_human_blocks"] == 0
    assert rep["summary"]["failed_blocks"] == 0
    # 883→885: the two "Wow! wow! wow!" blocks used to be quarantined because
    # TokenEchoMockProvider's long filler could not fit three sentences under the
    # length cap, so the mock violated its own sentence-parity contract and the
    # omission gate fired on the fixture. The mock now falls back to its micro
    # filler before giving up; the whole novel delivers with nothing quarantined.
    # Gate wired 2026-09-23 and its golden recorded in the same round (8bef284).
    # The corpus used to be excused as `network`-marked, which was untrue (local
    # EPUB, offline double) — that false marker is why it ran undefended for a week.
    assert_no_kpi_regression(out_epub, BASELINES_DIR / "standard-alice" / "metrics.golden.json")


@pytest.mark.asyncio
async def test_baseline_call_of_the_wild_streaming_e2e(tmp_path: Path) -> None:
    """Validate call-of-the-wild novel: 408 blocks, cursor pagination, and distribution metrics."""
    src_epub = BASELINES_DIR / "call-of-the-wild" / "call_of_the_wild.epub"
    assert src_epub.exists()
    out_epub = tmp_path / "call_of_the_wild_bilingual.epub"

    run = await _run_baseline(
        tmp_path=tmp_path,
        provider=TokenEchoMockProvider(default_response="《野性的呼唤》中文双语对照段落。"),
        src=src_epub,
        out=out_epub,
        db_name="db_call_of_the_wild",
        job_id="job_baseline_call_of_the_wild",
        default_score=0.89,
    )

    # The streaming half the test is named for: progress has to arrive as the
    # draft works through the book and end by agreeing with the deliverable. A
    # counter that moved backwards would mean a resumed cursor double-counting
    # blocks, and a single draft batch for 408 blocks would mean the batching
    # path never ran.
    events = run.events
    assert events[0].event_type is EventType.JOB_STARTED
    assert [e.completed_blocks for e in events] == sorted(e.completed_blocks for e in events)
    draft_batches = [e for e in events if e.event_type is EventType.DRAFT_BATCH_COMPLETED]
    assert len(draft_batches) > 1, f"408 blocks drafted in a single batch of {len(events)} events"
    assert events[-1].event_type is EventType.EXPORT_COMPLETED
    assert events[-1].completed_blocks == events[-1].total_blocks == 408

    assert out_epub.exists()
    assert out_epub.stat().st_size > 100_000

    rep = run.report()
    assert rep["summary"]["total_blocks"] == 408
    # The partition is the invariant; the old 405/0/3 triple was arithmetic
    # memorised from a run nobody re-read. It recorded the three Project
    # Gutenberg license paragraphs ("PLEASE READ THIS BEFORE YOU DISTRIBUTE…",
    # "1.F.2. LIMITED WARRANTY…") being quarantined as near-verbatim echoes —
    # right when the gate could not tell a Chinese translation from an
    # untouched source. 15c0af2 fixed exactly that blind spot (a CJK-script
    # target is translated by definition), so those paragraphs now clear the
    # gate like any other prose and the human queue is empty. Asserting an empty
    # queue keeps the promise the 405 was reaching for — nothing in a 408-block
    # novel needs a human — without pinning a count that no longer describes
    # this corpus.
    assert rep["summary"]["completed_blocks"] == rep["summary"]["total_blocks"]
    assert rep["summary"]["failed_blocks"] == 0
    assert rep["summary"]["blocked_human_blocks"] == 0
    # Score metrics quantile consistency
    metrics = rep["score_metrics"]
    if metrics.get("scored_blocks", 0) == 0:
        # This baseline's mock QE runner scores nothing, so the ordering below
        # would be `0 <= 0 <= 0 <= 0 <= 0` — true for any implementation. Assert
        # the real invariant (an unscored run reports zeros) so the check can
        # actually fail.
        assert metrics["avg_qe"] == 0.0
        assert metrics["min_qe"] == 0.0 and metrics["max_qe"] == 0.0
    else:
        assert (
            metrics["min_qe"]
            <= metrics["p10_qe"]
            <= metrics["p50_qe"]
            <= metrics["p90_qe"]
            <= metrics["max_qe"]
        )
    assert_no_kpi_regression(out_epub, BASELINES_DIR / "call-of-the-wild" / "metrics.golden.json")


def test_token_overhead_and_fast_pass_calibration() -> None:
    """Bound the token budget using the *real* pipeline knobs, not copies.

    The point of the budget model is that a config change that would blow
    the ~1.5x ceiling (a bigger ``bottom_percentile`` repair share, more
    ``max_repair_rounds``, or enabling the LLM judge on a heuristic engine
    while ``rerank_k``>1) actually fails this test. Reading the constants
    from ``UBTConfig`` / ``RepairLoop`` instead of hard-coding them is what
    keeps it honest for `bottom_percentile`; the fast-pass ceiling (0.75), the
    decay tail (0.30) and the 1.5x budget ceiling remain literals, so those three
    still cannot go red when the product knobs drift.
    """
    from ubt.core.config import UBTConfig

    cfg = UBTConfig()
    total_blocks = 1000
    draft_tokens_per_block = 200
    total_draft_tokens = total_blocks * draft_tokens_per_block

    # Real repair-selection share: the pipeline flags this fraction of blocks
    # as suspicious and sends at most it to repair each round.
    repair_candidate_count = int(total_blocks * cfg.bottom_percentile)
    # The selection fraction must stay a subset of what the fast-pass gate
    # leaves un-passed; a bottom_percentile that exceeds that ceiling means
    # the model's "0-token fast pass absorbs the rest" premise is broken.
    fast_pass_clean_blocks = int(total_blocks * 0.75)
    remaining_blocks = total_blocks - fast_pass_clean_blocks
    assert repair_candidate_count <= remaining_blocks, (
        f"bottom_percentile={cfg.bottom_percentile} selects {repair_candidate_count} "
        f"repair blocks but only {remaining_blocks} are non-fast-pass"
    )

    # Round 1 repair: all candidates. Rounds 2..N: a decaying tail (empirically
    # ~30% of the prior round still fails after one fix).
    repair_tokens = repair_candidate_count * draft_tokens_per_block
    tail = repair_candidate_count
    for _ in range(max(cfg.max_repair_rounds - 1, 0)):
        tail = int(tail * 0.30)
        repair_tokens += tail * draft_tokens_per_block

    # Bible extraction: one amortized pass over TOC/headings.
    bible_overhead_tokens = int(total_draft_tokens * 0.03)

    # best-of-n rerank multiplies the draft-side sampling by rerank_k.
    draft_multiplier = 1.0
    if cfg.rerank_k > 1 and cfg.qe_engine != "heuristic":
        draft_multiplier = float(cfg.rerank_k)

    total_tokens_consumed = int(total_draft_tokens * draft_multiplier)
    total_tokens_consumed += repair_tokens + bible_overhead_tokens
    multiplier = total_tokens_consumed / total_draft_tokens

    assert multiplier <= 1.60, (
        f"Token multiplier {multiplier:.3f}x exceeds the 1.6x design ceiling; "
        f"bottom_percentile={cfg.bottom_percentile}, "
        f"max_repair_rounds={cfg.max_repair_rounds}, rerank_k={cfg.rerank_k}"
    )
    # A guard against the model silently collapsing to draft-only (a 1.0x
    # reading would mean repair/rerank knobs are inert, not that we are cheap).
    assert multiplier >= 1.0, "budget model produced an impossible sub-draft total"
