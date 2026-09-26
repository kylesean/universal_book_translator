"""L1 (Heuristic) vs L2 (CometKiwi) agreement + L3 gray-zone sizing.

Scores live TranslateGemma outputs with both runners and prints the
calibration evidence used to set ``qe_threshold`` (0.75) and the L3 gray
zone [0.7, 0.8): agreement rate, danger quadrant (L1-pass/L2-low), and the
share of blocks that would actually hit the LLM judge. Reports only —
absolute neural values are domain-sensitive and must not be hard-gated.

Run locally with:
    uv run pytest tests/integration/test_qe_calibration.py -v -s
"""

import json
import sys
from pathlib import Path

import pytest

from tests.integration._live_helpers import (
    LOCAL_MT_API_BASE,
    MT_MODEL,
    find_cometkiwi_checkpoint,
    requires_cometkiwi,
    requires_local_mt,
    sample_corpus_sentences,
)
from ubt.core.config import packaged_comet_script
from ubt.core.ir.models import BlockStatus, BlockType, IRBlock
from ubt.core.qe.comet_runner import HeuristicQERunner, SubprocessQERunner
from ubt.core.qe.mt_gate import is_mt_suitable
from ubt.core.router.provider import OpenAICompatibleProvider
from ubt.core.router.router import ModelRouter

pytestmark = pytest.mark.slow  # live QE/COMET backend calls

CLEAN_KWARGS = {"has_terms": False, "has_few_shot": False, "has_masked_spans": False}
GRAY_LOW, GRAY_HIGH = 0.7, 0.8


@requires_local_mt
@requires_cometkiwi
@pytest.mark.asyncio
async def test_qe_l1_l2_agreement_report(tmp_path: Path) -> None:
    sents = sample_corpus_sentences(limit_per_book=8)
    admitted = [s for s in sents if is_mt_suitable(s, BlockType.NARRATIVE, **CLEAN_KWARGS)]
    assert admitted

    provider = OpenAICompatibleProvider(
        api_key="sk-local", base_url=LOCAL_MT_API_BASE, default_model=MT_MODEL
    )
    router = ModelRouter(provider=provider, draft_model=MT_MODEL, repair_model=MT_MODEL)
    try:
        outputs: list[str] = []
        for i, s in enumerate(admitted):
            block = IRBlock(id=f"cal-{i}", spine_index=i, source_text=s, status=BlockStatus.DRAFTED)
            outputs.append(
                await router.draft(block, target_lang="zh", source_lang="en", model=MT_MODEL)
            )
    finally:
        await provider.aclose()

    heuristic = HeuristicQERunner(target_lang="zh", source_lang="en")
    l1 = await heuristic.score_pairs(
        [{"src": s, "mt": t} for s, t in zip(admitted, outputs, strict=True)]
    )

    ckpt = find_cometkiwi_checkpoint()
    assert ckpt is not None
    l2_runner = SubprocessQERunner(
        python_bin=Path(sys.executable),
        script_path=packaged_comet_script(),
        model_name=str(ckpt),
        timeout_seconds=600,
    )
    try:
        l2 = await l2_runner.score_pairs(
            [{"src": s, "mt": t} for s, t in zip(admitted, outputs, strict=True)]
        )
    finally:
        # The resident scorer is a real child process. Without this teardown
        # its transport outlives the event loop, and GC then raises
        # "Event loop is closed" during a later test (2026-09 full-suite
        # PytestUnraisableExceptionWarning), while the worker holds its fds.
        await l2_runner.aclose()

    danger = sum(1 for a, b in zip(l1, l2, strict=True) if a >= 0.75 and b < 0.5)
    gray = sum(1 for b in l2 if GRAY_LOW <= b < GRAY_HIGH)
    mean_abs_diff = sum(abs(a - b) for a, b in zip(l1, l2, strict=True)) / len(l1)
    buckets: dict[str, int] = {"<0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, ">=0.8": 0}
    for b in l2:
        buckets[
            "<0.4" if b < 0.4 else "0.4-0.6" if b < 0.6 else "0.6-0.8" if b < 0.8 else ">=0.8"
        ] += 1

    l2_mean = sum(l2) / len(l2)
    report = {
        "n": len(l1),
        "l1_pass_rate": round(sum(1 for a in l1 if a >= 0.75) / len(l1), 3),
        "l2_mean": round(l2_mean, 4),
        "mean_abs_diff_l1_l2": round(mean_abs_diff, 4),
        "danger_quadrant_l1pass_l2low": danger,
        "l3_gray_zone_share": round(gray / len(l2), 3),
        "l2_buckets": buckets,
    }
    (tmp_path / "qe_calibration.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\n[qe-calibration] {json.dumps(report, indent=1)}")

    # Sanity floors only: catastrophic MT or a dead QE path collapses these.
    assert l2_mean > 0.5
    assert danger == 0


@requires_cometkiwi
@pytest.mark.asyncio
async def test_subprocess_qe_runner_real_neural_cometkiwi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SubprocessQERunner against real CometKiwi weights (moved out of tests/unit).

    the case lived in the unit suite, gated only on a checkpoint
    file existing — and presence proves nothing about usability. On this machine
    the 2.2GB checkpoint exists yet the subprocess needs minutes, so the unit
    suite paid the full 300s timeout by default. It is a real-model integration
    case: the module-level ``slow`` marker keeps it out of ``-m "not slow"``,
    and the nightly ``live-local`` job runs this file with the real weights.
    """
    ckpt = find_cometkiwi_checkpoint()
    assert ckpt is not None  # requires_cometkiwi skips before this otherwise
    script_path = packaged_comet_script()
    runner = SubprocessQERunner(
        python_bin=Path(sys.executable),
        script_path=script_path,
        model_name=str(ckpt),
        # 60s is enough on an idle machine (~10s), but under a full-suite run
        # the torch subprocess competes for CPU and blows past it — use the
        # production default (300s) so the test is load-tolerant.
        timeout_seconds=300,
    )
    # Hide GPUs from the worker subprocess: the machine's GPU may be shared
    # with unrelated processes and CUDA OOM (exit 3) would flake the run.
    # The script falls back to gpus=0 (CPU) when no CUDA device is visible,
    # which still exercises the real neural path end-to-end.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    pairs = [
        {"src": "Hello world", "mt": "你好世界"},
        {"src": "A completely nonsensical translation string", "mt": "1234567890"},
    ]
    try:
        scores = await runner.score_pairs(pairs)
        assert len(scores) == 2
        assert scores[0] > 0.7
        assert scores[1] < scores[0]
    finally:
        # Reap the resident scorer; see test_qe_l1_l2_agreement_report.
        await runner.aclose()
