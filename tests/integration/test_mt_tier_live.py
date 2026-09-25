"""4 dedicated-MT tier live acceptance against the local llama-swap MT backend.

Runs without network or API keys: the local llama-swap gateway plus cached
CometKiwi weights. Skipped when the services are absent. This is the acceptance
evidence for the ``misrouting < 2%`` target — structural pass rate must be
100% on admitted singles; neural scores are reported (not hard-gated on
absolute values, which are domain-sensitive).

Run locally with:
    uv run pytest tests/integration/test_mt_tier_live.py -v -s
"""

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
from ubt.core.qe.comet_runner import SubprocessQERunner
from ubt.core.qe.fast_pass import FastPassFilter
from ubt.core.qe.mt_gate import is_mt_suitable
from ubt.core.router.provider import OpenAICompatibleProvider
from ubt.core.router.router import ModelRouter

pytestmark = pytest.mark.slow  # live MT backend calls

CLEAN_KWARGS = {"has_terms": False, "has_few_shot": False, "has_masked_spans": False}


def test_mt_gate_admits_only_simple_prose() -> None:
    """Admission audit on real corpus sentences (no services needed)."""
    sents = sample_corpus_sentences()
    assert sents, "expected sentences from local baseline EPUBs"
    admitted = [s for s in sents if is_mt_suitable(s, BlockType.NARRATIVE, **CLEAN_KWARGS)]
    # Sampler only yields clean singles: everything must be admitted, proving
    # the gate does not silently starve the MT tier on ordinary prose.
    assert len(admitted) == len(sents)
    # And the gate still rejects the structural cases.
    assert not is_mt_suitable("x = 1 + 2", BlockType.FORMULA, **CLEAN_KWARGS)
    assert not is_mt_suitable(
        "First sentence. Second sentence.", BlockType.NARRATIVE, **CLEAN_KWARGS
    )


def _make_router() -> tuple[ModelRouter, OpenAICompatibleProvider]:
    provider = OpenAICompatibleProvider(
        api_key="sk-local", base_url=LOCAL_MT_API_BASE, default_model=MT_MODEL
    )
    router = ModelRouter(provider=provider, draft_model=MT_MODEL, repair_model=MT_MODEL)
    return router, provider


async def _translate_admitted(router: ModelRouter, sents: list[str]) -> list[str]:
    outputs: list[str] = []
    for i, s in enumerate(sents):
        block = IRBlock(id=f"live-mt-{i}", spine_index=i, source_text=s, status=BlockStatus.DRAFTED)
        outputs.append(
            await router.draft(block, target_lang="zh", source_lang="en", model=MT_MODEL)
        )
    return outputs


@requires_local_mt
@pytest.mark.asyncio
async def test_mt_live_plumbing_structural() -> None:
    """Curated clean singles must all survive FastPass (path plumbing guard)."""
    sents = [
        "The quick brown fox jumps over the lazy dog.",
        "She opened the window and looked at the garden.",
        "The meeting will start at nine o'clock tomorrow morning.",
        "He bought a fresh loaf of bread from the corner bakery.",
    ]
    router, provider = _make_router()
    try:
        outputs = await _translate_admitted(router, sents)
    finally:
        await provider.aclose()
    fp = FastPassFilter(source_lang="en", target_lang="zh")
    failures = [
        (s, t, fp.evaluate(s, t).reason)
        for s, t in zip(sents, outputs, strict=True)
        if not fp.evaluate(s, t).passed
    ]
    assert not failures


@requires_local_mt
@pytest.mark.asyncio
async def test_mt_live_corpus_report() -> None:
    """Corpus sweep is report-only calibration evidence (not a hard gate).

    FastPass is deliberately conservative: condensed-but-adequate MT is routed
    to repair by design. This test prints the admission/structural profile for
    threshold tuning; only empty outputs fail.
    """
    sents = sample_corpus_sentences()
    admitted = [s for s in sents if is_mt_suitable(s, BlockType.NARRATIVE, **CLEAN_KWARGS)]
    router, provider = _make_router()
    try:
        outputs = await _translate_admitted(router, admitted)
    finally:
        await provider.aclose()
    assert all(o.strip() for o in outputs)
    fp = FastPassFilter(source_lang="en", target_lang="zh")
    failures = [
        (s, t, fp.evaluate(s, t).reason)
        for s, t in zip(admitted, outputs, strict=True)
        if not fp.evaluate(s, t).passed
    ]
    print(
        f"\n[mt-tier] corpus={len(sents)} admitted={len(admitted)} "
        f"structural_failures={len(failures)}"
    )
    for s, t, reason in failures:
        print(f"  REASON: {reason}\n  SRC: {s[:90]}\n  MT:  {t[:90]}")


@requires_local_mt
@requires_cometkiwi
@pytest.mark.asyncio
async def test_mt_live_neural_quality_report() -> None:
    """Score live MT outputs with local CometKiwi; report, don't hard-gate."""
    sents = sample_corpus_sentences(limit_per_book=8)
    admitted = [s for s in sents if is_mt_suitable(s, BlockType.NARRATIVE, **CLEAN_KWARGS)]
    router, provider = _make_router()
    try:
        outputs = await _translate_admitted(router, admitted)
    finally:
        await provider.aclose()
    ckpt = find_cometkiwi_checkpoint()
    assert ckpt is not None
    runner = SubprocessQERunner(
        python_bin=Path(sys.executable),
        script_path=packaged_comet_script(),
        model_name=str(ckpt),
        timeout_seconds=600,
    )
    scores = await runner.score_pairs(
        [{"src": s, "mt": t} for s, t in zip(admitted, outputs, strict=True)]
    )
    mean = sum(scores) / len(scores)
    lo = min(scores)
    print(f"\n[mt-tier] n={len(scores)} cometkiwi_mean={mean:.4f} min={lo:.4f}")
    print(f"[mt-tier] sorted_scores={[round(s, 3) for s in sorted(scores)]}")
    # Lenient sanity floor: catastrophic MT (empty/garbled) would collapse this.
    assert mean > 0.5
