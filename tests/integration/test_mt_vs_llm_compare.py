"""MT vs LLM paired comparison.

Paired design: the same admitted singles are drafted by both tiers and scored
blind by local CometKiwi + Heuristic. Reports cost/quality deltas;
asserts are lenient sanity floors, the printed table is the evidence.

Run locally with:
    OPENCODE_SESSION_ID=<your-session-id> \
      uv run pytest tests/integration/test_mt_vs_llm_compare.py -v -s
"""

import json
import os
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
from ubt.core.config import UBTConfig, packaged_comet_script
from ubt.core.ir.models import BlockStatus, BlockType, IRBlock
from ubt.core.qe.comet_runner import HeuristicQERunner, SubprocessQERunner
from ubt.core.qe.mt_gate import is_mt_suitable
from ubt.core.router.provider import OpenAICompatibleProvider
from ubt.core.router.router import ModelRouter

pytestmark = pytest.mark.slow  # live local MT backend calls

CLEAN_KWARGS = {"has_terms": False, "has_few_shot": False, "has_masked_spans": False}

requires_zen = pytest.mark.skipif(
    not os.getenv("OPENCODE_SESSION_ID"),
    reason="set OPENCODE_SESSION_ID (the variable the opencode provider's "
    "extra_headers expands) to route Zen calls (see docs)",
)


async def _draft_all(
    router: ModelRouter, sents: list[str], model: str | None, prefix: str
) -> list[str]:
    outputs: list[str] = []
    for i, s in enumerate(sents):
        block = IRBlock(
            id=f"{prefix}-{i}", spine_index=i, source_text=s, status=BlockStatus.DRAFTED
        )
        outputs.append(
            await router.draft(
                block, target_lang="zh", source_lang="en", model=model or router.draft_model
            )
        )
    return outputs


@requires_local_mt
@requires_zen
@requires_cometkiwi
@pytest.mark.asyncio
async def test_mt_vs_llm_paired_report(tmp_path: Path) -> None:
    sents = sample_corpus_sentences(limit_per_book=4)
    admitted = [s for s in sents if is_mt_suitable(s, BlockType.NARRATIVE, **CLEAN_KWARGS)]
    assert admitted

    mt_provider = OpenAICompatibleProvider(
        api_key="sk-local", base_url=LOCAL_MT_API_BASE, default_model=MT_MODEL
    )
    mt_router = ModelRouter(provider=mt_provider, draft_model=MT_MODEL, repair_model=MT_MODEL)
    cfg = UBTConfig.from_env()
    llm_provider = OpenAICompatibleProvider(
        api_key=cfg.api_key.get_secret_value(),
        base_url=cfg.base_url,
        default_model=cfg.draft_model,
        api_mode=cfg.api_mode,
        extra_headers=cfg.extra_headers,
        reasoning_dialect=cfg.reasoning_dialect,
    )
    llm_router = ModelRouter(
        provider=llm_provider, draft_model=cfg.draft_model, repair_model=cfg.repair_model
    )
    try:
        mt_outs = await _draft_all(mt_router, admitted, MT_MODEL, "mt")
        llm_outs = await _draft_all(llm_router, admitted, None, "llm")
    finally:
        await mt_provider.aclose()
        await llm_provider.aclose()

    ckpt = find_cometkiwi_checkpoint()
    assert ckpt is not None
    kiwi = SubprocessQERunner(
        python_bin=Path(sys.executable),
        script_path=packaged_comet_script(),
        model_name=str(ckpt),
        timeout_seconds=600,
    )
    heuristic = HeuristicQERunner(target_lang="zh", source_lang="en")
    mt_scores = await kiwi.score_pairs(
        [{"src": s, "mt": t} for s, t in zip(admitted, mt_outs, strict=True)]
    )
    llm_scores = await kiwi.score_pairs(
        [{"src": s, "mt": t} for s, t in zip(admitted, llm_outs, strict=True)]
    )
    fp = heuristic.fast_pass if hasattr(heuristic, "fast_pass") else None
    assert fp is not None
    mt_fp = sum(1 for s, t in zip(admitted, mt_outs, strict=True) if fp.evaluate(s, t).passed)
    llm_fp = sum(1 for s, t in zip(admitted, llm_outs, strict=True) if fp.evaluate(s, t).passed)

    def mean(xs: list[float]) -> float:
        return round(sum(xs) / len(xs), 4) if xs else 0.0

    report = {
        "n": len(admitted),
        "mt_model": MT_MODEL,
        "llm_model": cfg.draft_model,
        "cometkiwi_mean": {"mt": mean(mt_scores), "llm": mean(llm_scores)},
        "fastpass_rate": {
            "mt": round(mt_fp / len(admitted), 3),
            "llm": round(llm_fp / len(admitted), 3),
        },
        "llm_usage": llm_router.usage_totals(),
    }
    (tmp_path / "mt_vs_llm.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\n[mt-vs-llm] {json.dumps(report, indent=1)}")

    assert mean(mt_scores) > 0.5
    assert mean(llm_scores) > 0.5
