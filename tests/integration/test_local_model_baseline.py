"""Zero-cost real-model baseline: a FIXED EN→ZH corpus through the real MT tier.

Why this file exists (review D1/§8-19): every other test that exercises the draft
path drives ``MockModelProvider``, so the project's core claim — that a real model
produces number-preserving, structurally clean Chinese — had no automated
evidence at all. The nightly "real-provider smoke" was supposed to cover it but
selected only Docling PDF-parsing tests (see ``test_live_provider_smoke.py``).

This runs a real model over a fixed corpus and hard-asserts invariants that must
hold for ANY competent translation. Neural quality scores are NOT computed here:
the reference-free CometKiwi subprocess did not complete within its 300s budget in
this environment (the same reason moved that case out of the unit
suite), so paying it again would recreate the cost F8 removed. Neural profiles
stay where they already are — ``tests/integration/test_qe_calibration.py``, run
by the nightly ``live-local`` job — and remain uncalibrated
(``docs/guides/evaluation-and-comparison-guide.md`` still lists L3 judge calibration as
待跑), so nothing gates on them.

Zero API cost — the local llama-swap MT backend plus the cached CometKiwi
checkpoint — and self-skips when either is absent, so a bare machine stays
green. The nightly ``live-local`` job runs this file as a hard gate.

The probe targets the llama-swap gateway, not the Ollama daemon that used to
serve this model: Ollama was retired 2026-09-23 (unit disabled, weights
deleted), so a ``:11434`` probe would have skipped this gate forever while
reporting green.
"""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from tests.integration._live_helpers import LOCAL_MT_API_BASE, MT_MODEL, requires_local_mt
from ubt.core.ir.models import BlockStatus, BlockType, FlowID, IRBlock
from ubt.core.qe.fast_pass import FastPassFilter
from ubt.core.router.provider import OpenAICompatibleProvider
from ubt.core.router.router import ModelRouter
from ubt.core.validators.consistency import NumericConsistencyValidator

pytestmark = pytest.mark.slow  # spawns the local model

# A FIXED corpus, not a sampled one: a baseline is only a regression signal if the
# inputs are identical across runs. It deliberately covers the failure classes the
# Review found by inspection — plain numbers (A4), a year, a decimal+unit, a
# sentence with no numbers at all, and a technical term — so a regression in the
# numeric gate or the mask/restore round trip shows up here.
BASELINE_CORPUS: tuple[str, ...] = (
    "The gate oxide thickness is 2 nanometers.",
    "The channel length was reduced to 45 nanometers in 2019.",
    "The device operates at 1.2 volts and consumes 30 milliwatts.",
    "Subthreshold swing degrades as the channel length shrinks.",
    "This chapter reviews the three main fabrication steps.",
    "Short-channel effects were first reported in the 1980s.",
)

_CJK_RANGES = ((0x4E00, 0x9FFF), (0x3400, 0x4DBF))


def _has_cjk(text: str) -> bool:
    return any(start <= ord(ch) <= end for ch in text for start, end in _CJK_RANGES)


def _block(index: int, source: str) -> IRBlock:
    return IRBlock(
        id=f"baseline-{index:03d}",
        flow_id=FlowID.MAIN_STORY,
        spine_index=index,
        block_type=BlockType.NARRATIVE,
        source_text=source,
        status=BlockStatus.PENDING,
    )


def _router() -> tuple[ModelRouter, OpenAICompatibleProvider]:
    provider = OpenAICompatibleProvider(
        api_key=SecretStr("sk-local"),
        base_url=LOCAL_MT_API_BASE,
        default_model=MT_MODEL,
    )
    return ModelRouter(provider=provider, draft_model=MT_MODEL, repair_model=MT_MODEL), provider


@requires_local_mt
@pytest.mark.asyncio
async def test_local_model_baseline_structural_invariants() -> None:
    """Every fixed-corpus sentence must survive the real model + real gates.

    A regression here means the shipped path stopped producing trustworthy
    Chinese: an empty/echoed target, a leaked mask token or prompt scaffold, a
    dropped or invented number, or a structural FastPass rejection.
    """
    router, provider = _router()
    try:
        outputs = []
        for i, source in enumerate(BASELINE_CORPUS):
            outputs.append(
                await router.draft(
                    _block(i, source),
                    target_lang="zh",
                    source_lang="en",
                    model=MT_MODEL,
                )
            )
    finally:
        await provider.aclose()

    fast_pass = FastPassFilter(source_lang="en", target_lang="zh")
    numeric = NumericConsistencyValidator()
    failures: list[str] = []

    for source, target in zip(BASELINE_CORPUS, outputs, strict=True):
        if not target.strip():
            failures.append(f"empty target for {source!r}")
            continue
        if "⟦" in target or "⟧" in target:
            failures.append(f"placeholder residue for {source!r}: {target!r}")
        for leaked in ("<translation>", "</translation>", "<issues>", "###"):
            if leaked in target:
                failures.append(f"prompt scaffold {leaked!r} leaked for {source!r}: {target!r}")
        if not _has_cjk(target):
            failures.append(
                f"target is not Chinese (echo or wrong script) for {source!r}: {target!r}"
            )
        decision = fast_pass.evaluate(source, target)
        if not decision.passed:
            failures.append(f"FastPass rejected {source!r} -> {target!r}: {decision.reason}")
        if not numeric.validate(source, target).is_valid:
            failures.append(f"numeric fidelity failed for {source!r} -> {target!r}")

    assert not failures, "real-model baseline regressions:\n  " + "\n  ".join(failures)
