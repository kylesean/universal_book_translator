"""The one test that spends real provider budget (nightly smoke job).

Why this file exists (inding B5/D1): every other test that exercises the
draft path drives ``MockModelProvider``, and the nightly "Real-provider smoke"
job (since retired 2026-09-26; recover it with
``git log -- .github/workflows/nightly-smoke.yml``) selected tests with
``-k "smoke or real"`` — which matched only ``test_docling_real_paper.py``
(matched on the module name containing "real"), a file with zero provider
references that parses a PDF. So the job could not verify a provider even in
principle, and its ``grep -c PASSED`` gate could never be satisfied by
``pytest -q`` output either.

Deliberately tiny (one or two calls, cents at most) and deliberately lenient: it
asserts invariants that must hold for ANY competent provider. The structural
*quality floor* (the golden review's D1, once deferred) is covered by
``test_live_draft_meets_structural_quality_gate`` below: the draft must clear the
pipeline's own zero-token FastPassFilter gate. It self-skips without a real
credential, so a local run stays free.
"""

from __future__ import annotations

import pytest

from tests.integration._live_helpers import requires_live_llm
from ubt.core.config import UBTConfig
from ubt.core.ir.models import BlockStatus, BlockType, FlowID, IRBlock
from ubt.core.qe.fast_pass import FastPassFilter
from ubt.core.router.provider import OpenAICompatibleProvider
from ubt.core.router.router import ModelRouter
from ubt.core.validators.consistency import NumericConsistencyValidator

pytestmark = [pytest.mark.slow, pytest.mark.network]

# Two numbers and a masked-span-free sentence: enough to catch an echo, a
# dropped number, or a leaked placeholder without needing a golden reference.
_SOURCE = "The gate oxide thickness is 200 nanometers; the channel length is 45 nanometers."


def _smoke_block() -> IRBlock:
    return IRBlock(
        id="smoke-0001",
        flow_id=FlowID.MAIN_STORY,
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        source_text=_SOURCE,
        status=BlockStatus.PENDING,
    )


@pytest.mark.asyncio
@requires_live_llm
async def test_live_draft_returns_usable_translation_and_records_usage(
    live_llm_env: None,
) -> None:
    """One real draft call: non-empty output, numbers kept, usage accounted.

    The usage assertion also guards the token-accounting regression live: the
    router's ``usage_totals`` accessor used to read a ``@property`` through a
    ``callable()`` guard and so returned ``{}`` for the only production
    provider, which pinned reported cost at ``$0.00`` for every job.
    """
    config = UBTConfig.from_env()
    provider = OpenAICompatibleProvider(
        api_key=config.api_key,
        base_url=config.base_url,
        default_model=config.draft_model,
        api_mode=config.api_mode,
    )
    router = ModelRouter(
        provider=provider,
        draft_model=config.draft_model,
        repair_model=config.repair_model,
    )
    try:
        translated = await router.draft(
            block=_smoke_block(),
            target_lang="zh",
            source_lang="en",
        )
    finally:
        await provider.aclose()

    assert translated.strip(), "live provider returned an empty draft"

    # A leaked opaque placeholder means the mask/unmask round trip did not close.
    assert "⟦" not in translated and "⟧" not in translated, (
        f"placeholder residue in live output: {translated!r}"
    )

    # Prompt scaffolding must not be shipped as the translation.
    for leaked in ("<translation>", "</translation>", "<issues>"):
        assert leaked not in translated, f"prompt scaffold leaked into output: {translated!r}"

    # Numeric fidelity, using the production validator (which also normalises
    # CJK numerals, so a fluent '两百纳米' rendering passes rather than flakes).
    numeric = NumericConsistencyValidator().validate(_SOURCE, translated)
    assert numeric.is_valid, f"numbers not preserved: {numeric.message}"

    # Usage must be observable, or the job cannot report what it spent.
    usage = router.usage_totals_by_model()
    assert usage, "provider reported no usage; cost/cache accounting is blind"
    assert sum(int(t.get("prompt_tokens", 0)) for t in usage.values()) > 0


# A harder golden than _SOURCE: several figures to track, an inline formula the
# math masker must round-trip intact, and a domain term. Long enough that an
# empty/echo/truncated/garbled-length draft trips the structural gate.
_QUALITY_SOURCE = (
    "The thermodynamic relation $E = m c^2$ implies that a 3.5 kg sample "
    "releases roughly 3.15e17 joules, while a 12 gram catalyst lowers the "
    "activation energy by 47 kJ per mole."
)


def _quality_block() -> IRBlock:
    return IRBlock(
        id="smoke-quality-0001",
        flow_id=FlowID.MAIN_STORY,
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        source_text=_QUALITY_SOURCE,
        status=BlockStatus.PENDING,
    )


@pytest.mark.asyncio
@requires_live_llm
async def test_live_draft_meets_structural_quality_gate(live_llm_env: None) -> None:
    """Real-model quality floor via the muse-spark profile (closes D1).

    Not a paraphrase-flaky BLEU/COMET number: the draft of the harder golden
    above must (a) come back in the *target* script and (b) clear the *same*
    zero-token ``FastPassFilter`` structural gate the pipeline auto-passes
    blocks on. So it fails only on a genuine regression — dropped figures, a
    corrupted masked formula, a source echo, or exploded length — and not on an
    acceptable rewording. Runs in the nightly smoke job (``UBT_SMOKE_API_KEY``);
    self-skips otherwise, so it never spends budget in the merge matrix.
    """
    config = UBTConfig.from_env()
    provider = OpenAICompatibleProvider(
        api_key=config.api_key,
        base_url=config.base_url,
        default_model=config.draft_model,
        api_mode=config.api_mode,
    )
    router = ModelRouter(
        provider=provider,
        draft_model=config.draft_model,
        repair_model=config.repair_model,
    )
    try:
        translated = await router.draft(
            block=_quality_block(),
            target_lang="zh",
            source_lang="en",
        )
    finally:
        await provider.aclose()

    # (a) Actually translated into Chinese — an echoed English paragraph is a
    # quality regression, not a stylistic variant.
    assert sum("一" <= c <= "鿿" for c in translated) >= 5, (
        f"draft not in target script (possible echo): {translated!r}"
    )

    # (b) Passes the production structural auto-pass gate.
    decision = FastPassFilter(source_lang="en", target_lang="zh").validate_structural_invariants(
        _QUALITY_SOURCE, translated, block_type=BlockType.NARRATIVE
    )
    assert decision.passed, f"live draft failed the structural quality gate: {decision.reason}"
