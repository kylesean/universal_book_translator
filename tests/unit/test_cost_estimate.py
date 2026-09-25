"""Pre-flight cost estimation: the number a user sees before spending anything."""

from typing import Any

import pytest

from ubt.core.engine.cost_estimate import (
    estimate_draft_cost,
    measure_prefix_tokens,
)
from ubt.core.ir.models import BlockType, FlowID, IRBlock
from ubt.core.router.provider import MockModelProvider
from ubt.core.router.router import ModelRouter

pytestmark = pytest.mark.fast


def _block(text: str, *, skip: bool = False, done: bool = False) -> IRBlock:
    return IRBlock(
        id=f"b{abs(hash(text)) % 10_000}",
        flow_id=FlowID.MAIN_STORY,
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        source_text=text,
        target_text="已译" if done else None,
        skip_translate=skip,
    )


@pytest.fixture
def router() -> Any:
    return ModelRouter(
        provider=MockModelProvider(), draft_model="deepseek-chat", repair_model="deepseek-chat"
    )


def test_prefix_is_measured_from_the_real_prompt(router: Any) -> None:
    """The scaffolding is not a constant: it follows the active prompt inputs."""
    plain = measure_prefix_tokens(router, target_lang="zh", source_lang="en")
    with_glossary = measure_prefix_tokens(
        router,
        target_lang="zh",
        source_lang="en",
        glossary_table="| term | 术语 |\n| --- | --- |\n" + "| router | 路由器 |\n" * 50,
    )
    assert plain is not None and with_glossary is not None
    assert plain > 0
    # A book bible rides in the prefix of *every* call, so ignoring it would
    # under-price a glossary-heavy run by that share on each block.
    assert with_glossary > plain


def test_only_untranslated_translatable_blocks_are_billed(router: Any) -> None:
    long_text = "A technical paragraph about metrology. " * 20
    blocks = [
        _block(long_text),
        _block(long_text, skip=True),  # verbatim cover chrome
        _block(long_text, done=True),  # already drafted by a previous run
    ]

    estimate = estimate_draft_cost(blocks, draft_model="deepseek-chat", prefix_tokens=1000)

    assert estimate.billable_blocks == 1
    assert estimate.prompt_tokens == 1000 + len(long_text) // 4
    assert estimate.completion_tokens == (len(long_text) // 4 * 3) // 2


def test_priced_model_yields_a_range_and_unpriced_stays_unknown() -> None:
    blocks = [_block("Some source paragraph with a reasonable length. " * 10)]

    priced = estimate_draft_cost(blocks, draft_model="deepseek-chat", prefix_tokens=1000)
    unpriced = estimate_draft_cost(blocks, draft_model="brand-new-model-9000", prefix_tokens=1000)

    assert priced.cost_usd_uncached is not None
    assert priced.cost_usd_cached is not None
    # Cache hits can only make the run cheaper, never dearer.
    assert priced.cost_usd_cached <= priced.cost_usd_uncached
    assert unpriced.cost_usd_uncached is None
    assert unpriced.money_is_unknown is True
    assert priced.money_is_unknown is False
    assert "unknown" in unpriced.describe()
    assert "Draft calls only" in priced.describe()


def test_zero_work_estimates_zero(router: Any) -> None:
    estimate = estimate_draft_cost([], draft_model="deepseek-chat", prefix_tokens=1000)
    assert estimate.billable_blocks == 0
    assert estimate.prompt_tokens == 0
    assert estimate.cost_usd_uncached is None


def test_unmeasurable_router_yields_no_estimate() -> None:
    """A test double / plugin router must not turn an estimate into a crash.

    The pipeline injects whatever router the caller built, and some of them
    (MagicMocks in the regression suite, plugin routers with another contract)
    do not return a text pair. Losing the forecast is fine; failing the run is
    not.
    """
    from unittest.mock import MagicMock

    silent = MagicMock()
    silent.build_draft_prompt.return_value = ()
    assert measure_prefix_tokens(silent, target_lang="zh", source_lang="en") is None

    weird = MagicMock()
    weird.build_draft_prompt.return_value = (1, 2)
    assert measure_prefix_tokens(weird, target_lang="zh", source_lang="en") is None
