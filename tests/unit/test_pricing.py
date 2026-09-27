"""Unit tests for per-model token pricing and cache discounts."""

import pytest

from ubt.core.router.pricing import (
    cache_hit_rate_from_usage,
    endpoint_is_local,
    estimate_cost_usd,
    has_price_entry,
    price_is_known,
    resolve_cached_input_price,
    resolve_model_prices,
)


def test_resolve_model_prices_longest_prefix() -> None:
    assert resolve_model_prices("deepseek-chat") == (0.27, 1.10)
    # Specific tier prefix (deepseek-v4-flash) wins over the general deepseek family prefix.
    assert resolve_model_prices("deepseek-v4-flash-0715") == (0.14, 0.55)
    # General family prefix resolves other derivatives without exhaustive enumeration.
    assert resolve_model_prices("deepseek-coder-0715") == (0.27, 1.10)
    assert resolve_model_prices("deepseek-reasoner-x") == (0.55, 2.19)
    # Unknown model -> (0, 0): reportable "unknown", not a fabricated number.
    assert resolve_model_prices("totally-unknown-model") == (0.0, 0.0)


def test_resolve_cached_input_price_known_and_fallback() -> None:
    # Published DeepSeek cache-hit rate.
    assert resolve_cached_input_price("deepseek-chat") == 0.028
    assert resolve_cached_input_price("deepseek-reasoner") == 0.11
    # Derivative names resolve through the family prefix.
    assert resolve_cached_input_price("deepseek-chat-0918") == 0.028
    # No published cached rate -> full input price (no invented discount).
    assert resolve_cached_input_price("gpt-4o-mini") == 0.15
    assert resolve_cached_input_price("totally-unknown-model") == 0.0


def test_resolve_cached_input_price_anthropic() -> None:
    """Claude cache reads bill at the published ~10% rate, not the full input.

    The cached-input table held only DeepSeek keys, so a cache-heavy Claude run
    billed reads at the full input price (10x) and could trip UBT_BUDGET_USD.
    """
    assert resolve_cached_input_price("claude-3-5-sonnet") == 0.30
    assert resolve_cached_input_price("claude-3-5-haiku") == 0.08
    assert resolve_cached_input_price("claude-3-opus") == 1.50
    # Dated / vendor-prefixed names resolve through the family prefix, and the
    # more specific entry wins over the generic ``claude`` default.
    assert resolve_cached_input_price("claude-3-5-sonnet-20241022") == 0.30
    assert resolve_cached_input_price("anthropic/claude-3-5-haiku") == 0.08
    assert resolve_cached_input_price("claude-opus-4") == 1.50


def test_estimate_cost_usd_without_cache_hits_unchanged() -> None:
    """No cached_tokens reported -> identical to the legacy full-input formula."""
    totals = {"deepseek-chat": {"prompt_tokens": 100_000, "completion_tokens": 10_000}}
    expected = (100_000 * 0.27 + 10_000 * 1.10) / 1_000_000
    assert estimate_cost_usd(totals) == round(expected, 6)


def test_estimate_cost_usd_applies_cache_discount() -> None:
    """Cache-hit tokens are billed at the cached rate, not full input.

    100k prompt with 80k cache hits on deepseek-chat:
      legacy  = (100k * 0.27 + 10k * 1.10) / 1M = 0.038
      discounted = (20k * 0.27 + 80k * 0.028 + 10k * 1.10) / 1M = 0.01864
    Charging full input price for hits overstates cost ~2x here; with the
    README's high hit ratios the gap approaches ~10x on input-heavy runs.
    """
    totals = {
        "deepseek-chat": {
            "prompt_tokens": 100_000,
            "completion_tokens": 10_000,
            "cached_tokens": 80_000,
        }
    }
    expected = (20_000 * 0.27 + 80_000 * 0.028 + 10_000 * 1.10) / 1_000_000
    assert estimate_cost_usd(totals) == round(expected, 6)
    legacy = (100_000 * 0.27 + 10_000 * 1.10) / 1_000_000
    discounted = estimate_cost_usd(totals)
    assert discounted is not None and discounted < legacy


def test_estimate_cost_usd_no_discount_for_unpublished_models() -> None:
    """Models without a published cached rate pay full input price even when
    the provider reports cache hits — never fabricate a discount."""
    totals = {
        "gpt-4o-mini": {
            "prompt_tokens": 100_000,
            "completion_tokens": 0,
            "cached_tokens": 80_000,
        }
    }
    expected = (100_000 * 0.15) / 1_000_000
    assert estimate_cost_usd(totals) == round(expected, 6)


def test_estimate_cost_usd_clamps_cached_above_prompt() -> None:
    """Defensive: a provider reporting more cached than prompt tokens must not
    produce negative uncached cost."""
    totals = {
        "deepseek-chat": {
            "prompt_tokens": 1_000,
            "completion_tokens": 0,
            "cached_tokens": 5_000,
        }
    }
    # cached clamps to prompt_tokens = 1_000 at the cached rate.
    expected = (1_000 * 0.028) / 1_000_000
    assert estimate_cost_usd(totals) == round(expected, 6)


def test_estimate_cost_usd_sums_across_models() -> None:
    totals = {
        "deepseek-chat": {"prompt_tokens": 1_000_000, "completion_tokens": 0},
        "deepseek-reasoner": {
            "prompt_tokens": 1_000_000,
            "completion_tokens": 0,
            "cached_tokens": 500_000,
        },
    }
    expected = (1_000_000 * 0.27 + 500_000 * 0.55 + 500_000 * 0.11) / 1_000_000
    assert estimate_cost_usd(totals) == round(expected, 6)


def test_cache_hit_rate_from_usage() -> None:
    """Run-scoped cache rate: the report and the progress event must agree."""
    assert cache_hit_rate_from_usage({}) == 0.0
    # No prompt tokens yet -> "unmeasured", never a divide-by-zero.
    assert cache_hit_rate_from_usage({"m": {"completion_tokens": 10}}) == 0.0
    assert cache_hit_rate_from_usage({"m": {"prompt_tokens": 100, "cached_tokens": 64}}) == 0.64
    # Summed across models, and clamped when the API over-reports hits.
    assert (
        cache_hit_rate_from_usage(
            {
                "a": {"prompt_tokens": 100, "cached_tokens": 50},
                "b": {"prompt_tokens": 100, "cached_tokens": 50},
            }
        )
        == 0.5
    )
    assert cache_hit_rate_from_usage({"m": {"prompt_tokens": 10, "cached_tokens": 999}}) == 1.0


def test_unpriced_model_with_usage_reports_unknown_not_zero(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The $0.00000-default bug: an unpriced model that spent tokens used to
    sum to a confident zero. It must now report unknown (None) and warn."""
    import logging

    with caplog.at_level(logging.WARNING):
        cost = estimate_cost_usd(
            {"brand-new-model-9000": {"prompt_tokens": 1000, "completion_tokens": 200}}
        )
    assert cost is None
    assert "price-table" in caplog.text


def test_free_tier_models_are_priced_zero_not_unknown() -> None:
    """Zero pricing is an ENDPOINT fact.

    The table used to carry ``ollama/``, ``localhost`` and
    ``host.docker.internal`` as if they were model names, so a real self-hosted
    run — model ``llama3.2``, endpoint ``127.0.0.1`` — never matched any of them
    and reported "unknown" (which refuses a capped run), while ``qwen3:8b`` on
    that same endpoint was billed at the *cloud* Qwen rate.
    """
    assert (
        estimate_cost_usd(
            {"llama3.2": {"prompt_tokens": 500, "completion_tokens": 90}},
            base_url="http://127.0.0.1:11434/v1",
        )
        == 0.0
    )
    # A genuinely free CLOUD tier stays a price-table entry (OpenCode zen).
    assert has_price_entry("opencode/deepseek-v4-flash")
    assert not has_price_entry("brand-new-model-9000")
    # The fake model-name entries are removed, not merely shadowed.
    assert not has_price_entry("ollama/llama3")
    assert not has_price_entry("localhost")


def test_shipped_default_muse_model_is_priced_zero() -> None:
    """The default draft/repair model must resolve to an explicit free entry.

    With no price-table entry the default run reported "unknown" cost, so
    UBT_BUDGET_USD silently enforced nothing out of the box.
    """
    assert has_price_entry("muse-spark-1.3-contributor")
    assert (
        estimate_cost_usd(
            {"muse-spark-1.3-contributor": {"prompt_tokens": 1000, "completion_tokens": 200}}
        )
        == 0.0
    )


def test_unpriced_model_without_usage_does_not_poison_the_total() -> None:
    totals = {
        "deepseek-chat": {"prompt_tokens": 1_000_000, "completion_tokens": 10},
        "ghost-model": {"prompt_tokens": 0, "completion_tokens": 0},
    }
    cost = estimate_cost_usd(totals)
    assert cost is not None
    assert cost > 0.0


def test_batch_served_tokens_are_discounted_and_interactive_tokens_are_not() -> None:
    """Only the tokens the batch parser marked bill at the half rate.

    Discounting the whole model would also halve the repair/QE calls that ran
    interactively on the same model; not discounting at all overstated a batch
    run ~2x, which tripped UBT_BUDGET_USD at roughly half the book while the
    assess quote — which *did* apply the discount — said the run would fit.
    """
    interactive = {"prompt_tokens": 100_000, "completion_tokens": 10_000}
    full = (100_000 * 0.27 + 10_000 * 1.10) / 1_000_000
    assert estimate_cost_usd({"deepseek-chat": interactive}) == round(full, 6)

    mixed = {
        "deepseek-chat": {
            "prompt_tokens": 200_000,
            "completion_tokens": 20_000,
            "batch_prompt_tokens": 100_000,
            "batch_completion_tokens": 10_000,
        }
    }
    # Half the tokens came back through the Batch API: that half bills at 50%.
    assert estimate_cost_usd(mixed) == round(full + full * 0.5, 6)


def test_pricing_anthropic_and_openai_models() -> None:
    # Official Claude 3 model IDs must resolve to their correct prices, not Sonnet fallback
    opus_prices = resolve_model_prices("claude-3-opus-20240229")
    assert opus_prices == (15.0, 75.0), f"Opus price mismatch: {opus_prices}"

    haiku_prices = resolve_model_prices("claude-3-haiku-20240307")
    assert haiku_prices == (0.80, 4.00), f"Haiku price mismatch: {haiku_prices}"

    sonnet_prices = resolve_model_prices("claude-3-5-sonnet-20241022")
    assert sonnet_prices == (3.0, 15.0), f"Sonnet price mismatch: {sonnet_prices}"

    # OpenAI standard models must exist in pricing table
    assert has_price_entry("gpt-4-turbo") is True
    assert has_price_entry("gpt-4") is True
    assert has_price_entry("o1") is True
    assert has_price_entry("o1-mini") is True


@pytest.mark.fast
def test_pricing_strips_provider_prefix() -> None:
    from ubt.core.router.pricing import resolve_cached_input_price

    # Direct vs provider-prefixed names
    direct_prices = resolve_model_prices("gpt-4o")
    prefixed_prices = resolve_model_prices("openai/gpt-4o")
    assert direct_prices[0] > 0
    assert prefixed_prices == direct_prices

    direct_cached = resolve_cached_input_price("deepseek-chat")
    prefixed_cached = resolve_cached_input_price("openrouter/deepseek-chat")
    assert direct_cached > 0
    assert prefixed_cached == direct_cached


# ---------------------------------------------------------------------------
# "Free" is a property of WHERE the request went, not of what the model is
# called. Self-hosting is the common case for this project, and
# every self-hosted user was either mis-billed or refused at startup.
# ---------------------------------------------------------------------------

_LOCAL_ENDPOINTS = (
    "http://127.0.0.1:11434/v1",
    "http://127.0.0.53:9090/v1",  # a loopback stub resolver is still loopback
    "http://localhost:9090/v1",
    "http://[::1]:9090/v1",
    "http://0.0.0.0:11434/v1",
    "http://host.docker.internal:11434/v1",
    "http://host.containers.internal:11434/v1",
    "http://ollama.localhost:11434/v1",
)

_REMOTE_ENDPOINTS = (
    "https://api.deepseek.com/v1",
    "https://api.openai.com/v1",
    "http://192.168.1.50:11434/v1",  # a LAN box is not free until declared
    "http://10.0.0.7:8000/v1",
    "http://127.0.0.1.evil.example/v1",  # a name that merely ends in a literal
    "not a url",
    "",
)


@pytest.mark.parametrize("endpoint", _LOCAL_ENDPOINTS)
def test_endpoint_is_local_accepts_loopback_and_container_hosts(endpoint: str) -> None:
    assert endpoint_is_local(endpoint), endpoint


@pytest.mark.parametrize("endpoint", _REMOTE_ENDPOINTS)
def test_endpoint_is_local_rejects_remote_and_lookalikes(endpoint: str) -> None:
    assert not endpoint_is_local(endpoint), endpoint


def test_local_endpoint_makes_any_model_name_free() -> None:
    """Including the names the table would otherwise price as cloud models."""
    local = "http://127.0.0.1:11434/v1"
    for model in ("llama3.2", "translategemma:4b", "gemma4-e4b"):
        # Absent from the table entirely: these used to report "unknown",
        # which refuses a capped run.
        assert not has_price_entry(model), model
        totals = {model: {"prompt_tokens": 1_000, "completion_tokens": 500}}
        assert estimate_cost_usd(totals, base_url=local) == 0.0, model
        assert price_is_known(model, base_url=local) is True, model

    # `qwen3:8b` DOES match the cloud Qwen entry — the old code charged a local
    # run (0.5, 1.5) USD/Mtok for it. The endpoint decides, not the name.
    assert has_price_entry("qwen3:8b")
    totals = {"qwen3:8b": {"prompt_tokens": 1_000_000, "completion_tokens": 0}}
    assert estimate_cost_usd(totals, base_url=local) == 0.0
    assert price_is_known("qwen3:8b", base_url=local) is True


def test_local_looking_model_name_on_a_remote_endpoint_is_still_billed() -> None:
    """Do not invert the bug: a ``:tag`` name is not evidence of self-hosting."""
    totals = {"qwen3:8b": {"prompt_tokens": 1_000_000, "completion_tokens": 0}}
    cost = estimate_cost_usd(totals, base_url="https://api.deepseek.com/v1")
    assert cost == round(1_000_000 * 0.5 / 1_000_000, 6)


def test_remote_endpoint_unpriced_model_stays_unknown() -> None:
    """Fail-closed on a paid channel is load-bearing and unchanged."""
    assert (
        estimate_cost_usd(
            {"brand-new-model-9000": {"prompt_tokens": 10, "completion_tokens": 5}},
            base_url="https://api.openai.com/v1",
        )
        is None
    )


def test_local_endpoint_that_hides_usage_still_reports_unknown() -> None:
    """Loopback is usually free — not *provably* free.

    A paid gateway behind ``127.0.0.1`` (LiteLLM, an opencode/qoder proxy)
    would otherwise get a confident $0.00 for spend the provider never
    reported, leaving UBT_BUDGET_USD satisfied on real money.
    """
    totals = {"llama3.2": {"prompt_tokens": 10, "completion_tokens": 5, "unmeasured_calls": 1}}
    assert estimate_cost_usd(totals, base_url="http://127.0.0.1:11434/v1") is None


def test_declared_lan_endpoint_is_free(monkeypatch: pytest.MonkeyPatch) -> None:
    """A self-hosted inference box on the LAN is free once declared."""
    monkeypatch.setenv("UBT_LOCAL_ENDPOINTS", "inference.lan, 10.0.0.7")
    assert endpoint_is_local("http://inference.lan:8080/v1")
    assert endpoint_is_local("http://10.0.0.7:8080/v1")
    assert (
        estimate_cost_usd(
            {"my-local-model": {"prompt_tokens": 5, "completion_tokens": 5}},
            base_url="http://inference.lan:8080/v1",
        )
        == 0.0
    )


def test_opt_out_bills_loopback_against_the_price_table(monkeypatch: pytest.MonkeyPatch) -> None:
    """``UBT_BILL_LOCAL_ENDPOINT`` for a paid proxy behind loopback."""
    monkeypatch.setenv("UBT_BILL_LOCAL_ENDPOINT", "true")
    assert (
        estimate_cost_usd(
            {"deepseek-chat": {"prompt_tokens": 1_000_000, "completion_tokens": 0}},
            base_url="http://127.0.0.1:11434/v1",
        )
        == 0.27
    )
    # ... and an unpriced model behind that proxy is unknown again, not free.
    assert (
        estimate_cost_usd(
            {"llama3.2": {"prompt_tokens": 5, "completion_tokens": 5}},
            base_url="http://127.0.0.1:11434/v1",
        )
        is None
    )


def test_mixed_endpoints_bill_only_the_remote_channel() -> None:
    """A local draft model plus a cloud OCR model must not zero the cloud one."""
    totals = {
        "translategemma:4b": {"prompt_tokens": 1_000, "completion_tokens": 500},
        "gpt-4o-mini": {"prompt_tokens": 1_000_000, "completion_tokens": 0},
    }
    cost = estimate_cost_usd(
        totals,
        base_url="http://127.0.0.1:9090/v1",
        endpoint_map={"gpt-4o-mini": "https://api.openai.com/v1"},
    )
    assert cost == round(1_000_000 * 0.15 / 1_000_000, 6)


def test_gpt_41_does_not_inherit_legacy_gpt4_rates() -> None:
    """A newer family must not silently resolve to the shorter legacy prefix."""
    from ubt.core.router.pricing import resolve_model_prices

    assert resolve_model_prices("gpt-4.1") == (2.00, 8.00)
    assert resolve_model_prices("gpt-4.1-mini") == (0.40, 1.60)
    assert resolve_model_prices("gpt-4.1") != resolve_model_prices("gpt-4")


@pytest.mark.fast
def test_pricing_multi_segment_namespace() -> None:
    standard = resolve_model_prices("claude-3-5-sonnet")
    namespaced = resolve_model_prices("openrouter/anthropic/claude-3-5-sonnet")
    assert standard != (0.0, 0.0)
    assert namespaced == standard, f"Namespaced model prices {namespaced} should match {standard}"


@pytest.mark.fast
def test_gemini_flash_does_not_inherit_legacy_gemini_rate() -> None:
    """A flash/lite variant must not silently inherit the legacy ``gemini`` rate."""
    assert resolve_model_prices("gemini-1.5-flash") == (0.075, 0.30)
    assert resolve_model_prices("gemini-2.5-flash") == (0.30, 2.50)
    assert resolve_model_prices("gemini-1.5-flash") != resolve_model_prices("gemini")


@pytest.mark.fast
def test_has_price_entry_matches_resolve_for_nested_namespace() -> None:
    """The two predicates must consider the same candidates (rsplit segment)."""
    from ubt.core.router.pricing import has_price_entry

    model = "openrouter/google/gemini-2.0-flash"
    assert resolve_model_prices(model) != (0.0, 0.0)
    assert has_price_entry(model) is True


@pytest.mark.fast
def test_2026_contemporary_models_priced_correctly() -> None:
    """Verify contemporary 2026 models resolve to expected prices and cached prices."""
    # Gemini 3.x
    assert resolve_model_prices("gemini-3.8-flash") == (0.10, 0.40)
    assert resolve_model_prices("gemini-3.1-pro") == (1.25, 10.00)
    assert resolve_cached_input_price("gemini-3.8-flash") == 0.025
    assert resolve_cached_input_price("gemini-3.1-pro") == 0.3125

    # Anthropic 3.7 / 4
    assert resolve_model_prices("claude-3-7-sonnet") == (3.00, 15.00)
    assert resolve_cached_input_price("claude-3-7-sonnet") == 0.30
    assert resolve_model_prices("claude-sonnet-4") == (3.00, 15.00)

    # DeepSeek v4
    assert resolve_model_prices("deepseek-v4-flash") == (0.14, 0.55)
    assert resolve_cached_input_price("deepseek-v4-flash") == 0.014
    assert resolve_model_prices("deepseek-v4") == (0.27, 1.10)

    # OpenAI o4
    assert resolve_model_prices("o4-mini") == (1.10, 4.40)
    assert resolve_model_prices("o4") == (2.50, 10.00)


@pytest.mark.fast
def test_custom_pricing_and_free_endpoints() -> None:
    """Verify custom pricing registration and declared free endpoints."""
    from ubt.core.router.pricing import (
        declare_custom_free_endpoint,
        register_custom_model_pricing,
        reset_custom_pricing,
    )

    try:
        # Before registration, custom model is unknown
        assert (
            price_is_known("my-custom-model", base_url="https://llm.internal.example/v1") is False
        )
        assert resolve_model_prices("my-custom-model") == (0.0, 0.0)

        # Register custom pricing
        register_custom_model_pricing("my-custom-model", (0.20, 0.80), cached_input=0.05)
        assert resolve_model_prices("my-custom-model") == (0.20, 0.80)
        assert resolve_cached_input_price("my-custom-model") == 0.05
        assert price_is_known("my-custom-model", base_url="https://llm.internal.example/v1") is True

        # Custom free endpoint
        assert price_is_known("another-unknown", base_url="https://free.internal.corp/v1") is False
        declare_custom_free_endpoint("https://free.internal.corp/v1")
        assert price_is_known("another-unknown", base_url="https://free.internal.corp/v1") is True
        # Spending estimation on free endpoint should be 0.0
        totals = {"another-unknown": {"prompt_tokens": 10000, "completion_tokens": 5000}}
        assert estimate_cost_usd(totals, base_url="https://free.internal.corp/v1") == 0.0
    finally:
        reset_custom_pricing()
