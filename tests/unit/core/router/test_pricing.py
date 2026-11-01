from __future__ import annotations

from pathlib import Path

import pytest

from ubt.core.router.pricing import (
    find_prices_file,
    has_price_entry,
    price_is_known,
    resolve_batch_discount,
    resolve_cached_input_price,
    resolve_model_prices,
)

pytestmark = pytest.mark.fast


def test_exact_model_pricing() -> None:
    assert has_price_entry("gpt-4o")
    assert resolve_model_prices("gpt-4o") == (2.50, 10.00)
    assert has_price_entry("gemini-3.8-flash")
    assert resolve_model_prices("gemini-3.8-flash") == (0.10, 0.40)
    assert resolve_cached_input_price("gemini-3.8-flash") == 0.025
    assert resolve_batch_discount("gpt-4o") == 0.5


def test_safe_date_snapshot_prefix_matching() -> None:
    assert has_price_entry("gpt-4o-2024-08-06")
    assert resolve_model_prices("gpt-4o-2024-08-06") == (2.50, 10.00)

    assert has_price_entry("claude-3-5-sonnet-20241022")
    assert resolve_model_prices("claude-3-5-sonnet-20241022") == (3.00, 15.00)

    assert has_price_entry("o1-2024-12-17")
    assert resolve_model_prices("o1-2024-12-17") == (15.00, 60.00)


def test_unsafe_tier_keywords_do_not_match_base_model() -> None:
    assert not has_price_entry("o1-pro")
    assert resolve_model_prices("o1-pro") == (0.0, 0.0)

    assert not has_price_entry("gpt-4-32k")
    assert resolve_model_prices("gpt-4-32k") == (0.0, 0.0)


def test_bare_family_keys_do_not_swallow_arbitrary_unpriced_variants() -> None:
    assert not has_price_entry("gemini-3.8-pro")
    assert resolve_model_prices("gemini-3.8-pro") == (0.0, 0.0)

    assert not has_price_entry("claude-future-model")
    assert resolve_model_prices("claude-future-model") == (0.0, 0.0)

    assert has_price_entry("gemini")
    assert resolve_model_prices("gemini") == (0.10, 0.40)


def test_price_is_known_reports_false_for_unpriced_cloud_models() -> None:
    assert not price_is_known("o1-pro", base_url="https://api.openai.com/v1")
    assert not price_is_known(
        "gemini-3.8-pro", base_url="https://generativelanguage.googleapis.com"
    )
    assert price_is_known("any-model", base_url="http://127.0.0.1:11434/v1")


def test_a_prices_file_set_in_config_is_used(tmp_path: Path) -> None:
    from ubt.core.config import UBTConfig
    from ubt.core.router.pricing import declare_custom_prices_file

    prices = tmp_path / "prices.toml"
    prices.write_text(
        '[prices."ubt-test-model-xyz"]\ninput = 1.0\noutput = 2.0\n', encoding="utf-8"
    )

    assert not has_price_entry("ubt-test-model-xyz"), "the name must start unknown"

    config = UBTConfig(prices_file=prices)
    config.bootstrap_runtime_environment()
    try:
        assert has_price_entry("ubt-test-model-xyz")
        assert resolve_model_prices("ubt-test-model-xyz") == (1.0, 2.0)
    finally:
        declare_custom_prices_file(None)
        from ubt.core.router import pricing

        pricing._load_prices_cached.cache_clear()


def test_a_missing_custom_prices_file_is_ignored() -> None:
    from ubt.core.router.pricing import declare_custom_prices_file

    declare_custom_prices_file("/nonexistent/definitely-not-here.toml")
    try:
        assert find_prices_file() is None
        assert resolve_model_prices("gpt-4o-mini") == (0.15, 0.6)
    finally:
        declare_custom_prices_file(None)


def test_an_explicit_prices_path_that_is_missing_raises() -> None:
    with pytest.raises(FileNotFoundError, match="Prices file not found"):
        find_prices_file("/nonexistent/definitely-not-here.toml")
