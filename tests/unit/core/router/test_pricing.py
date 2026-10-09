"""Unit tests for router pricing table resolution and budget safety."""

from __future__ import annotations

import pytest

from ubt.core.router.pricing import (
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
    # o1-pro is a 10x more expensive tier than o1; must not silently resolve to o1
    assert not has_price_entry("o1-pro")
    assert resolve_model_prices("o1-pro") == (0.0, 0.0)

    # gpt-4-32k has 2x higher rates than gpt-4; must not silently match gpt-4
    assert not has_price_entry("gpt-4-32k")
    assert resolve_model_prices("gpt-4-32k") == (0.0, 0.0)


def test_bare_family_keys_do_not_swallow_arbitrary_unpriced_variants() -> None:
    # gemini-3.8-pro is not in shipped prices.toml; must not silently resolve to bare 'gemini'
    assert not has_price_entry("gemini-3.8-pro")
    assert resolve_model_prices("gemini-3.8-pro") == (0.0, 0.0)

    # claude-novel-variant must not fall back to bare 'claude'
    assert not has_price_entry("claude-future-model")
    assert resolve_model_prices("claude-future-model") == (0.0, 0.0)

    # But verbatim exact match on bare key remains functional
    assert has_price_entry("gemini")
    assert resolve_model_prices("gemini") == (0.10, 0.40)


def test_price_is_known_reports_false_for_unpriced_cloud_models() -> None:
    assert not price_is_known("o1-pro", base_url="https://api.openai.com/v1")
    assert not price_is_known(
        "gemini-3.8-pro", base_url="https://generativelanguage.googleapis.com"
    )
    # Self-hosted / local endpoints are treated as free/known
    assert price_is_known("any-model", base_url="http://127.0.0.1:11434/v1")
