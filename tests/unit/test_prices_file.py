"""Unit tests for the externalized prices.toml pricing table and loader."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from ubt.core.router.pricing import (
    PriceEntry,
    find_prices_file,
    load_prices_table,
    resolve_model_prices,
)

pytestmark = pytest.mark.fast


def test_load_shipped_prices_table() -> None:
    """The packaged prices.toml contains 2026 models with verified prices."""
    table = load_prices_table()
    assert "gemini-3.8-flash" in table
    gemini_entry = table["gemini-3.8-flash"]
    assert isinstance(gemini_entry, PriceEntry)
    assert gemini_entry.input == 0.10
    assert gemini_entry.output == 0.40
    assert gemini_entry.cached_input == 0.025
    assert gemini_entry.verified_at != ""

    assert "claude-3-7-sonnet" in table
    claude_entry = table["claude-3-7-sonnet"]
    assert claude_entry.input == 3.00
    assert claude_entry.output == 15.00
    assert claude_entry.cached_input == 0.30

    assert "deepseek-v4-flash" in table
    assert "o4-mini" in table


def test_user_prices_file_overrides_shipped(tmp_path: Path) -> None:
    """A user prices.toml overrides shipped entries and adds custom models."""
    custom_toml = tmp_path / "custom_prices.toml"
    custom_toml.write_text(
        """
[prices.gemini-3.8-flash]
input = 0.08
output = 0.32
cached_input = 0.02

[prices.my-special-llm]
input = 0.50
output = 2.00
cached_input = 0.10
verified_at = "2026-09-28"
""",
        encoding="utf-8",
    )

    table = load_prices_table(custom_path=custom_toml)
    assert table["gemini-3.8-flash"].input == 0.08
    assert table["gemini-3.8-flash"].output == 0.32
    assert table["gemini-3.8-flash"].cached_input == 0.02

    assert "my-special-llm" in table
    assert table["my-special-llm"].input == 0.50
    assert table["my-special-llm"].output == 2.00


def test_env_ubt_prices_file_precedence(tmp_path: Path) -> None:
    """UBT_PRICES_FILE environment variable directs table resolution."""
    env_toml = tmp_path / "env_prices.toml"
    env_toml.write_text(
        """
[prices.custom-from-env]
input = 1.00
output = 4.00
""",
        encoding="utf-8",
    )

    with patch.dict(os.environ, {"UBT_PRICES_FILE": str(env_toml)}):
        found = find_prices_file()
        assert found == env_toml
        table = load_prices_table()
        assert "custom-from-env" in table
        assert table["custom-from-env"].input == 1.00


def test_prices_exact_match_beats_prefix(tmp_path: Path) -> None:
    """Exact model match takes precedence over family prefix."""
    prices_file = tmp_path / "prefix_test.toml"
    prices_file.write_text(
        """
[prices.myfamily]
input = 1.00
output = 2.00

[prices.myfamily-fast]
input = 0.20
output = 0.40
""",
        encoding="utf-8",
    )
    table = load_prices_table(custom_path=prices_file)
    with patch("ubt.core.router.pricing._current_prices_table", return_value=table):
        assert resolve_model_prices("myfamily-fast") == (0.20, 0.40)
        assert resolve_model_prices("myfamily-pro") == (1.00, 2.00)
