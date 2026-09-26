"""Cost-benchmark price resolution.

The committed artifact is the only reproducible bill for ``--budget-usd``
calibration (docs/benchmarks/README.md), so a default DeepSeek run must stay
*priced* (prices + cost written), while a model the built-in DeepSeek rates do
not cover must write ``null`` rather than a wrong DeepSeek-rate number.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "cost_benchmark.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ubt_cost_benchmark", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cb = _load()


def test_default_deepseek_run_is_priced() -> None:
    args = cb.build_args(["book.md"])
    price_input, _hit, _out, priced = cb._resolve_prices(args)
    assert priced is True
    assert price_input == 0.27


def test_non_deepseek_model_without_rates_is_unpriced() -> None:
    args = cb.build_args(["book.md", "--draft-model", "gpt-4o", "--repair-model", "gpt-4o"])
    assert cb._resolve_prices(args)[3] is False


def test_explicit_rate_prices_any_model() -> None:
    args = cb.build_args(["book.md", "--draft-model", "gpt-4o", "--price-input", "2.5"])
    assert cb._resolve_prices(args)[3] is True
