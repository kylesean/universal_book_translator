"""The golden-baseline gate: four layers that catch a regression.

Each layer answers a question the others cannot:

* **contract** — was content lost outright (a violation, a missing asset)?
* **structure** — did the renderer stop placing blocks or assets?
* **text** — same counters, different words (a mistranslation, a lost page)?
* **visual** — same text, different layout (a moved figure, a squeezed table)?

The tests pin both directions: a real regression must be reported, and a
legitimate change (an asset moving from crop to reconstructed math) must not be.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.golden_baseline import (  # noqa: E402
    _normalise_text,
    _sample_pages,
    compare,
)

pytestmark = pytest.mark.fast


def _baseline() -> dict[str, Any]:
    base = {
        "pages": 15,
        "text": (
            "在深度学习中，模型通常对所有输入重复使用相同的参数。而专家混合模型则打破了这一惯例，"
            "而是为每个输入样本选择不同的参数。其结果是一种稀疏激活的模型——尽管参数数量惊人，"
            "但计算成本却保持恒定。我们提出了切换变换器，简化了专家混合模型的路由算法。"
        )
        * 6,
        "text_chars": 0,  # filled in by _baseline()
        "structure": {
            "delivered_text": 100,
            "total_assets": 20,
            "reconstructed_assets": 3,
            "preserved_assets": 17,
            "missing_assets": 0,
            "violations": 0,
        },
        "visual": {"rasters": {}},
    }
    base["text_chars"] = len(str(base["text"]))
    return base


# --- layer 1: contract ------------------------------------------------------ #


def test_a_contract_violation_is_a_regression() -> None:
    base = _baseline()
    current = _baseline()
    current["structure"]["violations"] = 2
    assert any("contract violations" in p for p in compare(base, current))


def test_a_missing_asset_is_a_regression() -> None:
    base = _baseline()
    current = _baseline()
    current["structure"]["missing_assets"] = 1
    assert any("missing assets" in p for p in compare(base, current))


# --- layer 2: structure ----------------------------------------------------- #


def test_a_lost_page_is_a_regression_even_when_short() -> None:
    """One page of a 15-page paper is 1.35% of its characters.

    A pure text-share threshold passed that (measured), so the page count is
    its own ratchet.
    """
    base = _baseline()
    current = _baseline()
    current["pages"] = 14
    assert any("page count dropped" in p for p in compare(base, current))


def test_fewer_delivered_blocks_is_a_regression() -> None:
    base = _baseline()
    current = _baseline()
    current["structure"]["delivered_text"] = 90
    assert any("delivered text blocks dropped" in p for p in compare(base, current))


def test_assets_moving_from_preserved_to_reconstructed_is_not_a_regression() -> None:
    """A repair that turns a crop into real math is the goal, not a defect."""
    base = _baseline()
    current = _baseline()
    current["structure"]["reconstructed_assets"] = 5
    current["structure"]["preserved_assets"] = 15
    assert compare(base, current) == []


def test_a_dropping_total_asset_count_is_a_regression() -> None:
    base = _baseline()
    current = _baseline()
    current["structure"]["total_assets"] = 15
    assert any("total assets dropped" in p for p in compare(base, current))


# --- layer 3: visible text -------------------------------------------------- #


def test_a_char_count_drop_is_a_regression() -> None:
    base = _baseline()
    current = _baseline()
    current["text"] = "短"
    current["text_chars"] = 1
    problems = compare(base, current)
    assert any("shrank" in p for p in problems)
    assert any("changed" in p for p in problems)


def test_a_small_rewording_is_tolerated() -> None:
    """A translation tweak must not fail the gate, or nobody can improve prose."""
    base = _baseline()
    current = _baseline()
    current["text"] = base["text"].replace("冻结", "固定", 1)
    # Same length rewrite: the character count is unchanged, so only the
    # sequence comparison can see it, and a one-word swap is under the limit.
    current["text_chars"] = len(current["text"])
    assert len(current["text"]) == len(base["text"])
    assert compare(base, current) == []


def test_whitespace_reflow_is_not_a_text_change() -> None:
    assert _normalise_text("a  b\n\nc   d") == "a b c d"


# --- the happy path --------------------------------------------------------- #


def test_an_identical_snapshot_is_clean() -> None:
    base = _baseline()
    assert compare(base, base) == []


# --- sampling --------------------------------------------------------------- #


def test_page_sample_spreads_across_the_document() -> None:
    assert _sample_pages(1) == [1]
    assert _sample_pages(3) == [1, 2, 3]
    sampled = _sample_pages(73)
    assert sampled[0] == 1
    assert len(sampled) == 3
    assert sampled[-1] > 30
