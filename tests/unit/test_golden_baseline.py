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


# --------------------------------------------------------------------------- #
# Model-translated cases: content invariants instead of byte equality
#
# A case delivered by a real model is not reproducible. Measured on arXiv
# 2606.23344, two runs of IDENTICAL code share 0.83 of their word sequence and
# score SSIM 0.23-0.64 on the sampled pages — far below the 0.985 floor. So the
# byte-level layers, applied to such a case, flag every run and detect nothing.
# What does hold is the content: numbers 0.9987, identifiers 0.9912, citations
# 1.0000 across those same two runs.
# --------------------------------------------------------------------------- #


def _model_baseline() -> dict[str, Any]:
    from scripts.golden_baseline import content_features

    text = (
        "RT-DocLayout 在 2024 年达到 92.5% 的准确率，超过 Mask R-CNN [8] 与 "
        "DiT [4, 11]。模型仅需 33M 参数，训练于 1000 个批次。"
    ) * 4
    base = _baseline()
    base["text"] = text
    base["text_chars"] = len(text)
    base["features"] = content_features(text)
    return base


def test_a_model_translated_case_is_clean_against_itself() -> None:
    base = _model_baseline()
    assert compare(base, base, deterministic=False) == []


def test_rewording_a_model_translated_case_is_not_a_regression() -> None:
    """The whole point: different words, same content, must pass."""
    from scripts.golden_baseline import content_features

    base = _model_baseline()
    current = _model_baseline()
    current["text"] = base["text"].replace("准确率", "精确度").replace("模型", "该模型")
    current["text_chars"] = len(current["text"])
    current["features"] = content_features(current["text"])
    assert compare(base, current, deterministic=False) == []


def test_losing_the_citations_is_a_regression() -> None:
    import re

    from scripts.golden_baseline import content_features

    base = _model_baseline()
    current = _model_baseline()
    stripped = re.sub(r"\[\d+(?:\s*[,–—-]\s*\d+)*\]", "", base["text"])
    current["text"] = stripped
    current["text_chars"] = len(stripped)
    current["features"] = content_features(stripped)
    problems = compare(base, current, deterministic=False)
    assert any("citations lost" in p for p in problems), problems


def test_table_cell_fusion_does_not_lose_digits() -> None:
    """The text layer glues adjacent table cells: ``0.082 90.30`` / ``0.08290.30``.

    Whether it does so depends on the column widths the translation produces, so
    the SAME code renders the same table either way — measured on arXiv
    2606.23344: two identical runs produced 20 fused tokens, while a baseline run
    produced none. A token-keyed count called that a 4.4% number loss.

    The digit multiset is invariant under fusion (only the '.' moves): baseline
    vs run matched 2385/2385 digits. That is what the gate counts.
    """
    from scripts.golden_baseline import content_features

    separated = content_features("结果 0.082 90.30 与 93.81 水平")
    fused = content_features("结果 0.08290.30 与 93.81 水平")
    assert fused["digits"] == separated["digits"]


def test_a_dropped_digit_is_still_detected() -> None:

    from scripts.golden_baseline import _reproduction, content_features

    before = content_features("达到 93.81 与 2024 年")
    after = content_features("达到 93.8 与 20 年")
    assert _reproduction(before["digits"], after["digits"]) < 0.9


def test_losing_the_numbers_is_a_regression() -> None:
    import re

    from scripts.golden_baseline import content_features

    base = _model_baseline()
    current = _model_baseline()
    stripped = re.sub(r"\d+(?:[.,]\d+)*", "", base["text"])
    current["text"] = stripped
    current["text_chars"] = len(stripped)
    current["features"] = content_features(stripped)
    problems = compare(base, current, deterministic=False)
    assert any("digits lost" in p for p in problems), problems


def test_losing_the_identifiers_is_a_regression() -> None:
    from scripts.golden_baseline import content_features

    base = _model_baseline()
    current = _model_baseline()
    stripped = base["text"].replace("Mask R-CNN", "掩码模型").replace("DiT", "扩散")
    current["text"] = stripped
    current["text_chars"] = len(stripped)
    current["features"] = content_features(stripped)
    problems = compare(base, current, deterministic=False)
    assert any("identifiers lost" in p for p in problems), problems


def test_an_untranslated_page_moves_the_cjk_share() -> None:
    from scripts.golden_baseline import content_features

    base = _model_baseline()
    current = _model_baseline()
    english = "This page came out in the source language and was never translated. " * 10
    current["text"] = english
    current["text_chars"] = len(english)
    current["features"] = content_features(english)
    problems = compare(base, current, deterministic=False)
    assert any("CJK share" in p for p in problems), problems


def test_a_deterministic_case_still_gates_on_the_word_sequence() -> None:
    """The strict layers stay strict where the artifact really is reproducible."""
    base = _model_baseline()
    current = _model_baseline()
    current["text"] = base["text"].replace("准确率", "精确度") * 1
    current["text_chars"] = len(current["text"])
    assert compare(base, current, deterministic=True) != []
    assert compare(base, current, deterministic=False) == []


def test_a_baseline_frozen_without_features_still_gets_the_content_gate() -> None:
    """An older baseline has no ``features``; derive them, do not skip the layer."""
    base = _model_baseline()
    base.pop("features")
    current = _model_baseline()
    import re

    from scripts.golden_baseline import content_features

    stripped = re.sub(r"\[\d+(?:\s*[,–—-]\s*\d+)*\]", "", current["text"])
    current["text"] = stripped
    current["text_chars"] = len(stripped)
    current["features"] = content_features(stripped)
    problems = compare(base, current, deterministic=False)
    assert any("citations lost" in p for p in problems), problems
