"""Unit tests for the three-tier elasticity solver."""

from __future__ import annotations

import pytest

from ubt.render.typst.elastic_solver import TypeElasticitySolver

pytestmark = pytest.mark.fast


def test_zero_deficit_keeps_defaults() -> None:
    style = TypeElasticitySolver.resolve_params(base_font_size=10.0, deficit_ratio=0.0)
    assert style.font_size_pt == 10.0
    assert style.leading_em == 0.62
    assert style.stretch_pct == 100.0
    assert style.scale == 1.0


def test_negative_deficit_keeps_defaults() -> None:
    # Underflow (spacious box)
    style = TypeElasticitySolver.resolve_params(base_font_size=10.0, deficit_ratio=-0.15)
    assert style.font_size_pt == 10.0
    assert style.leading_em == 0.62
    assert style.stretch_pct == 100.0


def test_tier_1_reduces_leading_only() -> None:
    # Small deficit 8% -> only reduces leading, leaves stretch at 100% and scale at 1.0
    style = TypeElasticitySolver.resolve_params(base_font_size=10.0, deficit_ratio=0.08)
    assert style.font_size_pt == 10.0
    assert style.leading_em == 0.55
    assert style.stretch_pct == 100.0
    assert style.scale == 1.0


def test_tier_2_compresses_variable_font_stretch_preserves_font_size() -> None:
    # Medium deficit 18% -> compresses font_stretch to 95%, preserves base_font_size!
    style = TypeElasticitySolver.resolve_params(base_font_size=10.0, deficit_ratio=0.18)
    assert style.font_size_pt == 10.0
    assert style.leading_em == 0.48
    assert style.stretch_pct == 95.0
    assert style.scale == 1.0


def test_tier_3_clamps_at_minimum_scale_envelope() -> None:
    # Large deficit 40% -> compresses stretch to 90%, scales font gently, clamps at 0.92
    style = TypeElasticitySolver.resolve_params(base_font_size=10.0, deficit_ratio=0.40)
    assert style.font_size_pt == 9.2  # 10.0 * 0.92
    assert style.leading_em == 0.48
    assert style.stretch_pct == 90.0
    assert style.scale == 0.92

    # Extreme deficit 100% -> still clamps strictly at 0.92 (never collapses into ant text!)
    extreme = TypeElasticitySolver.resolve_params(base_font_size=10.0, deficit_ratio=1.0)
    assert extreme.font_size_pt == 9.2
    assert extreme.scale == 0.92
