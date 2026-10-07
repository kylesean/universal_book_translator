"""Contract tests for the (single) execution-granularity policy resolver.

The render-route axis is gone: every PDF renders through the single
source-canvas ``overlay`` engine (``ubt.core.config.RENDER_ENGINE``), so there
are no engine resolvers, threshold constants or legacy aliases left to test.
"""

from __future__ import annotations

import logging

import pytest

from ubt.core import config as config_mod
from ubt.core.config import UBTConfig
from ubt.core.ir.models import BookManifest
from ubt.core.ir.run_metadata import RunMetadata
from ubt.core.policy.adaptive_policy import (
    AdaptivePolicy,
    Granularity,
    resolve_adaptive_policy,
)
from ubt.core.router_mode import RouteDecision, RouteMode

pytestmark = pytest.mark.fast


def _route(mode: RouteMode = "short", pages: int = 5) -> RouteDecision:
    return RouteDecision(
        mode=mode,
        pages=pages,
        chars=1000,
        has_scan=False,
        formula_heavy=False,
        reason="test",
    )


def _manifest(run: RunMetadata | None = None) -> BookManifest:
    return BookManifest(doc_id="doc", title="t", source_path="s.pdf", run=run or RunMetadata())


# --------------------------------------------------------------------------- #
# The engine is a constant, not a choice
# --------------------------------------------------------------------------- #


def test_the_render_engine_is_a_single_constant() -> None:
    assert config_mod.RENDER_ENGINE == "overlay"


def test_config_has_no_render_engine_knob() -> None:
    assert not hasattr(UBTConfig(), "render_engine")
    assert not hasattr(AdaptivePolicy, "render_engine")


def test_adaptive_policy_has_no_render_engine_field() -> None:
    policy = AdaptivePolicy(
        granularity=Granularity.MICRO,
        fast_lane_bible=True,
        visual_blocking=False,
        deterministic_glossary=True,
        reason="because",
    )
    assert "render_engine" not in policy.to_dict()


def test_granularity_only_exposes_micro() -> None:
    assert list(Granularity) == [Granularity.MICRO]
    assert Granularity.MICRO.value == "micro"


def test_to_dict_round_trips_every_field() -> None:
    policy = AdaptivePolicy(
        granularity=Granularity.MICRO,
        fast_lane_bible=True,
        visual_blocking=False,
        deterministic_glossary=True,
        reason="because",
    )
    assert policy.to_dict() == {
        "granularity": "micro",
        "fast_lane_bible": True,
        "visual_blocking": False,
        "deterministic_glossary": True,
        "reason": "because",
    }


# --------------------------------------------------------------------------- #
# resolve_adaptive_policy
# --------------------------------------------------------------------------- #


def test_short_document_keeps_fast_lane_and_glossary() -> None:
    policy = resolve_adaptive_policy(_manifest(), _route("short", 5), UBTConfig())
    assert policy.granularity is Granularity.MICRO
    assert policy.fast_lane_bible is True
    assert policy.visual_blocking is True
    assert policy.deterministic_glossary is True
    assert policy.reason == (
        "Canonical frozen-math publication pipeline (5pp, fast-lane seed bible)"
    )


def test_long_document_drops_fast_lane_and_glossary() -> None:
    policy = resolve_adaptive_policy(_manifest(), _route("long", 300), UBTConfig())
    assert policy.fast_lane_bible is False
    assert policy.visual_blocking is False
    assert policy.deterministic_glossary is False
    assert policy.reason == "Canonical frozen-math publication pipeline (300pp)"


def test_long_document_honours_visual_blocking_gate_flag() -> None:
    config = UBTConfig(visual_blocking_gate_enabled=True)
    policy = resolve_adaptive_policy(_manifest(), _route("long", 300), config)
    assert policy.visual_blocking is True
    assert policy.fast_lane_bible is False


def test_forced_macro_granularity_still_runs_micro_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="ubt.core.policy.adaptive_policy"):
        policy = resolve_adaptive_policy(
            _manifest(), _route("short", 5), UBTConfig(), forced_granularity="macro"
        )
    assert policy.granularity is Granularity.MICRO
    assert any("retired" in record.message for record in caplog.records)


def test_forced_micro_granularity_is_silent(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="ubt.core.policy.adaptive_policy"):
        policy = resolve_adaptive_policy(
            _manifest(), _route("short", 5), UBTConfig(), forced_granularity="micro"
        )
    assert policy.granularity is Granularity.MICRO
    assert caplog.records == []
