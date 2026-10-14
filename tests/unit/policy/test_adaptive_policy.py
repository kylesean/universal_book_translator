"""Contract tests for the execution-policy resolver.

The Granularity axis is gone: the pipeline is always the frozen-math
micro-block architecture. The render engine is a single constant, not a choice.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ubt.core.config import UBTConfig
from ubt.core.ir.models import BookManifest
from ubt.core.ir.run_metadata import RunMetadata
from ubt.core.policy.adaptive_policy import (
    AdaptivePolicy,
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
# The engine is not a choice
# --------------------------------------------------------------------------- #


def test_config_has_no_render_engine_knob() -> None:
    assert not hasattr(UBTConfig(), "render_engine")
    assert not hasattr(AdaptivePolicy, "render_engine")


def test_adaptive_policy_has_no_render_engine_field() -> None:
    policy = AdaptivePolicy(
        fast_lane_bible=True,
        visual_blocking=False,
        deterministic_glossary=True,
        reason="because",
    )
    assert "render_engine" not in policy.to_dict()


def test_to_dict_round_trips_every_field() -> None:
    policy = AdaptivePolicy(
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


def test_long_route_glossary_is_config_opt_in() -> None:
    """The long route's deterministic glossary is an explicit config switch.

    Default off (the historical premise stays the default until an on/off
    comparison proves otherwise); on, the export stage enforces terms for
    books too, and the reason string says so. Re-export reads the same flag,
    so a re-render cannot silently drop the layer.
    """
    config = UBTConfig(deterministic_glossary_long_route=True)
    policy = resolve_adaptive_policy(_manifest(), _route("long", 300), config)
    assert policy.deterministic_glossary is True
    assert "config opt-in" in policy.reason
    # Short route unaffected: it keeps enforcement with or without the flag.
    short = resolve_adaptive_policy(_manifest(), _route("short", 5), UBTConfig())
    assert short.deterministic_glossary is True


def test_re_export_enforcement_follows_the_config(tmp_path: Path) -> None:
    """re_export's AdaptivePolicy reads deterministic_glossary_long_route.

    The old hard ``False`` meant a re-render shipped a *different* delivery
    than the original run whenever the long-route flag was on (enforced
    spans N -> 0) with nothing flagging the difference.
    """
    from ubt.core.engine.re_export import _build_services

    config = UBTConfig(
        db_dir=tmp_path / "db",
        deterministic_glossary_long_route=True,
    )
    manifest = _manifest()
    services = _build_services(config, router=object(), qe_runner=object(), manifest=manifest)
    assert services.adaptive_policy.deterministic_glossary is True
    # And off stays off.
    config_off = UBTConfig(db_dir=tmp_path / "db2")
    services_off = _build_services(
        config_off, router=object(), qe_runner=object(), manifest=manifest
    )
    assert services_off.adaptive_policy.deterministic_glossary is False
