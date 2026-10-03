"""Contract tests for the render-route / granularity policy resolvers."""

from __future__ import annotations

import logging

import pytest

from ubt.core.config import UBTConfig
from ubt.core.ir.models import BlockType, BookManifest, BoundingBox, IRBlock, make_element
from ubt.core.ir.run_metadata import RunMetadata
from ubt.core.policy.adaptive_policy import (
    MULTICOLUMN_SHARE_AUTO,
    STRUCT_SHARE_AUTO,
    AdaptivePolicy,
    Granularity,
    resolve_adaptive_policy,
    resolve_pdf_engine,
    resolve_render_engine_from_signals,
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


def _block(
    block_id: str = "b",
    block_type: BlockType = BlockType.NARRATIVE,
    *,
    bbox: BoundingBox | None = None,
    spine_index: int = 0,
) -> IRBlock:
    element = make_element(
        id=block_id,
        spine_index=spine_index,
        block_type=block_type,
        source_text="x",
        bbox=bbox,
        skip_translate=False,
    )
    return IRBlock(element=element)


def _box(x0: float = 0.0, y0: float = 0.0, x1: float = 10.0, y1: float = 10.0) -> BoundingBox:
    return BoundingBox(page=1, x0=x0, y0=y0, x1=x1, y1=y1)


# --------------------------------------------------------------------------- #
# Constants / enum / dataclass surface
# --------------------------------------------------------------------------- #


def test_auto_dispatch_thresholds_are_pinned() -> None:
    assert STRUCT_SHARE_AUTO == 0.20
    assert MULTICOLUMN_SHARE_AUTO == 0.25


def test_granularity_only_exposes_micro() -> None:
    assert list(Granularity) == [Granularity.MICRO]
    assert Granularity.MICRO.value == "micro"


def test_to_dict_round_trips_every_field() -> None:
    policy = AdaptivePolicy(
        granularity=Granularity.MICRO,
        render_engine="rigid",
        fast_lane_bible=True,
        visual_blocking=False,
        deterministic_glossary=True,
        reason="because",
    )
    assert policy.to_dict() == {
        "granularity": "micro",
        "render_engine": "rigid",
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
    assert policy.render_engine == "auto"
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


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("auto", "auto"),
        ("rigid", "rigid"),
        ("reflow", "publication"),
        ("publication", "publication"),
    ],
)
def test_render_engine_is_canonicalized_from_config(configured: str, expected: str) -> None:
    config = UBTConfig(render_engine=configured)  # type: ignore[arg-type]
    policy = resolve_adaptive_policy(_manifest(), _route("short", 5), config)
    assert policy.render_engine == expected


def test_retired_inplace_alias_folds_to_rigid() -> None:
    config = UBTConfig(render_engine="inplace")  # type: ignore[arg-type]
    policy = resolve_adaptive_policy(_manifest(), _route("long", 300), config)
    assert policy.render_engine == "rigid"


# --------------------------------------------------------------------------- #
# resolve_render_engine_from_signals
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("requested", ["publication", "reflow"])
def test_publication_request_passes_through(requested: str) -> None:
    assert (
        resolve_render_engine_from_signals(
            requested, has_math=True, struct_share=0.9, multicolumn_share=0.9
        )
        == "publication"
    )


def test_rigid_request_passes_through_without_geometry() -> None:
    assert (
        resolve_render_engine_from_signals(
            "rigid", has_math=False, struct_share=0.0, has_geometry=False
        )
        == "rigid"
    )


def test_inplace_alias_resolves_to_rigid() -> None:
    assert (
        resolve_render_engine_from_signals("inplace", has_math=False, struct_share=0.0) == "rigid"
    )


def test_hybrid_alias_falls_into_auto_dispatch() -> None:
    assert resolve_render_engine_from_signals("hybrid", has_math=True, struct_share=0.0) == "rigid"


def test_unknown_engine_warns_and_falls_back_to_publication(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="ubt.core.policy.adaptive_policy"):
        engine = resolve_render_engine_from_signals("weird", has_math=True, struct_share=0.9)
    assert engine == "publication"
    assert any("Unknown render_engine" in record.message for record in caplog.records)


def test_auto_with_math_takes_rigid() -> None:
    assert resolve_render_engine_from_signals("auto", has_math=True, struct_share=0.0) == "rigid"


@pytest.mark.parametrize(
    ("share", "expected"),
    [(0.199, "publication"), (0.20, "rigid"), (0.5, "rigid")],
)
def test_auto_structure_share_boundary(share: float, expected: str) -> None:
    assert (
        resolve_render_engine_from_signals("auto", has_math=False, struct_share=share) == expected
    )


@pytest.mark.parametrize(
    ("share", "expected"),
    [(0.249, "publication"), (0.25, "rigid"), (0.4, "rigid")],
)
def test_auto_multicolumn_share_boundary(share: float, expected: str) -> None:
    assert (
        resolve_render_engine_from_signals(
            "auto", has_math=False, struct_share=0.0, multicolumn_share=share
        )
        == expected
    )


def test_auto_without_geometry_reflows_even_with_math(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="ubt.core.policy.adaptive_policy"):
        engine = resolve_render_engine_from_signals(
            "auto", has_math=True, struct_share=0.9, has_geometry=False
        )
    assert engine == "publication"
    assert any("no usable geometry" in record.message for record in caplog.records)


def test_empty_request_is_treated_as_auto() -> None:
    assert resolve_render_engine_from_signals("", has_math=True, struct_share=0.0) == "rigid"


def test_plain_prose_auto_reflows() -> None:
    assert (
        resolve_render_engine_from_signals("auto", has_math=False, struct_share=0.0)
        == "publication"
    )


def test_request_is_case_and_whitespace_insensitive() -> None:
    assert (
        resolve_render_engine_from_signals("  RIGID ", has_math=False, struct_share=0.0) == "rigid"
    )


# --------------------------------------------------------------------------- #
# resolve_pdf_engine
# --------------------------------------------------------------------------- #


def test_pdf_engine_derives_geometry_from_blocks() -> None:
    blocks = [_block("a", BlockType.NARRATIVE, bbox=_box())]
    assert resolve_pdf_engine("auto", blocks) == "publication"


def test_pdf_engine_zero_area_bbox_is_not_geometry() -> None:
    # A stamped zero-area box (plain-text fallback) must not count as geometry:
    # a formula with no usable box reflows rather than routing rigid.
    blocks = [_block("f", BlockType.FORMULA, bbox=_box(x1=0.0, y1=0.0))]
    assert resolve_pdf_engine("auto", blocks) == "publication"


def test_pdf_engine_formula_block_drives_has_math() -> None:
    # 1 formula among 6 blocks: struct share ~0.167 stays below the threshold,
    # so rigid can only come from the formula block itself.
    blocks = [_block(f"p{i}", BlockType.NARRATIVE, bbox=_box()) for i in range(5)]
    blocks.append(_block("f", BlockType.FORMULA, bbox=_box()))
    assert resolve_pdf_engine("auto", blocks) == "rigid"


def test_pdf_engine_structural_share_reaches_threshold() -> None:
    blocks = [
        _block("p1", BlockType.NARRATIVE, bbox=_box()),
        _block("p2", BlockType.NARRATIVE, bbox=_box()),
        _block("p3", BlockType.NARRATIVE, bbox=_box()),
        _block("p4", BlockType.NARRATIVE, bbox=_box()),
        _block("t", BlockType.TABLE, bbox=_box()),
    ]
    # 1 / 5 = 0.20 exactly -> rigid.
    assert resolve_pdf_engine("auto", blocks) == "rigid"


def test_pdf_engine_no_blocks_reflows() -> None:
    assert resolve_pdf_engine("auto", []) == "publication"


def test_pdf_engine_requested_rigid_short_circuits() -> None:
    assert resolve_pdf_engine("rigid", [_block("p", BlockType.NARRATIVE)]) == "rigid"


def test_pdf_engine_manifest_formula_heavy_forces_math() -> None:
    blocks = [_block("p", BlockType.NARRATIVE, bbox=_box())]
    run = RunMetadata(route_decision={"formula_heavy": True})
    assert resolve_pdf_engine("auto", blocks, _manifest(run)) == "rigid"


def test_pdf_engine_manifest_multicolumn_share_routes_rigid() -> None:
    blocks = [_block("p", BlockType.NARRATIVE, bbox=_box())]
    run = RunMetadata(route_decision={"multicolumn_page_share": 0.3})
    assert resolve_pdf_engine("auto", blocks, _manifest(run)) == "rigid"


def test_pdf_engine_manifest_multicolumn_share_accepts_numeric_string() -> None:
    blocks = [_block("p", BlockType.NARRATIVE, bbox=_box())]
    run = RunMetadata(route_decision={"multicolumn_page_share": "0.25"})
    assert resolve_pdf_engine("auto", blocks, _manifest(run)) == "rigid"


@pytest.mark.parametrize("bad", [None, "abc", {}])
def test_pdf_engine_manifest_multicolumn_share_invalid_is_zero(bad: object) -> None:
    blocks = [_block("p", BlockType.NARRATIVE, bbox=_box())]
    run = RunMetadata(route_decision={"multicolumn_page_share": bad})
    assert resolve_pdf_engine("auto", blocks, _manifest(run)) == "publication"


def test_pdf_engine_manifest_route_decision_non_dict_is_ignored() -> None:
    blocks = [_block("p", BlockType.NARRATIVE, bbox=_box())]
    run = RunMetadata(route_decision={"formula_heavy": False})
    assert resolve_pdf_engine("auto", blocks, _manifest(run)) == "publication"


def test_pdf_engine_no_manifest_uses_block_signals_only() -> None:
    blocks = [
        _block("p", BlockType.NARRATIVE, bbox=_box()),
        _block("t", BlockType.TABLE, bbox=_box()),
    ]
    # 1 / 2 = 0.5 -> rigid without any manifest.
    assert resolve_pdf_engine("auto", blocks, None) == "rigid"


def test_pdf_engine_unknown_request_falls_back_to_publication() -> None:
    blocks = [_block("p", BlockType.NARRATIVE, bbox=_box())]
    assert resolve_pdf_engine("weird", blocks) == "publication"


def test_pdf_engine_manifest_structural_page_share_routes_rigid() -> None:
    blocks = [_block("p", BlockType.NARRATIVE, bbox=_box())]
    run = RunMetadata(route_decision={"structural_page_share": 0.18})
    assert resolve_pdf_engine("auto", blocks, _manifest(run)) == "rigid"


def test_pdf_engine_paper_profile_routes_rigid() -> None:
    assert (
        resolve_render_engine_from_signals(
            "auto", has_math=False, struct_share=0.0, profile="paper"
        )
        == "rigid"
    )


def test_pdf_engine_academic_paper_category_routes_rigid() -> None:
    assert (
        resolve_render_engine_from_signals(
            "auto", has_math=False, struct_share=0.0, category="DocCategory.ACADEMIC_PAPER"
        )
        == "rigid"
    )
