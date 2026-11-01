from __future__ import annotations

import pytest

from ubt.core.layout.style_group import (
    BODY_FLOOR_PT,
    DRIFT_TOLERANCE_PT,
    EXCEPTION_PAGE_SHARE,
    LADDER,
    FeasibilityProbe,
    MicroTypography,
    PageFit,
    body_band,
    derive_document_body_metrics,
    exception_shares,
    plan_body_group,
    plan_from_metadata,
)

pytestmark = pytest.mark.fast


def _metrics(sizes: dict[int, tuple[float, int]]) -> dict[int, tuple[float, float, int]]:
    return {page: (size, 1.35, weight) for page, (size, weight) in sizes.items()}


def _probe_from_table(
    max_size: dict[int, float],
    *,
    rescue_tracking: dict[int, float] | None = None,
    rescue_pitch: dict[int, float] | None = None,
) -> FeasibilityProbe:

    def probe(page: int, size_pt: float, rung: MicroTypography, pitch: float) -> PageFit:
        limit = max_size.get(page, 99.0)
        if size_pt <= limit:
            return PageFit(page=page, fits=True)
        if rescue_tracking is not None and rung.tracking_em <= rescue_tracking.get(page, 1.0):
            return PageFit(page=page, fits=True)
        if rescue_pitch is not None and pitch <= rescue_pitch.get(page, 99.0):
            return PageFit(page=page, fits=True)
        return PageFit(
            page=page,
            fits=False,
            deficit_pt=round((size_pt - limit) * 10.0, 3),
            block_ids=(f"p{page}-b1",),
        )

    return probe


def test_the_document_size_is_a_text_weighted_mode_not_a_page_median() -> None:
    metrics = _metrics({1: (9.0, 1), 2: (8.9, 120), 3: (8.9, 90), 4: (7.9, 30), 5: (8.0, 2)})
    size, pitch = derive_document_body_metrics(metrics)
    assert size == 8.9
    assert pitch == 1.35


def test_the_band_is_relative_and_absolute_whichever_is_higher() -> None:
    floor, ceiling = body_band(9.8)
    assert floor == pytest.approx(7.84)
    assert ceiling == pytest.approx(10.78)

    floor, ceiling = body_band(6.0)
    assert floor == BODY_FLOOR_PT
    assert ceiling == pytest.approx(BODY_FLOOR_PT)


def test_the_group_starts_at_the_source_size_and_spends_nothing() -> None:
    plan = plan_body_group(_metrics({1: (8.9, 100), 2: (8.9, 100)}), _probe_from_table({}))
    assert plan.group.size_pt == 8.9
    assert plan.group.rung == "plain"
    assert plan.group.pitch == pytest.approx(1.35)
    assert plan.group.tracking_em == 0.0
    assert plan.exceptions == ()


def test_the_mildest_rung_that_rescues_a_page_wins() -> None:
    plan = plan_body_group(
        _metrics({1: (8.9, 100), 2: (8.9, 100)}),
        _probe_from_table({1: 8.0}, rescue_tracking={1: -0.015}),
    )
    assert plan.group.size_pt == 8.9
    assert plan.group.rung == "tracking_-0.015"
    assert plan.group.tracking_em == pytest.approx(-0.015)


def test_one_page_inside_the_budget_keeps_the_source_size_and_is_declared() -> None:
    pages = dict.fromkeys(range(1, 21), (8.9, 100))
    plan = plan_body_group(
        _metrics(pages),
        _probe_from_table({19: 8.0, 20: 8.0}),
    )
    assert plan.group.size_pt == 8.9
    assert plan.exception_pages == (19, 20)
    assert {exception.reason for exception in plan.exceptions} == {"does_not_fit"}
    assert all(exception.deficit_pt > 0 for exception in plan.exceptions)


def test_the_group_steps_down_only_until_the_budget_is_met() -> None:
    pages = dict.fromkeys(range(1, 21), (8.9, 100))
    intro = _probe_from_table(dict.fromkeys((18, 19, 20), 8.4))
    plan = plan_body_group(_metrics(pages), intro)
    assert plan.group.size_pt == 8.4
    assert plan.exceptions == ()


def test_a_page_that_never_holds_the_group_is_declared_not_shrunk() -> None:
    plan = plan_body_group(
        _metrics({1: (8.9, 100), 2: (8.9, 100)}),
        _probe_from_table({1: 7.0, 2: 8.9}),
    )
    assert plan.group.size_pt == BODY_FLOOR_PT
    assert plan.exception_pages == (1,)
    assert plan.exceptions[0].deficit_pt > 0
    assert plan.group.size_pt >= plan.group.floor_pt


def test_deferred_pages_join_the_group_but_do_not_vote_on_its_size() -> None:
    plan = plan_body_group(
        _metrics({1: (8.9, 100), 2: (8.9, 100), 3: (8.9, 100)}),
        _probe_from_table({2: 7.0}),
        deferred_pages=(2,),
    )
    assert plan.group.size_pt == 8.9
    assert plan.deferred_pages == (2,)
    assert plan.exceptions == ()


def test_exception_shares_report_an_unknown_block_count_as_zero() -> None:
    pages = dict.fromkeys(range(1, 11), (8.9, 100))
    plan = plan_body_group(_metrics(pages), _probe_from_table({10: 7.0}))
    page_share, block_share = exception_shares(plan, block_count=0)
    assert page_share == pytest.approx(0.1)
    assert block_share == 0.0


def test_a_plan_round_trips_through_manifest_metadata() -> None:
    pages = dict.fromkeys(range(1, 11), (8.9, 100))
    plan = plan_body_group(_metrics(pages), _probe_from_table({10: 7.0}))
    restored = plan_from_metadata(plan.to_metadata_dict())
    assert restored is not None
    assert restored.group.size_pt == plan.group.size_pt
    assert restored.group.rung == plan.group.rung
    assert restored.exception_pages == plan.exception_pages
    assert restored.exceptions[0].reason == plan.exceptions[0].reason


def test_the_ladder_never_compresses_past_the_prd_bounds() -> None:
    assert LADDER[0].label == "plain"
    assert min(rung.tracking_em for rung in LADDER) >= -0.030
    tracking = [rung.tracking_em for rung in LADDER]
    assert tracking == sorted(tracking, reverse=True)
    pitch_drop = [rung.pitch_drop for rung in LADDER]
    assert pitch_drop == sorted(pitch_drop)
    assert DRIFT_TOLERANCE_PT == 0.5
    assert EXCEPTION_PAGE_SHARE == 0.10
