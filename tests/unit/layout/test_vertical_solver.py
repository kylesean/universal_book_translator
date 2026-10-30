"""Stage 3B vertical solve, against the real ``kiwisolver``.

The strength contract is the whole point of 3B, so these tests pin the
behaviour the PRD's acceptance depends on:

- required: blocks stay inside the track, consecutive blocks keep
  ``min_gap_pt``, and ``height + deficit >= H_req`` always holds —
  an over-full track surfaces as *deficit*, never as overlap and never as an
  exception from the required set.
- strong: ``deficit == 0`` is sacrificed first, exactly where the track runs
  out (Stage 4's trigger), while medium "keep source height/gap" and weak
  "keep source position" absorb what they can in that order.

``kiwisolver`` ships ``py.typed`` and is installed in the dev env via the
``[cels]`` extra, so these run in the fast tier with no subprocess.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from ubt.core.layout.solver import (
    DEFICIT_TRIGGER_PT,
    SolvedBlockSpec,
    VerticalBlockInput,
    VerticalTrack,
    solve_vertical,
)

pytestmark = pytest.mark.fast

TRACK = VerticalTrack(y_min_pt=60.0, y_max_pt=660.0)  # 600pt of usable column


def make_block(
    index: int,
    *,
    source_top: float,
    source_height: float,
    required: float | None = None,
) -> VerticalBlockInput:
    return VerticalBlockInput(
        index=index,
        required_height_pt=source_height if required is None else required,
        source_height_pt=source_height,
        source_top_pt=source_top,
    )


def assert_no_overlap_or_escape(specs: Sequence[SolvedBlockSpec], track: VerticalTrack) -> None:
    for spec in specs:
        assert spec.y_bottom_pt >= track.y_min_pt - 1e-6, f"block {spec.index} escaped below"
        assert spec.y_top_pt <= track.y_max_pt + 1e-6, f"block {spec.index} escaped above"
        assert spec.height_pt >= -1e-6, f"block {spec.index} has negative height"
    for above, below in zip(specs, specs[1:], strict=False):
        gap = above.y_bottom_pt - below.y_top_pt
        assert gap >= 2.0 - 1e-6, f"blocks {above.index}/{below.index} overlap (gap {gap}pt)"


def test_a_fitting_group_stays_near_its_source_positions() -> None:
    blocks = [
        make_block(0, source_top=650.0, source_height=200.0),
        make_block(1, source_top=440.0, source_height=180.0),
        make_block(2, source_top=250.0, source_height=170.0),
    ]
    specs = solve_vertical(blocks, track=TRACK)
    assert_no_overlap_or_escape(specs, TRACK)
    assert all(spec.deficit_pt == 0.0 for spec in specs), "the track has room; nothing is owed"
    assert all(not spec.needs_stage4 for spec in specs)
    for spec, block in zip(specs, blocks, strict=True):
        assert spec.y_top_pt == pytest.approx(block.source_top_pt, abs=1.0), (
            "the weak 'stay in place' anchor should hold when nothing conflicts"
        )


def test_expansion_grows_heights_and_compresses_gaps_not_deficits() -> None:
    """EN→ZH expansion: required heights exceed source heights, track still has room."""
    blocks = [
        make_block(0, source_top=650.0, source_height=200.0, required=260.0),
        make_block(1, source_top=440.0, source_height=180.0, required=240.0),
    ]
    specs = solve_vertical(blocks, track=TRACK)
    assert_no_overlap_or_escape(specs, TRACK)
    assert all(spec.deficit_pt == 0.0 for spec in specs), "600pt of track holds 500pt of demand"
    assert specs[0].height_pt == pytest.approx(260.0, abs=0.5), (
        "the strong 'no deficit' beats the medium 'keep source height'"
    )
    assert specs[1].height_pt == pytest.approx(240.0, abs=0.5)


def test_an_over_full_track_reports_deficit_instead_of_failing() -> None:
    """Σ H_req + gaps > track: heights shrink to the track, deficit records the shortfall."""
    blocks = [
        make_block(0, source_top=650.0, source_height=200.0, required=400.0),
        make_block(1, source_top=440.0, source_height=200.0, required=400.0),
    ]
    specs = solve_vertical(blocks, track=TRACK)
    assert_no_overlap_or_escape(specs, TRACK)
    total_deficit = sum(spec.deficit_pt for spec in specs)
    assert total_deficit == pytest.approx(800.0 + 2.0 - 600.0, abs=1.0), (
        "the shortfall (required total + min gap - track height) must be accounted, not hidden"
    )
    assert any(spec.needs_stage4 for spec in specs), (
        "deficit over the trigger hands work to Stage 4"
    )
    # The deficit never lies in the other direction: height + deficit covers the requirement.
    for spec in specs:
        assert spec.height_pt + spec.deficit_pt >= 400.0 - 1e-6


def test_deficit_lands_on_squeezed_blocks_and_flags_stage4_per_block() -> None:
    """A modest overflow squeezes only what it must; the flag is per block."""
    blocks = [
        make_block(0, source_top=650.0, source_height=150.0, required=150.0),
        make_block(1, source_top=490.0, source_height=150.0, required=150.0),
        make_block(2, source_top=330.0, source_height=150.0, required=170.0),
        make_block(3, source_top=170.0, source_height=150.0, required=190.0),
    ]
    specs = solve_vertical(blocks, track=TRACK)
    assert_no_overlap_or_escape(specs, TRACK)
    over_threshold = [spec for spec in specs if spec.needs_stage4]
    assert over_threshold, "a 600pt track facing 664pt of demand must owe Stage 4 work"
    assert all(spec.deficit_pt <= 190.0 for spec in specs)
    assert sum(spec.height_pt for spec in specs) <= 600.0 - 3 * 2.0 + 1e-6


def test_min_gap_is_required_even_when_the_source_had_less() -> None:
    """Source blocks nearly touching: the solve may not tighten below min_gap."""
    blocks = [
        make_block(0, source_top=650.0, source_height=200.0),
        make_block(1, source_top=451.0, source_height=200.0),  # source gap was 1pt
    ]
    specs = solve_vertical(blocks, track=TRACK)
    gap = specs[0].y_bottom_pt - specs[1].y_top_pt
    assert gap >= 2.0 - 1e-6, "required min gap beats the medium 'keep source gap'"


def test_a_track_that_cannot_even_hold_empty_blocks_raises() -> None:
    """Gaps alone exceed the track: a real geometry error, reported at add time.

    Three blocks in a 1pt-tall track need 2 x 2pt of required gaps; no y
    assignment exists, and kiwisolver must say so instead of relaxing a
    required constraint.
    """
    import kiwisolver

    blocks = [
        make_block(0, source_top=691.0, source_height=5.0, required=0.5),
        make_block(1, source_top=690.8, source_height=5.0, required=0.5),
        make_block(2, source_top=690.6, source_height=5.0, required=0.5),
    ]
    track_small = VerticalTrack(y_min_pt=690.0, y_max_pt=691.0)
    with pytest.raises(kiwisolver.UnsatisfiableConstraint):
        solve_vertical(blocks, track=track_small)


def test_empty_input_returns_empty_and_indices_survive_sorting() -> None:
    assert solve_vertical([], track=TRACK) == ()
    shuffled = [
        make_block(7, source_top=200.0, source_height=100.0),
        make_block(3, source_top=400.0, source_height=100.0),
        make_block(5, source_top=600.0, source_height=100.0),
    ]
    specs = solve_vertical(shuffled, track=TRACK)
    assert [spec.index for spec in specs] == [5, 3, 7], "top-down order, original indices kept"
    assert_no_overlap_or_escape(specs, TRACK)


def test_stage4_threshold_is_the_prd_value() -> None:
    """The trigger is 1.5pt (PRD §7.4.4); a hairline deficit does not queue Stage 4."""
    assert DEFICIT_TRIGGER_PT == 1.5
    blocks = [make_block(0, source_top=650.0, source_height=200.0, required=200.001)]
    specs = solve_vertical(blocks, track=TRACK)
    assert not specs[0].needs_stage4
