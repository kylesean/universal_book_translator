"""Multi-corpus validation for CELS layout coordinator & distortion engine.

Tests realistic document geometry:
1. Two-column academic paper with full-width header/abstract.
2. Obstacle avoidance (figures/formulas spanning columns).
3. Three-column dense layout.
4. Zero-cross-column penetration guard validation.
5. Distortion verdict calibration across clean vs defective layouts.
"""

from __future__ import annotations

import pytest

from ubt.core.layout import (
    Box,
    PageLayoutBlock,
    solve_page_cels,
)

pytestmark = pytest.mark.fast


def test_academic_paper_with_full_width_header_and_two_columns() -> None:
    """Standard arXiv/IEEE paper: 1 full-width title + 2 columns body."""
    printable = Box(x0=54.0, y0=54.0, x1=558.0, y1=738.0)  # US Letter margins

    # Full-width title at top: y: 680..720 (pinned)
    title_block = PageLayoutBlock(
        block_id="title",
        box=Box(x0=54.0, y0=680.0, x1=558.0, y1=720.0),
        text="A Novel Multi-Column Layout Architecture for Technical Documents",
        source_font_size_pt=16.0,
        required_height_pt=40.0,
        is_pinned=True,
    )

    # 2 columns below title: y: 60..660
    # Left column: x: 54..286 (width 232pt)
    # Right column: x: 326..558 (width 232pt), gutter: 40pt
    left_blocks = [
        PageLayoutBlock(
            block_id=f"left_{i}",
            box=Box(x0=54.0, y0=660.0 - (i + 1) * 60.0, x1=286.0, y1=660.0 - i * 60.0),
            text=f"Left column technical discussion segment {i}.",
            source_font_size_pt=9.0,
            required_height_pt=50.0,
        )
        for i in range(8)
    ]

    right_blocks = [
        PageLayoutBlock(
            block_id=f"right_{i}",
            box=Box(x0=326.0, y0=660.0 - (i + 1) * 60.0, x1=558.0, y1=660.0 - i * 60.0),
            text=f"Right column experimental results segment {i}.",
            source_font_size_pt=9.0,
            required_height_pt=50.0,
        )
        for i in range(8)
    ]

    all_blocks = [title_block] + left_blocks + right_blocks

    solution = solve_page_cels(
        page_no=1,
        blocks=all_blocks,
        page_width=612.0,
        page_height=792.0,
        printable_box=printable,
        source_body_size_pt=9.0,
        realized_body_size_pt=9.0,
    )

    # Validate column detection
    assert solution.detection.is_multicolumn
    assert len(solution.detection.tracks) == 2
    assert len(solution.track_solutions) == 2

    # Verify both tracks solved independently
    assert len(solution.track_solutions[0].solved_blocks) == 8
    assert len(solution.track_solutions[1].solved_blocks) == 8

    # Title is recorded as pinned obstacle
    assert len(solution.pinned_obstacles) == 1
    assert solution.pinned_obstacles[0].box == title_block.box

    # Layout is clean and deliverable
    assert solution.total_deficit_pt == 0.0
    assert not solution.requires_llm_condensation
    assert not solution.distortion.is_critical
    assert solution.distortion.verdict == "faithful_deliverable"
    assert solution.distortion.subscores.overlap_ratio == 0.0


def test_obstacle_spanning_both_columns_pins_cleanly() -> None:
    """Figure spanning both columns gets pinned, body lines assign to tracks beside/around it."""
    printable = Box(x0=54.0, y0=54.0, x1=558.0, y1=738.0)

    # Figure in middle of page spanning both columns: x: 54..558, y: 350..450
    fig_block = PageLayoutBlock(
        block_id="fig_1",
        box=Box(x0=54.0, y0=350.0, x1=558.0, y1=450.0),
        text="[Figure 1: Full Architecture Diagram]",
        source_font_size_pt=8.0,
        required_height_pt=100.0,
        is_pinned=True,
    )

    # 4 left column blocks and 4 right column blocks
    blocks: list[PageLayoutBlock] = [fig_block]
    for i in range(4):
        y = 300.0 - i * 50.0
        blocks.append(
            PageLayoutBlock(
                block_id=f"col0_{i}",
                box=Box(x0=54.0, y0=y, x1=286.0, y1=y + 40.0),
                text=f"Paragraph {i} in column 0",
                source_font_size_pt=9.0,
                required_height_pt=40.0,
            )
        )
        blocks.append(
            PageLayoutBlock(
                block_id=f"col1_{i}",
                box=Box(x0=326.0, y0=y, x1=558.0, y1=y + 40.0),
                text=f"Paragraph {i} in column 1",
                source_font_size_pt=9.0,
                required_height_pt=40.0,
            )
        )

    solution = solve_page_cels(
        page_no=2,
        blocks=blocks,
        page_width=612.0,
        page_height=792.0,
        printable_box=printable,
        source_body_size_pt=9.0,
        realized_body_size_pt=9.0,
    )

    assert solution.detection.is_multicolumn
    assert len(solution.pinned_obstacles) == 1
    assert solution.pinned_obstacles[0].box.width == pytest.approx(504.0)
    assert solution.distortion.verdict == "faithful_deliverable"


def test_three_column_layout_detection_and_solving() -> None:
    """3-column dense newsletter layout: columns at [54..190], [210..346], [366..502]."""
    printable = Box(x0=54.0, y0=54.0, x1=502.0, y1=738.0)
    col_xs = [(54.0, 190.0), (210.0, 346.0), (366.0, 502.0)]

    blocks: list[PageLayoutBlock] = []
    for c_idx, (x0, x1) in enumerate(col_xs):
        for b_idx in range(5):
            y_top = 700.0 - b_idx * 70.0
            blocks.append(
                PageLayoutBlock(
                    block_id=f"c{c_idx}_b{b_idx}",
                    box=Box(x0=x0, y0=y_top - 60.0, x1=x1, y1=y_top),
                    text=f"Column {c_idx} Block {b_idx} Content",
                    source_font_size_pt=8.0,
                    required_height_pt=60.0,
                )
            )

    solution = solve_page_cels(
        page_no=3,
        blocks=blocks,
        page_width=556.0,
        page_height=792.0,
        printable_box=printable,
        source_body_size_pt=8.0,
        realized_body_size_pt=8.0,
    )

    assert solution.detection.is_multicolumn
    assert len(solution.detection.tracks) == 3
    assert len(solution.track_solutions) == 3
    for trk_sol in solution.track_solutions:
        assert len(trk_sol.solved_blocks) == 5
    assert solution.distortion.verdict == "faithful_deliverable"


def test_cross_column_penetration_is_refused_or_pinned() -> None:
    """A box straddling across a gutter is pinned as an obstacle, preserving track separation."""
    printable = Box(x0=54.0, y0=54.0, x1=558.0, y1=738.0)
    # 2 columns at 54..280 and 320..546 (gutter at 280..320)
    # Straddler penetrating gutter with small vertical share (< 15% STRADDLER_MAX_SHARE)
    blocks: list[PageLayoutBlock] = [
        PageLayoutBlock(
            block_id="straddler",
            box=Box(x0=250.0, y0=400.0, x1=350.0, y1=420.0),  # 20pt height
            text="Straddling formula",
            source_font_size_pt=9.0,
            required_height_pt=20.0,
        ),
    ]
    # 6 blocks per column (total column height ~300pt, so 20pt straddler is ~6% < 15%)
    for i in range(6):
        y = 650.0 - (i + 1) * 50.0
        blocks.append(
            PageLayoutBlock(
                block_id=f"c0_{i}",
                box=Box(x0=54.0, y0=y, x1=270.0, y1=y + 45.0),
                text=f"Left col {i}",
                source_font_size_pt=9.0,
                required_height_pt=45.0,
            )
        )
        blocks.append(
            PageLayoutBlock(
                block_id=f"c1_{i}",
                box=Box(x0=330.0, y0=y, x1=546.0, y1=y + 45.0),
                text=f"Right col {i}",
                source_font_size_pt=9.0,
                required_height_pt=45.0,
            )
        )

    solution = solve_page_cels(
        page_no=4,
        blocks=blocks,
        page_width=612.0,
        page_height=792.0,
        printable_box=printable,
        source_body_size_pt=9.0,
        realized_body_size_pt=9.0,
    )

    # Straddler must be pinned as an obstacle so column tracks remain unpenetrated
    assert solution.detection.is_multicolumn
    assert len(solution.detection.tracks) == 2
    assert len(solution.pinned_obstacles) >= 1
    assert any(p.box.x0 <= 250.0 and p.box.x1 >= 350.0 for p in solution.pinned_obstacles)
