"""CELS layout coordinator unit tests: Stage 1 to Stage 3B chaining."""

from __future__ import annotations

import pytest

from ubt.core.layout import Box, PageLayoutBlock, solve_page_cels

pytestmark = pytest.mark.fast


def test_single_column_page_solves_cleanly() -> None:
    printable = Box(x0=54.0, y0=54.0, x1=558.0, y1=738.0)
    blocks = [
        PageLayoutBlock(
            block_id=f"b{i}",
            box=Box(x0=54.0, y0=60.0 + i * 40.0, x1=558.0, y1=60.0 + i * 40.0 + 30.0),
            text=f"段落文本 {i}",
            source_font_size_pt=10.0,
            required_height_pt=30.0,
        )
        for i in range(5)
    ]

    sol = solve_page_cels(
        page_no=1,
        blocks=blocks,
        page_width=612.0,
        page_height=792.0,
        printable_box=printable,
        source_body_size_pt=10.0,
        realized_body_size_pt=10.0,
    )

    assert not sol.detection.is_multicolumn
    assert len(sol.track_solutions) == 1
    track_sol = sol.track_solutions[0]
    assert len(track_sol.solved_blocks) == 5
    assert sol.total_deficit_pt == 0.0
    assert not sol.requires_llm_condensation
    assert not sol.distortion.is_critical
    assert sol.distortion.verdict == "faithful_deliverable"


def test_two_column_page_partitions_tracks_and_solves_independently() -> None:
    printable = Box(x0=54.0, y0=54.0, x1=558.0, y1=738.0)
    # Left column: x: 54..280, Right column: x: 332..558 (gutter 52pt)
    blocks: list[PageLayoutBlock] = []
    for i in range(4):
        y = 60.0 + i * 50.0
        blocks.append(
            PageLayoutBlock(
                block_id=f"left_{i}",
                box=Box(x0=54.0, y0=y, x1=280.0, y1=y + 40.0),
                text=f"左栏文本 {i}",
                source_font_size_pt=9.0,
                required_height_pt=40.0,
            )
        )
        blocks.append(
            PageLayoutBlock(
                block_id=f"right_{i}",
                box=Box(x0=332.0, y0=y, x1=558.0, y1=y + 40.0),
                text=f"右栏文本 {i}",
                source_font_size_pt=9.0,
                required_height_pt=40.0,
            )
        )

    sol = solve_page_cels(
        page_no=2,
        blocks=blocks,
        page_width=612.0,
        page_height=792.0,
        printable_box=printable,
        source_body_size_pt=9.0,
        realized_body_size_pt=9.0,
    )

    assert sol.detection.is_multicolumn
    assert len(sol.detection.tracks) == 2
    assert len(sol.track_solutions) == 2
    assert len(sol.track_solutions[0].solved_blocks) == 4
    assert len(sol.track_solutions[1].solved_blocks) == 4
    assert sol.total_deficit_pt == 0.0
    assert not sol.distortion.is_critical


def test_overfull_track_flags_condensation_and_reports_deficit() -> None:
    # Printable height = 200pt (50..250)
    printable = Box(x0=50.0, y0=50.0, x1=500.0, y1=250.0)
    # 5 blocks requiring 50pt each = 250pt + gaps > 200pt -> deficit!
    blocks = [
        PageLayoutBlock(
            block_id=f"b{i}",
            box=Box(x0=50.0, y0=50.0 + i * 30.0, x1=500.0, y1=50.0 + i * 30.0 + 25.0),
            text=f"长正文段落 {i}",
            source_font_size_pt=10.0,
            required_height_pt=50.0,
        )
        for i in range(5)
    ]

    sol = solve_page_cels(
        page_no=3,
        blocks=blocks,
        page_width=550.0,
        page_height=300.0,
        printable_box=printable,
        source_body_size_pt=10.0,
        realized_body_size_pt=10.0,
    )

    assert sol.total_deficit_pt > 1.5
    assert sol.requires_llm_condensation
    # Deficit blocks are identified
    deficit_blocks = [
        spec for spec in sol.track_solutions[0].solved_blocks if spec.deficit_pt > 0.0
    ]
    assert len(deficit_blocks) > 0

    # Stage 4 requests generated with valid budgets
    assert len(sol.stage4_requests) > 0
    for req in sol.stage4_requests:
        assert req.char_budget > 0
        assert req.deficit_pt > 0.0
        assert req.allocated_height_pt >= 0.0

    fit_specs = sol.to_container_fit_specs()
    assert len(fit_specs) == len(sol.stage4_requests)
    assert all(fs.max_target_chars > 0 for fs in fit_specs)


def test_stage4_condensation_loop_resolves_deficit() -> None:
    """Stage 4 closed-loop: deficit triggers character budget -> condensation -> re-solve clears deficit."""
    from ubt.render.typst.container_repair import repair_bottleneck_containers

    printable = Box(x0=50.0, y0=50.0, x1=500.0, y1=200.0)  # 150pt printable height
    # 3 blocks requiring 60pt each = 180pt > 150pt
    verbose_texts = [
        "我们可以看到，在大多数情况下，由于这个原因，系统需要进行复杂的分析和处理，以确保最终质量达到预期标准。",
        "根据上述内容，在当前情况下，我们可以看到相关的内容已经通过测试，并且由此可见，进行优化的效果非常显著。",
        "由此可见，针对这些技术指标，可以通过使用先进算法进行计算，从而在大多数情况下获得更好的性能。",
    ]
    blocks = [
        PageLayoutBlock(
            block_id=f"seg_{i}",
            box=Box(x0=50.0, y0=150.0 - i * 45.0, x1=500.0, y1=150.0 - i * 45.0 + 40.0),
            text=verbose_texts[i],
            source_font_size_pt=10.0,
            required_height_pt=60.0,
        )
        for i in range(3)
    ]

    # Initial solve fails with deficit
    sol1 = solve_page_cels(
        page_no=4,
        blocks=blocks,
        page_width=550.0,
        page_height=250.0,
        printable_box=printable,
        source_body_size_pt=10.0,
        realized_body_size_pt=10.0,
    )
    assert sol1.requires_llm_condensation
    assert len(sol1.stage4_requests) > 0

    # Stage 4: Run condensation via heuristic/LLM
    fit_specs = sol1.to_container_fit_specs()
    repaired = repair_bottleneck_containers(fit_specs)
    assert len(repaired) > 0

    # Re-solve with condensed text and reduced required height (fits in available height)
    condensed_blocks = [
        PageLayoutBlock(
            block_id=b.block_id,
            box=b.box,
            text=repaired.get(b.block_id, b.text),
            source_font_size_pt=b.source_font_size_pt,
            # Each condensed block now fits within 40pt
            required_height_pt=40.0,
        )
        for b in blocks
    ]

    sol2 = solve_page_cels(
        page_no=4,
        blocks=condensed_blocks,
        page_width=550.0,
        page_height=250.0,
        printable_box=printable,
        source_body_size_pt=10.0,
        realized_body_size_pt=10.0,
    )

    # Deficit completely resolved!
    assert sol2.total_deficit_pt == 0.0
    assert not sol2.requires_llm_condensation
    assert len(sol2.stage4_requests) == 0
    assert sol2.distortion.verdict == "faithful_deliverable"
