"""Unit tests for figure subcaption parsing, panel grouping, and grid generation."""

from __future__ import annotations

import pytest

from ubt.core.figures.detector import (
    calculate_grid_columns,
    detect_figure_panels,
    extract_main_and_subcaptions,
    parse_subcaptions,
)
from ubt.core.figures.generator import (
    render_figure_grid_html,
    render_figure_grid_typst,
)
from ubt.core.figures.models import (
    FigurePanel,
    SubFigure,
)
from ubt.core.ir.models import BlockType, IRBlock, make_element

pytestmark = pytest.mark.fast


def _make_block(
    btype: BlockType, text: str, asset_id: str | None = None, block_id: str = "b1"
) -> IRBlock:
    elem = make_element(
        id=block_id,
        spine_index=1,
        block_type=btype,
        source_text=asset_id or text,
    )
    # Only ``element`` and ``target_text`` are fields of ``IRBlock``; the rest
    # were silently ignored by ``extra="ignore"``.
    return IRBlock(element=elem, target_text=text)


def test_extract_main_and_subcaptions_english() -> None:
    text = "Figure 4: Ablation study. (a) Learning rate decay. (b) Batch size variation."
    main, subs = extract_main_and_subcaptions(text)
    assert main == "Figure 4: Ablation study."
    assert len(subs) == 2
    assert subs[0] == ("(a)", "Learning rate decay.")
    assert subs[1] == ("(b)", "Batch size variation.")


def test_extract_main_and_subcaptions_cjk() -> None:
    text = "图 3: 架构消融对比。（a）标准自注意力机制；（b）线性化注意力变体。"
    main, subs = extract_main_and_subcaptions(text)
    assert main == "图 3: 架构消融对比。"
    assert len(subs) == 2
    assert subs[0] == ("(a)", "标准自注意力机制")
    assert subs[1] == ("(b)", "线性化注意力变体。")


def test_extract_subcaptions_delimiters() -> None:
    text = "Figure 2: Subfigures without parens. a) Original image, b) Filtered mask."
    main, subs = extract_main_and_subcaptions(text)
    assert main == "Figure 2: Subfigures without parens."
    assert len(subs) == 2
    assert subs[0] == ("(a)", "Original image")
    assert subs[1] == ("(b)", "Filtered mask.")


def test_parse_subcaptions_convenience() -> None:
    text = "Fig 1: (a) Left (b) Center (c) Right"
    subs = parse_subcaptions(text)
    assert len(subs) == 3
    assert [s[0] for s in subs] == ["(a)", "(b)", "(c)"]
    assert [s[1] for s in subs] == ["Left", "Center", "Right"]


def test_calculate_grid_columns() -> None:
    assert calculate_grid_columns(1) == 1
    assert calculate_grid_columns(2) == 2
    assert calculate_grid_columns(3) == 3
    assert calculate_grid_columns(4) == 2  # 2x2 grid preferred
    assert calculate_grid_columns(6) == 3  # 2x3 grid
    assert calculate_grid_columns(8, max_cols=3) == 3


def test_detect_figure_panels_adjacent_images() -> None:
    blocks = [
        _make_block(BlockType.IMAGE, "fig1.png", asset_id="fig1.png", block_id="img1"),
        _make_block(BlockType.IMAGE, "fig2.png", asset_id="fig2.png", block_id="img2"),
        _make_block(
            BlockType.NARRATIVE,
            "Figure 1: Comparison. (a) Baseline model. (b) Proposed model.",
            block_id="cap1",
        ),
    ]

    panels = detect_figure_panels(blocks)
    assert len(panels) == 1
    p = panels[0]
    assert p.is_composite is True
    assert p.columns == 2
    assert p.main_caption == "Figure 1: Comparison."
    assert len(p.subfigures) == 2
    assert p.subfigures[0].label == "(a)"
    assert p.subfigures[0].caption == "Baseline model."
    assert p.subfigures[1].label == "(b)"
    assert p.subfigures[1].caption == "Proposed model."


def test_detect_figure_panels_single_image() -> None:
    blocks = [
        _make_block(BlockType.IMAGE, "diag.png", asset_id="diag.png", block_id="img1"),
        _make_block(
            BlockType.NARRATIVE, "Figure 5: High level overview of system.", block_id="cap1"
        ),
    ]

    panels = detect_figure_panels(blocks)
    assert len(panels) == 1
    p = panels[0]
    assert p.is_composite is False
    assert p.columns == 1
    assert p.main_caption == "Figure 5: High level overview of system."
    assert len(p.subfigures) == 1
    assert p.subfigures[0].asset == "diag.png"


def test_render_figure_grid_typst() -> None:
    panel = FigurePanel(
        panel_id="panel-1",
        main_caption="Figure 3: System comparison.",
        subfigures=[
            SubFigure(asset="img1.png", label="(a)", caption="Left panel"),
            SubFigure(asset="img2.png", label="(b)", caption="Right panel"),
        ],
        columns=2,
        gutter_pt=10.0,
        anchor_id="fig-comp",
    )

    markup = render_figure_grid_typst(panel)
    assert "#figure(" in markup
    assert "grid(" in markup
    assert "columns: (1fr, 1fr)" in markup
    assert "gutter: 10.0pt" in markup
    assert 'image("img1.png", width: 100%)' in markup
    assert "*(a)* Left panel" in markup
    assert 'image("img2.png", width: 100%)' in markup
    assert "*(b)* Right panel" in markup
    assert "caption: [Figure 3: System comparison.]" in markup
    assert "<fig-comp>" in markup


def test_render_figure_grid_html() -> None:
    panel = FigurePanel(
        panel_id="panel-1",
        main_caption="System performance",
        subfigures=[
            SubFigure(asset="loss.png", label="(a)", caption="Loss"),
            SubFigure(asset="acc.png", label="(b)", caption="Accuracy"),
        ],
        columns=2,
        gutter_pt=8.0,
        anchor_id="fig-perf",
    )

    markup = render_figure_grid_html(panel)
    assert '<figure class="ubt-figure-grid" id="fig-perf"' in markup
    assert "grid-template-columns: repeat(2, 1fr)" in markup
    assert "gap: 8.0pt" in markup
    assert '<div class="ubt-subfigure"' in markup
    assert '<img src="loss.png" alt="(a) Loss"' in markup
    assert "<strong>(a)</strong> Loss" in markup
    assert '<figcaption class="ubt-main-caption"' in markup
    assert "System performance" in markup
