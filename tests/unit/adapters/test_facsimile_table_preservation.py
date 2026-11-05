from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pikepdf
import pytest
from pikepdf import Dictionary, Name

from ubt.adapters.pdf.docling_render import (
    DoclingRenderStrategy,
    _sync_l1_geometry_with_artifact,
)
from ubt.core.content.adapt import graph_from_blocks
from ubt.core.content.nodes import AssetIntegrity, AssetNode
from ubt.core.engine.stages.export import apply_render_skip_flags
from ubt.core.ir.models import (
    BlockType,
    BookManifest,
    BoundingBox,
    FlowID,
    IRBlock,
    make_element,
)
from ubt.render.typst.facsimile_renderer import FacsimileRenderer

pytestmark = pytest.mark.fast

_NEEDS_ARTIFACT_TOOLS = pytest.mark.skipif(
    shutil.which("typst") is None or shutil.which("pdftotext") is None,
    reason="typst/pdftotext not installed",
)

_SOURCE_MARKER = "SRCGRIDMARKER"
_TRANSLATED_MARKER = "译表标记"


def _source_pdf(path: Path, *, words: list[tuple[str, float]], font_size: float = 9.0) -> Path:
    """A source page whose text sits at explicit ``(word, y)`` origins."""
    with pikepdf.Pdf.new() as pdf:
        page = pdf.add_blank_page(page_size=(500, 700))
        font = pdf.make_indirect(
            Dictionary(Type=Name.Font, Subtype=Name.Type1, BaseFont=Name.Helvetica)
        )
        page.Resources = Dictionary(Font=Dictionary(F1=font))
        ops = [f"BT /F1 {font_size} Tf"]
        for word, y in words:
            ops.append(f"1 0 0 1 50 {y} Tm ({word}) Tj")
        ops.append("ET")
        page.Contents = pdf.make_stream("\n".join(ops).encode())
        pdf.save(path)
    return path


def _table_block(
    *,
    grid: str,
    translated: str,
    bbox: tuple[float, float, float, float],
    block_id: str = "t1",
) -> IRBlock:
    block = IRBlock(
        element=make_element(
            id=block_id,
            spine_index=1,
            block_type=BlockType.TABLE,
            flow_id=FlowID.MAIN_STORY,
            source_text=grid,
            bbox=BoundingBox(page=1, x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3]),
        )
    )
    block.target_text = translated
    return block


def _render(manifest: BookManifest, blocks: list[IRBlock], out: Path) -> DoclingRenderStrategy:
    strategy = DoclingRenderStrategy(delivery_mode="facsimile", enable_pdf_optimization=False)
    asyncio.run(strategy.render_blocks(manifest, blocks, "zh", out))
    return strategy


def _delivered_text(pdf: Path) -> str:
    result = subprocess.run(
        ["pdftotext", str(pdf), "-"], capture_output=True, text=True, check=True
    )
    return result.stdout


def _asset_integrity(
    blocks: list[IRBlock], strategy: DoclingRenderStrategy, block_id: str
) -> object:
    apply_render_skip_flags(blocks, strategy.last_render_skips)
    graph = graph_from_blocks(blocks, reconstructions=strategy.last_render_reconstructions)
    asset = next(
        node for node in graph.nodes if isinstance(node, AssetNode) and node.id == block_id
    )
    return asset.descriptor.integrity


@_NEEDS_ARTIFACT_TOOLS
def test_a_table_that_cannot_fit_keeps_its_source_in_the_delivered_pdf(
    tmp_path: Path,
) -> None:
    """The canvas's "preserve the source table" must be true at the artifact.

    Regression: the strip plan erased the table's bbox, the canvas then skipped
    the grid because it could not fit, and nothing was drawn or kept — the
    delivered PDF lost the table entirely and the contract reported MISSING.
    """
    source = _source_pdf(tmp_path / "src.pdf", words=[(_SOURCE_MARKER, 320.0)])
    grid = "\n".join(
        ["| Index | Measurement |", "|---|---|"] + [f"| item_{i} | value_{i} |" for i in range(10)]
    )
    translated = "\n".join(
        [f"| 索引 | {_TRANSLATED_MARKER} |", "|---|---|"]
        + [f"| 项_{i} | 值_{i} |" for i in range(10)]
    )
    block = _table_block(grid=grid, translated=translated, bbox=(40.0, 300.0, 460.0, 330.0))
    manifest = BookManifest(
        doc_id="doc_table_kept", title="Table kept", source_path=str(source), metadata={}
    )
    out = tmp_path / "out.pdf"
    strategy = _render(manifest, [block], out)

    text = _delivered_text(out)
    assert _SOURCE_MARKER in text, (
        "the source table's ink is the deliverable when the canvas cannot fit it"
    )
    assert _TRANSLATED_MARKER not in text

    assert ("t1", "table_source_preserved") in strategy.last_render_flags
    assert _asset_integrity([block], strategy, "t1") is AssetIntegrity.PRESERVED_OPAQUE


@_NEEDS_ARTIFACT_TOOLS
def test_a_table_that_fits_replaces_its_source_in_the_delivered_pdf(
    tmp_path: Path,
) -> None:
    """A fitting table keeps the old contract: source erased, translation drawn."""
    source = _source_pdf(
        tmp_path / "src.pdf",
        words=[(_SOURCE_MARKER, 300.0), (_SOURCE_MARKER, 288.0), (_SOURCE_MARKER, 276.0)],
    )
    grid = "\n".join(
        ["| Index | Measurement |", "|---|---|"] + [f"| item_{i} | value_{i} |" for i in range(3)]
    )
    translated = "\n".join(
        ["| 索引 | 测量 |", "|---|---|"] + [f"| 项_{i} | 值_{i} |" for i in range(3)]
    )
    block = _table_block(grid=grid, translated=translated, bbox=(40.0, 100.0, 460.0, 330.0))
    manifest = BookManifest(
        doc_id="doc_table_drawn", title="Table drawn", source_path=str(source), metadata={}
    )
    out = tmp_path / "out.pdf"
    strategy = _render(manifest, [block], out)

    text = _delivered_text(out)
    assert _SOURCE_MARKER not in text, "a redrawn table's source text must be stripped"
    assert "项_0" in text and "值_0" in text

    assert not any(reason == "table_source_preserved" for _, reason in strategy.last_render_flags)
    assert _asset_integrity([block], strategy, "t1") is AssetIntegrity.RECONSTRUCTED


def test_an_unparseable_table_is_drawn_as_prose_not_blanked() -> None:
    """A parse failure is not a fit failure: the old behaviour was escaped text."""
    from ubt.render.typst.facsimile_renderer import FacsimileBlockPlacement

    renderer = FacsimileRenderer()
    placement = FacsimileBlockPlacement(
        block_id="t9",
        page=1,
        x0=50.0,
        y0=100.0,
        x1=250.0,
        y1=120.0,
        text="this is not a pipe table",
        is_table=True,
    )
    content = renderer._placement_content(placement, width_pt=200.0, target_lang="zh")

    assert content != ""
    assert "this is not a pipe table" in content
    assert renderer.preserved_placements == []


def test_the_render_records_the_same_obstacle_edge_as_the_planner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The render's solver records must clamp at the obstacle's ``y1``."""
    import ubt.adapters.pdf.visual_gate as visual_gate
    import ubt.render.layout_solver as layout_solver

    source = _source_pdf(tmp_path / "src.pdf", words=[("page body", 300.0)])
    heading = IRBlock(
        element=make_element(
            id="h1",
            spine_index=1,
            block_type=BlockType.HEADING,
            flow_id=FlowID.MAIN_STORY,
            source_text="A heading",
            bbox=BoundingBox(page=1, x0=40.0, y0=300.0, x1=460.0, y1=320.0),
        ),
        target_text="一个标题",
    )
    prose = IRBlock(
        element=make_element(
            id="p1",
            spine_index=2,
            block_type=BlockType.NARRATIVE,
            flow_id=FlowID.MAIN_STORY,
            source_text="Prose above the heading",
            bbox=BoundingBox(page=1, x0=40.0, y0=400.0, x1=460.0, y1=500.0),
        ),
        target_text="标题上方的正文。",
    )

    captured: dict[str, list[list[dict[str, Any]]]] = {}
    original = layout_solver.solve_page_2d

    def _capture(
        columns: list[list[dict[str, Any]]],
        *,
        source_size_pt: float,
        source_leading_pt: float,
        **kwargs: Any,
    ) -> Any:
        captured["columns"] = columns
        return original(
            columns,
            source_size_pt=source_size_pt,
            source_leading_pt=source_leading_pt,
            **kwargs,
        )

    monkeypatch.setattr(layout_solver, "solve_page_2d", _capture)

    def _fake_render(self: FacsimileRenderer, pages, bg, out, build_dir, **kwargs):  # type: ignore[no-untyped-def]
        out.write_bytes(source.read_bytes())
        return out

    monkeypatch.setattr(FacsimileRenderer, "render_document", _fake_render)
    monkeypatch.setattr(visual_gate, "detect_colliding_l1_pages", lambda *a, **k: {})

    manifest = BookManifest(
        doc_id="doc_obstacle", title="Obstacle", source_path=str(source), metadata={}
    )
    _render(manifest, [heading, prose], tmp_path / "out.pdf")

    record = next(rec for column in captured["columns"] for rec in column if rec.get("id") == "p1")
    # The heading's TOP edge is y1=320; y0=300 would let the prose expand 20pt
    # into the heading's own band.
    assert record["obstacle_top"] == 320.0


def test_a_shrunk_l1_page_is_recorded_at_the_artifact_size_and_budgeted(
    tmp_path: Path,
) -> None:
    """The contract measures the glyphs; geometry must not keep promising the plan."""
    from ubt.render.layout_solver import FitReport

    artifact = _source_pdf(
        tmp_path / "artifact.pdf", words=[("body text here", 300.0)], font_size=7.5
    )
    geometry = [
        {
            "page_no": 1,
            "tier": "L1",
            "delivered_size_pt": 8.9,
            "fits": True,
            "style_exception": False,
        }
    ]
    blocks_by_page = {
        1: [
            IRBlock(
                element=make_element(
                    id="b1",
                    spine_index=1,
                    block_type=BlockType.NARRATIVE,
                    flow_id=FlowID.MAIN_STORY,
                    source_text="body text here",
                    bbox=BoundingBox(page=1, x0=40.0, y0=100.0, x1=460.0, y1=500.0),
                )
            )
        ]
    }
    page_fits: dict[int, FitReport] = {}

    _sync_l1_geometry_with_artifact(
        output_path=artifact,
        page_geometry=geometry,
        page_fits=page_fits,
        blocks_by_page=blocks_by_page,
        group_size_pt=8.9,
    )

    assert geometry[0]["delivered_size_pt"] == 7.5
    assert geometry[0]["style_exception"] is True
    assert geometry[0]["fits"] is False
    assert page_fits[1].reason == "fit_to_box_shrink"
    assert page_fits[1].deficit_pt == pytest.approx(1.4)
