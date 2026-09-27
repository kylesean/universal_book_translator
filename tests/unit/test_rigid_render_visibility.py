"""Anchored render fail-closed + loss-visibility guards.

Two confirmed defects:
* A page-level overlay compile failure was recorded as ``("page:N",
  "overlay_compile")`` and the adapter filtered those entries out, so whole
  pages reverted to source with no trace in the quality report.
* A zero-bbox decode path (pypdf/pdfium fallback) produced no zones and the
  engine emitted an unmodified copy of the source PDF as the "translation".
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ubt.adapters.pdf.rigid.typesetter import (
    RigidReport,
    RigidTypesetter,
    _assert_paintable,
    _demote_page_blocks,
    _with_list_marker,
)
from ubt.core.exceptions import DocumentParseError
from ubt.core.ir.models import BlockType, IRBlock


def test_zero_zone_render_fails_closed() -> None:
    report = RigidReport()
    report.skipped.append(("b1", "no_zone"))
    report.skipped.append(("b2", "no_zone"))
    with pytest.raises(DocumentParseError, match="no paintable zones"):
        _assert_paintable({}, report)


def test_zero_zone_allows_empty_document() -> None:
    # No prose blocks at all (nothing skipped as no_zone) is not a failure.
    report = RigidReport()
    report.skipped.append(("b1", "non_prose"))
    _assert_paintable({}, report)


def test_paintable_render_is_not_blocked() -> None:
    report = RigidReport()
    report.skipped.append(("b1", "no_zone"))
    _assert_paintable({1: [(None, 10.0, ["x"])]}, report)  # type: ignore[list-item]


def test_failed_page_overlay_demotes_blocks_to_render_skips() -> None:
    report = RigidReport()
    report.rendered_blocks.extend(["b1", "b2"])
    report.blocks_by_page[3] = ["b1", "b2"]
    _demote_page_blocks(report, 3)
    assert report.rendered_blocks == []
    assert ("b1", "overlay_compile") in report.skipped
    assert ("b2", "overlay_compile") in report.skipped


def test_demote_ignores_other_pages() -> None:
    report = RigidReport()
    report.rendered_blocks.extend(["b1", "b9"])
    report.blocks_by_page[3] = ["b1"]
    _demote_page_blocks(report, 3)
    assert report.rendered_blocks == ["b9"]


def _blk(block_type: BlockType, text: str) -> IRBlock:
    from ubt.core.ir.models import FlowID

    return IRBlock(
        id="b1",
        flow_id=FlowID.MAIN_STORY,
        spine_index=1,
        block_type=block_type,
        source_text=text,
        target_text=text,
    )


def test_list_item_gets_its_bullet_restored() -> None:
    """A translated LIST_ITEM whose bullet glyph was dropped at extraction
    gets a bullet back so the overlay keeps the list structure."""
    from ubt.core.ir.models import BlockType

    block = _blk(BlockType.LIST_ITEM, "Breadth expands hypothesis coverage.")
    assert _with_list_marker(block, "广度拓展假设的覆盖范围。") == "• 广度拓展假设的覆盖范围。"


def test_list_marker_not_doubled_when_model_already_emitted_one() -> None:
    from ubt.core.ir.models import BlockType

    block = _blk(BlockType.LIST_ITEM, "item")
    assert _with_list_marker(block, "• already bulleted") == "• already bulleted"
    assert _with_list_marker(block, "1. numbered item") == "1. numbered item"


def test_narrative_block_never_gets_a_bullet() -> None:
    from ubt.core.ir.models import BlockType

    block = _blk(BlockType.NARRATIVE, "just a paragraph")
    assert _with_list_marker(block, "只是一个段落") == "只是一个段落"


def test_page_overlay_disables_cjk_latin_spacing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The rigid width model is only conservative with CJK/Latin spacing off.

    ``font_metrics.text_width_pt`` sums glyph advances, so Typst's default
    ``cjk-latin-spacing`` adds a gap the model never measured. At the font floor
    the zone clips its overflow, so the excess would be truncated silently
    rather than overlap-flagged — the overlay must turn the spacing off.
    """
    from ubt.adapters.pdf.rigid.typesetter import RigidTypesetter
    from ubt.adapters.pdf.rigid.zones import PageFacts

    typesetter = RigidTypesetter()
    monkeypatch.setattr(typesetter, "_font_tuple", lambda: '"Noto Serif CJK SC"')
    overlay = typesetter._page_overlay(PageFacts(page=1, width=595.0, height=842.0), [])
    assert "cjk-latin-spacing: none" in overlay


def test_geometry_less_blocks_are_recorded_as_skips() -> None:
    """A block with no bbox must be a recorded skip, not a silent drop.

    ``render_coverage`` counts rendered vs planned blocks; a block dropped
    without a skip entry was counted as rendered.
    """
    from ubt.adapters.pdf.rigid.typesetter import RigidTypesetter
    from ubt.core.ir.models import BlockType, BoundingBox, IRBlock

    typesetter = RigidTypesetter()
    no_bbox = IRBlock(id="b1", spine_index=0, block_type=BlockType.NARRATIVE, source_text="x")
    off_page = IRBlock(
        id="b2",
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        source_text="y",
        bbox=BoundingBox(page=99, x0=0.0, y0=0.0, x1=10.0, y1=10.0),
    )
    _paints, report = typesetter._plan_blocks([no_bbox, off_page], {}, {})
    assert (no_bbox.id, "no_bbox") in report.skipped
    assert (off_page.id, "no_page_height") in report.skipped


def test_unrenderable_inline_math_is_a_visible_block_skip() -> None:
    """Never ship a translated box whose math fell back to raw LaTeX.

    The ledger may contain an unsupported command after a model/parser failure.
    Keeping the source paragraph and reporting ``math_unrenderable`` is honest;
    silently painting ``$\\foo{x}$`` as escaped text is not a translation.
    """
    from ubt.adapters.pdf.rigid.zones import Zone
    from ubt.core.ir.models import BoundingBox

    typesetter = RigidTypesetter()
    block = IRBlock(
        id="bad-math",
        spine_index=0,
        block_type=BlockType.NARRATIVE,
        source_text="An unsupported expression.",
        target_text=r"译文含 $\foo{x}$。",
        bbox=BoundingBox(page=1, x0=60.0, y0=600.0, x1=520.0, y1=680.0),
    )
    zone = Zone(
        block_id=block.id,
        page=1,
        x0=60.0,
        y0=600.0,
        x1=520.0,
        y1=680.0,
        base_size=10.0,
    )

    paints, report = typesetter._plan_blocks([block], {block.id: (zone,)}, {1: 842.0})

    assert paints == {}
    assert (block.id, "math_unrenderable") in report.skipped


@pytest.mark.fast
def test_assert_paintable_rejects_geometry_skips() -> None:
    """No paintable zone because no page geometry was decoded must abort."""
    from ubt.adapters.pdf.rigid.typesetter import RigidReport, _assert_paintable
    from ubt.core.exceptions import DocumentParseError

    for reason in ("no_page_height", "no_bbox", "no_zone"):
        report = RigidReport()
        report.skipped.append(("b1", reason))
        with pytest.raises(DocumentParseError):
            _assert_paintable({}, report)


@pytest.mark.fast
def test_assert_paintable_allows_documents_with_nothing_to_translate() -> None:
    """A document whose only blocks are non-prose is legitimately empty."""
    from ubt.adapters.pdf.rigid.typesetter import RigidReport, _assert_paintable

    report = RigidReport()
    report.skipped.append(("img1", "non_prose"))
    report.skipped.append(("fig1", "empty_target"))
    _assert_paintable({}, report)  # must not raise


@pytest.mark.fast
def test_partition_render_skips_separates_intentional_preserved_from_fail_closed() -> None:
    """_partition_render_skip_counts must separate intentional preserved elements
    (policy, non_prose, chrome, footer) from true fail-closed skips (spill, no_zone, math_unrenderable)."""
    from ubt.core.engine.stages.export import _partition_render_skip_counts

    checkpoints = (
        [{"block_id": f"p{i}", "error_flags": ["render_skip:policy"]} for i in range(56)]
        + [{"block_id": f"n{i}", "error_flags": ["render_skip:non_prose"]} for i in range(14)]
        + [{"block_id": "c1", "error_flags": ["render_skip:chrome"]}]
        + [{"block_id": "c2", "error_flags": ["render_skip:chrome"]}]
        + [{"block_id": "f1", "error_flags": ["render_skip:footer"]}]
        + [
            {"block_id": "fc1", "error_flags": ["render_skip:no_zone"]},
            {"block_id": "fc2", "error_flags": ["render_skip:math_unrenderable"]},
            {"block_id": "fc3", "error_flags": ["render_skip:spill"]},
        ]
    )
    fail_closed, preserved = _partition_render_skip_counts(checkpoints)
    assert fail_closed == 3
    assert preserved == 73


def test_ubt_fit_clips_at_the_floor_instead_of_overprinting(tmp_path: Path) -> None:
    """A block that cannot fit even at the floor must not overprint the next.

    ``#ubt-fit`` shrank to ``min-sz`` and then emitted ``clip: false``
    unconditionally, so an oversized block flowed past its zone and garbled the
    block below. The rendered artifact is the ground truth: the top zone is
    oversized and the band below its clip limit must stay free of text ink.
    """
    import shutil
    import subprocess

    if shutil.which("typst") is None:
        pytest.skip("typst binary unavailable")
    pdfium = pytest.importorskip("pypdfium2")
    pytest.importorskip("PIL")

    from ubt.adapters.pdf.rigid.typesetter import RigidTypesetter

    body = " ".join(["overflow"] * 200)
    typ = (
        "#set page(width: 200pt, height: 200pt, margin: 0pt, fill: white)\n"
        '#set text(size: 12pt, font: "Liberation Serif")\n'
        + RigidTypesetter._typst_fit_preamble()
        + "\n#place(top + left, dx: 0pt, dy: 0pt)"
        f"[#ubt-fit(200pt, 40pt, 12pt, 12pt, 0pt)[{body}]]\n"
        "#place(top + left, dx: 0pt, dy: 60pt)"
        '[#rect(width: 200pt, height: 40pt, fill: rgb("#ff0000"))]\n'
    )
    src = tmp_path / "fit.typ"
    out = tmp_path / "fit.pdf"
    src.write_text(typ, encoding="utf-8")
    subprocess.run(["typst", "compile", str(src), str(out)], check=True)

    image = pdfium.PdfDocument(str(out))[0].render(scale=1.0).to_pil().convert("RGB")
    # Below the top zone's 42.5pt clip limit only the red band is painted; a dark
    # (text) pixel there means the block overprinted.
    dark = sum(
        1
        for y in range(50, image.height)
        for x in range(image.width)
        if all(channel < 128 for channel in image.getpixel((x, y)))
    )
    assert dark == 0, f"{dark} overprinted text pixels below the zone"
