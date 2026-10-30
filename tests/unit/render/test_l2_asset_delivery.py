"""L2 (page-locked) pages must draw their assets, never typeset their paths.

The delivery defect this pins is measured, not hypothetical: on a real 26-page
BSIM-CMG chapter (``chapter-3.pdf``) 25 of 26 pages routed to L2, and the L2
channel carried no background layer and had no ``IMAGE`` branch at all. Each
asset block's ``source_text`` *is* the extracted file's absolute path (that is
how the parser stores it), so the renderer set those paths as body prose. The
delivered page 4 read, verbatim:

    /home/kyle/.cache/ubt/docling_cache/assets/fc1d7…/pic_p4_22.dpi300-tight2.png

while the contract reported 94 ``MISSING`` assets for figures the reader could
not see. Two independent things had to be true afterwards: the ink has to be on
the page, and the contract has to say so.

An L1 facsimile page does *not* need this: its stripped background layer carries
the asset's source ink at its exact box, which is why the copier is only wired
into the L2 branch of the hybrid document.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ubt.core.ir.models import BlockType, BoundingBox, IRBlock, StyleMeta, make_element
from ubt.render.typst.facsimile_renderer import (
    FacsimilePageSpec,
    FacsimileRenderer,
)
from ubt.render.typst.page_locked_renderer import (
    PageGeometry,
    PageLockedPageSpec,
    PageLockedReflowRenderer,
    _block_height,
    make_asset_copier,
)

pytestmark = pytest.mark.fast

_LEAK_PATH = "/home/kyle/.cache/ubt/docling_cache/assets/fc1d7ab/pic_p4_22.dpi300-tight2.png"


def _asset_block(
    block_id: str,
    path: str = _LEAK_PATH,
    *,
    page: int = 4,
    bbox: tuple[float, float, float, float] = (36.0, 500.0, 300.0, 700.0),
) -> IRBlock:
    """An image block in the exact shape the PDF parser writes: path as text."""
    bb = BoundingBox(page=page, x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3])
    el = make_element(
        id=block_id, spine_index=1, block_type=BlockType.IMAGE, source_text=path, bbox=bb
    )
    return IRBlock(element=el, style=StyleMeta())


def _prose_block(block_id: str, text: str, *, page: int = 4) -> IRBlock:
    bb = BoundingBox(page=page, x0=36.0, y0=400.0, x1=500.0, y1=470.0)
    el = make_element(
        id=block_id, spine_index=1, block_type=BlockType.NARRATIVE, source_text=text, bbox=bb
    )
    return IRBlock(element=el, target_text=text, style=StyleMeta())


def _geom(columns: int = 1) -> PageGeometry:
    return PageGeometry(
        page_w=540.0,
        page_h=666.0,
        margin_top=40.0,
        margin_bottom=40.0,
        margin_left=40.0,
        margin_right=40.0,
        columns=columns,
    )


def _spec(blocks: list[IRBlock], *, columns: int = 1) -> PageLockedPageSpec:
    return PageLockedPageSpec(
        page_no=4,
        geometry=_geom(columns),
        uniform_size_pt=9.0,
        leading_em=0.35,
        par_spacing_em=0.65,
        blocks=blocks,
    )


class TestAssetPathNeverBecomesProse:
    """The path is machinery. Under no configuration may it be typeset."""

    def test_no_copier_drops_the_asset_instead_of_printing_the_path(self) -> None:
        markup = PageLockedReflowRenderer().build_page_markup(_spec([_asset_block("a1")]))
        assert "docling_cache" not in markup
        assert ".png" not in markup
        assert _LEAK_PATH not in markup
        assert "pic_p4_22" not in markup

    def test_a_copier_draws_the_image(self, tmp_path: Path) -> None:
        asset = tmp_path / "assets" / "pic_p4_22.dpi300-tight2.png"
        asset.parent.mkdir()
        asset.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 40)
        build = tmp_path / "build"
        copier = make_asset_copier(build, asset.parent)

        markup = PageLockedReflowRenderer().build_page_markup(
            _spec([_asset_block("a1", str(asset))]), asset_copier=copier
        )
        assert "image(" in markup
        # The reference is the content-addressed copy that actually exists in
        # the build dir next to the .typ, not the absolute source path.
        name = markup.split('image("')[1].split('"')[0]
        assert (build / name).exists()
        assert name.startswith("pic_p4_22.dpi300-tight2-")
        assert name.endswith(".png")
        assert str(asset) not in markup
        assert "docling_cache" not in markup

    def test_the_image_is_clamped_to_the_column_width(self, tmp_path: Path) -> None:
        """A wider-than-column figure must not overflow the text block."""
        asset = tmp_path / "wide.png"
        asset.write_bytes(b"\x89PNG\r\n\x1a\n")
        copier = make_asset_copier(tmp_path / "build", tmp_path)
        # bbox width 600pt on a 540pt page: content width is 460pt.
        block = _asset_block("a1", str(asset), bbox=(20.0, 500.0, 620.0, 700.0))
        markup = PageLockedReflowRenderer().build_page_markup(_spec([block]), asset_copier=copier)
        width = float(markup.split("width: ")[-1].split("pt")[0])
        assert width <= 540.0 - 40.0 - 40.0 + 0.01

    def test_a_missing_asset_file_is_dropped_not_broken(self, tmp_path: Path) -> None:
        """``copier`` returns ``None`` for an unreadable path -> block dropped."""
        copier = make_asset_copier(tmp_path / "build", tmp_path / "nope")
        markup = PageLockedReflowRenderer().build_page_markup(
            _spec([_asset_block("a1", str(tmp_path / "gone.png"))]), asset_copier=copier
        )
        assert "image(" not in markup
        assert "gone.png" not in markup

    def test_document_markup_threads_the_copier_to_every_page(self, tmp_path: Path) -> None:
        asset = tmp_path / "p.png"
        asset.write_bytes(b"\x89PNG\r\n\x1a\n")
        copier = make_asset_copier(tmp_path / "build", tmp_path)
        renderer = PageLockedReflowRenderer()
        markup = renderer.build_document_markup(
            [_spec([_asset_block("a1", str(asset)), _prose_block("p1", "正文")])],
            asset_copier=copier,
        )
        assert "image(" in markup
        assert "正文" in markup
        assert "docling_cache" not in markup


class TestAssetCopier:
    def test_name_is_content_addressed_against_same_basename_collisions(
        self, tmp_path: Path
    ) -> None:
        """Flat build dirs plus shared basenames silently overwrite each other."""
        root = tmp_path / "assets"
        root.mkdir()
        one = root / "pic_p4_22.png"
        two = root / "sub" / "pic_p4_22.png"
        two.parent.mkdir()
        one.write_bytes(b"A" * 8)
        two.write_bytes(b"B" * 8)
        copier = make_asset_copier(tmp_path / "build", root)
        n1, n2 = copier(str(one)), copier(str(two))
        assert n1 and n2 and n1 != n2
        assert (tmp_path / "build" / n1).read_bytes() == b"A" * 8
        assert (tmp_path / "build" / n2).read_bytes() == b"B" * 8

    def test_the_same_asset_twice_copies_once(self, tmp_path: Path) -> None:
        asset = tmp_path / "x.png"
        asset.write_bytes(b"SAME")
        copier = make_asset_copier(tmp_path / "build")
        assert copier(str(asset)) == copier(str(asset))

    def test_relative_names_resolve_under_the_assets_root(self, tmp_path: Path) -> None:
        root = tmp_path / "assets"
        root.mkdir()
        (root / "pic.png").write_bytes(b"R")
        copier = make_asset_copier(tmp_path / "build", root)
        assert copier("pic.png") is not None

    def test_a_missing_build_dir_is_created_rather_than_dropping_every_asset(
        self, tmp_path: Path
    ) -> None:
        """The copier must not need the render stage to have run first.

        A caller that builds markup before ``build_dir`` exists (a preflight, a
        preview, a test) would otherwise get ``None`` for every asset and an
        artifact with no figures at all — silent loss through a missing
        ``mkdir``, which is the exact failure this copier exists to prevent.
        """
        asset = tmp_path / "assets" / "pic.png"
        asset.parent.mkdir()
        asset.write_bytes(b"\x89PNG\r\n\x1a\n")
        build = tmp_path / "not" / "made" / "yet"
        assert not build.exists()
        name = make_asset_copier(build)(str(asset))
        assert name is not None
        assert (build / name).exists()

    def test_empty_and_unreadable_inputs_return_none(self, tmp_path: Path) -> None:
        copier = make_asset_copier(tmp_path / "build")
        assert copier("") is None
        assert copier("/nonexistent/pic.png") is None


class TestHeightBudgetCountsAssets:
    """An asset occupies vertical space, or the page silently overflows."""

    def test_an_image_reserves_its_scaled_height(self) -> None:
        block = _asset_block("a1", bbox=(36.0, 500.0, 300.0, 700.0))
        h = _block_height(block, col_w=460.0, size=9.0, leading=0.35, spacing=0.65, tracking=0.0)
        assert h > 0.0, "a 200pt figure must not budget as zero height"

    def test_a_full_width_image_budgets_its_whole_box(self) -> None:
        """At or below the column width the height is the source region's height."""
        # 264pt wide (<= col_w) and 200pt tall: no scaling, so ~200pt reserved.
        block = _asset_block("a1", bbox=(36.0, 500.0, 300.0, 700.0))
        h = _block_height(block, col_w=460.0, size=9.0, leading=0.35, spacing=0.65, tracking=0.0)
        assert 195.0 < h < 210.0, h

    def test_an_image_wider_than_the_column_budgets_the_scaled_height(self) -> None:
        """Scaled down to the column, its height shrinks by the same factor."""
        # 920pt wide, 200pt tall, into a 460pt column -> half scale -> ~100pt.
        block = _asset_block("a1", bbox=(20.0, 500.0, 940.0, 700.0))
        h = _block_height(block, col_w=460.0, size=9.0, leading=0.35, spacing=0.65, tracking=0.0)
        assert 95.0 < h < 110.0, h

    def test_a_zero_area_image_budgets_nothing(self) -> None:
        flat = _asset_block("f", bbox=(36.0, 500.0, 36.0, 700.0))
        h = _block_height(flat, col_w=460.0, size=9.0, leading=0.35, spacing=0.65, tracking=0.0)
        assert h == 0.0

    def test_prose_budget_is_unchanged_by_the_asset_branch(self) -> None:
        prose = _prose_block("p1", "一句话。")
        h = _block_height(prose, col_w=460.0, size=9.0, leading=0.35, spacing=0.65, tracking=0.0)
        assert h > 0.0


class TestHybridFacsimileOnlyL2GetsCopier:
    """L1 pages keep their figures in the background layer; L2 pages must draw."""

    def test_build_markup_passes_the_copier_and_l1_still_uses_the_background(
        self, tmp_path: Path
    ) -> None:
        asset = tmp_path / "pic_p4_22.png"
        asset.write_bytes(b"\x89PNG\r\n\x1a\n")
        copier = make_asset_copier(tmp_path / "build", tmp_path)

        l2 = PageLockedPageSpec(
            page_no=4,
            geometry=_geom(),
            uniform_size_pt=9.0,
            leading_em=0.35,
            par_spacing_em=0.65,
            blocks=[_asset_block("a1", str(asset))],
        )
        pages = [
            FacsimilePageSpec(
                page_no=1,
                page_w=540.0,
                page_h=666.0,
                uniform_size_pt=9.0,
                leading_em=0.35,
                tracking_em=0.0,
                placements=[],
                tier="L1",
            ),
            FacsimilePageSpec(
                page_no=4,
                page_w=540.0,
                page_h=666.0,
                uniform_size_pt=9.0,
                leading_em=0.35,
                tracking_em=0.0,
                placements=[],
                tier="L2",
                l2_spec=l2,
            ),
        ]
        markup = FacsimileRenderer().build_markup(pages, "bg_stripped.pdf", asset_copier=copier)

        # L1 still pages over the stripped background, and names no asset file.
        assert 'background: image("bg_stripped.pdf", page: 1' in markup
        # L2 draws the figure instead of printing its path.
        assert "pic_p4_22" in markup
        assert "docling_cache" not in markup
        assert str(asset) not in markup

    def test_without_a_copier_an_l2_page_drops_the_asset_rather_than_leaking(
        self, tmp_path: Path
    ) -> None:
        l2 = PageLockedPageSpec(
            page_no=4,
            geometry=_geom(),
            uniform_size_pt=9.0,
            leading_em=0.35,
            par_spacing_em=0.65,
            blocks=[_asset_block("a1")],
        )
        pages = [
            FacsimilePageSpec(
                page_no=4,
                page_w=540.0,
                page_h=666.0,
                uniform_size_pt=9.0,
                leading_em=0.35,
                tracking_em=0.0,
                placements=[],
                tier="L2",
                l2_spec=l2,
            )
        ]
        markup = FacsimileRenderer().build_markup(pages, "bg_stripped.pdf")
        assert "docling_cache" not in markup
        assert ".png" not in markup
