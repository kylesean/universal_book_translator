from __future__ import annotations

import re

import pytest

pytestmark = pytest.mark.fast


def test_the_block_cache_identity_covers_the_crop_code() -> None:
    from ubt.adapters.pdf.analysis_cache import docling_identity

    assert "visual_scalpel" in docling_identity()


def test_the_asset_name_carries_the_crop_policy() -> None:
    from ubt.adapters.pdf.visual_scalpel import (
        DELIVERED_CROP_POLICY_TAG,
        FIGURE_ASSET_DPI,
        FIGURE_ASSET_DPI_TAG,
    )

    assert DELIVERED_CROP_POLICY_TAG
    assert DELIVERED_CROP_POLICY_TAG in FIGURE_ASSET_DPI_TAG
    assert str(FIGURE_ASSET_DPI) in FIGURE_ASSET_DPI_TAG


def test_the_policy_tag_is_filename_safe() -> None:
    from ubt.adapters.pdf.visual_scalpel import DELIVERED_CROP_POLICY_TAG

    assert re.fullmatch(r"[A-Za-z0-9_-]+", DELIVERED_CROP_POLICY_TAG)


def test_the_delivered_helpers_are_distinct_from_the_prompt_crop() -> None:
    from ubt.adapters.pdf import visual_scalpel

    assert visual_scalpel.delivered_crop_pil is not visual_scalpel.crop_block_pil
    import inspect

    src = inspect.getsource(visual_scalpel.delivered_crop_pil)
    assert "crop_block_pil" in src
