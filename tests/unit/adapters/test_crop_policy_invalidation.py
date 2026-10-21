"""A crop-policy change must reach documents that were already converted.

The delivered crop is cut during *extraction* and the resulting IMAGE blocks are
cached: a block's ``source_text`` is the asset path. So there are two independent
caches to bust, and missing either one makes a crop fix invisible on exactly the
documents it was found on:

* the **block cache**, keyed by ``docling_identity()`` — if the crop code is not
  part of that identity, ``cached_blocks`` returns the old IMAGE blocks;
* the **asset filename**, built from ``FIGURE_ASSET_DPI_TAG`` — the asset
  directory is keyed by the source PDF's sha alone, and the extraction path
  skips the crop when ``asset_path.exists()``, so an unchanged name replays the
  previous PNG.

Pinned here so a future margin/DPI change cannot silently look like a no-op.
"""

from __future__ import annotations

import re

import pytest

pytestmark = pytest.mark.fast


def test_the_block_cache_identity_covers_the_crop_code() -> None:
    """``visual_scalpel`` shapes the cached blocks, so it must key the cache."""
    from ubt.adapters.pdf.analysis_cache import docling_identity

    assert "visual_scalpel" in docling_identity()


def test_the_asset_name_carries_the_crop_policy() -> None:
    """The delivered-asset tag names the policy, so a change cuts new files."""
    from ubt.adapters.pdf.visual_scalpel import (
        DELIVERED_CROP_POLICY_TAG,
        FIGURE_ASSET_DPI,
        FIGURE_ASSET_DPI_TAG,
    )

    assert DELIVERED_CROP_POLICY_TAG
    assert DELIVERED_CROP_POLICY_TAG in FIGURE_ASSET_DPI_TAG
    assert str(FIGURE_ASSET_DPI) in FIGURE_ASSET_DPI_TAG


def test_the_policy_tag_is_filename_safe() -> None:
    """The tag lands in a filename: no separators, no spaces."""
    from ubt.adapters.pdf.visual_scalpel import DELIVERED_CROP_POLICY_TAG

    assert re.fullmatch(r"[A-Za-z0-9_-]+", DELIVERED_CROP_POLICY_TAG)


def test_the_delivered_helpers_are_distinct_from_the_prompt_crop() -> None:
    """Two entry points, so a caller cannot silently get the wrong margin."""
    from ubt.adapters.pdf import visual_scalpel

    assert visual_scalpel.delivered_crop_pil is not visual_scalpel.crop_block_pil
    # The delivered one is what the two delivered-asset call sites import.
    import inspect

    src = inspect.getsource(visual_scalpel.delivered_crop_pil)
    assert "crop_block_pil" in src
