from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from pdf_builders import write_text_pdf

from ubt.adapters.pdf.docling_blocks import (
    code_text_layer_is_lossy,
    crop_lossy_code_asset,
)
from ubt.core.ir.models import BlockType

pytestmark = pytest.mark.fast


def _item(page: int, left: float, bottom: float, right: float, top: float) -> SimpleNamespace:
    return SimpleNamespace(
        prov=[SimpleNamespace(page_no=page, bbox=SimpleNamespace(l=left, b=bottom, r=right, t=top))]
    )


def test_a_long_single_line_listing_is_lossy() -> None:
    assert code_text_layer_is_lossy("def load balance loss(router probs): " + "x = 1 " * 40)


def test_a_listing_with_line_breaks_is_not_lossy() -> None:
    text = "\n".join(f"line {i} of a real listing with plenty of length" for i in range(10))
    assert not code_text_layer_is_lossy(text)


def test_short_prose_is_never_lossy() -> None:
    assert not code_text_layer_is_lossy("x = load_balance_loss(router_probs)")


def test_a_lossy_listing_is_recovered_as_a_source_crop(tmp_path: Path) -> None:
    pdf = write_text_pdf(tmp_path / "code.pdf", [["def load balance loss(p):", "    return p"]])
    assets = tmp_path / "assets"
    block = crop_lossy_code_asset(_item(1, 54.0, 600.0, 400.0, 700.0), assets, pdf, ordinal=3)
    assert block is not None
    assert block.block_type is BlockType.IMAGE
    assert block.skip_translate is True
    asset = Path(block.source_text)
    assert asset.exists()
    assert "dpi300" in asset.name


def test_an_unusable_region_keeps_the_text_block(tmp_path: Path) -> None:
    assert crop_lossy_code_asset(_item(1, 0, 0, 0, 0), tmp_path, None, 1) is None
    pdf = write_text_pdf(tmp_path / "tiny.pdf", [["x"]])
    assert crop_lossy_code_asset(_item(1, 0.0, 0.0, 5.0, 5.0), tmp_path, pdf, 1) is None
