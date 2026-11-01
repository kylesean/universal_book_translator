from __future__ import annotations

from pathlib import Path

import pytest

from ubt.core.ir.models import BlockType, IRBlock, make_element
from ubt.render.typst.document_renderer import TypstDocumentRenderer

pytestmark = pytest.mark.fast


def _code_block(text: str) -> IRBlock:
    return IRBlock(
        element=make_element(
            id="b1",
            spine_index=1,
            block_type=BlockType.CODE,
            source_text=text,
            skip_translate=True,
        ),
        target_text=text,
    )


def test_a_listing_keeps_its_line_breaks_and_monospace(tmp_path: Path) -> None:
    listing = "def load_balance_loss(p):\n    return p"
    source, _stats = TypstDocumentRenderer().build_source([_code_block(listing)], tmp_path)

    assert 'raw("' in source, source
    assert "def load_balance_loss(p):\n    return p" in source


def test_code_metacharacters_survive_the_raw_string(tmp_path: Path) -> None:
    listing = 'x = f("# not markup"); s = "quoted"'
    source, _stats = TypstDocumentRenderer().build_source([_code_block(listing)], tmp_path)

    assert '\\"' in source
    assert source.count('s = \\"quoted\\"') == 1


def test_an_empty_listing_renders_nothing(tmp_path: Path) -> None:
    source, _stats = TypstDocumentRenderer().build_source([_code_block("")], tmp_path)
    assert "raw(" not in source
