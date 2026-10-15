"""A text code listing is typeset as code, never as prose.

Typst folds a markup newline into a space and prose escaping leaves the
listing set in the body face, so the old prose path delivered a listing as one
justified paragraph — no line structure, no monospace. The listing is passed
to Typst ``raw`` as a plain string, which no markup or code mode can enter.
"""

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
    # The body must reach the compiler with its newline intact, not folded.
    assert "def load_balance_loss(p):\n    return p" in source


def test_code_metacharacters_survive_the_raw_string(tmp_path: Path) -> None:
    listing = 'x = f("# not markup"); s = "quoted"'
    source, _stats = TypstDocumentRenderer().build_source([_code_block(listing)], tmp_path)

    # A double quote inside the payload must not terminate the raw string.
    assert '\\"' in source
    # And the escaped payload is the only occurrence of the quote chars.
    assert source.count('s = \\"quoted\\"') == 1


def test_an_empty_listing_renders_nothing(tmp_path: Path) -> None:
    source, _stats = TypstDocumentRenderer().build_source([_code_block("")], tmp_path)
    assert "raw(" not in source
