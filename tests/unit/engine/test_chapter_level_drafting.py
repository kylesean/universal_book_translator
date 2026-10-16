"""Chapter-level long-context drafting is the default (module B, item 2).

The spec's translation unit is a logical section/chapter packed into one long
request (``<block id=...>`` markup, one generation), not a per-block micro
window. The macro path existed but was gated behind ``macro_chunk_size > 1`` with
a default of 1, so the live path stayed per-block. These tests pin that the
default now packs, and that the packing never crosses a section boundary.
"""

from __future__ import annotations

import pytest

from ubt.core.config import UBTConfig
from ubt.core.engine.stages.draft import _DraftInputs, chunk_by_section
from ubt.core.ir.models import BlockType, IRBlock, make_element

pytestmark = pytest.mark.fast


def _block(bid: str, block_type: BlockType) -> IRBlock:
    return IRBlock(
        element=make_element(
            id=bid, spine_index=1, block_type=block_type, source_text="Some source text here."
        )
    )


def _prepared(*types: BlockType) -> list[tuple[IRBlock, _DraftInputs]]:
    return [
        (
            _block(f"b{i}", t),
            _DraftInputs(
                masked_source="x",
                code_map=None,
                cite_map=None,
                math_map=None,
                glossary_table="",
                neighbor_ctx="",
                macro_ctx="",
                few_shot_reference="",
            ),
        )
        for i, t in enumerate(types)
    ]


def test_macro_chunk_size_defaults_to_the_chapter_level_path() -> None:
    # >1 selects the packed/chapter path; the old default of 1 forced one call
    # per block (the 300-char micro window the spec retired).
    assert UBTConfig().macro_chunk_size > 1


def test_narrative_blocks_pack_into_one_chunk() -> None:
    prepared = _prepared(*(BlockType.NARRATIVE for _ in range(6)))
    chunks = chunk_by_section(prepared, chunk_size=8)
    assert len(chunks) == 1
    assert len(chunks[0]) == 6


def test_a_heading_starts_a_new_chunk() -> None:
    # A heading and the prose it introduces are one unit; a following heading
    # (after prose) opens a fresh chunk so sections never mix.
    prepared = _prepared(
        BlockType.HEADING,
        BlockType.NARRATIVE,
        BlockType.NARRATIVE,
        BlockType.HEADING,
        BlockType.NARRATIVE,
    )
    chunks = chunk_by_section(prepared, chunk_size=8)
    assert [len(c) for c in chunks] == [3, 2]


def test_chunk_size_bounds_a_long_section() -> None:
    prepared = _prepared(*(BlockType.NARRATIVE for _ in range(10)))
    chunks = chunk_by_section(prepared, chunk_size=4)
    assert all(len(c) <= 4 for c in chunks)
    assert sum(len(c) for c in chunks) == 10
