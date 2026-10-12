"""Section-aware draft chunking (RFC-2026-02 §2.3).

A fixed ``prepared[i:i+size]`` split cuts sections in half. Measured on the
arXiv 2101.03961 run with the shipped default (``macro_chunk_size=5``), one
chunk ended in the middle of the Abstract, so the model received a heading whose
prose lived in the *next* request. A heading and the paragraph it introduces are
one translation unit; splitting them is how a draft loses the antecedent for a
pronoun or mistranslates a title in isolation.

These tests pin the packing rules, because a wrong packing is invisible in a
rendered artifact — the page still compiles, the prose is just subtly worse.
"""

from __future__ import annotations

import pytest

from ubt.core.engine.stages.draft import _DraftInputs, chunk_by_section
from ubt.core.ir.models import BlockType, IRBlock, make_element

pytestmark = pytest.mark.fast

_MAPPING = {"H": BlockType.HEADING, "p": BlockType.NARRATIVE, "T": BlockType.TABLE}


def _prepared(*kinds: str) -> list[tuple[IRBlock, _DraftInputs]]:
    """Real ``IRBlock``s: a duck-typed stand-in cannot catch a rename."""
    out: list[tuple[IRBlock, _DraftInputs]] = []
    for i, kind in enumerate(kinds):
        element = make_element(
            id=f"b{i}",
            spine_index=i,
            block_type=_MAPPING[kind],
            source_text=f"text {i}",
            level=1,
            skip_translate=False,
        )
        block = IRBlock(element=element)
        inputs = _DraftInputs(
            masked_source=block.source_text,
            code_map={},
            cite_map={},
            math_map={},
            glossary_table="",
            neighbor_ctx="",
            macro_ctx="",
            few_shot_reference="",
        )
        out.append((block, inputs))
    return out


def _shape(chunks: list[list[tuple[IRBlock, _DraftInputs]]]) -> list[list[str]]:
    return [
        ["H" if b.block_type is BlockType.HEADING else "." for b, _ in chunk] for chunk in chunks
    ]


def test_a_heading_starts_a_new_chunk() -> None:
    chunks = chunk_by_section(_prepared("p", "p", "H", "p"), chunk_size=10)
    assert _shape(chunks) == [[".", "."], ["H", "."]]


def test_a_run_of_headings_stays_together() -> None:
    """A title block (arXiv id, journal, submitted, title) is one unit.

    Splitting it into four single-heading calls pays four times for one page.
    """
    chunks = chunk_by_section(_prepared("H", "H", "H", "H"), chunk_size=5)
    assert _shape(chunks) == [["H", "H", "H", "H"]]


def test_a_heading_after_prose_does_split() -> None:
    chunks = chunk_by_section(_prepared("H", "p", "H", "H", "p"), chunk_size=10)
    assert _shape(chunks) == [["H", "."], ["H", "H", "."]]


def test_the_size_limit_still_bounds_a_long_section() -> None:
    """A section longer than the chunk size is split at the limit."""
    chunks = chunk_by_section(_prepared("H", "p", "p", "p", "p", "p"), chunk_size=3)
    assert _shape(chunks) == [["H", ".", "."], [".", ".", "."]]


def test_a_short_section_is_not_merged_with_its_neighbour() -> None:
    """Merging would reintroduce the cross-boundary packing this prevents."""
    chunks = chunk_by_section(_prepared("H", "p", "H", "p"), chunk_size=10)
    assert len(chunks) == 2
    assert _shape(chunks) == [["H", "."], ["H", "."]]


def test_chunk_size_one_yields_one_block_per_chunk() -> None:
    chunks = chunk_by_section(_prepared("H", "p", "p"), chunk_size=1)
    assert _shape(chunks) == [["H"], ["."], ["."]]


def test_order_and_identity_are_preserved() -> None:
    """The caller aligns results with ``prepared`` by index, so order is a contract."""
    prepared = _prepared("H", "p", "H", "p", "p", "H")
    chunks = chunk_by_section(prepared, chunk_size=2)
    flattened = [block for chunk in chunks for block, _ in chunk]
    assert flattened == [block for block, _ in prepared]


def test_an_empty_input_yields_no_chunks() -> None:
    assert chunk_by_section([], chunk_size=5) == []


def test_a_headingless_document_packs_to_the_size_limit() -> None:
    chunks = chunk_by_section(_prepared(*("p" for _ in range(7))), chunk_size=3)
    assert _shape(chunks) == [[".", ".", "."], [".", ".", "."], ["."]]
