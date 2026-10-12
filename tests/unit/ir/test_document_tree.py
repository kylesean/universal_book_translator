"""Section grouping: the translation unit for the semantic reflow path.

RFC-2026-02 §2.3 calls for section-scope translation (one unit per section)
instead of the per-block drafting that produced the pronoun-broken output the
old path shipped. These tests pin the grouping rules, because a wrong grouping
is invisible in a rendered artifact — it just produces subtly worse prose:

- a top-level heading opens a section; deeper headings stay inside it;
- a heading-less document still yields one translatable unit (an EPUB chapter,
  a plain article);
- the heading path reflects the source's own numbering, so a skipped level in
  the source cannot leave a stale ancestor in the path;
- blocks the policy marked ``skip_translate`` are excluded from the
  translatable view but stay in the section (reading order is preserved).
"""

from __future__ import annotations

import pytest

from ubt.core.ir.document_tree import build_section_tree, section_context_for, section_texts
from ubt.core.ir.models import BlockType, IRBlock, make_element

pytestmark = pytest.mark.fast

_BLOCK_TYPES = {
    "heading": BlockType.HEADING,
    "narrative": BlockType.NARRATIVE,
    "table": BlockType.TABLE,
}


def _Block(  # noqa: N802 - mirrors the IR constructor the tests read as a noun
    block_type: str,
    source_text: str,
    level: int = 1,
    skip_translate: bool = False,
    policy_translate: bool | None = None,
    id: str = "",
) -> IRBlock:
    """A real ``IRBlock``; a stand-in cannot catch a rename in the IR.

    Kept as a function rather than a dataclass so the call sites read unchanged.
    """
    element = make_element(
        id=id or f"b{abs(hash(source_text)) % 10_000}",
        spine_index=0,
        block_type=_BLOCK_TYPES.get(block_type, BlockType.NARRATIVE),
        source_text=source_text,
        level=level,
        skip_translate=skip_translate,
    )
    return IRBlock(element=element, policy_translate=policy_translate)


def test_deeper_headings_stay_inside_the_open_section() -> None:
    blocks = [
        _Block("heading", "1 Introduction", level=1),
        _Block("narrative", "intro", level=1),
        _Block("heading", "2 Switch Transformer", level=1),
        _Block("heading", "2.1 Simplifying Sparse Routing", level=2),
        _Block("narrative", "routing", level=1),
    ]
    sections = build_section_tree(blocks, max_level=2)
    assert [s.title for s in sections] == [
        "1 Introduction",
        "2 Switch Transformer",
        "2.1 Simplifying Sparse Routing",
    ]
    # The parent section holds only its own heading; the child opened a new one.
    assert len(sections[1]) == 1
    assert len(sections[2]) == 2


def test_a_headingless_document_is_one_translatable_unit() -> None:
    blocks = [_Block("narrative", "para one"), _Block("narrative", "para two")]
    sections = build_section_tree(blocks)
    assert len(sections) == 1
    assert sections[0].heading is None
    assert sections[0].title == ""
    assert section_texts(sections) == ["para one\n\npara two"]


def test_the_path_uses_the_sources_own_numbering() -> None:
    blocks = [
        _Block("heading", "2 Switch Transformer", level=1),
        _Block("heading", "2.1 Simplifying Sparse Routing", level=2),
        _Block("heading", "3 Scaling Properties", level=1),
    ]
    sections = build_section_tree(blocks, max_level=2)
    assert sections[0].path == ("2",)
    assert sections[1].path == ("2", "2.1")
    # A new level-1 heading replaces the path rather than appending to it.
    assert sections[2].path == ("3",)


def test_a_skipped_level_does_not_leave_a_stale_ancestor() -> None:
    blocks = [
        _Block("heading", "2 Parent", level=1),
        _Block("heading", "2.1 Child", level=2),
        # Jump straight back to level 1: the child must not survive in the path.
        _Block("heading", "3 Sibling", level=1),
    ]
    sections = build_section_tree(blocks, max_level=2)
    assert sections[2].path == ("3",)


def test_skip_translate_blocks_are_excluded_from_the_translatable_view() -> None:
    blocks = [
        _Block("heading", "1 Introduction"),
        _Block("narrative", "translate me"),
        _Block("narrative", "keep me", skip_translate=True),
        _Block("narrative", "policy says keep", policy_translate=False),
    ]
    sections = build_section_tree(blocks)
    translatable = sections[0].translatable
    assert [b.source_text for b in translatable] == ["1 Introduction", "translate me"]
    # ...but they stay in the section, so reading order is untouched.
    assert len(sections[0]) == 4


def test_an_empty_block_list_yields_no_sections() -> None:
    assert build_section_tree([]) == []


# --- section context (RFC-2026-02 §2.3) ------------------------------------


def test_section_context_gives_a_headingless_chunk_its_section() -> None:
    """A chunk that does not hold the heading still learns *where* it is.

    ``chunk_by_section`` guarantees a chunk never straddles a heading, but a
    long section spans several chunks and only the first holds the heading.
    Without the trail, the model meets "this approach" with no antecedent.

    The answer carries the **whole ancestry**, not just the leaf title: "2.1
    Simplifying Sparse Routing" alone does not say which chapter "2.1" belongs
    to, and that is the fact a back-reference needs.
    """
    blocks = [
        _Block("heading", "2 Switch Transformer", level=1, id="b1"),
        _Block("narrative", "intro", id="b2"),
        _Block("heading", "2.1 Simplifying Sparse Routing", level=2, id="b3"),
        _Block("narrative", "routing one", id="b4"),
        _Block("narrative", "routing two", id="b5"),
    ]
    # b5 is a later chunk of the 2.1 section: it carries no heading of its own.
    assert (
        section_context_for(blocks, [blocks[4]])
        == "2 Switch Transformer > 2.1 Simplifying Sparse Routing"
    )


def test_section_context_keeps_the_whole_ancestry_for_a_deep_section() -> None:
    """Three levels deep, the trail names every ancestor in order."""
    blocks = [
        _Block("heading", "1 Introduction", level=1, id="h1"),
        _Block("heading", "1.1 Background", level=2, id="h2"),
        _Block("heading", "1.1.1 Details", level=3, id="h3"),
        _Block("narrative", "prose", id="p1"),
    ]
    # ``1.1.1`` is deeper than ``max_level=2``, so it folds into ``1.1``.
    assert section_context_for(blocks, [blocks[3]]) == "1 Introduction > 1.1 Background"


def test_the_leaf_only_shortcut_is_not_used_when_an_ancestor_exists() -> None:
    """A one-level section needs no separator; a nested one must have it.

    Pins the regression directly: the old implementation returned
    ``section.title``, which for the nested case is the leaf alone.
    """
    blocks = [
        _Block("heading", "2 Method", level=1, id="h1"),
        _Block("heading", "2.1 Sub", level=2, id="h2"),
        _Block("narrative", "prose", id="p1"),
    ]
    nested = section_context_for(blocks, [blocks[2]])
    assert nested == "2 Method > 2.1 Sub"
    assert nested.count(">") == 1


def test_section_context_is_stable_across_a_sections_chunks() -> None:
    """Same string for every chunk of a section, so the cache key cannot flip.

    The content-addressed prompt cache keys on the prompt text; if the context
    changed halfway through a section, the second half would always miss.
    """
    blocks = [
        _Block("heading", "2.1 Simplifying Sparse Routing", level=2, id="b1"),
        _Block("narrative", "a", id="b2"),
        _Block("narrative", "b", id="b3"),
    ]
    assert section_context_for(blocks, [blocks[0]]) == section_context_for(blocks, [blocks[2]])


def test_section_context_is_empty_without_headings() -> None:
    blocks = [
        _Block("narrative", "a", id="b1"),
        _Block("narrative", "b", id="b2"),
    ]
    assert section_context_for(blocks, [blocks[0]]) == ""


def test_section_context_is_empty_for_an_empty_chunk() -> None:
    assert section_context_for([], []) == ""
