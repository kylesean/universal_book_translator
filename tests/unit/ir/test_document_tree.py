from __future__ import annotations

import pytest

from ubt.core.ir.document_tree import build_section_tree, section_context_for
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
    assert len(sections[1]) == 1
    assert len(sections[2]) == 2


def test_a_headingless_document_is_one_translatable_unit() -> None:
    blocks = [_Block("narrative", "para one"), _Block("narrative", "para two")]
    sections = build_section_tree(blocks)
    assert len(sections) == 1
    assert sections[0].heading is None
    assert sections[0].title == ""
    assert sections[0].text == "para one\n\npara two"


def test_the_path_uses_the_sources_own_numbering() -> None:
    blocks = [
        _Block("heading", "2 Switch Transformer", level=1),
        _Block("heading", "2.1 Simplifying Sparse Routing", level=2),
        _Block("heading", "3 Scaling Properties", level=1),
    ]
    sections = build_section_tree(blocks, max_level=2)
    assert sections[0].path == ("2",)
    assert sections[1].path == ("2", "2.1")
    assert sections[2].path == ("3",)


def test_a_skipped_level_does_not_leave_a_stale_ancestor() -> None:
    blocks = [
        _Block("heading", "2 Parent", level=1),
        _Block("heading", "2.1 Child", level=2),
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
    assert len(sections[0]) == 4


def test_an_empty_block_list_yields_no_sections() -> None:
    assert build_section_tree([]) == []


def test_section_context_gives_a_headingless_chunk_its_section() -> None:
    blocks = [
        _Block("heading", "2 Switch Transformer", level=1, id="b1"),
        _Block("narrative", "intro", id="b2"),
        _Block("heading", "2.1 Simplifying Sparse Routing", level=2, id="b3"),
        _Block("narrative", "routing one", id="b4"),
        _Block("narrative", "routing two", id="b5"),
    ]
    assert (
        section_context_for(blocks, [blocks[4]])
        == "2 Switch Transformer > 2.1 Simplifying Sparse Routing"
    )


def test_section_context_keeps_the_whole_ancestry_for_a_deep_section() -> None:
    blocks = [
        _Block("heading", "1 Introduction", level=1, id="h1"),
        _Block("heading", "1.1 Background", level=2, id="h2"),
        _Block("heading", "1.1.1 Details", level=3, id="h3"),
        _Block("narrative", "prose", id="p1"),
    ]
    assert section_context_for(blocks, [blocks[3]]) == "1 Introduction > 1.1 Background"


def test_the_leaf_only_shortcut_is_not_used_when_an_ancestor_exists() -> None:
    blocks = [
        _Block("heading", "2 Method", level=1, id="h1"),
        _Block("heading", "2.1 Sub", level=2, id="h2"),
        _Block("narrative", "prose", id="p1"),
    ]
    nested = section_context_for(blocks, [blocks[2]])
    assert nested == "2 Method > 2.1 Sub"
    assert nested.count(">") == 1


def test_section_context_is_stable_across_a_sections_chunks() -> None:
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


def test_a_skipped_level_does_not_nest_a_sibling_under_its_sibling() -> None:
    blocks = [
        _Block("heading", "1 Intro", level=1),
        _Block("heading", "1.0.1 Sub", level=3),
        _Block("heading", "1.0.2 Sub2", level=3),
    ]
    sections = build_section_tree(blocks, max_level=3)
    paths = {s.title: s.path for s in sections}

    assert paths["1.0.1 Sub"] == ("1", "1.0.1")
    assert paths["1.0.2 Sub2"] == ("1", "1.0.2")


def test_a_deeper_heading_still_nests() -> None:
    blocks = [
        _Block("heading", "1 A", level=1),
        _Block("heading", "1.1 B", level=2),
        _Block("heading", "1.1.1 C", level=3),
        _Block("heading", "1.1.2 D", level=3),
    ]
    sections = build_section_tree(blocks, max_level=3)
    paths = {s.title: s.path for s in sections}

    assert paths["1.1.1 C"] == ("1", "1.1", "1.1.1")
    assert paths["1.1.2 D"] == ("1", "1.1", "1.1.2")


def test_the_acknowledgments_section_carries_the_personal_name_rule() -> None:
    blocks = [
        _Block("heading", "9 Acknowledgments", id="h1"),
        _Block("narrative", "We thank Yinqian Zhang for the discussions.", id="p1"),
        _Block("narrative", "This work was supported by Ant Group.", id="p2"),
        _Block("heading", "Appendix A Benchmarks", id="h2"),
        _Block("narrative", "Numbers follow.", id="p3"),
    ]
    ack = section_context_for(blocks, [blocks[1], blocks[2]])
    appendix = section_context_for(blocks, [blocks[4]])

    assert "Personal Names" in ack
    assert "never transliterate" in ack
    assert ack.startswith("9 Acknowledgments")
    assert "Personal Names" not in appendix
