"""Unit tests for memory subsystem: CJK matching, neighbor sliding window, and Translation Bible."""

from ubt.core.ir.models import FlowID, IRBlock
from ubt.core.memory.bible import (
    BibleEntry,
    clean_bible_entry,
    merge_bible_entries,
)
from ubt.core.memory.cjk_matcher import (
    contains_cjk,
    count_term_in_text,
    format_terms_markdown_table,
    select_terms_for_chunk,
    term_appears_in_text,
)
from ubt.core.memory.neighbor_window import NeighborContextBuilder


def test_cjk_matcher_boundary_and_length_rules() -> None:
    """Validate ASCII boundary-awareness and CJK length sensitivity."""
    text = "The cat sat in the category catalog. 我们在图书馆阅读一本书。"

    # ASCII boundary test: "cat" must not match inside "category" or "catalog"
    assert count_term_in_text("cat", text) == 1

    # CJK single character test: must return 0 to prevent false-positive overmatching
    assert count_term_in_text("书", text) == 0
    assert count_term_in_text("图", text) == 0

    # CJK multi-character test: matches properly
    assert count_term_in_text("图书馆", text) == 1
    assert contains_cjk("书") is True
    assert contains_cjk("hello") is False


def test_select_terms_for_chunk_and_table_format() -> None:
    """Validate local hit priority and markdown table rendering."""
    terms = [
        {"source": "cat", "translation": "猫", "frequency": 10},
        {"source": "dog", "translation": "狗", "frequency": 100},
        {"source": "bird", "translation": "鸟", "frequency": 50},
    ]
    chunk = "The cat ran away."

    selected = select_terms_for_chunk(terms, chunk, top_n=1, max_terms=2)
    sources = [s["source"] for s in selected]
    # 'cat' is a local hit (must be included)
    assert "cat" in sources
    # 'dog' is top global frequency
    assert "dog" in sources
    assert len(selected) == 2

    table = format_terms_markdown_table(selected)
    assert "| 原文 | 别名 | 译文 |" in table
    assert "| cat |  | 猫 |" in table


def test_neighbor_window_flow_isolation() -> None:
    """Validate that sliding context window never bleeds across different FlowIDs."""
    main_b1 = IRBlock(
        id="m1", flow_id=FlowID.MAIN_STORY, spine_index=1, source_text="Main paragraph 1."
    )
    side_b = IRBlock(
        id="s1", flow_id=FlowID.SIDEBAR_ASIDE, spine_index=2, source_text="Sidebar callout box."
    )
    main_b2 = IRBlock(
        id="m2", flow_id=FlowID.MAIN_STORY, spine_index=3, source_text="Main paragraph 2."
    )

    builder = NeighborContextBuilder(neighbor_chars=300)

    # For main_b2, its predecessor should be main_b1, completely skipping side_b!
    ctx = builder.extract_from_blocks(main_b2, [main_b1, side_b, main_b2])
    assert "Main paragraph 1." in ctx
    assert "Sidebar callout box." not in ctx


def test_translation_bible_pruning_and_merging() -> None:
    """Validate 0-token pruning of noisy slogans and multi-chapter deduplication."""
    # Slogan with > 4 words must be dropped
    slogan = clean_bible_entry(
        source="Four legs good two legs bad",
        translation="四条腿好两条腿坏",
    )
    assert slogan is None

    # Normal entry: punctuation stripped
    entry1 = clean_bible_entry(
        source="Snowball",
        translation="《雪球》",
        aliases=["Snow-ball"],
    )
    assert entry1 is not None
    assert entry1.translation == "雪球"

    # Chapter 2 has same character with alternative alias and slightly different spelling
    entry2 = clean_bible_entry(
        source="snowball",
        translation="白球",  # Second translation must be overridden by first
        aliases=["The Pig Snowball"],
    )
    assert entry2 is not None

    merged = merge_bible_entries([entry1, entry2])
    assert len(merged) == 1
    assert merged[0].source == "Snowball"
    assert merged[0].translation == "雪球"  # First attested translation preserved
    assert "The Pig Snowball" in merged[0].aliases
    assert "Snow-ball" in merged[0].aliases


def test_merge_bible_entries_populates_missing_translation() -> None:
    """If first entry has empty translation, subsequent valid translation should populate it."""
    entry1 = BibleEntry(source="Darcy", translation="", aliases=["Mr. Darcy"], kind="person")
    entry2 = BibleEntry(source="Darcy", translation="达西", aliases=["Fitzwilliam"], kind="person")
    merged = merge_bible_entries([entry1, entry2])
    assert len(merged) == 1
    assert merged[0].translation == "达西"
    assert "Mr. Darcy" in merged[0].aliases
    assert "Fitzwilliam" in merged[0].aliases


def test_neighbor_context_builder_fallback_prev_text() -> None:
    """Fix 5: Verify NeighborContextBuilder uses fallback_prev_text when preceding block is outside batch."""
    builder = NeighborContextBuilder()

    block1 = IRBlock(id="b101", spine_index=101, source_text="Current batch first block.")
    block2 = IRBlock(id="b102", spine_index=102, source_text="Current batch second block.")
    batch = [block1, block2]

    # Without fallback: block1 has no preceding context
    ctx_no_fallback = builder.extract_from_blocks(block1, batch)
    assert "READ-ONLY PRECEDING CONTEXT" not in ctx_no_fallback

    # With fallback from ledger: preceding context is seamlessly injected
    ctx_with_fallback = builder.extract_from_blocks(
        block1, batch, fallback_prev_text="Tail excerpt from block 100 in previous batch."
    )
    assert "READ-ONLY PRECEDING CONTEXT" in ctx_with_fallback
    assert "Tail excerpt from block 100" in ctx_with_fallback


def test_cjk_matcher_case_insensitivity() -> None:
    """Verify that CJK matcher matches terms case-insensitively for Latin scripts."""
    # 1. Term uppercase, text lowercase
    term = {"source": "Attention", "translation": "注意力", "aliases": []}
    text = "The self-attention mechanism revolutionized natural language processing."

    assert term_appears_in_text(term, text) is True
    assert count_term_in_text("Attention", text) == 1

    # 2. Term lowercase, text title-case
    term_lower = {"source": "transformer", "translation": "转换器", "aliases": []}
    text_upper = "Transformer models are scalable."
    assert term_appears_in_text(term_lower, text_upper) is True
    assert count_term_in_text("transformer", text_upper) == 1

    # 3. Term selection for chunk
    selected = select_terms_for_chunk([term], text)
    assert len(selected) == 1
    assert selected[0]["source"] == "Attention"


def test_neighbor_context_builder_bidirectional() -> None:
    """Verify NeighborContextBuilder formats both preceding and subsequent excerpts."""
    builder = NeighborContextBuilder(neighbor_chars=100)
    ctx = builder.format_prompt_block(
        prev_text="Preceding context paragraph.",
        next_text="Subsequent context paragraph.",
    )
    assert "[READ-ONLY PRECEDING CONTEXT: DO NOT TRANSLATE OR ECHO]" in ctx
    assert "Preceding context paragraph." in ctx
    assert "[READ-ONLY SUBSEQUENT CONTEXT: DO NOT TRANSLATE OR ECHO]" in ctx
    assert "Subsequent context paragraph." in ctx


def test_neighbor_window_fallback_next_text() -> None:
    """Verify NeighborContextBuilder correctly incorporates fallback_next_text."""
    builder = NeighborContextBuilder()
    b1 = IRBlock(
        id="b1",
        spine_index=1,
        flow_id=FlowID.MAIN_STORY,
        source_text="First paragraph.",
    )
    # When b1 is the only block in the batch, fallback_next_text provides lookahead
    ctx = builder.extract_from_blocks(
        target_block=b1,
        surrounding_blocks=[b1],
        fallback_next_text="Next batch lookahead paragraph.",
    )
    assert "[READ-ONLY SUBSEQUENT CONTEXT: DO NOT TRANSLATE OR ECHO]" in ctx
    assert "Next batch lookahead paragraph." in ctx


def test_in_batch_neighbor_window_prefers_the_finished_target() -> None:
    from ubt.core.memory.neighbor_window import NeighborContextBuilder

    builder = NeighborContextBuilder(neighbor_chars=200)
    prev = IRBlock(
        id="n1",
        flow_id=FlowID.MAIN_STORY,
        spine_index=1,
        source_text="The elf closed the door quietly.",
        target_text="L'elfe ferma la porte en silence.",
    )
    current = IRBlock(
        id="n2",
        flow_id=FlowID.MAIN_STORY,
        spine_index=2,
        source_text="Nobody heard it.",
    )
    ctx = builder.extract_from_blocks(current, [prev, current])
    assert "L'elfe ferma la porte" in ctx
    assert "PRECEDING CONTEXT" in ctx
