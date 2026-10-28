"""Tests for PromptContextAssembler, ContextBudget, and cross-stage prompt assembly."""

from __future__ import annotations

import pytest

from ubt.core.prompts.assembler import (
    PromptContextAssembler,
    assemble_continuity_context,
    assemble_critique_issues,
    assemble_terminology_table,
)
from ubt.core.prompts.budget import (
    ContextBudget,
    estimate_prompt_tokens,
    truncate_markdown_table_to_budget,
    truncate_text_to_budget,
)
from ubt.core.router.prompts import (
    build_hybrid_repair_prompt,
    build_minimal_repair_prompt,
    build_rich_repair_prompt,
    draft_source_from_prompt,
)

pytestmark = pytest.mark.fast


def test_context_budget_initialization_and_scaling() -> None:
    budget = ContextBudget()
    assert budget.max_context_tokens == 16384
    assert budget.max_terminology_tokens == 3072
    assert budget.max_history_tokens == 2048

    # Scaled down for small model (e.g. 4096 context window)
    scaled_small = ContextBudget.from_context_window(4096)
    assert scaled_small.max_context_tokens == 4096
    assert scaled_small.max_terminology_tokens < budget.max_terminology_tokens
    assert scaled_small.max_history_tokens < budget.max_history_tokens

    # Scaled up for large model (e.g. 64k window)
    scaled_large = ContextBudget.from_context_window(65536)
    assert scaled_large.max_context_tokens == 65536
    assert scaled_large.max_terminology_tokens > budget.max_terminology_tokens


def test_estimate_prompt_tokens_accuracy() -> None:
    ascii_text = "Hello world this is a test paragraph for token counting."
    assert estimate_prompt_tokens(ascii_text) > 5

    cjk_text = "这是一段用于测试中文词汇估算的文本段落。"
    assert estimate_prompt_tokens(cjk_text) > 10

    # Multi-argument sum
    combined = estimate_prompt_tokens(ascii_text, cjk_text)
    assert combined == estimate_prompt_tokens(ascii_text) + estimate_prompt_tokens(cjk_text)


def test_truncate_markdown_table_preserves_structure() -> None:
    header = "| Source Term | Target Term | Notes |\n| :--- | :--- | :--- |"
    rows = [f"| Term{i} | 术语{i} | Domain note for term {i} |" for i in range(100)]
    full_table = header + "\n" + "\n".join(rows)

    # Tight budget: e.g. 100 tokens
    truncated = truncate_markdown_table_to_budget(full_table, max_tokens=100)

    # Header and delimiter are strictly preserved
    assert "| Source Term | Target Term | Notes |" in truncated
    assert "| :--- | :--- | :--- |" in truncated
    # Rows are truncated cleanly without broken lines
    assert truncated.count("\n| Term") < 100
    assert "omitted to fit context budget" in truncated

    # Short table is returned untouched
    short_table = header + "\n| Alpha | 阿尔法 | Note |"
    assert truncate_markdown_table_to_budget(short_table, max_tokens=500) == short_table


def test_truncate_text_to_budget_preserves_boundaries() -> None:
    paragraphs = [
        f"Paragraph {i}: This is sentence content that extends the text." for i in range(50)
    ]
    full_text = "\n\n".join(paragraphs)

    truncated = truncate_text_to_budget(full_text, max_tokens=80)
    assert "[... truncated to fit budget ...]" in truncated
    assert len(truncated) < len(full_text)
    # Check it still has leading paragraphs
    assert "Paragraph 0:" in truncated


def test_assemble_terminology_table_budget_partition() -> None:
    global_table = "| GSource | GTarget |\n|---|---|\n" + "\n".join(
        f"| G{i} | 目标{i} |" for i in range(80)
    )
    chunk_table = "| CSource | CTarget |\n|---|---|\n" + "\n".join(
        f"| C{i} | 局部{i} |" for i in range(80)
    )

    budget = ContextBudget(max_terminology_tokens=200)
    b_global, b_chunk = assemble_terminology_table(global_table, chunk_table, budget=budget)

    assert "| GSource |" in b_global
    assert "| CSource |" in b_chunk
    assert estimate_prompt_tokens(b_global) <= 220
    assert estimate_prompt_tokens(b_chunk) <= 220


def test_assemble_continuity_context() -> None:
    epoch_text = "Epoch Summary: The protagonist traveled across the ancient kingdoms."
    rolling_text = "Rolling Summary: Chapter 3 introduced the dragon mountains."

    budget = ContextBudget(max_history_tokens=100)
    b_epoch, b_rolling = assemble_continuity_context(epoch_text, rolling_text, budget=budget)

    assert b_epoch == epoch_text
    assert b_rolling == rolling_text


def test_assemble_critique_issues_deduplication_and_limit() -> None:
    flags = [
        "Glossary violation: 'Alpha' -> '阿尔法'",
        "Repetition detected in sentence 2",
        "Glossary violation: 'Alpha' -> '阿尔法'",  # duplicate
        "Omission of equation clause",
    ]
    formatted = assemble_critique_issues(flags)
    assert formatted.count("Glossary violation") == 1
    assert "- Repetition detected in sentence 2" in formatted
    assert "- Omission of equation clause" in formatted

    empty_formatted = assemble_critique_issues([])
    assert "General fluency / quality score below threshold" in empty_formatted


def test_prompt_context_assembler_draft_assembly() -> None:
    assembler = PromptContextAssembler()
    source = "The fundamental theorem of calculus establishes the relationship between differentiation and integration."

    # Test Hybrid Strategy
    system_hybrid, user_hybrid = assembler.assemble_draft(
        source_text=source,
        strategy="hybrid",
        target_lang="zh",
        source_lang="en",
        genre_profile="academic",
        global_glossary="| Calculus | 微积分 |\n|---|---|",
        glossary_table="| Theorem | 定理 |\n|---|---|",
        rolling_summary="Previous section discussed limits.",
        epoch_summary="Book history: mathematical foundations.",
    )

    assert "scholarly scientific translator" in system_hybrid
    assert "Terminology Glossary" in user_hybrid
    assert "Book Continuity" in user_hybrid
    assert "Story Context" in user_hybrid
    assert draft_source_from_prompt(user_hybrid) == source

    # Test Rich Strategy
    system_rich, user_rich = assembler.assemble_draft(
        source_text=source,
        strategy="rich",
        target_lang="zh",
        source_lang="en",
        genre_profile="academic",
        global_glossary="| Calculus | 微积分 |\n|---|---|",
        glossary_table="| Theorem | 定理 |\n|---|---|",
    )
    assert "Translation Bible (Strict Global Terminology Mappings)" in system_rich
    assert "Translation Bible (Strict Mappings)" in user_rich
    assert draft_source_from_prompt(user_rich) == source


def test_prompt_context_assembler_macro_chunk_assembly() -> None:
    assembler = PromptContextAssembler()
    blocks = [("b001", "First paragraph."), ("b002", "Second paragraph.")]

    system, user = assembler.assemble_macro_chunk(
        blocks=blocks,
        target_lang="zh",
        source_lang="en",
        section_context="Section 2.1 Overview",
        global_glossary="| Paragraph | 段落 |\n|---|---|",
    )

    assert "Translation Bible (Strict Global Terminology Mappings)" in system
    assert '<block id="b001">First paragraph.</block>' in user
    assert '<block id="b002">Second paragraph.</block>' in user
    assert "Section Context" in user


def test_prompt_context_assembler_repair_with_global_glossary() -> None:
    assembler = PromptContextAssembler()
    source = "The Hamiltonian operator represents total system energy."
    draft = "哈密顿算子代表系统的能量。"
    error_flags = ["Terminology drift: Hamiltonian should be 哈密顿量"]

    # Assemble Rich Repair
    system_repair, user_repair = assembler.assemble_repair(
        source_text=source,
        draft_text=draft,
        error_flags=error_flags,
        strategy="rich",
        target_lang="zh",
        source_lang="en",
        global_glossary="| Hamiltonian | 哈密顿量 |\n|---|---|",
        glossary_table="| Operator | 算子 |\n|---|---|",
    )

    assert "Translation Bible (Global Terminology)" in user_repair
    assert "Translation Bible\n| Operator | 算子 |" in user_repair
    assert "Terminology drift: Hamiltonian should be 哈密顿量" in user_repair
    # Source recovery must not leak scaffold
    assert draft_source_from_prompt(user_repair) == source


def test_prompt_context_assembler_consistency_assembly() -> None:
    assembler = PromptContextAssembler()
    source = "The quantum wavefunction collapses upon observation."
    draft = "量子波函数在观测时坍塌。"
    consistency_flags = [
        "[GLOSSARY_VIOLATION]: 'wavefunction' translated as '波函数' but book standard is '态函数'"
    ]

    system, user = assembler.assemble_consistency(
        source_text=source,
        draft_text=draft,
        consistency_flags=consistency_flags,
        strategy="hybrid",
        target_lang="zh",
        source_lang="en",
        global_glossary="| Wavefunction | 态函数 |\n|---|---|",
    )

    assert "Terminology Glossary" in user
    assert "态函数" in user
    assert "[GLOSSARY_VIOLATION]" in user
    assert draft_source_from_prompt(user) == source


def test_repair_builders_directly_accept_global_glossary() -> None:
    source = "Coordinate descent iteratively minimizes along coordinate directions."
    draft = "坐标下降法沿着坐标方向迭代最小化。"
    global_glossary = "| Coordinate descent | 坐标下降算法 |\n|---|---|"
    chunk_glossary = "| Iteratively | 迭代地 |\n|---|---|"

    # Minimal Repair
    _sys_min, user_min = build_minimal_repair_prompt(
        source_text=source,
        draft_text=draft,
        error_flags=["Minor terminology inconsistency"],
        glossary_table=chunk_glossary,
        global_glossary=global_glossary,
    )
    assert "坐标下降算法" in user_min
    assert "迭代地" in user_min
    assert draft_source_from_prompt(user_min) == source

    # Hybrid Repair
    _sys_hyb, user_hyb = build_hybrid_repair_prompt(
        source_text=source,
        draft_text=draft,
        error_flags=["Minor terminology inconsistency"],
        glossary_table=chunk_glossary,
        global_glossary=global_glossary,
    )
    assert "Terminology Glossary" in user_hyb
    assert "Terminology Glossary (this section)" in user_hyb
    assert draft_source_from_prompt(user_hyb) == source

    # Rich Repair
    _sys_rich, user_rich = build_rich_repair_prompt(
        source_text=source,
        draft_text=draft,
        error_flags=["Minor terminology inconsistency"],
        glossary_table=chunk_glossary,
        global_glossary=global_glossary,
    )
    assert "Translation Bible (Global Terminology)" in user_rich
    assert "Translation Bible\n| Iteratively | 迭代地 |" in user_rich
    assert draft_source_from_prompt(user_rich) == source
