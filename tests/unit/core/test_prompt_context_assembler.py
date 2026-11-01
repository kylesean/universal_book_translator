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


def test_estimate_prompt_tokens_accuracy() -> None:
    ascii_text = "Hello world this is a test paragraph for token counting."
    assert estimate_prompt_tokens(ascii_text) > 5

    cjk_text = "这是一段用于测试中文词汇估算的文本段落。"
    assert estimate_prompt_tokens(cjk_text) > 10

    combined = estimate_prompt_tokens(ascii_text, cjk_text)
    assert combined == estimate_prompt_tokens(ascii_text) + estimate_prompt_tokens(cjk_text)


def test_truncate_markdown_table_preserves_structure() -> None:
    header = "| Source Term | Target Term | Notes |\n| :--- | :--- | :--- |"
    rows = [f"| Term{i} | 术语{i} | Domain note for term {i} |" for i in range(100)]
    full_table = header + "\n" + "\n".join(rows)

    truncated = truncate_markdown_table_to_budget(full_table, max_tokens=100)

    assert "| Source Term | Target Term | Notes |" in truncated
    assert "| :--- | :--- | :--- |" in truncated
    assert truncated.count("\n| Term") < 100
    assert "omitted to fit context budget" in truncated

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
        "Glossary violation: 'Alpha' -> '阿尔法'",
        "Omission of equation clause",
    ]
    formatted = assemble_critique_issues(flags)
    assert formatted.count("Glossary violation") == 1
    assert "- Repetition detected in sentence 2" in formatted
    assert "- Omission of equation clause" in formatted

    empty_formatted = assemble_critique_issues([])
    assert "General fluency / quality score below threshold" in empty_formatted


def test_prompt_context_assembler_repair_with_global_glossary() -> None:
    assembler = PromptContextAssembler()
    source = "The Hamiltonian operator represents total system energy."
    draft = "哈密顿算子代表系统的能量。"
    error_flags = ["Terminology drift: Hamiltonian should be 哈密顿量"]

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
    assert draft_source_from_prompt(user_repair) == source


def test_repair_builders_directly_accept_global_glossary() -> None:
    source = "Coordinate descent iteratively minimizes along coordinate directions."
    draft = "坐标下降法沿着坐标方向迭代最小化。"
    global_glossary = "| Coordinate descent | 坐标下降算法 |\n|---|---|"
    chunk_glossary = "| Iteratively | 迭代地 |\n|---|---|"

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
