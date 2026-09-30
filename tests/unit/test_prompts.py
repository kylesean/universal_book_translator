"""Unit tests for prompt construction and table repair structural guardrails."""

import pytest

from ubt.core.router.prompts import (
    build_hybrid_repair_prompt,
    build_minimal_repair_prompt,
    build_rich_repair_prompt,
)


@pytest.mark.fast
def test_table_repair_prompt_includes_grid_preservation_guardrails() -> None:
    """When repairing a markdown table, repair prompts must include structural
    guardrails to strictly preserve exact row and column counts (N x M) and never delete rows.
    """
    src_table = (
        "| Model | Accuracy | F1 Score |\n"
        "|---|---|---|\n"
        "| GPT-4 | 0.92 | 0.91 |\n"
        "| Claude | 0.93 | 0.92 |\n"
        "| Llama | 0.88 | 0.87 |"
    )
    draft_table = (
        "| 模型 | 准确率 | F1得分 |\n"
        "|---|---|---|\n"
        "| GPT-4 | 0.92 | 0.91 |\n"
        "| Claude | 0.93 | 0.92 |\n"
        "| Llama | 0.88 | 0.87 |"
    )
    flags = ["Repetitive loop hallucination detected"]

    # Hybrid repair prompt
    sys_h, user_h = build_hybrid_repair_prompt(src_table, draft_table, flags)
    prompt_text_h = (sys_h + "\n" + user_h).lower()
    assert "preserve" in prompt_text_h or "row" in prompt_text_h
    assert "column" in prompt_text_h or "grid" in prompt_text_h
    assert "delete" in prompt_text_h or "exact" in prompt_text_h

    # Rich repair prompt
    sys_r, user_r = build_rich_repair_prompt(src_table, draft_table, flags)
    prompt_text_r = (sys_r + "\n" + user_r).lower()
    assert "preserve" in prompt_text_r or "row" in prompt_text_r
    assert "column" in prompt_text_r or "grid" in prompt_text_r
    assert "delete" in prompt_text_r or "exact" in prompt_text_r

    # Minimal repair prompt
    sys_m, user_m = build_minimal_repair_prompt(src_table)
    prompt_text_m = (sys_m + "\n" + user_m).lower()
    assert "row" in prompt_text_m or "grid" in prompt_text_m or "table" in prompt_text_m


@pytest.mark.fast
def test_table_repair_prompt_with_intro_preamble_preserves_guardrail() -> None:
    """A table accompanied by descriptive preamble lines must still be detected as a table
    and receive table preservation guardrails in repair prompts.
    """
    src_table = (
        "Table 1: Main experimental results on benchmark datasets.\n"
        "Below we compare our proposed method against standard baselines across three tasks.\n"
        "All models are evaluated under the same conditions.\n"
        "Line 4 of descriptive context.\n"
        "Line 5 of experimental details.\n"
        "| Model | Accuracy | F1 Score |\n"
        "|---|---|---|\n"
        "| GPT-4 | 0.92 | 0.91 |\n"
        "| Claude | 0.93 | 0.92 |"
    )
    draft_table = "..."
    flags = ["Repetitive loop hallucination detected"]

    sys_h, user_h = build_hybrid_repair_prompt(src_table, draft_table, flags)
    prompt_text_h = (sys_h + "\n" + user_h).lower()
    assert (
        "critical table structure guardrail" in prompt_text_h
        or "table preservation requirement" in prompt_text_h
    )


def test_minimal_repair_prompt_carries_the_draft_and_flags() -> None:
    """MINIMAL is small, not blind: the draft and its defects must be included.

    The minimal repair prompt only carried the source, so a "repair" under a
    minimal-model profile re-translated from scratch with no error context —
    the MQM repair chain silently degraded to a fresh draft while still
    billing repair rounds.
    """
    from ubt.core.router.prompts import build_minimal_repair_prompt

    _system, user = build_minimal_repair_prompt(
        source_text="The valve opens at dawn.",
        draft_text="The valve opens at dawn.",
        error_flags=["target_missing", "echo_detected"],
        target_lang="zh",
    )
    assert "The valve opens at dawn." in user
    assert "target_missing" in user
    assert "echo_detected" in user
