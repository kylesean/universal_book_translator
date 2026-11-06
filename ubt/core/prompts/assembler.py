"""Budgeted prompt context assembly shared by the prompt builders.

Every stage's builder needs the same three decisions before it can format a
prompt: how much terminology to keep, how much continuity summary to keep, how
many critique issues to list. Those helpers live here,
:mod:`ubt.core.router.prompts` calls them from its draft and repair builders, and
:class:`PromptContextAssembler` is the budget-aware entry point for the repair
strategies. Keeping the budgeting in one module is what stops each stage from
growing its own truncation rules that silently disagree.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ubt.core.prompts.budget import (
    ContextBudget,
    truncate_markdown_table_to_budget,
    truncate_text_to_budget,
)

if TYPE_CHECKING:
    from ubt.core.router.capabilities import PromptStrategy

logger = logging.getLogger(__name__)


def assemble_terminology_table(
    global_glossary: str = "",
    chunk_glossary: str = "",
    budget: ContextBudget | None = None,
) -> tuple[str, str]:
    """Fit the global and chunk glossaries into ``max_terminology_tokens``.

    A single glossary gets the whole quota; when both are present the quota is
    divided evenly, because nothing at this point in the pipeline makes one more
    valuable than the other, and an uneven rule would quietly starve one stage.
    """
    b = budget or ContextBudget()
    max_term_tokens = b.max_terminology_tokens

    clean_global = global_glossary.strip()
    clean_chunk = chunk_glossary.strip()

    if not clean_global and not clean_chunk:
        return "", ""

    if clean_global and not clean_chunk:
        return truncate_markdown_table_to_budget(clean_global, max_term_tokens), ""

    if clean_chunk and not clean_global:
        return "", truncate_markdown_table_to_budget(clean_chunk, max_term_tokens)

    # Both are present: allocate budget proportionally (default 50/50)
    half_budget = max(32, max_term_tokens // 2)
    budgeted_global = truncate_markdown_table_to_budget(clean_global, half_budget)
    budgeted_chunk = truncate_markdown_table_to_budget(clean_chunk, half_budget)
    return budgeted_global, budgeted_chunk


def assemble_continuity_context(
    epoch_summary: str = "",
    rolling_summary: str = "",
    budget: ContextBudget | None = None,
) -> tuple[str, str]:
    """Assemble and budget epoch and rolling continuity summaries.

    Returns ``(budgeted_epoch, budgeted_rolling)`` respecting ``max_history_tokens``.
    """
    b = budget or ContextBudget()
    max_history = b.max_history_tokens

    clean_epoch = epoch_summary.strip()
    clean_rolling = rolling_summary.strip()

    if not clean_epoch and not clean_rolling:
        return "", ""

    if clean_epoch and not clean_rolling:
        return truncate_text_to_budget(clean_epoch, max_history), ""

    if clean_rolling and not clean_epoch:
        return "", truncate_text_to_budget(clean_rolling, max_history)

    half_budget = max(32, max_history // 2)
    return (
        truncate_text_to_budget(clean_epoch, half_budget),
        truncate_text_to_budget(clean_rolling, half_budget),
    )


def assemble_critique_issues(
    error_flags: list[str] | None = None,
    max_issues: int = 20,
) -> str:
    """Format defect flags into a standardized critique list for repair and consistency."""
    if not error_flags:
        return "- General fluency / quality score below threshold"

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_flags: list[str] = []
    for flag in error_flags:
        clean = flag.strip()
        if clean and clean not in seen:
            seen.add(clean)
            unique_flags.append(clean)

    if not unique_flags:
        return "- General fluency / quality score below threshold"

    trimmed = unique_flags[:max_issues]
    formatted = "\n".join(f"- {flag}" for flag in trimmed)
    if len(unique_flags) > max_issues:
        formatted += f"\n- ... and {len(unique_flags) - max_issues} more issues"
    return formatted


class PromptContextAssembler:
    """Budget-aware entry point for the three repair prompt strategies.

    One ``strategy`` string selects minimal, hybrid or rich; terminology is
    budgeted once here so the strategies cannot differ from each other in how
    much glossary they smuggle into the context. Draft and consistency prompts
    are not routed through this class -- their builders call the same helpers
    directly, which is why only repair has a wrapper.
    """

    def __init__(self, budget: ContextBudget | None = None) -> None:
        self.budget = budget or ContextBudget()

    def assemble_repair(
        self,
        source_text: str,
        draft_text: str,
        error_flags: list[str] | None = None,
        *,
        strategy: PromptStrategy | str = "rich",
        glossary_table: str = "",
        global_glossary: str = "",
        target_lang: str = "zh",
        source_lang: str = "en",
        annotated_draft: str = "",
        has_error_spans: bool = False,
        context_block: str = "",
    ) -> tuple[str, str]:
        """Assemble a budgeted repair prompt with unified terminology and critique formatting."""
        from ubt.core.router.prompts import (
            build_hybrid_repair_prompt,
            build_minimal_repair_prompt,
            build_rich_repair_prompt,
        )

        b_global, b_chunk = assemble_terminology_table(
            global_glossary=global_glossary,
            chunk_glossary=glossary_table,
            budget=self.budget,
        )
        clean_flags = list(error_flags or [])
        strat_str = str(strategy).lower()

        if "minimal" in strat_str:
            return build_minimal_repair_prompt(
                source_text=source_text,
                draft_text=draft_text,
                error_flags=clean_flags,
                glossary_table=b_chunk,
                target_lang=target_lang,
                source_lang=source_lang,
                global_glossary=b_global,
            )
        if "hybrid" in strat_str:
            return build_hybrid_repair_prompt(
                source_text=source_text,
                draft_text=draft_text,
                error_flags=clean_flags,
                glossary_table=b_chunk,
                target_lang=target_lang,
                source_lang=source_lang,
                annotated_draft=annotated_draft,
                has_error_spans=has_error_spans,
                global_glossary=b_global,
            )
        return build_rich_repair_prompt(
            source_text=source_text,
            draft_text=draft_text,
            error_flags=clean_flags,
            glossary_table=b_chunk,
            target_lang=target_lang,
            source_lang=source_lang,
            annotated_draft=annotated_draft,
            has_error_spans=has_error_spans,
            global_glossary=b_global,
            # Only the rich strategy carries the excerpt: minimal/hybrid are
            # deliberately terse tiers, and growing them would change their
            # whole cost profile for a gain their prompt shape cannot hold.
            context_block=context_block,
        )


__all__ = [
    "PromptContextAssembler",
    "assemble_continuity_context",
    "assemble_critique_issues",
    "assemble_terminology_table",
]
