"""Unified prompt context assembler across Draft, Repair, and Consistency stages.

Provides deterministic context budgeting, consistent terminology injection,
and clean separation of system and user prompt roles across translation workflows.
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
    """Assemble and budget global and chunk terminology tables.

    Returns ``(budgeted_global, budgeted_chunk)``, ensuring neither exceeds
    allocated token quotas.
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
    """Unified coordinator for assembling prompt payloads across Draft, Repair, and Consistency.

    Enforces context budgeting, prevents token blowout, and standardizes terminology
    and error critique structure across all stages.
    """

    def __init__(self, budget: ContextBudget | None = None) -> None:
        self.budget = budget or ContextBudget()

    def assemble_draft(
        self,
        source_text: str,
        *,
        strategy: PromptStrategy | str = "hybrid",
        glossary_table: str = "",
        neighbor_context: str = "",
        target_lang: str = "zh",
        source_lang: str = "en",
        genre_profile: str = "general",
        rolling_summary: str = "",
        global_glossary: str = "",
        few_shot_reference: str = "",
        epoch_summary: str = "",
        domain: str | None = None,
        block_type: str | None = None,
    ) -> tuple[str, str]:
        """Assemble a budgeted single-block draft prompt using the configured strategy."""
        from ubt.core.router.prompts import (
            build_hybrid_draft_prompt,
            build_minimal_draft_prompt,
            build_rich_draft_prompt,
        )

        b_global, b_chunk = assemble_terminology_table(
            global_glossary=global_glossary,
            chunk_glossary=glossary_table,
            budget=self.budget,
        )
        b_epoch, b_rolling = assemble_continuity_context(
            epoch_summary=epoch_summary,
            rolling_summary=rolling_summary,
            budget=self.budget,
        )
        b_neighbor = truncate_text_to_budget(neighbor_context, self.budget.max_neighbor_tokens)
        b_few_shot = truncate_text_to_budget(few_shot_reference, self.budget.max_few_shot_tokens)

        strat_str = str(strategy).lower()
        if "minimal" in strat_str:
            return build_minimal_draft_prompt(
                source_text=source_text,
                glossary_table=b_chunk,
                target_lang=target_lang,
                source_lang=source_lang,
                global_glossary=b_global,
                few_shot_reference=b_few_shot,
                genre_profile=genre_profile,
                domain=domain,
                block_type=block_type,
            )
        if "hybrid" in strat_str:
            return build_hybrid_draft_prompt(
                source_text=source_text,
                glossary_table=b_chunk,
                neighbor_context=b_neighbor,
                target_lang=target_lang,
                source_lang=source_lang,
                genre_profile=genre_profile,
                rolling_summary=b_rolling,
                global_glossary=b_global,
                few_shot_reference=b_few_shot,
                epoch_summary=b_epoch,
                domain=domain,
                block_type=block_type,
            )
        return build_rich_draft_prompt(
            source_text=source_text,
            glossary_table=b_chunk,
            neighbor_context=b_neighbor,
            target_lang=target_lang,
            source_lang=source_lang,
            genre_profile=genre_profile,
            rolling_summary=b_rolling,
            global_glossary=b_global,
            few_shot_reference=b_few_shot,
            epoch_summary=b_epoch,
            domain=domain,
            block_type=block_type,
        )

    def assemble_macro_chunk(
        self,
        blocks: list[tuple[str, str]],
        *,
        glossary_table: str = "",
        neighbor_context: str = "",
        target_lang: str = "zh",
        source_lang: str = "en",
        genre_profile: str = "general",
        rolling_summary: str = "",
        global_glossary: str = "",
        few_shot_reference: str = "",
        epoch_summary: str = "",
        domain: str | None = None,
        section_context: str = "",
    ) -> tuple[str, str]:
        """Assemble a budgeted multi-block macro-chunk draft prompt."""
        from ubt.core.router.prompts import build_macro_chunk_draft_prompt

        b_global, b_chunk = assemble_terminology_table(
            global_glossary=global_glossary,
            chunk_glossary=glossary_table,
            budget=self.budget,
        )
        b_epoch, b_rolling = assemble_continuity_context(
            epoch_summary=epoch_summary,
            rolling_summary=rolling_summary,
            budget=self.budget,
        )
        b_neighbor = truncate_text_to_budget(neighbor_context, self.budget.max_neighbor_tokens)
        b_few_shot = truncate_text_to_budget(few_shot_reference, self.budget.max_few_shot_tokens)
        b_section = truncate_text_to_budget(section_context, self.budget.max_neighbor_tokens)

        return build_macro_chunk_draft_prompt(
            blocks=blocks,
            glossary_table=b_chunk,
            neighbor_context=b_neighbor,
            target_lang=target_lang,
            source_lang=source_lang,
            genre_profile=genre_profile,
            rolling_summary=b_rolling,
            global_glossary=b_global,
            few_shot_reference=b_few_shot,
            epoch_summary=b_epoch,
            domain=domain,
            section_context=b_section,
        )

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
        )

    def assemble_consistency(
        self,
        source_text: str,
        draft_text: str,
        consistency_flags: list[str],
        *,
        strategy: PromptStrategy | str = "rich",
        glossary_table: str = "",
        global_glossary: str = "",
        target_lang: str = "zh",
        source_lang: str = "en",
        prior_error_flags: list[str] | None = None,
    ) -> tuple[str, str]:
        """Assemble a consistency re-translation prompt with strict terminology enforcement."""
        combined_flags = list(prior_error_flags or []) + list(consistency_flags)
        return self.assemble_repair(
            source_text=source_text,
            draft_text=draft_text,
            error_flags=combined_flags,
            strategy=strategy,
            glossary_table=glossary_table,
            global_glossary=global_glossary,
            target_lang=target_lang,
            source_lang=source_lang,
        )


__all__ = [
    "PromptContextAssembler",
    "assemble_continuity_context",
    "assemble_critique_issues",
    "assemble_terminology_table",
]
