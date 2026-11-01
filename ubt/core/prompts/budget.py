"""Token budgeting and section truncation for compiled prompts.

Truncation here is structure-aware: a markdown table is cut on row boundaries so
it stays a valid table, and plain text is cut in units the estimator agrees on.
A section that gets trimmed therefore never becomes a section that gets
malformed. The caps themselves belong to :class:`ContextBudget`.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass


def _get_default_estimator() -> Callable[[str], int]:
    from ubt.core.engine.cost_estimate import count_text_tokens

    return count_text_tokens


def estimate_prompt_tokens(*texts: str) -> int:
    """Estimate token count across multiple text segments using script-aware heuristics."""
    from ubt.core.engine.cost_estimate import count_text_tokens

    return sum(count_text_tokens(text) for text in texts if text)


@dataclass(frozen=True, slots=True)
class ContextBudget:
    """Token budget caps across prompt sections for an LLM generation call.

    Prevents unbounded glossary, continuity, or few-shot injection from pushing
    requests past context boundaries or crowding out the source text.
    """

    max_context_tokens: int = 16384
    max_terminology_tokens: int = 3072  # ~12,000 characters
    max_history_tokens: int = 2048  # ~8,000 characters (epoch + rolling context)


_TABLE_HEADER_SEP_RE = re.compile(r"^\|?(\s*:?-+:?\s*\|)+\s*$")


def truncate_markdown_table_to_budget(
    table: str,
    max_tokens: int,
    estimate_fn: Callable[[str], int] | None = None,
) -> str:
    """Truncate a markdown table to fit within ``max_tokens`` without row splitting.

    Preserves the header and delimiter rows, greedily including data rows in
    order until the next row would breach ``max_tokens``.
    """
    stripped = table.strip()
    if not stripped or max_tokens <= 0:
        return ""

    estimator = estimate_fn or _get_default_estimator()
    if estimator(stripped) <= max_tokens:
        return stripped

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if len(lines) < 3 or "|" not in lines[0]:
        return truncate_text_to_budget(stripped, max_tokens, estimate_fn=estimator)

    header = lines[0]
    delimiter = lines[1]
    if not _TABLE_HEADER_SEP_RE.match(delimiter):
        return truncate_text_to_budget(stripped, max_tokens, estimate_fn=estimator)

    base_table = f"{header}\n{delimiter}"
    base_cost = estimator(base_table)
    if base_cost >= max_tokens:
        return base_table

    kept_rows: list[str] = []
    omitted_count = 0
    current_text = base_table

    for row in lines[2:]:
        candidate = f"{current_text}\n{row}"
        if estimator(candidate) <= max_tokens:
            kept_rows.append(row)
            current_text = candidate
        else:
            omitted_count += 1

    if not kept_rows:
        return base_table

    result = f"{base_table}\n" + "\n".join(kept_rows)
    if omitted_count > 0:
        notice = f"\n<!-- [{omitted_count} term entries omitted to fit context budget] -->"
        if estimator(result + notice) <= max_tokens + 20:
            result += notice
    return result


def truncate_text_to_budget(
    text: str,
    max_tokens: int,
    suffix: str = "\n[... truncated to fit budget ...]",
    estimate_fn: Callable[[str], int] | None = None,
) -> str:
    """Truncate free-form text to fit within ``max_tokens``, preserving paragraph bounds."""
    stripped = text.strip()
    if not stripped or max_tokens <= 0:
        return ""

    estimator = estimate_fn or _get_default_estimator()
    if estimator(stripped) <= max_tokens:
        return stripped

    suffix_cost = estimator(suffix)
    allowed_tokens = max(10, max_tokens - suffix_cost)

    paragraphs = stripped.split("\n\n")
    kept_paragraphs: list[str] = []
    current_cost = 0

    for para in paragraphs:
        para_cost = estimator(para)
        if kept_paragraphs and (current_cost + para_cost > allowed_tokens):
            break
        kept_paragraphs.append(para)
        current_cost += para_cost
        if current_cost >= allowed_tokens:
            break

    if kept_paragraphs:
        return "\n\n".join(kept_paragraphs).strip() + suffix

    lines = stripped.splitlines()
    kept_lines: list[str] = []
    current_cost = 0
    for line in lines:
        line_cost = estimator(line)
        if kept_lines and (current_cost + line_cost > allowed_tokens):
            break
        kept_lines.append(line)
        current_cost += line_cost
        if current_cost >= allowed_tokens:
            break

    if kept_lines:
        return "\n".join(kept_lines).strip() + suffix

    char_budget = max(20, allowed_tokens * 3)
    return stripped[:char_budget].rstrip() + suffix


__all__ = [
    "ContextBudget",
    "estimate_prompt_tokens",
    "truncate_markdown_table_to_budget",
    "truncate_text_to_budget",
]
