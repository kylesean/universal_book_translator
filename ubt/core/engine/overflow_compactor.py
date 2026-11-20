"""Bounded LLM overflow compaction loop for 1:1 in-place PDF delivery.

When :class:`NativePDFCodec` reports :class:`OverflowItem`s whose translated
prose exceeds the source container even at the font floor, this module sends a
targeted single-shot compaction prompt per overflowing unit to tighten phrasing
while strictly preserving meaning and all inline anchor tags (``<f1/>``,
``<b1>``).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from ubt.codecs.protocol import OverflowItem, TranslationUnit
from ubt.core.validators.inline_tags import strip_inline_tags, validate_inline_tags

CompleteFn = Callable[[str, str], Awaitable[str]]


@dataclass(frozen=True, slots=True)
class CompactionSummary:
    """Audit metrics for one overflow compaction pass."""

    attempted: int
    accepted: int
    rejected_tag_violation: int
    rejected_not_shorter: int
    compacted: dict[str, str]


def build_compaction_prompt(
    *,
    source_marked: str,
    current_target: str,
    overflow: OverflowItem,
    target_lang: str,
) -> tuple[str, str]:
    """Return ``(system_prompt, user_prompt)`` for bounded academic tightening."""
    system = (
        "You are a senior technical editor specializing in concise publication typesetting. "
        "Tighten the translation to fit the exact target character budget without dropping "
        "facts, numbers, or technical terms. Preserve every XML-like inline tag "
        "(such as <f1/>, <b1>...</b1>, <i2>...</i2>) verbatim and output ONLY the revised "
        "translation with no commentary."
    )
    pct_over = max(5, int(round((overflow.overflow_ratio - 1.0) * 100)))
    user = (
        f"Target language: {target_lang}\n"
        f"Overflow: +{pct_over}% taller than source column box.\n"
        f"Target length budget: <= {overflow.target_char_budget} characters (excluding tags).\n\n"
        f"Source:\n{source_marked}\n\n"
        f"Current translation (too long):\n{current_target}\n\n"
        "Concise revised translation:"
    )
    return system, user


async def compact_overflowing_units(
    *,
    overflows: Sequence[OverflowItem],
    units_by_id: dict[str, TranslationUnit],
    translations: dict[str, str],
    complete_fn: CompleteFn,
    target_lang: str = "zh",
    min_overflow_ratio: float = 1.05,
    max_units: int = 32,
    concurrency: int = 6,
) -> CompactionSummary:
    """Compact up to ``max_units`` overflowing translations concurrently (max 1 pass)."""
    candidates = [
        ov
        for ov in sorted(overflows, key=lambda item: -item.overflow_ratio)
        if ov.overflow_ratio >= min_overflow_ratio
        and ov.unit_id in units_by_id
        and ov.unit_id in translations
    ][:max_units]

    if not candidates:
        return CompactionSummary(
            attempted=0,
            accepted=0,
            rejected_tag_violation=0,
            rejected_not_shorter=0,
            compacted={},
        )

    sem = asyncio.Semaphore(max(1, concurrency))
    compacted: dict[str, str] = {}
    rejected_tags = 0
    rejected_len = 0

    async def _one(ov: OverflowItem) -> tuple[str, str | None, str]:
        unit = units_by_id[ov.unit_id]
        cur = translations[ov.unit_id]
        sys_p, usr_p = build_compaction_prompt(
            source_marked=unit.source_marked,
            current_target=cur,
            overflow=ov,
            target_lang=target_lang,
        )
        async with sem:
            revised = (await complete_fn(sys_p, usr_p)).strip()
        if not revised:
            return ov.unit_id, None, "empty"
        check = validate_inline_tags(unit.source_marked, revised)
        if not check.valid:
            return ov.unit_id, None, "tag"
        if len(strip_inline_tags(check.normalized_text).strip()) >= len(
            strip_inline_tags(cur).strip()
        ):
            return ov.unit_id, None, "length"
        return ov.unit_id, check.normalized_text, "ok"

    results = await asyncio.gather(*(_one(ov) for ov in candidates), return_exceptions=True)
    for res in results:
        if isinstance(res, BaseException):
            continue
        uid, new_text, status = res
        if status == "ok" and new_text is not None:
            compacted[uid] = new_text
        elif status == "tag":
            rejected_tags += 1
        elif status == "length":
            rejected_len += 1

    return CompactionSummary(
        attempted=len(candidates),
        accepted=len(compacted),
        rejected_tag_violation=rejected_tags,
        rejected_not_shorter=rejected_len,
        compacted=compacted,
    )
