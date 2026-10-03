"""Rolling cross-chapter continuity summaries.

Feeds a compact summary of each completed chapter into the draft prompts of
the next chapter, so character names, established renderings, and narrative
state carry across chapter boundaries. One bulk draft-tier call per chapter
transition; failures degrade to a deterministic excerpt so the pipeline never
blocks on summarization.
"""

import logging
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)

_MAX_INPUT_CHARS = 3000
_MAX_SUMMARY_CHARS = 800
_MIN_SUMMARY_CHARS = 10
_DETERMINISTIC_CHARS = 400
# A sentence-boundary cut shorter than this is an abbreviation fragment rather
# than a sentence: the longest common abbreviation with a trailing period
# ("Fig.", "e.g.", "etc.", "Ref.") is four characters.
_MIN_BOUNDARY_CHARS = 5


def extract_chapter_id(block_id: str) -> str:
    """Chapter namespace from a block id (``ch_001#b0002`` or ``ch_001:b0002`` -> ``ch_001``).

    Ids without a delimiter are treated as one single-chapter
    document (returns ``""``), disabling cross-chapter summaries.
    """
    for sep in ("#", ":"):
        if sep in block_id:
            return block_id.split(sep, 1)[0]
    return ""


def collect_chapter_text(blocks: list[Any], max_chars: int = _MAX_INPUT_CHARS) -> str:
    """Assemble the chapter's text for summarization.

    Prefers the translated ``target_text`` (reinforces established renderings),
    falls back to ``draft_text`` then ``source_text``. Truncated to the head
    ``max_chars`` to bound the summary call cost.
    """
    parts: list[str] = []
    total = 0
    for b in blocks:
        text = getattr(b, "target_text", None) or getattr(b, "draft_text", None) or b.source_text
        if not text or not text.strip():
            continue
        parts.append(text.strip())
        total += len(text.strip()) + 1
        if total >= max_chars:
            break
    return "\n".join(parts)[:max_chars].strip()


def deterministic_summary(chapter_text: str, max_chars: int = _DETERMINISTIC_CHARS) -> str:
    """Sentence-boundary truncation of ``chapter_text`` to ``max_chars``.

    Used both as the head excerpt for an unusable LLM summary and to cap an
    over-long but valid one. A boundary that lands very early — the only period
    in the window belongs to an abbreviation ("Fig.", "e.g.", "v1.2") — would
    otherwise collapse the whole summary to a four-character fragment, so a
    boundary shorter than ``_MIN_BOUNDARY_CHARS`` is ignored in favour of the
    hard cut.
    """
    text = chapter_text.strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_end = max((cut.rfind(sep) for sep in ("。", ".", "!", "?", "！", "？")), default=-1)
    if last_end + 1 >= _MIN_BOUNDARY_CHARS:
        return cut[: last_end + 1]
    return cut


def build_summary_prompt(chapter_text: str, target_lang: str) -> tuple[str, str]:
    """Build the bulk chapter-continuation-summary prompt."""
    system_prompt = (
        f"You are a concise story-bible summarizer for a book being translated into {target_lang}.\n"
        "Output ONLY the summary text."
    )
    user_prompt = (
        "### Chapter content (translated so far)\n"
        f"{chapter_text}\n\n"
        "### Task\n"
        f"Write a compact continuation summary (3-5 sentences, in {target_lang}) capturing "
        "the key events or points, established terminology renderings, and tone, "
        "so the next chapter can be translated coherently."
    )
    return system_prompt, user_prompt


async def llm_summary(
    content: str,
    complete: Callable[[str, str], Awaitable[str]],
    target_lang: str,
    *,
    context: str,
) -> str | None:
    """The LLM-summary fold shared by the rolling and hierarchical summarizers.

    Prompt -> complete -> strip echo quotes -> normalize whitespace -> accept a
    summary at or above ``_MIN_SUMMARY_CHARS``, truncated at a sentence
    boundary (the tail is the part continuation needs, and the bulk call was
    already paid for). Returns ``None`` when there is nothing usable — call
    failure, empty echo, or a sub-minimum fragment — so each caller applies its
    own :func:`deterministic_summary` fallback policy.
    """
    try:
        system_prompt, user_prompt = build_summary_prompt(content, target_lang)
        raw = await complete(system_prompt, user_prompt)
        cleaned = (raw or "").strip().strip('"“”').strip()
        normalized = " ".join(cleaned.split())
        if len(normalized) >= _MIN_SUMMARY_CHARS:
            return deterministic_summary(normalized, _MAX_SUMMARY_CHARS)
    except Exception as exc:
        logger.warning("%s: %s", context, exc)
    return None


async def summarize_chapter(
    blocks: list[Any],
    complete: Any,  # Callable[[str, str], Awaitable[str]]
    target_lang: str = "zh",
) -> str:
    """Summarize one completed chapter; degrade to a deterministic excerpt."""
    chapter_text = collect_chapter_text(blocks)
    if not chapter_text:
        return ""
    summary = await llm_summary(
        chapter_text,
        complete,
        target_lang,
        context="Rolling summary LLM call failed",
    )
    return summary or deterministic_summary(chapter_text)
