"""Summary primitives for the hierarchical (L2/L3) memory manager.

``collect_chapter_text`` / ``llm_summary`` / ``deterministic_summary`` are the
one assembly + call + fallback policy for a summary of drafted blocks. The
hierarchical manager calls them on step / chapter transitions; there is no
separate chapter-boundary summarizer anymore -- that second path wrote the
same ``macro_ctx`` slot through its own LLM call and was merged away.
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
    """Assemble the text for summarization, biased to its most recent end.

    Prefers the translated ``target_text`` (reinforces established renderings),
    falls back to ``draft_text`` then ``source_text``.

    Selection is *tail-first*: the summary this feeds is the continuation
    context ("what just happened, so the next block stays coherent"), and the
    L1 neighbour window only reaches ~300 chars back. Taking the head of a
    step buffer dropped precisely the newest blocks every time — measured: a
    just-crossed 3,500-char step (the snapshot trigger) collected 3,000 chars
    ending at block 15 of 20, whose established renderings the summary was
    supposed to capture. Blocks are walked newest-first and re-ordered to
    chronological before joining, so the *content* is recency-biased while the
    text itself still reads in narrative order.
    """
    parts: list[str] = []
    total = 0
    for b in reversed(blocks):
        text = getattr(b, "target_text", None) or getattr(b, "draft_text", None) or b.source_text
        if not text or not text.strip():
            continue
        stripped = text.strip()
        parts.append(stripped)
        total += len(stripped) + 1
        if total >= max_chars:
            break
    parts.reverse()
    joined = "\n".join(parts)
    if len(joined) <= max_chars:
        return joined
    # The over-budget block is the *oldest* one (newest-first selection), so
    # the trim keeps its tail — the part nearest the recap's subject — and the
    # newest blocks survive whole. A head slice here cut the newest block
    # mid-sentence.
    return joined[-max_chars:].strip()


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
