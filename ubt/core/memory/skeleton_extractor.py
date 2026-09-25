"""Language-agnostic LLM Document-Skeleton Terminology Extractor (Tier-2 Bible).

Samples the high-information density skeleton of a document (Title, Abstract,
Section Headings, and Leading Introduction Paragraphs) and executes a single
structured LLM call to identify the exact academic/technical sub-discipline and
extract canonical target-language terminology. Works across all source and
target languages without relying on brittle ASCII or honorific regexes.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from ubt.core.ir.models import BlockType, IRBlock
from ubt.core.memory.bible import BibleEntry, clean_bible_entry

logger = logging.getLogger(__name__)

_SKELETON_MAX_CHARS = 6000
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_CJK_CHAR_RE = re.compile(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]")


def _document_text(blocks: Sequence[IRBlock] | Sequence[str]) -> str:
    parts: list[str] = []
    for entry in blocks:
        parts.append(entry if isinstance(entry, str) else (entry.source_text or ""))
    return "\n".join(parts)


def _count_term_occurrences(term: str, text: str) -> int:
    """Occurrences of ``term`` as a standalone unit (word-bounded for Latin).

    Allows the final word's ordinary English inflections (``coeffect`` matches
    ``coeffects``; ``agent harness`` matches ``agent harnesses``) so a real term
    is not dropped for appearing in its plural/gerund form. A different word that
    merely starts with the term (``cat`` in ``category``) still does not match.
    """
    if not term:
        return 0
    if _CJK_CHAR_RE.search(term):
        return text.count(term)
    words = term.split()
    pattern = r"\s+".join(re.escape(w) for w in words[:-1])
    if pattern:
        pattern += r"\s+"
    pattern += rf"{re.escape(words[-1])}(?:s|es|ed|ing)?"
    return len(re.findall(rf"(?<![A-Za-z0-9]){pattern}(?![A-Za-z0-9])", text, re.IGNORECASE))


def build_document_skeleton(
    blocks: Sequence[IRBlock] | Sequence[str],
    *,
    max_chars: int = _SKELETON_MAX_CHARS,
) -> str:
    """Assemble a compact, high-signal document skeleton from headings and intro prose."""
    if not blocks:
        return ""
    first = blocks[0]
    if isinstance(first, str):
        parts: list[str] = []
        used = 0
        for raw in blocks:
            txt = str(raw).strip()
            if not txt:
                continue
            if used + len(txt) > max_chars:
                rem = max_chars - used
                if rem > 120:
                    parts.append(txt[:rem])
                break
            parts.append(txt)
            used += len(txt) + 1
        return "\n".join(parts)

    ir_blocks: Sequence[IRBlock] = blocks  # type: ignore[assignment]
    headings: list[str] = []
    intro_prose: list[str] = []
    for blk in ir_blocks:
        if blk.skip_translate:
            continue
        src = (blk.source_text or "").strip()
        if not src:
            continue
        if blk.block_type == BlockType.HEADING:
            if len(headings) < 40:
                headings.append(f"[H] {src}")
        elif blk.block_type in (BlockType.NARRATIVE, BlockType.LIST_ITEM) and sum(
            len(x) for x in intro_prose
        ) < int(max_chars * 0.75):
            intro_prose.append(src)

    skeleton_lines: list[str] = []
    budget = max_chars
    for h in headings:
        if len(h) + 1 > budget:
            break
        skeleton_lines.append(h)
        budget -= len(h) + 1
    for p in intro_prose:
        if budget <= 120:
            break
        snippet = p if len(p) <= budget else p[:budget]
        skeleton_lines.append(snippet)
        budget -= len(snippet) + 1

    return "\n".join(skeleton_lines)


def _parse_terms_json(raw: str) -> list[dict[str, Any]]:
    """Extract the terms list from a JSON or markdown-fenced JSON response."""
    text = (raw or "").strip()
    if not text:
        return []
    fence_match = _JSON_FENCE_RE.search(text)
    candidate = fence_match.group(1) if fence_match else text
    if not candidate.startswith("{"):
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            candidate = candidate[start : end + 1]
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        terms = data.get("terms", [])
        if isinstance(terms, list):
            return [t for t in terms if isinstance(t, dict)]
    return []


async def extract_skeleton_terms_llm(
    blocks: Sequence[IRBlock] | Sequence[str],
    *,
    complete_raw_fn: Callable[[str, str], Awaitable[str]] | None,
    source_lang: str,
    target_lang: str,
    max_chars: int = _SKELETON_MAX_CHARS,
) -> list[BibleEntry]:
    """Extract canonical domain terminology from the document skeleton via one LLM call."""
    if complete_raw_fn is None:
        return []
    skeleton = build_document_skeleton(blocks, max_chars=max_chars)
    if len(skeleton.strip()) < 30:
        return []

    system_prompt = (
        f"You are an expert bilingual domain lexicographer and scholarly terminology "
        f"architect ({source_lang} -> {target_lang}).\n"
        "Your task:\n"
        "1. Identify the exact academic/technical sub-discipline of the document.\n"
        "2. Extract 15 to 35 high-value domain-specific technical terms, recurring compound "
        "concepts, and proper nouns/framework names that require strict translation consistency.\n"
        "3. Provide the authoritative, sub-discipline-standard translation in the target language "
        "(e.g. in Programming Language Theory / Category Theory, dual 'co-' concepts like "
        "'coeffect', 'comonad', 'coalgebra' MUST be rendered as '余效应', '余单子', '余代数', "
        "never '上效应'; software/framework names like 'Cordis' or 'VSCode' stay in their "
        "canonical Latin form).\n"
        "Return ONLY valid JSON matching this schema:\n"
        '{"domain": "<sub-discipline>", "terms": [{"source": "<source term>", '
        '"translation": "<target rendering>", "kind": "term"}]}'
    )
    user_prompt = (
        f"Source language: {source_lang}\n"
        f"Target language: {target_lang}\n\n"
        f"Document Skeleton:\n{skeleton}"
    )

    try:
        raw_response = await complete_raw_fn(system_prompt, user_prompt)
    except Exception as exc:
        logger.warning("Skeleton terminology extraction failed: %s", exc)
        return []

    raw_items = _parse_terms_json(raw_response)
    document_text = _document_text(blocks)
    entries: list[BibleEntry] = []
    for item in raw_items:
        src = str(item.get("source", "")).strip()
        tgt = str(item.get("translation", "")).strip()
        kind = str(item.get("kind", "term")).strip() or "term"
        if not src or not tgt:
            continue
        # The skeleton channel is the only one whose terms are not mined from the
        # text, so an LLM can invent a term from a neighbouring sub-discipline.
        # Anchor every entry to the document and rank by real occurrence instead
        # of a sentinel frequency that would otherwise put unverified rows at the
        # top of every block's glossary prompt.
        occurrences = _count_term_occurrences(src, document_text)
        if occurrences == 0:
            logger.debug("Dropping skeleton term %r: not present in the document", src)
            continue
        cleaned = clean_bible_entry(
            source=src,
            translation=tgt,
            kind=kind if kind in ("term", "person", "place", "org") else "term",
        )
        if cleaned is not None:
            entries.append(cleaned.model_copy(update={"frequency": occurrences}))
    return entries
