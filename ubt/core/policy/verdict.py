"""Translation verdict: zero-LLM-cost pre-filter (roadmap P8).

UBT already gates *after* drafting (0-Token FastPass); this gate runs
*before* any model call. :func:`judge_block` applies ten ordered
``KEEP_ORIGIN`` short-circuit rules — the first match wins and the block
ships verbatim. Anything surviving all ten rules is translated.

Rules only ever *withhold* translation; they never rewrite text, so a
false positive costs style, never correctness — and every keep carries a
``policy_reason`` for audit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ubt.core.ir.models import IRBlock, LayoutRole, SemanticRole
from ubt.core.policy.layout_policy import (
    BYTE_WORD_RE,
    HEXDUMP_MIN_SHARE,
    HEXDUMP_MIN_TOKENS,
    INDEX_MIN_LINES,
    INDEX_MIN_SINGLE_SHARE,
    ISBN_DIGITS_RE,
    NON_PROSE_FLOWS,
    NON_TEXT_BLOCK_TYPES,
    SHORT_LABEL_MAX_LEN,
    SHORT_LABEL_RE,
    SINGLE_TOKEN_LINE_RE,
    UNICODE_WORD_RE,
    URL_DOI_RE,
)

# Two or more purely-lowercase words read as prose ("effective mobility"), not
# as a figure/table label ("B1", "Fig. 3", "kg"). Rule 10 kept such prose in the
# source language; only genuine short labels should be kept.
_PROSE_PHRASE_RE = re.compile(r"\b[a-z]{2,}\b[ _]+\b[a-z]{2,}\b")


@dataclass(frozen=True)
class Verdict:
    """Pre-draft judgement for one block."""

    translate: bool
    reason: str  # "verdict:translate" or "verdict:<rule>"
    should_call_model: bool = False

    def __post_init__(self) -> None:
        if self.translate != self.should_call_model:
            raise ValueError("translate and should_call_model must agree")


def _is_hexdump(text: str) -> bool:
    tokens = text.split()
    if len(tokens) < HEXDUMP_MIN_TOKENS:
        return False
    byte_words = sum(1 for t in tokens if BYTE_WORD_RE.match(t))
    return byte_words >= HEXDUMP_MIN_TOKENS and byte_words / len(tokens) >= HEXDUMP_MIN_SHARE


def _is_index_list(text: str) -> bool:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < INDEX_MIN_LINES:
        return False
    single = sum(1 for ln in lines if SINGLE_TOKEN_LINE_RE.match(ln))
    return single / len(lines) >= INDEX_MIN_SINGLE_SHARE


def judge_block(block: IRBlock, *, translate_chrome: bool = False) -> Verdict:
    """Return the pre-draft verdict for ``block`` (pure function).

    ``translate_chrome`` lets HEADER/FOOTER roles (never PAGE_NUMBER) reach
    the translator; the caller is responsible for clearing the adapter's
    parse-time chrome skip first.
    """
    text = block.source_text or ""
    stripped = text.strip()

    # Rule 1: an explicit prior verdict is respected, never recomputed — it
    # runs before the content rules so an explicitly kept block keeps its
    # audit reason even when its text is empty or placeholder-only.
    if block.policy_translate is False:
        return Verdict(False, block.policy_reason or "verdict:explicit_keep")
    # Rule 2: empty blocks.
    if not stripped:
        return Verdict(False, "verdict:empty")
    # Rule 3: pure placeholders (no word/CJK characters at all).
    if not UNICODE_WORD_RE.search(stripped):
        return Verdict(False, "verdict:placeholder")
    # Rule 4: non-text block types / adapter-marked skips keep origin.
    if block.skip_translate or block.block_type in NON_TEXT_BLOCK_TYPES:
        return Verdict(False, "verdict:non_text")
    # Rule 5: chrome layout roles never enter translation (page numbers
    # never do, even under the chrome opt-in).
    if block.layout_role in {LayoutRole.HEADER, LayoutRole.FOOTER, LayoutRole.PAGE_NUMBER} and (
        not translate_chrome or block.layout_role is LayoutRole.PAGE_NUMBER
    ):
        return Verdict(False, "verdict:chrome")
    # Rule 6: non-prose semantic roles never enter translation.
    if block.semantic_role in {
        SemanticRole.REFERENCE,
        SemanticRole.METADATA,
        SemanticRole.AFFILIATION,
    }:
        return Verdict(False, "verdict:non_prose_role")
    # Rule 7: identifier-only blocks (bare URL / DOI / ISBN).
    if URL_DOI_RE.match(stripped):
        return Verdict(False, "verdict:identifier_only")
    digits_only = re.sub(r"[- ]", "", stripped)
    if ISBN_DIGITS_RE.match(digits_only):
        return Verdict(False, "verdict:identifier_only")
    # Rule 8: hexdumps / byte tables.
    if _is_hexdump(stripped):
        return Verdict(False, "verdict:hexdump")
    # Rule 9: index-like single-token line lists.
    if _is_index_list(stripped):
        return Verdict(False, "verdict:index_list")
    # Rule 10: short alnum labels outside the main story flow.
    if (
        block.flow_id in NON_PROSE_FLOWS
        and len(stripped) <= SHORT_LABEL_MAX_LEN
        and SHORT_LABEL_RE.match(stripped)
        and not _PROSE_PHRASE_RE.search(stripped)
    ):
        return Verdict(False, "verdict:short_label")
    return Verdict(True, "verdict:translate", should_call_model=True)


def apply_verdict(block: IRBlock, verdict: Verdict) -> None:
    """Stamp the verdict onto the block (auditable via ledger v5 columns)."""
    block.policy_translate = verdict.translate
    block.policy_reason = verdict.reason
    if not verdict.translate:
        block.skip_translate = True


__all__ = [
    "Verdict",
    "apply_verdict",
    "judge_block",
]
