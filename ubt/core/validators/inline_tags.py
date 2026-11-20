"""Lightweight XML-like inline anchor validator & normalizer for UBT v2 codecs.

Validates that LLM output preserves every rigid self-closing anchor (``<f1/>``,
``<n1/>``, ``<x1/>``) and properly balances every paired style/link anchor
(``<b1>...</b1>``, ``<i2>...</i2>``, ``<a3>...</a3>``, ``<c4>...</c4>``,
``<r5>...</r5>``, ``<s6>...</s6>``) declared in the source unit.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

_TAG_RE = re.compile(r"<\s*(/?)\s*([a-zA-Z]\d+)\s*(/?)\s*>")


@dataclass(frozen=True, slots=True)
class TagValidationResult:
    """Deterministic outcome of validating inline anchors on one translated unit."""

    valid: bool
    normalized_text: str
    errors: tuple[str, ...] = ()
    missing_rigid: tuple[str, ...] = ()
    extra_tags: tuple[str, ...] = ()


def extract_declared_tags(source_marked: str) -> tuple[list[str], set[str]]:
    """Return ``(rigid_tag_ids, pair_tag_ids)`` declared in ``source_marked``."""
    rigid: list[str] = []
    pairs: set[str] = set()
    for m in _TAG_RE.finditer(source_marked):
        is_close = bool(m.group(1))
        tag_id = m.group(2).lower()
        is_self = bool(m.group(3))
        if is_self:
            rigid.append(tag_id)
        elif not is_close:
            pairs.add(tag_id)
    return rigid, pairs


def strip_inline_tags(marked: str) -> str:
    """Return plain text with all UBT inline anchor tags removed."""
    return _TAG_RE.sub("", marked)


def validate_inline_tags(source_marked: str, target_marked: str) -> TagValidationResult:
    """Validate and lightly normalize inline anchor tags in ``target_marked``."""
    src_rigid_list, src_pairs = extract_declared_tags(source_marked)
    src_rigid_counts = Counter(src_rigid_list)
    src_rigid_set = set(src_rigid_counts)

    # Normalize harmless whitespace/case drift (e.g. `<F1 />` -> `<f1/>`, `<f1>` -> `<f1/>`).
    def _norm(m: re.Match[str]) -> str:
        is_close = bool(m.group(1))
        tag_id = m.group(2).lower()
        is_self = bool(m.group(3))
        if tag_id in src_rigid_set:
            if is_close:
                return ""  # Drop stray closing `</f1>` if LLM wrapped a rigid token
            return f"<{tag_id}/>"
        if is_self:
            return f"<{tag_id}/>"
        return f"</{tag_id}>" if is_close else f"<{tag_id}>"

    normalized = _TAG_RE.sub(_norm, target_marked)

    errors: list[str] = []
    extra_tags: list[str] = []
    seen_rigid: Counter[str] = Counter()
    stack: list[str] = []

    for m in _TAG_RE.finditer(normalized):
        is_close = bool(m.group(1))
        tag_id = m.group(2).lower()
        is_self = bool(m.group(3))

        if is_self:
            if tag_id not in src_rigid_set:
                extra_tags.append(tag_id)
                errors.append(f"unexpected rigid tag <{tag_id}/>")
            else:
                seen_rigid[tag_id] += 1
            continue

        if tag_id not in src_pairs:
            extra_tags.append(tag_id)
            errors.append(f"undeclared pair tag <{'/' if is_close else ''}{tag_id}>")
            continue

        if not is_close:
            stack.append(tag_id)
        else:
            if not stack:
                errors.append(f"unmatched closing tag </{tag_id}>")
            elif stack[-1] != tag_id:
                errors.append(f"mismatched nesting: expected </{stack[-1]}>, got </{tag_id}>")
            else:
                stack.pop()

    if stack:
        errors.append(f"unclosed pair tag(s): {', '.join(f'<{t}>' for t in stack)}")

    missing_rigid: list[str] = []
    for tag_id, expected_count in src_rigid_counts.items():
        actual = seen_rigid[tag_id]
        if actual < expected_count:
            missing_rigid.append(tag_id)
            errors.append(
                f"missing rigid tag <{tag_id}/> (expected {expected_count}, got {actual})"
            )
        elif actual > expected_count:
            errors.append(
                f"duplicated rigid tag <{tag_id}/> (expected {expected_count}, got {actual})"
            )

    return TagValidationResult(
        valid=not errors,
        normalized_text=normalized,
        errors=tuple(errors),
        missing_rigid=tuple(missing_rigid),
        extra_tags=tuple(extra_tags),
    )
