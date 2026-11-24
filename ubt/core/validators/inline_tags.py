"""Lightweight XML-like inline anchor validator & normalizer for UBT v2 codecs.

Validates that LLM output preserves every rigid self-closing anchor (``<f1/>``,
``<n1/>``, ``<x1/>``) and properly balances every paired style/link anchor
(``<b1>...</b1>``, ``<i2>...</i2>``, ``<a3>...</a3>``, ``<c4>...</c4>``,
``<r5>...</r5>``, ``<s6>...</s6>``) declared in the source unit.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

_TAG_RE = re.compile(r"<\s*(/?)\s*([a-zA-Z]\d+)\s*(/?)\s*>")

# Weak/lightweight LLMs frequently wrap rigid formula anchors like ``<f1/>`` in
# LaTeX math delimiters (``$\text{<f1/>}$``, ``$\mathrm{<f1/>}$``, ``$<f1/>$``,
# ``\(<f1/>\)``, or backticks `` `<f1/>` ``). Unwrapping those restores the exact
# anchor so layout/rendering never prints literal ``$\text{`` around the formula.
_LATEX_WRAPPED_ANCHOR_RE = re.compile(
    r"(?:"
    r"\${1,2}\s*(?:\\(?:text|mathrm|mathbf|mathit|operatorname)\s*\{\s*)?"
    r"(<\s*[fnxFNX]\d+\s*/?\s*>)"
    r"(?:\s*\})?\s*\${1,2}"
    r"|\\\(\s*(?:\\(?:text|mathrm|mathbf|mathit|operatorname)\s*\{\s*)?"
    r"(<\s*[fnxFNX]\d+\s*/?\s*>)"
    r"(?:\s*\})?\s*\\\)"
    r"|`\s*(<\s*[fnxFNX]\d+\s*/?\s*>)\s*`"
    r")"
)


_LATEX_TEXT_ANCHOR_RE = re.compile(
    r"\\(?:text|mathrm|mathbf|mathit|operatorname)\s*\{\s*(<\s*[fnxFNX]\d+\s*/?\s*>)\s*\}"
)
_DOLLAR_ANCHOR_SPAN_RE = re.compile(r"\$([^$\n]{1,120})\$")


def unwrap_latex_wrapped_anchors(text: str, declared_rigid: set[str] | None = None) -> str:
    """Strip accidental LaTeX math or code delimiters around rigid inline anchors."""
    if not text or "<" not in text:
        return text

    def _tag_matches(inner: str) -> str | None:
        tag_m = _TAG_RE.fullmatch(inner.strip())
        if tag_m is None:
            return None
        tag_id = tag_m.group(2).lower()
        if declared_rigid is not None and tag_id not in declared_rigid:
            return None
        return f"<{tag_id}/>"

    def _replace(m: re.Match[str]) -> str:
        inner = m.group(1) or m.group(2) or m.group(3) or ""
        norm = _tag_matches(inner)
        return norm if norm is not None else m.group(0)

    out = _LATEX_WRAPPED_ANCHOR_RE.sub(_replace, text)

    # Also unwrap compound inline-math wrappers where a weak LLM placed a rigid
    # anchor alongside simple operators/digits inside one `$...$` span, e.g.
    # ``$\text{<f2/>} = 0$`` -> ``<f2/> = 0`` or ``$\text{<f5/>} = \text{<f6/>}$`` -> ``<f5/> = <f6/>``.
    def _unwrap_compound(m: re.Match[str]) -> str:
        body = m.group(1)
        if "<" not in body:
            return m.group(0)
        unwrapped = _LATEX_TEXT_ANCHOR_RE.sub(
            lambda tm: _tag_matches(tm.group(1)) or tm.group(0), body
        )
        # Only drop the surrounding `$...$` if a declared anchor is present and no
        # other LaTeX macro commands remain inside the span.
        has_declared = any(
            _tag_matches(tm.group(0)) is not None for tm in _TAG_RE.finditer(unwrapped)
        )
        if has_declared and "\\" not in unwrapped:
            return unwrapped.strip()
        return m.group(0)

    return _DOLLAR_ANCHOR_SPAN_RE.sub(_unwrap_compound, out)


def resolve_inline_tags_for_reflow(
    text: str,
    raw_by_tag: Mapping[str, str] | None = None,
) -> str:
    """Convert native-codec marked text into plain prose safe for reflow renderers.

    Rigid anchors (``<f1/>``) are replaced with their source ``raw_text`` when
    available so inline symbols/variables survive degradation; paired style tags
    (``<b1>...</b1>``, ``<c1>...</c1>``) are stripped while preserving inner text.
    """
    if not text or "<" not in text:
        return text
    cleaned = unwrap_latex_wrapped_anchors(text)
    raw_map = {k.lower(): str(v) for k, v in (raw_by_tag or {}).items() if v}

    def _sub(m: re.Match[str]) -> str:
        is_close = bool(m.group(1))
        tag_id = m.group(2).lower()
        is_self = bool(m.group(3))
        if (is_self or tag_id in raw_map) and not is_close:
            return raw_map.get(tag_id, "")
        return ""

    return _TAG_RE.sub(_sub, cleaned)


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

    # Unwrap accidental LaTeX math/code wrappers around declared rigid anchors
    # (e.g. ``$\text{<f1/>}$`` -> ``<f1/>``) before normalizing tag spelling.
    cleaned_target = (
        unwrap_latex_wrapped_anchors(target_marked, src_rigid_set)
        if src_rigid_set
        else target_marked
    )

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

    normalized = _TAG_RE.sub(_norm, cleaned_target)

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
