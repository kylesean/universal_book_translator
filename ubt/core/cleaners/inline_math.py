"""Shared, currency-aware detection of inline ``$...$`` math spans.

A naive ``\\$[^\\$\\n]+\\$`` pairs the first ``$`` with the next one on the line,
so ``"$5 and $10"`` becomes the "math" span ``"$5 and $"`` and every term inside
it is shielded from glossary enforcement / CJK spacing. Three passes would
carry their own copy of that rule and drifted; this module is the single home.

The rule has two parts:

- **Delimiters must not touch whitespace** — ``$5 and $10`` has a space before
  the second ``$``, so it is not a math pair at all.
- **The interior must not be currency/prose** — ``$5$`` and ``$3.50$`` are
  money; a bracket-glued interior with no LaTeX signal is prose.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from ubt.core.cjk_ranges import HAN_KANA_HANGUL_RANGES

#: Inline ``$...$`` whose delimiters do not touch whitespace.
INLINE_DOLLAR_PATTERN = re.compile(r"\$(?!\s)([^$\n]+?)(?<!\s)\$")

# Bare numbers with optional thousands/decimal separators: money, not math.
_CURRENCY_PATTERN = re.compile(r"^\d[\d,.]*$")
# Currency chains/ranges ($10-$20, $5–$10, $10-20$)
_CURRENCY_RANGE_PATTERN = re.compile(r"^\d[\d,.]*\s*[-–—]\s*(?:\$?\d[\d,.]*)?$")
# Signs the interior is real math: a \command, sub/superscript, or brace.
_LATEX_SIGNAL_PATTERN = re.compile(r"\\[A-Za-z]+|[_^{}]")
# Han (incl. Extension A) + kana + hangul, from the shared tier table.
_CJK_RANGES = HAN_KANA_HANGUL_RANGES


def _has_bare_cjk(content: str) -> bool:
    """True when content contains CJK characters outside LaTeX text commands."""
    has_cjk = any(any(lo <= ord(ch) <= hi for lo, hi in _CJK_RANGES) for ch in content)
    if not has_cjk:
        return False
    # If wrapped in a LaTeX text environment, it may be legitimate math-mode text
    return not any(cmd in content for cmd in ("\\text", "\\mathrm", "\\mbox", "\\operatorname"))


def is_math_content(content: str) -> bool:
    """True when a ``$...$`` interior is math rather than currency/prose."""
    stripped = content.strip()
    if not stripped:
        return False
    if _CURRENCY_PATTERN.match(stripped) or _CURRENCY_RANGE_PATTERN.match(stripped):
        return False
    if _has_bare_cjk(stripped):
        return False
    # A lone ``$`` is a currency-unit marker, not a delimiter. Two of them on
    # one line -- a table header's ``Eff. ($) ... Token ($)`` -- pair up into a
    # span whose content is a row of prose, so Gate 4 failed the block for a
    # "math span mismatch" the draft can never fix. The tell is a span glued to
    # brackets with no math inside it; a stray-dollar-shattered formula
    # (``$\psi = V \ln($``) ends with ``(`` too, but carries \commands.
    bracket_glued = stripped.startswith(")") or stripped.endswith("(")
    return not bracket_glued or bool(_LATEX_SIGNAL_PATTERN.search(stripped))


def iter_inline_math(text: str) -> Iterator[re.Match[str]]:
    """Yield regex matches for genuine inline ``$...$`` math spans in ``text``."""
    pos = 0
    while pos < len(text):
        match = INLINE_DOLLAR_PATTERN.search(text, pos)
        if match is None:
            break
        if is_math_content(match.group(1)):
            yield match
            pos = match.end()
        else:
            pos = match.start() + 1


def inline_math_spans(text: str) -> list[tuple[int, int]]:
    """Character intervals of genuine inline ``$...$`` math spans."""
    return [(m.start(), m.end()) for m in iter_inline_math(text)]
