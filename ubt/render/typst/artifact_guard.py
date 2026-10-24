r"""Pipeline sentinels that must never reach the typesetter.

The masking families (math, code, citation, email, ``SOUP``), the emphasis
sentinels, and the semantic-break delimiter all write ``⟦…⟧`` tokens into text
that is *supposed* to be consumed before delivery. Each family owns its own
prefix and its own restore pass, but restore is a chain
(``citation → soup → math → code → email``), so a token one pass cannot account
for reaches the next stage, and eventually the renderer.

Nothing used to notice. The token then entered :func:`typstify_math`, and Typst's
math mode mangled it in a family-specific way — each one a compile failure that
lost the **whole document**, because the renderer compiles the document as a
single unit:

===================  ==================================================
sentinel             what Typst did
===================  ==================================================
``⟦SOUP_MASK_0001⟧`` ``_`` is the subscript operator, so ``#"SOUP"_…``
                     entered code mode → ``unknown variable: SOUP``
``⟦B⟧`` / ``⟦/B⟧``    ``/`` is the fraction operator → ``unexpected slash``
``\llbracket B …``   ``\llbracket`` is not a Typst name →
                     ``unknown variable: lbracket``
===================  ==================================================

Five gold-corpus documents failed this way (alphafold2,
attention-is-all-you-need, mamba, neural-ode, and clip through the table path).

This module is the *check layer* for that class: one owner of "what a pipeline
sentinel looks like", and one predicate — :func:`has_sentinel` — the renderer
consults before it transpiles anything. A sentinel found inside a would-be math
span makes the span prose, which renders the token literally. That is the honest
outcome: the span's restore failed, and a visible token is attributable where a
vanished equation is not.

A model may also *transcribe* the emphasis marker as the LaTeX brackets it
resembles (:data:`_LATEX_EMPHASIS_RE`), so both spellings are recognised.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: The literal sentinel prefixes the pipeline writes. Kept as literals rather
#: than imported from the maskers so this module has no dependency and can be
#: used by any layer; ``tests/unit/render/test_artifact_guard.py`` asserts they
#: match each masker's ``default_prefix``, which is what stops the two drifting.
_SENTINEL_PREFIXES: tuple[tuple[str, str], ...] = (
    ("\u27e6B\u27e7", "emphasis"),
    ("\u27e6/B\u27e7", "emphasis"),
    ("\u27e6I\u27e7", "emphasis"),
    ("\u27e6/I\u27e7", "emphasis"),
    ("\u27e6SEMANTIC_BREAK\u27e7", "semantic_break"),
    ("\u27e6MATH_MASK_", "math"),
    ("\u27e6SOUP_MASK_", "soup"),
    ("\u27e6CITE_MASK_", "citation"),
    ("\u27e6CODE_MASK_", "code"),
    ("\u27e6EMAIL_MASK_", "email"),
    ("\u27e6MASK_", "mask"),
)

#: Any ``⟦…⟧`` span. A sentinel never legitimately spans a line break, so
#: ``[^⟧\n]*`` bounds the match and a lone ``⟦`` in ordinary prose cannot
#: swallow the rest of the paragraph.
_SENTINEL_RE = re.compile(r"\u27e6[^\u27e7\n]*\u27e7")

#: The emphasis marker as a model may transcribe it. A model reading ``⟦B⟧`` as
#: a mathematical bracket writes ``\llbracket B \rrbracket`` — the same marker,
#: which is why :mod:`ubt.core.ir.emphasis` accepts both spellings. ``\\+``
#: tolerates the extra backslash Typst escaping adds (``\llbracket`` →
#: ``\\llbracket`` in the emitted source), so the same pattern recognises the
#: marker in raw text *and* counts it in the rendered source.
#:
#: ``[BI]`` covers bold *and* italic: ``⟦I⟧`` was missing from the registry and
#: from this pattern, so a target whose italic marker the model rewrote as
#: ``\llbracket I \rrbracket`` was not recognised as a sentinel at all. It then
#: reached the math transpiler as a stray command, and one such formula fails
#: the whole document compile.
_LATEX_EMPHASIS_RE = re.compile(r"\\+llbracket\s*(?:/\s*)?[BbIi]\s*\\+rrbracket")


@dataclass(frozen=True, slots=True)
class Sentinel:
    """One pipeline sentinel found in text that should no longer carry it."""

    kind: str
    text: str


def find_sentinels(text: str) -> list[Sentinel]:
    """Every pipeline sentinel in ``text``, in order of appearance."""
    found: list[Sentinel] = []
    for match in _SENTINEL_RE.finditer(text):
        token = match.group(0)
        kind = next(
            (name for prefix, name in _SENTINEL_PREFIXES if token.startswith(prefix)),
            "unknown",
        )
        found.append(Sentinel(kind=kind, text=token))
    for match in _LATEX_EMPHASIS_RE.finditer(text):
        found.append(Sentinel(kind="emphasis", text=match.group(0)))
    return found


def has_sentinel(text: str) -> bool:
    """True when ``text`` still carries any pipeline sentinel.

    The renderer's precondition: a span that answers ``True`` is escaped as
    prose instead of transpiled as math. One predicate covers every family, so a
    new masker cannot reintroduce the bug by having a prefix the renderer does
    not know.
    """
    return _SENTINEL_RE.search(text) is not None or _LATEX_EMPHASIS_RE.search(text) is not None


def sentinel_kinds(text: str) -> list[str]:
    """The distinct sentinel families in ``text`` (for a quality report)."""
    return sorted({s.kind for s in find_sentinels(text)})


__all__ = ["Sentinel", "find_sentinels", "has_sentinel", "sentinel_kinds"]
