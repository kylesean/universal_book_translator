"""Inline-math masking for LLM translation protection (BabelDOC placeholder paradigm).

Display math normally arrives as whole ``FORMULA`` blocks (``skip_translate``)
and never reaches the LLM. But *inline* math inside narrative prose —
``$K_{t}$``, ``\\(x^2\\)``, ``$$…$$`` pasted into Markdown/EPUB/HTML sources —
would otherwise be translated as prose ("K 的 t" etc.) or corrupted by
glossary substitution. Like citations and code, inline math is therefore
masked behind deterministic tokens before drafting and unmasked afterwards —
the same placeholder-protection contract as
:class:`~ubt.core.cleaners.code_masker.CodeMasker`.

Deliberately there is NO "formula hint" instruction to the LLM (BabelDOC's
``--add-formula-placehold-hint`` is off by default upstream for the same
reason: hints degrade quality). The token simply replaces the math, so the
model translates around an opaque placeholder.

Currency guard: ``$5``, ``$100`` and ``$3.50`` are money, not math, and are
never masked. A ``$…$`` span is math only when its content is not a bare
number.
"""

import re

from ubt.core.cleaners.inline_math import (
    is_math_content,
)
from ubt.core.cleaners.mask_tokens import (
    BaseMasker,
    RestoreStyle,
    TokenFactory,
    token_patterns,
)
from ubt.core.cleaners.mask_tokens import order_by_position as _order_by_position

_MASK_PREFIX = "⟦MATH_MASK_"


_DISPLAY_DOLLAR_PATTERN = re.compile(r"\$\$\s*(.+?)\s*\$\$", re.DOTALL)
# Multi-line LaTeX math environments (B-track): align/gather/equation blocks
# span newlines so none of the single-line patterns below can see them.
# Runs FIRST — a nested \begin{cases} inside $...$ is masked as one unit and
# the surviving outer delimiters are picked up by the inline pass after.
_MATH_ENV_NAMES = (
    "align",
    "aligned",
    "gather",
    "gathered",
    "multline",
    "flalign",
    "eqnarray",
    "equation",
    "split",
    "cases",
    "matrix",
    "pmatrix",
    "bmatrix",
    "vmatrix",
    "Vmatrix",
    "array",
)
_MATH_ENV_PATTERN = re.compile(
    r"\\begin\{(" + "|".join(_MATH_ENV_NAMES) + r")\*?\}"
    r".*?"
    r"\\end\{\1\*?\}",
    re.DOTALL,
)
# LaTeX \(...\) and \[...\] groups (single-line; display blocks are FORMULA).
_PAREN_MATH_PATTERN = re.compile(r"\\\((.+?)\\\)")
_BRACKET_MATH_PATTERN = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)


class MathMasker(BaseMasker):
    """Masks inline math spans before translation and unmasks afterwards."""

    default_prefix = _MASK_PREFIX

    def mask(self, text: str) -> tuple[str, dict[str, str]]:
        """Replace math spans with protective tokens; returns (masked, mapping).

        Tokens carry a checksum suffix (``⟦MATH_MASK_0001-a3f⟧``) binding the
        index to the masked original; see :func:`~ubt.core.cleaners.mask_tokens.token_checksum`.
        """
        factory = TokenFactory(self.mask_prefix)

        def _replace(match: re.Match[str]) -> str:
            return factory.next(match.group(0))

        def _replace_guarded(match: re.Match[str]) -> str:
            if not is_math_content(match.group(1)):
                return match.group(0)
            return _replace(match)

        masked = _MATH_ENV_PATTERN.sub(_replace, text)
        masked = _DISPLAY_DOLLAR_PATTERN.sub(_replace, masked)
        masked = _PAREN_MATH_PATTERN.sub(_replace, masked)
        masked = _BRACKET_MATH_PATTERN.sub(_replace, masked)
        from ubt.core.cleaners.inline_math import iter_inline_math

        inline_matches = list(iter_inline_math(masked))
        if inline_matches:
            pieces: list[str] = []
            last_end = 0
            for m in inline_matches:
                pieces.append(masked[last_end : m.start()])
                pieces.append(_replace(m))
                last_end = m.end()
            pieces.append(masked[last_end:])
            masked = "".join(pieces)
        return masked, _order_by_position(masked, factory.mapping)

    def _restore_style(self) -> RestoreStyle:
        fuzzy, scan = token_patterns(self.mask_prefix, "MATH")
        return RestoreStyle(
            scan_pattern=scan,
            fuzzy_pattern=fuzzy,
            checksumless_restores=True,
            nested=True,
        )


def extract_math_spans(text: str) -> list[str]:
    """Inline-math span contents in order (for preservation comparison)."""
    _, mapping = MathMasker().mask(text)

    def _key_index(item: tuple[str, str]) -> int:
        match = re.search(r"(\d+)", item[0])
        return int(match.group(1)) if match is not None else 0

    ordered = sorted(mapping.items(), key=_key_index)
    return [original for _, original in ordered]
