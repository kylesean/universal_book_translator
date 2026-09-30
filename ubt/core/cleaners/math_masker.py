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

import functools
import re

from ubt.core.cleaners.inline_math import (
    is_math_content,
)
from ubt.core.cleaners.mask_tokens import RestoreStyle, UnmaskReport, restore_masked
from ubt.core.cleaners.mask_tokens import order_by_position as _order_by_position
from ubt.core.cleaners.mask_tokens import token_checksum as _token_checksum

_MASK_PREFIX = "⟦MATH_MASK_"


@functools.lru_cache(maxsize=16)
def _get_math_fuzzy_pattern(mask_prefix: str) -> re.Pattern[str]:
    """Token shape surviving bracket/casing/spacing mutations and an optional checksum."""
    clean_prefix = re.sub(r"^[⟦\[【(\s]+|[⟧\]】)\s]+$", "", mask_prefix).rstrip(":_")
    prefix_pattern = re.sub(r"[:_]+", r"[:_]?", re.escape(clean_prefix))
    if "MATH" in clean_prefix.upper() and clean_prefix.upper() != "MATH_MASK":
        prefix_pattern = rf"(?:{prefix_pattern}|MATH[:_]?MASK|MATH)"
    return re.compile(
        r"[⟦\[【(]?\s*" + prefix_pattern + r"[:_]?(\d{1,6})(?:-([0-9a-z]{3}))?\s*[⟧\]】)]?",
        re.IGNORECASE,
    )


@functools.lru_cache(maxsize=16)
def _get_math_scan_pattern(mask_prefix: str) -> re.Pattern[str]:
    clean_prefix = re.sub(r"^[⟦\[【(\s]+|[⟧\]】)\s]+$", "", mask_prefix).rstrip(":_")
    prefix_pattern = re.sub(r"[:_]+", r"[:_]?", re.escape(clean_prefix))
    if "MATH" in clean_prefix.upper() and clean_prefix.upper() != "MATH_MASK":
        prefix_pattern = rf"(?:{prefix_pattern}|MATH[:_]?MASK|MATH)"
    return re.compile(
        prefix_pattern + r"[:_]?(\d{1,6})(?:-([0-9a-z]{3}))?",
        re.IGNORECASE,
    )


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


class MathMasker:
    """Masks inline math spans before translation and unmasks afterwards."""

    def __init__(self, mask_prefix: str = _MASK_PREFIX) -> None:
        self.mask_prefix = mask_prefix

    def mask(self, text: str) -> tuple[str, dict[str, str]]:
        """Replace math spans with protective tokens; returns (masked, mapping).

        Tokens carry a checksum suffix (``⟦MATH_MASK_0001-a3f⟧``) binding the
        index to the masked original; see :func:`_token_checksum`.
        """
        mapping: dict[str, str] = {}
        counter = 1

        def _replace(match: re.Match[str]) -> str:
            nonlocal counter
            original = match.group(0)
            token = f"{self.mask_prefix}{counter:04d}-{_token_checksum(counter, original)}⟧"
            mapping[token] = original
            counter += 1
            return token

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
        return masked, _order_by_position(masked, mapping)

    def unmask(self, text: str, mapping: dict[str, str]) -> str:
        """Restore math tokens (mutations included); see :func:`restore_masked`."""
        return restore_masked(text, mapping, self._restore_style()).text

    def unmask_checked(self, text: str, mapping: dict[str, str]) -> UnmaskReport:
        """Restore plus integrity verification; see :class:`UnmaskReport`."""
        return restore_masked(text, mapping, self._restore_style())

    def _restore_style(self) -> RestoreStyle:
        return RestoreStyle(
            scan_pattern=_get_math_scan_pattern(self.mask_prefix),
            fuzzy_pattern=_get_math_fuzzy_pattern(self.mask_prefix),
            checksumless_restores=True,
            nested=True,
        )


def count_math_spans(text: str) -> int:
    """Number of inline-math spans in text (0-token diagnostics)."""
    probe = MathMasker()
    _, mapping = probe.mask(text)
    return len(mapping)


def extract_math_spans(text: str) -> list[str]:
    """Inline-math span contents in order (for preservation comparison)."""
    _, mapping = MathMasker().mask(text)
    ordered = sorted(
        mapping.items(),
        key=lambda kv: int(re.search(r"(\d+)", kv[0]).group(1)),  # type: ignore[union-attr]
    )
    return [original for _, original in ordered]
