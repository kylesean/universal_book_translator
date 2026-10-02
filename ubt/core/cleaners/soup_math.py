"""Unicode-soup math protection for PDF-extracted narrative text.

Display math normally arrives as whole ``FORMULA`` blocks and never reaches
the LLM, and LaTeX-delimited inline math is masked by
:class:`~ubt.core.cleaners.math_masker.MathMasker`. But dense textbook pages
make docling merge display equations *into* narrative blocks as delimiter-free
unicode soup (``whereQ0``, ``βSI=e−ψpert/(2Vtm)``, ``5CfinVtm``): no ``$``,
no backslash, nothing ``MathMasker`` can see. Left unmasked, the LLM
"translates" the soup (``——√((——))`` debris in the PDF) and the anchored
overlay then blanks the perfect vector original underneath.

This module detects such spans (precision-first: a span must carry real math
signal, never bare identifiers) and masks them behind opaque tokens reusing
``MathMasker``'s checksum-verified restore machinery, so a swallowed token
fails closed into the repair loop instead of shipping silent math loss.
"""

import re

from ubt.core.cjk_ranges import HAN_UNIFIED_CLASS
from ubt.core.cleaners.math_masker import MathMasker

_GREEK_RE = re.compile(r"[\u0370-\u03ff\u1f00-\u1fff]")
_MATH_OP_RE = re.compile(
    r"[∀∂∃∅∇∈∋∏∑√∞∧∨∩∪∴∵∼≈≠≡≤≥≪≫⊂⊃⊆⊇⊕⊖⊗⊘⋅∘∙±×÷¬¯°′″¼½¾¹²³∀∃∠⊥∥→←↑↓⇒⇔−│‖⟨⟩⌈⌉⌊⌋]"
)
_SUB_SUP_RE = re.compile(r"[₀-₉₍₎₊₋₌₎ⁿ⁰-⁹⁺⁻⁼⁽⁾²³¹]")
_CJK_RE = re.compile(f"[{HAN_UNIFIED_CLASS}]")

_RUN_CHAR_RE = re.compile(
    r"[A-Za-z0-9\u0370-\u03ff\u1f00-\u1fff_^'{}\[\]()\/|+\-*=<>,.;:!%&?°′″±×÷¬¯²³¹₀-₉₍₎₊₋₌₎ⁿ⁰-⁹⁺⁻⁼⁽⁾∀∂∃∅∇∈∋∏∑√∞∧∨∩∪∴∵∼≈≠≡≤≥≪≫⊂⊃⊆⊇⊕⊖⊗⊘⋅∘∙∠⊥∥→←↑↓⇒⇔−│‖⟨⟩⌈⌉⌊⌋]"
)
_WS_RE = re.compile(r"\s+")


def _run_spans(text: str) -> list[tuple[int, int]]:
    """Maximal runs of math-adjacent characters (CJK/whitespace break runs)."""
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for i, ch in enumerate(text):
        if _RUN_CHAR_RE.match(ch):
            if start is None:
                start = i
        else:
            if start is not None:
                spans.append((start, i))
                start = None
    if start is not None:
        spans.append((start, len(text)))
    return spans


def _qualifies(run: str) -> bool:
    """True when a run carries real math signal (precision-first)."""
    if len(run) < 2:
        # Lone operator: only when glued to neighbours (``e−ψ``). Spaced
        # singles (``100 ° C``, ``F ≪ F``) are ordinary prose the model
        # copies safely; masking each would litter placeholder tokens.
        return False
    math_hits = (
        len(_GREEK_RE.findall(run)) + len(_MATH_OP_RE.findall(run)) + len(_SUB_SUP_RE.findall(run))
    )
    if math_hits >= 2:
        return True
    if math_hits == 1:
        if "_" in run or "^" in run or "(" in run or ")" in run:
            return True
        if _GREEK_RE.match(run[0]):
            return True
        return bool(len(run) >= 2 and any(c.isdigit() for c in run))
    return bool(
        ("_" in run or "^" in run)
        and len(run) >= 4
        and any(c.isupper() or c.isdigit() for c in run)
    )


_PARAM_ASSIGN_RE = re.compile(
    r"\b([A-Za-z\u0370-\u03ff][A-Za-z0-9_,\u0370-\u03ff]{0,12}(?:\([A-Za-z0-9_,\s=\u0370-\u03ff]+\))?)\s*([=≈~≤≥<>])\s*"
    r"([+-]?\d+(?:\.\d+)?(?:\s*(?:[×x*·]\s*10\^?[-−]?\d+|\^[-−]?\d+))?)"
    # The bare-letter unit alternative ("s" in "5 s") must not end mid-word:
    # without the trailing guard "5 square meters" matched "area = 5 s" and
    # masking split "square" into "⟦…⟧quare". Multi-char units need no guard.
    r"(\s*(?:cm[-−^0-9]+|cm[³²]|%|eV|Hz|deg|°C|[μuµncdmkMGT]?[mgsSAFVWΩ](?:[-−^0-9]+)?(?:/[A-Za-z·0-9]+)*(?![A-Za-z])))?"
)


_TOKEN_SPAN_RE = re.compile(r"⟦[^⟧]*⟧")


def _subtract_token_spans(spans: list[tuple[int, int]], text: str) -> list[tuple[int, int]]:
    """Cut ``⟦…⟧`` mask-token intervals out of candidate spans.

    The pipeline masks code → cite → math → soup, so by soup time the text
    already carries tokens whose checksum cores ("_", uppercase, digits)
    qualify as math runs. Masking them again nests tokens (⟦⟦SOUP…⟧⟧) and
    doubles the loss surface when the LLM normalizes brackets.
    """
    protected = [(m.start(), m.end()) for m in _TOKEN_SPAN_RE.finditer(text or "")]
    if not protected:
        return spans
    out: list[tuple[int, int]] = []
    for s, e in spans:
        fragments = [(s, e)]
        for ps, pe in protected:
            nxt: list[tuple[int, int]] = []
            for fs, fe in fragments:
                if pe <= fs or ps >= fe:
                    nxt.append((fs, fe))
                    continue
                if ps > fs:
                    nxt.append((fs, ps))
                if pe < fe:
                    nxt.append((pe, fe))
            fragments = nxt
        out.extend(fragments)
    # Subtraction can leave single-char slivers; masking those would litter
    # placeholder tokens (runs already require len >= 2 via _qualifies).
    return [(fs, fe) for fs, fe in out if fe - fs >= 2]


def find_soup_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) spans of unicode-soup math and physical parameter assignments inside prose."""
    raw_spans = [(s, e) for s, e in _run_spans(text or "") if _qualifies(text[s:e])]
    if text:
        for m in _PARAM_ASSIGN_RE.finditer(text):
            raw_spans.append((m.start(), m.end()))
    raw_spans = _subtract_token_spans(raw_spans, text)

    cleaned: list[tuple[int, int]] = []
    for s, e in raw_spans:
        while e > s and text[e - 1] in ",.;:!?":
            e -= 1
        if e > s:
            cleaned.append((s, e))

    if not cleaned:
        return []

    cleaned.sort(key=lambda sp: (sp[0], -(sp[1] - sp[0])))
    merged: list[tuple[int, int]] = []
    for s, e in cleaned:
        if not merged:
            merged.append((s, e))
        else:
            last_s, last_e = merged[-1]
            if s >= last_e:
                merged.append((s, e))
            elif e > last_e:
                merged[-1] = (last_s, e)
    return merged


class SoupMathMasker(MathMasker):
    """Masks delimiter-free unicode math; restores via MathMasker machinery."""

    def __init__(self) -> None:
        super().__init__(mask_prefix="\u27e6SOUP_MASK_")

    def mask(self, text: str) -> tuple[str, dict[str, str]]:
        """Replace soup spans with checksum-bound tokens; (masked, mapping)."""
        from ubt.core.cleaners.mask_tokens import token_checksum as _token_checksum

        mapping: dict[str, str] = {}
        if not text:
            return text, mapping
        out: list[str] = []
        cursor = 0
        for index, (start, end) in enumerate(find_soup_spans(text), 1):
            original = text[start:end]
            token = f"{self.mask_prefix}{index:04d}-{_token_checksum(index, original)}\u27e7"
            mapping[token] = original
            out.append(text[cursor:start])
            out.append(token)
            cursor = end
        out.append(text[cursor:])
        return "".join(out), mapping
