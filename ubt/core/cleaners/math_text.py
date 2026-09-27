"""C-track: translatable natural-language spans inside math (``\\text{…}``).

BabelDOC freezes everything inside formulas (``Z(\\text{Hit Rate})`` stays
English forever). UBT holds the LaTeX source, so natural-language spans
can go through the LLM while the math skeleton stays byte-identical.

Conservative by design (calibration: KV handbook has exactly 3 spans —
``softmax``/``GiB`` kept, ``by terms`` translated):

- single tokens never translate (``softmax``, ``GiB``, ``ReLU`` are terms,
  units or symbols — translating them is how C-track bulk-mistranslates);
- nested math/commands, digits and overlong spans never translate;
- only multi-word plain phrases translate (``by terms``, ``Hit Rate``).

The C-era invariant: ``skeleton(source) == skeleton(target)`` — the
formula with every span body blanked must match exactly, i.e. translation
may change span *contents* and nothing else. Any violation fails closed
to source-verbatim (Gate 3 holds).
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass


def get_math_text_system_prompt(source_lang: str = "en", target_lang: str = "zh") -> str:
    """Generate system prompt parameterized by source and target language."""
    from ubt.core.language_profile import PROFILES

    src_prof = PROFILES.get((source_lang or "").strip().lower())
    tgt_prof = PROFILES.get((target_lang or "").strip().lower())
    src_label = src_prof.name if src_prof else (source_lang.title() or "English")
    tgt_label = tgt_prof.name if tgt_prof else (target_lang.title() or "Chinese")
    return (
        f"Translate the following short mathematical label from {src_label} to {tgt_label}. "
        "Return ONLY the translation, no quotes, no explanation."
    )


_COMMAND_RE = re.compile(r"\\(text|mathrm|operatorname|textrm|mbox)\s*\{")
_FORBIDDEN_INNER_RE = re.compile(r"[\\{}$^_&%#]")
_MAX_INNER_LEN = 120


@dataclass(frozen=True)
class TextSpan:
    """One ``\\cmd{inner}`` occurrence (offsets into the source formula)."""

    cmd: str
    inner: str
    start: int
    end: int


def _read_braced(s: str, open_idx: int) -> tuple[str, int] | None:
    """Read ``{…}`` starting at ``open_idx``; None on unbalanced input."""
    depth = 0
    for i in range(open_idx, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[open_idx + 1 : i], i + 1
    return None


def extract_text_spans(latex: str) -> list[TextSpan]:
    """All ``\\text{…}``-family spans in reading order (balanced braces)."""
    spans: list[TextSpan] = []
    last_end = 0
    for m in _COMMAND_RE.finditer(latex):
        if m.start() < last_end:
            continue
        parsed = _read_braced(latex, m.end() - 1)
        if parsed is None:
            continue
        inner, end = parsed
        spans.append(TextSpan(cmd=m.group(1), inner=inner, start=m.start(), end=end))
        last_end = end
    return spans


def is_translatable_text(inner: str) -> bool:
    """True only for multi-word plain phrases (conservative, see module doc)."""
    s = inner.strip()
    if not s or len(s) > _MAX_INNER_LEN:
        return False
    if _FORBIDDEN_INNER_RE.search(s):
        return False
    if not re.search(r"\s", s):
        return False
    # Digits are Gate-4 territory ("Figure 1", units) — never translate.
    return not re.search(r"\d", s)


def translatable_spans(latex: str) -> list[TextSpan]:
    """Spans of ``latex`` that should go through the LLM."""
    return [sp for sp in extract_text_spans(latex) if is_translatable_text(sp.inner)]


def skeleton(latex: str) -> str:
    """Formula with every span body blanked — the C-era invariant key."""
    spans = extract_text_spans(latex)
    if not spans:
        return latex
    parts: list[str] = []
    cursor = 0
    for i, sp in enumerate(spans):
        parts.append(latex[cursor : sp.start])
        parts.append(f"\x00SPAN{i}\x00")
        cursor = sp.end
    parts.append(latex[cursor:])
    return "".join(parts)


def skeleton_holds(source: str, target: str) -> bool:
    """True when translation changed span contents and nothing else."""
    return skeleton(source) == skeleton(target)


def reassemble(source: str, translations: dict[int, str]) -> str:
    """Splice translated inners back into a copy of ``source``."""
    spans = extract_text_spans(source)
    parts: list[str] = []
    cursor = 0
    for i, sp in enumerate(spans):
        parts.append(source[cursor : sp.start])
        new_inner = translations.get(i, sp.inner)
        parts.append(f"\\{sp.cmd}{{{new_inner}}}")
        cursor = sp.end
    parts.append(source[cursor:])
    return "".join(parts)


async def translate_math_text(
    source: str,
    target_lang: str,
    complete: Callable[[str, str], Awaitable[str]],
) -> str | None:
    """Translate a formula's text spans; None = nothing to do or fail-closed.

    ``complete(inner, target_lang)`` performs one LLM call per span.
    Returns the reassembled formula on skeleton-invariant success,
    otherwise None (caller keeps source verbatim — Gate 3 holds).
    """
    all_spans = extract_text_spans(source)
    spans = [sp for sp in all_spans if is_translatable_text(sp.inner)]
    if not spans:
        return None
    wanted = {(sp.start, sp.end) for sp in spans}
    translations: dict[int, str] = {}
    for i, sp in enumerate(all_spans):
        if (sp.start, sp.end) not in wanted:
            continue
        try:
            rendered = (await complete(sp.inner.strip(), target_lang)).strip()
        except Exception:
            return None
        if not rendered or "\\" in rendered or "{" in rendered or "}" in rendered:
            return None  # model returned commands — fail closed, not escaped
        translations[i] = rendered
    candidate = reassemble(source, translations)
    if not skeleton_holds(source, candidate):
        return None
    if candidate == source:
        return None
    return candidate
