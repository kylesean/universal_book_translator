"""ContainerFitRepair: Adaptive container fitting and code/token boundary repair for Typst.

Prevents horizontal layout spills and container clipping in technical books and papers:
1. Adaptive Monospace Code Scaling: Computes max line widths (accounting for 2.0x CJK characters),
   dynamically adjusting monospace font size from 8.5pt down to 7.0pt.
2. Smart Token-Aware Code Wrapping: Wraps lines exceeding container width at operator,
   comma, or bracket boundaries with hanging indent.
3. Soft Break Injection for Unbroken Tokens: Injects zero-width break opportunities (\\u200b)
   into long URLs, file paths, and identifiers to prevent right-margin prose spills.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence

from pydantic import BaseModel, ConfigDict, Field


def effective_char_width(char: str) -> float:
    """Return effective width unit for character (1.0 for ASCII, 2.0 for CJK/full-width)."""
    code = ord(char)
    # CJK Unified Ideographs, Hiragana, Katakana, Hangul, Fullwidth Forms
    if (
        (0x4E00 <= code <= 0x9FFF)
        or (0x3400 <= code <= 0x4DBF)
        or (0x3040 <= code <= 0x30FF)
        or (0xAC00 <= code <= 0xD7AF)
        or (0xFF01 <= code <= 0xFF60)
        or (0xFFE0 <= code <= 0xFFE6)
    ):
        return 2.0
    return 1.0


def measure_line_width(line: str) -> float:
    """Calculate effective character units for a line of code."""
    # Expand tabs to 4 spaces for consistent measurement
    expanded = line.expandtabs(4)
    return sum(effective_char_width(c) for c in expanded)


def smart_wrap_code_line(
    line: str,
    max_units: int,
    hanging_indent: str = "    ",
) -> list[str]:
    """Wrap a long code line at natural syntactic token boundaries with hanging indent."""
    expanded = line.expandtabs(4)
    if measure_line_width(expanded) <= max_units or not expanded.strip():
        return [expanded]

    m_indent = re.match(r"^(\s*)", expanded)
    orig_indent = m_indent.group(1) if m_indent else ""
    cont_indent = orig_indent + hanging_indent

    content = expanded[len(orig_indent) :].strip()
    current_prefix = orig_indent
    sub_lines: list[str] = []

    while content:
        avail = max(15, max_units - int(measure_line_width(current_prefix)))
        if measure_line_width(content) <= avail:
            sub_lines.append(current_prefix + content)
            break

        # Search for best break in content within avail
        best_break = -1
        curr_u = 0.0
        for i, ch in enumerate(content):
            w = effective_char_width(ch)
            if curr_u + w > avail:
                break
            curr_u += w
            if ch in (",", ";") and i + 1 < len(content) and content[i + 1] == " ":
                best_break = i + 2
            elif ch in ("(", "[", "{") or ch == " " and i > 0:
                best_break = i + 1
            elif ch == "." and i > 0 and content[i - 1].isalnum():
                best_break = i

        if best_break <= 0:
            # Hard break at character limit
            cut = 0
            u = 0.0
            for i, ch in enumerate(content):
                u += effective_char_width(ch)
                if u > avail:
                    cut = max(1, i)
                    break
            if cut == 0:
                cut = len(content)
            best_break = cut

        sub_lines.append(current_prefix + content[:best_break].rstrip())
        content = content[best_break:].lstrip()
        current_prefix = cont_indent

    return sub_lines if sub_lines else [expanded]


def fit_code_block(
    code_text: str,
    container_width_pt: float,
    *,
    base_font_size_pt: float = 8.5,
    min_font_size_pt: float = 7.0,
    inset_pt: float = 16.0,
) -> tuple[str, float]:
    """Adaptively scale monospace font size and wrap long lines to fit container width.

    Returns:
        (fitted_code_text, chosen_font_size_pt)
    """
    lines = code_text.splitlines()
    if not lines:
        return code_text, base_font_size_pt

    net_width = max(80.0, container_width_pt - inset_pt)
    max_line_units = max(measure_line_width(ln) for ln in lines)

    # Unit width coefficient for monospace font (~0.58 pt per unit at 1.0pt font size)
    unit_coeff = 0.58

    # Phase 1: Adaptive font size scaling
    needed_font_size = net_width / max(1.0, max_line_units * unit_coeff)

    if needed_font_size >= base_font_size_pt:
        # Fits comfortably at base size
        chosen_font_size = base_font_size_pt
        wrapped_lines = lines
    elif needed_font_size >= min_font_size_pt:
        # Scale font down to avoid line breaking
        chosen_font_size = round(needed_font_size, 2)
        wrapped_lines = lines
    else:
        # Phase 2: At minimum font size (7.0pt), lines still exceed container -> wrap them
        chosen_font_size = min_font_size_pt
        max_units_at_min = max(20, math.floor(net_width / (unit_coeff * min_font_size_pt)))
        wrapped_lines = []
        for ln in lines:
            wrapped_lines.extend(smart_wrap_code_line(ln, max_units=max_units_at_min))

    return "\n".join(wrapped_lines), chosen_font_size


def repair_code_block_markup(
    code_text: str,
    container_width_pt: float,
    *,
    base_font_size_pt: float = 8.5,
) -> str:
    """Generate container-bounded Typst code block markup with adaptive font and smart wrapping."""
    cleaned = code_text.strip()
    if not cleaned:
        return ""

    fitted_code, font_size = fit_code_block(
        cleaned,
        container_width_pt=container_width_pt,
        base_font_size_pt=base_font_size_pt,
    )

    # Typst literal string escaping
    payload = fitted_code.replace("\\", "\\\\").replace('"', '\\"')

    # If font size was scaled down from default, apply #set text(size: ...)
    size_override = (
        f"#set text(size: {font_size:.1f}pt)\n  " if font_size < base_font_size_pt else ""
    )
    return (
        f"#block(fill: luma(245), inset: 8pt, radius: 4pt, width: 100%)[\n"
        f"  {size_override}"
        f'#raw("{payload}")\n'
        f"]"
    )


def inject_soft_breaks(text: str, max_token_len: int = 32) -> str:
    """Inject zero-width break hints (\\u200b) into long unbreakable tokens in prose.

    Targets URLs, filesystem paths, and long identifiers to prevent right-margin spills.
    Does not touch LaTeX math delimiters ($...$) or Typst directives.
    """
    if len(text) < max_token_len:
        return text

    # Word boundary regex matching contiguous non-space strings >= max_token_len
    token_pattern = re.compile(r"(\S{" + str(max_token_len) + r",})")

    def _break_token(match: re.Match[str]) -> str:
        tok = match.group(1)
        # Skip tokens inside Typst math ($...$) or code
        if tok.startswith("$") or tok.startswith("#") or tok.startswith("@"):
            return tok

        # Inject \u200b after slashes, dots, underscores, hyphens, and query params
        return re.sub(r"([/_.\-?&=])", lambda m: m.group(1) + "\u200b", tok)

    return token_pattern.sub(_break_token, text)


class ContainerFitSpec(BaseModel):
    """Specification for container space capacity budget and constrained concise repair."""

    model_config = ConfigDict(frozen=True)

    seg_id: str
    current_translation: str
    max_target_chars: int
    locked_glossary_terms: list[str] = Field(default_factory=list)
    locked_numbers: list[str] = Field(default_factory=list)


def compute_container_char_budget(
    container_width_pt: float,
    max_lines: int = 1,
    floor_pt: float = 7.5,
    target_lang: str = "zh",
) -> int:
    """Compute maximum character capacity budget for a rigid container.

    Formula from docs/design/adaptive_delivery_prd.md §8.6:
    B_chars = floor(W / (alpha_lang * s_floor)) * N_max
    where alpha_zh = 1.0 (full-width ideograph), alpha_en = 0.48 (Latin average).
    """
    lang_lower = (target_lang or "zh").lower()
    if any(lang_lower.startswith(p) for p in ("zh", "ja", "ko")):
        alpha = 1.0
    elif any(lang_lower.startswith(p) for p in ("en", "de", "fr", "es", "it", "pt", "nl")):
        alpha = 0.48
    else:
        alpha = 0.60

    chars_per_line = max(1, math.floor(container_width_pt / max(1.0, alpha * floor_pt)))
    return chars_per_line * max(1, max_lines)


def _effective_glyph_len(text: str) -> int:
    """Length of text excluding zero-width/break hints and leading/trailing whitespace."""
    cleaned = text.replace("\u200b", "").replace("\ufeff", "").strip()
    return len(cleaned)


def extract_locked_numbers(text: str) -> list[str]:
    """Extract distinct numerical tokens to lock during concise rewriting."""
    raw_nums = re.findall(r"(?<![A-Za-z0-9_])\d+(?:\.\d+)?%?(?![A-Za-z0-9_])", text)
    seen: set[str] = set()
    result: list[str] = []
    for n in raw_nums:
        if n not in seen:
            seen.add(n)
            result.append(n)
    return result


def verify_concise_candidate(spec: ContainerFitSpec, candidate: str) -> bool:
    """WMT26 dual-gate validation:
    1. Character count must be <= spec.max_target_chars.
    2. Locked glossary terms (locked_glossary_terms) must not be dropped or altered.
    3. Locked numbers (locked_numbers) must not be dropped or altered.
    4. Inline math ($...$, $$...$$) and sentinels (e.g. ⟦...⟧) must be preserved intact.
    5. Candidate must not be empty.
    """
    cleaned_candidate = candidate.strip()
    if not cleaned_candidate:
        return False

    if _effective_glyph_len(cleaned_candidate) > spec.max_target_chars:
        return False

    for term in spec.locked_glossary_terms:
        if term and term not in cleaned_candidate:
            return False

    for num in spec.locked_numbers:
        if num and num not in cleaned_candidate:
            return False

    math_spans = re.findall(r"\$[^$]+\$|\$\$[^$]+\$\$", spec.current_translation)
    for m in math_spans:
        if m not in cleaned_candidate:
            return False

    sentinel_spans = re.findall(r"⟦[^⟧]+⟧|\[SENTINEL_[^\]]+\]", spec.current_translation)
    return all(s in cleaned_candidate for s in sentinel_spans)


_ZH_CONDENSATIONS: list[tuple[str, str]] = [
    ("我们可以看到，", ""),
    ("我们可以看到", ""),
    ("由此可见，", "可见，"),
    ("由于这个原因，", "因此，"),
    ("由于这个原因", "因此"),
    ("可以通过使用", "可通过"),
    ("可以通过", "可通过"),
    ("进行计算", "计算"),
    ("进行分析", "分析"),
    ("进行处理", "处理"),
    ("进行评估", "评估"),
    ("进行优化", "优化"),
    ("进行测试", "测试"),
    ("相关的内容", "相关内容"),
    ("等等", "等"),
    ("根据上述内容", "据此"),
    ("在大多数情况下", "通常"),
    ("在当前情况下", "目前"),
]

_EN_CONDENSATIONS: list[tuple[str, str]] = [
    ("in order to", "to"),
    ("as a result of", "due to"),
    ("a number of", "several"),
    ("at the present time", "currently"),
    ("is able to", "can"),
    ("has the capability to", "can"),
    ("prior to", "before"),
    ("subsequent to", "after"),
    ("for the purpose of", "for"),
    ("with reference to", "regarding"),
]


def concise_rewrite_heuristic(
    text: str,
    target_budget: int,
    *,
    locked_terms: Sequence[str] = (),
    locked_numbers: Sequence[str] = (),
    target_lang: str = "zh",
) -> str:
    """Deterministic condensing heuristic for rigid containers.

    Tries progressive substitutions of verbose connective phrases without altering
    locked terms, numbers, or inline math.
    """
    candidate = text
    lang_lower = (target_lang or "zh").lower()
    replacements = _ZH_CONDENSATIONS if lang_lower.startswith("zh") else _EN_CONDENSATIONS

    for pattern, replacement in replacements:
        if _effective_glyph_len(candidate) <= target_budget:
            break
        if pattern in candidate:
            test_cand = candidate.replace(pattern, replacement, 1)
            if all(t in test_cand for t in locked_terms if t) and all(
                n in test_cand for n in locked_numbers if n
            ):
                candidate = test_cand

    if _effective_glyph_len(candidate) > target_budget:
        condensed_ws = re.sub(r"[ \t]+", " ", candidate).strip()
        if all(t in condensed_ws for t in locked_terms if t) and all(
            n in condensed_ws for n in locked_numbers if n
        ):
            candidate = condensed_ws

    return candidate


def repair_bottleneck_containers(
    specs: Sequence[ContainerFitSpec],
    rewriter: Callable[[ContainerFitSpec], str] | None = None,
) -> dict[str, str]:
    """Attempt to repair bottleneck containers that exceed character budgets.

    Returns a mapping of seg_id -> repaired_concise_text for all successfully
    condensed and verified candidates.
    """
    repaired: dict[str, str] = {}
    for spec in specs:
        candidate: str | None = None
        if rewriter is not None:
            try:
                candidate = rewriter(spec)
            except Exception:
                candidate = None
        if candidate is None or not verify_concise_candidate(spec, candidate):
            candidate = concise_rewrite_heuristic(
                spec.current_translation,
                spec.max_target_chars,
                locked_terms=spec.locked_glossary_terms,
                locked_numbers=spec.locked_numbers,
            )
        if verify_concise_candidate(spec, candidate):
            repaired[spec.seg_id] = candidate
    return repaired


__all__ = [
    "ContainerFitSpec",
    "compute_container_char_budget",
    "concise_rewrite_heuristic",
    "effective_char_width",
    "extract_locked_numbers",
    "fit_code_block",
    "inject_soft_breaks",
    "measure_line_width",
    "repair_bottleneck_containers",
    "repair_code_block_markup",
    "smart_wrap_code_line",
    "verify_concise_candidate",
]
