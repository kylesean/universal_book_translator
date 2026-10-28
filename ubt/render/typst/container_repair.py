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


__all__ = [
    "effective_char_width",
    "fit_code_block",
    "inject_soft_breaks",
    "measure_line_width",
    "repair_code_block_markup",
    "smart_wrap_code_line",
]
