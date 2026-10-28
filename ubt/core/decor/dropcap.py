"""Drop cap initial extraction and multi-target rendering for Typst and HTML."""

from __future__ import annotations

import html
from typing import Final

from ubt.core.decor.models import DropCapConfig

_OPENING_QUOTES: Final[frozenset[str]] = frozenset(
    {"“", "‘", '"', "'", "「", "『", "《", "（", "("}
)

TYPST_DROPCAP_MACRO: Final[str] = """// --- UBT Publication Drop Cap Macro ---
#let ubt-dropcap(letter, prefix: "", lines: 2, fill: rgb("#1e3a8a")) = {
  box(
    baseline: 0%,
    inset: (right: 4pt, top: 0pt, bottom: 0pt),
    outset: 0pt,
    [
      #if prefix != "" {
        text(size: 1.15em, weight: "bold", fill: fill)[#prefix]
      }
      #text(size: lines * 1.35em, weight: "bold", fill: fill)[#letter]
    ]
  )
}
"""

HTML_DROPCAP_CSS: Final[str] = """/* --- UBT Publication Drop Cap Styler --- */
.ubt-dropcap-wrap {
  float: left;
  line-height: 0.82;
  margin-right: 0.25em;
  margin-top: 0.08em;
}
.ubt-dropcap-prefix {
  font-size: 1.4em;
  font-weight: bold;
  vertical-align: top;
  color: var(--dropcap-color, #1e3a8a);
}
.ubt-dropcap-letter {
  font-size: 3.2em;
  font-weight: bold;
  font-family: serif;
  color: var(--dropcap-color, #1e3a8a);
}
"""


def extract_dropcap_initial(text: str) -> tuple[str, str, str]:
    """Extract leading opening quotation mark (if any), initial letter, and remaining text."""
    clean = text.lstrip()
    if not clean:
        return "", "", ""

    prefix = ""
    start_idx = 0

    if clean[0] in _OPENING_QUOTES:
        prefix = clean[0]
        start_idx = 1
        while start_idx < len(clean) and clean[start_idx].isspace():
            start_idx += 1

    if start_idx >= len(clean):
        return "", clean[0], clean[1:]

    initial_letter = clean[start_idx]
    remainder = clean[start_idx + 1 :]
    return prefix, initial_letter, remainder


def apply_dropcap_typst(text: str, config: DropCapConfig | None = None) -> str:
    """Wrap opening paragraph initial character in a Typst `#ubt-dropcap` call."""
    cfg = config or DropCapConfig()
    if not cfg.enabled or not text.strip():
        return text

    # Do not apply to headings, code, equations, or list markers
    trimmed = text.lstrip()
    if trimmed.startswith(("#", "`", "$", "-", "*", ">", "|")):
        return text

    prefix, letter, remainder = extract_dropcap_initial(text)
    if not letter:
        return text

    # Escape Typst special syntax inside the letter and prefix
    safe_prefix = prefix.replace("[", r"\[").replace("]", r"\]").replace('"', r"\"")
    safe_letter = letter.replace("[", r"\[").replace("]", r"\]").replace('"', r"\"")

    return f'#ubt-dropcap("{safe_letter}", prefix: "{safe_prefix}", lines: {cfg.lines}, fill: rgb("{cfg.color}")){remainder}'


def apply_dropcap_html(text: str, config: DropCapConfig | None = None) -> str:
    """Wrap opening paragraph initial character in modern HTML drop cap markup."""
    cfg = config or DropCapConfig()
    if not cfg.enabled or not text.strip():
        return text

    trimmed = text.lstrip()
    if trimmed.startswith(("#", "`", "$", "-", "*", ">", "|")):
        return text

    prefix, letter, remainder = extract_dropcap_initial(text)
    if not letter:
        return text

    esc_prefix = html.escape(prefix)
    esc_letter = html.escape(letter)
    esc_color = html.escape(cfg.color)

    prefix_span = f'<span class="ubt-dropcap-prefix">{esc_prefix}</span>' if esc_prefix else ""
    return (
        f'<span class="ubt-dropcap-wrap" style="--dropcap-color: {esc_color};">'
        f'{prefix_span}<span class="ubt-dropcap-letter">{esc_letter}</span></span>{remainder}'
    )
