"""Publication-grade chapter header decoration and styling for Typst and HTML."""

from __future__ import annotations

import html
import re
from typing import Final

from ubt.core.decor.models import ChapterHeaderConfig, HeaderStyle

_CHAPTER_LABEL_RE: Final[re.Pattern[str]] = re.compile(
    r"^(第\s*[0-9一二三四五六七八九十百零]+\s*[章回节篇卷部]|Chapter\s*[0-9IVXLCDM]+|Section\s*[0-9IVXLCDM]+|[0-9]+(?:\.[0-9]+)*\.?)[：:\s]+",
    re.IGNORECASE,
)


def parse_chapter_heading(text: str) -> tuple[str, str, str]:
    """Parse heading text into (chapter_label, title, subtitle)."""
    clean = text.strip()
    if not clean:
        return "", "", ""

    m = _CHAPTER_LABEL_RE.match(clean)
    if m:
        label = m.group(1).strip()
        rest = clean[m.end() :].strip()
    else:
        label = ""
        rest = clean

    # Check for subtitle delimiters
    if " -- " in rest:
        title, subtitle = rest.split(" -- ", 1)
    elif " —— " in rest:
        title, subtitle = rest.split(" —— ", 1)
    elif " — " in rest:
        title, subtitle = rest.split(" — ", 1)
    elif "：" in rest and not label:
        title, subtitle = rest.split("：", 1)
    elif ": " in rest and not label:
        title, subtitle = rest.split(": ", 1)
    else:
        title, subtitle = rest, ""

    return label.strip(), title.strip(), subtitle.strip()


def render_chapter_header_typst(
    heading_text: str, config: ChapterHeaderConfig | None = None
) -> str:
    """Render a publication decorative chapter opening header in Typst."""
    cfg = config or ChapterHeaderConfig()

    # Escape Typst markup in strings
    def esc(s: str) -> str:
        return s.replace("[", r"\[").replace("]", r"\]").replace("#", r"\#")

    if not cfg.enabled:
        return f"#heading(level: 1)[{esc(heading_text)}]"

    label, title, subtitle = parse_chapter_heading(heading_text)

    e_label = esc(label)
    e_title = esc(title)
    e_sub = esc(subtitle)

    top_v = f"{cfg.top_spacing_cm}cm"
    bot_v = f"{cfg.bottom_spacing_cm}cm"

    sub_typst = (
        f'\n    #v(0.3em)\n    #text(size: 1.05em, style: "italic", fill: rgb("{cfg.secondary_color}"))[{e_sub}]'
        if e_sub
        else ""
    )

    if cfg.style == HeaderStyle.CLASSIC:
        label_line = (
            f'#text(size: 1.05em, weight: "bold", tracking: 0.25em, fill: rgb("{cfg.secondary_color}"))[{e_label}]\n    #v(0.35em)\n    '
            if e_label
            else ""
        )
        fleuron_markup = (
            f'\n    #v(0.25em)\n    #text(size: 0.9em, fill: rgb("{cfg.accent_color}"))[{cfg.fleuron_char}]'
            if cfg.show_fleuron
            else ""
        )
        return f"""#v({top_v})
#align(center)[
    {label_line}#text(size: 2.0em, weight: "bold", fill: rgb("#0f172a"))[{e_title}]{sub_typst}
    #v(0.65em)
    #line(length: 35%, stroke: 1.2pt + rgb("{cfg.accent_color}")){fleuron_markup}
    #v(0.25em)
    #line(length: 22%, stroke: 0.6pt + rgb("{cfg.secondary_color}"))
]
#v({bot_v})"""

    if cfg.style == HeaderStyle.MODERN:
        label_line = (
            f'#text(size: 0.95em, weight: "bold", tracking: 0.15em, fill: rgb("{cfg.accent_color}"))[{e_label}]\n    #v(0.2em)\n    '
            if e_label
            else ""
        )
        return f"""#v({top_v})
#block(width: 100%, inset: (left: 14pt), stroke: (left: 4.5pt + rgb("{cfg.accent_color}")))[
    {label_line}#text(size: 1.95em, weight: "bold", fill: rgb("#0f172a"))[{e_title}]{sub_typst}
]
#v({bot_v})"""

    if cfg.style == HeaderStyle.ACADEMIC:
        label_line = (
            f'#text(size: 1.1em, weight: "bold", fill: rgb("{cfg.secondary_color}"))[{e_label}]\n    #v(0.2em)\n    '
            if e_label
            else ""
        )
        return f"""#v({top_v})
#block(width: 100%)[
    {label_line}#text(size: 1.85em, weight: "bold", fill: rgb("#0f172a"))[{e_title}]{sub_typst}
    #v(0.4em)
    #line(length: 100%, stroke: 0.75pt + rgb("{cfg.secondary_color}"))
]
#v({bot_v})"""

    # Minimal
    label_line = (
        f'#text(size: 0.9em, tracking: 0.2em, fill: rgb("{cfg.secondary_color}"))[{e_label}]\n    #v(0.3em)\n    '
        if e_label
        else ""
    )
    return f"""#v({top_v})
#block(width: 100%)[
    {label_line}#text(size: 1.75em, weight: "medium", fill: rgb("#0f172a"))[{e_title}]{sub_typst}
]
#v({bot_v})"""


def render_chapter_header_html(heading_text: str, config: ChapterHeaderConfig | None = None) -> str:
    """Render a publication decorative chapter opening header in HTML."""
    cfg = config or ChapterHeaderConfig()
    if not cfg.enabled:
        return f"<h1>{html.escape(heading_text)}</h1>"
    label, title, subtitle = parse_chapter_heading(heading_text)

    e_label = html.escape(label)
    e_title = html.escape(title)
    e_sub = html.escape(subtitle)
    style_class = f"ubt-header-{cfg.style.value}"

    label_html = f'<div class="ubt-chapter-label">{e_label}</div>' if e_label else ""
    sub_html = f'<div class="ubt-chapter-subtitle">{e_sub}</div>' if e_sub else ""
    fleuron_html = (
        f'<div class="ubt-chapter-ornament"><span class="ubt-fleuron">{html.escape(cfg.fleuron_char)}</span></div>'
        if cfg.show_fleuron and cfg.style == HeaderStyle.CLASSIC
        else ""
    )

    return f"""<header class="ubt-chapter-header {style_class}">
  {label_html}
  <h1 class="ubt-chapter-title">{e_title}</h1>
  {sub_html}
  {fleuron_html}
</header>"""
