"""Publishing generator for Typst and HTML backmatter structures and numbering switches."""

from __future__ import annotations

import html

from ubt.core.backmatter.models import (
    BackmatterConfig,
    BackmatterKind,
    BackmatterSection,
)


def _esc_typst(s: str) -> str:
    return s.replace("[", r"\[").replace("]", r"\]").replace("#", r"\#")


def emit_appendix_switch_typst(config: BackmatterConfig | None = None) -> str:
    """Emit the Typst counter reset and heading numbering switch for appendices."""
    cfg = config or BackmatterConfig()
    pagebreak_line = "#pagebreak(weak: true)\n" if cfg.auto_pagebreak else ""
    reset_line = "#counter(heading).update(0)\n" if cfg.reset_heading_counter else ""
    numbering_line = f'#set heading(numbering: "{cfg.appendix_numbering}")\n'

    return (
        f"// --- UBT Backmatter Appendix Mode Switch ---\n"
        f"{pagebreak_line}"
        f"{reset_line}"
        f"{numbering_line}"
    )


def render_backmatter_heading_typst(
    section: BackmatterSection,
    config: BackmatterConfig | None = None,
    anchor_id: str | None = None,
) -> str:
    """Render a backmatter section heading in Typst with correct numbering and styling."""
    cfg = config or BackmatterConfig()
    aid = anchor_id or section.section_id
    anchor_part = f" <{aid}>" if aid else ""

    if section.kind == BackmatterKind.APPENDIX:
        clean = _esc_typst(section.clean_title)
        # If the clean title already contains "附录" or "Appendix", do not prepend
        if clean.startswith((cfg.appendix_prefix_zh, cfg.appendix_prefix_en)):
            display_title = clean
        else:
            prefix = (
                cfg.appendix_prefix_zh
                if any("\u4e00" <= c <= "\u9fff" for c in clean)
                else cfg.appendix_prefix_en
            )
            letter = section.letter_index
            sep = "：" if prefix == cfg.appendix_prefix_zh else ": "
            display_title = f"{prefix} {letter}{sep}{clean}" if letter else f"{prefix}{sep}{clean}"

        # If heading numbering is set to "A.1", Typst automatically numbers if not numbering: none.
        # But to be robust across custom styles, we emit a well-defined heading with anchor
        return f"#heading(level: 1)[{display_title}]{anchor_part}"

    # Non-appendix backmatter (Bibliography, Index, Glossary, Afterword)
    clean_title = _esc_typst(section.clean_title)
    num_opt = "numbering: none" if cfg.suppress_backmatter_numbering else ""
    args = f"level: 1{', ' + num_opt if num_opt else ''}"
    return f"#heading({args})[{clean_title}]{anchor_part}"


def render_backmatter_section_html(
    section: BackmatterSection,
    content_html: str = "",
    config: BackmatterConfig | None = None,
) -> str:
    """Render an HTML5 semantic backmatter section container with ARIA doc-roles."""
    _ = config or BackmatterConfig()
    role_map = {
        BackmatterKind.APPENDIX: "doc-appendix",
        BackmatterKind.BIBLIOGRAPHY: "doc-bibliography",
        BackmatterKind.GLOSSARY: "doc-glossary",
        BackmatterKind.INDEX: "doc-index",
        BackmatterKind.AFTERWORD: "doc-afterword",
        BackmatterKind.COLOPHON: "doc-colophon",
        BackmatterKind.OTHER_BACKMATTER: "doc-backmatter",
    }
    role = role_map.get(section.kind, "doc-backmatter")
    sec_id = html.escape(section.section_id)
    title_esc = html.escape(section.raw_title)

    return (
        f'<section class="ubt-backmatter ubt-{section.kind.value}" id="{sec_id}" role="{role}">\n'
        f'  <h1 class="ubt-backmatter-heading">{title_esc}</h1>\n'
        f"{content_html}\n"
        f"</section>"
    )


def render_backmatter_skeleton_markdown(sections: list[BackmatterSection]) -> str:
    """Render markdown outline skeleton for all detected backmatter sections."""
    lines: list[str] = []
    for s in sections:
        if s.kind == BackmatterKind.APPENDIX:
            lines.append(f"# 附录 {s.letter_index}：{s.clean_title}\n")
        else:
            lines.append(f"# {s.clean_title}\n")
    return "\n".join(lines)
