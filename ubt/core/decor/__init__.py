"""Publication chapter header decorations and paragraph initial drop caps."""

from __future__ import annotations

from ubt.core.decor.dropcap import (
    HTML_DROPCAP_CSS,
    TYPST_DROPCAP_MACRO,
    apply_dropcap_html,
    apply_dropcap_typst,
    extract_dropcap_initial,
)
from ubt.core.decor.header import (
    parse_chapter_heading,
    render_chapter_header_html,
    render_chapter_header_typst,
)
from ubt.core.decor.models import (
    ChapterHeaderConfig,
    DecorSpec,
    DropCapConfig,
    HeaderStyle,
)

__all__ = [
    "HTML_DROPCAP_CSS",
    "TYPST_DROPCAP_MACRO",
    "ChapterHeaderConfig",
    "DecorSpec",
    "DropCapConfig",
    "HeaderStyle",
    "apply_dropcap_html",
    "apply_dropcap_typst",
    "extract_dropcap_initial",
    "parse_chapter_heading",
    "render_chapter_header_html",
    "render_chapter_header_typst",
]
