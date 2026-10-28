"""Publication-grade chapter epigraph and abstract formatter."""

from __future__ import annotations

from ubt.core.epigraph.detector import (
    detect_chapter_epigraphs_and_abstracts,
    parse_abstract_text,
    parse_epigraph_text,
)
from ubt.core.epigraph.generator import (
    render_abstract_html,
    render_abstract_typst,
    render_epigraph_html,
    render_epigraph_typst,
)
from ubt.core.epigraph.models import (
    ChapterAbstract,
    Epigraph,
    EpigraphConfig,
    EpigraphStyle,
)

__all__ = [
    "ChapterAbstract",
    "Epigraph",
    "EpigraphConfig",
    "EpigraphStyle",
    "detect_chapter_epigraphs_and_abstracts",
    "parse_abstract_text",
    "parse_epigraph_text",
    "render_abstract_html",
    "render_abstract_typst",
    "render_epigraph_html",
    "render_epigraph_typst",
]
