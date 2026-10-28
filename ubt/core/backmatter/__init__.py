"""Appendix and backmatter layout management for professional book publication."""

from __future__ import annotations

from ubt.core.backmatter.detector import (
    classify_heading,
    detect_backmatter_manifest,
)
from ubt.core.backmatter.generator import (
    emit_appendix_switch_typst,
    render_backmatter_heading_typst,
    render_backmatter_section_html,
    render_backmatter_skeleton_markdown,
)
from ubt.core.backmatter.models import (
    BackmatterConfig,
    BackmatterKind,
    BackmatterManifest,
    BackmatterSection,
)

__all__ = [
    "BackmatterConfig",
    "BackmatterKind",
    "BackmatterManifest",
    "BackmatterSection",
    "classify_heading",
    "detect_backmatter_manifest",
    "emit_appendix_switch_typst",
    "render_backmatter_heading_typst",
    "render_backmatter_section_html",
    "render_backmatter_skeleton_markdown",
]
