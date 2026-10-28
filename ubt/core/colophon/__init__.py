"""Publication colophon, CIP cataloging, and copyright metadata package."""

from __future__ import annotations

from ubt.core.colophon.generator import (
    extract_colophon_from_blocks,
    render_colophon_html,
    render_colophon_markdown,
    render_colophon_typst,
)
from ubt.core.colophon.isbn import (
    calculate_isbn10_check_digit,
    calculate_isbn13_check_digit,
    clean_isbn,
    format_isbn,
    validate_isbn,
    validate_isbn10,
    validate_isbn13,
)
from ubt.core.colophon.models import ColophonMetadata

__all__ = [
    "ColophonMetadata",
    "calculate_isbn10_check_digit",
    "calculate_isbn13_check_digit",
    "clean_isbn",
    "extract_colophon_from_blocks",
    "format_isbn",
    "render_colophon_html",
    "render_colophon_markdown",
    "render_colophon_typst",
    "validate_isbn",
    "validate_isbn10",
    "validate_isbn13",
]
