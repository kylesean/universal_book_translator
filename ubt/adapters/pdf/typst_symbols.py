"""LaTeX-command → Typst-symbol vocabulary (adapter import path).

The canonical tables live in :mod:`ubt.core.math_symbols`, so the QE's
renderability whitelist (``ubt.core.validators.math_guard``) and the two
converters read one source and cannot drift. This module keeps the historical
adapter import path for the display (``typst_math``) and inline
(``overlay_text``) converters; see the core module for the table layout.
"""

from __future__ import annotations

from ubt.core.math_symbols import (
    DISPLAY_SYMBOLS,
    DISPLAY_VALUE_OVERRIDES,
    INLINE_RENDERABLE_COMMANDS,
    INLINE_SYMBOLS,
    INLINE_VALUE_OVERRIDES,
    SPECIAL_LATEX_COMMANDS,
)

__all__ = [
    "DISPLAY_SYMBOLS",
    "DISPLAY_VALUE_OVERRIDES",
    "INLINE_RENDERABLE_COMMANDS",
    "INLINE_SYMBOLS",
    "INLINE_VALUE_OVERRIDES",
    "SPECIAL_LATEX_COMMANDS",
]
