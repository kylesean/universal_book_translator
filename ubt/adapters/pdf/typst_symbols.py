"""LaTeX-command → Typst-symbol vocabulary (adapter import path).

The canonical table lives in :mod:`ubt.core.math_symbols`, so the QE's
renderability whitelist (``ubt.core.validators.math_guard``) and the inline
converter (``overlay_text``) read one source and cannot drift. This module keeps
the historical adapter import path; see the core module for the table layout.
"""

from __future__ import annotations

from ubt.core.math_symbols import (
    INLINE_RENDERABLE_COMMANDS,
    INLINE_SYMBOLS,
    INLINE_VALUE_OVERRIDES,
    SPECIAL_LATEX_COMMANDS,
)

__all__ = [
    "INLINE_RENDERABLE_COMMANDS",
    "INLINE_SYMBOLS",
    "INLINE_VALUE_OVERRIDES",
    "SPECIAL_LATEX_COMMANDS",
]
