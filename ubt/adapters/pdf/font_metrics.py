"""Font helpers the overlay path actually uses.

``resolve_cjk_ttc`` is the piece the runtime doctor check uses to resolve a CJK
font; ``sanitize_font_family`` filters a family name for Typst's ``#set text``.

The width-metrics half of this module (fontTools advance sums, a width fitter)
was deleted with the rigid typesetter: overlay fitting measures through Typst
compiles instead, and the advance-sum path had no remaining caller.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from ubt.core.exceptions import DocumentParseError
from ubt.core.policy.layout_policy import CJK_FONT_CANDIDATES


def resolve_cjk_ttc() -> str:
    override = os.environ.get("UBT_CJK_FONT", "").strip()
    if override:
        return override
    for cand in CJK_FONT_CANDIDATES:
        if Path(cand).exists():
            return cand
    raise DocumentParseError(
        "No CJK font found for overlay width metrics "
        f"(tried {', '.join(CJK_FONT_CANDIDATES)}); "
        "set UBT_CJK_FONT to any CJK TrueType *collection* (.ttc) — a bare .ttf/.otf "
        "will not do, this is read with fontTools' TTCollection"
    )


_FONT_FAMILY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._-]*")


def sanitize_font_family(value: str | None) -> str | None:
    """Return a font family usable in ``#set text(font: ...)`` or ``None``."""
    clean = (value or "").strip()
    return clean if _FONT_FAMILY_RE.fullmatch(clean) else None


__all__ = ["resolve_cjk_ttc", "sanitize_font_family"]
