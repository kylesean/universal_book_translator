"""CJK width metrics (fontTools advance sums, conservative vs Typst).

The width authority for overlay fitting: an unshaped advance-sum width that
stays conservative against Typst's shaped output (kerning and punctuation
compression only ever shrink it). :func:`resolve_cjk_ttc` is the piece the
runtime doctor check uses to resolve a CJK font.

Unshaped advance-sum width. Conservative vs Typst's shaped output
(kerning/punctuation compression only ever shrink it) *given* the rigid
overlay disables CJK-Latin auto-spacing: with Typst's default spacing,
mixed-script lines lay out ~0.25em wider per script boundary and the
fitter's fit is worthless.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
from pathlib import Path
from typing import Any

from ubt.core.exceptions import DocumentParseError
from ubt.core.policy.layout_policy import CJK_FONT_CANDIDATES

logger = logging.getLogger(__name__)


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


WIDTH_FONT_FALLBACK = "Noto Sans CJK SC"


def _face_name(font: Any) -> str:
    return str(font["name"].getDebugName(4) or "")


def load_width_font(ttc_path: str, family: str | None = None) -> Any:
    """Pick the face whose advance widths the fitter should measure with.

    ``family`` is the font the page will actually render in, so measuring it
    keeps the fitted size honest; then the historical
    :data:`WIDTH_FONT_FALLBACK`; then a normalized substring of either (packaged
    names differ -- ``NotoSerifCJKsc-Regular`` vs ``Noto Serif CJK SC``); then
    the first face in the collection.

    The old version raised unless the exact string ``Noto Sans CJK SC`` was
    present, so pointing ``UBT_CJK_FONT`` at a Serif collection -- one of the
    paths ``resolve_cjk_ttc`` itself recommends, and one ``ubt doctor`` reports
    as OK -- aborted the render over a *measurement* font.
    """
    from fontTools.ttLib import TTCollection

    fonts = list(TTCollection(ttc_path).fonts)
    if not fonts:
        raise DocumentParseError(f"No font face inside {ttc_path}")
    wanted = [name for name in (family, WIDTH_FONT_FALLBACK) if name]
    for name in wanted:
        for font in fonts:
            if _face_name(font) == name:
                return font

    def folded(name: str) -> str:
        return name.replace(" ", "").replace("-", "").lower()

    for name in wanted:
        target = folded(name)
        for font in fonts:
            if target and target in folded(_face_name(font)):
                return font
    logger.debug(
        "width metrics: no %s-like face in %s, measuring %s",
        "/".join(wanted) or "named",
        ttc_path,
        _face_name(fonts[0]) or "first face",
    )
    return fonts[0]


def _cached_cmap(font: Any) -> dict[int, str]:
    """Best cmap for ``font``, built once per loaded font object.

    The rigid fitter measures thousands of lines against the same TTFont;
    rebuilding the cmap dict on every call dominated the fit. Caching on the
    font object keeps the existing public signature and lives/dies with it.
    """
    cached: dict[int, str] | None = getattr(font, "_ubt_cmap_cache", None)
    if cached is not None:
        return cached
    cmap: dict[int, str] = dict(font.getBestCmap())
    with contextlib.suppress(AttributeError, TypeError):
        font._ubt_cmap_cache = cmap
    return cmap


_MATH_SPAN_WIDTH_RE = re.compile(r"\$([^$]+)\$")
_MATH_WRAPPER_WIDTH_RE = re.compile(r"\\(?:text|mathrm|mathbb|mathfrak|mathcal)\{([^{}]*)\}")
_MATH_CMD_WIDTH_RE = re.compile(r"\\[a-zA-Z]+")


def _visual_math_text(text: str) -> str:
    """Collapse LaTeX control sequences inside ``$...$`` to single-glyph width equivalents."""
    if "$" not in text:
        return text

    def _shrink(m: re.Match[str]) -> str:
        inner = m.group(1)
        inner = _MATH_WRAPPER_WIDTH_RE.sub(r"\1", inner)
        inner = _MATH_CMD_WIDTH_RE.sub("M", inner)
        return inner.replace("{", "").replace("}", "").replace("_", "").replace("^", "")

    return _MATH_SPAN_WIDTH_RE.sub(_shrink, text)


def text_width_pt(font: Any, text: str, size_pt: float) -> float:
    upm = int(font["head"].unitsPerEm)
    cmap = _cached_cmap(font)
    hmtx = font["hmtx"]
    notdef = int(hmtx[".notdef"][0])
    total = 0
    for ch in _visual_math_text(text):
        glyph = cmap.get(ord(ch), ".notdef")
        try:
            total += int(hmtx[glyph][0])
        except KeyError:
            total += notdef
    return total / upm * size_pt


_FONT_FAMILY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._-]*")


def sanitize_font_family(value: str | None) -> str | None:
    """Return a font family usable in ``#set text(font: ...)`` or ``None``."""
    clean = (value or "").strip()
    return clean if _FONT_FAMILY_RE.fullmatch(clean) else None


__all__ = ["load_width_font", "resolve_cjk_ttc", "sanitize_font_family", "text_width_pt"]
