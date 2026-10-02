"""``ubt.layout`` -- theme resolution and element-level fidelity assignment (layout and theme presentation layer).

``theme`` is the single owner of the presentation policy the sink stages read:
the language-resolved font stack, base size, leading, bilingual pairing style,
and text direction. It *composes* the existing owners (``language_profile``'s
per-language fonts, ``font_probe``'s availability, ``typst_constants``' style
profile, the ``font_family`` override) rather than restating them, so the
"one attribute, one source" rule holds.
"""

from __future__ import annotations

from ubt.layout.theme import BilingualStyle, Direction, Theme, direction_for, resolve_theme

__all__ = ["BilingualStyle", "Direction", "Theme", "direction_for", "resolve_theme"]
