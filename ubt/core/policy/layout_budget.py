"""Fixed-container character budget estimation for one-pass layout-aware drafting."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ubt.core.ir.models import StyleMeta

_INLINE_TAG_RE = re.compile(r"</?[a-z]\d+/?\s*>")
_PLACEHOLDER_TOKEN_RE = re.compile(r"⟦[^⟧]+⟧")
_CJK_CHAR_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")
_CJK_TARGET_LANGS = frozenset({"zh", "ja", "ko"})


def _plain_content_chars(text: str) -> tuple[str, int]:
    without_tags = _INLINE_TAG_RE.sub("", text or "")
    without_masks = _PLACEHOLDER_TOKEN_RE.sub("xx", without_tags)
    cleaned = " ".join(without_masks.split())
    return cleaned, len(cleaned)


def _estimate_unconstrained_target_len(source_text: str, target_lang: str) -> float:
    cleaned, length = _plain_content_chars(source_text)
    if length == 0:
        return 0.0
    cjk_src = len(_CJK_CHAR_RE.findall(cleaned))
    src_is_cjk = cjk_src / max(1, length) >= 0.25
    tgt_base = (target_lang or "").strip().lower().split("-")[0]
    tgt_is_cjk = tgt_base in _CJK_TARGET_LANGS
    if src_is_cjk and not tgt_is_cjk:
        return length * 2.1
    if not src_is_cjk and tgt_is_cjk:
        return length * 0.72
    return length * 1.15


def _box_comfortable_capacity(style: StyleMeta, target_lang: str) -> int | None:
    width_pt = getattr(style, "box_width_pt", None)
    height_pt = getattr(style, "box_height_pt", None)
    font_size = getattr(style, "font_size", None)
    if not width_pt or not height_pt or not font_size:
        return None
    if width_pt <= 0 or height_pt <= 0 or font_size <= 0:
        return None

    # Model a gentle 0.90 scale (well above the 0.68-0.70 floor) so the draft
    # targets natural legibility rather than cramming at minimum size.
    eff_size = max(5.5, float(font_size) * 0.90)
    raw_leading = getattr(style, "line_height_pt", None)
    base_leading = (
        float(raw_leading) if raw_leading and raw_leading > 0 else float(font_size) * 1.25
    )
    eff_leading = max(eff_size * 1.15, base_leading * 0.90)

    tgt_base = (target_lang or "").strip().lower().split("-")[0]
    char_advance = eff_size * (0.95 if tgt_base in _CJK_TARGET_LANGS else 0.52)
    chars_per_line = max(1, int(float(width_pt) / max(1.0, char_advance)))
    usable_h = max(eff_size * 1.12, float(height_pt) + 1.5)
    lines_fit = max(1, int((usable_h - eff_size * 1.12) // eff_leading) + 1)
    return max(4, chars_per_line * lines_fit)


def estimate_tight_char_budget(
    style: StyleMeta | None,
    source_text: str,
    target_lang: str,
    *,
    tightness_threshold: float = 0.85,
) -> int | None:
    """Return a target character budget only when a fixed source box is tight.

    Roomy paragraphs and reflow documents return ``None`` so unconstrained prose
    keeps identical prompts and cache keys.
    """
    if style is None:
        return None
    explicit = getattr(style, "char_budget", None)
    if explicit is not None and explicit > 0:
        return int(explicit)

    _, plain_len = _plain_content_chars(source_text)
    if plain_len < 12:
        return None

    capacity = _box_comfortable_capacity(style, target_lang)
    if capacity is None:
        return None

    expected = _estimate_unconstrained_target_len(source_text, target_lang)
    if expected < capacity * tightness_threshold:
        return None

    return max(8, int(capacity * 0.95))


__all__ = ["estimate_tight_char_budget"]
