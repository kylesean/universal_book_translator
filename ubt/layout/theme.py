"""Theme: one value for a run's presentation policy (presentation policy theme).

Before this, "the theme" had five owners: ``language_profile`` (per-language
fonts and localized figure/table prefixes), ``font_probe`` (which of those fonts
the machine actually has), bilingual style profiles (the emitted bilingual
size/fill constants), ``config.font_family`` (the user override), and the
reconstructor's own size/leading defaults. A :class:`Theme` *composes* those into
one frozen value resolved once per run, so a renderer reads one object instead of
five sources -- the single source of truth rule ("one attribute, one source").

It also owns **text direction**, a concept that had no owner at all: Typst runs
the Unicode Bidirectional Algorithm on logical-order text and lays runs out from
``dir:``/``lang:``, so the theme's job is detection and emission -- not
reordering text in Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Direction(StrEnum):
    """Text direction of one language."""

    LTR = "ltr"
    RTL = "rtl"


#: Base language subtags that are written right-to-left.
_RTL_LANGUAGES = frozenset(
    {
        "ar",
        "he",
        "fa",
        "ur",
        "ps",
        "sd",
        "ug",
        "yi",
        "dv",
        "ku",
        "ckb",
        "arc",
        "syr",
        "nqo",
        "rhg",
        "adx",
    }
)

#: ISO 15924 script subtags that are written right-to-left.
_RTL_SCRIPTS = frozenset({"arab", "hebr", "thaa", "nkoo", "adlm", "rohg", "yezi", "mand", "samr"})


def direction_for(language: str | None) -> Direction:
    """The direction of a BCP-47-ish language tag. Pure.

    A script subtag wins over the base language (``ar-Latn`` is left-to-right,
    ``en-Arab`` is right-to-left); otherwise the base subtag decides. Unknown
    tags are left-to-right, the same default the rest of the pipeline assumes.
    """
    tag = (language or "").strip().lower().replace("_", "-")
    if not tag:
        return Direction.LTR
    subtags = tag.split("-")
    for part in subtags[1:]:
        if len(part) == 4 and part.isalpha():
            return Direction.RTL if part in _RTL_SCRIPTS else Direction.LTR
    return Direction.RTL if subtags[0] in _RTL_LANGUAGES else Direction.LTR


@dataclass(frozen=True, slots=True)
class BilingualStyle:
    """How source and target are distinguished in a bilingual layout."""

    source_size: str = "9.5pt"
    source_fill: str = "luma(90)"
    target_size: str = "10.5pt"
    target_fill: str = "black"


@dataclass(frozen=True, slots=True)
class Theme:
    """A run's resolved presentation policy: fonts, metrics, pairing, direction."""

    source_lang: str = "en"
    target_lang: str = "zh"
    fonts: tuple[str, ...] = ()
    font_override: str | None = None
    base_size_pt: float = 10.5
    leading_em: float = 0.85
    paper_size: str = "a4"
    figure_prefix: str = "Fig."
    table_prefix: str = "Table"
    bilingual: BilingualStyle = field(default_factory=BilingualStyle)

    @property
    def source_direction(self) -> Direction:
        return direction_for(self.source_lang)

    @property
    def target_direction(self) -> Direction:
        return direction_for(self.target_lang)

    @property
    def font_stack(self) -> tuple[str, ...]:
        """The resolved stack, the user's override first when one is set."""
        if self.font_override:
            return (self.font_override, *self.fonts)
        return self.fonts


def resolve_theme(
    source_lang: str = "en",
    target_lang: str = "zh",
    *,
    font_override: str | None = None,
    base_size_pt: float | None = None,
    leading_em: float | None = None,
    paper_size: str = "a4",
) -> Theme:
    """Compose the existing owners into one :class:`Theme`.

    The font stack and localized prefixes come from
    :func:`ubt.core.language_profile.resolve_font_config`; the metric defaults
    match the reconstructor's constructor. Nothing here duplicates a policy that
    already has an owner.
    """
    from ubt.core.language_profile import resolve_font_config

    config = resolve_font_config(target_lang)
    return Theme(
        source_lang=(source_lang or "en").strip().lower().replace("_", "-"),
        target_lang=(target_lang or "zh").strip().lower().replace("_", "-"),
        fonts=tuple(config.typst_fonts),
        font_override=font_override,
        base_size_pt=10.5 if base_size_pt is None else base_size_pt,
        leading_em=0.85 if leading_em is None else leading_em,
        paper_size=paper_size,
        figure_prefix=config.figure_prefix,
        table_prefix=config.table_prefix,
    )


__all__ = ["BilingualStyle", "Direction", "Theme", "direction_for", "resolve_theme"]
