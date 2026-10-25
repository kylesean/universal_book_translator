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
class TypographicConvention:
    """Per-language body-text metrics, from published typesetting standards.

    **Why this exists as data rather than a literal in the preamble.** The
    renderer emitted one rule for every target: ``first-line-indent: 2em``. For
    Chinese that is correct — GB/T 15834-2011 and the ROC handbook both require
    two characters (首行缩进两格). For Japanese it is wrong: the convention is
    ``一字下げ``, one character. Korean takes one character as well, inherited
    from 원고지 manuscript practice. So two of the three languages this engine
    renders most were indented twice as far as their own tradition — a defect
    that reads as "the typesetting was done by someone who only knows Chinese",
    and that no test could catch because nothing owned the value.

    ``first_line_indent_em`` is in **em**, which equals one character width for
    a CJK target and one lowercase-advance-unit for a Latin one. Zero means the
    convention is a block paragraph with spacing and no indent (the modern
    English trade default).

    ``leading_em`` is Typst's ``par(leading:)``, which is the *gap* added
    between lines, not the total line advance. The relationship measured
    against Typst 0.15.1 with a CJK face is roughly
    ``advance ≈ 1 + leading``, so 0.78em of leading yields ~1.78em advance —
    inside the 1.5-1.8em band that CJK books use. It is kept per-language so a
    future measure that wants looser Thai or Latin can be expressed here
    instead of in the renderer.

    RTL targets are absent on purpose: they set from a different set of
    conventions (no first-line indent in Arabic book typography, and
    ``leading`` interacts with the taller Naskh line box), and inventing
    values without a source would be worse than leaving the Latin default.
    """

    #: First-line paragraph indent, in em (= one CJK character).
    first_line_indent_em: float
    #: ``par(leading:)`` in em, i.e. the gap between line boxes.
    leading_em: float
    #: Where the numbers come from, for whoever tunes them next.
    source: str


#: Measured, not guessed. Each entry names the standard it encodes so a reader
#: can check it rather than trust it.
_TYPOGRAPHIC_CONVENTIONS: dict[str, TypographicConvention] = {
    # GB/T 15834-2011 and the ROC《重訂標點符號手冊》 both require 首行缩进
    # two characters; the rule dates to the 1919/1920 punctuation reform.
    "zh": TypographicConvention(
        first_line_indent_em=2.0,
        leading_em=0.78,
        source="GB/T 15834-2011 (首行缩进两格); 行送り 1.5-1.8em",
    ),
    # 一字下げ: one character, not two. Codified in Japanese publishing practice
    # after WWII and standard on 原稿用紙.
    "ja": TypographicConvention(
        first_line_indent_em=1.0,
        leading_em=0.78,
        source="一字下げ (one-character indent); JIS X 4051 行間 1/2-1em",
    ),
    # 원고지 practice leaves one 칸 blank; W3C klreq §4.2 line-head indentation.
    # Book leading runs 1.5-1.9x (Morisawa Hangeul guide).
    "ko": TypographicConvention(
        first_line_indent_em=1.0,
        leading_em=0.78,
        source="원고지 한 칸 들여쓰기; W3C klreq §4.2; Morisawa 150-190%",
    ),
    # Latin book practice: a one-em first-line indent (Chicago Manual of Style
    # §2.11 "indent the first line of each paragraph"), with ~1.2-1.35x leading.
    # The renderer emitted the Chinese two-character indent for English too,
    # which is twice the convention *and* twice the intended visual step.
    "en": TypographicConvention(
        first_line_indent_em=1.0,
        # Typst's `leading` is the gap, not the advance: measured with Typst
        # 0.15.1, advance ≈ 1 + leading, so 0.35em gives ~1.35x line height.
        leading_em=0.35,
        source="Chicago Manual of Style §2.11 (1em indent); 1.2-1.35x leading",
    ),
}

#: Languages that set from their own conventions but have none recorded here.
#: They keep whatever the renderer emitted before this module existed rather
#: than receiving a value invented for them: Arabic and Hebrew book typography
#: do not simply inherit the Latin indent rule, and Thai/Khmer line-breaking
#: conventions are their own subject. Recording a guess would make the
#: delivery *look* researched while being wrong, which is worse than an
#: admitted default.
_UNRECORDED_CONVENTION_LANGUAGES: frozenset[str] = frozenset(
    {"ar", "he", "fa", "ur", "ps", "sd", "ug", "yi", "dv", "th", "lo", "my", "km"}
)

#: What the renderer emitted before this table existed. Used for a language
#: whose convention is not recorded, so no unrelated output changes.
_LEGACY_CONVENTION = TypographicConvention(
    first_line_indent_em=2.0,
    leading_em=0.78,
    source="pre-table renderer literal (CJK two-character indent)",
)


def convention_for(language: str | None) -> TypographicConvention:
    """The typographic convention for a target, or the Latin default.

    Resolution order: an exact tag, then its base language, then the default.
    A Chinese target keeps the two-character indent the renderer has always
    emitted, so Chinese output does not change. Japanese and Korean stop being
    indented twice as far as their own convention. Every other language with a
    recorded convention (today: the Latin default) gets one-em paragraphs;
    languages whose convention is *not* recorded keep the previous literal so
    this change cannot silently restyle them.
    """
    tag = (language or "").strip().lower().replace("_", "-")
    base = tag.split("-")[0]
    for key in (tag, base):
        if key in _TYPOGRAPHIC_CONVENTIONS:
            return _TYPOGRAPHIC_CONVENTIONS[key]
    if base in _UNRECORDED_CONVENTION_LANGUAGES:
        return _LEGACY_CONVENTION
    # `zh-TW`/`zh-HK` share the Chinese rule; anything else with a recorded
    # script falls to the Latin book convention.
    if base == "zh":
        return _TYPOGRAPHIC_CONVENTIONS["zh"]
    return _TYPOGRAPHIC_CONVENTIONS["en"]


@dataclass(frozen=True, slots=True)
class Theme:
    """A run's resolved presentation policy: fonts, metrics, pairing, direction."""

    source_lang: str = "en"
    target_lang: str = "zh"
    fonts: tuple[str, ...] = ()
    font_override: str | None = None
    base_size_pt: float = 10.5
    #: Body leading, in em. **The value the renderer actually emits** for a
    #: single-column page: ``document_renderer`` used to hardcode its own
    #: leading (0.78em here, 0.65/0.75em for the two-column academic template)
    #: while this field claimed 0.85, so a maintainer who read this module got a
    #: setting no page ever used. ``leading_em`` is now the owner for the default
    #: layout; the academic template keeps its tighter per-column values because
    #: the tighter measure genuinely wants them.
    leading_em: float = 0.78
    #: First-line paragraph indent, in em (= one CJK character). Resolved per
    #: language from :data:`_TYPOGRAPHIC_CONVENTIONS`; ``None`` means "not
    #: resolved", and the renderer then falls back to the Chinese two-character
    #: value it has always emitted.
    first_line_indent_em: float | None = None
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
    :func:`ubt.core.language_profile.resolve_font_config`, then the stack is
    resolved against the machine through :mod:`ubt.adapters.pdf.font_probe`:
    the per-language profile is a static name list, and a stack nobody probed
    renders an entire CJK/RTL book as tofu with the compile warnings discarded.
    The metric defaults match the reconstructor's constructor. Nothing here
    duplicates a policy that already has an owner.
    """
    from ubt.adapters.pdf.font_probe import resolve_for_render
    from ubt.core.language_profile import resolve_font_config

    # One resolver for every renderer: profile + machine probe + the
    # protected-override rule all live in ``resolve_for_render``, so this
    # orchestrator cannot drift from what the preamble builders emit.
    resolved = resolve_for_render(target_lang, font_override=font_override)
    config = resolve_font_config(target_lang)
    convention = convention_for(target_lang)
    return Theme(
        source_lang=(source_lang or "en").strip().lower().replace("_", "-"),
        target_lang=(target_lang or "zh").strip().lower().replace("_", "-"),
        fonts=resolved.families,
        font_override=None,
        base_size_pt=10.5 if base_size_pt is None else base_size_pt,
        # The per-language convention supplies the default; an explicit
        # argument still wins, so a caller that measured its own value keeps it.
        leading_em=convention.leading_em if leading_em is None else leading_em,
        first_line_indent_em=convention.first_line_indent_em,
        paper_size=paper_size,
        figure_prefix=config.figure_prefix,
        table_prefix=config.table_prefix,
    )


__all__ = ["BilingualStyle", "Direction", "Theme", "direction_for", "resolve_theme"]
