"""Language profiles: per-target-language policy for the 0-token defense net.

Mechanism lives in validators/cleaners; policy (thresholds, script identity
checks) lives here, so supporting a new language pair is a configuration
change, not a code change. The ``zh`` profile defines the standard en→zh behavior.

Design notes:
- ``min_length_ratio`` / ``max_length_ratio`` bound the target/source character
  ratio prior. Latin-script pairs sit near 1.0; CJK pairs are far looser.
  Values for non-zh pairs are conservative starting calibrations.
- ``min_target_ratio`` gates "target-language identity": the fraction of
  target-script characters in prose. A value of 0.0 disables the gate —
  required for same-script pairs (e.g. en→fr) where character-class ratios
  cannot distinguish English residue from French (that needs language ID,
  which is out of scope for 0-token checks).
"""

from collections.abc import Callable
from dataclasses import dataclass


def _prose_total(text: str) -> int:
    return max(1, len(text.replace(" ", "")))


def cjk_script_ratio(text: str) -> float:
    """Fraction of CJK Unified Ideographs / Extension A among non-space chars."""
    total = _prose_total(text)
    count = sum(
        1 for c in text for lo, hi in ((0x3400, 0x4DBF), (0x4E00, 0x9FFF)) if lo <= ord(c) <= hi
    )
    return count / total


def kana_script_ratio(text: str) -> float:
    """Fraction of Hiragana + Katakana among non-space chars (Japanese identity)."""
    total = _prose_total(text)
    count = sum(
        1 for c in text for lo, hi in ((0x3040, 0x309F), (0x30A0, 0x30FF)) if lo <= ord(c) <= hi
    )
    return count / total


def hangul_script_ratio(text: str) -> float:
    """Fraction of Hangul syllables among non-space chars (Korean identity)."""
    total = _prose_total(text)
    count = sum(
        1 for c in text for lo, hi in ((0xAC00, 0xD7AF), (0x1100, 0x11FF)) if lo <= ord(c) <= hi
    )
    return count / total


def latin_script_ratio(text: str) -> float:
    """Fraction of Latin letters (incl. extended/A+B) among non-space chars.

    Note: for same-script source pairs (en→fr, en→de, ...) this cannot detect
    English echo; ``min_target_ratio=0.0`` disables the identity gate there.
    It still catches gross failures such as CJK echo in a Latin target.
    """
    total = _prose_total(text)
    count = sum(
        1
        for c in text
        for lo, hi in ((0x41, 0x5A), (0x61, 0x7A), (0xC0, 0x24F))
        if lo <= ord(c) <= hi and ord(c) not in (0xD7, 0xF7)
    )
    return count / total


def cyrillic_script_ratio(text: str) -> float:
    """Fraction of Cyrillic characters among non-space chars (Russian identity)."""
    total = _prose_total(text)
    count = sum(1 for c in text if 0x0400 <= ord(c) <= 0x04FF or 0x0500 <= ord(c) <= 0x052F)
    return count / total


@dataclass(frozen=True, slots=True)
class LanguageProfile:
    """Per-target-language policy consumed by FastPassFilter and friends."""

    code: str
    name: str
    min_length_ratio: float
    max_length_ratio: float
    min_target_ratio: float  # 0.0 disables the target-identity gate
    target_script_ratio: Callable[[str], float]
    supports_backfill: bool = True


# en→zh: default baseline profile for FastPassFilter.
# NOTE: this profile is the no-argument fallback for ``FastPassFilter()``.
# Production construction paths passing source_lang/target_lang resolve through
# :func:`get_pair_policy`, whose ("en", "zh") band is calibrated to (0.2, 1.5).
ZH = LanguageProfile("zh", "Chinese", 0.2, 3.0, 0.25, cjk_script_ratio)
JA = LanguageProfile("ja", "Japanese", 0.2, 2.5, 0.10, kana_script_ratio)
KO = LanguageProfile("ko", "Korean", 0.2, 2.5, 0.30, hangul_script_ratio)
FR = LanguageProfile("fr", "French", 0.6, 1.8, 0.0, latin_script_ratio)
DE = LanguageProfile("de", "German", 0.7, 2.0, 0.0, latin_script_ratio)
ES = LanguageProfile("es", "Spanish", 0.6, 1.8, 0.0, latin_script_ratio)
EN = LanguageProfile("en", "English", 0.6, 1.8, 0.0, latin_script_ratio)
RU = LanguageProfile("ru", "Russian", 0.6, 2.0, 0.25, cyrillic_script_ratio)

PROFILES: dict[str, LanguageProfile] = {p.code: p for p in (ZH, JA, KO, FR, DE, ES, EN, RU)}


@dataclass(frozen=True, slots=True)
class LanguagePairPolicy:
    """Dynamic joint policy for a directional (source_lang -> target_lang) translation pair.

    Encapsulates expansion/contraction priors, script identity gates, and terminology rules.
    Structurally compatible with LanguageProfile so existing validators consume it directly.
    """

    source_code: str
    target_code: str
    source_name: str
    target_name: str
    min_length_ratio: float
    max_length_ratio: float
    min_target_ratio: float
    target_script_ratio: Callable[[str], float]
    supports_backfill: bool = True

    @property
    def code(self) -> str:
        """Compatibility property matching LanguageProfile.code (target language)."""
        return self.target_code

    @property
    def name(self) -> str:
        """Compatibility property matching LanguageProfile.name (target language)."""
        return self.target_name


# Pair-specific expansion/contraction ratios (source, target) -> (min_ratio, max_ratio)
_PAIR_RATIO_BOUNDS: dict[tuple[str, str], tuple[float, float]] = {
    # English -> Mainstream targets (Phase 1 focus)
    ("en", "zh"): (0.2, 1.5),
    ("en", "ja"): (0.2, 1.8),
    ("en", "ko"): (0.25, 1.8),
    ("en", "de"): (0.7, 2.0),
    ("en", "fr"): (0.7, 1.9),
    ("en", "es"): (0.7, 1.9),
    ("en", "ru"): (0.6, 2.0),
    # Reverse pairs: CJK -> English (significant expansion)
    ("zh", "en"): (1.0, 5.0),
    ("ja", "en"): (0.8, 4.5),
    ("ko", "en"): (0.8, 4.5),
    # Latin -> English
    ("de", "en"): (0.6, 1.6),
    ("fr", "en"): (0.6, 1.6),
    ("es", "en"): (0.6, 1.6),
    ("ru", "en"): (0.5, 1.8),
    # Cross-pairs for Spanish (es)
    ("es", "zh"): (0.2, 1.5),
    ("zh", "es"): (1.0, 5.0),
    ("es", "fr"): (0.8, 1.3),
    ("fr", "es"): (0.8, 1.3),
    ("es", "de"): (0.7, 1.4),
    ("de", "es"): (0.8, 1.5),
    # Cross-pairs for Korean (ko)
    ("ko", "zh"): (0.5, 1.8),
    ("zh", "ko"): (0.6, 2.0),
    ("ko", "ja"): (0.7, 1.6),
    ("ja", "ko"): (0.7, 1.6),
}

_CJK_SCRIPTS = frozenset({"zh", "ja", "ko"})
_LATIN_SCRIPTS = frozenset({"en", "fr", "de", "es", "it", "pt", "nl"})
_CYRILLIC_SCRIPTS = frozenset({"ru", "uk", "bg"})


def resolve_script_family_bounds(
    src: str, tgt: str, fallback_bounds: tuple[float, float]
) -> tuple[float, float]:
    """Dynamically compute expansion/contraction bounds based on script family transitions."""
    if (src, tgt) in _PAIR_RATIO_BOUNDS:
        return _PAIR_RATIO_BOUNDS[(src, tgt)]

    # Latin to CJK (Contractive)
    if src in _LATIN_SCRIPTS and tgt in _CJK_SCRIPTS:
        return (0.18, 1.8)

    # CJK to Latin (Expansive)
    if src in _CJK_SCRIPTS and tgt in _LATIN_SCRIPTS:
        return (0.8, 5.0)

    # Same script family (Latin to Latin, CJK to CJK)
    if (src in _LATIN_SCRIPTS and tgt in _LATIN_SCRIPTS) or (
        src in _CJK_SCRIPTS and tgt in _CJK_SCRIPTS
    ):
        return (0.5, 2.0)

    # Cyrillic transitions
    if src in _CYRILLIC_SCRIPTS and tgt in _CJK_SCRIPTS:
        return (0.2, 1.8)
    if src in _CJK_SCRIPTS and tgt in _CYRILLIC_SCRIPTS:
        return (0.8, 4.5)
    if src in _CYRILLIC_SCRIPTS and tgt in _LATIN_SCRIPTS:
        return (0.5, 1.8)
    if src in _LATIN_SCRIPTS and tgt in _CYRILLIC_SCRIPTS:
        return (0.6, 2.0)

    return fallback_bounds


def normalize_lang_code(code: str) -> str:
    """Reduce a BCP-47-ish tag to the base code used by :data:`PROFILES`.

    Region and script subtags are dropped when the full tag is not itself a
    profile: ``zh-CN``/``zh-TW``/``zh-Hans`` -> ``zh``, ``en-US`` -> ``en``.
    An unknown base language is returned unchanged, so callers that must reject
    it (entry points) can do so explicitly instead of getting a silent fallback.
    """
    if not code:
        return ""
    normalized = code.strip().lower().replace("_", "-")
    if normalized in PROFILES:
        return normalized
    return normalized.split("-", 1)[0]


def is_supported_lang(code: str) -> bool:
    """Whether ``code`` (optionally region/script-tagged) has a language profile."""
    return normalize_lang_code(code) in PROFILES


def get_profile(code: str) -> LanguageProfile:
    """Resolve a profile by BCP-47-ish code; raises on unknown codes (no silent fallback).

    Region/script subtags resolve to their base language (``zh-CN`` -> ``zh``),
    so an entry point that accepts such a tag no longer ingests a whole book and
    only then fails with ``Unknown language profile: 'zh-cn'``.
    """
    profile = PROFILES.get(normalize_lang_code(code))
    if profile is None:
        raise ValueError(f"Unknown language profile: {code!r}. Available: {sorted(PROFILES)}")
    return profile


def get_pair_policy(source_lang: str = "en", target_lang: str = "zh") -> LanguagePairPolicy:
    """Resolve a bidirectional pair policy with calibrated length bounds and script guards.

    For unknown source languages, safely falls back to the target profile defaults.

    Region/script subtags are normalized first, so ``en-US -> zh-CN`` hits the
    calibrated ``("en", "zh")`` ratio band rather than the generic script-family
    fallback.
    """
    src = normalize_lang_code(source_lang) or "en"
    tgt = normalize_lang_code(target_lang) or "zh"

    tgt_profile = get_profile(tgt)
    src_profile = PROFILES.get(src)
    src_name = src_profile.name if src_profile else src.upper()

    min_len, max_len = resolve_script_family_bounds(
        src, tgt, (tgt_profile.min_length_ratio, tgt_profile.max_length_ratio)
    )

    # Script gating: if source and target share Latin script, character-ratio gate cannot
    # distinguish source residue from target text, so min_target_ratio is 0.0.
    min_target = tgt_profile.min_target_ratio
    if src in _LATIN_SCRIPTS and tgt in _LATIN_SCRIPTS:
        min_target = 0.0

    return LanguagePairPolicy(
        source_code=src,
        target_code=tgt,
        source_name=src_name,
        target_name=tgt_profile.name,
        min_length_ratio=min_len,
        max_length_ratio=max_len,
        min_target_ratio=min_target,
        target_script_ratio=tgt_profile.target_script_ratio,
        supports_backfill=tgt_profile.supports_backfill,
    )


@dataclass(frozen=True, slots=True)
class FontConfig:
    """Per-language typography and figure caption configuration."""

    typst_fonts: tuple[str, ...]
    latex_main_font: str = "Liberation Serif"
    figure_prefix: str = "Fig."
    table_prefix: str = "Table"


_DEFAULT_LATIN_FONTS: tuple[str, ...] = ("Liberation Serif", "Linux Libertine", "DejaVu Serif")

_FONT_CONFIGS: dict[str, FontConfig] = {
    "zh": FontConfig(
        typst_fonts=(
            "Noto Serif CJK SC",
            "Source Han Serif SC",
            "Noto Sans CJK SC",
            "Liberation Serif",
        ),
        figure_prefix="图",
        table_prefix="表",
    ),
    "zh-cn": FontConfig(
        typst_fonts=(
            "Noto Serif CJK SC",
            "Source Han Serif SC",
            "Noto Sans CJK SC",
            "Liberation Serif",
        ),
        figure_prefix="图",
        table_prefix="表",
    ),
    "zh-hans": FontConfig(
        typst_fonts=(
            "Noto Serif CJK SC",
            "Source Han Serif SC",
            "Noto Sans CJK SC",
            "Liberation Serif",
        ),
        figure_prefix="图",
        table_prefix="表",
    ),
    "zh-tw": FontConfig(
        typst_fonts=(
            "Noto Serif CJK TC",
            "Source Han Serif TC",
            "Noto Sans CJK TC",
            "Liberation Serif",
        ),
        figure_prefix="圖",
        table_prefix="表",
    ),
    "zh-hant": FontConfig(
        typst_fonts=(
            "Noto Serif CJK TC",
            "Source Han Serif TC",
            "Noto Sans CJK TC",
            "Liberation Serif",
        ),
        figure_prefix="圖",
        table_prefix="表",
    ),
    "zh-hk": FontConfig(
        typst_fonts=(
            "Noto Serif CJK TC",
            "Source Han Serif TC",
            "Noto Sans CJK TC",
            "Liberation Serif",
        ),
        figure_prefix="圖",
        table_prefix="表",
    ),
    "ja": FontConfig(
        typst_fonts=(
            "Noto Serif CJK JP",
            "Source Han Serif JP",
            "Noto Sans CJK JP",
            "Liberation Serif",
        ),
        figure_prefix="図",
        table_prefix="表",
    ),
    "ko": FontConfig(
        typst_fonts=(
            "Noto Serif CJK KR",
            "Source Han Serif KR",
            "Noto Sans CJK KR",
            "Liberation Serif",
        ),
        figure_prefix="그림",
        table_prefix="표",
    ),
    "en": FontConfig(
        typst_fonts=_DEFAULT_LATIN_FONTS,
        figure_prefix="Fig.",
        table_prefix="Table",
    ),
    "fr": FontConfig(
        typst_fonts=_DEFAULT_LATIN_FONTS,
        figure_prefix="Figure",
        table_prefix="Tableau",
    ),
    "de": FontConfig(
        typst_fonts=_DEFAULT_LATIN_FONTS,
        figure_prefix="Abb.",
        table_prefix="Tabelle",
    ),
    "es": FontConfig(
        typst_fonts=_DEFAULT_LATIN_FONTS,
        figure_prefix="Figura",
        table_prefix="Tabla",
    ),
    "ru": FontConfig(
        typst_fonts=("Liberation Serif", "DejaVu Serif", "Noto Serif"),
        figure_prefix="Рис.",
        table_prefix="Таблица",
    ),
}


def resolve_font_config(target_lang: str) -> FontConfig:
    """Resolve font family and figure caption prefix settings for ``target_lang``."""
    code = (target_lang or "zh").strip().lower().replace("_", "-")
    if code in _FONT_CONFIGS:
        return _FONT_CONFIGS[code]
    prefix = code.split("-")[0]
    if prefix in _FONT_CONFIGS:
        return _FONT_CONFIGS[prefix]
    if "zh" in code:
        return _FONT_CONFIGS["zh"]
    return _FONT_CONFIGS["en"]
