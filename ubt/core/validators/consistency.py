"""0-Token consistency validators for numbers and glossary terms."""

import re
from decimal import Decimal
from typing import Any

from ubt.core.cjk_ranges import HAN_UNIFIED_CLASS
from ubt.core.language_profile import LanguagePairPolicy, LanguageProfile
from ubt.core.policy.layout_policy import CN_MEASURE_WORDS
from ubt.core.validators.base import ContentValidator, ValidationResult

_NUM = re.compile(r"\d[\d,.\-–—/]*\d|\d")

_CN_DIGIT_VALUES = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_CN_UNIT_VALUES = {"十": 10, "百": 100, "千": 1000}
# Magnitude-suffixed numbers: '250万' = 2500000, '1.5亿' = 150000000
_WAN_YI_RE = re.compile(r"(\d+(?:\.\d+)?)([万亿])")
# A run of adjacent magnitudes is ONE quantity ("1亿2000万" = 120000000); the
# per-magnitude regex above must not scale and concatenate them separately.
_WAN_YI_SEQ_RE = re.compile(r"(?:\d+(?:\.\d+)?[万亿])+")
_FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")
# Arabic-Indic (U+0660-0669) and Extended Arabic-Indic (U+06F0-06F9) digits.
# An Arabic target may render a source number in either; without this fold the
# ASCII source digit has no match and a correct translation reports a lost
# number. Match-only view, stored text untouched.
_ARABIC_INDIC_DIGITS = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)
# Unicode sub/superscript digits -> ASCII (match-only view, stored text is
# untouched): PDF extraction spaces subscripts ('β 2') while translation
# normalizes them ('β²' / 'β₀'). Without this, a correct 'β²项' fails the
# '2'-presence check — the same false-positive family as CJK numerals.
# The ASCII digit is prefixed with a separator ('β²' -> 'β^2', 'H₂O' -> 'H_2O')
# instead of being concatenated: a superscript directly after a digit ('10²')
# would otherwise fold into the phantom token '102', and the source's own '10'
# and '2' would both read as lost — a correct translation quarantined as a
# dropped number. The separator keeps every digit a standalone token.
_SUB_SUP_DIGITS_RE = re.compile(r"[⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉]")
_SUB_SUP_DIGITS_MAP = {
    **{sup: f"^{d}" for sup, d in zip("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789", strict=True)},
    **{sub: f"_{d}" for sub, d in zip("₀₁₂₃₄₅₆₇₈₉", "0123456789", strict=True)},
}


def _cn_numeral_value(s: str) -> int:
    """Convert a Chinese numeral string to int.

    Handles the complete form ('一百二十三'=123), positional concatenation
    ('一九八四'=1984), and the colloquial shorthand where a trailing digit
    stands for the next-lower unit position ('一百二'=120, '两千五'=2500). A
    零 placeholder marks an order-of-magnitude gap and disables the shorthand
    ('一千零五'=1005, not 1500).
    """
    total = 0
    current = 0
    last_unit = 0
    tail_has_zero = False  # a 零 appeared in the digit run after the last unit
    for ch in s:
        if ch in _CN_DIGIT_VALUES:
            digit = _CN_DIGIT_VALUES[ch]
            if digit == 0:
                tail_has_zero = True
            # Consecutive digits concatenate positionally: '一九八四' = 1984
            current = current * 10 + digit if current else digit
        elif ch in _CN_UNIT_VALUES:
            if current == 0:
                current = 1  # '十' alone means 10, not 0*10
            total += current * _CN_UNIT_VALUES[ch]
            current = 0
            last_unit = _CN_UNIT_VALUES[ch]
            tail_has_zero = False
    if current:
        if last_unit >= 100 and not tail_has_zero and current < 10:
            # Shorthand: the lone trailing digit fills the position just below
            # the last unit. '百'(100)->×10, '千'(1000)->×100.
            total += current * (last_unit // 10)
        else:
            total += current
    return total


# Chinese numerals are normalized for *matching only* (the stored text is
# never touched), but only in genuine numeral contexts: a structural marker
# ('第七章' -> '第7章'), a numeral run bounded by non-CJK text, or a run followed
# by a known measure word/unit ('为两百元' -> '为200元'). A bare numeral
# character embedded inside a CJK word ('统一' -> '统1') is not a number and
# must not mask a genuinely lost digit.
#
# Keying the normalisation off a bare non-CJK boundary fails in BOTH
# directions:
#   * '为两百元' / '共二十人' / '长达三十米' would go unnormalised, because the
#     run is CJK-flanked on both sides; the digit check then reports the value
#     as missing and quarantines a *correct* translation as a critical defect.
#   * '统一' at a string boundary gets normalised to '统1', fabricating a "1"
#     out of a word and thereby masking a genuinely lost digit — the exact
#     failure the boundary rule claims to prevent. '十分' ('10分'), '一度',
#     '一面', '一部分' hit the same flaw through the left-boundary variant.
# The measure-word clause fixes the first family; requiring two or more
# characters for the boundary clauses fixes the second, because a bare
# single numeral character next to a boundary is far more often a word fragment
# than a number. Verified: 61 non-numeric numeral-containing words normalise to
# themselves ('统一', '万一', '十分', '一度', '一部分', '一概而论', '一视同仁', …).
#
# Deliberate residual (a lexicon, not a regex, would be needed to remove it):
#   * a chengyu shaped like <numeral><measure-word> still normalises —
#     '三番五次' -> '三番5次', '五花八门' -> '五花8门';
#   * a single-character numeral with neither a measure word nor a
#     multi-character run ('数量为三。') is left unnormalised.
# The trade-off is deliberate: shrinking the measure-word rule to
# multi-character runs would re-break ordinary prose ('有五项', '三人同行',
# '五年'), whose failure mode is the *user-visible* one — a correct translation
# reported as a missing number, escalated to repair and possibly quarantined as
# BLOCKED_HUMAN — while the chengyu cases only mask a loss when the source
# happens to carry exactly the digit the idiom fabricates. Prefer the common
# false negative over a rare one.
_CN_NUMERAL_CHARS = "零一二两三四五六七八九十百千"
_CN_MEASURE_RE = "|".join(
    re.escape(word) for word in sorted(CN_MEASURE_WORDS, key=len, reverse=True)
)
_CN_NUMERAL_CONTEXT_RE = re.compile(
    rf"(?<=第)[{_CN_NUMERAL_CHARS}]+"
    rf"|(?<![{HAN_UNIFIED_CLASS}])[{_CN_NUMERAL_CHARS}]{{2,}}"
    rf"|[{_CN_NUMERAL_CHARS}]+(?={_CN_MEASURE_RE})"
    rf"|[{_CN_NUMERAL_CHARS}]{{2,}}(?![{HAN_UNIFIED_CLASS}])"
)
# Century/decade idiom: '20世纪80年代' == '二十世纪八十年代' == EN 'the 1980s'.
# (century - 1) * 100 + decade, e.g. 20世纪80年代 -> 1980年代. Either side may be
# written in Chinese numerals, so the character class accepts both scripts.
_CN_CENTURY_DECADE_RE = re.compile(
    rf"([0-9{_CN_NUMERAL_CHARS}]{{1,3}})\s*世纪\s*([0-9{_CN_NUMERAL_CHARS}]{{1,3}})\s*年代"
)


def _cn_or_ascii_value(s: str) -> int:
    """Value of a run written in either ASCII digits or Chinese numerals."""
    return int(s) if s.isdigit() else _cn_numeral_value(s)


# Scientific notation is a *number*, not a digit run plus a stray digit: '1e5'
# denotes 100000, so a target that writes the expanded value ('100000') is
# correct and must not be reported as dropping '1' and '5'. Expanded for the
# match-only view on both sides so either spelling satisfies the gate.
_SCI_NOTATION_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)[eE]([+-]?\d+)(?!\w)")


def _expand_scientific_not(text: str) -> str:
    """Rewrite ``NeM`` to its expanded decimal string (match-only view)."""

    def _repl(match: re.Match[str]) -> str:
        try:
            exponent = int(match.group(2))
            # Bound the expansion: a pathological exponent would otherwise build
            # a multi-megabyte string (or hang) in a match-only preparation step.
            if abs(exponent) > 1000:
                return match.group(0)
            return format(Decimal(match.group(1)) * (Decimal(10) ** exponent), "f")
        except (ValueError, ArithmeticError):
            return match.group(0)

    return _SCI_NOTATION_RE.sub(_repl, text)


# PDF extraction commonly puts spaces around decimal points ('19 . 6', '0 . 007', '+ 24 . 7').
# Collapse them so they tokenize as single decimals instead of separate integers.
_SPACED_DECIMAL_RE = re.compile(r"(?<![\w.])(\d+)\s*\.\s*(\d+)(?![\w.])")


def _collapse_spaced_decimals(text: str) -> str:
    """Collapse PDF extraction spaces around decimal points ('19 . 6' -> '19.6')."""
    return _SPACED_DECIMAL_RE.sub(r"\1.\2", text)


# Locale separator canonicalization: '1,500' (EN thousands),
# '1.500' (DE thousands) and decimal '15,6' (DE/FR) must all survive the
# digit-presence check regardless of the target language's convention.
# Leading '0.' / '0,' (e.g. '0.125') is never a thousands separator.
_THOUSANDS_COMMA_RE = re.compile(r"(?:(?<=[1-9])|(?<=\d\d)),(?=\d{3}(?!\d))")
_THOUSANDS_DOT_RE = re.compile(r"(?:(?<=[1-9])|(?<=\d\d))\.(?=\d{3}(?!\d))")
_DECIMAL_COMMA_RE = re.compile(r"(?<=\d),(?=\d+(?!\d))")
# Decimal magnitude equivalence: '1.50' and '1.5' are the same number. Strip
# insignificant trailing fractional zeros so a correct rendering is not read
# as a dropped number.
_TRAILING_ZERO_DEC_RE = re.compile(r"(?<=\d)\.(\d*?)0+(?!\d)")


def _strip_trailing_decimal_zeros(text: str) -> str:
    """Drop insignificant trailing zeros from decimal fractions ('1.50' -> '1.5')."""

    def _repl(match: re.Match[str]) -> str:
        frac = match.group(1)
        return f".{frac}" if frac else ""

    return _TRAILING_ZERO_DEC_RE.sub(_repl, text)


def canonicalize_numeric_token(s: str) -> str:
    """Canonicalize a numeric token: strip thousands separators, unify decimals."""
    t = s.strip()
    t = _THOUSANDS_COMMA_RE.sub("", t)
    t = _THOUSANDS_DOT_RE.sub("", t)
    t = _DECIMAL_COMMA_RE.sub(".", t)
    t = re.sub(r"[–—~～]", "-", t)
    return _strip_trailing_decimal_zeros(t)


def normalize_for_numeric_matching(text: str, lang: str = "zh") -> str:
    """Canonicalize target text before digit-presence checks.

    Order matters: full-width digits first (so '１９８４' participates), then
    genuine Chinese numerals in numeral contexts ('第七章' -> '第7章', '五项' ->
    '5项' — a match-only view, stored text is untouched), then 万/亿 magnitude
    scaling ('250万' → '2500000'), then locale separator and decimal
    canonicalization.
    """
    text = text.translate(_FULLWIDTH_DIGITS)
    text = text.translate(_ARABIC_INDIC_DIGITS)
    text = _expand_scientific_not(text)
    text = _collapse_spaced_decimals(text)
    text = _SUB_SUP_DIGITS_RE.sub(lambda m: _SUB_SUP_DIGITS_MAP[m.group(0)], text)
    if lang == "zh" or any(c in text for c in _CN_DIGIT_VALUES):
        # Century/decade idiom BEFORE the context normaliser: the latter can
        # convert only one of the two numerals ('二十世纪八十年代' ->
        # '20世纪八十年代'), after which neither the Chinese nor the digit-only
        # pattern can fold the other side.
        text = _CN_CENTURY_DECADE_RE.sub(
            lambda m: (
                f"{(_cn_or_ascii_value(m.group(1)) - 1) * 100 + _cn_or_ascii_value(m.group(2))}年代"
            ),
            text,
        )
        text = _CN_NUMERAL_CONTEXT_RE.sub(lambda m: str(_cn_numeral_value(m.group(0))), text)

        def _scale_sequence(m: re.Match[str]) -> str:
            # Sum the whole adjacent run: "1亿2000万" is 120000000, not
            # "10000000020000000" (each magnitude scaled and concatenated).
            # Decimal, not float: 万亿-scale sums exceed float's 2^53
            # exact-integer range, and this must fold exactly like
            # _cn_compound_runs below, which already uses Decimal.
            total = Decimal(0)
            for num, unit in _WAN_YI_RE.findall(m.group(0)):
                total += Decimal(num) * _CN_SCALE_FACTORS[unit]
            return str(round(total))

        text = _WAN_YI_SEQ_RE.sub(_scale_sequence, text)
    text = _THOUSANDS_COMMA_RE.sub("", text)
    text = _THOUSANDS_DOT_RE.sub("", text)
    text = _DECIMAL_COMMA_RE.sub(".", text)
    text = _strip_trailing_decimal_zeros(text)
    text = re.sub(r"[–—~～]", "-", text)
    return text.replace(",", "").replace("，", "")


_RANGE_DELIMITERS = re.compile(r"[-–—/]")
_GLUED_PAGE_RANGE_SOURCE_RE = re.compile(r"\bpp\.?\s*(?P<digits>\d{4})\b", re.IGNORECASE)
_FLATTENED_FOOTNOTE_RE = re.compile(r"(?<=\w)\s+(?P<marker>[1-9])\s+(?=[.,])")
# A leading ordered-list marker ("(1) ...", "1. ...", "1) ...") is editorial
# structure, not a numeric fact: the translator drops it and the renderer
# restores the marker, so the numeric check must not read it as a lost number.
_LEADING_LIST_MARKER_RE = re.compile(
    r"^\s*(?:\(\s*\d{1,3}\s*\)|\uff08\s*\d{1,3}\s*\uff09|\d{1,3}\s*[.)\u3001])",
    re.MULTILINE,
)


def _glued_page_range_is_preserved(original: str, translated: str, num: str) -> bool:
    """Accept a repaired ``pp. 4046`` extraction when target restores ``40–46``."""
    for match in _GLUED_PAGE_RANGE_SOURCE_RE.finditer(original):
        digits = match.group("digits")
        if digits != num:
            continue
        first, last = digits[:2], digits[2:]
        if first == last:
            continue
        target_range = re.compile(rf"(?<!\d){first}\s*[-–—/]\s*{last}(?!\d)")
        if target_range.search(translated):
            return True
    return False


# Scale-word equivalence (both directions). "250万像素" and "a 2.5-megapixel
# camera" are the same quantity, but the digit-presence check can only ever see
# '250' vs '2.5', so every magnitude rewritten by a unit word reads as a
# dropped number — which escalates a correct sentence to flagship repair and,
# when that fails, quarantines it as BLOCKED_HUMAN (Chinese source text would
# otherwise ship inside the English deliverable). Chinese books are dense in
# 万/亿, so this is the single largest false-positive source on the advertised
# en<->zh pair.
#
# The exemption is value-exact and unit-attested: a number may only be satisfied
# by a target number that equals it after multiplying by a scale word that is
# physically present next to one of the two numbers. A plain lost '250' still
# cannot be explained by a target '2.5' (2500000 != 2.5), so real omissions are
# still caught.
_CN_SCALE_FACTORS: dict[str, Decimal] = {
    # Multi-char magnitudes (百万 = 10^6, 千万 = 10^7, 万亿 = 10^12) are how a
    # Chinese translation restates "million"/"billion"; without them a correct
    # '12.5 million' -> '12.5 百万' read as a dropped magnitude.
    "十万": Decimal(100_000),
    "百万": Decimal(1_000_000),
    "千万": Decimal(10_000_000),
    "百亿": Decimal(10_000_000_000),
    "千亿": Decimal(100_000_000_000),
    "万亿": Decimal(1_000_000_000_000),
    "百": Decimal(100),
    "千": Decimal(1000),
    "万": Decimal(10000),
    "亿": Decimal(100_000_000),
}
_EN_SCALE_WORDS: dict[str, Decimal] = {
    "hundred": Decimal(100),
    "thousand": Decimal(1000),
    "million": Decimal(1_000_000),
    "billion": Decimal(1_000_000_000),
    "trillion": Decimal(1_000_000_000_000),
}
# SI prefixes attach to a unit without a separator ("2.5-megapixel", "50mm"),
# so they are matched as prefixes rather than whole words.
_SI_PREFIX_FACTORS: dict[str, Decimal] = {
    "tera": Decimal(1_000_000_000_000),
    "giga": Decimal(1_000_000_000),
    "mega": Decimal(1_000_000),
    "kilo": Decimal(1000),
    "hecto": Decimal(100),
    "deca": Decimal(10),
    "deci": Decimal("0.1"),
    "centi": Decimal("0.01"),
    "milli": Decimal("0.001"),
    "micro": Decimal("0.000001"),
    "nano": Decimal("0.000000001"),
}
_SCALE_FACTORS: dict[str, Decimal] = {**_CN_SCALE_FACTORS, **_EN_SCALE_WORDS, **_SI_PREFIX_FACTORS}
_SCALE_ADJACENCY_RE = re.compile(
    # CJK scale word next to the digits: '250万', '12.5 百万' (translations often
    # put a space before the magnitude). Multi-char magnitudes first so '百万'
    # is not read as '百'.
    r"(?P<cn>\d[\d.]*)\s*(?P<cn_unit>万亿|千亿|百亿|千万|百万|十万|[万亿千百])"
    # Whole English scale word: '2.5 million', '250,000 pixels per inch'
    r"|(?P<en>\d[\d,]*(?:\.\d+)?)\s*[-/]?\s*"
    r"(?P<en_unit>thousand|million|billion|trillion|hundred)(?![a-z])"
    # SI prefix on a unit: '2.5-megapixel', '30millisecond'
    r"|(?<![a-z])(?P<si>\d[\d,]*(?:\.\d+)?)[\s-]*"
    r"(?P<si_unit>tera|giga|mega|kilo|hecto|deca|deci|centi|milli|micro|nano)(?=[a-z])"
    # SI prefix as its symbol, which is how scientific prose abbreviates it:
    # '2.5 MPa', '50 kHz', '3 nm'. Only the spelled-out words were known, so a
    # correct translation of "2500000 Pa" was reported as having LOST the number
    # and escalated to repair -- and a repair cannot invent a better unit, so it
    # can only end up BLOCKED_HUMAN with the source placeholder shipped.
    # Case matters here (M = mega, m = milli), so this branch is looked up
    # case-sensitively; a bare unit word ("2500000 Pa") never matches, because
    # its leading letter is not an SI prefix symbol.
    r"|(?<![a-zA-Z])(?P<sym>\d[\d,]*(?:\.\d+)?)[ \t]?"
    # The unit tail is one uppercase-led symbol (Pa, Hz, mol? no) or a single
    # lowercase letter (kg, ms, nm), never a longer lowercase run: that shape is
    # a word ("3 mol"), and reading its "m" as milli would invent a scale and
    # wave a genuine 1000x error through the gate.
    # Exclude "pp" (percentage points) and "ppm"/"ppb"/"ppt" from being misread
    # as pico-unit combinations.
    r"(?P<sym_unit>(?!pp[mbt]?(?![a-zA-Z]))[kMmGTµμnp](?:[A-Z][a-z]{0,2}|[a-z]))(?![a-zA-Z])"
    # Bare count multiplier: '390K', '30k', '10M', '2B' (common in technical counts)
    r"|(?<![a-zA-Z])(?P<count>\d[\d,]*(?:\.\d+)?)[ \t]?"
    r"(?P<count_unit>[KkMBb])(?![a-zA-Z0-9])"
)

#: Bare count multipliers (390K -> 390,000, 10M -> 10,000,000, 2B -> 2,000,000,000).
_COUNT_SCALE_FACTORS: dict[str, Decimal] = {
    "k": Decimal(1000),
    "K": Decimal(1000),
    "M": Decimal(1_000_000),
    "b": Decimal(1_000_000_000),
    "B": Decimal(1_000_000_000),
}

#: SI prefix symbols, case-sensitive (``M`` mega vs ``m`` milli).
_SI_SYMBOL_FACTORS: dict[str, Decimal] = {
    "T": Decimal(1_000_000_000_000),
    "G": Decimal(1_000_000_000),
    "M": Decimal(1_000_000),
    "k": Decimal(1000),
    "m": Decimal("0.001"),
    "µ": Decimal("0.000001"),
    "μ": Decimal("0.000001"),
    "n": Decimal("0.000000001"),
    "p": Decimal("0.000000000001"),
}


def _to_decimal(token: str) -> Decimal | None:
    try:
        return Decimal(token)
    except Exception:
        return None


def _canon_value(value: Decimal) -> str:
    """Render a value without exponent or trailing zeros so both sides compare."""
    normalized = value.normalize()
    if normalized == normalized.to_integral_value():
        return str(normalized.quantize(Decimal(1)))
    return format(normalized, "f")


#: A CJK char that turns a leading 千/百 into a *measure-unit prefix* rather
#: than a magnitude: 千克 (kg), 千米 (km), 千字节 (kB), 百帕 (hPa), 千瓦 (kW) …
#: '10千克' states ten kilograms, not 10000; scaling it licensed a fabricated
#: 1000x value as "equivalent". 万/亿 are always magnitudes (250万, 1亿).
_CN_UNIT_PREFIX_SUFFIXES = frozenset("克米字帕瓦赫秒升欧伏安焦卡吨牛贝特巴")


def _scale_map(text: str) -> dict[str, set[str]]:
    """Map each unit-scaled number's textual form to the values it may denote here."""
    out: dict[str, set[str]] = {}
    for match in _SCALE_ADJACENCY_RE.finditer(text):
        raw = (
            match.group("cn")
            or match.group("en")
            or match.group("si")
            or match.group("sym")
            or match.group("count")
        )
        unit = (
            match.group("cn_unit")
            or match.group("en_unit")
            or match.group("si_unit")
            or match.group("sym_unit")
            or match.group("count_unit")
        )
        if raw is None or unit is None:
            continue
        if match.group("cn_unit") is not None and unit in "千百":
            nxt = text[match.end("cn_unit") : match.end("cn_unit") + 1]
            if nxt and nxt in _CN_UNIT_PREFIX_SUFFIXES:
                # '10千克' — a unit prefix, not a magnitude.
                continue
        if match.group("count") is not None:
            factor = _COUNT_SCALE_FACTORS.get(unit)
        elif match.group("sym") is not None:
            factor = _SI_SYMBOL_FACTORS.get(unit[0])
        else:
            factor = _SCALE_FACTORS.get(unit.lower())
        value = _to_decimal(canonicalize_numeric_token(raw))
        if factor is None or value is None:
            continue
        key = canonicalize_numeric_token(raw)
        out.setdefault(key, set()).add(_canon_value(value * factor))
    return out


def _denoted_values(text: str) -> set[str]:
    """Every value this text states: bare numbers plus unit-scaled numbers."""
    values: set[str] = set()
    for token in _NUM.findall(text):
        value = _to_decimal(canonicalize_numeric_token(token))
        if value is not None:
            values.add(_canon_value(value))
    for targets in _scale_map(text).values():
        values |= targets
    return values


def scale_equivalent_values(text: str) -> dict[str, set[str]]:
    """Public view of :func:`_scale_map`: number token -> values it may denote.

    A scale word restates a magnitude ("1.5 million" == "150万" == "1500000"),
    so a correct rendering shares no raw digit string with the source. Consumers
    outside the numeric validator (the omission gate's number metric and chrF
    residue) need the same equivalence.
    """
    return _scale_map(text)


def denoted_numeric_values(text: str) -> set[str]:
    """Public view of :func:`_denoted_values`: every value the text states."""
    return _denoted_values(text)


def magnitude_rewritten(text: str) -> str:
    """Rewrite each magnitude-scaled number to its value (``"12.5 million"`` -> ``"12500000"``).

    A magnitude word restates the number itself, so the omission gate scores a
    scale-scoped source number at its value and must rewrite the target the same
    way — otherwise a correct ``"12.5 百万"`` rendering shares no digit string
    with the source ``"12.5 million"``. SI prefixes are left alone: ``20 nm``
    keeps the number literal on both sides.
    """

    def _repl(match: re.Match[str]) -> str:
        raw = match.group("cn") or match.group("en")
        unit = match.group("cn_unit") or match.group("en_unit")
        if raw is None or unit is None:
            return match.group(0)  # SI prefix/symbol: the number stays literal.
        if match.group("cn_unit") is not None and unit in "千百":
            nxt = text[match.end("cn_unit") : match.end("cn_unit") + 1]
            if nxt and nxt in _CN_UNIT_PREFIX_SUFFIXES:
                return match.group(0)  # '10千克' — a unit prefix, not a magnitude.
        factor = _SCALE_FACTORS.get(unit.lower())
        value = _to_decimal(canonicalize_numeric_token(raw))
        if factor is None or value is None:
            return match.group(0)
        return _canon_value(value * factor)

    return _SCALE_ADJACENCY_RE.sub(_repl, text)


# Compound Chinese magnitudes ('3亿5000万','1万2千') are several digit+unit pairs
# that together state one value. Neither constituent matches a target written as
# a plain number ('350000000'), so without this the per-token check reports both
# as lost and a correct translation is escalated to repair.
_CN_COMPOUND_RE = re.compile(r"(?:\d[\d.]*[万亿千百]){2,}")
_CN_COMPOUND_PAIR_RE = re.compile(r"(\d[\d.]*)([万亿千百])")


def _cn_compound_runs(text: str) -> list[tuple[list[str], str]]:
    """(constituent canonical tokens, total value) for each compound magnitude run."""
    runs: list[tuple[list[str], str]] = []
    for match in _CN_COMPOUND_RE.finditer(text):
        pairs = _CN_COMPOUND_PAIR_RE.findall(match.group(0))
        total = Decimal(0)
        tokens: list[str] = []
        ok = True
        for raw, unit in pairs:
            value = _to_decimal(raw)
            if value is None:
                ok = False
                break
            total += value * _CN_SCALE_FACTORS[unit]
            tokens.append(canonicalize_numeric_token(raw))
        if ok and tokens:
            runs.append((tokens, _canon_value(total)))
    return runs


# Common rhetorical idioms where numbers are traditionally translated into
# Chinese idiomatic phrases (chengyu / dynamic equivalence) without literal digits.
_NUMERIC_IDIOM_PATTERNS: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (re.compile(r"\bcatch[-\s]?22\b", re.IGNORECASE), ("22",)),
    (re.compile(r"\b24\s*[/／]\s*7\b", re.IGNORECASE), ("24", "7", "24/7")),
    (re.compile(r"\b9\s*(?:-|to|-to-)\s*5\b", re.IGNORECASE), ("9", "5")),
    (re.compile(r"\btop\s*10\b", re.IGNORECASE), ("10",)),
    (re.compile(r"\btop\s*5\b", re.IGNORECASE), ("5",)),
    (re.compile(r"\b360[-\s]?degree\b", re.IGNORECASE), ("360",)),
    (re.compile(r"\b180[-\s]?degree\b", re.IGNORECASE), ("180",)),
    (re.compile(r"\b50[-\s/]?50\b", re.IGNORECASE), ("50",)),
    (re.compile(r"\b1[-\s]?on[-\s]?1\b", re.IGNORECASE), ("1",)),
    (re.compile(r"\b(?:at\s+)?sixes\s+and\s+sevens\b", re.IGNORECASE), ("6", "7")),
    (re.compile(r"\bcloud\s+nine\b", re.IGNORECASE), ("9",)),
    (re.compile(r"\btwo\s+peas\s+in\s+a\s+pod\b", re.IGNORECASE), ("2",)),
    (re.compile(r"\b(?:to|dressed\s+up\s+to)\s+the\s+nines\b", re.IGNORECASE), ("9",)),
    (re.compile(r"\ba\s+hundred\s+and\s+one\b", re.IGNORECASE), ("100", "1", "101")),
    (re.compile(r"\bone\s+in\s+a\s+million\b", re.IGNORECASE), ("1", "1000000")),
    (re.compile(r"\bkill\s+two\s+birds\s+with\s+one\s+stone\b", re.IGNORECASE), ("2", "1")),
    (re.compile(r"\bsecond\s+to\s+none\b", re.IGNORECASE), ("2",)),
    (re.compile(r"\bback\s+to\s+square\s+one\b", re.IGNORECASE), ("1",)),
)


def _has_numeric_token(target: str, num_str: str) -> bool:
    # Reject a fragment of a larger number: a decimal point/comma followed by a
    # digit ("3" inside "3.5") or preceded by a digit ("5" inside "3.5"). A
    # bare sentence period ("3.") is still a standalone token.
    def _standalone(value: str) -> bool:
        return bool(
            re.search(
                rf"(?<!\d)(?<!\d[.,]){re.escape(value)}(?!\d)(?![.,]\d)",
                target,
            )
        )

    if _standalone(num_str):
        return True
    # Strip leading zeros ONLY for a two-digit day/month token ('05' -> '5').
    # Applying it unconditionally let a changed identifier ('007' -> '7') pass
    # the numeric gate.
    if len(num_str) == 2 and num_str.startswith("0") and num_str.isdigit():
        stripped = num_str.lstrip("0")
        return bool(stripped and _standalone(stripped))
    return False


def _has_negative_token(target: str, num_str: str) -> bool:
    """True when ``num_str`` appears in ``target`` with a leading minus/负."""
    return bool(
        re.search(
            rf"(?<![\d.])[-−负]\s*{re.escape(num_str)}(?!\d)(?![.,]\d)",
            target,
        )
    )


class NumericConsistencyValidator(ContentValidator):
    """Ensures standalone numbers and years (e.g. 1984, percentages, stats) are preserved."""

    def __init__(self, profile: LanguageProfile | LanguagePairPolicy | None = None) -> None:
        self.profile = profile

    def validate(self, original: str, translated: str) -> ValidationResult:
        # Source tokens pass through the same locale canonicalization as
        # the target text, so '15,6' (decimal comma) vs '15.6' or '1.500' vs
        # '1500' style convention differences do not read as data loss.
        # Full-width source digits fold to ASCII ('１９８４' == '1984'):
        # str.isdigit() counts them as digits, but the target-side match view
        # normalizes them to ASCII, so an un-folded source token would never be
        # found and a correct translation would be quarantined as a dropped
        # number. Superscript/subscript digits are deliberately NOT folded
        # here: folding before tokenization would merge '10²' into the phantom
        # token '102', and since those characters are not in \d's class they
        # are simply not numeric tokens on the source side.
        src_view = _expand_scientific_not(original.translate(_FULLWIDTH_DIGITS))
        src_view = _collapse_spaced_decimals(src_view)
        src_nums = {canonicalize_numeric_token(m) for m in _NUM.findall(src_view)}
        # Residual ambiguity: a dot followed by exactly three digits is
        # BOTH a German-style thousands separator and a three-decimal
        # fraction. canonicalize strips it ('1.500' -> '1500'), so the
        # textual check can only ever require the integer reading — a
        # translation rendering the decimal reading ('1.500 g' -> '1.5 g')
        # would share no token with the source and be quarantined as a lost
        # number. Such a token licenses BOTH readings: it is satisfied when
        # the target denotes either value. A genuinely different value
        # denotes neither and is still caught.
        ambiguous_readings: dict[str, set[str]] = {}
        for match in _NUM.findall(src_view):
            if _THOUSANDS_DOT_RE.search(match):
                readings = {_to_decimal(canonicalize_numeric_token(match)), _to_decimal(match)}
                values = {_canon_value(v) for v in readings if v is not None}
                if values:
                    ambiguous_readings.setdefault(canonicalize_numeric_token(match), set()).update(
                        values
                    )
        src_nums.discard("")
        # A digit run immediately preceded by a minus sign is a negative
        # quantity; the sign is part of the fact, so a dropped '−' must fail
        # even though the digits survive. Hyphenated words ('epoch-5', 'tier-1')
        # are not negative quantities.
        negative_tokens: set[str] = set()
        for match in _NUM.finditer(src_view):
            if match.start() > 0 and src_view[match.start() - 1] in "-−":
                if match.start() > 1 and src_view[match.start() - 2].isalpha():
                    continue
                canon = canonicalize_numeric_token(match.group(0))
                if canon:
                    negative_tokens.add(canon)

        if not src_nums:
            return ValidationResult.success()

        target_code = self.profile.code if self.profile is not None else "zh"
        normalized_tgt = normalize_for_numeric_matching(translated, lang=target_code)

        # Idiom / flattened-footnote exemptions are OCCURRENCE-SCOPED: a token
        # is exempt only when EVERY occurrence in the source lies inside an
        # exempt span. The previous flat set let one idiom ("top 10") exempt a
        # standalone "Chapter 10" as well, so a dropped chapter reference
        # shipped as valid.
        # Spans must be computed on ``src_view`` (the same expanded view the
        # token offsets come from): scientific-notation expansion ('1e5' ->
        # '100000') changes length, so spans measured on ``original`` were
        # shifted and a correctly exempted token elsewhere was reported lost.
        exempt_spans: list[tuple[int, int]] = []
        for pat, _nums in _NUMERIC_IDIOM_PATTERNS:
            exempt_spans.extend((m.start(), m.end()) for m in pat.finditer(src_view))
        # PDF extraction commonly flattens a superscript footnote into a
        # spaced single digit before punctuation ("plugins 5 ,"). Such a
        # reference is editorial metadata, not a numeric fact to translate.
        exempt_spans.extend((m.start(), m.end()) for m in _FLATTENED_FOOTNOTE_RE.finditer(src_view))
        exempt_spans.extend(
            (m.start(), m.end()) for m in _LEADING_LIST_MARKER_RE.finditer(src_view)
        )

        exempt_numbers: set[str] = set()
        if exempt_spans:
            occurrences: dict[str, list[tuple[int, int]]] = {}
            for m in _NUM.finditer(src_view):
                canon = canonicalize_numeric_token(m.group(0))
                if canon:
                    occurrences.setdefault(canon, []).append((m.start(), m.end()))
            for canon, occs in occurrences.items():
                if all(any(s <= o_s and o_e <= e for s, e in exempt_spans) for o_s, o_e in occs):
                    exempt_numbers.add(canon)

        lost_numbers: list[str] = []
        # Scale equivalence is computed once per pair: the values the source
        # states next to a scale word, and every value the target states.
        src_scales = _scale_map(src_view)
        # Canonical digit runs that occur *bare* somewhere (not part of a scale
        # match). ``src_scales`` is keyed by the digits, so a bare '250' and a
        # scaled '250万' collapse to one key; without this set the loop below
        # checked that key against the magnitude only and skipped the bare
        # occurrence, so a dropped (or altered) bare number shipped as valid.
        # Spans are measured on ``src_view`` so they line up with ``_NUM``'s
        # tokens -- the same view ``src_nums`` came from.
        scaled_spans: list[tuple[int, int]] = []
        for scale_match in _SCALE_ADJACENCY_RE.finditer(src_view):
            for group in ("cn", "en", "si", "sym", "count"):
                if scale_match.group(group) is not None:
                    scaled_spans.append(scale_match.span(group))
                    break
        bare_numbers: set[str] = set()
        for num_match in _NUM.finditer(src_view):
            if any(s <= num_match.start() and num_match.end() <= e for s, e in scaled_spans):
                continue
            bare_canon = canonicalize_numeric_token(num_match.group(0))
            if bare_canon:
                bare_numbers.add(bare_canon)
        source_compounds = _cn_compound_runs(original.translate(_FULLWIDTH_DIGITS))
        tgt_values = _denoted_values(translated) | _denoted_values(normalized_tgt)
        tgt_values |= {
            total
            for text in (translated, normalized_tgt)
            for _tokens, total in _cn_compound_runs(text)
        }
        compound_satisfied = {
            token for tokens, total in source_compounds if total in tgt_values for token in tokens
        }

        def _bare_satisfied(num: str) -> bool:
            """Whether ``num``'s own (unsigned, unscaled) occurrence is honoured."""
            # Sign: a dropped minus on an unscaled quantity is a changed fact.
            if num in negative_tokens and not _has_negative_token(normalized_tgt, num):
                return False
            if _RANGE_DELIMITERS.search(num):
                sub_parts = [
                    canonicalize_numeric_token(p) for p in _RANGE_DELIMITERS.split(num) if p.strip()
                ]
                if sub_parts and all(p in exempt_numbers for p in sub_parts):
                    return True
            if _has_numeric_token(normalized_tgt, num):
                return True
            # A broken PDF text layer can concatenate a two-page range
            # ("pp. 40–46" -> "pp. 4046"). Accept only when the source has
            # the explicit plural-page cue and the target restores that exact
            # pair as a range.
            if _glued_page_range_is_preserved(original, normalized_tgt, num):
                return True
            # If num is a compound range (e.g. 1984-1985, 10/20), check if all sub-numbers are preserved
            if _RANGE_DELIMITERS.search(num):
                sub_parts = [
                    canonicalize_numeric_token(p) for p in _RANGE_DELIMITERS.split(num) if p.strip()
                ]
                if len(sub_parts) > 1 and all(
                    _has_numeric_token(normalized_tgt, sub) for sub in sub_parts
                ):
                    return True
            # A bare quantity restated at the same value but spelled out shares
            # no digit string, so the textual check above cannot see it.
            if num in tgt_values:
                return True
            # A thousands-dot/three-decimals token ('1.500') is satisfied by
            # either of its two readings, not only the stripped integer form.
            return bool((ambiguous_readings.get(num) or set()) & tgt_values)

        for num in sorted(src_nums):
            if num in exempt_numbers:
                continue
            if num in compound_satisfied:
                # The compound total being restated satisfies only its
                # constituents. The same digits may also occur bare elsewhere
                # in the source; that bare occurrence still needs its own
                # value, so fall through to the bare check (same rule as the
                # scaled path below) instead of waving the key through.
                if num not in bare_numbers or _bare_satisfied(num):
                    continue
                lost_numbers.append(num)
                continue
            # Magnitude: a source quantity written with a scale word ('250万',
            # '2.5 million') must be restated at that magnitude. Bare surviving
            # digits ('250') are not enough — that is a dropped x10^4..10^6.
            scaled = src_scales.get(num)
            if scaled is not None:
                if num in negative_tokens:
                    magnitude_ok = any(_has_negative_token(normalized_tgt, c) for c in scaled)
                else:
                    magnitude_ok = bool(scaled & tgt_values)
                if not magnitude_ok:
                    lost_numbers.append(num)
                    continue
                # The same digit run may also occur bare elsewhere. The
                # magnitude satisfies only the scaled occurrence; the bare one
                # still needs its own value, so fall through to the bare check
                # rather than waving the whole key through.
                if num not in bare_numbers or _bare_satisfied(num):
                    continue
                lost_numbers.append(num)
                continue
            if _bare_satisfied(num):
                continue
            lost_numbers.append(num)

        if lost_numbers:
            return ValidationResult.failure(
                error_code="NUMERIC_INCONSISTENCY",
                message=f"Missing numeric tokens in translation: {lost_numbers}",
                suggested_action="RETRY",
                details={"lost_numbers": lost_numbers, "original_numbers": sorted(src_nums)},
            )

        return ValidationResult.success()


class GlossaryConsistencyValidator(ContentValidator):
    """Verifies that terms appearing in original text are translated consistently."""

    def __init__(self, glossary: list[dict[str, Any]]) -> None:
        # Sort by source length descending to prevent sub-term false misses
        self.glossary = sorted(glossary, key=lambda g: -len(str(g.get("source", ""))))

    def validate(self, original: str | None, translated: str | None) -> ValidationResult:
        # Deferred import, not top-level: ``ubt.core.qe``'s package __init__
        # pulls in comet_runner and fast_pass, both of which import THIS
        # module — a module-level ``from ubt.core.qe.term_drift import ...``
        # would make the validators-first import order (what every test module
        # and most callers use) die on a partially-initialized ImportError. At
        # call time the lookup is a plain sys.modules hit.
        from ubt.core.qe.term_drift import detect_term_drift

        # Detection (source-side [source] + aliases, boundary-aware, folding
        # case on both sides) is single-sourced in term_drift — the same scan
        # evaluate_terms and the span annotator consume — so the quality gate
        # and the consistency planner can no longer disagree about which
        # blocks drifted. detect_term_drift derives the structural spans of
        # each side once and reuses them for every term of the block.
        findings = detect_term_drift(original or "", translated or "", self.glossary)
        missing_terms: list[dict[str, str]] = [
            {"source": finding.source, "expected": finding.expected}
            for finding in findings
            if finding.drifted
        ]

        if missing_terms:
            return ValidationResult.failure(
                error_code="GLOSSARY_DRIFT",
                message=f"Glossary terms drifted: {[m['source'] + ' -> ' + m['expected'] for m in missing_terms]}",
                suggested_action="RETRY",
                details={"missing_terms": missing_terms},
            )

        return ValidationResult.success()
