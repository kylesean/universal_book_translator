"""Inline-math overlay rendering (math-mode pass-through + CJK spacing).

Root cause it fixes (ch3 appendix): docling flattens inline math to bare
text (``F th,SI``), the draft model re-emits it as prose (``Fth,SI`` or
undelimited ``e^{...}``), and :func:`typst_escape` then prints every
``$ \\ { } _ ^`` literally — so translated prose shows ``Fth,SI`` /
``e^{F-F_th}`` where a native layout shows $F_{th,SI}$.

Mainstream loop (Markdown-translation / BabelDOC placeholder paradigm):
delimited math is a non-translatable atom carried through as ``$...$`` and
rendered in Typst math mode. This module is the render half:

- :func:`split_math_spans` — split a line on the :class:`MathMasker`
  placeholder tokens, so math spans (``$$``, ``\\(..\\)``, ``$..$``,
  environments) are located with the masker's own currency guards
  (``$5`` is never math) instead of a second regex dialect.
- :func:`repair_malformed_math_spans` — merge one formula fragmented by a
  stray ``$`` (``$\\psi_B = V_{tm} \\ln($N_ch $= n_i)$``) into a single
  balanced span before fitting, so the probe-gated math path renders it
  instead of degrading to literal LaTeX (chapter-3 ``\\psi_B`` class).
- :func:`strip_cjk_latin_spaces` — MT models emit ASCII spaces at every
  CJK<->Latin boundary (``式 (A.6) 对于``); the overlay disables Typst's
  own CJK-Latin spacing, so these survive verbatim as stutter. Stripped
  on text spans only — math content is byte-identical for the probe cache.
- :func:`render_overlay_line` — text spans are escaped as before; math
  spans are validated by a Typst math compile probe and emitted raw in
  ``$...$``. Anything failing validation (or split across fitter line
  breaks, i.e. unbalanced ``$`` in one line) falls back to stripped
  escaped text — never worse than the old literal output.

All functions are pure except the injected probe (defaults to
escape-everything, i.e. the historical behavior, when no probe is given).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Sequence

from ubt.adapters.pdf.textgeom import SUPERSCRIPT_DECODE_MAP
from ubt.adapters.pdf.typst_symbols import INLINE_SYMBOLS
from ubt.core.cjk_ranges import HAN_UNIFIED_CLASS, HAN_UNIFIED_RANGES
from ubt.core.cleaners.cjk_spacing import apply_pangu_spacing, normalize_publishing_cjk
from ubt.core.cleaners.html_sanitizer import strip_html_mark_tags
from ubt.core.cleaners.math_masker import MathMasker

_CJK = r"一-鿿㐀-䶿豈-﫿぀-ヿ가-힯"


_LINE_START_MARKUP_RE = re.compile(r"(?m)^([ \t]*)(=|\+|-|/|\d+\.)(?=\s|$)")


def escape_line_start_markup(text: str) -> str:
    """Escape Typst block markers that are special only at the start of a line.

    ``=``/``+``/``-``/``/``/``N.`` open a heading/list/enum/term-list in Typst
    *only* at line start, so they are deliberately absent from the mid-line
    escape sets (escaping them everywhere would litter ordinary prose and
    maths). A translated paragraph can still carry a newline immediately before
    such a marker, which silently re-lays-out the page as a heading or list (or
    fails the whole compile, as a ``/`` term marker does); escape the marker
    (and, for ``N.``, its dot) so it renders as literal text.
    """

    # The marker must be followed by whitespace or end-of-line: ``12.5`` is a
    # decimal, not an enumerated-list item, and must be left untouched.
    def _repl(m: re.Match[str]) -> str:
        lead, marker = m.group(1), m.group(2)
        if marker.endswith("."):
            return f"{lead}{marker[:-1]}\\."
        return f"{lead}\\{marker}"

    return _LINE_START_MARKUP_RE.sub(_repl, text)


def typst_escape(text: str) -> str:
    text = strip_html_mark_tags(text)
    out: list[str] = []
    for ch in text:
        # `[` / `]` are structural (they open/close content blocks): a chunk
        # split mid-span (`[26]` -> `[2` | `6]`, chapter-3 p2/p17) would
        # otherwise unbalance the statement, swallowing every following
        # statement on the page ~200pt away (A.1 garble, eq-3.1 intrusion).
        if ch in ("\\", "{", "}", "$", "#", "@", "<", ">", "`", "*", "_", "[", "]", "~"):
            out.append("\\" + ch)
        elif ch == '"':
            out.append('\\"')
        else:
            out.append(ch)
    # `//` opens a Typst line comment even inside
    # [#text[...]] content, swallowing the closing brackets and failing the
    # whole overlay compile. A zero-width space breaks the comment token
    # while rendering invisibly (URLs stay copyable modulo one ZWSP).
    escaped = escape_line_start_markup("".join(out))
    return escaped.replace("//", "/\u200b/")


_BARE_NUMBER_RE = re.compile(r"^\d[\d,.]*$")

# A stray ``$`` is only a fitter-split math fragment (safe to drop) when the
# line carries LaTeX markers; a lone currency ``$5`` must survive verbatim.
_MATHY_RE = re.compile(r"[\\_{}^\u221a\u222b\u2211\u220f\u2202\u221e\u2208]")


def strip_cjk_latin_spaces(text: str, target_lang: str = "zh") -> str:
    """Normalize CJK spacing conforming to Chinese typography guidelines (Pangu spacing).

    Maintains single half-width spaces between CJK characters and Latin/digits/math,
    while removing stray spaces between CJK characters or adjacent to CJK punctuation.
    """
    if not text:
        return text
    return normalize_publishing_cjk(text, target_lang=target_lang)


def split_math_spans(text: str) -> list[tuple[bool, str]]:
    """Split text into ``(is_math, content)`` spans on MathMasker tokens.

    The masker locates ``$$…$$``, ``\\\\(...\\\\)``, ``$…$`` and LaTeX
    environments with its own currency guards; math-span content keeps its
    original delimiters. Text without spans returns a single text part.
    """
    if "$" not in text and "\\(" not in text and "\\begin{" not in text:
        return [(False, text)]
    masker = MathMasker()
    masked, mapping = masker.mask(text)
    if not mapping:
        return [(False, text)]
    token_re = re.compile("(" + "|".join(re.escape(t) for t in mapping) + ")")
    parts: list[tuple[bool, str]] = []
    for chunk in token_re.split(masked):
        if not chunk:
            continue
        if chunk in mapping:
            parts.append((True, mapping[chunk]))
        else:
            parts.append((False, chunk))
    return parts


def math_span_extents(text: str) -> list[tuple[int, int]]:
    """Char offsets of ``$...$`` math spans, end-exclusive, in order.

    Cumulative offsets over :func:`split_math_spans` parts (masking is a
    pure substitution, so concatenation reconstructs the input exactly).
    The fitter uses these for span-atomic flow: breaking inside one of
    these ranges leaves unbalanced ``$`` per overlay line, which degrades
    to literal backslash text downstream (chapter-1 ``$T_{\\text{si}}``
    class).
    """

    extents: list[tuple[int, int]] = []
    pos = 0
    for is_math, content in split_math_spans(text or ""):
        end = pos + len(content)
        if is_math:
            extents.append((pos, end))
        pos = end
    return extents


# -- malformed-span repair -------------------------------------------------
# Models occasionally fragment a single formula with a stray ``$``
# (``$\psi_B = V_{tm} \ln($N_ch $= n_i)$``). Each fragment then fails the
# Typst math probe on its unclosed bracket and the literal fallback prints
# the LaTeX. Merging the fragments back into one span *before* the fitter
# measures keeps extents and render consistent (fitter and probe cache see
# the same merged body).
_MERGE_GAP_MAX = 40
_MERGE_BREAK_RE = re.compile(r"[。．.!！?？;；\n]")
_MERGE_CJK_RE = re.compile(f"[{_CJK}]")


def _math_balance(body: str) -> int:
    """Unclosed ``(``/``[`` count; escaped LaTeX chars never count."""
    clean = re.sub(r"\\.", "", body)
    return clean.count("(") + clean.count("[") - clean.count(")") - clean.count("]")


def _mergeable_gap(text: str) -> bool:
    """Latin-only glue short enough to belong to one formula."""
    return (
        bool(text)
        and len(text) <= _MERGE_GAP_MAX
        and not (_MERGE_BREAK_RE.search(text) or _MERGE_CJK_RE.search(text))
    )


def _merge_candidate_ok(candidate: str) -> bool:
    """Balanced, delimiter-free and convertible to Typst math."""
    return "$" not in candidate and typstify_math(candidate) is not None


def _merge_spans(
    parts: list[tuple[bool, str]], lo: int, hi: int, body: str
) -> list[tuple[bool, str]]:
    return parts[:lo] + [(True, "$" + body + "$")] + parts[hi + 1 :]


def _repair_once(parts: list[tuple[bool, str]]) -> list[tuple[bool, str]] | None:
    """One left-to-right repair pass; rewritten parts or None."""
    for i, (is_math, content) in enumerate(parts):
        if not is_math:
            continue
        body = _math_inner(content)
        if body is None or _math_balance(body) == 0:
            continue
        # Forward: this fragment closes a bracket an earlier span opened.
        j = i + 1
        gap = ""
        while j < len(parts):
            is_m, chunk = parts[j]
            if is_m:
                nxt = _math_inner(chunk)
                if nxt is None:
                    break
                candidate = body + gap + nxt
                balance = _math_balance(candidate)
                if balance == 0 and _merge_candidate_ok(candidate):
                    return _merge_spans(parts, i, j, candidate)
                if balance < 0:
                    break
                body = candidate
                # The gap is now part of ``body``; keeping it would re-insert it
                # before the next fragment and duplicate the text.
                gap = ""
            elif _mergeable_gap(chunk):
                gap += chunk
            else:
                break
            j += 1
        # Backward: this fragment starts with a closer, opener precedes it.
        j = i - 1
        gap = ""
        while j >= 0:
            is_m, chunk = parts[j]
            if is_m:
                prev = _math_inner(chunk)
                if prev is None:
                    break
                candidate = prev + gap + body
                balance = _math_balance(candidate)
                if balance == 0 and _merge_candidate_ok(candidate):
                    return _merge_spans(parts, j, i, candidate)
                if balance > 0:
                    break
                body = candidate
                gap = ""
            elif _mergeable_gap(chunk):
                gap = chunk + gap
            else:
                break
            j -= 1
    return None


def repair_malformed_math_spans(text: str) -> str:
    """Merge math fragments split by a stray ``$`` into balanced spans.

    Only fires when a span's own ``(``/``[`` balance is nonzero and the
    merged body zeroes it across a short latin gap — sentence punctuation,
    CJK and oversized gaps veto the merge, so legitimately adjacent spans
    and currency dollars are never touched. A merge that still fails Typst
    conversion is rejected, keeping today's fail-closed literal path.
    """
    if "$" not in text:
        return text
    parts = split_math_spans(text)
    if not any(is_math for is_math, _ in parts):
        return text
    while True:
        repaired = _repair_once(parts)
        if repaired is None:
            return "".join(content for _, content in parts)
        parts = repaired


def _math_inner(content: str) -> str | None:
    """Unwrap one math span to its Typst-math body; None when not emittable."""
    body = content.strip()
    for opener, closer in (("$$", "$$"), ("\\[", "\\]"), ("\\(", "\\)")):
        if (
            body.startswith(opener)
            and body.endswith(closer)
            and len(body) > len(opener) + len(closer)
        ):
            body = body[len(opener) : -len(closer)].strip()
            break
    else:
        if body.startswith("$") and body.endswith("$") and len(body) > 2:
            body = body[1:-1].strip()
        elif body.startswith("\\begin{"):
            pass
        else:
            return None
    if not body or _BARE_NUMBER_RE.match(body):
        return None
    if body.count("$") % 2:
        return None
    return body


# LaTeX -> Typst math dialect map (Typst symbols take no backslash).
# Commands outside this table fail closed to escaped text (probe decides).
_LATEX_CMD_RE = re.compile(r"\\([a-zA-Z]+)")
_LATEX_CMD_MAP = INLINE_SYMBOLS
# Converted/builtin Typst math identifiers that must NOT be quoted as text.
_TYPST_MATH_WORDS = frozenset(_LATEX_CMD_MAP.values()) | {
    "bb",
    "frak",
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "sinh",
    "cosh",
    "tanh",
    "log",
    "ln",
    "lg",
    "exp",
    "min",
    "max",
    "sup",
    "inf",
    "lim",
    "liminf",
    "limsup",
    "det",
    "dim",
    "gcd",
    "lcm",
    "arg",
    "mod",
    "sqrt",
    "frac",
    "sum",
    "product",
    "integral",
    "partial",
    "cal",
    "hat",
    "tilde",
    "vec",
    "overline",
    "dot",
    "ddot",
    "diaer",
}
# Word runs for the upright-label quoter. Dotted Typst identifiers
# (plus.minus, lt.eq, lt.double from the map above) must match as ONE
# token: quoting "lt"/"double" separately emits #"lt".#"double", which
# compiles yet renders the literal text "ltdouble" (avoids literal concatenation
# of dotted Typst identifiers like \pm \leq \geq \neq). Single letters
# never match (math-italic variables stay bare); a trailing dot never
# joins (letter required after each dot), so "word." still splits.
_WORD_RUN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:\.[A-Za-z][A-Za-z0-9]*)+|[A-Za-z][A-Za-z0-9]+")
_FONT_STYLE_CMD_RE = re.compile(r"\\(mathbb|mathfrak|mathcal)\{([^{}]*)\}")
_DOUBLE_BACKSLASH_CMD_RE = re.compile(r"\\\\([a-zA-Z]+)")


# \text{...} / \mathrm{...} mean upright roman — exactly what the #"..."
# code strings below render (SI-correct label style). Rewriting them to
# the marked-string form keeps models' favourite wrappers compilable
# ($I_{\text{off}}$ for source "Ioff" class, $2\mathrm{D}$ for "(2D)"
# class) instead of failing closed to literal backslash garbage. Nested
# braces do not match and stay fail-closed. Bold/italic wrappers
# (\mathbf, \mathit, ...) are deliberately NOT mapped: upright would be
# a silent styling lie, so those still refuse.
_TEXTLIKE_CMD_RE = re.compile(r"\\(?:text|mathrm)\{([^{}]*)\}")
_LATEX_SPACES_RE = re.compile(
    r"\\(?:quad|qquad|enspace|thinspace|thickspace|medspace|negthinspace|[,;! ])"
)
#: Delimiter sizing/auto-sizing modifiers that carry no glyph of their own.
#: The negative lookahead stops ``\right`` from eating the head of a real
#: command (``\rightleftharpoons``, ``\leftrightarrow``), and ``[lmr]?``
#: consumes the ``\bigl``/``\bigr``/``\bigm`` variants before that check.
_LATEX_DELIM_SIZE_RE = re.compile(r"\\(?:bigg|Bigg|big|Big|left|right)(?:[lmr])?(?![a-zA-Z])")


def typstify_math_span(content: str) -> str | None:
    """Convert one delimited math span to a Typst math body, or ``None``.

    The single entry point for "span in, body out" so the renderer and the
    probe prefetch agree on exactly which bodies reach :func:`typstify_math`.
    """
    body = _math_inner(content)
    return typstify_math(body) if body is not None else None


def _styled_unicode_to_typst(ch: str) -> str | None:
    """Map a single Unicode Script/Fraktur/Double-Struck symbol to its Typst math call."""
    cp = ord(ch)
    if 0x1D49C <= cp <= 0x1D503 or ch in "ℬℰℱℋℐℒℳℛℯℊℴ":
        base = unicodedata.normalize("NFKC", ch)
        return f"cal({base})"
    if 0x1D504 <= cp <= 0x1D537 or 0x1D56C <= cp <= 0x1D59F or ch in "ℜℑℌℭℨ":
        base = unicodedata.normalize("NFKC", ch)
        return f"frak({base})"
    if 0x1D538 <= cp <= 0x1D56B or ch in "ℝℕℤℚℂ":
        base = unicodedata.normalize("NFKC", ch)
        return f"bb({base})"
    return None


def _expand_styled_unicode_in_math(text: str) -> str:
    """Replace Unicode Script/Fraktur/Double-Struck chars with cal()/frak()/bb() before NFKC."""
    cleaned = text.replace("\ufe00", "").replace("\ufe01", "")
    out: list[str] = []
    for i, ch in enumerate(cleaned):
        rep = _styled_unicode_to_typst(ch)
        if rep is not None:
            need_post_space = i + 1 < len(cleaned) and (
                cleaned[i + 1].isalnum() or "\u0370" <= cleaned[i + 1] <= "\u03ff"
            )
            out.append(rep + (" " if need_post_space else ""))
        else:
            out.append(ch)
    return "".join(out)


def _wrap_prose_styled_unicode(text: str) -> str:
    """Wrap bare Unicode Script/Fraktur/Double-Struck symbols in prose into $cal(...)$ spans."""
    if not text:
        return text
    parts = split_math_spans(text)
    out: list[str] = []
    for is_math, content in parts:
        if is_math:
            out.append(content)
            continue
        cleaned = content.replace("\ufe00", "").replace("\ufe01", "")
        buf: list[str] = []
        for ch in cleaned:
            rep = _styled_unicode_to_typst(ch)
            if rep is not None:
                buf.append(f"${ch}$")
            else:
                buf.append(ch)
        out.append("".join(buf))
    return "".join(out)


def typstify_math(body: str) -> str | None:
    """Translate a LaTeX inline-math body to compilable Typst math."""
    if not body:
        return None

    # Convert Script (𝒞 -> cal(C)), Fraktur (𝔈 -> frak(E)), and Double-Struck (ℝ -> bb(R))
    # BEFORE NFKC, because NFKC would flatten them to plain ASCII Latin letters (C, E, R).
    body = _expand_styled_unicode_in_math(body)
    body = unicodedata.normalize("NFKC", body).strip()
    if "#" in body or '"' in body or "//" in body.replace(" ", ""):
        return None
    body = _DOUBLE_BACKSLASH_CMD_RE.sub(r"\\\1", body)
    body = body.replace("≔", ":=").replace("⊧", " models ")
    # Separate adjacent Greek characters (e.g. σγ from 𝜎𝛾) so Typst doesn't
    # tokenize them as a single unknown identifier.
    body = re.sub(r"(?<=[\u0370-\u03ff])(?=[\u0370-\u03ffA-Za-z])", " ", body)
    body = re.sub(r"(?<=[A-Za-z])(?=[\u0370-\u03ff])", " ", body)
    # Bare superscripts/subscripts (e.g. $^{-3}$) attach to an empty string in Typst
    if body.startswith(("^", "_")):
        body = '""' + body

    # LaTeX explicit spacing and delimiter sizing modifiers
    body = _LATEX_SPACES_RE.sub(" ", body)
    body = _LATEX_DELIM_SIZE_RE.sub("", body)
    # Typst spells these math alphabets as function calls rather than LaTeX
    # commands. Leave syntax validation to the caller's TypstMathProbe.
    _font_fn = {"mathbb": "bb", "mathfrak": "frak", "mathcal": "cal"}
    body = _FONT_STYLE_CMD_RE.sub(lambda m: f"{_font_fn[m.group(1)]}({m.group(2)})", body)

    # Convert \frac{a}{b} -> frac(a, b) and \sqrt{a} -> sqrt(a)
    for _ in range(3):
        t1 = _FRAC_RE.sub(r"frac(\1, \2)", body)
        t2 = _SQRT_RE.sub(r"sqrt(\1)", t1)
        if t2 == body:
            break
        body = t2

    # Convert math accents \hat{a} -> hat(a), \hat a -> hat(a), etc.
    for _ in range(3):
        t1 = _ACCENT_CMD_RE.sub(
            lambda m: f"{_ACCENT_FN_MAP.get(m.group(1), m.group(1))}({m.group(2)})", body
        )
        t2 = _ACCENT_BARE_CMD_RE.sub(
            lambda m: f"{_ACCENT_FN_MAP.get(m.group(1), m.group(1))}({m.group(2)})", t1
        )
        if t2 == body:
            break
        body = t2

    marked = _TEXTLIKE_CMD_RE.sub(lambda m: "\0" + m.group(1) + "\0", body)

    def _cmd(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in _LATEX_CMD_MAP:
            rep = _LATEX_CMD_MAP[name]
            start = match.start()
            end = match.end()
            pre = (
                " "
                if (start > 0 and (marked[start - 1].isalnum() or marked[start - 1] == "\0"))
                else ""
            )
            post = (
                " "
                if (end < len(marked) and (marked[end].isalnum() or marked[end] == "\0"))
                else ""
            )
            return f"{pre}\0{rep}\0{post}"
        raise _UnknownCommand(name)

    try:
        marked = _LATEX_CMD_RE.sub(_cmd, marked)
    except _UnknownCommand:
        return None

    def _quote(match: re.Match[str]) -> str:
        word = match.group(0)
        if word in _TYPST_MATH_WORDS:
            return word
        return f'#"{word}"'

    quoted = _WORD_RUN_RE.sub(_quote, marked).replace("\0", "")
    # Bare strings choke inside ``_{...}`` groups (``A_{#"g0"}`` renders
    # literally); a pure-label group (strings + separators only, no math)
    # collapses to one code string (``A_#"g0"``), which parses cleanly.
    res = _LABEL_GROUP_RE.sub(_collapse_label_group, quoted)
    # Avoid #"foo"(...) being parsed as a Typst code function call
    return re.sub(r'(#"[^"]*")(?=\()', r"\1 ", res)


_LABEL_GROUP_RE = re.compile(r'([_^])\{((?:[^#"{}]|#"[^"]*")*)\}')
_LABEL_SEP_RE = re.compile(r"^[\s,;:|/\-\u00b7\d]*$")
_STRING_RE = re.compile(r'#"([^"]*)"')
_FRAC_RE = re.compile(r"\\frac\{([^{}]*)\}\{([^{}]*)\}")
_SQRT_RE = re.compile(r"\\sqrt\{([^{}]*)\}")
_ACCENT_CMD_RE = re.compile(r"\\(hat|tilde|vec|bar|overline|dot|ddot)\{([^{}]*)\}")
_ACCENT_BARE_CMD_RE = re.compile(r"\\(hat|tilde|vec|bar|overline|dot|ddot)\s+([A-Za-z0-9])")
_ACCENT_FN_MAP: dict[str, str] = {"bar": "overline", "ddot": "diaer"}


def _collapse_label_group(match: re.Match[str]) -> str:
    script, inner = match.group(1), match.group(2)
    if _STRING_RE.search(inner) is None:
        return script + "(" + inner + ")"
    if _LABEL_SEP_RE.fullmatch(_STRING_RE.sub("", inner)) is None:
        return script + "(" + inner + ")"
    merged = _STRING_RE.sub(lambda m: m.group(1), inner)
    return script + "#" + '"' + merged + '"'


class _UnknownCommand(Exception):
    pass


_UNICODE_SUPER_RUN_RE = re.compile(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾˒]+")


def restore_zone_superscripts(text: str, zone_rows: Sequence[str]) -> str:
    """Restore geometric prefix/suffix superscripts from zone.rows onto target text."""
    if not text or not zone_rows:
        return text
    joined_rows = " ".join(r for r in zone_rows if r)
    if not _UNICODE_SUPER_RUN_RE.search(joined_rows):
        return text

    out = text
    # 1. Prefix superscripts (e.g. '¹Peking University' or ' ²DeepSeek-AI' or '¹Data retrieved...')
    prefix_supers = re.findall(
        rf"(?:^|(?<=\s))([⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾˒]+)(?=[A-Za-z{HAN_UNIFIED_CLASS}])", joined_rows
    )
    for sup_tok in prefix_supers:
        ascii_tok = sup_tok.translate(SUPERSCRIPT_DECODE_MAP)
        if not ascii_tok:
            continue
        pat = re.compile(rf"(^|(?<=\s)){re.escape(ascii_tok)}\s+(?=[A-Za-z{HAN_UNIFIED_CLASS}])")
        out = pat.sub(rf"\g<1>{sup_tok}", out, count=1)

    # 2. Suffix superscripts (e.g. 'Yifan Shi¹˒²', 'Wei Zhang¹', 'executable code¹', 'extensions.¹')
    suffix_supers = re.findall(
        rf"(?<=[A-Za-z0-9{HAN_UNIFIED_CLASS}.])[ \t]*([⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾˒]+)", joined_rows
    )
    for sup_tok in suffix_supers:
        ascii_tok = sup_tok.translate(SUPERSCRIPT_DECODE_MAP)
        if not ascii_tok:
            continue
        pat = re.compile(
            rf"(?:(?<=[A-Za-z{HAN_UNIFIED_CLASS}])[ \t]+|(?<=[。．.）)])[ \t]*){re.escape(ascii_tok)}(?=\s|[，,。.;；:：!！?？{HAN_UNIFIED_CLASS}]|$)"
        )
        out = pat.sub(sup_tok, out, count=1)

    # 3. Restore Latin comma + quad spacing after author-name superscripts (e.g. 'Yifan Shi¹˒²，Wei Zhang¹')
    out = re.sub(r"([⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾˒]+)\s*[，,]\s*(?=[A-Z])", r"\1,  ", out)
    return out


def _emit_typst_superscripts(escaped_text: str) -> str:
    """Convert Unicode superscript runs into Typst native #super[...] blocks with quad spacing for mid-line prefix affiliations."""
    if not _UNICODE_SUPER_RUN_RE.search(escaped_text):
        return escaped_text

    def _repl(m: re.Match[str]) -> str:
        raw_ascii = m.group(0).translate(SUPERSCRIPT_DECODE_MAP)
        sup_node = f"#super(typographic: false, size: 0.62em)[{typst_escape(raw_ascii)}]"
        start, end = m.start(), m.end()
        # Mid-line prefix superscript (preceded by whitespace after a token, and immediately followed by a word/CJK char)
        # represents a multi-institution separator (LaTeX \quad, e.g. '¹北京大学  ²DeepSeek-AI')
        if (
            start > 0
            and escaped_text[start - 1].isspace()
            and end < len(escaped_text)
            and (
                escaped_text[end].isalnum()
                or any(lo <= ord(escaped_text[end]) <= hi for lo, hi in HAN_UNIFIED_RANGES)
            )
        ):
            return f"#h(1.2em){sup_node}"
        return sup_node

    # Also preserve double-space after comma in author lists as #h(0.5em)
    res = _UNICODE_SUPER_RUN_RE.sub(_repl, escaped_text)
    res = re.sub(r"(\]),  (?=[A-Z])", r"\1, #h(0.5em)", res)
    return res


def _wrap_style(
    segment: str, bold: bool, italic: bool, superscript: bool, color_hex: str | None
) -> str:
    """Wrap an escaped segment in the Typst markup its source style calls for.

    Applied after escaping, so the injected markup is never escaped. Superscript
    reuses the same ``#super`` shape :func:`_emit_typst_superscripts` emits, so a
    geometrically-raised dagger and a Unicode superscript render alike.
    """
    if superscript:
        segment = f"#super(typographic: false, size: 0.62em)[{segment}]"
    if italic:
        segment = f"#emph[{segment}]"
    if bold:
        segment = f"#strong[{segment}]"
    if color_hex:
        segment = f'#text(fill: rgb("{color_hex}"))[{segment}]'
    return segment


def render_overlay_line(
    line: str,
    math_probe: Callable[[str], bool] | None = None,
    target_lang: str = "zh",
    run_spans: Sequence[tuple[int, int, bool, bool, bool, str | None]] = (),
) -> str:
    """Render one fitter-flowed line to Typst source.

    ``run_spans`` are character ranges of ``line`` to wrap in style markup
    (colour / superscript / weight); a range overlapping a math span is dropped in
    favour of the math rendering. They are applied after escaping, so the markup
    survives.
    """
    if math_probe is not None:
        line = _wrap_prose_styled_unicode(line)
    math_spans: list[tuple[int, int]] = []
    offset = 0
    for is_math, content in split_math_spans(line):
        if is_math:
            math_spans.append((offset, offset + len(content)))
        offset += len(content)
    boundaries: set[int] = {0, len(line)}
    for start, end in math_spans:
        boundaries.update((start, end))
    for span in run_spans:
        boundaries.update((span[0], span[1]))
    ordered = sorted(bound for bound in boundaries if 0 <= bound <= len(line))
    out: list[str] = []
    for start, end in zip(ordered, ordered[1:], strict=False):
        if end <= start:
            continue
        content = line[start:end]
        if any(s <= start and end <= e for s, e in math_spans):
            typst = typstify_math_span(content)
            if typst is not None and math_probe is not None and math_probe(typst):
                out.append(f"${typst}$")
            else:
                fallback = content.replace("$", "")
                escaped = typst_escape(strip_cjk_latin_spaces(fallback, target_lang=target_lang))
                out.append(_emit_typst_superscripts(escaped))
            continue
        if content.count("$") % 2 and _MATHY_RE.search(content):
            content = content.replace("$", "")
        escaped = typst_escape(strip_cjk_latin_spaces(content, target_lang=target_lang))
        rendered = _emit_typst_superscripts(escaped)
        style = next((s for s in run_spans if s[0] <= start and end <= s[1]), None)
        if style is not None:
            rendered = _wrap_style(rendered, style[2], style[3], style[4], style[5])
        out.append(rendered)
    return _break_run_call_chains(out)


def _break_run_call_chains(segments: Sequence[str]) -> str:
    """Join segments without letting a styled run absorb a following ``(``.

    A styled run ends in ``]`` (``#strong[...]``). Typst parses a following ``(``
    as another call on that run's *result* — ``#strong[x](y)`` calls the content
    ``x`` as a function — so a citation's opening paren right after a bold run
    (``#strong[Firecracker microVMs ](Agache et al., 2020)``) fails to compile
    and the whole fragment descends to the source. Emit that one literal ``(`` as
    an explicit ``#text("(")``, which renders identically and ends the code
    expression cleanly. A literal ``[`` needs no such guard: ``typst_escape``
    already backslash-escapes it.
    """
    merged: list[str] = []
    for segment in segments:
        if merged and merged[-1].endswith("]") and segment[:1] == "(":
            merged.append('#text("(")')
            merged.append(segment[1:])
        else:
            merged.append(segment)
    return "".join(merged)


def prepare_overlay_text(raw: str, target_lang: str = "zh") -> str:
    """Pre-fitter normalization: repair math, then Pangu spacing on text spans."""
    raw = _wrap_prose_styled_unicode(raw)
    raw = repair_malformed_math_spans(raw)
    raw = apply_pangu_spacing(raw, target_lang=target_lang)
    parts = split_math_spans(raw)
    out: list[str] = []
    for is_math, content in parts:
        if is_math:
            out.append(content)
            continue
        out.append(strip_cjk_latin_spaces(content, target_lang=target_lang))
    return "".join(out)
