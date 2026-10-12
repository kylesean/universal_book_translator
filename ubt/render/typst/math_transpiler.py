r"""LaTeX math → Typst math transpiler.

The semantic reflow renderer needs this transform, and
``ubt/render/table_compiler.py`` needs it too, so it lives in ``render`` rather
than in the PDF adapter (``render`` no longer reaches into ``adapters/pdf`` for
math).

Two entry points:

``typstify_math(body)``
    A LaTeX *math body* (no delimiters) → a Typst math body, or ``None`` when
    the input cannot be emitted safely. ``None`` is a deliberate fail-closed
    answer, not an error: the caller falls back to keeping the source span.
``typstify_math_span(content)``
    A *delimited* span (``$...$``, ``$$...$$``, ``\(...\)``, ``\[...\]``) →
    the body, or ``None``.

The module is pure: ``re``, ``unicodedata`` and the shared symbol table only.
"""

from __future__ import annotations

import re
import unicodedata

from ubt.adapters.pdf.typst_symbols import INLINE_SYMBOLS

#: A bare number is not math worth converting (``$5`` is currency).
_BARE_NUMBER_RE = re.compile(r"^\d[\d,.]*$")


def typstify_math_span(content: str) -> str | None:
    """Convert one delimited math span to a Typst math body, or ``None``.

    The single entry point for "span in, body out" so the renderer and the
    probe prefetch agree on exactly which bodies reach :func:`typstify_math`.
    """
    body = _math_inner(content)
    return typstify_math(body) if body is not None else None


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
    # Emitted by the font-style mapping for ``\mathbf``/``\mathit``. Without
    # these the word-run quoter below turns them into literal text
    # (``#"bold" (W)``), so the formula renders the *name* of the function
    # instead of a bold variable.
    "bold",
    "italic",
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
_FONT_STYLE_CMD_RE = re.compile(
    r"\\(mathbb|mathfrak|mathcal|mathbf|mathit|boldsymbol|bm)\{([^{}]*)\}"
)
_DOUBLE_BACKSLASH_CMD_RE = re.compile(r"\\\\([a-zA-Z]+)")


# \text{...} / \mathrm{...} mean upright roman — exactly what the #"..."
# code strings below render (SI-correct label style). Rewriting them to
# the marked-string form keeps models' favourite wrappers compilable
# ($I_{\text{off}}$ for source "Ioff" class, $2\mathrm{D}$ for "(2D)"
# class) instead of failing closed to literal backslash garbage. Nested
# braces do not match and stay fail-closed.
#
# Bold/italic are real styling requests, not labels, so they are mapped to
# Typst's own math commands (``bold``/``italic``) rather than dropped. Leaving
# them unmapped refused the **whole formula**: measured, ``$\mathbf{W} x + b$``
# rendered as "W x + b" — the math layout was lost for a formula whose only
# unusual feature was a bold matrix, and ``\mathbf`` is among the most common
# commands in machine-learning papers.
_TEXTLIKE_CMD_RE = re.compile(r"\\(?:text|mathrm)\{([^{}]*)\}")
_LATEX_SPACES_RE = re.compile(
    r"\\(?:quad|qquad|enspace|thinspace|thickspace|medspace|negthinspace|[,;! ])"
)
#: Delimiter sizing/auto-sizing modifiers that carry no glyph of their own.
#: The negative lookahead stops ``\right`` from eating the head of a real
#: command (``\rightleftharpoons``, ``\leftrightarrow``), and ``[lmr]?``
#: consumes the ``\bigl``/``\bigr``/``\bigm`` variants before that check.
_LATEX_DELIM_SIZE_RE = re.compile(r"\\(?:bigg|Bigg|big|Big|left|right)(?:[lmr])?(?![a-zA-Z])")

#: LaTeX permits whitespace between a control word and its brace argument
#: (``\frac {a}{b}``); every regex below requires them adjacent. Docling emits
#: the spaced form for **every** command, so without this normalization the
#: whole transpiler fails closed: measured on arXiv 2101.03961, 6 of 7 display
#: formulas returned ``None`` (only the one with no command argument survived).
#: ``\cmd {`` appears 15 times in that paper's formulas alone.
_LATEX_CMD_SPACE_RE = re.compile(r"\\([A-Za-z]+)\s+\{")

#: LaTeX operator commands that have no entry in the symbol table because Typst
#: spells them as plain identifiers. Without them the unknown-command guard
#: fails the *whole* formula closed over one unrecognized name: ``\arg`` alone
#: cost 2101.03961's fifth formula. ``\argmax``/``\argmin`` are already in the
#: table; these are the spellings that are not.
_LATEX_OPERATOR_RESCUE: dict[str, str] = {
    "arg": "arg",
    "operatorname": "",  # \operatorname{foo} -> the bare name, handled below
    "mathop": "",
    "bmod": "mod",
    "pmod": "mod",
    "limits": "",
    "nolimits": "",
    "displaystyle": "",
    "textstyle": "",
    "scriptstyle": "",
}
#: NOTE: ``\mathrm`` is deliberately NOT in the table above. It is handled by
#: ``_TEXTLIKE_CMD_RE``, which preserves its argument as an upright label; a
#: rescue entry would strip the command first and leave a bare ``{...}`` group.


def _normalize_latex_spacing(body: str) -> str:
    """Make Docling's spaced LaTeX canonical so the patterns below can match.

    Three normalizations, all of them no-ops on canonical LaTeX:

    1. ``\\cmd {`` → ``\\cmd{`` — the load-bearing one (see
       ``_LATEX_CMD_SPACE_RE``).
    2. ``{ a }`` → ``{a}`` — padding inside a group. Docling pads every group,
       and ``\\frac { a } { b }`` otherwise yields ``frac( a ,  b )``.
    3. ``\\cmd   x`` → ``\\cmd x`` — runs of whitespace collapse to one, so
       the downstream spacing heuristics see a single separator.
    """
    body = _LATEX_CMD_SPACE_RE.sub(r"\\\1{", body)
    body = re.sub(r"\{\s+", "{", body)
    body = re.sub(r"\s+\}", "}", body)
    body = re.sub(r"[ \t]{2,}", " ", body)
    return body


def _rescue_operators(body: str) -> str:
    """Replace operator commands Typst spells as bare identifiers.

    ``\\arg`` → ``arg``, ``\\operatorname{rank}`` → ``rank``, and the spacing
    hints (``\\limits``, ``\\displaystyle``) drop out entirely. Applied only to
    commands absent from the symbol table, so it can never shadow a real
    mapping.
    """
    for name, replacement in _LATEX_OPERATOR_RESCUE.items():
        if name == "operatorname":
            body = re.sub(r"\\operatorname\{([^{}]*)\}", r"\1", body)
            continue
        body = re.sub(rf"\\{name}\b", replacement, body)
    return body


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


def _extract_braced_chunk(text: str, start: int) -> tuple[str, int] | None:
    """Extract a balanced `{...}` chunk starting at text[start]."""
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i], i + 1
    return None


def _convert_binoms(text: str) -> str:
    r"""Convert LaTeX ``\binom{n}{k}`` to Typst ``binom(n, k)``.

    Two brace arguments, like ``\frac``, so it needs its own pass: the command
    table only maps names to glyphs and cannot consume arguments. A formula
    containing one previously failed closed (3 on the gold corpus).
    """
    out: list[str] = []
    i = 0
    while i < len(text):
        if text.startswith(r"\binom", i) and (i + 6 == len(text) or not text[i + 6].isalpha()):
            pos = i + 6
            while pos < len(text) and text[pos].isspace():
                pos += 1
            top_res = _extract_braced_chunk(text, pos)
            if top_res is not None:
                top, after_top = top_res
                while after_top < len(text) and text[after_top].isspace():
                    after_top += 1
                bot_res = _extract_braced_chunk(text, after_top)
                if bot_res is not None:
                    bot, after_bot = bot_res
                    if out and (out[-1].isalnum() or out[-1] in "_^"):
                        out.append(" ")
                    out.append(f"binom({_convert_binoms(top)}, {_convert_binoms(bot)})")
                    i = after_bot
                    continue
        out.append(text[i])
        i += 1
    return "".join(out)


def _convert_fractions(text: str) -> str:
    """Convert LaTeX \\frac{num}{den} to Typst frac(num, den) with balanced brace support.

    A separator is emitted before the converted call when the preceding
    character is a script marker or a word character: without it
    ``a_b\\frac{1}{2}`` became ``a_bfrac(1, 2)`` and the word-run quoter read
    ``bfrac`` as one word (the same defect as :func:`_convert_sqrts`).
    """
    out: list[str] = []
    i = 0
    while i < len(text):
        if text.startswith(r"\frac", i) and (i + 5 == len(text) or not text[i + 5].isalpha()):
            pos = i + 5
            while pos < len(text) and text[pos].isspace():
                pos += 1
            num_res = _extract_braced_chunk(text, pos)
            if num_res is not None:
                num_content, after_num = num_res
                while after_num < len(text) and text[after_num].isspace():
                    after_num += 1
                den_res = _extract_braced_chunk(text, after_num)
                if den_res is not None:
                    den_content, after_den = den_res
                    rec_num = _convert_fractions(num_content)
                    rec_den = _convert_fractions(den_content)
                    if out and (out[-1].isalnum() or out[-1] in "_^"):
                        out.append(" ")
                    out.append(f"frac({rec_num}, {rec_den})")
                    i = after_den
                    continue
        out.append(text[i])
        i += 1
    return "".join(out)


def _convert_sqrts(text: str) -> str:
    """Convert LaTeX \\sqrt{arg} to Typst sqrt(arg) with balanced brace support.

    A separator is emitted before the converted call when the preceding
    character is a script marker or a word character. Without it,
    ``\\beta_t\\sqrt{x}`` became ``beta_tsqrt(x)`` — the subscript's ``t``
    merged with the function name, and the word-run quoter downstream then read
    ``tsqrt`` as one word and emitted ``beta_#"tsqrt" (x)``, a formula whose
    root was lost (measured on the gold corpus, 6 formulas).
    """
    out: list[str] = []
    i = 0
    while i < len(text):
        if text.startswith(r"\sqrt", i) and (i + 5 == len(text) or not text[i + 5].isalpha()):
            pos = i + 5
            while pos < len(text) and text[pos].isspace():
                pos += 1
            arg_res = _extract_braced_chunk(text, pos)
            if arg_res is not None:
                arg_content, after_arg = arg_res
                rec_arg = _convert_sqrts(arg_content)
                if out and (out[-1].isalnum() or out[-1] in "_^"):
                    out.append(" ")
                out.append(f"sqrt({rec_arg})")
                i = after_arg
                continue
        out.append(text[i])
        i += 1
    return "".join(out)


def typstify_math(body: str) -> str | None:
    """Translate a LaTeX inline-math body to compilable Typst math.

    The input is normalized first (``_normalize_latex_spacing``) because the
    extractors emit a spaced LaTeX dialect (``\\frac { a } { b }``) that every
    pattern below would otherwise miss. Order after that matters: font
    alphabets and structure functions are rewritten to Typst calls before the
    generic command substitution, so no command is quoted as text after it has
    already become a function.
    """
    if not body:
        return None

    # Convert Script (𝒞 -> cal(C)), Fraktur (𝔈 -> frak(E)), and Double-Struck (ℝ -> bb(R))
    # BEFORE NFKC, because NFKC would flatten them to plain ASCII Latin letters (C, E, R).
    body = _expand_styled_unicode_in_math(body)
    body = unicodedata.normalize("NFKC", body).strip()
    # A bare ``#`` enters Typst code mode and a bare ``"`` opens a string, so
    # either makes the formula unrepresentable. ``\#`` is the LaTeX *escape*
    # for a literal hash (used as a counter name, ``\#L``), so it is exempt:
    # without the exemption the escape was rejected before the conversion pass
    # that turns it into Typst's ``hash`` glyph could run.
    if body.replace(r"\#", "").count("#") or '"' in body or "//" in body.replace(" ", ""):
        return None
    body = _normalize_latex_spacing(body)
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
    # Operators Typst spells as bare identifiers (\arg, \operatorname{...});
    # done before the unknown-command guard, which would otherwise fail the
    # whole formula closed over one unrecognized name.
    body = _rescue_operators(body)
    # Typst spells these math alphabets as function calls rather than LaTeX
    # commands. Leave syntax validation to the caller's TypstMathProbe.
    # ``boldsymbol``/``bm`` are LaTeX's bold-for-anything (including symbols),
    # which Typst's ``bold`` also accepts, so they share the mapping.
    _font_fn = {
        "mathbb": "bb",
        "mathfrak": "frak",
        "mathcal": "cal",
        "mathbf": "bold",
        "boldsymbol": "bold",
        "bm": "bold",
        "mathit": "italic",
    }
    body = _FONT_STYLE_CMD_RE.sub(lambda m: f"{_font_fn[m.group(1)]}({m.group(2)})", body)

    # Convert \frac{a}{b} -> frac(a, b) and \sqrt{a} -> sqrt(a) with balanced brace support
    body = _convert_fractions(body)
    body = _convert_sqrts(body)
    body = _convert_binoms(body)
    # ``\#`` is an escaped hash and the command regex below only matches
    # letters, so it is rewritten here. Typst spells the glyph ``hash``.
    body = body.replace(r"\#", " hash ")

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

    marked = _TEXTLIKE_CMD_RE.sub(
        lambda m: (
            f'#"{m.group(1)}"'
            if any(ord(c) > 127 for c in m.group(1))
            else "\0" + m.group(1) + "\0"
        ),
        body,
    )

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

    # ``\text{tokens per batch}`` is one upright label, but the word-run quoter
    # above split it into ``#"tokens" #"per" #"batch"``. That renders the same
    # in *display* math, but NOT in a script position: ``A_(#"i" #"j")``
    # typesets ``A_i j`` with a space, while ``A_#"ij"`` typesets ``A_ij``.
    # Re-join a run of adjacent quoted strings into one, keeping the whitespace
    # that separated them.
    def _rejoin_quoted(match: re.Match[str]) -> str:
        # ``#"a" #"b" #"c"`` -> ``"a b c"``: drop the leading ``#``, then
        # turn each ``" #"`` seam back into the space it stood for.
        return match.group(1)[1:].replace('" #"', " ")

    res = re.sub(r'((?:#"[^"]*"\s*){2,})', _rejoin_quoted, quoted)
    # Bare strings choke inside ``_{...}`` groups (``A_{#"g0"}`` renders
    # literally); a pure-label group (strings + separators only, no math)
    # collapses to one code string (``A_#"g0"``), which parses cleanly.
    res = _LABEL_GROUP_RE.sub(_collapse_label_group, res)
    # Avoid #"foo"(...) being parsed as a Typst code function call
    return re.sub(r'(#"[^"]*")(?=\()', r"\1 ", res)


_LABEL_GROUP_RE = re.compile(r'([_^])\{((?:[^#"{}]|#"[^"]*")*)\}')
_LABEL_SEP_RE = re.compile(r"^[\s,;:|/\-\u00b7\d]*$")
_STRING_RE = re.compile(r'#"([^"]*)"')

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


def strip_math_delimiters(text: str) -> str:
    """The body of a display-math span, whatever delimiters the extractor used.

    Single ``$`` is deliberately *not* stripped: a display formula's body never
    needs it (the extractor writes ``$$…$$`` or ``\\[…\\]``), and stripping a
    lone pair would turn a formula that genuinely is one dollar sign into an
    empty body — which :func:`typstify_math` then rejects, silently dropping the
    formula instead of reporting the malformed markup.

    A bare ``$$`` (the placeholder ``docling_parser`` writes for a formula it
    could not read) returns ``""`` so callers can recognise the placeholder
    rather than passing ``$$`` to the compiler.
    """
    body = text.strip()
    for opener, closer in (("$$", "$$"), (r"\[", r"\]"), (r"\(", r"\)")):
        if body == opener == closer:
            # The placeholder for a formula the extractor could not read: the
            # opener and closer are the same two characters, so the length test
            # below would treat it as a body of zero and fall through.
            return ""
        if body.startswith(opener) and body.endswith(closer):
            if len(body) == len(opener) + len(closer):
                return ""
            if len(body) > len(opener) + len(closer):
                return body[len(opener) : -len(closer)].strip()
    return body


#: LaTeX command -> Unicode glyph, for degrading a formula that will not
#: compile. Deliberately small: it exists to keep raw command names off the
#: page, not to typeset the equation.
_LATEX_TO_UNICODE_FALLBACK: dict[str, str] = {
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "epsilon": "ε",
    "zeta": "ζ",
    "eta": "η",
    "theta": "θ",
    "iota": "ι",
    "kappa": "κ",
    "lambda": "λ",
    "mu": "μ",
    "nu": "ν",
    "xi": "ξ",
    "pi": "π",
    "rho": "ρ",
    "sigma": "σ",
    "tau": "τ",
    "upsilon": "υ",
    "phi": "ϕ",
    "chi": "χ",
    "psi": "ψ",
    "omega": "ω",
    "Gamma": "Γ",
    "Delta": "Δ",
    "Theta": "Θ",
    "Lambda": "Λ",
    "Xi": "Ξ",
    "Pi": "Π",
    "Sigma": "Σ",
    "Upsilon": "Υ",
    "Phi": "Φ",
    "Psi": "Ψ",
    "Omega": "Ω",
    "le": "≤",
    "leq": "≤",
    "ge": "≥",
    "geq": "≥",
    "neq": "≠",
    "ne": "≠",
    "approx": "≈",
    "equiv": "≡",
    "sim": "∼",
    "propto": "∝",
    "in": "∈",
    "notin": "∉",
    "subset": "⊂",
    "supset": "⊃",
    "cup": "∪",
    "cap": "∩",
    "emptyset": "∅",
    "forall": "∀",
    "exists": "∃",
    "nabla": "∇",
    "partial": "∂",
    "infty": "∞",
    "sum": "∑",
    "prod": "∏",
    "int": "∫",
    "pm": "±",
    "mp": "∓",
    "times": "×",
    "div": "÷",
    "cdot": "·",
    "to": "→",
    "rightarrow": "→",
    "leftarrow": "←",
    "Rightarrow": "⇒",
    "Leftarrow": "⇐",
    "leftrightarrow": "↔",
    "mapsto": "↦",
    "sqrt": "√",
    "angle": "∠",
    "perp": "⊥",
    "parallel": "∥",
}


def _convert_fractions_to_slash(text: str) -> str:
    """Convert LaTeX \\frac{num}{den} to (num)/(den) with balanced brace support."""
    out: list[str] = []
    i = 0
    while i < len(text):
        if text.startswith(r"\frac", i) and (i + 5 == len(text) or not text[i + 5].isalpha()):
            pos = i + 5
            while pos < len(text) and text[pos].isspace():
                pos += 1
            num_res = _extract_braced_chunk(text, pos)
            if num_res is not None:
                num_content, after_num = num_res
                while after_num < len(text) and text[after_num].isspace():
                    after_num += 1
                den_res = _extract_braced_chunk(text, after_num)
                if den_res is not None:
                    den_content, after_den = den_res
                    rec_num = _convert_fractions_to_slash(num_content)
                    rec_den = _convert_fractions_to_slash(den_content)
                    num_str = (
                        rec_num
                        if (
                            rec_num.isalnum() or (rec_num.startswith("(") and rec_num.endswith(")"))
                        )
                        else f"({rec_num})"
                    )
                    den_str = (
                        rec_den
                        if (
                            rec_den.isalnum() or (rec_den.startswith("(") and rec_den.endswith(")"))
                        )
                        else f"({rec_den})"
                    )
                    out.append(f"{num_str}/{den_str}")
                    i = after_den
                    continue
        out.append(text[i])
        i += 1
    return "".join(out)


def clean_math_fallback_text(content: str) -> str:
    """Degrade an uncompilable LaTeX math span to clean readable Unicode text.

    Guarantees that raw LaTeX command names (e.g. \\frac, \\phi, \\gamma, \\sum)
    and backslashes NEVER leak into rendered deliverables when Typst math
    compilation fails.
    """
    text = content.strip()
    for opener, closer in (("$$", "$$"), (r"\[", r"\]"), (r"\(", r"\)")):
        if (
            text.startswith(opener)
            and text.endswith(closer)
            and len(text) >= len(opener) + len(closer)
        ):
            text = text[len(opener) : -len(closer)].strip()
            break
    else:
        if text.startswith("$") and text.endswith("$") and len(text) >= 2:
            text = text[1:-1].strip()

    text = _convert_fractions_to_slash(text)
    text = re.sub(r"\\(?:text|mathrm|mathbf|mathit|textbf|textit)\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\sqrt\{([^{}]*)\}", r"√(\1)", text)

    def _sub_sym(m: re.Match[str]) -> str:
        name = m.group(1)
        return _LATEX_TO_UNICODE_FALLBACK.get(name, name)

    text = re.sub(r"\\([a-zA-Z]+)", _sub_sym, text)
    text = text.replace(r"\{", "{").replace(r"\}", "}").replace("\\", "")
    return re.sub(r"\s+", " ", text).strip()
