r"""LaTeX math → Typst math transpiler.

The renderer needs this transform, and
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

from ubt.core.math_symbols import INLINE_SYMBOLS
from ubt.render.typst.math_ast_transpiler import (
    has_latex_environments,
    transpile_latex_ast,
)

#: A bare number is not math worth converting (``$5`` is currency).
_BARE_NUMBER_RE = re.compile(r"^\d[\d,.]*$")


def typstify_math_span(content: str) -> str | None:
    """Convert one delimited math span to a Typst math body, or ``None``.

    The single entry point for "span in, body out", so the renderer and the
    compile gate agree on exactly which bodies reach :func:`typstify_math`.
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
# Commands outside this table fail closed to escaped text.
_LATEX_CMD_RE = re.compile(r"\\([a-zA-Z]+)")
_LATEX_CMD_MAP = INLINE_SYMBOLS
# Converted/builtin Typst math identifiers that must NOT be quoted as text.
_TYPST_MATH_WORDS = frozenset(_LATEX_CMD_MAP.values()) | {
    "bb",
    "frak",
    "mat",
    "cases",
    "delim",
    # Emitted by the font-style mapping for ``\mathbf``/``\mathit``. Without
    # these the word-run quoter below turns them into literal text
    # (``#"bold" (W)``), so the formula renders the *name* of the function
    # instead of a bold variable.
    "bold",
    "italic",
    # Emitted by the \sqrt[n]{x} -> root(n, x) pre-pass: same failure mode as
    # bold/italic above — the quoter quoted the function name into
    # ``#"root" (3, x)`` and the PDF carried the literal word "root".
    "root",
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
            # Optional LaTeX root degree: \sqrt[3]{x} / \sqrt[n]{x+1}. Left
            # alone the bracket group sailed through and the deliverable
            # printed the literal text ``sqrt[3]{x}`` (the compile gate sees
            # no backslash, so it passes — zero signal). Typst spells the
            # nth root ``root(n, x)``.
            degree: str | None = None
            if pos < len(text) and text[pos] == "[":
                close = text.find("]", pos + 1)
                if close != -1 and "[" not in text[pos + 1 : close]:
                    degree = text[pos + 1 : close].strip()
                    pos = close + 1
                    while pos < len(text) and text[pos].isspace():
                        pos += 1
            arg_res = _extract_braced_chunk(text, pos)
            if arg_res is not None:
                arg_content, after_arg = arg_res
                rec_arg = _convert_sqrts(arg_content)
                if out and (out[-1].isalnum() or out[-1] in "_^"):
                    out.append(" ")
                if degree:
                    out.append(f"root({_convert_sqrts(degree)}, {rec_arg})")
                else:
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

    # The AST path runs AFTER normalization: the extractors emit a spaced
    # dialect (``\begin { pmatrix }``), so gating on the raw body let the real
    # Docling spelling slip past the environment check and fail closed with
    # ``None`` — the matrix then rendered as plain text. Normalizing first makes
    # the gate and the AST see the same, canonical ``\begin{pmatrix}`` shape.
    if has_latex_environments(body):
        ast_res = transpile_latex_ast(body, leaf_transpiler=typstify_math)
        if ast_res is not None:
            return ast_res

    # LaTeX's ``\|`` is the double-vertical-bar NORM delimiter (``\Vert``):
    # left unmapped it slipped past the ``\\([a-zA-Z]+)`` command regex and
    # rendered as Typst's single bar, silently halving every norm in the book
    # (measured: ``\|y - X\beta\|_2^2`` printed ``|y − Xβ|₂²``).
    body = body.replace(r"\|", r"\Vert ")
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
    # commands. A body that still fails to compile (an empty argument, a
    # dangling script) is caught by the caller's ``TypstMathProbe`` compile gate
    # (``ubt.adapters.pdf.typst_math_probe``), which degrades it to readable
    # Unicode rather than emitting markup the compiler rejects.
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

    def _font_style(match: re.Match[str]) -> str:
        # ``\in\mathcal{T}`` (unspaced, and the common compact spelling) would
        # otherwise fuse: the match starts right after ``\in``, so emitting
        # ``cal(T)`` with no separator produced ``\incal`` — an unknown command
        # that failed the whole formula closed. Keep a space when a command or
        # a word character runs into the call. A separator before a bare
        # letter subscript (``x^2``) or an opening delimiter is harmless.
        call = f"{_font_fn[match.group(1)]}({match.group(2)})"
        start = match.start()
        if start > 0 and (body[start - 1] == "\\" or body[start - 1].isalnum()):
            return " " + call
        return call

    body = _FONT_STYLE_CMD_RE.sub(_font_style, body)

    # Convert \frac{a}{b} -> frac(a, b) and \sqrt{a} -> sqrt(a) with balanced brace support
    body = _convert_fractions(body)
    body = _convert_sqrts(body)
    body = _convert_binoms(body)
    # ``\#`` is an escaped hash and the command regex below only matches
    # letters, so it is rewritten here. Typst spells the glyph ``hash``.
    body = body.replace(r"\#", " hash ")

    # Convert math accents \hat{a} -> hat(a), \hat a -> hat(a), etc.
    # A bare accent whose operand became a Typst function call (\hat\mathcal{T}
    # -> \hat cal(T), the font pass above rewrote it) must take the *whole* call.
    # The single-character form below would otherwise steal the call's first
    # letter: ``hat(c)al(T)`` — a silent mis-render.
    _accent_names = "hat|tilde|vec|bar|overline|dot|ddot"

    def _accent_call(match: re.Match[str]) -> str:
        return f"{_ACCENT_FN_MAP.get(match.group(1), match.group(1))}({match.group(2)})"

    _accent_call_re = re.compile(
        rf"\\({_accent_names})\s*((?:bb|frak|cal|bold|italic)\((?:[^()]|\([^()]*\))*\))"
    )
    # A bare accent whose operand is another LaTeX command takes that whole
    # command as its argument: LaTeX's ``\hat\alpha`` accents alpha, so the
    # command must survive *inside* the call (``hat(\alpha)``) and be
    # substituted normally afterwards. Left alone it produced None.
    _accent_over_cmd_re = re.compile(rf"\\({_accent_names})\s*(\\[a-zA-Z]+)")

    def _accent_over_cmd(match: re.Match[str]) -> str:
        name = _ACCENT_FN_MAP.get(match.group(1), match.group(1))
        return f"{name}({match.group(2)})"

    for _ in range(4):
        previous = body
        body = _accent_call_re.sub(_accent_call, body)
        body = _accent_over_cmd_re.sub(_accent_over_cmd, body)
        body = _ACCENT_CMD_RE.sub(_accent_call, body)
        body = _ACCENT_BARE_CMD_RE.sub(_accent_call, body)
        if body == previous:
            break

    # A ``\text{...}`` label that contains CJK is emitted already quoted, because
    # the word-run quoter below would otherwise split it on its ASCII words. It
    # must then be *hidden* from that quoter, which cannot tell an already-quoted
    # label from bare source: ``\text{总 token 数}`` became
    # ``#"总 #"token" 数"`` -- the inner quotes close the label early and Typst
    # fails with ``unknown variable: token``, losing the whole document (measured
    # on switch-transformers, a figure caption). Park the label behind a
    # non-word sentinel and restore it once quoting is done.
    cjk_labels: list[str] = []

    def _textlike(match: re.Match[str]) -> str:
        argument = match.group(1)
        if any(ord(c) > 127 for c in argument):
            cjk_labels.append(argument)
            return f"\x01{len(cjk_labels) - 1}\x02"
        # The marked form keeps the label's own spaces out of the quoter's way;
        # ``_WORD_RUN_RE`` then quotes it as one word (``\0off\0`` -> ``#"off"``).
        return "\0" + argument + "\0"

    marked = _TEXTLIKE_CMD_RE.sub(_textlike, body)

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
        # Dynamic fallback for simple alphabetical operator/function names (e.g. \sinc, \relu, \sign, \poly)
        # Emits the identifier so the word-run quoter turns it into a valid Typst string operator #"name".
        # Parameterized commands (followed by '{') stay fail-closed so they degrade cleanly.
        if name.isalpha() and 2 <= len(name) <= 15:
            rem = marked[match.end() :].lstrip()
            if not rem.startswith("{"):
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
                return f"{pre}\0{name}\0{post}"
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
    # Restore the CJK labels the quoter was shielded from. ``re.escape`` is not
    # needed: the replacement is a literal string, and a backslash in a label
    # would have to survive Typst's string syntax, which it does verbatim.
    quoted = re.sub(
        r"\x01(\d+)\x02",
        lambda m: f'#"{cjk_labels[int(m.group(1))]}"',
        quoted,
    )

    # ``\text{tokens per batch}`` is one upright label, but the word-run quoter
    # above split it into ``#"tokens" #"per" #"batch"``. That renders the same
    # in *display* math, but NOT in a script position: ``A_(#"i" #"j")``
    # typesets ``A_i j`` with a space, while ``A_#"ij"`` typesets ``A_ij``.
    # Re-join a run of adjacent quoted strings into one, keeping the whitespace
    # that separated them.
    def _rejoin_quoted(match: re.Match[str]) -> str:
        # ``#"a" #"b" #"c"`` -> ``#"a b c"``: keep the leading ``#`` (a bare
        # string is not a value in math mode, and ``_LABEL_GROUP_RE`` below
        # deliberately refuses to match a bare ``"``), then turn each ``" #"``
        # seam back into the space it stood for.
        return match.group(1).replace('" #"', " ")

    res = re.sub(r'((?:#"[^"]*"\s*){2,})', _rejoin_quoted, quoted)
    # Bare strings choke inside ``_{...}`` groups (``A_{#"g0"}`` renders
    # literally); a pure-label group (strings + separators only, no math)
    # collapses to one code string (``A_#"g0"``), which parses cleanly.
    res = _LABEL_GROUP_RE.sub(_collapse_label_group, res)
    # Avoid #"foo"(...) being parsed as a Typst code function call
    res = re.sub(r'(#"[^"]*")(?=\()', r"\1 ", res)
    # A script operator with no operand fails the *whole* document: ``$a^$``
    # compiles to ``error: expected expression``, and the publish path answers a
    # compile failure by raising ``DocumentParseError`` -- so one dangling ``^``
    # anywhere in the book loses every page. The model emits these when it
    # re-encodes a flattened superscript as LaTeX (``x²`` -> ``x^``) or drops a
    # brace. ``x^2``/``a^{}`` are fine; only ``^``/``_`` directly followed by
    # another script operator or by end-of-body is broken. Attach an empty
    # string, the same spelling the leading-script guard above uses, which Typst
    # accepts in every position.
    return _DANGLING_SCRIPT_RE.sub(r'\1""', res)


#: A subscript/superscript with nothing to attach to (see the call site).
_DANGLING_SCRIPT_RE = re.compile(r"([_^])(?=[_^]|\s*$)")


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
    # Symbols with a direct Unicode spelling that were missing entirely (the
    # fallback used to leak the command name for every one of these).
    "otimes": "⊗",
    "oplus": "⊕",
    "ominus": "⊖",
    "odot": "⊙",
    "ast": "∗",
    "star": "⋆",
    "circ": "∘",
    "bullet": "•",
    "dagger": "†",
    "ddagger": "‡",
    "prime": "′",
    "cdots": "⋯",
    "ldots": "…",
    "vdots": "⋮",
    "ddots": "⋱",
    "dotsb": "…",
    "dots": "…",
    "implies": "⇒",
    "impliedby": "⇐",
    "iff": "⟺",
    "lvert": "|",
    "rvert": "|",
    "lVert": "‖",
    "rVert": "‖",
    "vert": "|",
    "Vert": "‖",
    "mid": "∣",
    "langle": "⟨",
    "rangle": "⟩",
    "lbrace": "{",
    "rbrace": "}",
    "lbrack": "[",
    "rbrack": "]",
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
    compilation fails: every mapped command degrades to its Unicode spelling,
    structure commands (\\frac, \\underline, \\begin{...}) are unwrapped to
    their argument content, and any command left unrecognised degrades to a
    neutral ``⟨math⟩`` placeholder rather than the command name.
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
    # nth roots: \sqrt[3]{x} -> √[3](x) — the degree is information, keep it.
    text = re.sub(r"\\sqrt\[([^\[\]]*)\]\{([^{}]*)\}", r"√[\1](\2)", text)
    # Two-argument structure BEFORE the one-argument shell-stripper below
    # would eat the first brace pair and strand the second as ``1{2}``:
    # \tfrac{1}{2} -> (1)/(2), the same slash shape
    # _convert_fractions_to_slash produces for \frac.
    text = re.sub(r"\\(?:tfrac|dfrac|cfrac)\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1)/(\2)", text)
    # Structure commands: keep the argument, drop the shell. ``\underline{x}``
    # used to print the literal ``underline{x}`` — the guarantee in this
    # docstring was false for 169 of the 241 table commands.
    text = re.sub(
        r"\\(?:binom|underline|overline|widehat|widetilde"
        r"|overbrace|underbrace|overrightarrow|mathbf|mathit|mathbb|mathcal"
        r"|mathfrak|boldsymbol|bm|displaystyle|operatorname|limits)\{([^{}]*)\}",
        r"\1",
        text,
    )
    # Array environments: \begin{...} & \end{...} — drop the wrappers, keep
    # rows readable (cell & cell; \\ separates rows as a line break).
    text = re.sub(r"\\(?:begin|end)\{[^{}]*\}", " ", text)
    text = text.replace(" & ", ", ").replace("&", ", ")
    text = text.replace("\\\\", "; ")
    # Delimiter-size modifiers carry no content and are not in the symbol
    # table, so they used to degrade to the ``⟨math⟩`` placeholder
    # (``a ⟨math⟩( x ⟨math⟩)``). Drop them before the generic pass.
    text = re.sub(r"\\(?:big|Big|bigg|Bigg|bigl|bigr|Bigl|Bigr|left|right)\b\s*", " ", text)

    def _sub_sym(m: re.Match[str]) -> str:
        name = m.group(1)
        if name in _LATEX_TO_UNICODE_FALLBACK:
            return _LATEX_TO_UNICODE_FALLBACK[name]
        # Unmapped command: a neutral placeholder, never the command name.
        return "⟨math⟩"

    text = re.sub(r"\\([a-zA-Z]+)", _sub_sym, text)
    text = text.replace(r"\{", "{").replace(r"\}", "}").replace("\\", "")
    return re.sub(r"\s+", " ", text).strip()
