"""LaTeX/math-to-Typst conversion chain.

Group reader, Docling-math normalization, pandoc + regex converters, OCR
cleanup, delimiter checks, and formula emission. No table/text/image logic.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess

from ubt.adapters.pdf.typst_symbols import DISPLAY_SYMBOLS
from ubt.core.env import subprocess_env

logger = logging.getLogger(__name__)

_TEX_STRIP_PATTERN = re.compile(
    r"\\(?:left|right|big|Big|bigg|Bigg|displaystyle|limits|quad|qquad)(?![a-zA-Z])"
)
# NOTE: LaTeX's explicit line break `\\` is deliberately NOT a spacing token.
# Docling preserves where the author broke the line (chapter-3 Eq. (3.11) breaks
# before `+ 2 eps / (T_fin C_ox)`), and flattening it to a space makes Typst
# re-wrap the whole equation by itself -- which lands the wrap at the last
# relation and orphans a trailing `= 0` on its own line, while the author's
# break point is lost. See _latex_math_to_typst_regex, which protects and
# restores it as a Typst continuation line.
_TEX_SPACING = {r"\,": " ", r"\;": " ", r"\:": " ", r"\!": ""}
_LATEX_LINEBREAK_RE = re.compile(r"\\\\+")
# A bare NUL, deliberately letter-free: _sanitize_typst_math_variables quotes
# every bare multi-letter token, and a sentinel spelled with letters ("BRK")
# came back as \x00"BRK"\x00 and never matched the restore pattern.
_LINEBREAK_SENTINEL = "\x00"
# A Typst continuation line: a bare `quad` (1em) reproduces the author's
# half-quad indent on the wrapped part. Deliberately NO alignment point (`&`):
# `&` splits an equation row into alignment columns, and a lone `&` on the
# continuation row (the author's rows carry none) puts row 1 into column 1 and
# row 2 into column 2 — Typst then sizes the block as the sum of both rows and
# the whole equation overflows both page margins by half the excess
# (chapter-3 Eq. 3.11: 47pt past each margin, first row starting off-column).
_LINEBREAK_TYPST = " \\\n  quad "
_TEX_SYMBOL_MAP = DISPLAY_SYMBOLS
_TEX_QUEST_FUNCTION_MAP = {
    "text": "upright",
    "mathrm": "upright",
    "operatorname": "upright",
    "mbox": "upright",
    "textrm": "upright",
    "mathbf": "bold",
    "boldsymbol": "bold",
    "mathit": "italic",
    "mathbb": "bb",
    "mathcal": "cal",
    "mathfrak": "frak",
    "hat": "hat",
    "vec": "vec",
    "tilde": "tilde",
    "overline": "overline",
    "underline": "underline",
}
_TEX_TWO_ARG_FUNCS = {"frac": "frac", "dfrac": "frac", "tfrac": "frac", "log_": "log"}

_TYPST_MATH_KEYWORDS: frozenset[str] = frozenset(
    {
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "zeta",
        "eta",
        "theta",
        "iota",
        "kappa",
        "lambda",
        "mu",
        "nu",
        "xi",
        "pi",
        "rho",
        "sigma",
        "tau",
        "upsilon",
        "phi",
        "chi",
        "psi",
        "omega",
        "Gamma",
        "Delta",
        "Theta",
        "Lambda",
        "Xi",
        "Pi",
        "Sigma",
        "Upsilon",
        "Phi",
        "Psi",
        "Omega",
        "sin",
        "cos",
        "tan",
        "cot",
        "sec",
        "csc",
        "sinh",
        "cosh",
        "tanh",
        "ln",
        "log",
        "exp",
        "sqrt",
        "root",
        "frac",
        "min",
        "max",
        "lim",
        "dim",
        "det",
        "inf",
        "sup",
        "infinity",
        "times",
        "dot",
        "div",
        "plus",
        "minus",
        "pm",
        "mp",
        "partial",
        "equiv",
        "approx",
        "in",
        "subset",
        "supset",
        "forall",
        "exists",
        "dif",
        "upright",
        "bold",
        "italic",
        "bb",
        "cal",
        "frak",
        "gt",
        "lt",
        "eq",
        "ne",
        "le",
        "ge",
        "arrow",
        "prop",
        "parallel",
        "perp",
        "and",
        "or",
        "not",
        "without",
        "nabla",
        "planck",
        "dots",
        "integral",
        # Large operators and relations pandoc emits for Typst. Missing from
        # this set they were quoted as literal text by
        # _sanitize_typst_math_variables ("sum", "product", "union", "oo"...),
        # so \sum/\prod/\cup/\infty rendered as the words instead of the
        # symbols. Verified against `typst compile` for each identifier.
        "sum",
        "product",
        "union",
        "inter",
        "oo",
        "therefore",
        "because",
        "arg",
        "compose",
    }
)


_CODE_SPAN_START_RE = re.compile(r"#[a-zA-Z][a-zA-Z0-9]*\s*[\(\[]")


def _mask_code_spans(s: str) -> tuple[str, list[str]]:
    """Lift pandoc's ``#call(...)``/``#call[...]`` spans out of a math string.

    Their contents are Typst *code*, not math: quoting bare identifiers inside
    (as :func:`_sanitize_typst_math_variables` does for math) would corrupt the
    call. Returns the masked text plus the spans, restored by the caller.
    """
    spans: list[str] = []
    out: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        m = _CODE_SPAN_START_RE.match(s, i)
        if m is None:
            out.append(s[i])
            i += 1
            continue
        depth = 0
        k = m.end() - 1
        while k < n:
            c = s[k]
            if c == "\\" and k + 1 < n:
                k += 2  # escaped delimiter (\( \) \[ \]) is not a delimiter
                continue
            if c in "([":
                depth += 1
            elif c in ")]":
                depth -= 1
                if depth == 0:
                    k += 1
                    break
            k += 1
        spans.append(s[i:k])
        out.append(f"\x01{len(spans) - 1}\x01")
        i = k
    return "".join(out), spans


def _sanitize_typst_math_variables(s: str) -> str:
    """Quote bare multi-letter tokens the *regex* tier invented, so Typst never sees an unknown variable.

    Never apply it to pandoc's output. The allowlist below is a hand-written
    list of symbols, while pandoc emits Typst's whole math namespace —
    ``hat``, ``tilde``, ``overline``, ``binom``, ``divides``, ``cases``, ``mat``
    — and every one of those it did not know became a quoted *word*. That
    corruption is silent: it compiles with exit 0 and the atom-count gates see
    the same content, so the reader just gets ``hat(θ)`` printed where θ̂
    belonged. A pandoc identifier Typst truly rejects fails the compile and the
    healer degrades that one formula to a verbatim line, which is the honest
    failure mode.
    """
    s, masked_spans = _mask_code_spans(s)
    str_pat = re.compile(r'"(?:\\.|[^"\\])*"')
    pos = 0
    chunks: list[str] = []

    def _repl(m: re.Match[str], current_part: str) -> str:
        t = m.group(1)
        start, end = m.start(1), m.end(1)
        pre = current_part[start - 1] if start > 0 else ""
        post = current_part[end] if end < len(current_part) else ""
        if (
            pre in (".", "_", "\\")
            or post == "."
            or (start > 0 and current_part[start - 1] == "\\")
        ):
            return t
        if post == ":" and t in ("delim", "augment", "gap"):
            return t
        if post == "(" and t in (
            "upright",
            "bold",
            "italic",
            "bb",
            "cal",
            "frak",
            "sqrt",
            "root",
            "frac",
            "mat",
            "cases",
            "binom",
            "vec",
            "hat",
            "tilde",
            "overline",
            "underline",
            "sin",
            "cos",
            "tan",
            "cot",
            "sec",
            "csc",
            "sinh",
            "cosh",
            "tanh",
            "ln",
            "log",
            "exp",
            "min",
            "max",
            "lim",
            "dim",
            "det",
            "integral",
        ):
            return t
        if t.lower() in _TYPST_MATH_KEYWORDS or t in _TYPST_MATH_KEYWORDS:
            return t
        return f'"{t}"'

    var_pat = re.compile(r"(?<![\\a-zA-Z0-9_])([a-zA-Z][a-zA-Z0-9]*[a-zA-Z0-9])(?![a-zA-Z0-9])")

    for m in str_pat.finditer(s):
        start, end = m.span()
        if start > pos:
            part = s[pos:start]

            def _sub_match(vm: re.Match[str], _part: str = part) -> str:
                return _repl(vm, _part)

            chunks.append(var_pat.sub(_sub_match, part))
        chunks.append(m.group(0))
        pos = end
    if pos < len(s):
        part = s[pos:]
        chunks.append(var_pat.sub(lambda vm: _repl(vm, part), part))
    result = "".join(chunks)
    for idx, span in enumerate(masked_spans):
        result = result.replace(f"\x01{idx}\x01", span)
    return result


def is_typst_math_well_formed(expr: str) -> bool:
    """Validate that a Typst math expression has balanced delimiters and closed quotes."""
    stack: list[str] = []
    in_string = False
    i = 0
    n = len(expr)

    while i < n:
        ch = expr[i]

        if in_string:
            if ch == "\\":
                i += 2
                continue
            elif ch == '"':
                in_string = False
                i += 1
                continue
            else:
                i += 1
                continue
        else:
            if ch == "\\":
                i += 2
                continue
            elif ch == '"':
                in_string = True
                i += 1
                continue
            elif ch in "([{":
                stack.append(ch)
                i += 1
                continue
            elif ch in ")]}":
                expected_open = {")": "(", "]": "[", "}": "{"}[ch]
                if expected_open in stack:
                    if stack[-1] == expected_open:
                        stack.pop()
                    else:
                        # Typst supports interleaved interval delimiters like
                        # ([)] — the expected opener is present but not on top.
                        idx = len(stack) - 1 - stack[::-1].index(expected_open)
                        stack.pop(idx)
                elif ch in ")]" and any(o in stack for o in ("(", "[")):
                    # Interval notation ([a, b) / (a, b]): a round/square close
                    # may match the *other* round/square opener. Without this the
                    # documented case returned False and a valid interval was
                    # degraded to a verbatim code span.
                    idx = max(k for k, o in enumerate(stack) if o in ("(", "["))
                    stack.pop(idx)
                else:
                    return False
                i += 1
                continue
            elif ch == "$":
                return False
            else:
                i += 1
                continue

    return not (in_string or stack)


def _strip_math_delimiters(span: str) -> str:
    """Remove one layer of math delimiters ($, $$, \\(...\\), \\[...\\])."""
    s = span.strip()
    if s.startswith("$$") and s.endswith("$$") and len(s) >= 4:
        return s[2:-2].strip()
    if s.startswith("$") and s.endswith("$") and len(s) >= 2:
        return s[1:-1].strip()
    if s.startswith("\\(") and s.endswith("\\)"):
        return s[2:-2].strip()
    if s.startswith("\\[") and s.endswith("\\]"):
        return s[2:-2].strip()
    return s


def _skip_ws(s: str, i: int) -> int:
    while i < len(s) and s[i].isspace():
        i += 1
    return i


def _read_group(s: str, i: int) -> tuple[str, int] | tuple[None, int]:
    """Read a balanced {...} group starting at s[i]; returns (content, next_index).

    The failure return is ``(None, i)`` — "no group here, nothing consumed".
    It must NOT be ``(None, len(s))``: every caller treats the returned index
    as the next parse position, so claiming the whole remainder was consumed
    makes the caller drop the rest of the formula on the floor. That is not
    hypothetical — Docling emits benignly unbalanced LaTeX (missing ``}``
    before a stray ``\\right)``), and the old sentinel silently truncated
    ``\\mathcal{E}_{xs}=\\sqrt{\\frac{...}}{...}`` (483 chars) down to
    ``cal(E)_("x s")= sqrt`` (20 chars) with no warning, no residual-LaTeX
    signal and a syntactically well-formed result that sailed through Gate 4.
    See ``_content_lost`` for the independent content-preservation gate.
    """
    i = _skip_ws(s, i)
    if i >= len(s) or s[i] != "{":
        return None, i
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1 : j], j + 1
    return None, i


_DOCLING_BARE_CMDS = {
    "colon",
    "cdots",
    "vdots",
    "ldots",
    "dots",
    "cdot",
    "rightarrow",
    "leftarrow",
    "Rightarrow",
    "Leftarrow",
    "Leftrightarrow",
    "rightarrowtail",
    "mapsto",
    "to",
    "in",
    "notin",
    "subset",
    "subseteq",
    "supset",
    "supseteq",
    "cup",
    "cap",
    "circ",
    "partial",
    "infty",
    "sum",
    "prod",
    "int",
    "iint",
    "iiint",
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "varepsilon",
    "zeta",
    "eta",
    "theta",
    "vartheta",
    "iota",
    "kappa",
    "lambda",
    "mu",
    "nu",
    "xi",
    "pi",
    "varpi",
    "rho",
    "varrho",
    "sigma",
    "varsigma",
    "tau",
    "upsilon",
    "phi",
    "varphi",
    "chi",
    "psi",
    "omega",
    "Gamma",
    "Delta",
    "Theta",
    "Lambda",
    "Xi",
    "Pi",
    "Sigma",
    "Upsilon",
    "Phi",
    "Psi",
    "Omega",
    "mathbb",
    "mathbf",
    "mathrm",
    "mathit",
    "mathcal",
    "quad",
    "qquad",
    "times",
}
_DOCLING_RE = re.compile(r"(?<![\\a-zA-Z])(" + "|".join(_DOCLING_BARE_CMDS) + r")(?=[^a-zA-Z]|$)")

# Trailing ``\text{...}`` groups holding natural-language prose, e.g. Docling
# gluing the "where ..." sentence into a formula block as
# ``\\ \text{is the electrostatic potential ...}``. The prose is duplicated
# in the following narrative block, so it is dropped from the formula;
# short subscript tags (``\text{ch}``, no spaces) are real math and stay.
_TAIL_UNIT_RE = re.compile(r"(?:\\\\\s*)?(\\text\s*\{([^{}]*)\}|[A-Za-z])\s*$")
_HEAD_TEXT_RE = re.compile(r"^[\s&\\\\]*\\text\s*\{([^{}]*)\}\s*")
_PROSE_WORD_RE = re.compile(r"[A-Za-z]{4,}")


def _is_prose_text(inner: str | None) -> bool:
    return inner is not None and " " in inner and _PROSE_WORD_RE.search(inner) is not None


def _strip_trailing_text_prose(s: str) -> str:
    """Remove natural-language ``\text{...}`` tails from a formula source.

    Docling glues sentence fragments as ``... \\\\ \\text{A} a \\text{B}``;
    the loop strips prose ``\\text`` groups right-to-left, and lone letters
    sandwiched between them — but lone letters strip only after at least one
    prose group established artifact mode, so genuine endings (``y=ax+b``,
    aligned ``a \\\\ b``) never match.
    """
    stripped_prose = False
    while True:
        match = _TAIL_UNIT_RE.search(s)
        if not match:
            return s
        unit, inner = match.group(1), match.group(2)
        if unit.startswith("\\text"):
            if _is_prose_text(inner):
                s = s[: match.start()].rstrip()
                stripped_prose = True
                continue
            return s
        if stripped_prose:
            s = s[: match.start()].rstrip()
            continue
        return s


def _strip_leading_text_prose(s: str) -> str:
    """Remove natural-language ``\\text{...}`` heads (mirror of the tail strip).

    Docling also prepends sentence starts (``\\text{ can be integrated ...}``).
    Short tags (``\\text{for }``, ``\\text{if }`` — no 4-letter word) stay.
    """
    while True:
        match = _HEAD_TEXT_RE.match(s)
        if not match or not _is_prose_text(match.group(1)):
            return s
        s = s[match.end() :].lstrip()


def _needs_latex_conversion(formula: str) -> bool:
    """Determine if a formula contains LaTeX markup or Docling OCR artifacts."""
    return (
        "\\" in formula
        or "&" in formula
        or bool(_DOCLING_RE.search(formula))
        or bool(re.search(r"[_^]\s*\{", formula))
    )


_GROUP_BODY_RE = re.compile(r"\{([^{}]*)\}")


def _normalize_docling_math(s: str) -> str:
    r"""Restore backslashes for Docling OCR stripped commands and normalize groups.

    Brace-group bodies are masked before the restore and put back untouched.
    Inside ``{...}`` the characters are the author's own text, and a stripped
    backslash cannot be told apart from an ordinary subscript word — guessing
    wrong silently changes the formula. ``R_{in}`` (input resistance) came back
    as ``R_{\in}`` and Typst renders that as ``R ∈``; left alone, pandoc emits
    ``R_(i n)`` and the label survives. A Greek name in a subscript degrades to
    plain letters instead, which is the safe direction: visibly imperfect
    rather than quietly a different statement.
    """
    bodies: list[str] = []

    def _mask(match: re.Match[str]) -> str:
        bodies.append(match.group(1))
        return "\x00" + str(len(bodies) - 1) + "\x00"

    while _GROUP_BODY_RE.search(s):
        s = _GROUP_BODY_RE.sub(_mask, s)
    masked = s
    masked = _DOCLING_RE.sub(r"\\\1", masked)
    while "\x00" in masked:
        head, _, rest = masked.partition("\x00")
        index, _, tail = rest.partition("\x00")
        if not index.isdigit():
            masked = head + "\x00" + rest
            break
        masked = head + "{" + bodies[int(index)] + "}" + tail
    s = masked
    s = re.sub(r"_\s*\{\s*", "_{", s)
    s = re.sub(r"\^\s*\{\s*", "^{", s)
    s = re.sub(r"\s*\}\s*", "}", s)
    return s


def _restore_line_breaks(text: str) -> str:
    """Turn protected ``\\\\`` sentinels back into Typst continuation lines.

    On the regex tier it runs *after*
    :func:`_sanitize_typst_math_variables`, which quotes every bare
    multi-letter token — restoring earlier would hand it a bare ``quad`` and get
    ``"quad"`` back, which Typst sets as the literal word "quad" instead of an
    indent. The restored form is a bare ``quad`` with no ``&`` (see
    ``_LINEBREAK_TYPST``).

    A sentinel sitting at the very end of the equation (Docling's trailing
    ``\\quad \\\\`` on chapter-3 Eq. 3.13) carries no continuation content:
    restoring it leaves a dangling ``\\`` + ``quad`` row that Typst sets as
    the literal word "quad" after the equation. Drop it here.
    """
    out = re.sub(r"[ \t]*" + _LINEBREAK_SENTINEL + r"[ \t]*", _LINEBREAK_TYPST, text)
    return re.sub(r"(?:\\\n\s*&?quad\s*)+$", "", out).rstrip()


def _latex_math_to_typst(latex: str) -> str:
    """Convert LaTeX display math to Typst math syntax.

    A-track: pandoc's AST-based converter is primary (complete content
    preservation, correct subscript attachment — verified against the KV
    handbook formulas where the regex path dropped half the equation).
    The legacy regex converter below stays as the zero-dependency fallback
    for environments without pandoc (missing binary, timeout, parse
    failure): any pandoc failure returns None and this function degrades
    gracefully instead of raising.

    Never fails closed -- unlike :func:`overlay_text.typstify_math`, which
    returns ``None`` so the rigid engine can drop a span it cannot typeset.
    See that docstring for why the two converters are not merged.
    """
    s = latex.strip()

    # 0. Normalize Docling spaced-out format back to LaTeX (shared
    # preprocessor: pandoc's reader needs the same canonical input).
    s = _normalize_docling_math(s)

    pandoc_out = _pandoc_math_to_typst(s)
    if pandoc_out and not _has_residual_latex(pandoc_out):
        return _accept_conversion(s, _restore_line_breaks(pandoc_out), "pandoc")
    # Pandoc "succeeds" (exit 0) on some polluted inputs while echoing raw
    # LaTeX back through. Residual backslash-commands mean the AST pass did
    # not convert, so degrade to the regex converter instead of emitting
    # fail-open garbage that breaks Typst compilation document-wide.
    if pandoc_out:
        logger.warning(
            "Pandoc math passthrough detected (residual LaTeX kept); using regex fallback",
        )
    return _accept_conversion(
        s,
        _restore_line_breaks(_sanitize_typst_math_variables(_latex_math_to_typst_regex(s))),
        "regex",
    )


def _accept_conversion(source: str, converted: str, tier: str) -> str:
    """Apply Gate 5 to one tier's output.

    Returns ``converted`` when the math content survived, otherwise the raw
    source. Handing back the source is deliberate fail-loud: it still carries
    residual LaTeX, so the caller's Gate 4 rejects it and emits a verbatim
    code span. A visibly imperfect formula beats an invisibly wrong one.
    """
    if not _content_lost(source, converted):
        return converted
    logger.warning(
        "Math conversion (%s) lost content: %d -> %d math atoms; keeping source verbatim",
        tier,
        _math_atom_count(source),
        _math_atom_count(converted),
    )
    return source


latex_math_to_typst = _latex_math_to_typst


# Per-formula pandoc budget: formulas are short; anything slower means a
# wedged subprocess that must not stall the render stage.

_PANDOC_TIMEOUT_S = 15


def _pandoc_math_to_typst(normalized: str) -> str | None:
    """AST-based LaTeX→Typst via pandoc; None on any failure (fallback path).

    Takes canonical LaTeX (already through :func:`_normalize_docling_math`),
    wraps it as display math for pandoc's LaTeX reader, and unwraps one
    ``$`` layer from the Typst output for our ``$ … $`` emitter.
    """
    pandoc = shutil.which("pandoc")
    if pandoc is None or not normalized.strip():
        return None
    try:
        proc = subprocess.run(
            [pandoc, "-f", "latex", "-t", "typst", "--wrap=none"],
            input="$$" + normalized.strip() + "$$",
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=_PANDOC_TIMEOUT_S,
            # pandoc needs PATH/HOME, not this host's credentials.
            env=subprocess_env(),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    out = proc.stdout.replace("\n", " ").strip()
    # Unwrap pandoc's display-math delimiters (our emitter adds its own).
    if out.startswith("$$") and out.endswith("$$") and len(out) >= 4:
        out = out[2:-2].strip()
    elif out.startswith("$") and out.endswith("$") and len(out) >= 2:
        out = out[1:-1].strip()
    else:
        # Pandoc failed to convert to math mode and echoed escaped plain text (e.g. \$\$...\$\$)
        return None
    out = re.sub(r"[ \t]{2,}", " ", out).strip()
    return out or None


def _latex_math_to_typst_regex(latex: str) -> str:
    """Regex-based LaTeX→Typst converter (zero-dependency fallback)."""
    s = latex.strip()

    # 0. Normalize Docling spaced-out format back to LaTeX
    s = _normalize_docling_math(s)

    # 0. Convert LaTeX environments (matrices, cases, aligned)
    _MATRIX_DELIMS = {
        "pmatrix": 'delim: "("',
        "bmatrix": 'delim: "["',
        "Bmatrix": 'delim: "{"',
        "vmatrix": 'delim: "|"',
        "Vmatrix": 'delim: "||"',
        "matrix": "",
    }

    def _replace_matrix(m: re.Match[str]) -> str:
        env = m.group(1)
        body = m.group(2).strip()
        delim = _MATRIX_DELIMS.get(env, "")
        raw_rows = re.split(r"\\\\+|\\newline", body)
        rows: list[str] = []
        for r in raw_rows:
            r = r.strip()
            if not r:
                continue
            cols = [c.strip() for c in r.split("&")]
            rows.append(", ".join(cols))
        rows_str = "; ".join(rows)
        if delim:
            return f"mat({delim}, {rows_str})"
        return f"mat({rows_str})"

    def _replace_cases(m: re.Match[str]) -> str:
        body = m.group(1).strip()
        raw_rows = re.split(r"\\\\+|\\newline", body)
        rows: list[str] = []
        for r in raw_rows:
            r = r.strip()
            if not r:
                continue
            cols = [c.strip() for c in r.split("&")]
            rows.append(" \x01 ".join(cols))
        return f"cases({', '.join(rows)})"

    s = re.sub(
        r"\\begin\{(matrix|pmatrix|bmatrix|Bmatrix|vmatrix|Vmatrix)\}(.*?)\\end\{\1\}",
        _replace_matrix,
        s,
        flags=re.DOTALL,
    )
    s = re.sub(r"\\begin\{cases\}(.*?)\\end\{cases\}", _replace_cases, s, flags=re.DOTALL)
    s = re.sub(
        r"\\begin\{(?:aligned|split|align\*?|gather\*?)\}(.*?)\\end\{(?:aligned|split|align\*?|gather\*?)\}",
        r"\1",
        s,
        flags=re.DOTALL,
    )

    # Protect alignment ampersands in expressions like &= or & =
    s = re.sub(r"&\s*([=<>&|/+\-])", lambda m: f"\x01 {m.group(1)}", s)

    # 0a. Protect the author's explicit line breaks. Everything downstream
    # treats `\\` as layout noise, so it has to be lifted out of the stream
    # before the spacing/strip passes and put back at the end.
    s = _LATEX_LINEBREAK_RE.sub(_LINEBREAK_SENTINEL, s)

    # 0. Normalize bare or backslashed text / font macros into quoted strings for Typst
    s = re.sub(r"\\?(?:text|mathrm|operatorname|textrm|mbox)\s*\{([^}]+)\}", r'"\1"', s)
    s = re.sub(r"\\?(?:text|mathrm|operatorname|textrm|mbox)\s*\(([^\"'\)]+)\)", r'"\1"', s)
    s = re.sub(r"\\?mathbb\s*\{([^}]+)\}", r"bb(\1)", s)
    s = re.sub(r"\\?mathbf\s*\{([^}]+)\}", r"bold(\1)", s)
    s = re.sub(r"\b(?:nolimits|displaystyle)\b", " ", s)

    # 1. Strip alignment ampersands and equation tags like '&& (1)'.
    s = re.sub(r"(?:&\s*)+\(\s*\d+\s*\)", " ", s)
    s = s.replace("&", " ")
    s = s.replace("\x01", "&")

    # 2. Drop layout-only tokens and normalize spacing commands.
    s = _TEX_STRIP_PATTERN.sub(" ", s)
    for tok, repl in _TEX_SPACING.items():
        s = s.replace(tok, repl)

    # 3. Rewrite \command{arg...} sequences with balanced-brace parsing.
    out: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch != "\\":
            out.append(ch)
            i += 1
            continue

        m = re.match(r"\\([a-zA-Z]+)", s[i:])
        if m is None:
            out.append(ch)  # escaped literal (\{, \%, \\ already handled above)
            i += 1
            continue
        cmd = m.group(1)
        i += 1 + len(cmd)

        if cmd == "sqrt":
            j = i
            index: str | None = None
            if j < n and s[j] == "[":
                close = s.find("]", j)
                if close != -1:
                    index = s[j + 1 : close].strip()
                    j = close + 1
            arg, j = _read_group(s, j)
            if index:
                out.append(f"root({index.strip()}, {_latex_math_to_typst_regex(arg or '')})")
            elif arg is not None:
                out.append(f"sqrt({_latex_math_to_typst_regex(arg)})")
            else:
                out.append("sqrt")
            i = j
        elif cmd in _TEX_TWO_ARG_FUNCS:
            arg1, j = _read_group(s, i)
            arg2, j2 = _read_group(s, j)
            if arg1 is not None and arg2 is not None:
                out.append(
                    f"{_TEX_TWO_ARG_FUNCS[cmd]}({_latex_math_to_typst_regex(arg1)}, "
                    f"{_latex_math_to_typst_regex(arg2)})"
                )
                i = j2
            else:
                out.append(_TEX_TWO_ARG_FUNCS[cmd])
        elif cmd in _TEX_QUEST_FUNCTION_MAP:
            fn = _TEX_QUEST_FUNCTION_MAP[cmd]
            arg, j = _read_group(s, i)
            if arg is not None:
                if fn == "upright":
                    inner = arg.replace('"', "'")
                    out.append(f'{fn}("{inner}")')
                else:
                    out.append(f"{fn}({_latex_math_to_typst_regex(arg)})")
                i = j
            else:
                out.append(fn)
        elif cmd in _TEX_SYMBOL_MAP:
            # The separator is not cosmetic: adjacent commands otherwise glue
            # into one identifier (``4\pi\varepsilon`` -> ``4piepsilon``), which
            # Typst rejects as an unknown variable -- and the identifier
            # sanitizer will not touch a token that starts after a digit, so the
            # whole document's compile fails until the healer quotes it into a
            # visibly wrong equation.
            out.append(_TEX_SYMBOL_MAP[cmd] + " ")
        else:
            # Check if cmd starts with a known TeX symbol (glued command like \llF -> lt.double + F)
            split_found = False
            for prefix in sorted(_TEX_SYMBOL_MAP.keys(), key=len, reverse=True):
                if len(prefix) >= 2 and cmd.startswith(prefix):
                    remainder = cmd[len(prefix) :]
                    if remainder and (remainder[0].isupper() or remainder[0].isdigit()):
                        out.append(_TEX_SYMBOL_MAP[prefix] + " ")
                        i = i - len(cmd) + len(prefix)
                        split_found = True
                        break
            if split_found:
                continue

            # Unknown command: degrade safely so Typst compiles
            if len(cmd) > 1 and cmd.lower() not in _TYPST_MATH_KEYWORDS:
                out.append(f'"{cmd}"')
            else:
                # Same separator rule as the symbol map above: two adjacent
                # commands (``4\pi\varepsilon_0``) otherwise joined into one
                # identifier Typst cannot resolve.
                out.append(cmd + " ")
    s = "".join(out)

    # 4. Remaining bare LaTeX groups are Typst-invalid: convert to parens.
    s = s.replace("{", "(").replace("}", ")")

    # 4.1 Subscript sanitization: quote multi-letter subscripts to prevent Typst variable errors
    def _fix_paren_sub(m: re.Match[str]) -> str:
        prefix = m.group(1)
        inner = m.group(2).strip()
        clean = inner.replace('"', "").strip()
        if "," in clean:
            parts = [p.strip() for p in clean.split(",") if p.strip()]
            quoted_parts = [
                p if (p.lower() in _TYPST_MATH_KEYWORDS or len(p) == 1) else f'"{p}"' for p in parts
            ]
            return f"{prefix}_({', '.join(quoted_parts)})"
        if len(clean) > 1 and clean.lower() not in _TYPST_MATH_KEYWORDS:
            return f'{prefix}_("{clean}")'
        return f"{prefix}_({clean})"

    s = re.sub(r"([a-zA-Z0-9\)])_\(([^()]+)\)", _fix_paren_sub, s)

    def _fix_bare_sub(m: re.Match[str]) -> str:
        prefix = m.group(1)
        sub = m.group(2)
        if sub.lower() in _TYPST_MATH_KEYWORDS or len(sub) == 1:
            return f"{prefix}_{sub}"
        return f'{prefix}_("{sub}")'

    s = re.sub(r"([a-zA-Z0-9\)])_([a-zA-Z]{2,})(?![a-zA-Z0-9_])", _fix_bare_sub, s)
    # Post-normalization: bare 'cdot' -> 'dot.c', residual text(...) -> "..."
    s = re.sub(r"\bcdot\b", "dot.c", s)
    s = re.sub(r"\btext\s*\(([^\"'\)]+)\)", r'"\1"', s)

    # 5. Collapse redundant whitespace.
    return re.sub(r"[ \t]{2,}", " ", s).strip()


# Pandoc's Typst writer emits real code expressions for a few LaTeX
# constructs: ``\Big``-family delimiters -> ``#scale(x: ..%, y: ..%)[..]``,
# ``\boxed`` -> ``#box(...)``, ``\phantom`` -> ``#hide[..]``. They are pure
# layout calls, but their argument list is Typst *code*: ``#box(raw(read("x")))``
# carries no second ``#`` and would still run. A call therefore keeps its ``#``
# only when its bracket body is plain data; otherwise it is demoted to text.
_PANDOC_CALL_NAME_RE = re.compile(r"#(?:scale|box|hide)\s*([\(\[])")

#: Characters that mark a bracket body as code rather than a parameter list:
#: a nested call, a string literal or a raw block.
_PANDOC_CALL_BODY_FORBIDDEN = ("(", ")", '"', "'", "`")


def _pandoc_call_body_is_data(body: str) -> bool:
    """True when a ``#scale/#box/#hide`` argument body contains no code."""
    return not any(ch in body for ch in _PANDOC_CALL_BODY_FORBIDDEN)


def _sanitize_math_content(formula: str) -> str:
    """Neutralize Typst code-injection vectors inside math mode.

    In Typst math mode a ``#`` switches into code execution, letting a crafted
    formula run arbitrary Typst code (file access, loops). Every ``#`` is
    dropped except pandoc's own side-effect-free calls — ``\\Big``-family
    delimiters, ``\\boxed`` and ``\\phantom`` become ``#scale(``, ``#box(``
    and ``#hide[`` in the Typst writer — AND only when the call's argument body
    is plain data. A call whose body contains a nested call or string literal
    (``#box(raw(read("x")))``) is demoted to inert text, because its arguments
    are code. Blanket-stripping the safe calls turned chapter-3 Eq. (3.9) into
    the literal text ``"scale"(x: 180%, ...)``.
    """
    if "#" not in formula:
        return formula
    out: list[str] = []
    i = 0
    n = len(formula)
    while i < n:
        match = _PANDOC_CALL_NAME_RE.search(formula, i)
        if match is None:
            out.append(formula[i:].replace("#", ""))
            break
        out.append(formula[i : match.start()].replace("#", ""))
        open_idx = match.end() - 1
        close_idx = _matching_close(formula, open_idx)
        if close_idx is not None and _pandoc_call_body_is_data(formula[open_idx + 1 : close_idx]):
            out.append(formula[match.start() : close_idx + 1])
            i = close_idx + 1
        else:
            # Drop only this call's ``#`` and keep scanning, so a later safe
            # pandoc call is still protected.
            i = match.start() + 1
    return "".join(out)


# The lookahead form above only locates the ``#``; masking to end-of-call needs
# the bracket position and a balanced scan.
_PANDOC_TYPST_CALL_OPEN_RE = re.compile(r"#(?:scale|box|hide)\s*([\(\[])")


def _matching_close(text: str, open_idx: int) -> int | None:
    """Index of the bracket matching ``text[open_idx]``, or None if unbalanced."""
    depth = 0
    i = open_idx
    n = len(text)
    in_string = False
    while i < n:
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_string = False
            i += 1
            continue
        if ch == "\\":
            i += 2
            continue
        if ch == '"':
            in_string = True
            i += 1
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


def _mask_pandoc_typst_calls(expr: str) -> str:
    """Blank balanced pandoc ``#scale``/``#box``/``#hide`` calls for a syntax check.

    Pandoc's Typst writer emits these as code-mode content blocks, e.g.
    ``#box(inset: 3pt, [$ x + 1 $])``. The inner ``$`` is that block's own math
    mode and compiles fine, but :func:`is_typst_math_well_formed` scans in math
    mode and rejects every ``$`` — so ``\\boxed`` equations were degraded to a
    verbatim code span. Masking the call lets Gate 4 judge the math *around* it;
    the real, already-sanitized text is still what gets emitted.
    """
    if "#" not in expr:
        return expr
    out: list[str] = []
    cursor = 0
    for match in _PANDOC_TYPST_CALL_OPEN_RE.finditer(expr):
        open_idx = match.end() - 1
        if match.start() < cursor:
            continue
        close_idx = _matching_close(expr, open_idx)
        if close_idx is None:
            continue
        out.append(expr[cursor : match.start()])
        out.append("x")
        cursor = close_idx + 1
    out.append(expr[cursor:])
    return "".join(out)


def _is_emittable_math(expr: str) -> bool:
    """Gate 4 with pandoc layout calls masked (see :func:`_mask_pandoc_typst_calls`)."""
    return is_typst_math_well_formed(_mask_pandoc_typst_calls(expr))


def _has_residual_latex(text: str) -> bool:
    """True when converted math still carries raw LaTeX backslash commands or escaped markup.

    Typst math mode has no ``\\command`` syntax, so any residual (``\\frac``,
    ``\\varepsilon``, ``\\left`` …) or escaped markup (``\\$``, ``\\_``, ``\\"``)
    is uncompilable. Callers treat this as a converter failure and fall through
    to the next tier (regex, then verbatim) instead of emitting fail-open garbage.
    """
    return bool(re.search(r"\\[a-zA-Z]", text)) or "\\$" in text or "\\_" in text or '\\"' in text


# Content-preservation gate (Gate 5). Gate 4 only proves the converted math is
# *syntactically* valid Typst; a fragment is syntactically valid too. The
# b0040 regression produced ``cal(E)_("x s")= sqrt`` — balanced, no residual
# LaTeX, and semantically empty — so it passed Gate 4 and shipped. Comparing
# the number of math-bearing atoms before and after conversion catches any
# conversion that structurally loses most of the formula, independently of
# which tier produced it and of whether the source was well-formed.
_MATH_ATOM_RE = re.compile(r"[A-Za-z]+|\d+")
# Ignore very short formulas: a 3-atom input has no reliable ratio signal.
_MIN_ATOMS_FOR_CONTENT_CHECK = 8
# Calibrated against the 74 real formula blocks of the chapter-3 ledger:
# every legitimate conversion lands at ratio >= 0.79 (typ. ~1.0); the one
# silent truncation measured 0.08.
_MIN_MATH_ATOM_RATIO = 0.5


def _math_atom_count(text: str) -> int:
    """Count math-bearing atoms (letter runs and numbers) in math source/output."""
    return len(_MATH_ATOM_RE.findall(text))


def _content_lost(source: str, converted: str) -> bool:
    """True when ``converted`` dropped most of ``source``'s math content.

    Pure structural loss detector: independent of delimiter balance and of
    residual LaTeX, so it also catches tiers that "succeed" while silently
    emitting a fragment.
    """
    src_atoms = _math_atom_count(source)
    if src_atoms < _MIN_ATOMS_FOR_CONTENT_CHECK:
        return False
    return _math_atom_count(converted) < src_atoms * _MIN_MATH_ATOM_RATIO


_INTERTEXT_RE = re.compile(r"\\(?:short)?intertext\s*\{")


def _split_on_intertext(s: str) -> list[str]:
    """Split ``s`` on ``\\intertext{...}``, honouring brace-balanced payloads.

    A plain ``\\intertext\\s*\\{[^}]*\\}`` regex cannot handle a payload that
    itself contains braces — and Docling's payloads routinely do (any
    ``_{...}`` subscript). The regex then consumed only up to the payload's
    *first inner* ``}``, so the remainder of the payload leaked back into the
    equation. In chapter-3 that remainder happened to be a second copy of the
    same formula: Eq. (3.8) rendered twice, once per aligned line.
    """
    out: list[str] = []
    cursor = 0
    while True:
        m = _INTERTEXT_RE.search(s, cursor)
        if m is None:
            break
        out.append(s[cursor : m.start()])
        # m.end() sits just past "{"; _read_group wants the index of "{".
        arg, end = _read_group(s, m.end() - 1)
        # Unbalanced payload: drop only the command, keep the content rather
        # than silently discarding an unknown amount of math.
        cursor = end if arg is not None else m.end()
    out.append(s[cursor:])
    return out


# Docling drops the braces on ``\mathrm`` and leaves the label as separate
# letters: ``\mathrm f b`` where LaTeX would be ``\mathrm{fb}``. The
# converter's function branch needs a brace group, so without this repair it
# emitted the *bare* word ``upright`` and Typst rendered the letters
# u·p·r·i·g·h·t·f·b — chapter-3 Eq. (3.8) displayed ``V_upright fb`` where the
# source reads ``V_fb``. Single-letter runs are folded too (``\mathrm m`` →
# ``upright("m")``, which was already correct but is now uniform).
_BARE_ROMAN_RE = re.compile(
    r"\\(mathrm|text|operatorname|textrm|mbox)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+){0,4})"
    r"(?=\s*[^a-zA-Z]|$)"
)


def _rejoin_bare_roman(m: re.Match[str]) -> str:
    """``\\mathrm f b`` -> ``\\mathrm{fb}`` (multi-letter upright labels)."""
    label = re.sub(r"\s+", "", m.group(2))
    return "\\" + m.group(1) + "{" + label + "}"


# CodeFormulaV2 sometimes over-captures a region: the formula item's text then
# carries a trailing prose line (the "where ..." sentence that follows the
# equation, already extracted as its own narrative block). The misread
# subscripts collapse to punctuation (``Q_{,y}``, ``C_{;Y}``, ``T_{;}``) —
# no legitimate formula has a punctuation-only subscript — and the line still
# contains a prose connector like ``\text{with}``.
_OCR_PUNCT_SUBSCRIPT_RE = re.compile(r"_\s*\{\s*\\text\s*\{\s*[,;.]")
_OCR_PROSE_CONNECTOR_RE = re.compile(
    r"\\text\s*\{\s*(?:with|where|using|but|and|for)\b", re.IGNORECASE
)

# One trailing equation-number group: ``\quad (A.8) \\`` at end of string.
# Deliberately *not* wrapped in a ``+`` repetition. The separator run before
# ``(`` and the ``\s*`` after ``)`` both match whitespace, so nesting them
# inside an iterating group gives a run of N spaces 2**N partitions to try
# once the final ``$`` fails — a measured 12 s for 16 repeats. Repeats are
# peeled one at a time in the caller instead, which is linear.
_EQ_NUM_TAIL_RE = re.compile(
    # Trailing equation number after an EXPLICIT separator — ``\quad (3.2)``,
    # ``\qquad (3.2) \\`` or ``\\ (A.8)``. The number is a letter prefix plus
    # digits/dots only; a space or operator inside the parenthesis means it is
    # content, not a number, so a legitimate tail such as ``f(x) = (1 - x)`` is
    # no longer truncated to ``f(x) =`` (a bare `` (3.2)`` with only a space
    # before it is left alone too — equation numbers follow a separator).
    r"(?:\\qquad|\\quad|\\\\)[&\s]*"
    r"\(\s*(?:[A-Za-z]\s*[.\-]\s*)?\d[\dA-Za-z.]*\)\s*"
    r"(?:\\\\)?$"
)


def _is_ocr_prose_row(row: str) -> bool:
    """True for a CodeFormulaV2 region-overcapture row of prose residue."""
    return bool(_OCR_PUNCT_SUBSCRIPT_RE.search(row)) and bool(_OCR_PROSE_CONNECTOR_RE.search(row))


# Words that mark a glued-on narrative row in an OCR'd formula. Matched with
# boundaries on purpose: a substring test judged "elsewhere" a prose row (and
# deleted the second branch of a ``cases`` environment with it), "uniform" or
# "information" likewise matched "for".
_OCR_PROSE_HINT_RE = re.compile(
    r"\b(?:function|equation|using|for|with|where|can be|given by|represented"
    r"|defined as|written as)\b"
)


def _clean_ocr_formula(raw: str) -> str:
    """Clean up OCR-damaged formula text from Docling extraction.

    Common OCR artifacts in academic PDFs:
    - Spaced-out letters: 'A t t e n t i o n' -> 'Attention'
    - Garbled tags: 'not nolimits _ { mathbb { F } }' -> removed
    - Truncated fragments: 'P E _ { (' -> dropped as invalid
    - Trailing equation numbers: '& & ( 1 )' -> removed
    """
    s = raw.strip()
    if not s:
        return s

    # Drop Docling-escaped display delimiters (``\$\$``): they are wrapper
    # artifacts, not math content, and they make pandoc's LaTeX reader pass
    # the whole formula through unconverted (exit 0, garbage out).
    s = s.replace("\\$", "")

    # Drop natural-language ``\text{...}`` tails glued by Docling (the prose
    # survives in the following narrative block; keeping it renders English
    # words inside the equation line).
    s = _strip_trailing_text_prose(s)
    s = _strip_leading_text_prose(s)

    # Strip OCR-glued sentence headers e.g. "Eqs. (3.7), (3.8) can be written as a single equation:"
    s = re.sub(
        r"^[\s&\\\\]*(?:E\s*q\s*s?|Equation|Eqs?)\b.*?(?:can be written as|is given by|as follows|given by)[^:]*:\s*\}?\s*(?:\\\\)?",
        "",
        s,
        flags=re.IGNORECASE,
    )
    # Strip multiline glued prose line before \\ (e.g. "a \text{function on ... using ...} \\ Q_{inv} = ...")
    if "\\\\" in s:
        parts = re.split(r"\s*\\\\\s*", s)
        if len(parts) > 1:
            rel = re.compile(r"(=|\\approx|\\le|\\ge|\\sim|\\equiv|\\propto)")
            if not rel.search(parts[0]) and (
                _OCR_PROSE_HINT_RE.search(parts[0]) or parts[0].strip().endswith((":", r"\colon"))
            ):
                s = " \\\\ ".join(parts[1:])
            # Also strip trailing prose line without relation operator (e.g. "\\ v and q are represented by")
            parts = re.split(r"\s*\\\\\s*", s)
            if len(parts) > 1 and (
                (not rel.search(parts[-1]) and _OCR_PROSE_HINT_RE.search(parts[-1]))
                or _is_ocr_prose_row(parts[-1])
            ):
                dropped = len(parts[-1].strip())
                s = " \\\\ ".join(parts[:-1])
                if _is_ocr_prose_row(parts[-1]):
                    # Never silent: the payload duplicates the following
                    # narrative line (already translated) and cannot ship as
                    # an equation row.
                    logger.warning(
                        "Dropped %d chars of OCR prose residue while cleaning a formula",
                        dropped,
                    )

    # Strip OCR-glued sentence tails e.g. "\\ \text {where} \psi_{pert} is given by...", "In order to find...", etc.
    m_where = re.search(
        # The lookbehind is load-bearing: without it "where" matched inside
        # "elsewhere"/"anywhere" and the cutter dropped the rest of a real
        # formula (and unbalanced its brace group) on a prose word the
        # equation legitimately contained.
        r"[\s\\\\]*(?:\\text\s*\{)?\s*\\?\s*(?<![A-Za-z])(?:where|using this approximation|in order to|it is also possible to|note that|we can|then,? we have)\b",
        s,
        re.IGNORECASE,
    )
    if m_where:
        # Everything after a prose glue word ("where …", "we can …") is narrative
        # that the surrounding text already carries, so it is dropped from the
        # formula — but dropping source characters silently is exactly the kind
        # of loss a FidelityReport can never recover. Log it.
        logger.warning(
            "Truncated %d chars of formula tail at prose glue %r",
            len(s) - m_where.start(),
            s[m_where.start() : m_where.start() + 40].strip(),
        )
        s = s[: m_where.start()].rstrip()

    # Clean dangling trailing \text{ or \text
    s = re.sub(r"\\text\s*\{\s*$", "", s).rstrip()

    # Drop orphaned \right) after brackets (common OCR hallucination in long multiline equations)
    s = re.sub(r"(\]\s*)\\right\s*\)", r"\1", s)
    # Close unclosed \text{tm} } in exponent fractions before + or -
    s = re.sub(r"(\\\s*text\s*\{\s*tm\s*\}\s*\})\s*([+-])", r"\1 } \2", s)

    # Strip trailing backslashes/ampersands/spacing left after tail stripping.
    # ``\quad`` / ``\qquad`` must be in the class: Docling ends display
    # equations with ``\quad \\`` (chapter-3 Eq. 3.13), and a surviving quad
    # reaches Typst as the literal word "quad" after the equation.
    s = re.sub(r"(?:[\s&\\\\]|\\quad|\\qquad|\\,|\\;|\\:|\\!)+$", "", s).rstrip()

    # Strip orphan leading subscripts/periods (e.g. "_ { d s } . \\")
    s = re.sub(r"^[\s&\\\\]*_\s*\{[^}]*\}\s*\.?[\s\\\\]*", "", s)

    # Docling emits brace-less roman labels (\mathrm f b). Rejoin them into a
    # real group so the converter applies the style instead of emitting a bare
    # "upright" word (see _BARE_ROMAN_RE).
    s = _BARE_ROMAN_RE.sub(_rejoin_bare_roman, s)

    # Clean \intertext prose / deduplicate OCR repeated equation branches
    if r"\intertext" in s:
        len_before_intertext = len(s)
        parts = _split_on_intertext(s)
        p1 = parts[0].strip()
        p2 = parts[1].strip() if len(parts) > 1 else ""
        clean_p1 = re.sub(r"[\s&\\\\]+", "", re.sub(r"\(\s*[0-9A-Za-z.]+\s*\)", "", p1))
        clean_p2 = re.sub(r"[\s&\\\\]+", "", re.sub(r"\(\s*[0-9A-Za-z.]+\s*\)", "", p2))
        if (
            clean_p1
            and clean_p2
            and (
                clean_p1 == clean_p2
                or clean_p1.startswith(clean_p2)
                or clean_p2.startswith(clean_p1)
                or clean_p1.endswith(clean_p2)
                or clean_p2.endswith(clean_p1)
                or clean_p1 in clean_p2
                or clean_p2 in clean_p1
            )
            or not clean_p2
            or any(
                w in p2.lower()
                for w in ["show", "obtain", "solve", "using", "where", "with", "from", "eq", "beta"]
            )
        ):
            s = p1
        else:
            s = p1 + " " + p2
        dropped = len_before_intertext - len(s)
        if dropped > 0:
            # Never silent: the payload is either a duplicate of this very
            # equation or prose Docling glued into the formula block (which
            # cannot be translated here anyway, since FORMULA blocks are
            # skip_translate). Surfacing the count keeps it auditable.
            logger.warning(
                "Dropped %d chars of \\intertext payload while cleaning a formula "
                "(Docling repeat or glued prose)",
                dropped,
            )

    # Clean repeated OCR leader dots e.g. \dot { \cdot } \quad ...
    s = re.sub(
        r"(?:\\\\(?:\\\\\s*)?)?(?:\\dot\s*\{\s*\\cdot\s*\}\s*(?:\\quad|\s)*){2,}",
        "",
        s,
    ).strip()
    s = re.sub(r"(?:\\quad|\s)*(?:\\cdot\s*){3,}", "", s).strip()

    # Strip trailing and embedded equation numbers e.g. (3.2), && (3.2), \quad (A.8) \\
    s = s.strip()
    tail = _EQ_NUM_TAIL_RE.search(s)
    while tail:
        s = s[: tail.start()].rstrip()
        tail = _EQ_NUM_TAIL_RE.search(s)
    s = re.sub(r"(?:&\s*)+\(\s*(?:[A-Za-z]\s*[\.\-]\s*)?\d[\d.A-Za-z\s\-–]*\)", "", s)
    # Strip OCR-glued equation numbers embedded inside brackets/terms e.g. [\psi_{pert} (11) 2 V_{um}]
    s = re.sub(
        r"(?<=[a-zA-Z_\}\]\)])\s*\(\s*(?:[A-Za-z]\s*[\.\-]\s*)?\d+(?:\.\d+)?\s*\)\s*(?=[0-9a-zA-Z_\\\[\{])",
        " ",
        s,
    )

    # Clean Docling OCR 'upright' artifacts (e.g. \text{upright fb} -> \text{fb}, C_{upright ox} -> C_{ox})
    s = re.sub(r"\\(?:text|mathrm|operatorname)\s*\{\s*upright\s+([^}]+)\}", r"\\text{\1}", s)
    s = re.sub(r"(?<=_)\{\s*upright\s+([^}]+)\}", r"{\\text{\1}}", s)
    s = re.sub(r"\bupright\s+([a-zA-Z0-9_]+)", r"\1", s)

    # Clean truncated trailing radicals without argument (e.g. '= \text{sqrt}' or '= \sqrt')
    s = re.sub(r"=\s*(?:\\?(?:text\s*\{\s*)?sqrt(?:\s*\})?|\\sqrt\s*\{\s*\})\s*$", "", s).rstrip()

    # Detect spaced-out single letters pattern (OCR artifact).
    # If >50% of non-space tokens are single characters, attempt to rejoin.
    tokens = s.split()
    if tokens:
        single_char_count = sum(1 for t in tokens if len(t) == 1)
        ratio = single_char_count / len(tokens)
        if (
            ratio > 0.5
            and "\\" not in s
            and "_" not in s
            and "^" not in s
            and not _DOCLING_RE.search(s)
        ):
            # Try to reconstruct known function/variable names from spaced letters
            collapsed = s.replace(" ", "")
            # Known math function names to restore
            known_funcs = [
                "Attention",
                "MultiHead",
                "Concat",
                "softmax",
                "FFN",
                "head",
                "model",
                "rate",
                "min",
                "max",
                "cos",
                "sin",
                "pos",
                "step",
                "num",
                "warmup",
                "steps",
            ]
            for func in known_funcs:
                collapsed = collapsed.replace(func.lower(), func)
            s = collapsed

    # Remove garbled OCR artifacts
    s = re.sub(r"not\s*nolimits\s*_\s*\{\s*mathbb\s*\{[^}]*\}\s*\}", "", s)
    s = re.sub(r"nolimits\s*_\s*\{\s*\}", "", s)

    # Drop truncated/invalid fragments (unbalanced parens with no content)
    if s.count("(") != s.count(")") and len(s.strip("$ \t\n\r{()}_^")) < 5:
        return ""  # Fragment too broken to be useful

    return s.strip()


def _balanced_delimiters(text: str) -> bool:
    """Paren/bracket balance check on converted Typst math (no braces remain)."""
    depth_paren = depth_bracket = 0
    for ch in text:
        if ch == "(":
            depth_paren += 1
        elif ch == ")":
            depth_paren -= 1
        elif ch == "[":
            depth_bracket += 1
        elif ch == "]":
            depth_bracket -= 1
        if depth_paren < 0 or depth_bracket < 0:
            return False
    return depth_paren == 0 and depth_bracket == 0


# Minimum characters a split's right-hand side must carry for the relation
# to count as a chain link; shorter than this it is a terminal ("= 0").
_MIN_CHAIN_TAIL_CHARS = 12


_TYPST_ROW_SPLIT_RE = re.compile(r"\\\n\s*")


# Display-width model for Typst math (reflow column fit).
#
# Raw ``len()`` counts Typst markup — ``upright("..")``, ``\,`` / ``\/``
# escapes, ``quad`` indents — inflating chapter-3 equations ~1.6-2x. Judging
# fit on markup chars shredded single-line originals (Eq. 3.14, 60 glyphs but
# 105 markup chars) into two lines. Glyph count tracks the 471pt page-strict
# column far better (fractions/sums stack vertically, so this still errs on
# the safe side for 2D constructs).
_MATH_FIT_GLYPHS = 78
# Per-equation fallback size for equations that miss the body-size budget but
# fit one step down — standard print practice, and what the source textbook
# itself does for its wide display equations. Applied via a scoped
# ``#block[#show ...]`` so labels, refs and the equation counter are untouched.
# Budget: 96 glyphs × ~4.25pt at 8.5pt ≈ 410pt < 431pt usable (471pt column
# minus the equation number) — chapter-3 Eq. 3.11 rows (84/93) fit; anything
# wider still re-derives breaks instead of overflowing.
_MATH_SHRINK_SIZE_PT = 8.5
_MATH_FIT_GLYPHS_SMALL = 96

_MARKUP_NOISE_RES = (
    # Function/font wrappers print no glyphs of their own (frac stacks its
    # args vertically; scripting attaches without advancing much).
    re.compile(r"\b(?:upright|bb|bold|cal|frak|sans|serif|mono)\("),
    re.compile(r"[_^]\("),
    re.compile(r'"'),
    re.compile(r"\\([(),/])"),
    re.compile(r"\\,"),
)
_STYLED_LITERAL_RE = re.compile(r"\b(?:upright|bb|bold|cal|frak|sans|serif|mono)\(\"([^\"]*)\"\)")


def _strip_balanced(t: str, opener: str) -> str:
    """Drop ``opener`` and its matching close paren, keeping the inner text.

    ``frac(V,2)`` -> ``V,2``; ``Q_(inv)`` -> ``Qinv``. Unbalanced input is
    returned unchanged (fail-safe: never destroy content while measuring).
    """
    res: list[str] = []
    i = 0
    n = len(t)
    m = len(opener)
    while True:
        j = t.find(opener, i)
        if j < 0:
            res.append(t[i:])
            break
        res.append(t[i:j])
        depth = 0
        end = -1
        for k in range(j + m - 1, n):
            if t[k] == "(":
                depth += 1
            elif t[k] == ")":
                depth -= 1
                if depth == 0:
                    end = k
                    break
        if end < 0:
            res.append(t[j:])
            break
        res.append(t[j + m : end])
        i = end + 1
    return "".join(res)


def _widest_frac_branch(t: str) -> str:
    """Replace balanced ``frac(NUM,DEN)`` with the wider branch's text.

    A stacked fraction is only as wide as its widest part; summing both
    branches (as naive char counts do) doubles the estimate. Unbalanced or
    comma-less input is left untouched.
    """
    res: list[str] = []
    i = 0
    n = len(t)
    while True:
        j = t.find("frac(", i)
        if j < 0:
            res.append(t[i:])
            break
        res.append(t[i:j])
        # Depth counts from inside frac's own paren (seen first at j+4, held
        # at 1): the NUM/DEN separator comma therefore sits at depth 1, and
        # frac's own close paren is the one returning to 0.
        depth = 0
        end = -1
        comma = -1
        for k in range(j + 4, n):
            if t[k] == "(":
                depth += 1
            elif t[k] == ")":
                if depth <= 1:
                    end = k
                    break
                depth -= 1
            elif t[k] == "," and depth == 1 and comma < 0:
                comma = k
        if end < 0 or comma < 0:
            res.append(t[j:])
            break
        num = t[j + 5 : comma]
        den = t[comma + 1 : end]
        res.append(num if _math_display_len(num) >= _math_display_len(den) else den)
        i = end + 1
    return "".join(res)


def _math_display_len(eq: str) -> int:
    """Approximate rendered glyph count of Typst math (markup excluded).

    Whitespace in Typst math source is insignificant (layout is automatic),
    so it is dropped entirely — except inside ``"quoted"`` literals, where
    spaces really print. ``quad``-family indents carry no printed glyph once
    layout whitespace is removed (matching the historical ``&quad``
    accounting); ``&=`` and line breaks count as their visible remainder.
    Two-dimensional constructs
    are measured, not summed: ``frac`` contributes its widest branch, script
    and style wrappers contribute only their inner text.
    """
    t = _STYLED_LITERAL_RE.sub(r"\1", eq)
    t = _widest_frac_branch(t)
    for opener in (
        "upright(",
        "bb(",
        "bold(",
        "cal(",
        "frak(",
        "sans(",
        "serif(",
        "mono(",
        "sqrt(",
        "root(",
        "sum(",
        "product(",
        "integral(",
        "_(",
        "^(",
    ):
        t = _strip_balanced(t, opener)
    lits = re.findall(r'"(?:\\.|[^"\\])*"', t)
    t = re.sub(r'"(?:\\.|[^"\\])*"', "", t)
    for pat in _MARKUP_NOISE_RES:
        t = pat.sub(lambda m: m.group(1) if m.lastindex else "", t)
    t = re.sub(r"&?(?:quad|qquad|nbsp|ensp|emsp)\b", "  ", t)
    t = t.replace("&=", "=")
    t = re.sub(r"\\\n", "", t)
    t = re.sub(r"\s+", "", t)
    return len(t) + sum(max(len(lit) - 2, 0) for lit in lits)


# Fail-safe net before emission: whatever tier left spacing/layout residue at
# the end of the equation (bare or ``"quoted"`` quad/qquad, a dangling
# ``\\\n quad`` row, stray ``&``) is stripped. A trailing ``quad`` word is
# never legitimate math content — the chapter-3 Eq. 3.13 regression shipped
# ``... d Q_("inv") "quad"`` and printed the word "quad" after the equation.
_TRAILING_TYPST_NOISE_RE = re.compile(
    r"(?:\s|\\\n\s*|&quad\b|&\s*|\"quad\"|\"qquad\"|\bquad\b|\bqquad\b|\\quad|\\qquad)+$"
)


def _strip_trailing_typst_noise(eq: str) -> str:
    """Remove layout residue dangling at the end of converted Typst math."""
    return _TRAILING_TYPST_NOISE_RE.sub("", eq).rstrip()


def _split_long_typst_equation(eq: str, max_len: int = _MATH_FIT_GLYPHS) -> str:
    """Adaptively split overly long display equations at top-level relation signs to prevent page overflow.

    ``max_len`` is a *display glyph* budget (see :func:`_math_display_len`),
    not a markup character count: markup inflates equations ~1.6x, and the
    old character budget shredded single-line originals such as chapter-3
    Eq. 3.14 into two lines.

    Continuation lines are indented with Typst's ``quad``, never LaTeX's
    ``\\quad``: Typst math has no backslash-command syntax, so ``&\\quad``
    parses as the alignment point followed by two undefined identifiers and
    the whole equation dies with ``unknown variable: uad``. Because this
    helper runs after Gate 4 (a syntax-only check that ``&\\quad`` passes),
    that failure was invisible until the render-time compile probe degraded
    the line — 8 of chapter-3's 74 formula blocks were lost this way while
    the identical unsplit input compiled fine.
    """
    if _math_display_len(eq) < max_len or "\\ " in eq:
        return eq

    # An author's explicit break is worth keeping -- it is where the source
    # actually broke -- while every resulting row fits the column at the body
    # size, or fits one step down (the emitter shrink-wraps that band at
    # ``_MATH_SHRINK_SIZE_PT`` instead of re-deriving the breaks). Typst does
    # not wrap display math, so an over-long row sticks out past the page
    # edge, which is worse than re-deriving the break ourselves -- but only
    # rows that miss even the shrunk budget take that path now (chapter-3
    # Eq. 3.11 keeps its two author rows at 8.5pt instead of becoming five).
    rows = [r.strip() for r in _TYPST_ROW_SPLIT_RE.split(eq) if r.strip()]
    if len(rows) > 1:
        if max(_math_display_len(r) for r in rows) <= _MATH_FIT_GLYPHS_SMALL:
            return eq
        # Too long to keep as-is: flatten back to one row and re-derive below.
        eq = " ".join(
            re.sub(r"^(?:&)?(?:quad|qquad)\b\s*", "", r)
            .replace('"quad"', "")
            .replace('"qquad"', "")
            for r in rows
        ).strip()
        eq = re.sub(r"\s{2,}", " ", eq)

    # Find top-level relation signs at depth 0. A ``=``/``+``/``-`` inside a
    # quoted string literal (e.g. ``upright("p-type")``) is text, not an
    # operator — splitting there tears the label apart.
    depth = 0
    in_string = False
    splits: list[int] = []
    for i, ch in enumerate(eq):
        if ch == '"' and (i == 0 or eq[i - 1] != "\\"):
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        elif (
            depth == 0
            and ch == "="
            and i > 0
            and eq[i - 1] not in "!<>="
            and (i + 1 == len(eq) or eq[i + 1] != "=")
        ):
            splits.append(i)

    # A terminal relation — a trailing "= 0" — is not a chain link. Breaking
    # before it strands the tail on its own line while the first line stays
    # full-width (chapter-3 Eq. (3.11) shipped exactly that: a lone "&= 0"
    # under one long line, and the equation number pulled to the vertical
    # middle of the resulting two-row block). Keep only the splits that carry a
    # substantive right-hand side.
    splits = [p for p in splits if len(eq) - p > _MIN_CHAIN_TAIL_CHARS]

    if len(splits) >= 2:
        # Multiple '=' (e.g. A = B = C): align at second and subsequent '='
        res = eq[: splits[0]] + "&="
        prev = splits[0] + 1
        for s_pos in splits[1:]:
            res += eq[prev:s_pos].rstrip() + " \\\n  &="
            prev = s_pos + 1
        res += eq[prev:]
        return res
    elif len(splits) == 1 and _math_display_len(eq[: splits[0]]) < max_len // 2:
        # Single '=' with very long RHS: break at top-level '+' or '-' after length threshold
        lhs = eq[: splits[0]]
        rhs = eq[splits[0] + 1 :]
        r_depth = 0
        r_in_string = False
        r_splits: list[int] = []
        for j, c in enumerate(rhs):
            if c == '"' and (j == 0 or rhs[j - 1] != "\\"):
                r_in_string = not r_in_string
                continue
            if r_in_string:
                continue
            if c in "([{":
                r_depth += 1
            elif c in ")]}":
                r_depth = max(0, r_depth - 1)
            elif r_depth == 0 and c in ("+", "-") and j >= 35:
                r_splits.append(j)
        if r_splits:
            res = lhs + "&= "
            prev = 0
            for r_pos in r_splits:
                # Unlike author breaks, this row-1 ``&=`` IS an alignment
                # point: the ``&quad`` continuation pairs with it, so the
                # wrapped terms align under the RHS (no lone ``&``).
                res += rhs[prev:r_pos].rstrip() + " \\\n  &quad " + rhs[r_pos] + " "
                prev = r_pos + 1
            res += rhs[prev:]
            return res
    return eq


def _emit_formula_math(raw_content: str, block_id: str) -> str:
    """Render one formula block as a Typst math line — never silently drop.

    Gate 4 (verified rendering): converted math with unbalanced delimiters,
    or content that cleaning reduces to nothing, falls back to a verbatim
    monospace line so the content stays visible (fail-loud) instead of
    vanishing from the document. Callers log the fallback via the warning
    emitted here.
    """
    stripped = (raw_content or "").strip("$ \t\n\r")
    clean = _clean_ocr_formula(stripped)
    if not clean:
        logger.warning(
            "Formula block %s unusable after cleaning; emitting verbatim source",
            block_id,
        )
        verbatim = stripped.replace("`", "'").replace("\n", " ")
        return f"`{verbatim}`" if verbatim else f"// Empty formula block: {block_id}"
    if _needs_latex_conversion(clean):
        clean = _latex_math_to_typst(clean)
    clean = _sanitize_math_content(clean)
    if not _is_emittable_math(clean):
        # Gracefully handle single unclosed/unopened trailing delimiters
        trimmed = clean.rstrip()
        if trimmed.count("(") < trimmed.count(")") and trimmed.endswith(")"):
            excess = trimmed.count(")") - trimmed.count("(")
            for _ in range(excess):
                if trimmed.endswith(")"):
                    trimmed = trimmed[:-1].rstrip()
            if _is_emittable_math(trimmed):
                clean = trimmed
        elif trimmed.count("[") < trimmed.count("]") and trimmed.endswith("]"):
            excess = trimmed.count("]") - trimmed.count("[")
            for _ in range(excess):
                if trimmed.endswith("]"):
                    trimmed = trimmed[:-1].rstrip()
            if _is_emittable_math(trimmed):
                clean = trimmed

    # Split BEFORE the Gate 4 check so the emitted line is the thing that gets
    # validated. Running it after the gate left the splitter's output unchecked
    # — which is how the ``&\quad`` bug shipped: `\quad` is a LaTeX command,
    # ``is_typst_math_well_formed`` only counts delimiters so it passed, and
    # 9 formula blocks died at the render-time compile probe instead.
    clean = _split_long_typst_equation(clean)
    # Last-chance net: no tier may leave spacing residue (``"quad"`` word,
    # dangling ``\\\n quad`` row) at the end of the equation — it would print
    # literally after the formula (chapter-3 Eq. 3.13).
    clean = _strip_trailing_typst_noise(clean)

    if not _is_emittable_math(clean) or _has_residual_latex(clean):
        logger.warning(
            "Formula block %s unusable after conversion (residual LaTeX or "
            "malformed math); emitting verbatim source",
            block_id,
        )
        # Show the *cleaned* source: already-handled artifacts (orphan
        # \text{prose}, $$\$$ wrappers) stay out of the reader's face while
        # the residual LaTeX that actually broke conversion remains visible.
        verbatim = clean.replace("`", "'").replace("\n", " ")
        return f"`{verbatim}`" if verbatim else f"// Empty formula block: {block_id}"
    return f"$ {clean} $"
