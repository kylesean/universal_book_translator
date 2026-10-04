"""Shared LaTeX-command → Typst-symbol vocabulary (the render single source).

Two converters own a symbol table:

- ``typst_math`` (display-block reflow math) → :data:`DISPLAY_SYMBOLS`
- ``overlay_text`` (inline, in-place math)   → :data:`INLINE_SYMBOLS`

They serve different output contexts, so this module keeps them deliberately
separate while single-sourcing the vocabulary:

- :data:`_SHARED`       — commands both recognise, with the display spelling.
- :data:`_DISPLAY_ONLY` — commands the display converter owns; the inline
  table admits them too (see the note on ``INLINE_SYMBOLS``) because
  paragraph-level context math shares the display vocabulary.
- :data:`_INLINE_ONLY`  — commands only the inline converter recognises
  (Greek letters, the frac/sqrt/det branch, and the operator symbols the
  inline converter gained for real papers).
- :data:`DISPLAY_VALUE_OVERRIDES` / :data:`INLINE_VALUE_OVERRIDES` — the seven
  shared commands each side intentionally spells differently (both valid Typst).

A value fix (e.g. ``\\cap`` → ``inter``) therefore lands once and reaches both
tables, so the two maps cannot silently drift apart again.

:data:`INLINE_RENDERABLE_COMMANDS` is the set the inline converter can actually
emit: the symbol-table keys plus the commands handled by dedicated pre-passes
(font styles, upright text, accents, spacing, delimiter sizing). The QE's
``RENDERABLE_LATEX_COMMANDS`` imports it, so "what the gate allows" and "what
the renderer can compile" are the same set by construction rather than by a
hand-maintained mirror that drifts.

The module lives in ``ubt.core`` (not the adapter) so the gate can read it
without importing the adapter package; the adapter re-exports it for the
historical import path.
"""

from __future__ import annotations

# Commands recognised by BOTH converters. Values are the display spelling;
# INLINE_VALUE_OVERRIDES restores the seven divergent inline spellings.
_SHARED: dict[str, str] = {
    "approx": "approx",
    "cdot": "dot.c",
    "cos": "cos",
    "div": "div",
    "exp": "exp",
    "geq": ">=",
    "gg": "gt.double",
    "infty": "infinity",
    "int": "integral",
    "leq": "<=",
    "lim": "lim",
    "ll": "lt.double",
    "ln": "ln",
    "log": "log",
    "max": "max",
    "min": "min",
    "neq": "!=",
    "partial": "partial",
    "pm": "plus.minus",
    "prod": "product",
    "sin": "sin",
    "sum": "sum",
    "tan": "tan",
    "times": "times",
    "circ": "compose",
    "mapsto": "|->",
    "to": "->",
    "rightsquigarrow": "⇝",
    "varepsilon": "epsilon.alt",
    "varphi": "phi.alt",
    "vartheta": "theta.alt",
}

# Commands only the display converter (typst_math) maps.
_DISPLAY_ONLY: dict[str, str] = {
    "Leftarrow": "<=",
    "Leftrightarrow": "<=>",
    "Rightarrow": "=>",
    "argmax": "arg",
    "argmin": "arg",
    "arrow": "->",
    "arrow.l": "<-",
    "arrow.r": "->",
    "asymp": "approx",
    "because": "because",
    "bigcap": "inter.big",
    "bigcup": "union.big",
    "cap": "inter",
    "cdots": "dots.h",
    "colon": ":",
    "cong": "approx",
    "cup": "union",
    "dots": "dots.h",
    "equiv": "equiv",
    "exists": "exists",
    "forall": "forall",
    "ge": ">=",
    "gggtr": "gt.triple",
    "hbar": "ℏ",
    "in": "in",
    "inf": "inf",
    "land": "and",
    "ldots": "dots.h",
    "le": "<=",
    "leftarrow": "<-",
    "llless": "lt.triple",
    "lnot": "not",
    "lor": "or",
    "mp": "minus.plus",
    "nabla": "nabla",
    "ne": "!=",
    "neg": "not",
    "notin": "not in",
    "parallel": "parallel",
    "perp": "perp",
    "propto": "prop",
    "rightarrow": "->",
    "rightarrowtail": "->",
    "setminus": "without",
    "sim": "tilde.op",
    "simeq": "approx",
    "subset": "subset",
    "subseteq": "subset.eq",
    "sup": "sup",
    "supset": "supset",
    "supseteq": "supset.eq",
    "therefore": "therefore",
    "triangleq": "equiv",
    "varnothing": "emptyset",
    "varpi": "pi.alt",
    "varpropto": "prop",
    "varrho": "rho.alt",
    "varsigma": "sigma.alt",
    "vdash": "tack.r",
    "dashv": "tack.l",
    "models": "models",
    "emptyset": "emptyset",
    "vdots": "dots.v",
}

#: Display-only commands whose value is not safe for the inline converter's
#: word-run quoter (it would quote the string literal's contents). They are in
#: :data:`DISPLAY_SYMBOLS` but deliberately not in :data:`INLINE_SYMBOLS`.
_DISPLAY_TEXT_ONLY: dict[str, str] = {
    "ReLU": 'upright("ReLU")',
}

# Commands only the inline converter (overlay_text) maps.
_INLINE_ONLY: dict[str, str] = {
    "Delta": "Delta",
    "Gamma": "Gamma",
    "Lambda": "Lambda",
    "Omega": "Omega",
    "Phi": "Phi",
    "Pi": "Pi",
    "Psi": "Psi",
    "Sigma": "Sigma",
    "Theta": "Theta",
    "Xi": "Xi",
    "alpha": "alpha",
    "beta": "beta",
    "chi": "chi",
    "delta": "delta",
    "det": "det",
    "epsilon": "epsilon",
    "eta": "eta",
    "frac": "frac",
    "gamma": "gamma",
    "iota": "iota",
    "kappa": "kappa",
    "lambda": "lambda",
    "mu": "mu",
    "nu": "nu",
    "omega": "omega",
    "phi": "phi",
    "pi": "pi",
    "psi": "psi",
    "rho": "rho",
    "sigma": "sigma",
    "sqrt": "sqrt",
    "tau": "tau",
    "theta": "theta",
    "xi": "xi",
    "zeta": "zeta",
    # Operator/symbol vocabulary real papers need but the table lacked, so the
    # model's conventional LaTeX was flagged as unrenderable and quarantined.
    # Every spelling below was verified to compile with TypstMathProbe.
    "Im": "Im",
    "Re": "Re",
    "bullet": "bullet",
    "circledast": "ast.o",
    "circledcirc": "circle.small",
    "coloneqq": ":=",
    "Cup": "union.big",
    "Cap": "inter.big",
    "curlyeqprec": "prec.eq.curly",
    "diamond": "diamond",
    "divideontimes": "times.div",
    "doteq": "eq.dot",
    "hookrightarrow": "arrow.r.hook",
    "iff": "<=>",
    "langle": "<.",
    "leadsto": "~>",
    "leftharpoonup": "harpoon.lt",
    "longleftarrow": "<--",
    "longrightarrow": "-->",
    "ltimes": "times.l",
    "mid": "mid",
    "multimap": "arrow.r.long.bar",
    "ngeq": "gt.eq.not",
    "nleq": "lt.eq.not",
    "not": "not",
    "nprec": "prec.not",
    "nvdash": "tack.r.not",
    "oplus": "plus.o",
    "prec": "prec",
    "rangle": ".>",
    "rhd": "triangle.r",
    "rightleftharpoons": "harpoon.rt.lb",
    "rightharpoonup": "harpoon.rt",
    "risingdotseq": "eq.dot",
    "rtimes": "times.r",
    "sqsubseteq": "subset.eq.sq",
    "sqsupseteq": "supset.eq.sq",
    "square": "square",
    "Box": "square.stroked",
    "subsetneq": "subset.neq",
    "supsetneq": "supset.neq",
    "top": "top",
    "bot": "bot",
    "triangleleft": "triangle.l",
    "triangleright": "triangle.r",
    "twoheadrightarrow": "arrow.r.twohead",
    "approxeq": "approx.eq",
    "vDash": "models",
    # Delimiters the model writes with \left...\right; the sizing modifiers are
    # stripped by the converter, these supply the glyph. All verified to compile.
    "lfloor": "floor.l",
    "rfloor": "floor.r",
    "lceil": "ceil.l",
    "rceil": "ceil.r",
    "lvert": "bar.v",
    "rvert": "bar.v",
    "lVert": "bar.v.double",
    "rVert": "bar.v.double",
    "vert": "bar.v",
    "Vert": "bar.v.double",
    "lbrace": "brace.l",
    "rbrace": "brace.r",
    "llbracket": "⟦",
    "rrbracket": "⟧",
    # Arrow aliases the model uses and the Unicode map already names.
    "leftrightarrow": "<->",
    "longleftrightarrow": "arrow.l.r.long",
    "Longleftrightarrow": "arrow.l.r.double.long",
    "Longrightarrow": "arrow.r.double.long",
    "Longleftarrow": "arrow.l.double.long",
}

# The seven shared commands each context spells differently (both valid Typst).
DISPLAY_VALUE_OVERRIDES: dict[str, str] = {
    "cdot": "dot.c",
    "geq": ">=",
    "leq": "<=",
    "neq": "!=",
    "varepsilon": "epsilon.alt",
    "varphi": "phi.alt",
    "vartheta": "theta.alt",
}
INLINE_VALUE_OVERRIDES: dict[str, str] = {
    "cdot": "dot",
    "geq": "gt.eq",
    "leq": "lt.eq",
    "neq": "eq.not",
    "varepsilon": "epsilon",
    "varphi": "phi",
    "vartheta": "theta",
}

#: Commands the inline converter handles with a dedicated pre-pass rather than
#: a symbol-table entry. Kept beside the table so the renderable set is derived
#: from one place; each name mirrors a regex in ``overlay_text.typstify_math``.
SPECIAL_LATEX_COMMANDS: frozenset[str] = frozenset(
    {
        # font styles -> bb()/frak()/cal()
        "mathbb",
        "mathfrak",
        "mathcal",
        # upright strings
        "text",
        "mathrm",
        # math accents
        "hat",
        "tilde",
        "vec",
        "bar",
        "overline",
        "dot",
        "ddot",
        # explicit spacing
        "quad",
        "qquad",
        "enspace",
        "thinspace",
        "thickspace",
        "medspace",
        "negthinspace",
        # delimiter sizing modifiers and auto-sizing delimiters
        "big",
        "Big",
        "bigg",
        "Bigg",
        "left",
        "right",
    }
)


def _merged(*layers: dict[str, str]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for layer in layers:
        merged.update(layer)
    return merged


DISPLAY_SYMBOLS: dict[str, str] = _merged(
    _SHARED, DISPLAY_VALUE_OVERRIDES, _DISPLAY_ONLY, _DISPLAY_TEXT_ONLY
)
# Paragraph-level context math uses the same command vocabulary as display
# math (``\to``, ``\mapsto``, ``\simeq``, set operators, ...). Restricting the
# inline table to _INLINE_ONLY made valid LaTeX fail closed and the rigid
# renderer painted the escaped command bytes. Keep the value overrides, then
# admit the general/display vocabulary; TypstMathProbe remains the final
# compile gate for syntax this lightweight converter cannot represent.
INLINE_SYMBOLS: dict[str, str] = _merged(
    _SHARED, INLINE_VALUE_OVERRIDES, _DISPLAY_ONLY, _INLINE_ONLY
)

#: Every LaTeX command the inline overlay converter recognises. The QE gate
#: (``ubt.core.validators.math_guard.RENDERABLE_LATEX_COMMANDS``) imports this,
#: so the gate and the renderer cannot disagree about what compiles.
INLINE_RENDERABLE_COMMANDS: frozenset[str] = frozenset(INLINE_SYMBOLS) | SPECIAL_LATEX_COMMANDS
