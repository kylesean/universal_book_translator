"""Shared LaTeX-command → Typst-symbol vocabulary.

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
  (Greek letters and the frac/sqrt/det branch it handles itself).
- :data:`DISPLAY_VALUE_OVERRIDES` / :data:`INLINE_VALUE_OVERRIDES` — the seven
  shared commands each side intentionally spells differently (both valid Typst).

A value fix (e.g. ``\\cap`` → ``inter``) therefore lands once and reaches both
tables, so the two maps cannot silently drift apart again.
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
    "ReLU": 'upright("ReLU")',
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
    "hbar": "planck.reduce",
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
    "alpha": "alpha",
    "beta": "beta",
    "chi": "chi",
    "delta": "delta",
    "det": "det",
    "epsilon": "epsilon",
    "eta": "eta",
    "frac": "frac",
    "gamma": "gamma",
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


def _merged(*layers: dict[str, str]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for layer in layers:
        merged.update(layer)
    return merged


DISPLAY_SYMBOLS: dict[str, str] = _merged(_SHARED, DISPLAY_VALUE_OVERRIDES, _DISPLAY_ONLY)
# Paragraph-level context math uses the same command vocabulary as display
# math (``\to``, ``\mapsto``, ``\simeq``, set operators, ...). Restricting the
# inline table to _INLINE_ONLY made valid LaTeX fail closed and the rigid
# renderer painted the escaped command bytes. Keep the value overrides, then
# admit the general/display vocabulary; TypstMathProbe remains the final
# compile gate for syntax this lightweight converter cannot represent.
INLINE_SYMBOLS: dict[str, str] = _merged(
    _SHARED, INLINE_VALUE_OVERRIDES, _DISPLAY_ONLY, _INLINE_ONLY
)
