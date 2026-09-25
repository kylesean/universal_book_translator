"""Gate 1/3 shared math-debris detection + Gate 3 export invariant (CAT QA paradigm).

Industry consensus (BabelDOC ACL'26, XLIFF ``translate="no"`` + placeholder QA,
MinerU formula categories) is one principle: math is a non-translatable atom
with a closed loop. This module is the 0-token enforcement side of that loop:

- :func:`looks_like_math_debris` — text-content classifier for math debris
  (shattered display equations: isolated single-letter tokens + math symbols).
  Structural detection (Docling ``FORMULA`` labels) stays primary; this is the
  safety net for plain-text paths (pdfium/oxide) that never emit ``FORMULA``.
  Calibrated on the KV-Cache handbook ledger: debris pages
  score positive, clean prose/heading pages negative — see unit tests.
- :func:`normalize_math` — whitespace-insensitive comparison for the formula
  invariant (LaTeX spacing carries no semantics).
- :func:`apply_math_guards` — export-stage enforcement (Gate 3):
  (a) ``FORMULA`` blocks whose target drifted from source LaTeX are reset to
  the source (the LLM must never be the source of formula content);
  (b) ``FAILED`` non-formula blocks whose *source* is math debris get their
  target reset to the source (faithful debris over hallucinated garbage).
  Block statuses are NEVER touched here (no pass-rate cosmetics) — every
  action appends an ``error_flags`` entry and is counted in the returned
  tally, which the caller logs and persists like any other checkpoint.
"""

import re
import unicodedata
from typing import Any

from ubt.core.cleaners.math_text import skeleton_holds
from ubt.core.ir.models import BlockStatus, BlockType, IRBlock

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
# Bracketed numeric citations ([1], [12-14], [3, 7]) are prose accessories,
# not math: strip them before analysis so "Research anchors: [1], [6]" stays
# prose. (Same shape as CitationMasker._CITATION_PATTERN, duplicated to keep
# this module dependency-free.)
_CITATION_STRIP_RE = re.compile(r"\[\s*\d+(?:\s*[-–]\s*\d+)?(?:\s*,\s*\d+(?:\s*[-–]\s*\d+)?)*\s*\]")
# Strong math signals. Deliberately excludes comma/period (ordinary prose
# punctuation); isolated-letter density carries those cases instead.
_MATH_STRONG_CHARS = frozenset("=^_{}[]()\\/√∫∑∏∂∞∈∀∃×÷:;…")
_SUBSCRIPT_RE = re.compile(r"[A-Za-z]\s*[_^]")
_ISOLATED_LETTER_RE = re.compile(r"[A-Za-z]")
_PUNCT_STRIP = ".,;:!?()[]{}'\"-–—"
# ASCII words (length 3+) that are ordinary prose, not equation fallout.
# A single real word ("Layer", "Mechanism") vetoes the debris verdict —
# shattered equations never contain running words. Known math-function names
# that legitimately appear inside debris are allowlisted.
_WORD_RE = re.compile(r"[A-Za-z]{3,}")
_MATH_WORDS = frozenset(
    {
        "softmax",
        "frac",
        "sqrt",
        "sum",
        "prod",
        "lim",
        "log",
        "exp",
        "sin",
        "cos",
        "tan",
        "min",
        "max",
        "det",
        "argmax",
        "argmin",
    }
)

# Punctuation-only runs ("…" / ". . .") are layout debris, not prose.
_MIN_TOKENS = 2
_MAX_CHARS = 160
_SINGLE_RATIO = 0.5
_MIN_ISOLATED_LETTERS = 3
# Isolated-letter density floor for the 3c math-signal rung. An absolute
# count (>= 3) misfires on ordinary long prose — a 75-word paragraph
# easily holds six indefinite articles ("a ... a ..."),
# manufacturing a repair round that "reconstructs" plain labels like 2D
# into $2\mathrm{D}$. Shattered-equation rows (e.g. "where Ag 0 , r , F ,
# and F th,SI ...", 3 singles / 16 tokens ≈ 0.19) are letter-DENSE;
# running prose sits well below 0.15. Calibrated: true positive ≈ 0.19,
# false positive ≈ 0.08.
_ISOLATED_LETTER_MIN_DENSITY = 0.15


_ISOLATED_LETTER_RE2 = re.compile(r"\b[A-Za-z]\b")
# A single capital variable glued to a following digit ('N 2', 'F 1') is
# flattened-subscript evidence. Abbreviations ('Fig 3', 'Tab 1', 'Eq 3') have
# more than one letter, so they do not score here and cannot push ordinary
# prose callouts over the >= 2 threshold.
_SUBSCRIPT_FLAT_RE = re.compile(r"\b[A-Z]\s+\d")
_GREEK_NAMES = frozenset(
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
        "omicron",
        "pi",
        "rho",
        "sigma",
        "tau",
        "upsilon",
        "phi",
        "chi",
        "psi",
        "omega",
    }
)


def _is_math_subscript_residue(token: str) -> bool:
    """True when an underscored token looks like flattened math rather than code identifier."""
    if "_" not in token:
        return False
    if "_{" in token:
        return True
    parts = token.split("_")
    if len(parts) < 2:
        return False
    base, sub = parts[0], parts[1]
    if len(base) == 1 and base.isalpha():
        return True
    if base.lower() in _GREEK_NAMES:
        return True
    return bool(len(base) <= 3 and sub.isdigit())


def _has_math_soup_residue(text: str) -> bool:
    for m in re.finditer(r"[A-Za-z0-9\\]+_[A-Za-z0-9{]+", text):
        tok = m.group(0).lstrip("\\")
        if _is_math_subscript_residue(tok):
            return True
    return False


def _strip_code_snake_case(text: str) -> str:
    """Strip standard snake_case programming identifiers (create_task, code_change)."""

    def _repl(m: re.Match[str]) -> str:
        tok = m.group(0)
        return tok if _is_math_subscript_residue(tok) else " "

    return re.sub(r"[A-Za-z0-9\\]+_[A-Za-z0-9{]+", _repl, text)


# Something repair can actually re-delimit: a letter glued to digits (Ag 0,
# F1), a sub/superscript or LaTeX escape, or a Greek letter. Isolated-letter
# debris alone is not a math signal — a drop-cap TOC run ("C OVER T ITLE P AGE")
# scores high on letter density but contains no math to redelimit, so routing
# it to repair asks the model for something that does not exist and the block
# can never clear the gate.
_MATH_CANDIDATE_RE = re.compile(r"[A-Za-z]\s*\d|[\\_^]|[\u0370-\u03ff]")


def target_missing_math_delimiters(source: str, target: str) -> bool:
    """True when undelimited source math would render as flat prose.

    Delimited sources are covered by the span-mismatch gate (3b); this is
    the docling-flattened counterpart (3c): the source carries math
    signals (``Ag 0``, ``F th,SI``, ``psi_pert`` residue, debris) yet the
    target has no ``$...$`` span for the overlay to render in math mode.
    Fires only when the target actually contains ASCII runs worth
    delimiting — pure-CJK targets take the omission path instead.
    Conservative by design: score >= 2 required, so ordinary prose with
    one or two stray single letters never trips it. The source must also
    carry at least one concrete candidate (``_MATH_CANDIDATE_RE``), so
    isolated-letter debris with no math to redelimit stays quiet.
    Programming identifiers (snake_case like create_task, code_change)
    are explicitly exempted from math residue scoring.
    """
    from ubt.core.cleaners.math_masker import extract_math_spans

    if not source or not target:
        return False
    if extract_math_spans(target):
        return False
    if not re.search(r"[A-Za-z]", target):
        return False
    if extract_math_spans(source):
        return True
    src_clean_code = _strip_code_snake_case(source)
    # No concrete math candidate => repair has nothing to re-delimit. Without
    # this, isolated-letter debris (drop-cap TOC/file-header runs) is routed to
    # repair forever and the block is quarantined as MQM-critical.
    if not _MATH_CANDIDATE_RE.search(src_clean_code):
        return False
    score = 0
    if looks_like_math_debris(source):
        score += 2
    if re.search(r"[\\_^]", src_clean_code):
        score += 1
    # Density, not absolute count: three lone "a"s make a math row in a
    # 16-token fragment but are ordinary articles in a 75-word paragraph.
    isolated = _ISOLATED_LETTER_RE2.findall(source)
    tokens = source.split()
    if (
        len(isolated) >= _MIN_ISOLATED_LETTERS
        and tokens
        and len(isolated) / len(tokens) >= _ISOLATED_LETTER_MIN_DENSITY
    ):
        score += 2
    if _SUBSCRIPT_FLAT_RE.search(source):
        score += 1
    if _has_math_soup_residue(target):
        score += 2
    return score >= 2


# LaTeX control sequences the Typst overlay probe compiles, mirrored from
# the render side (mathtext._LATEX_CMD_MAP keys plus the text/mathrm
# upright-string pre-pass). Anything else in a target is unrenderable in
# the in-place overlay (probe rejects -> literal "\mathrm"-style garbage)
# and therefore sanctioned NOWHERE: the draft/repair prompts must only
# emit these. A unit test pins this set in sync with the render table.
RENDERABLE_LATEX_COMMANDS = frozenset(
    {
        "psi",
        "beta",
        "alpha",
        "mu",
        "nu",
        "epsilon",
        "varepsilon",
        "theta",
        "vartheta",
        "lambda",
        "pi",
        "sigma",
        "omega",
        "gamma",
        "delta",
        "phi",
        "varphi",
        "chi",
        "rho",
        "tau",
        "eta",
        "kappa",
        "xi",
        "zeta",
        "Gamma",
        "Delta",
        "Theta",
        "Lambda",
        "Pi",
        "Sigma",
        "Omega",
        "Phi",
        "Psi",
        "times",
        "cdot",
        "div",
        "pm",
        "leq",
        "geq",
        "neq",
        "approx",
        "infty",
        "partial",
        "sqrt",
        "frac",
        "ln",
        "log",
        "exp",
        "sin",
        "cos",
        "tan",
        "min",
        "max",
        "lim",
        "det",
        "sum",
        "prod",
        "int",
        "ll",
        "gg",
        # Shared with the display/overlay converter. TypstMathProbe remains
        # the final syntax gate; membership means "known command", not
        # "blindly paint raw LaTeX".
        "Leftarrow",
        "Leftrightarrow",
        "ReLU",
        "Rightarrow",
        "argmax",
        "argmin",
        "arrow",
        "arrow.l",
        "arrow.r",
        "asymp",
        "because",
        "bigcap",
        "bigcup",
        "cap",
        "cdots",
        "circ",
        "colon",
        "cong",
        "cup",
        "dots",
        "equiv",
        "exists",
        "forall",
        "ge",
        "gggtr",
        "hbar",
        "in",
        "inf",
        "land",
        "ldots",
        "le",
        "leftarrow",
        "llless",
        "lnot",
        "lor",
        "mapsto",
        "mp",
        "nabla",
        "ne",
        "neg",
        "notin",
        "parallel",
        "perp",
        "propto",
        "rightarrow",
        "rightarrowtail",
        "rightsquigarrow",
        "setminus",
        "sim",
        "simeq",
        "subset",
        "subseteq",
        "sup",
        "supset",
        "supseteq",
        "therefore",
        "to",
        "triangleq",
        "varnothing",
        "varpi",
        "varpropto",
        "varrho",
        "varsigma",
        "vdash",
        "dashv",
        "models",
        "emptyset",
        "vdots",
        "text",
        "mathrm",
    }
)

_LATEX_CMD_RE = re.compile(r"\\([a-zA-Z]+)")


#: Unicode math symbol -> its LaTeX control sequence. Docling flattens most
#: source math to plain text, so ``x ∈ S`` reaches the gate as literal U+2208
#: while a translator re-encoding it conventionally emits ``\in``. Comparing
#: control-sequence sets across that encoding boundary reported 244 of 262
#: quarantined blocks as "hallucinated" on a real 92-page paper — led by
#: ``\in`` (62), ``\simeq`` (47), ``\circ`` (34) and ``\to`` (21). None were
#: fabricated; only the encoding moved. Normalizing the source to LaTeX before
#: the comparison fixes the class rather than growing a per-symbol allowlist.
_UNICODE_TO_LATEX = {
    "∈": "in",
    "∉": "notin",
    "⊆": "subseteq",
    "⊂": "subset",
    "∪": "cup",
    "∩": "cap",
    "∅": "emptyset",
    "⌀": "emptyset",
    "⊢": "vdash",
    "⊣": "dashv",
    "⊨": "models",
    "⊧": "models",
    "≃": "simeq",
    "≈": "approx",
    "≅": "cong",
    "≠": "neq",
    "≤": "le",
    "≥": "geq",
    "→": "to",
    "⇒": "Rightarrow",
    "⟹": "Longrightarrow",
    "↦": "mapsto",
    "∘": "circ",
    "⊥": "perp",
    "∥": "parallel",
    "∀": "forall",
    "∃": "exists",
    "∄": "nexists",
    "¬": "neg",
    "∧": "land",
    "∨": "lor",
    "×": "times",
    "⋅": "cdot",
    "⋯": "cdots",
    "…": "dots",
    "⋱": "ddots",
    "⋮": "vdots",
    "Δ": "Delta",
    "Σ": "Sigma",
    "Ω": "Omega",
    "Π": "Pi",
    "Φ": "Phi",
    "Ψ": "Psi",
    "Γ": "Gamma",
    "Θ": "Theta",
    "Λ": "Lambda",
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ε": "epsilon",
    "ζ": "zeta",
    "η": "eta",
    "θ": "theta",
    "ι": "iota",
    "κ": "kappa",
    "λ": "lambda",
    "μ": "mu",
    "ν": "nu",
    "ξ": "xi",
    "π": "pi",
    "ρ": "rho",
    "σ": "sigma",
    "τ": "tau",
    "υ": "upsilon",
    "φ": "phi",
    "χ": "chi",
    "ψ": "psi",
    "ω": "omega",
    "ℝ": "mathbb",
    "ℕ": "mathbb",
    "ℤ": "mathbb",
    "ℚ": "mathbb",
    "ℂ": "mathbb",
    "R_BLACKBOARD": "mathbb",
}
_UNICODE_MATH_CHARS = frozenset(_UNICODE_TO_LATEX)

_UNICODE_MATH_STYLE_PREFIXES = (
    ("MATHEMATICAL BOLD ITALIC", "mathbfit"),
    ("MATHEMATICAL BOLD FRAKTUR", "mathbffrak"),
    ("MATHEMATICAL BOLD SCRIPT", "mathbcal"),
    ("MATHEMATICAL DOUBLE-STRUCK", "mathbb"),
    ("MATHEMATICAL SANS-SERIF", "mathsf"),
    ("MATHEMATICAL MONOSPACE", "mathtt"),
    ("MATHEMATICAL FRAKTUR", "mathfrak"),
    ("MATHEMATICAL SCRIPT", "mathcal"),
    ("MATHEMATICAL BOLD", "mathbf"),
    ("MATHEMATICAL ITALIC", "mathit"),
)


def _unicode_math_style_command(char: str) -> str | None:
    """Return the LaTeX font command represented by a styled math letter."""
    name = unicodedata.name(char, "")
    return next(
        (command for prefix, command in _UNICODE_MATH_STYLE_PREFIXES if name.startswith(prefix)),
        None,
    )


def _source_command_set(source: str) -> set[str]:
    """LaTeX commands in the source, counting Unicode symbols as their names.

    Built as a set union rather than by rewriting the source string: expanding
    ``\forall x`` inline yields ``\\forallx``, and the control-sequence
    pattern ``\\([a-zA-Z]+)`` is greedy, so the command would read as
    ``forallx`` and fail to excuse a target's ``\\forall``. Only the source
    side is expanded -- a bare Unicode symbol in the *target* is prose-level
    math and must not license a control sequence the source never carried.
    """
    src = source or ""
    cmds = set(_LATEX_CMD_RE.findall(src))
    cmds.update(_UNICODE_TO_LATEX[ch] for ch in src if ch in _UNICODE_MATH_CHARS)
    cmds.update(command for ch in src if (command := _unicode_math_style_command(ch)) is not None)
    return cmds


def novel_unsupported_latex_commands(source: str, target: str) -> list[str]:
    """LaTeX commands fabricated by the model.

    Returns the sorted commands present in ``target`` but absent from
    ``source`` AND outside :data:`RENDERABLE_LATEX_COMMANDS` — e.g.
    ``\\mathrm`` wrapped around a plain "(2D)" label. Renderable-novel
    commands (``\\text``, Greek, ``\\frac``...) pass: they compile
    downstream, so flagging them would fight the math-reconstruction
    prompt rule. Empty list means clean.

    The source comparison is Unicode-aware: a source symbol flattened to
    plain text (``∈``) excuses its LaTeX spelling (``\\in``) in the target,
    because the two denote the same math.
    """

    src_cmds = _source_command_set(source)
    novel = [
        cmd
        for cmd in sorted(set(_LATEX_CMD_RE.findall(target or "")))
        if cmd not in src_cmds and cmd not in RENDERABLE_LATEX_COMMANDS
    ]
    return novel


def normalize_math(text: str) -> str:
    """Collapse all whitespace for LaTeX comparison (spacing is not semantic)."""
    return re.sub(r"\s+", "", text or "")


def looks_like_math_debris(
    text: str,
    *,
    min_tokens: int = _MIN_TOKENS,
    max_chars: int = _MAX_CHARS,
    single_ratio: float = _SINGLE_RATIO,
) -> bool:
    """True when text reads as shattered-equation debris rather than prose.

    Positive signals: high isolated-single-letter density (``K``, ``V``,
    ``x`` — equation fallout, excluding nothing: prose almost never does
    this) combined with math symbols, subscript shapes, or 3+ isolated
    letters. CJK text, long passages, and single tokens are never debris
    (conservative: a missed debris block still goes through normal QE).
    """
    t = (text or "").strip()
    if not t or len(t) > max_chars:
        return False
    if _CJK_RE.search(t):
        return False
    t = _CITATION_STRIP_RE.sub(" ", t)
    # Running prose words veto debris: "(1) Layer 1: K, V" is a caption to
    # translate, while ", k q , v t t t ," has no words at all.
    if any(w.lower() not in _MATH_WORDS for w in _WORD_RE.findall(t)):
        return False
    tokens = t.split()
    if len(tokens) < min_tokens:
        return False
    if all(not tok.strip(_PUNCT_STRIP) for tok in tokens):
        return True
    singles = sum(1 for tok in tokens if len(tok.strip(_PUNCT_STRIP)) == 1)
    if singles / len(tokens) < single_ratio:
        return False
    if any(ch in _MATH_STRONG_CHARS for ch in t):
        return True
    if _SUBSCRIPT_RE.search(t):
        return True
    return len(_ISOLATED_LETTER_RE.findall(t)) >= _MIN_ISOLATED_LETTERS


def formula_target_intact(block: IRBlock) -> bool:
    """Formula invariant: target must be byte-identical (mod whitespace) source."""
    if block.block_type is not BlockType.FORMULA:
        return True
    if not block.target_text:
        return True
    return normalize_math(block.target_text) == normalize_math(block.source_text)


def formula_skeleton_intact(block: IRBlock) -> bool:
    """Skeleton invariant: skeleton-identical targets are legal translations.

    Byte-identity (above) stays the default bar; a target that differs
    ONLY inside ``\\text{…}``-family span bodies — i.e. a natural-language
    span translation — passes the skeleton check and must NOT be repaired.
    Anything else (changed math, deleted spans) still fails.
    """
    if block.block_type is not BlockType.FORMULA:
        return True
    if not block.target_text:
        return True
    return skeleton_holds(block.source_text or "", block.target_text)


def apply_math_guards(blocks: list[IRBlock]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Enforce Gate 3 on export-bound blocks.

    Returns ``(checkpoints, counts)`` where checkpoints follow the
    ``save_checkpoints_batch`` dict shape and counts holds
    ``formula_invariant_repairs`` / ``math_debris_fallbacks`` /
    ``c_text_accepted``. Mutates the in-memory blocks (target + error_flags)
    like the surrounding export loop. Flag appends are idempotent: export
    re-runs on resume/re-export, and a second pass must not duplicate flags
    or inflate defect counts.
    """
    checkpoints: list[dict[str, Any]] = []
    counts = {"formula_invariant_repairs": 0, "math_debris_fallbacks": 0, "c_text_accepted": 0}

    def _note(block: IRBlock, flag: str) -> bool:
        """Append ``flag`` once; True when newly added (countable)."""
        if flag in block.error_flags:
            return False
        block.error_flags.append(flag)
        return True

    for block in blocks:
        if block.block_type is BlockType.FORMULA and block.target_text:
            if formula_target_intact(block):
                continue
            if formula_skeleton_intact(block):
                # Legal span translation: prose spans changed, math untouched.
                if _note(
                    block,
                    "c_text_span_translation: formula target differs from "
                    "source only inside \\text{...} span bodies (accepted)",
                ):
                    counts["c_text_accepted"] += 1
                    checkpoints.append(
                        {
                            "block_id": block.id,
                            "target_text": block.target_text,
                            "status": block.status,
                            "error_flags": block.error_flags,
                        }
                    )
                continue
            prev_target = block.target_text
            noted = _note(
                block,
                "math_invariant_repair: formula target drifted from source "
                "LaTeX; reset to source (LLM output is never formula source)",
            )
            block.target_text = block.source_text
            if noted or block.target_text != prev_target:
                counts["formula_invariant_repairs"] += 1
                checkpoints.append(
                    {
                        "block_id": block.id,
                        "target_text": block.target_text,
                        "status": block.status,
                        "error_flags": block.error_flags,
                    }
                )
        elif (
            block.block_type is not BlockType.FORMULA
            and block.status is BlockStatus.FAILED
            and block.target_text
            and looks_like_math_debris(block.source_text)
        ):
            prev_target = block.target_text
            noted = _note(
                block,
                "math_debris_source_fallback: failed math-debris block renders "
                "source instead of hallucinated draft (re-run with docling "
                "for real formulas)",
            )
            block.target_text = block.source_text
            if noted or block.target_text != prev_target:
                counts["math_debris_fallbacks"] += 1
                checkpoints.append(
                    {
                        "block_id": block.id,
                        "target_text": block.target_text,
                        "status": block.status,
                        "error_flags": block.error_flags,
                    }
                )
    return checkpoints, counts
