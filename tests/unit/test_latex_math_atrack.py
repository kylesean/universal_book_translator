"""Unit tests for A-track: pandoc AST-based LaTeX→Typst with regex fallback.

Evidence basis: KV handbook formula
``K_{1:t-1} = [k_1; \\dots]`` — the regex path dropped half the equation
and detached subscripts (``K _(...)``); pandoc renders it complete with
attached subscripts (``K_(...)``).
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from ubt.adapters.pdf import typst_math
from ubt.adapters.pdf.typst_math import (
    _clean_ocr_formula,
    _normalize_docling_math,
    _sanitize_typst_math_variables,
)
from ubt.adapters.pdf.typst_reconstructor import (
    _balanced_delimiters,
    _emit_formula_math,
    _latex_math_to_typst,
    _latex_math_to_typst_regex,
    _pandoc_math_to_typst,
)

KV_F1 = "K _ { 1 \\colon t - 1 } = [ k _ { 1 } ; \\dots ; k _ { t - 1 } ] , \\quad V _ { 1 \\colon t - 1 } = [ v _ { 1 } ; \\dots ; v _ { t - 1 } ] ."
KV_SOFTMAX = "\\text {softmax} \\left ( \\frac { q _ { t } K _ { 1 \\colon t } ^ { T } } { \\sqrt { d _ { h } } } \\right ) V _ { 1 \\colon t } ."


def test_pandoc_primary_kv_formula_complete() -> None:
    out = _latex_math_to_typst(KV_F1)
    # Complete: both K and V halves survive (regex path lost the V half).
    assert "V" in out and "K" in out
    # Attached subscripts: no detached " _(...)".
    assert " _" not in out
    assert _balanced_delimiters(out)


def test_pandoc_primary_softmax_frac() -> None:
    out = _latex_math_to_typst(KV_SOFTMAX)
    assert "frac(" in out and "sqrt(" in out
    assert _balanced_delimiters(out)


def test_pandoc_returns_none_on_empty() -> None:
    assert _pandoc_math_to_typst("") is None
    assert _pandoc_math_to_typst("   ") is None


def test_fallback_when_pandoc_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pandoc missing → output identical to the legacy regex converter."""
    monkeypatch.setattr(typst_math, "_pandoc_math_to_typst", lambda _s: None)
    assert _latex_math_to_typst(KV_F1) == _latex_math_to_typst_regex(KV_F1)
    assert _latex_math_to_typst(KV_SOFTMAX) == _latex_math_to_typst_regex(KV_SOFTMAX)


def test_fallback_when_pandoc_binary_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _b: None)
    assert _pandoc_math_to_typst("x^2") is None


def test_emit_formula_math_uses_pandoc_output() -> None:
    line = _emit_formula_math(KV_F1, "b1")
    assert line.startswith("$") and line.endswith("$")
    assert "K_(" in line  # attached subscript = pandoc path signature


def test_emit_formula_math_keeps_boxed_equation() -> None:
    r"""``\boxed`` converts to a cleartext ``#box(...)[$ … $]`` call.

    Gate 4 rejects any ``$`` inside the math expression, but the inner ``$`` is
    the content-block's own math mode — valid, compiling Typst. Rejecting it
    degraded every boxed equation to a verbatim code span.
    """
    line = _emit_formula_math(r"\boxed{x+1}", "b_box")
    assert line.startswith("$") and line.endswith("$"), f"boxed eq degraded: {line!r}"
    assert "#box" in line
    assert not line.startswith("`")


def test_large_operators_are_not_quoted_as_literal_text() -> None:
    """\\sum/\\prod/\\cup/\\cap/\\infty must render as symbols, not the words.

    Regression: _sanitize_typst_math_variables quoted any bare multi-letter
    token absent from _TYPST_MATH_KEYWORDS, and pandoc emits ``sum``/``product``/
    ``union``/``inter``/``oo`` for these commands. The page therefore showed the
    literal word "sum" instead of ∑ (and "oo" instead of ∞).
    """
    cases = {
        r"\sum_{i=1}^n x_i": "sum_",
        r"\prod_i A_i": "product_",
        r"A \cup B": "union",
        r"A \cap B": "inter",
        r"\infty": "oo",
        r"\therefore": "therefore",
    }
    for latex, expected in cases.items():
        out = _latex_math_to_typst(latex)
        assert expected in out, out
        for word in ("sum", "product", "union", "inter", "oo", "therefore"):
            assert f'"{word}"' not in out, f"{latex} rendered {word!r} as literal text: {out}"


def test_ll_and_gg_symbol_conversion() -> None:
    out = _latex_math_to_typst(r"a \ll b \gg c")
    assert "lt.double" in out
    assert "gt.double" in out


def test_glued_command_and_subscript_sanitization() -> None:
    # Test the exact formula pattern from user report: (F \llF_th,SI
    out = _latex_math_to_typst(r"(F \llF_th,SI")
    assert "lt.double" in out
    # Multi-letter unknown variable must be quoted to avoid Typst compilation error
    assert '"SI"' in out or '"th"' in out


def test_clean_ocr_formula_keeps_content_parenthesis_with_operators() -> None:
    r"""Regression: a trailing ``(1 - x)`` is content, not an equation number.

    ``_EQ_NUM_TAIL_RE`` matched any ``(digit …)`` after a space, so
    ``f(x) = (1 - x)`` was silently truncated to ``f(x) =`` (a real, silent
    formula-content loss).
    """
    from ubt.adapters.pdf.typst_math import _clean_ocr_formula

    cleaned = _clean_ocr_formula(r"f(x) = (1 - x)")
    assert "(1 - x)" in cleaned

    # A real equation number after an explicit separator is still stripped.
    numbered = _clean_ocr_formula(r"E = m c^2 \qquad (3.2) \\")
    assert "3.2" not in numbered


def test_clean_ocr_formula_upright_and_embedded_eq_num() -> None:
    from ubt.adapters.pdf.typst_math import _clean_ocr_formula

    # Test stripping embedded equation number (11)
    raw = r"= \ln(\beta) - \frac{V_{gs}}{2 V_{tm}} [\psi_{pert} (11) 2 V_{um}]"
    cleaned = _clean_ocr_formula(raw)
    assert "(11)" not in cleaned
    assert "2 V_{um}" in cleaned

    # Test cleaning Docling OCR 'upright' artifacts
    raw_upright = r"V_{gs} = V_{\text{upright fb}} + \frac{\mathcal{E}_{xs}}{C_{upright ox}}"
    cleaned_upright = _clean_ocr_formula(raw_upright)
    assert "upright" not in cleaned_upright
    assert r"V_{\text{fb}}" in cleaned_upright or "fb" in cleaned_upright
    assert r"C_{ox}" in cleaned_upright or "ox" in cleaned_upright


# ---------------------------------------------------------------------------
# Regression: CodeFormulaV2 region over-capture glued the following "where
# Q_0 = ..." prose line onto Eq. (3.15) as a third row, misread as
# ``Q_{no} = Q_{,y} + 5C_{;Y} with C_{;} = eps, sqrt(T_{;})``.
# ---------------------------------------------------------------------------
REGION_OVERCAPTURE_EQ = (
    r"a \text { function on } Q _ { \text {inv} } \text { using } \\ "
    r"Q _ { \text {inv} } ( y ) \approx \sqrt { 2 q n _ { i } } \\ "
    r"Q _ { \text {no} } = Q _ { \text {, } y } + 5 C _ { \text {; } Y } "
    r"\text { with } C _ { \text {; } } = \varepsilon \, , \sqrt { T _ { \text {; } } } "
    r"\text { \ using this approximation } F ( 3 )"
)


def test_clean_ocr_formula_drops_region_overcapture_row() -> None:
    from ubt.adapters.pdf.typst_math import _clean_ocr_formula

    cleaned = _clean_ocr_formula(REGION_OVERCAPTURE_EQ)
    assert r"Q _ { \text {no} }" not in cleaned
    assert r"\sqrt { T _ { \text {; } } }" not in cleaned
    assert r"Q _ { \text {inv} } ( y )" in cleaned


def test_clean_ocr_formula_keeps_clean_trailing_row() -> None:
    from ubt.adapters.pdf.typst_math import _clean_ocr_formula

    cleaned = _clean_ocr_formula(r"a = b \\ c = d \text { with } e = f")
    assert "c = d" in cleaned


def test_emit_formula_math_splits_long_equation() -> None:
    chain = " + ".join(f"psi_{i}(y)" for i in range(12))
    long_eq = rf"V_{{gs}} = V_{{fb}} + {chain} = V_{{fb}} + {chain}"
    res = _emit_formula_math(long_eq, "b_long")
    assert r"&=" in res
    assert "\\\n" in res


# ---------------------------------------------------------------------------
# Regression: silent truncation of unbalanced Docling LaTeX (chapter-3 b0040)
# ---------------------------------------------------------------------------
# The real payload from ledger ``job_fc1d7bd7b799`` block ``pdf_main#b0040``
# (Eq. 3.9). Docling emitted benignly unbalanced LaTeX — 38 ``{`` vs 36 ``}``,
# and 2 ``\left`` vs 3 ``\right``. The old ``_read_group`` reported "consumed
# everything" on failure, so the ``\sqrt`` branch advanced its cursor to the
# end of the string and the whole equation collapsed to
# ``cal(E)_("x s")= sqrt`` — 483 chars to 20, silently.
DOCLING_UNBALANCED_EQ = (
    r"\mathcal { E } _ { x s } = \sqrt { \frac { 2 q n _ { i } } { \varepsilon _ { \text {ch} } }"
    r" \left [ V _ { \text {tm} } \left ( e ^ { \frac { \gamma _ { s } ( y ) }"
    r" { \gamma _ { \text {tm} } } - e ^ { \frac { \gamma _ { s } ( y ) } { \gamma _ { \text {tm} } } }"
    r" \right ) e ^ { \frac { - \gamma _ { B } - V _ { \text {s} } ( y ) } { \gamma _ { \text {tm} } }"
    r" + e ^ { \frac { \gamma _ { B } } { \gamma _ { \text {tm} } } } ( \psi _ { s } ( y )"
    r" - \psi _ { 0 } ( y ) ) \right ) \right ] }"
)
# What the old regex fallback produced for the payload above.
TRUNCATED_FRAGMENT = 'cal(E)_("x s")= sqrt'


def test_docling_payload_really_is_unbalanced() -> None:
    """Guard the fixture: the regression only means something while it holds."""
    assert DOCLING_UNBALANCED_EQ.count("{") > DOCLING_UNBALANCED_EQ.count("}")


def test_read_group_failure_does_not_claim_the_remainder() -> None:
    """A failed group read must report "nothing consumed", never ``len(s)``.

    Callers treat the returned index as the next parse position; returning
    ``len(s)`` made them discard every character after the failed group.
    """
    from ubt.adapters.pdf.typst_math import _read_group

    assert _read_group("{ a { b }", 0) == (None, 0)
    content, nxt = _read_group("{ a }", 0)
    assert content == " a " and nxt == 5


def test_unbalanced_docling_formula_is_not_silently_truncated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regex fallback must keep the content, not collapse to a bare ``sqrt``."""
    from ubt.adapters.pdf.typst_math import _math_atom_count

    monkeypatch.setattr(typst_math, "_pandoc_math_to_typst", lambda _s: None)
    out = _latex_math_to_typst_regex(DOCLING_UNBALANCED_EQ)
    assert out != TRUNCATED_FRAGMENT
    assert not out.rstrip().endswith("sqrt")
    # Content, not formatting, is the invariant: the bulk of the math survives.
    assert _math_atom_count(out) >= 0.5 * _math_atom_count(DOCLING_UNBALANCED_EQ)


def test_conversion_never_loses_math_content() -> None:
    """Gate 5 invariant: every tier either preserves content or yields the source.

    Deterministic despite pandoc: ``_accept_conversion`` returns the source
    verbatim when a tier would have dropped content, so the guarantee holds
    whether or not pandoc is installed.
    """
    from ubt.adapters.pdf.typst_math import _content_lost

    for payload in (DOCLING_UNBALANCED_EQ, KV_F1, KV_SOFTMAX):
        assert not _content_lost(payload, _latex_math_to_typst(payload))


def test_content_gate_recognises_the_truncated_fragment() -> None:
    from ubt.adapters.pdf.typst_math import _content_lost

    assert _content_lost(DOCLING_UNBALANCED_EQ, TRUNCATED_FRAGMENT)
    # Short formulas carry no reliable ratio signal and must not be flagged.
    assert not _content_lost(r"x ^ { 2 }", "x ^ 2")


def test_emit_formula_math_fails_loud_instead_of_shipping_a_fragment() -> None:
    """End-to-end: the artifact must keep the content and never look healthy.

    Before the fix this emitted ``$ cal(E)_("x s")= sqrt $`` — balanced, no
    residual LaTeX, syntactically valid, semantically empty, and therefore
    invisible to Gate 4 while being wrong on the page.
    """
    from ubt.adapters.pdf.typst_math import _math_atom_count

    line = _emit_formula_math(DOCLING_UNBALANCED_EQ, "pdf_main#b0040")
    assert not line.strip().startswith("$")
    assert line.startswith("`"), "must degrade to the verbatim fail-loud form"
    assert _math_atom_count(line) >= 0.5 * _math_atom_count(DOCLING_UNBALANCED_EQ)


# ---------------------------------------------------------------------------
# Regression: the long-equation splitter emitted LaTeX ``\quad`` into Typst
# ---------------------------------------------------------------------------
# Typst math has no backslash-command syntax, so ``&\quad`` parsed as an
# alignment point followed by two undefined identifiers: ``unknown variable:
# uad``. Gate 4 counts delimiters and passed it, and the splitter used to run
# *after* Gate 4, so nothing validated its output — 9 of chapter-3's 74
# formula blocks were degraded at the render-time compile probe while the
# identical unsplit input compiled cleanly.
SPLITTABLE_EQ = "f_0 = " + " + ".join(f"alpha_{i} beta_{i}" for i in range(14))


def test_splitter_emits_typst_syntax_not_latex() -> None:
    from ubt.adapters.pdf.typst_math import _has_residual_latex, _split_long_typst_equation

    split = _split_long_typst_equation(SPLITTABLE_EQ)
    assert "\\\n" in split, "fixture must actually trigger a split"
    # A backslash-newline is Typst's legitimate math line break; every other
    # backslash would be a LaTeX command Typst cannot parse.
    residue = split.replace("\\\n", "")
    assert "\\" not in residue, f"LaTeX command leaked into Typst math: {split!r}"
    assert not _has_residual_latex(split)


def test_emit_formula_math_validates_the_split_result() -> None:
    """The split output must go through Gate 4, not skip it."""
    from ubt.adapters.pdf.typst_math import _has_residual_latex

    line = _emit_formula_math(SPLITTABLE_EQ, "b_split")
    assert line.startswith("$") and line.endswith("$")
    assert "\\\n" in line, "fixture must exercise the split path"
    assert not _has_residual_latex(line)


# ---------------------------------------------------------------------------
# Regression: Docling artifacts seen in chapter-3 Eq. (3.8) / b0038
# ---------------------------------------------------------------------------
# Docling writes multi-letter roman labels without braces: ``\mathrm f b``.
# The converter's function branch needs a group, so it emitted the *bare* word
# ``upright`` and Typst set the letters u·p·r·i·g·h·t·f·b — the page showed
# ``V_upright fb`` where the source reads ``V_fb``.
BARE_ROMAN_EQ = r"V _ { g s } & = V _ { \mathrm f b } + \psi _ { s } ( y ) + C _ { \mathrm o x }"

# Docling also stuffs a duplicate of the equation into ``\intertext{}``. The
# old splitter regex ``\intertext\s*\{[^}]*\}`` could not cross the payload's
# own braces, so it removed only up to the first inner ``}`` and the remainder
# leaked back — Eq. (3.8) rendered twice, once per aligned line.
INTERTEXT_DUP_EQ = (
    r"V _ { g s } & = V _ { \mathrm f b } + \psi _ { s } ( y ) \intertext { v - "
    r"V _ { g s } = V _ { \mathrm f b } + \psi _ { s } ( y ) }"
)


def test_bare_roman_label_is_rejoined_into_a_group() -> None:
    """``\\mathrm f b`` must become a real group, not a bare ``upright`` word."""
    from ubt.adapters.pdf.typst_math import _BARE_ROMAN_RE, _clean_ocr_formula

    cleaned = _clean_ocr_formula(BARE_ROMAN_EQ)
    assert r"\mathrm{fb}" in cleaned
    assert r"\mathrm{ox}" in cleaned
    assert not _BARE_ROMAN_RE.search(cleaned)
    # Single-token labels are normalised the same way (idempotent, uniform).
    assert r"\mathrm{m}" in _clean_ocr_formula(r"V _ { \mathrm m }")


def test_emit_formula_math_leaks_no_bare_upright_word() -> None:
    """End-to-end: no bare ``upright`` identifier may reach Typst math.

    The label letters must stay bound together — as ``upright("fb")`` on the
    pandoc tier, or as a quoted subscript ``_("fb")`` after the regex tier's
    subscript sanitiser. Both render upright; the failure this guards is the
    bare word ``upright`` followed by loose letters.
    """
    import re as _re

    cases = {
        BARE_ROMAN_EQ: r"f\s?b",
        r"T _ { b } = e ^ { \psi _ { \mathrm r s } / V _ { t m } }": r"r\s?s",
    }
    for payload, label in cases.items():
        line = _emit_formula_math(payload, "b_roman")
        assert not _re.search(r"upright(?![(\"'])", line), line
        assert _re.search(label, line), line


def test_intertext_payload_with_nested_braces_is_not_leaked() -> None:
    """The nested-brace payload must be removed whole, not half-swallowed."""
    from ubt.adapters.pdf.typst_math import _clean_ocr_formula, _split_on_intertext

    parts = _split_on_intertext(INTERTEXT_DUP_EQ)
    assert len(parts) == 2
    assert "intertext" not in parts[0]
    assert parts[1].strip() == "", "trailing remainder must be empty, not the payload"

    cleaned = _clean_ocr_formula(INTERTEXT_DUP_EQ)
    assert "intertext" not in cleaned
    assert cleaned.count("=") == 1, f"duplicate equation branch survived: {cleaned!r}"


def test_eq_3_8_renders_once_not_twice() -> None:
    """The exact chapter-3 b0038 payload must not duplicate itself."""
    line = _emit_formula_math(INTERTEXT_DUP_EQ, "pdf_main#b0038")
    assert line.startswith("$") and line.endswith("$")
    assert "\\\n" not in line, "must not need an aligned second line"
    # One '=' only: a second copy would add another one.
    assert line.count("=") == 1, line
    assert 'V _("fb")' in line


# ---------------------------------------------------------------------------
# Author line breaks: `\\` must survive conversion as a Typst continuation.
# ---------------------------------------------------------------------------
# chapter-3 Eq. (3.11) is long enough that the author broke it before the
# `+ 2 eps / (T_fin C_ox)` term. `_TEX_SPACING` used to map `\\` to a plain
# space, so the break was lost and Typst re-wrapped the equation itself —
# landing the wrap at the final relation and stranding `= 0` on its own line
# while the equation number slid to the vertical middle of the resulting
# two-row block. 37 of the 74 formula blocks carry such a break.
AUTHOR_BREAK_EQ = (
    r"f ( \beta ) = \ln ( \beta ) - \ln ( \cos ( \beta ) ) "
    r"\\ + \frac { 2 \epsilon } { T _ { \text {fin} } } \sqrt { \beta ^ { 2 } } = 0"
)


def test_author_line_break_survives_conversion() -> None:
    out = _emit_formula_math(AUTHOR_BREAK_EQ, "b_break")
    assert out.lstrip().startswith("$"), out
    # A Typst continuation line, not a flattened space.
    assert "\\\n" in out, "the author's explicit break was flattened"
    assert "  quad + frac" in out, out


def test_author_line_break_has_no_alignment_point() -> None:
    """A lone ``&`` on the wrapped row turns the rows into alignment columns.

    Typst sizes such a block as the sum of both rows, which pushed chapter-3
    Eq. 3.11 47pt past both page margins. The indent must stay a bare
    ``quad``; ``&\\quad`` also dies with ``unknown variable: uad``.
    """
    out = _emit_formula_math(AUTHOR_BREAK_EQ, "b_break")
    assert r"&\quad" not in out
    assert "&quad" not in out
    assert "quad" in out


def test_line_break_sentinel_never_leaks() -> None:
    """The placeholder is stripped before the caller ever sees the string."""
    out = _emit_formula_math(AUTHOR_BREAK_EQ, "b_break")
    assert "\x00" not in out


def test_converted_author_break_still_compiles() -> None:
    """Restoring the break must not hand the renderer an invalid equation."""
    from ubt.adapters.pdf.typst_math import is_typst_math_well_formed

    body = _emit_formula_math(AUTHOR_BREAK_EQ, "b_break").strip("$ ")
    assert is_typst_math_well_formed(body)


# ---------------------------------------------------------------------------
# Splitter: a terminal `= 0` is not a chain link.
# ---------------------------------------------------------------------------
def test_splitter_does_not_orphan_a_terminal_relation() -> None:
    """`A = <long RHS> = 0` used to break before the final `= 0`."""
    from ubt.adapters.pdf.typst_math import _split_long_typst_equation

    eq = "f ( beta ) = " + " + ".join(f"a{i} b{i}" for i in range(12)) + " = 0"
    split = _split_long_typst_equation(eq)
    lines = split.split("\n")
    if len(lines) > 1:
        last = lines[-1].strip()
        assert last != "&= 0", "a lone `= 0` was stranded on its own line"
    assert "= 0" in split


def test_splitter_still_aligns_genuine_chains() -> None:
    """A real `A = B = C` chain keeps the aligned layout."""
    from ubt.adapters.pdf.typst_math import _split_long_typst_equation

    eq = "a very long left hand side here = " + "b " * 40 + "= " + "c " * 40
    split = _split_long_typst_equation(eq)
    assert "&=" in split


# ---------------------------------------------------------------------------
# Regression: chapter-3 Eq. 3.13 printed the word "quad" after the equation,
# Eq. 3.14 (single-line original) was shredded into two lines
# ---------------------------------------------------------------------------
# Docling ends display equations with ``\quad \\``. The quad survived cleaning
# (the trailing strip class had no quad) and the dangling break survived the
# restore, so Typst set a literal "quad" word after Eq. 3.13. Separately, the
# splitter judged fit on raw markup length (~1.6x the glyph count), so the
# 44-glyph single-line Eq. 3.14 was split in two.


def test_trailing_quad_break_never_ships() -> None:
    """Docling ``\\quad \\\\`` tails must vanish at every tier."""
    from ubt.adapters.pdf.typst_math import (
        _clean_ocr_formula,
        _emit_formula_math,
        _latex_math_to_typst,
        _restore_line_breaks,
    )

    src = r"I _ { d s } = \frac { W } { L } \mu ( T ) d Q _ { \text {inv} } \quad \\"
    assert "quad" not in _clean_ocr_formula(src)
    converted = _latex_math_to_typst(src)
    assert "quad" not in converted
    # Even a hand-built dangling break restores to nothing ...
    assert _restore_line_breaks("a + b\x00") == "a + b"
    # ... and the emission net catches whatever is left.
    line = _emit_formula_math(src, "b_quad_tail")
    assert line.startswith("$") and line.endswith("$")
    assert "quad" not in line


def test_display_len_ignores_markup_and_math_whitespace() -> None:
    """``upright(..)`` wrappers and math-mode spaces occupy no width."""
    from ubt.adapters.pdf.typst_math import _math_display_len

    assert _math_display_len("a b") == _math_display_len("ab") == 2
    assert _math_display_len('Q_(upright("inv"))') <= len("Q_(inv)")
    assert _math_display_len("f ( beta )") < len("f ( beta )")


def test_single_line_original_is_not_split() -> None:
    """A 44-glyph single-line equation (chapter-3 Eq. 3.14 class) stays whole."""
    from ubt.adapters.pdf.typst_math import _emit_formula_math

    src = (
        r"Q _ { \text {inv} , d / s } = C _ { \text {ox} } "
        r"( V _ { g s } - V _ { \text {fb} } - \psi _ { d / s } ) "
        r"- Q _ { \text {bulk} }"
    )
    line = _emit_formula_math(src, "b_single")
    assert line.startswith("$") and line.endswith("$")
    assert "\\\n" not in line
    assert "quad" not in line


def test_pandoc_math_identifiers_are_not_quoted_into_words() -> None:
    """The identifier sanitizer must not run on the pandoc tier.

    Pandoc's Typst writer emits ``hat(theta)``, ``overline(z)``, ``divides``,
    ``binom(n, k)`` — all valid Typst. The hand-written allowlist did not know
    them, so it quoted them into strings that compile with exit 0 and render as
    English words: ``hat(θ)`` printed instead of θ̂, ``P(A divides B)`` instead
    of P(A∣B). Nothing downstream can catch it: the atom-count gate sees the
    same content and typst reports no error.
    """
    quoted = ("hat", "tilde", "overline", "underline", "divides", "binom")
    for latex in (
        r"\hat{\theta}_{MLE}",
        r"\tilde{A}x=\tilde{b}",
        r"\overline{z} + \underline{y}",
        r"P(A \mid B)",
        r"\binom{n}{k}",
    ):
        converted = _latex_math_to_typst(latex)
        assert _pandoc_math_to_typst(latex) is not None, "pandoc must be the tier under test"
        for name in quoted:
            assert f'"{name}"' not in converted, f"{latex} -> {converted}"


@pytest.mark.skipif(shutil.which("typst") is None, reason="typst is not installed")
def test_accent_formulas_compile_as_math(tmp_path: Path) -> None:
    """The delivered form must be the one Typst accepts as math."""
    out = tmp_path / "accent.typ"
    for latex in (r"\hat{\theta}_{MLE}", r"\overline{z} + \underline{y}", r"P(A \mid B)"):
        line = _emit_formula_math(latex, "pdf_main#b1")
        out.write_text("#set page(width: 250pt, height: 100pt)\n" + line + "\n", encoding="utf-8")
        proc = subprocess.run(
            ["typst", "compile", "--root", str(tmp_path), str(out), str(tmp_path / "o.pdf")],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, f"{latex} -> {line}\n{proc.stderr}"


def test_subscript_words_are_not_reinterpreted_as_stripped_commands() -> None:
    r"""``R_{in}`` is input resistance; the OCR repair must not make it "R ∈".

    The Docling restore added a backslash to any bare command word anywhere,
    including inside ``_{...}`` and ``\text{...}``: ``R_{in}`` became
    ``R_{\in}`` and pandoc faithfully emitted ``R_in`` — which Typst renders as
    the membership symbol. Inside a group the letters are the author's text, and
    a stripped backslash cannot be told apart from a subscript word, so the
    guess is not ours to make.
    """
    assert _normalize_docling_math(r"R_{in}") == r"R_{in}"
    assert _normalize_docling_math(r"\text{ in general}") == r"\text{ in general}"
    # Outside a group the repair still does its job.
    assert _normalize_docling_math("alpha beta") == r"\alpha \beta"
    assert "_(i n)" in _latex_math_to_typst(r"R_{in} = \frac{V}{I}")


def test_prose_glue_cutter_does_not_slice_inside_a_word() -> None:
    """``\\text{elsewhere}`` is a cases row's content, not a sentence tail.

    The glue search had no left boundary, so "where" matched inside
    "elsewhere"/"anywhere"/"everywhere" and the cutter dropped everything from
    there — losing a formula branch and leaving an unbalanced brace group.
    """
    cases = r"f(x)=\begin{cases}1 & x>0\\0 & \text{elsewhere}\end{cases}"
    assert _clean_ocr_formula(cases) == cases
    assert _clean_ocr_formula(r"F = ma \text{anywhere}") == r"F = ma \text{anywhere}"
    # The intended cuts still happen: a glued narrative row and a "where" tail.
    assert _clean_ocr_formula(r"E = mc^2 \text{ where } m \text{ is mass}") == "E = mc^2"
    assert _clean_ocr_formula(r"a \text{function on f} \\ Q_{inv} = 2") == r"Q_{inv} = 2"


def test_regex_tier_does_not_glue_adjacent_commands() -> None:
    """``4\\pi\\varepsilon_0`` must not become the identifier ``4piepsilon``.

    The fallback converter appended each symbol straight onto the previous one,
    and the identifier sanitizer refuses to touch a token that starts after a
    digit — so the equation failed to compile for the *whole document* until the
    healer quoted it into a visibly wrong formula.
    """
    converted = _latex_math_to_typst_regex(r"V(x) = \frac{e}{4\pi\varepsilon_0 r}")
    assert "piepsilon" not in converted
    assert "pi" in converted and "epsilon.alt" in converted


@pytest.mark.skipif(shutil.which("typst") is None, reason="typst is not installed")
def test_regex_tier_output_compiles(tmp_path: Path) -> None:
    out = tmp_path / "regex.typ"
    body = _sanitize_typst_math_variables(
        _latex_math_to_typst_regex(r"V(x) = \frac{e}{4\pi\varepsilon_0 r}")
    )
    out.write_text(f"#set page(width: 250pt, height: 120pt)\n$ {body} $\n", encoding="utf-8")
    proc = subprocess.run(
        ["typst", "compile", "--root", str(tmp_path), str(out), str(tmp_path / "o.pdf")],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"{body}\n{proc.stderr}"


def test_latex_to_typst_matrix_and_cases_in_regex_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Matrix, cases, and aligned environments convert correctly even in regex fallback mode."""
    monkeypatch.setattr(typst_math, "_pandoc_math_to_typst", lambda _s: None)

    # 1. pmatrix: \begin{pmatrix} a & b \\ c & d \end{pmatrix} -> mat(...)
    pmatrix = r"\begin{pmatrix} a & b \\ c & d \end{pmatrix}"
    out_p = _latex_math_to_typst(pmatrix)
    assert "mat(" in out_p
    assert '"begin"' not in out_p
    assert '"end"' not in out_p
    assert ";" in out_p or "\\" in out_p

    # 2. bmatrix: \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} -> mat(delim: "[", ...)
    bmatrix = r"\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}"
    out_b = _latex_math_to_typst(bmatrix)
    assert "mat(" in out_b
    assert 'delim: "["' in out_b or "delim: [" in out_b
    assert '"begin"' not in out_b

    # 3. cases: \begin{cases} x & x > 0 \\ -x & x \le 0 \end{cases} -> cases(...)
    cases = r"\begin{cases} x & x > 0 \\ -x & x \le 0 \end{cases}"
    out_c = _latex_math_to_typst(cases)
    assert "cases(" in out_c
    assert '"begin"' not in out_c

    # 4. aligned: \begin{aligned} y &= ax + b \\ z &= cx + d \end{aligned}
    aligned = r"\begin{aligned} y &= ax + b \\ z &= cx + d \end{aligned}"
    out_a = _latex_math_to_typst(aligned)
    assert '"begin"' not in out_a
    assert '"end"' not in out_a


def test_normalize_docling_math_nested_braces() -> None:
    """Nested braces in text/groups must be masked recursively so 'in' is not turned into '\\in'."""
    from ubt.adapters.pdf.typst_math import _normalize_docling_math

    input_math = r"\text{plug in {x}}"
    normalized = _normalize_docling_math(input_math)
    assert r"\in" not in normalized
    assert "plug in" in normalized


def test_clean_ocr_formula_preserves_function_arguments() -> None:
    """_clean_ocr_formula must not strip trailing function arguments like P(0) or x(0)."""
    from ubt.adapters.pdf.typst_math import _clean_ocr_formula

    assert _clean_ocr_formula("P(0)") == "P(0)"
    assert _clean_ocr_formula("x(0)") == "x(0)"
    assert _clean_ocr_formula("f(1)") == "f(1)"
    assert _clean_ocr_formula(r"y(t) = x(0)") == r"y(t) = x(0)"
    assert _clean_ocr_formula(r"P(A|B) = P(0)") == r"P(A|B) = P(0)"


def test_math_content_sanitizer_blocks_code_in_pandoc_call_arguments() -> None:
    r"""A pandoc layout call's arguments are Typst code, not data.

    Regression: ``_sanitize_math_content`` kept the ``#`` of ``#box(...)``, but
    its argument list can call any function (``#box(raw(read("x")))``) — no
    second ``#`` needed — so a crafted source formula could execute Typst code.
    """
    from ubt.adapters.pdf.typst_math import _sanitize_math_content

    # Injected call is demoted to inert text (its '#' stripped).
    assert "#" not in _sanitize_math_content('#box(raw(read("secret.txt")))')
    # A safe outer call keeps its '#', but the injected inner call is demoted.
    scaled = _sanitize_math_content('#scale(x: 180%)[#read("x.txt")]')
    assert "#read" not in scaled
    assert 'read("x.txt")' in scaled
    # The known-safe pandoc calls keep their '#' (Eq. 3.9 regression guard).
    assert (
        _sanitize_math_content("#scale(x: 180%, y: 180%)[$ x $]")
        == "#scale(x: 180%, y: 180%)[$ x $]"
    )
    assert _sanitize_math_content("#hide[$ x $]") == "#hide[$ x $]"
    # A plain '#' is still dropped.
    assert _sanitize_math_content("a #evil b") == "a evil b"
