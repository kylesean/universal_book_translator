"""Inline-math overlay rendering + undelimited-math gate (ch3 Fth,SI class)."""

from pathlib import Path
from typing import Any

import pytest

from tests.unit.ir_seed import SeedDoc, seed_job
from ubt.adapters.pdf.overlay_text import (
    prepare_overlay_text,
    render_overlay_line,
    split_math_spans,
    strip_cjk_latin_spaces,
    typstify_math,
)
from ubt.core.qe.fast_pass import FastPassFilter
from ubt.core.validators.math_guard import target_missing_math_delimiters


def _src136() -> str:
    return "where Ag 0 , r , F , and F th,SI are defined as follows:"


def test_cjk_latin_spaces_pangu_spacing() -> None:
    assert strip_cjk_latin_spaces("式 (A.6) 对于轻掺杂同样是一个很好的近似") == (
        "式 (A.6) 对于轻掺杂同样是一个很好的近似"
    )
    assert strip_cjk_latin_spaces("都是对 β 的良好近似") == "都是对 β 的良好近似"
    assert "F > F_th" in strip_cjk_latin_spaces("当 F > F_th 时")


def test_currency_dollar_survives() -> None:
    out = render_overlay_line("价格 $5 不变")
    assert "\\$5" in out and "$5" not in out.replace("\\$5", "")


def test_math_span_renders_math_mode_with_probe() -> None:
    out = render_overlay_line("其中 $F_{th,SI}$ 的定义如下：", math_probe=lambda body: True)
    assert '$F_#"th,SI"$' in out
    assert "其中" in out and "的定义如下：" in out


def test_math_span_falls_back_without_probe() -> None:
    out = render_overlay_line("其中 $F_{th,SI}$ 的定义如下：")
    assert "$" not in out
    assert "F" in out


def test_failed_probe_falls_back_to_text() -> None:
    out = render_overlay_line("其中 $F_{th,SI}$ 的定义如下：", math_probe=lambda body: False)
    assert "$" not in out


def test_unbalanced_dollar_from_line_split_degrades_gracefully() -> None:
    out = render_overlay_line("其中 $F_{th", math_probe=lambda body: True)
    assert out.count("$") == 0


def test_prepare_keeps_math_bytes_identical() -> None:
    prepared = prepare_overlay_text("其中 Ag0 和 $F_{th,SI}$ 的定义如下：")
    assert "$F_{th,SI}$" in prepared
    assert prepared.startswith("其中 Ag0 和")


def test_typstify_converts_latex_dialect() -> None:
    assert typstify_math("\\psi(x,y)") == "psi(x,y)"
    assert typstify_math("V_tm") == 'V_#"tm"'
    assert typstify_math("\\unknowncmd x") is None
    assert typstify_math("#set x") is None


def test_split_math_spans() -> None:
    parts = split_math_spans("a $x$ b")
    assert parts == [(False, "a "), (True, "$x$"), (False, " b")]
    assert split_math_spans("plain") == [(False, "plain")]


def test_repair_merges_formula_fragmented_by_stray_dollar() -> None:
    raw = r"沟道掺杂浓度，$\psi_B = V_{tm} \ln($N_ch $= n_i)$。注意。"
    prepared = prepare_overlay_text(raw)
    math_parts = [c for is_m, c in split_math_spans(prepared) if is_m]
    assert len(math_parts) == 1
    assert r"N_ch = n_i" in math_parts[0] and r"\ln" in math_parts[0]
    out = render_overlay_line(prepared, math_probe=lambda body: True)
    assert "$" in out and "\\" not in out


def test_repair_does_not_leak_dollars_when_probe_rejects() -> None:
    raw = r"$\psi_B = V_{tm} \ln($N_ch $= n_i)$"
    out = render_overlay_line(prepare_overlay_text(raw), math_probe=lambda body: False)
    assert out.count("$") == 0


def test_repair_leaves_balanced_neighbors_alone() -> None:
    assert prepare_overlay_text("式中 $a$ 与 $b$ 的关系") == "式中 $a$ 与 $b$ 的关系"


def test_repair_never_merges_across_cjk_or_sentence_breaks() -> None:
    raw = r"$\ln($ 中文 $= n_i)$"
    prepared = prepare_overlay_text(raw)
    assert prepared.count("$") == 4 and "中文" in prepared
    raw2 = r"$\ln($。$= n_i)$"
    assert prepare_overlay_text(raw2).count("$") == 4


def test_repair_leaves_currency_and_lone_fragment_untouched() -> None:
    assert prepare_overlay_text("价格 $5 到 $10 不等") == "价格 $5 到 $10 不等"
    assert prepare_overlay_text("其中 $F_{th") == "其中 $F_{th"


def test_undelimited_gate_fires_on_136() -> None:
    assert target_missing_math_delimiters(_src136(), "其中 Ag0、r、F 和 Fth,SI 的定义如下：")


def test_undelimited_gate_quiet_cases() -> None:
    assert not target_missing_math_delimiters(
        "Using these parameters, a FinFET can be modeled.", "利用这些参数，可以对FinFET建模。"
    )
    assert not target_missing_math_delimiters(_src136(), "其中定义如下。")
    assert not target_missing_math_delimiters("The price is $5.", "价格是$5。")


def test_fastpass_routes_undelimited_math_to_repair() -> None:
    gate = FastPassFilter()
    decision = gate.validate_structural_invariants(
        _src136(), "其中 Ag0、r、F 和 Fth,SI 的定义如下："
    )
    assert not decision.passed and "Undelimited math" in decision.reason


def test_undelimited_gate_exempts_formula_and_skipped() -> None:
    from ubt.core.ir.models import BlockType

    gate = FastPassFilter()
    src = "A _ { g 0 } = frac { r psi _ { p e r t } } { V _ { t m } }"
    d = gate.validate_structural_invariants(src, src, block_type=BlockType.FORMULA)
    assert d.passed, d.reason
    d = gate.validate_structural_invariants(src, src, skip_translate=True)
    assert d.passed, d.reason
    d = gate.validate_structural_invariants(
        "where Ag 0 , r , F , and F th,SI are defined as follows:",
        "其中 Ag0、r、F 和 Fth,SI 的定义如下：",
        block_type=BlockType.NARRATIVE,
    )
    assert not d.passed and "Undelimited math" in d.reason


# --- Hallucinated-LaTeX class (chapter-1 b0004: repair turned "(2D)" into
# $2\mathrm{D}$, invisible in publication, literal garbage in overlay) ---

_B0004_SRC = (
    "After the industry embraced the FinFET technology for production, "
    "BSIM-CMG was selected as the industry standard FinFET model, without "
    "competition, by the Compact Model Council in 2012. The council members "
    "are major integrated IC companies, fabless companies, design automation "
    "companies, and IC foundries."
)
_B0004_CLEAN_DRAFT = "在业界将FinFET技术投入量产之后，BSIM-CMG被选为业界标准的FinFET模型（2D）。"
_B0004_REPAIRED = (
    "在业界将FinFET技术投入量产之后，BSIM-CMG被选为业界标准的FinFET模型（$2\\mathrm{D}$）。"
)
_LONG_PROSE_WITH_ARTICLES = (
    "After the industry embraced the new technology for production, a major "
    "switch to a better transistor structure is unavoidable and a new "
    "generation of engineers celebrated the milestone together."
)


def test_undelimited_gate_quiet_on_long_prose_with_articles() -> None:
    """3c density fix: six lone articles in a 75-word paragraph are prose,
    not a math signal (previously score 2 -> bogus repair round)."""
    assert not target_missing_math_delimiters(_B0004_SRC, _B0004_CLEAN_DRAFT)
    assert not target_missing_math_delimiters(
        _LONG_PROSE_WITH_ARTICLES, "业界采用新技术后，一次重大转向不可避免（2D）。"
    )


def test_undelimited_gate_still_fires_on_dense_fragments() -> None:
    """The ch3 true positive (3 singles / 16 tokens ≈ 0.19) still trips."""
    assert target_missing_math_delimiters(_src136(), "其中 Ag0、r、F 和 Fth,SI 的定义如下：")


def test_novel_command_detector_flags_unrenderable() -> None:
    r"""\mathbf is real LaTeX but outside the overlay probe's dialect:
    invented in target, uncompilable downstream -> flagged. (\mathrm is
    NOT flagged: the pre-pass compiles it, so it renders correctly.)"""
    from ubt.core.validators.math_guard import novel_unsupported_latex_commands

    assert novel_unsupported_latex_commands(_B0004_SRC, "二维（$\\mathbf{D}$）平面。") == ["mathbf"]
    assert novel_unsupported_latex_commands(_B0004_SRC, _B0004_REPAIRED) == []


def test_novel_command_detector_allows_renderable_newcomers() -> None:
    """\\text / Greek / \\frac compile downstream — flagging them would
    fight the math-reconstruction prompt rule."""
    from ubt.core.validators.math_guard import novel_unsupported_latex_commands

    assert (
        novel_unsupported_latex_commands(
            "The off current Ioff flows.", "关态电流$I_{\\text{off}}$在此流动。"
        )
        == []
    )
    assert novel_unsupported_latex_commands("Area 2 pi R.", "面积$2\\pi R$。") == []


def test_novel_command_detector_exempts_source_commands() -> None:
    from ubt.core.validators.math_guard import novel_unsupported_latex_commands

    src = "With $\\frac{a}{b}$ and $\\mathrm{X}$ as defined."
    assert novel_unsupported_latex_commands(src, "其中$\\frac{a}{b}$和$\\mathrm{X}$如定义。") == []


def test_fastpass_routes_hallucinated_latex_to_repair() -> None:
    gate = FastPassFilter()
    decision = gate.validate_structural_invariants(_B0004_SRC, "二维（$\\mathbf{D}$）平面。")
    assert not decision.passed and "Hallucinated LaTeX" in decision.reason
    # The shipped b0004 form compiles downstream, so the gate passes it;
    # the prompt guard (don't-invent rule) is its layer, tested separately.
    ok = gate.validate_structural_invariants(_B0004_SRC, _B0004_REPAIRED)
    assert ok.passed, ok.reason


def test_fastpass_passes_renderable_math_wrapping() -> None:
    gate = FastPassFilter()
    decision = gate.validate_structural_invariants(
        "The off current Ioff flows through the channel today.",
        "关态电流$I_{\\text{off}}$流经沟道。",
    )
    assert decision.passed, decision.reason


def test_typstify_converts_textlike_wrappers() -> None:
    assert typstify_math("2\\mathrm{D}") == "2D"
    assert typstify_math("I_{\\text{off}}") == 'I_#"off"'
    converted = typstify_math("T_{\\text{si}} \\ll L_g")
    assert converted == 'T_#"si" lt.double L_g'
    assert typstify_math("a \\pm b") == "a plus.minus b"
    assert typstify_math("a \\leq b") == "a lt.eq b"
    assert typstify_math("\\mathbf{x}") is None  # styling lies stay fail-closed
    assert typstify_math("\\text{a{b}}") is None  # nested braces stay fail-closed


def test_renderable_set_pinned_to_render_table() -> None:
    """Core allowlist and adapter map must move together (vacant discipline:
    core owns the set, the test — not imports — enforces sync)."""
    from ubt.adapters.pdf.overlay_text import _LATEX_CMD_MAP
    from ubt.core.validators.math_guard import RENDERABLE_LATEX_COMMANDS

    assert set(_LATEX_CMD_MAP) | {"text", "mathrm"} == RENDERABLE_LATEX_COMMANDS


def test_overlay_renders_converted_mathrm() -> None:
    out = render_overlay_line("二维（$2\\mathrm{D}$）平面", math_probe=lambda body: True)
    assert "$2D$" in out
    assert "\\" not in out.replace("\\\\", "")


def test_typstify_math_scientific_and_operators() -> None:
    assert typstify_math(r"N_{ch}=5\times10^{18}") == 'N_#"ch"=5 times 10^(18)'
    assert typstify_math(r"1\times 10^{15}") == "1 times 10^(15)"
    assert typstify_math(r"5\times10") == "5 times 10"


def test_typstify_math_bare_scripts() -> None:
    assert typstify_math("^{-3}") == '""^(-3)'
    assert typstify_math("^{18}") == '""^(18)'
    assert typstify_math("^2") == '""^2'


def test_typstify_math_avoids_code_function_call_collision() -> None:
    # #"ch"(0) would be parsed as calling string #"ch" as a function in Typst
    converted = typstify_math("V_{ch}(0) = V_s")
    assert converted == 'V_#"ch" (0) = V_s'


def test_typstify_math_partial_and_latex_spaces() -> None:
    converted = typstify_math(r"H_{FIN}=L=1\ \mu\mathrm{m}")
    assert converted == 'H_#"FIN"=L=1 mu m'
    diff_expr = typstify_math(r"f_n = \frac{\partial^n f}{\partial \beta^n}\Big|_{\beta = \beta_0}")
    assert diff_expr is not None
    assert "partial" in diff_expr
    assert "diff" not in diff_expr


def test_math_probe_normalizes_enclosing_dollars() -> None:
    from ubt.adapters.pdf.typst_math_probe import TypstMathProbe

    probe = TypstMathProbe()
    # Both bare Typst math expression and $-enclosed expression should probe correctly
    assert probe.check('N_#"ch" = 5')
    assert probe.check('$N_#"ch" = 5$')


def test_math_probe_compiles_with_the_configured_binary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The probe must use the same typst as the real compile.

    A probe pinned to PATH ``typst`` while the document honors
    ``UBT_TYPST_BINARY`` reports a false negative for every formula whenever
    typst lives off PATH — and being pure-cache, that one wrong verdict escapes
    all math to text for the whole run.
    """
    import ubt.adapters.pdf.typst_compile as typst_compile_mod
    from ubt.adapters.pdf.typst_math_probe import TypstMathProbe

    seen: list[str] = []

    def fake_compile(
        typ_path: str, pdf_path: str, binary: str = "typst", timeout: float = 120.0
    ) -> tuple[bool, str]:
        seen.append(binary)
        return True, ""

    monkeypatch.setattr(typst_compile_mod, "typst_compile", fake_compile)
    TypstMathProbe("/opt/typst-0.15.1").check("x + y")
    assert seen == ["/opt/typst-0.15.1"]


def test_inline_and_display_symbol_maps_do_not_drift() -> None:
    """The two LaTeX→Typst tables must agree except for documented overrides.

    overlay_text.typstify_math and typst_math._latex_math_to_typst serve
    different output contexts; seven commands intentionally map to different
    (both valid) Typst spellings. Every other shared command must agree, so a
    symbol added to one table cannot silently miss the other.
    """
    from ubt.adapters.pdf.overlay_text import _LATEX_CMD_MAP
    from ubt.adapters.pdf.typst_math import _TEX_SYMBOL_MAP

    documented_overrides = {
        "cdot",
        "leq",
        "geq",
        "neq",
        "vartheta",
        "varphi",
        "varepsilon",
    }
    shared = set(_LATEX_CMD_MAP) & set(_TEX_SYMBOL_MAP)
    drifted = {k for k in shared if _LATEX_CMD_MAP[k] != _TEX_SYMBOL_MAP[k]}
    assert drifted == documented_overrides, (
        f"symbol tables drifted; update both or document the override: {drifted ^ documented_overrides}"
    )


# ---------------------------------------------------------------------------
# Unicode-vs-LaTeX equivalence in the hallucination gate (2026-09-24).
# ---------------------------------------------------------------------------


def test_novel_command_detector_treats_unicode_source_as_equivalent_latex() -> None:
    """A source symbol encoded as Unicode is the SAME symbol in LaTeX.

    Docling flattens most source math to plain text: ``x ∈ S`` arrives as a
    literal U+2208, not ``x \\in S``. A translator that re-encodes it the
    conventional way emits ``\\in``, and a set-membership check over control
    sequences then reports the target as having invented a command the source
    never mentioned. The math is identical; only the encoding moved.
    """
    from ubt.core.validators.math_guard import novel_unsupported_latex_commands

    assert novel_unsupported_latex_commands("for x ∈ S we have", "对$x \\in S$有") == []
    assert (
        novel_unsupported_latex_commands("a ≃ b and a ≈ b", "有$a \\simeq b$和$a \\approx b$") == []
    )
    assert novel_unsupported_latex_commands("f: X → Y", "映射$f: X \\to Y$") == []
    assert (
        novel_unsupported_latex_commands("A ∪ B and A ∩ B", "并集$A \\cup B$与交集$A \\cap B$")
        == []
    )
    assert (
        novel_unsupported_latex_commands(
            "x ≠ y, x ≤ y, ∀x", "有$x \\neq y$、$x \\le y$、$\\forall x$"
        )
        == []
    )
    assert novel_unsupported_latex_commands("f ∘ g", "复合$f \\circ g$") == []
    assert novel_unsupported_latex_commands("S = ∅", "有$S = \\emptyset$") == []


def test_unicode_equivalence_does_not_weaken_the_hallucination_gate() -> None:
    """Normalizing encodings must not open a hole: a command with no Unicode
    counterpart and no source presence is still fabrication, and a Unicode
    symbol in the source does not license unrelated invented commands."""
    from ubt.core.validators.math_guard import novel_unsupported_latex_commands

    assert "coloneqq" in novel_unsupported_latex_commands("x ∈ S", "有$x \\in S$和$y \\coloneqq z$")
    assert novel_unsupported_latex_commands("x ∈ S", "有$x \\in S$和$\\mathbf{v}$") == ["mathbf"]


def test_fastpass_passes_unicode_to_latex_math_reencoding() -> None:
    """End-to-end through the gate itself."""
    gate = FastPassFilter()
    decision = gate.validate_structural_invariants(
        "The set membership x ∈ S holds for all cases.",
        "集合隶属关系对$x \\in S$成立。",
    )
    assert decision.passed, decision.reason


# ---------------------------------------------------------------------------
# ubt recheck-gates: judging a quarantined block's draft_text, not the
# placeholder triage leaves in target_text.
# ---------------------------------------------------------------------------


def _qblock(source: str, draft: str) -> Any:
    from ubt.core.ir.models import BlockType, FlowID, IRBlock

    return IRBlock(
        id="pdf_main#b1",
        flow_id=FlowID.MAIN_STORY,
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        source_text=source,
        draft_text=draft,
        target_text=draft,
    )


def _quarantined_ledger(tmp_path: Path, *, source: str, draft: str) -> None:
    """A job whose one block is quarantined the way triage leaves it: the real
    draft in draft_text, an HTML placeholder in target_text. The file is named
    ``<job_id>.sqlite`` because that is what the ledger consumers resolve."""
    from ubt.core.engine.ledger import SQLiteJobLedger
    from ubt.core.ir.models import BlockStatus

    ledger = SQLiteJobLedger(tmp_path / "job_q.sqlite")
    seed_job(
        ledger,
        "job_q",
        SeedDoc(
            doc_id="d1",
            source_path="x.pdf",
            format_type="pdf",
            metadata={},
            blocks=[_qblock(source, draft)],
        ),
        target_lang="zh",
    )
    ledger.save_checkpoint(
        block_id="pdf_main#b1",
        status=BlockStatus.BLOCKED_HUMAN,
        target_text=(
            '<mark class="ubt-blocked-human" title="MQM Critical unresolved">'
            "【待人工审校】source</mark>"
        ),
        draft_text=draft,
        error_flags=["mqm_critical_blocked"],
    )
    ledger.close()


def test_recheck_reads_draft_text_not_the_quarantine_placeholder(tmp_path: Path) -> None:
    """The whole point: a block quarantined for hallucinated \\simeq now passes
    the Unicode-aware gate. Read through ``target_text`` this can never happen —
    the placeholder has no LaTeX, so every quarantined block looks clean."""
    from ubt.cli.commands.status import recheck_gates

    _quarantined_ledger(tmp_path, source="a ≃ b and c ∈ S", draft="有 $a \\simeq b$ 和 $c \\in S$")
    report = recheck_gates("job_q", db_dir=tmp_path)
    assert report["total"] == 1
    assert report["would_pass"] == 1, report
    assert report["still_failing"] == 0


def test_recheck_still_reports_a_genuine_hallucination(tmp_path: Path) -> None:
    """\\mathcal has no Unicode counterpart here, so it is real fabrication."""
    from ubt.cli.commands.status import recheck_gates

    _quarantined_ledger(
        tmp_path,
        source="In the simply typed lambda calculus, Γ ⊢ t : T",
        draft="在简单类型演算中，有 $\\mathcal{E}$ 与 $\\mathfrak{F}$",
    )
    report = recheck_gates("job_q", db_dir=tmp_path)
    assert report["total"] == 1
    assert report["would_pass"] == 0, report
    assert report["still_failing"] == 1


def test_recheck_reports_missing_ledger_without_raising(tmp_path: Path) -> None:
    from ubt.cli.commands.status import recheck_gates

    report = recheck_gates("job_nope", db_dir=tmp_path)
    assert report["total"] == 0
    assert report.get("error")


def test_typstify_math_handles_context_math_from_formula_dense_page() -> None:
    """arXiv 2608.25512 p9 exposed a display/inline symbol-table split.

    These expressions are ordinary paragraph-level context types, not display
    formulas. If ``to``/``mapsto``/``circ`` remain display-only, rigid overlay
    falls back to escaped LaTeX and the delivered PDF shows reader-visible
    commands such as ``\\Gamma``.
    """
    samples = {
        r"\Gamma \to \Gamma \times (\Gamma \to \Gamma)": "Gamma -> Gamma times (Gamma -> Gamma)",
        r"f : X \rightsquigarrow Y": "f : X ⇝ Y",
        r"\gamma \mapsto \mathrm{pr}_1(f(\gamma, x)) : \Gamma \to \Gamma": 'gamma |-> #"pr"_1(f(gamma, x)) : Gamma -> Gamma',
        r"g \circ f": "g compose f",
    }
    for latex, expected in samples.items():
        converted = typstify_math(latex)
        assert converted == expected
        assert converted is not None
