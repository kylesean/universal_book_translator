"""Regression fixtures for OCR-damaged formulas from tests/fixtures/synthetic-duo.pdf.

The documented failure classes (b0040/b0114/b0176/b0186/b0178) come from the
real CodeFormulaV2 output stored in job fc1d7bd7b799 and pin what the engines
and the pipeline do with each before T0 repair automation exists. b0114 is the
one case with a verified deterministic repair (strip the extra trailing brace);
the other two syntax cases are Gate-4 refusals that ship as source crops.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from ubt.adapters.pdf.math_renderer import MathjaxRenderer
from ubt.adapters.pdf.typst_math import _clean_ocr_formula

# The case data used to live in tests/fixtures/formula_ocr_damage.json; that
# JSON was removed in 13568a7 (the directory now holds the generated PDF
# corpus), so the cases are inlined here verbatim to keep this regression file
# self-contained (no file I/O at collection time).
_CASES_JSON = r"""{
  "version": 1,
  "source": "tests/fixtures/synthetic-duo.pdf - Docling CodeFormulaV2 OCR LaTeX from job fc1d7bd7b799. raw_latex is the ledger string; latex is what _clean_ocr_formula hands the MathJax engine.",
  "cases": [
    {
      "id": "pdf_main#b0030",
      "damage": "none",
      "engine_ok": true,
      "pipeline_disposition": "native_svg",
      "note": "Control: a well-formed OCR formula from the same corpus renders cleanly.",
      "raw_latex": "\\frac { \\partial ^ { 2 } \\psi _ { 2 } ( x , y ) } { \\partial x ^ { 2 } } = \\frac { q N _ { c h } } { \\varepsilon _ { c h } }",
      "latex": "\\frac { \\partial ^ { 2 } \\psi _ { 2 } ( x , y ) } { \\partial x ^ { 2 } } = \\frac { q N _ { c h } } { \\varepsilon _ { c h } }"
    },
    {
      "id": "pdf_main#b0040",
      "damage": "brace_imbalance",
      "engine_ok": false,
      "pipeline_disposition": "gate4_verbatim_crop",
      "note": "Gate 4 refuses the emission, so _substitute_verbatim_formulas swaps in the source crop before the engine ever sees it; MathJax reports 'Missing close brace' on the cleaned string.",
      "raw_latex": "\\mathcal { E } _ { x s } = \\sqrt { \\frac { 2 q n _ { i } } { \\varepsilon _ { \\text {ch} } } \\left [ V _ { \\text {tm} } \\left ( e ^ { \\frac { \\gamma _ { s } ( y ) } { \\gamma _ { \\text {tm} } } - e ^ { \\frac { \\gamma _ { s } ( y ) } { \\gamma _ { \\text {tm} } } } \\right ) e ^ { \\frac { - \\gamma _ { B } - V _ { \\text {s} } ( y ) } { \\gamma _ { \\text {tm} } } + e ^ { \\frac { \\gamma _ { B } } { \\gamma _ { \\text {tm} } } } ( \\psi _ { s } ( y ) - \\psi _ { 0 } ( y ) ) \\right ) \\right ] }",
      "latex": "\\mathcal { E } _ { x s } = \\sqrt { \\frac { 2 q n _ { i } } { \\varepsilon _ { \\text {ch} } } \\left [ V _ { \\text {tm} } \\left ( e ^ { \\frac { \\gamma _ { s } ( y ) } { \\gamma _ { \\text {tm} } } - e ^ { \\frac { \\gamma _ { s } ( y ) } { \\gamma _ { \\text {tm} } } } \\right ) e ^ { \\frac { - \\gamma _ { B } - V _ { \\text {s} } ( y ) } { \\gamma _ { \\text {tm} } } + e ^ { \\frac { \\gamma _ { B } } { \\gamma _ { \\text {tm} } } } ( \\psi _ { s } ( y ) - \\psi _ { 0 } ( y ) ) \\right ) \\right ] }"
    },
    {
      "id": "pdf_main#b0044",
      "damage": "glued_prose",
      "engine_ok": true,
      "pipeline_disposition": "engine_witness_crop",
      "note": "_clean_ocr_formula strips the glued 'Eqs. (3.7), (3.8) can be written as a single equation:' header; the engine then renders, but the witness flags aspect 4.17x and the source crop ships.",
      "raw_latex": "E q s . \\, ( 3 . 7 ) , ( 3 . 8 ) \\, \\text { can be written as a single equation:} \\\\ f ( \\beta ) \\, = \\, \\ln ( \\beta ) \\, - \\, \\ln ( \\cos ( \\beta ) ) \\, - \\, \\frac { V _ { \\text {gs} } - V _ { \\text {h} } - V _ { \\text {ch} } } { 2 V _ { \\text {tm} } } \\, + \\, \\ln \\left ( \\frac { 2 } { T _ { \\text {fin} } } \\, \\sqrt { \\frac { 2 e \\text {h} V _ { \\text {um} } N _ { \\text {ch} } } { q n _ { \\text {i} } ^ { 2 } } } \\right ) \\\\ + \\, \\frac { 2 \\epsilon } { T _ { \\text {fin} } \\cos \\left ( \\beta \\right ) } \\sqrt { \\beta ^ { 2 } \\left ( \\frac { e \\text {gen} } { v _ { \\text {tm} } } - 1 \\right ) } \\, + \\, \\frac { \\psi _ { \\text {pert} } } { V _ { \\text {t} } ^ { 2 } } \\, [ \\psi _ { \\text {pert} } - 2 V _ { \\text {um} } \\ln ( \\cos ( \\beta ) ) ] \\right ) \\, = \\, 0 \\\\ \\text {where} \\, \\psi _ { \\text {pert} } \\text { is given by } \\psi _ { \\text {t} } \\, \\text {evaluated at } x = T _ { \\text {fm} } / 2 . \\, E q . \\, ( 3 . 1 1 ) \\, \\text { is an implicit equation in}",
      "latex": "f ( \\beta ) \\, = \\, \\ln ( \\beta ) \\, - \\, \\ln ( \\cos ( \\beta ) ) \\, - \\, \\frac { V _ { \\text {gs} } - V _ { \\text {h} } - V _ { \\text {ch} } } { 2 V _ { \\text {tm} } } \\, + \\, \\ln \\left ( \\frac { 2 } { T _ { \\text {fin} } } \\, \\sqrt { \\frac { 2 e \\text {h} V _ { \\text {um} } N _ { \\text {ch} } } { q n _ { \\text {i} } ^ { 2 } } } \\right ) \\\\ + \\, \\frac { 2 \\epsilon } { T _ { \\text {fin} } \\cos \\left ( \\beta \\right ) } \\sqrt { \\beta ^ { 2 } \\left ( \\frac { e \\text {gen} } { v _ { \\text {tm} } } - 1 \\right ) } \\, + \\, \\frac { \\psi _ { \\text {pert} } } { V _ { \\text {t} } ^ { 2 } } \\, [ \\psi _ { \\text {pert} } - 2 V _ { \\text {um} } \\ln ( \\cos ( \\beta ) ) ]  \\, = \\, 0"
    },
    {
      "id": "pdf_main#b0114",
      "damage": "extra_close_brace",
      "engine_ok": false,
      "pipeline_disposition": "engine_render_error_crop",
      "note": "Slipped past the brace gate: one extra '}' at the end. MathJax reports 'Extra close brace or missing open brace'. Stripping one trailing brace renders.",
      "raw_latex": "W = 2 \\sqrt { \\frac { ( T _ { \\text {Fin} , b a s e } - T _ { \\text {Fin} , o p } ) ^ { 2 } } { 4 } + H _ { \\text {Fin} } ^ { 2 } + T _ { \\text {Fin} , o p } } }",
      "latex": "W = 2 \\sqrt { \\frac { ( T _ { \\text {Fin} , b a s e } - T _ { \\text {Fin} , o p } ) ^ { 2 } } { 4 } + H _ { \\text {Fin} } ^ { 2 } + T _ { \\text {Fin} , o p } } }",
      "repair": {
        "rule": "strip_one_trailing_brace",
        "engine_ok": true,
        "latex": "W = 2 \\sqrt { \\frac { ( T _ { \\text {Fin} , b a s e } - T _ { \\text {Fin} , o p } ) ^ { 2 } } { 4 } + H _ { \\text {Fin} } ^ { 2 } + T _ { \\text {Fin} , o p } } "
      }
    },
    {
      "id": "pdf_main#b0176",
      "damage": "merged_equations",
      "engine_ok": true,
      "pipeline_disposition": "engine_witness_crop",
      "note": "Renders, but the OCR merged eq. (A.14c) and (A.14d) into one line: witness flags aspect 4.31x and components 36 vs 10, so the source crop ships.",
      "raw_latex": "T _ { 2 } & = \\sqrt { T _ { 1 } } ^ { \\dots } & ( A . 1 4 c ) 2 \\beta + T _ { a b b } T _ { c } \\tan g 0 + 2 T _ { b } \\beta \\sec g 0 s q T _ { 0 } & & ( A . 1 4 d )",
      "latex": "T _ { 2 } & = \\sqrt { T _ { 1 } } ^ { \\dots }  2 \\beta + T _ { a b b } T _ { c } \\tan g 0 + 2 T _ { b } \\beta \\sec g 0 s q T _ { 0 }"
    },
    {
      "id": "pdf_main#b0186",
      "damage": "missing_close_brace",
      "engine_ok": false,
      "pipeline_disposition": "gate4_verbatim_crop",
      "note": "Gate 4 refusal like b0040; MathJax reports 'Extra open brace or missing close brace' on the cleaned string.",
      "raw_latex": "\\beta _ { \\text {doped} } = e ^ { \\frac { - \\frac { \\sqrt { \\text {ev} } } { 2 } A _ { g } } { 2 ^ { \\frac { \\sqrt { \\text {ev} } } { 2 } \\frac { 1 } { 4 g } } } \\frac { A _ { g } } { r } \\sqrt { \\left ( \\frac { \\ln ( 1 + e ^ { 2 ( F - F _ { \\text {th} } ) } ) } { 2 A _ { g } } + 1 \\right ) ^ { 2 } - 1 } \\quad ( A . 1 2 g )",
      "latex": "\\beta _ { \\text {doped} } = e ^ { \\frac { - \\frac { \\sqrt { \\text {ev} } } { 2 } A _ { g } } { 2 ^ { \\frac { \\sqrt { \\text {ev} } } { 2 } \\frac { 1 } { 4 g } } } \\frac { A _ { g } } { r } \\sqrt { \\left ( \\frac { \\ln ( 1 + e ^ { 2 ( F - F _ { \\text {th} } ) } ) } { 2 A _ { g } } + 1 \\right ) ^ { 2 } - 1 }"
    },
    {
      "id": "pdf_main#b0178",
      "damage": "stacked_reflow",
      "engine_ok": true,
      "pipeline_disposition": "typst_witness_crop",
      "note": "Renders and passes the engine witness; only the legacy typst witness flags the stacked reflow.",
      "raw_latex": "T _ { 4 } = - 2 + 2 T _ { 0 } \\beta _ { 0 } ^ { 2 } \\sec g 0 s q ^ { 2 } + \\sec g 0 s q ( 2 T _ { b } + T _ { a } T _ { c } + 8 T _ { b } \\beta \\tan g 0 + 4 T _ { b } \\beta _ { 0 } ^ { 2 } \\tan g 0 ^ { 2 } ) ( A . 1 4 e )",
      "latex": "T _ { 4 } = - 2 + 2 T _ { 0 } \\beta _ { 0 } ^ { 2 } \\sec g 0 s q ^ { 2 } + \\sec g 0 s q ( 2 T _ { b } + T _ { a } T _ { c } + 8 T _ { b } \\beta \\tan g 0 + 4 T _ { b } \\beta _ { 0 } ^ { 2 } \\tan g 0 ^ { 2 } )"
    }
  ]
}"""
CASES: list[dict[str, Any]] = json.loads(_CASES_JSON)["cases"]
BY_ID = {case["id"]: case for case in CASES}

requires_node = pytest.mark.skipif(
    not MathjaxRenderer().available(), reason="Node + scripts/mathjax deps not installed"
)


@pytest.fixture(scope="module")
def renderer() -> Any:
    engine = MathjaxRenderer()
    yield engine
    engine.close()


@requires_node
def test_engine_outcomes_match_recorded_failures(renderer: MathjaxRenderer) -> None:
    for case in CASES:
        result = renderer.render(case["latex"])
        assert result.ok is case["engine_ok"], (
            f"{case['id']} ({case['damage']}): expected ok={case['engine_ok']}, "
            f"got ok={result.ok} error={result.error!r}"
        )
        if not case["engine_ok"]:
            assert result.error, f"{case['id']}: a failed render must carry a reason"


@requires_node
def test_verifiable_repair_renders(renderer: MathjaxRenderer) -> None:
    case = BY_ID["pdf_main#b0114"]
    repair = case["repair"]
    assert repair["rule"] == "strip_one_trailing_brace"
    assert case["latex"].rstrip().endswith("}")
    broken = renderer.render(case["latex"])
    repaired = renderer.render(repair["latex"])
    assert broken.ok is False
    assert repaired.ok is True


@requires_node
def test_pipeline_cleaning_keeps_gate4_refusals(renderer: MathjaxRenderer) -> None:
    # Cleaning must not "fix" the unbalanced strings silently: the gate and the
    # source-crop fallback own that decision, and a repair here would have to
    # render before it could ship.
    for bid in ("pdf_main#b0040", "pdf_main#b0186"):
        case = BY_ID[bid]
        assert case["pipeline_disposition"] == "gate4_verbatim_crop"
        result = renderer.render(_clean_ocr_formula(case["latex"]))
        assert result.ok is False


def test_cleaner_strips_glued_prose_without_node() -> None:
    # b0044's raw OCR text glues the sentence "Eqs. (3.7), (3.8) can be written
    # as a single equation:" in front of the math; _clean_ocr_formula removes it
    # so the engine sees math only (verified end-to-end in the same run).
    case = BY_ID["pdf_main#b0044"]
    cleaned = _clean_ocr_formula(case["raw_latex"])
    assert cleaned != case["raw_latex"]
    assert cleaned.lstrip().startswith("f ( \\beta )")
    assert "can be written as" not in cleaned
    # And the merged-equation case must stay intact for the witness to catch:
    # cleaning is not allowed to split b0176 into its original two equations
    # (it only strips the trailing "(A.14c)/(A.14d)" number tails).
    merged = BY_ID["pdf_main#b0176"]
    assert "A . 1 4 c" in merged["raw_latex"]
    assert "A . 1 4 c" not in _clean_ocr_formula(merged["raw_latex"])
    assert "\\sqrt { T _ { 1 } }" in _clean_ocr_formula(merged["raw_latex"])
