#!/usr/bin/env python3
"""Generate the repo's synthetic PDF corpus (tests/fixtures/synthetic-*.pdf).

Replaces the retired Elsevier chapter samples (2026-09 legal review) with
typographically equivalent fixtures that are original text owned by this
project. The structural properties the test suite depends on are encoded
here and asserted at the end of this script:

  synthetic-mono.pdf  13 pages, US Letter, single column, running heads,
                      tagged display formulas, a multi-line table, vector
                      figures, and an IEEE-shaped References section.
  synthetic-duo.pdf   26 pages, 540pt x 665.972pt, two columns, tagged
                      equations (1.1)..(3.9), (A.1)..(A.8), appendix.

Regenerate after editing the templates:
    uv run python scripts/make_sample_corpus.py
Requires the ``typst`` binary (see README system dependencies).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CORPUS_DIR = REPO / "tests" / "fixtures"

# The equation helper: body on the left/center, printed tag in parentheses on
# the right margin, mirroring the IEEE look the formula-tag recovery tests
# consume. Tags are plain text in the PDF text layer on purpose.
PRELUDE = r"""
#let eqnum = counter("eqnum")
#let eq(tag, body) = {
  grid(
    columns: (1fr, auto),
    gutter: 1em,
    align: (center, right),
    block(body),
    text(9.2pt)[#tag],
  )
}
"""

BODY = r"""
= The Synthetron Effect: Charge Redistribution in a Fictitious Narrow-Gap Device

== Introduction

This paper describes the synthetron, an invented device used to exercise every
typographic path of a translation pipeline: running heads, tagged display
formulas, a multi-line table, vector figures, and an IEEE-shaped reference
list. The physics here is deliberately fictional; the layout is deliberately
real. Any resemblance to actual semiconductor behavior is coincidental and
unverified.

The remainder of this document is organized as follows. Section 2 introduces
the compact model. Section 3 derives the central charge relation. Section 4
presents the simulation methodology. Section 5 reports synthetic results, and
the appendix collects the auxiliary identities reused throughout.

== Compact Model Overview

The synthetron operates by shuffling charge between a gate plate and a
fictitious channel reservoir. In the on state, the reservoir density obeys

#eq("(1.1)", $ rho_"on" (x) = rho_0 dot "exp" ( -x / ell ) $)

where $rho_0$ is the inversion prefactor and $ell$ is the screening length.
Fig. 1 sketches the cross section. The transfer characteristic follows the
square law up to the onset velocity saturation, which Section 3 captures in
closed form. The threshold condition for reservoir inversion reads

#eq("(1.2)", $ V_"th" = V_"FB" + phi_s + E_"ox" ell $)

with flat-band voltage $V_"FB"$, surface potential $phi_s$, and field $E_"ox"$ in
the oxide. Equation (1.2) anchors everything that follows; its symbols are
tabulated in Section 4.

== Central Charge Relation

Differentiating (1.2) under the quasistatic approximation yields the channel
charge per unit area,

#eq("(2.1)", $ Q_"ch" = -C_"ox" ( V_G - V_"th" ) dot ( 1 + lambda V_D^2 / V_c^2 ) $)

where $C_"ox"$ is the oxide capacitance per unit area and $lambda$ is the
fictional drain-modulation coefficient. For small drain bias (2.1) collapses
to the linear-response form

#eq("(2.2)", $ Q_"ch"^"lin" = -C_"ox" ( V_G - V_"th" ) $)

The measurable quantity in synthetic experiments is the transconductance
derived from (2.1) with the series expansion truncated at second order,

#eq("(3.1)", $ g_m = d I_D / d V_G = mu_"eff" C_"ox" ( W / L ) ( V_G - V_"th" ) $)

which is the central result of this paper. The effective mobility $mu_"eff"$
absorbs all model error and is fitted per device. The noise extension replaces
the deterministic field with its two-point correlator,

#eq("(3.2)", $ S_I ( f ) = 4 k_B T gamma g_m dot ( 1 + eta / f ) $)

with excess-noise factor $gamma$ and flicker corner $eta$; (3.2) serves as a
consistency check only, since the synthetic devices were never measured. Fig. 2
shows the predicted $g_m$ response against gate and drain bias. We further
record, without verifying,

#eq("(3.3)", $ Z_"ch" = ( j omega C_"ch" )^(-1) $)

#eq("(3.4)", $ ell_"phi" approx sqrt( D tau_"phi" ) $)

#eq("(3.5)", $ Delta Q = alpha ( V_G - V_"th" )^2 $)

#eq("(3.6)", $ R_c = R_"spreader" + R_"interface" $)

#eq("(3.7)", $ V_c = ell sqrt( 2 q rho_0 / epsilon_"ch" ) $)

#eq("(3.8)", $ gamma = 2/3 + delta $)

and the matching condition for the synthetic amplifier stage,

#eq("(3.9)", $ Z_"in" = Z_0^* $)

Each relation above reuses symbols defined earlier, keeping the formula
inventory dense on purpose.

== Simulation Methodology

The synthetic devices were defined on a non-uniform mesh with $N_x = 64$ and
$N_y = 32$ cells and relaxed until the residual dropped below $10^-8$. Table 1
lists the default parameters. All numbers are fabricated; every value was
chosen for digit variety so downstream numeric-consistency gates have real
work to do. The solver iterates the coupled system with a fixed-point loop;
convergence typically needed 14 iterations, and never fewer than 7. The
extraction step maps the solved potential back to (2.1) and (3.1); the
resulting currents are reported without re-normalization. We repeated every
simulation at 300 K, 225 K, and 45.1 K to sample the temperature axis broadly.

== Results and Discussion

The extracted transconductance tracks (3.1) within 4.5% for
$V_G - V_"th" < 0.4$ and drifts to 11.2% near the onset, consistent with the
curvature term (3.5). Fig. 3 plots the residuals. The noise floor sits at
$4.1 x 10^-22 "A"^2/"Hz"$, above the thermal expectation by the factor in
(3.2), which we attribute to the fictitious excess-noise term rather than to
any physical mechanism.

The temperature sweep shows no phase transition, unsurprising for an invented
material: $mu_"eff"$ falls off as $T^(-1.5)$ with an unexplained shoulder near
45.1 K that a braver paper would chase. We leave the shoulder, the 2.2% mass
discrepancy, and the sign of $eta$ to future synthetic work.

== Conclusion

We presented the synthetron, a fictional device whose equations are real
enough to typeset. The central relation (3.1) reproduces the expected gate
dependence, the noise model (3.2) fits within measurement error of numbers
nobody measured, and the layout stresses every path a bilingual reflow engine
must survive: display math with printed tags, inline math, subscripts, Greek
symbols, units, a table, three figures, and an appendix of eight identities.

#heading(numbering: none)[References]

#set par(first-line-indent: 0em, justify: false)

#block(spacing: 0.5em)[
  [1. K. Alekhine and B. Montton, "Charge redistribution in narrow-gap
  fictional materials," #emph[Synth. J. Appl. Phys.], vol. 12, no. 3,
  pp. 341-358, Mar. 2019, doi: 10.5555/SJAP.2019.0341.]

  [2. P. Valmary, "The invented transistor at forty," #emph[Proc. Fictitious
  Intl. Conf. Synth. Electron. (FICSE)], 2021, pp. 88-97,
  doi: 10.5555/FICSE.2021.0088.]

  [3. R. Tsubasa, J. Iyer, and M. Kowalski, "Screening lengths in
  reservoir-channel systems: a review," #emph[Synth. Rev. Lett.], vol. 4,
  pp. 15-31, 2022.]

  [4. A. Novak, #emph[Lecture Notes on Fictional Devices], vol. 2.
  Imagina Press, 2018, ch. 7.]

  [5. T. Emeka and G. Lindqvist, "Beta-statistics prefactors for the
  unmeasured: a user's guide," #emph[J. Apocryphal Eng.], vol. 77,
  no. 1, pp. 1-22, Jan. 2024.]
]

#heading(numbering: none)[Appendix: Auxiliary Identities]

The auxiliary identities reused in the text are collected here with their own
tag series, so two numbering namespaces exist in one document.

#eq("(A.1)", $ I_1 = integral_0^"inf" rho_"on" (x) d x = rho_0 ell $)

#eq("(A.2)", $ I_2 = integral_0^"inf" x rho_"on" (x) d x = rho_0 ell^2 $)

#eq("(A.3)", $ kappa = epsilon_"ch" / epsilon_0 $)

The beta-statistics prefactor used in the shoulder analysis reads

#eq("(A.4)", $ beta_"ST" = ( 1 + 3 lambda V_D^2 / ( 2 V_c^2 ) )^(-1/2) $)

followed by the two bookkeeping identities

#eq("(A.5)", $ eta = eta_"th" + eta_"flicker" $)

#eq("(A.6)", $ D = mu_"eff" k_B T / q $)

#eq("(A.7)", $ S_0 = k_B T / q $)

#eq("(A.8)", $ L_D = sqrt( epsilon_"ch" V_t / ( q rho_0 ) ) $)
"""

FIGURES = r"""
#figure(
  place(center)[
    block(width: 6cm, height: 3cm)[
      #place(top+left, dx: 0.2cm, dy: 2.6cm)[#line(length: 5.4cm, stroke: 0.8pt)]
      #place(top+left, dx: 0.4cm, dy: 0.2cm)[#line(angle: 90deg, length: 2.4cm, stroke: 0.8pt)]
      #place(top+left, dx: 1.6cm, dy: 0.4cm)[#line(angle: 90deg, length: 2.0cm, stroke: 0.5pt)]
      #place(top+left, dx: 3.4cm, dy: 0.4cm)[#line(angle: 90deg, length: 2.0cm, stroke: 0.5pt)]
      #place(top+left, dx: 2.2cm, dy: 0.9cm)[#circle(radius: 0.5cm, stroke: 0.8pt)]
      #place(top+left, dx: 0.8cm, dy: 1.6cm)[#line(length: 1.3cm, angle: -21deg, stroke: 0.7pt)]
      #place(top+left, dx: 2.0cm, dy: 1.1cm)[#line(length: 1.4cm, angle: 8deg, stroke: 0.7pt)]
      #place(top+left, dx: 3.4cm, dy: 1.3cm)[#line(length: 1.6cm, angle: -35deg, stroke: 0.7pt)]
    ]
  ],
  caption: [Cross section of the fictitious synthetron: gate plate, screening
    boundaries, reservoir, and the invented potential path.],
) <fig-one>

#figure(
  place(center)[
    block(width: 6cm, height: 3cm)[
      #place(top+left, dx: 0.2cm, dy: 2.7cm)[#line(length: 5.4cm, stroke: 0.8pt)]
      #place(top+left, dx: 0.2cm, dy: 2.7cm)[#line(angle: 270deg, length: 2.4cm, stroke: 0.8pt)]
      #place(top+left, dx: 0.6cm, dy: 2.2cm)[#line(length: 1.6cm, angle: -18deg, stroke: 0.9pt)]
      #place(top+left, dx: 2.1cm, dy: 1.7cm)[#line(length: 1.6cm, angle: -12deg, stroke: 0.9pt)]
      #place(top+left, dx: 3.6cm, dy: 1.4cm)[#line(length: 1.7cm, angle: -36deg, stroke: 0.9pt)]
      #for i in range(8) [
        #place(top+left, dx: 0.8cm + 0.55cm * i, dy: 2.55cm - 0.24cm * i)[
          #line(length: 0.16cm, stroke: 0.6pt)
        ]
      ]
    ]
  ],
  caption: [Predicted transconductance from (3.1): sampled operating points on
    the model curve.],
) <fig-two>

#figure(
  place(center)[
    block(width: 6cm, height: 2.8cm)[
      #let heights = (0.4, 0.66, 0.9, 1.14, 1.38, 0.52, 0.78, 1.02, 1.26, 1.5, 0.6, 0.86)
      #for i in range(12) [
        #place(top+left, dx: 0.35cm * i, dy: 2.5cm)[
          #line(angle: 270deg, length: heights.at(i) * 1cm, stroke: 1.4pt + gray)
        ]
      ]
    ]
  ],
  caption: [Residual histogram versus (3.1); the long right tail is the
    shoulder near 45.1 K.],
) <fig-three>
"""

TABLE = r"""
#figure(
  table(
    columns: 4,
    align: (left, left, center, left),
    stroke: 0.5pt,
    inset: 5pt,
    [$C_"ox"$], [oxide capacitance], [2.5e-3 F/cm²], [fabricated],
    [$mu_"eff"$], [effective mobility], [340 "cm"^2/"Vs"], [fitted],
    [$ell$], [screening length], [22 "nm"], [assumed],
    [$V_c$], [critical voltage], [0.87 "V"], [derived],
    [$gamma$], [excess-noise factor], [1.2], [asserted],
    [$N_x$], [mesh cells], [64], [solver],
    [$T$], [temperature], [300 "K"], [nominal],
    [$eta$], [flicker corner], [0.0125 "Hz"], [fitted],
  ),
  caption: [Default parameters of the synthetic devices. Every number exists
    to give numeric-fidelity gates digit shapes to guard; none is real.],
)
"""

PADDING = r"""
#heading(numbering: none)[Supplementary Variants]

The following numbered variants stretch the document to the page counts the
fixture contract needs while keeping the sampled text math-carrying: every
variant quotes an evaluated identity so font-density heuristics (the TUI
advisor) see a real paper's texture, not prose filler.

#for i in range(NPAD) [
  #par[Variant #(i + 1). Re-measuring the reservoir of Section 3 with
  coefficient #(i / 7 + 0.13) gives $g = #(i - calc.floor(i / 9) * 9)$, $V_c = #(i + 1)$,
  root $sqrt(V_c) = #((i + 1) / 10)$ and a screening length of
  #(18 + 2 * i) nm, within the (3.1) tolerance whenever the shoulder term
  stays below #(i / 11 + 2) percent. The sign of the correction remains
  unexplained, as promised.]
]
"""


def build_doc(running_head: str, columns: str, paper: str, npad: int) -> str:
    body = BODY.replace("== Simulation Methodology", TABLE + "\n== Simulation Methodology")
    body = body + FIGURES
    padding = PADDING.replace("NPAD", str(npad))
    return f"""
{PRELUDE}
#let running_head = "{running_head}"
#set page({paper}, columns: {columns}, header: context align(horizon)[
  #text(7.3pt)[SYNTHETIC SAMPLE JOURNAL OF ENGINEERING, VOL. 1, NO. 1, JANUARY 2026 \\\\
  #running_head — Charge Redistribution in a Fictitious Narrow-Gap Device]
])
#set text(font: "New Computer Modern", size: 9.2pt, lang: "en")
#set par(justify: true)
#set heading(numbering: "1.")

{body}
{padding}
"""


def compile_pdf(typst_src: str, out: Path) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".typ", delete=False) as fh:
        fh.write(typst_src)
        src = Path(fh.name)
    # Compile into a sibling temp file and rename: concurrent sessions (pytest
    # xdist workers self-healing a missing corpus) must never observe a
    # half-written PDF at the final path. Keep the .pdf suffix -- typst infers
    # the output format from it.
    tmp = out.with_name(f"{out.name}.tmp{os.getpid()}.pdf")
    try:
        proc = subprocess.run(
            ["typst", "compile", str(src), str(tmp)],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print(proc.stderr[:4000])
            raise SystemExit(f"typst compile failed for {out}")
        tmp.replace(out)
    finally:
        src.unlink(missing_ok=True)
        tmp.unlink(missing_ok=True)


def page_count(pdf: Path) -> int:
    import pypdfium2 as pdfium  # noqa: PLC0415

    doc = pdfium.PdfDocument(str(pdf))
    try:
        return len(doc)
    finally:
        doc.close()


def make_damaged_variant(src: Path, dst: Path) -> None:
    """Rewrite a slice of every ToUnicode map to MacRoman-residue codepoints.

    Reproduces the real-world damage mode the extraction witness exists for
    (a font that CLAIMS the wrong unicode for its glyphs, so extraction bleeds
    ¼/ð/Þ where the page actually shows latin letters) without shipping a
    third-party copyrighted file that happened to suffer it.
    """
    import re

    import pikepdf

    with pikepdf.open(src) as pdf:
        done: set[int] = set()
        idx = 0
        for page in pdf.pages:
            fonts = page.get("/Resources", pikepdf.Dictionary()).get("/Font", {}) or {}
            for fkey in list(fonts.keys()):
                try:
                    font = fonts[fkey]
                    if "/ToUnicode" not in font:
                        continue
                    tu = font["/ToUnicode"]
                    objgen = tu.objgen
                    if objgen in done:
                        continue
                    done.add(objgen)
                    text = tu.read_bytes().decode("latin-1")

                    def line_repl(m: re.Match[str]) -> str:
                        nonlocal idx
                        idx += 1
                        keep = f"<{m.group(1)}> <{m.group(2)}>" if idx % 9 else None
                        if keep is not None:
                            return f"<{m.group(1)}> <{m.group(2)}>"
                        dst_code = "00BC" if idx % 27 == 0 else "00F0"
                        return f"<{m.group(1)}> <{dst_code}>"

                    new = re.sub(r"<([0-9A-Fa-f]{2,4})>\s*<([0-9A-Fa-f]{4})>", line_repl, text)
                    font["/ToUnicode"] = pikepdf.Stream(pdf, new.encode("latin-1"))
                except Exception:
                    continue
        tmp = dst.with_name(f"{dst.name}.tmp{os.getpid()}.pdf")
        try:
            pdf.save(tmp)
            tmp.replace(dst)
        finally:
            tmp.unlink(missing_ok=True)


#: Files that make up the corpus; used by the pytest self-heal completeness check.
CORPUS_NAMES = ("synthetic-mono.pdf", "synthetic-duo.pdf", "synthetic-duo-damaged.pdf")


def generate_all(force: bool = False) -> list[Path]:
    """Build any missing corpus PDF under ``CORPUS_DIR``; return the paths.

    Programmatic entry shared by ``main()`` and the pytest self-heal hook
    (``tests/conftest.py``): with ``force=False`` only absent files are built,
    so a test session pays the typst compile cost exactly once per gap. The
    damaged variant follows its source: rebuilding ``synthetic-duo.pdf``
    invalidates it.
    """
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    duo = CORPUS_DIR / "synthetic-duo.pdf"
    damaged = CORPUS_DIR / "synthetic-duo-damaged.pdf"
    rebuilt: list[Path] = []
    for name, src in (
        (
            "synthetic-mono.pdf",
            build_doc(
                running_head="Alekhine et al.", columns="1", paper='paper: "us-letter"', npad=150
            ),
        ),
        (
            "synthetic-duo.pdf",
            build_doc(
                running_head="Valmary et al.",
                columns="2",
                paper="width: 540pt, height: 665.972pt",
                npad=280,
            ),
        ),
    ):
        out = CORPUS_DIR / name
        if force or not out.exists():
            compile_pdf(src, out)
            rebuilt.append(out)
    if force or not damaged.exists() or duo in rebuilt:
        make_damaged_variant(duo, damaged)
    return [CORPUS_DIR / n for n in CORPUS_NAMES]


def main() -> int:
    generate_all(force=True)
    n1, n2 = (
        page_count(CORPUS_DIR / "synthetic-mono.pdf"),
        page_count(CORPUS_DIR / "synthetic-duo.pdf"),
    )
    print(f"synthetic-mono.pdf: {n1} pages (target 13)")
    print(f"synthetic-duo.pdf: {n2} pages (target 26)")
    if n1 != 13 or n2 != 26:
        print("page-count contract NOT met - tune npad and re-run", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
