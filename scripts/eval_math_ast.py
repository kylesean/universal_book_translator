#!/usr/bin/env python3
"""Formula AST agreement: does the transpiler preserve the formula's structure?

RFC §9's fifth target is "公式 AST 一致率 ≥ 99%" and it was **unmeasured** — the
corpus only carried ``math_residue_ratio``, which asks a different question ("did
any LaTeX survive"). A formula can be residue-free and still wrong: a dropped
``\\frac``, a superscript turned into a subscript, a ``\\sum`` rendered as plain
text all pass the residue check.

This script measures the RFC's question directly, without a Typst parser (none
exists). Both sides are reduced to the same **structural signature** — the count
of each construct a reader perceives:

============  ============================  ==============================
construct     LaTeX side (MathML AST)       Typst side (emitted source)
============  ============================  ==============================
fraction      ``mfrac``                      ``frac(``
root          ``msqrt``/``mroot``            ``sqrt(``/``root(``
superscript   ``msup``/``msubsup``/``mover`` ``^``
subscript     ``msub``/``msubsup``/``munder`` ``_``
big operator  ``mo`` holding ∑/∏/∫/∮        ``sum``/``product``/``integral``
accent        ``mover``/``munder``           ``hat(``/``tilde(``/…
============  ============================  ==============================

A formula agrees when every count matches. The signature is deliberately coarse:
it cannot tell ``x^2`` from ``y^2`` (a glyph difference a reader sees but a
structure check should not fail on) and it cannot see order — what it does catch
is exactly the defect class this target exists for, a construct lost or turned
into a different construct.

If the LaTeX side cannot be parsed (an unsupported command) the formula is
reported ``unparsed`` rather than agreeing, so an unmeasurable formula never
counts as a pass.

Usage:
    uv run python scripts/eval_math_ast.py [--ledgers .ubt/ledgers] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ubt.render.typst.math_transpiler import (  # noqa: E402
    strip_math_delimiters,
    typstify_math,
)

_MATHML_NS = "{http://www.w3.org/1998/Math/MathML}"

#: The constructs the signature is built from, in report order.
_CONSTRUCTS = ("fraction", "binom", "root", "superscript", "subscript", "bigop", "accent")

#: MathML ``mo`` content that counts as a big operator.
_BIG_OPERATORS = frozenset({"∑", "∏", "∫", "∮", "∬", "∭", "⋃", "⋂"})

#: Typst function names that are big operators.
_TYPST_BIGOPS = (
    "sum",
    "product",
    "integral",
    "integral.double",
    "integral.triple",
    "integral.cont",
    "union",
    "sect",
)

#: Typst function names that are accents (a stretchy glyph over/under a base).
_TYPST_ACCENTS = ("hat", "tilde", "vec", "bar", "dot", "ddot", "overline", "underline", "accent")


#: Unicode primes the MathML encoder wraps in a script node.
_PRIME_MARKS = frozenset({"\u2032", "\u2033", "\u2034", "'", "\u2019"})


def _is_prime_script(element: ET.Element) -> bool:
    """Whether an ``msup``/``msub``/``msubsup`` script is only a prime mark.

    ``x'`` and ``x^\\prime`` both encode as a script over a prime character,
    while in both source and target the prime is written literally; only a real
    exponent is a superscript a reader perceives as one. MathML puts the base
    first, then the subscript, then the superscript, so the script to inspect is
    the **last** child for a superscript and the second for a plain subscript —
    examining every child would read the base and never match.
    """
    if len(element) < 2:
        return False
    return (element[-1].text or "").strip() in _PRIME_MARKS


def latex_signature(latex: str) -> dict[str, int] | None:
    """Structural signature of a LaTeX formula, or ``None`` when unparseable.

    The LaTeX is converted to MathML (a real AST) and the constructs are counted
    from it, so ``x^{2}`` and ``x^2`` produce the same signature while ``x^2`` and
    ``x_2`` do not.
    """
    try:
        import latex2mathml.converter as converter

        root = ET.fromstring(converter.convert(latex))
    except Exception:
        return None

    counts = dict.fromkeys(_CONSTRUCTS, 0)
    # A prime is a script only to the MathML encoder: ``x'`` becomes an ``msup``
    # over ``′``, while Typst (and LaTeX) render the same prime as a literal
    # ``'``. Counting it as a superscript would report a disagreement where the
    # two spellings agree, so primed ``msup``/``msub`` nodes are skipped.
    for element in root.iter():
        tag = element.tag.replace(_MATHML_NS, "")
        if tag == "mfrac":
            # MathML encodes both ``\frac`` and ``\binom`` as ``mfrac``; a
            # binomial carries ``linethickness="0"``. They are separate
            # constructs here because the transpiler emits ``frac(...)`` and
            # ``binom(...)``, so counting both as fractions would report a
            # disagreement on every binomial.
            if element.get("linethickness") == "0":
                counts["binom"] += 1
            else:
                counts["fraction"] += 1
        elif tag in ("msqrt", "mroot"):
            counts["root"] += 1
        elif tag in ("msup", "msub", "msubsup"):
            if _is_prime_script(element):
                # A prime in the superscript slot: ``x'_t`` is a subscripted
                # ``x'``, not a real exponent, and both spellings write the
                # prime literally.
                if tag == "msubsup" or tag == "msub":
                    counts["subscript"] += 1
                continue
            if tag in ("msup", "msubsup"):
                counts["superscript"] += 1
            if tag in ("msub", "msubsup"):
                counts["subscript"] += 1
        elif tag in ("mover", "munder", "munderover"):
            # A script over/under a base: an accent, or a big operator's limits.
            # Classified by the base so ``\hat{x}`` and ``\sum_{i}`` differ.
            base = next(iter(element), None)
            text = "".join(base.itertext()).strip() if base is not None else ""
            if text and text[0] in _BIG_OPERATORS:
                counts["bigop"] += 1
                counts["subscript"] += 1
            else:
                counts["accent"] += 1
        elif tag == "mo" and (element.text or "").strip() in _BIG_OPERATORS:
            counts["bigop"] += 1
    return counts


def typst_signature(typst: str) -> dict[str, int]:
    """Structural signature of transpiled Typst math source.

    A bare ``_`` is a subscript in Typst, whether or not a letter precedes it
    (``I_1`` is a subscripted ``I``, and so is ``f_1``), so no look-behind
    applies. The only false positive to exclude is an underscore *inside* a
    quoted label (``#"a_b"``), which is text rather than a script operator — so
    quoted strings are removed from the scan first.
    """
    scanned = re.sub(r'#"[^"]*"', " ", typst)
    # An escaped underscore is a literal character, not a script operator:
    # ``#"batch"\_#"size"`` renders the label ``batch_size``. Drop the escape
    # pair before counting.
    scanned = scanned.replace("\\_", " ").replace("\\^", " ")
    counts = dict.fromkeys(_CONSTRUCTS, 0)
    counts["fraction"] = len(re.findall(r"\bfrac\(", scanned))
    counts["binom"] = len(re.findall(r"\bbinom\(", scanned))
    counts["root"] = len(re.findall(r"\b(?:sqrt|root)\(", scanned))
    # ``_`` and ``^`` are not word characters for ``\b`` here, and ``sum_(``
    # must match, so the boundary is spelled out: no letter may abut the name.
    counts["bigop"] = sum(
        len(re.findall(rf"(?<![A-Za-z]){re.escape(op)}(?![A-Za-z])", scanned))
        for op in _TYPST_BIGOPS
    )
    counts["accent"] = sum(
        len(re.findall(rf"\b{re.escape(name)}\(", scanned)) for name in _TYPST_ACCENTS
    )
    counts["superscript"] = len(re.findall(r"\^", scanned))
    counts["subscript"] = len(re.findall(r"_", scanned))
    return counts


def compare(latex: str) -> dict[str, Any]:
    """One formula's verdict: ``agree`` / ``disagree`` / ``unparsed`` / ``undelivered``."""
    stripped = strip_math_delimiters(latex)
    converted = typstify_math(stripped)
    if converted is None:
        # The transpiler refused it, so the caller degrades it to Unicode text;
        # the formula did not reach the page as math at all.
        return {"latex": latex[:160], "verdict": "undelivered", "typst": None}
    expected = latex_signature(stripped)
    if expected is None:
        return {"latex": latex[:160], "verdict": "unparsed", "typst": converted[:160]}
    actual = typst_signature(converted)
    verdict = "agree" if expected == actual else "disagree"
    return {
        "latex": latex[:160],
        "typst": converted[:160],
        "verdict": verdict,
        "expected": expected,
        "actual": actual,
    }


def _formulas_from_ledger(ledger: Path) -> list[str]:
    """Every LaTeX formula the transpiler was asked to convert, per job ledger.

    The LaTeX lives in the *target* text, not the source: the PDF adapter
    extracts Unicode math (``𝐷``), the model re-encodes it as LaTeX
    (``$\\frac{D}{N}$``), and the transpiler converts that for Typst. So the
    formula set this measures is exactly the set the renderer had to handle.

    Both carriers are read: a ``formula`` block's whole ``target_text``, and the
    ``$…$`` spans inside prose blocks (where most inline math lives).
    """
    conn = sqlite3.connect(f"file:{ledger}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = list(conn.execute("SELECT block_type, target_text FROM blocks"))
    finally:
        conn.close()

    found: list[str] = []
    for row in rows:
        text = row["target_text"] or ""
        if str(row["block_type"]) == "formula":
            found.append(text.strip())
        found.extend(re.findall(r"\$\$[\s\S]*?\$\$|\$[^$\n]+\$", text))
    return [f for f in found if f.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ledgers", type=Path, default=ROOT / ".ubt" / "ledgers")
    ap.add_argument("--json", type=Path, default=None, help="write the full report here")
    ap.add_argument("--show", type=int, default=8, help="how many disagreeing formulas to print")
    args = ap.parse_args()

    ledgers = sorted(args.ledgers.glob("job_*.sqlite"))
    if not ledgers:
        print(f"no job ledgers under {args.ledgers}", file=sys.stderr)
        return 1

    totals = dict.fromkeys(("agree", "disagree", "unparsed", "undelivered"), 0)
    seen_docs: set[str] = set()
    per_doc: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []

    for ledger in ledgers:
        conn = sqlite3.connect(f"file:{ledger}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            meta = next(conn.execute("SELECT source_path FROM job_meta LIMIT 1"), None)
        finally:
            conn.close()
        source = str(meta["source_path"] or "") if meta else ""
        if "semantic_gold" not in source or Path(source).stem in seen_docs:
            continue
        name = Path(source).stem
        seen_docs.add(name)

        doc_counts = dict.fromkeys(("agree", "disagree", "unparsed", "undelivered"), 0)
        for latex in _formulas_from_ledger(ledger):
            result = compare(latex)
            doc_counts[result["verdict"]] += 1
            totals[result["verdict"]] += 1
            if result["verdict"] in ("disagree", "unparsed") and len(examples) < args.show:
                examples.append(result)
        per_doc.append({"id": name, **doc_counts})

    considered = totals["agree"] + totals["disagree"]
    rate = (totals["agree"] / considered) if considered else 0.0

    print(f"{'case':<28} {'agree':>7} {'disagree':>9} {'unparsed':>9} {'undelivered':>12}")
    print("-" * 70)
    for row in sorted(per_doc, key=lambda r: r["disagree"] + r["unparsed"], reverse=True):
        print(
            f"{row['id']:<28} {row['agree']:>7} {row['disagree']:>9} "
            f"{row['unparsed']:>9} {row['undelivered']:>12}"
        )
    print("-" * 70)
    print(
        f"{'TOTAL':<28} {totals['agree']:>7} {totals['disagree']:>9} "
        f"{totals['unparsed']:>9} {totals['undelivered']:>12}"
    )
    print(
        f"\nformula AST agreement: {rate:.4f} over {considered} comparable formulas "
        f"(target >= 0.99)"
    )
    print(f"  {totals['unparsed']} unparseable by the LaTeX parser (not counted as agreement)")
    print(f"  {totals['undelivered']} the transpiler refused (degraded to Unicode text)")

    if examples:
        print("\ndisagreeing / unparsed examples:")
        for ex in examples:
            print(f"  [{ex['verdict']}] {ex['latex']!r}")
            if ex.get("typst"):
                print(f"      typst: {ex['typst']!r}")
            if ex.get("expected") is not None:
                print(f"      expected={ex['expected']} actual={ex['actual']}")

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "agreement_rate": round(rate, 4),
                    "totals": totals,
                    "per_document": per_doc,
                    "examples": examples,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
