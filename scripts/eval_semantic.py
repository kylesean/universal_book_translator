#!/usr/bin/env python3
"""Semantic-path baseline metrics for the reflow (RFC-2026-02) programme.

Measures the *product* of a translation run on the dimensions the semantic
rearchitecture is meant to move, so a refactor can be judged against numbers
instead of impressions. Every metric is computable from the delivered PDF plus
the run's ledger; nothing here needs a paid judge or a human.

Metrics
-------
cjk_ratio            CJK chars / (CJK + Latin letters) in the delivered text.
                     The old overlay path scored ~0.31 on the arXiv reference
                     paper (source language survives on most of the page).
                     **Whole-document**, so the blocks the policy deliberately
                     keeps in the source (bibliography, affiliations, code)
                     count against it — read it beside
                     ``ledger.cjk_ratio_translatable``.
untranslated_marks   Occurrences of the quarantine badge / placeholder. The
                     overlay path shipped 17 on the reference paper.
p95_font_size_pt     P95 of character-weighted line font sizes. The overlay
                     path produced 4pt..12pt jitter from box-fitting; a reflow
                     should sit near one body size.
table_header_cjk     Fraction of table header cells carrying CJK. The overlay
                     path scored 0.0 (tables keep the source layer).
formula_failures     Math blocks whose Typst source still contains LaTeX
                     residue (``\\cmd``) or an un-transpiled word run.
ledger_status        Block status histogram from the job ledger.
blocked_ratio        (blocked_human + failed) / total blocks.
cjk_ratio_translatable
                     The same CJK ratio over **only** the blocks the pipeline
                     was supposed to translate. This is the number that says
                     whether translation happened; ``cjk_ratio`` also carries
                     the deliberate source keeps. Measured on the gold corpus
                     the two diverge sharply (e.g. 0.475 vs 0.792 on one
                     gpt3-sized job), so reporting only the whole-document
                     figure understated the result by ~0.3.

Usage:
    uv run python scripts/eval_semantic.py delivered.pdf [--ledger PATH] [--json]
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any

CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
LATIN_RE = re.compile(r"[A-Za-z]")
#: LaTeX commands that must never survive into a compiled Typst math block.
LATEX_RESIDUE_RE = re.compile(r"\\[A-Za-z]+")
#: Quarantine badges / placeholders that mean "shipped untranslated".
UNTRANSLATED_MARKERS = ("未译", "待人工审校", "untranslated", "Human review required")


def _pdftotext(path: Path, *, layout: bool = False) -> str | None:
    """pdftotext output, or ``None`` when the binary is unavailable."""
    import subprocess

    cmd = ["pdftotext", "-q"] + (["-layout"] if layout else []) + [str(path), "-"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 and out.stdout.strip() else None


def _pdf_text(path: Path) -> str:
    """Extract the delivered text layer (pdftotext, then pdfium fallback)."""
    text = _pdftotext(path)
    if text is not None:
        return text
    # pdfium fallback: no external binary dependency.
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(path))
    try:
        return "\n".join(page.get_textpage().get_text_range() for page in doc)
    finally:
        doc.close()


def _font_sizes(path: Path) -> list[tuple[float, int]]:
    """Character-weighted (font_size, count) pairs across every page.

    Reuses the adapter's own line extractor so the number matches what the
    engine believes a line's size is, rather than a second opinion from a
    different parser.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ubt.adapters.pdf.textgeom import extract_lines

    pairs: list[tuple[float, int]] = []
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(path))
    try:
        n_pages = len(doc)
    finally:
        doc.close()
    for page_no in range(1, n_pages + 1):
        try:
            lines, _ = extract_lines(path, page_no)
        except Exception:
            continue
        for ln in lines:
            text = (ln.text or "").strip()
            if not text or ln.font_size < 4.5:
                continue
            pairs.append((round(ln.font_size, 2), len(text)))
    return pairs


def _weighted_p95(pairs: list[tuple[float, int]]) -> float:
    if not pairs:
        return 0.0
    ordered = sorted(pairs)
    total = sum(n for _s, n in ordered)
    target = total * 0.95
    acc = 0
    for size, n in ordered:
        acc += n
        if acc >= target:
            return size
    return ordered[-1][0]


#: Tokens that are identifiers, not translatable prose. A table header is
#: "translated" when the *translatable* words in it stop appearing verbatim;
#: model names and metric abbreviations are expected to survive either way, so
#: counting them would dilute the signal to noise.
HEADER_IDENTIFIER_STOPWORDS = frozenset(
    {
        "glue",
        "squad",
        "xsum",
        "anli",
        "arc",
        "easy",
        "chal",
        "winogrande",
        "xl",
        "r3",
        "cnndm",
        "superglue",
        "base",
        "large",
        "model",
        "t5",
        "moe",
        "switch",
        "float32",
        "bfloat16",
        "flops",
        "geglu",
        "relu",
        "sst",
        "mnli",
        "qqp",
        "qnli",
        "rte",
        "mrpc",
        "cola",
        "sts",
    }
)


def _source_header_term_leakage(body_text: str, source_tables: list[list[str]]) -> float:
    """Fraction of translatable source-table-header words still verbatim.

    The one unambiguous "did the table get translated" signal. The overlay path
    keeps the source table layer, so its translatable header words survive at
    ~1.0 on a page whose surrounding prose is Chinese — the defect a reader
    sees. The reflow path rebuilds the grid, so they drop sharply.

    Caveat, stated because the number is otherwise easy to over-read: a word
    like "capacity" also occurs in translated *prose* ("capacity factor" is
    deliberately retained as terminology), so this is a ceiling, not an exact
    table-cell count. Only the **delta between render modes** is meaningful; a
    single run's absolute value is not.
    """
    if not source_tables:
        return 0.0
    words: set[str] = set()
    for header in source_tables:
        for cell in header:
            for word in re.findall(r"[A-Za-z]{4,}", cell):
                if word.lower() not in HEADER_IDENTIFIER_STOPWORDS:
                    words.add(word.lower())
    if not words:
        return 0.0
    lowered = body_text.lower()
    leaked = sum(1 for w in words if w in lowered)
    return leaked / len(words)


def _table_header_cjk(typst: Path | None) -> float:
    """Fraction of *translatable* table-header cells carrying CJK, from Typst source.

    Read from the generated Typst rather than the compiled PDF: a Typst table
    is a ``#table(...)`` call whose cells are literal markup, so the header row
    is directly addressable. In the PDF the same cells come back as loose text
    lines with no column structure left, and any heuristic that tries to
    recover the grid also sweeps up wrapped prose (measured: it scored the
    *untranslated* overlay artifact at 0.61, which is how a bad metric hides a
    real regression).

    Only cells holding at least one non-identifier Latin word are counted.
    ``CB Natural QA``, ``FLOPs/seq`` and ``FFN GEGLU`` are dataset and
    architecture names that must stay verbatim; counting them as "untranslated"
    would make the number look worse the more correct the translation is.
    """
    if typst is None or not typst.exists():
        return 0.0
    source = typst.read_text(encoding="utf-8", errors="replace")
    total = 0
    hit = 0
    for table in re.findall(r"#table\((.*?)\n\)", source, re.S):
        header = re.search(
            r"stroke: 1\.1pt\),\n\s*(.*?),(?:\n\s*)table\.hline\(stroke: 0\.5pt\)",
            table,
            re.S,
        )
        if not header:
            continue
        for raw in re.findall(r"\[(.*?)\]", header.group(1)):
            cell = re.sub(r"[*\\]", "", raw).strip()
            if not cell:
                continue
            words = [
                w
                for w in re.findall(r"[A-Za-z]{3,}", cell)
                if w.lower() not in HEADER_IDENTIFIER_STOPWORDS
            ]
            if not words:
                continue
            total += 1
            if CJK_RE.search(cell):
                hit += 1
    return (hit / total) if total else 0.0


def _table_header_cells(typst: Path | None) -> list[str]:
    """Translatable header cells that stayed in the source language.

    Reported alongside the ratio so the number is auditable: on the reference
    paper five of six "untranslated" cells are dataset/architecture names
    (``CB Natural QA``, ``FLOPs/seq``, ``FFN GEGLU``) that *should* stay
    verbatim, so the raw ratio understates the translation. Listing them is how
    a reader tells a real miss from a metric artefact.
    """
    if typst is None or not typst.exists():
        return []
    source = typst.read_text(encoding="utf-8", errors="replace")
    misses: list[str] = []
    for table in re.findall(r"#table\((.*?)\n\)", source, re.S):
        header = re.search(
            r"stroke: 1\.1pt\),\n\s*(.*?),(?:\n\s*)table\.hline\(stroke: 0\.5pt\)",
            table,
            re.S,
        )
        if not header:
            continue
        for raw in re.findall(r"\[(.*?)\]", header.group(1)):
            cell = re.sub(r"[*\\]", "", raw).strip()
            if not cell or CJK_RE.search(cell):
                continue
            words = [
                w
                for w in re.findall(r"[A-Za-z]{3,}", cell)
                if w.lower() not in HEADER_IDENTIFIER_STOPWORDS
            ]
            if words:
                misses.append(cell)
    return misses


def _typst_math_residue(typ_path: Path) -> tuple[int, int]:
    """(total math blocks, blocks with LaTeX/word-run residue) in a Typst source."""
    if not typ_path.exists():
        return 0, 0
    source = typ_path.read_text(encoding="utf-8", errors="replace")
    blocks = re.findall(r"\$([^$]+)\$", source)
    bad = 0
    for body in blocks:
        if LATEX_RESIDUE_RE.search(body):
            bad += 1
            continue
        # An un-transpiled word run survives as a quoted identifier that was
        # really a LaTeX command (``"mathcal"``), or a stray paren pair.
        if re.search(r'"(mathbb|mathcal|mathbf|mathrm|text|frac|sqrt)"', body):
            bad += 1
    return len(blocks), bad


def _ledger_stats(ledger_path: Path) -> dict[str, Any]:
    conn = sqlite3.connect(str(ledger_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = list(
            conn.execute(
                "SELECT status, error_flags_json, target_text, draft_text, source_text,"
                " skip_translate, policy_translate FROM blocks"
            )
        )
    except sqlite3.OperationalError as exc:
        return {"error": str(exc)}
    finally:
        conn.close()

    statuses = Counter(r["status"] for r in rows)
    flag_counter: Counter[str] = Counter()
    for row in rows:
        try:
            flags = json.loads(row["error_flags_json"] or "[]")
        except (TypeError, ValueError):
            continue
        for flag in flags:
            flag_counter[flag.split(":")[0]] += 1

    total = len(rows)
    quarantined = statuses.get("blocked_human", 0) + statuses.get("failed", 0)

    # The translatable-content ratio: the same CJK/(CJK+Latin) measure, but over
    # only the blocks the pipeline was supposed to translate. A bibliography
    # entry, an author affiliation or a code listing is *deliberately* kept in
    # the source language, and those blocks drag the whole-document ratio down
    # without saying anything about translation quality. Measured on the gold
    # corpus the gap is large: one gpt3-sized job reads 0.475 whole-document and
    # 0.792 translatable-only. Both are reported, so the deliberate keeps are
    # visible rather than merely depressing one number.
    kept = [r for r in rows if _keeps_source_row(r)]
    translatable = [r for r in rows if not _keeps_source_row(r)]

    def _ratio(selected: list[sqlite3.Row]) -> float | None:
        body = "".join(
            (r["target_text"] or r["draft_text"] or r["source_text"] or "") for r in selected
        )
        cjk = len(CJK_RE.findall(body))
        latin = len(LATIN_RE.findall(body))
        return round(cjk / (cjk + latin), 4) if (cjk + latin) else None

    return {
        "total_blocks": total,
        "source_kept_blocks": len(kept),
        "translatable_blocks": len(translatable),
        "statuses": dict(statuses.most_common()),
        "blocked_ratio": (quarantined / total) if total else 0.0,
        "cjk_ratio_translatable": _ratio(translatable),
        "top_flags": dict(flag_counter.most_common(10)),
    }


def _keeps_source_row(row: sqlite3.Row) -> bool:
    """Whether the pipeline decided this block stays in the source language.

    Mirrors ``ubt.render.typst.document_renderer._keeps_source``: the policy
    verdict is authoritative when present, and ``skip_translate`` covers the
    extractor's default. Keeping the two in step is what makes
    ``cjk_ratio_translatable`` measure the same population the renderer treated
    as translated.
    """
    try:
        policy = row["policy_translate"]
    except (IndexError, KeyError):
        policy = None
    if policy is not None and not policy:
        return True
    return bool(row["skip_translate"])


def evaluate(
    pdf: Path,
    ledger: Path | None,
    typst: Path | None,
    source_tables: list[list[str]] | None = None,
) -> dict[str, Any]:
    text = _pdf_text(pdf)
    cjk = len(CJK_RE.findall(text))
    latin = len(LATIN_RE.findall(text))
    sizes = _font_sizes(pdf)
    total_math, bad_math = _typst_math_residue(typst) if typst else (0, 0)
    table_cjk = _table_header_cjk(typst)
    header_leakage = _source_header_term_leakage(text, source_tables or [])

    report: dict[str, Any] = {
        "artifact": str(pdf),
        "cjk_chars": cjk,
        "latin_chars": latin,
        "cjk_ratio": round(cjk / (cjk + latin), 4) if (cjk + latin) else 0.0,
        "untranslated_marks": sum(text.count(m) for m in UNTRANSLATED_MARKERS),
        "p95_font_size_pt": _weighted_p95(sizes),
        "distinct_font_sizes": len({s for s, _n in sizes}),
        "source_header_term_leakage": round(header_leakage, 4),
        "translated_table_header_cjk": round(table_cjk, 4),
        "untranslated_header_cells": _table_header_cells(typst),
        "math_blocks": total_math,
        "math_residue_blocks": bad_math,
        "math_residue_ratio": round(bad_math / total_math, 4) if total_math else 0.0,
    }
    if ledger and ledger.exists():
        report["ledger"] = _ledger_stats(ledger)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("pdf", type=Path, help="delivered PDF to measure")
    ap.add_argument("--ledger", type=Path, default=None, help="job ledger SQLite for block stats")
    ap.add_argument(
        "--typst", type=Path, default=None, help="generated Typst source (math residue)"
    )
    ap.add_argument(
        "--source-tables",
        type=Path,
        default=None,
        help="JSON list of source table headers (list[list[str]]) for survival scoring",
    )
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = ap.parse_args()

    if not args.pdf.exists():
        print(f"error: {args.pdf} not found", file=sys.stderr)
        return 2

    source_tables = None
    if args.source_tables and args.source_tables.exists():
        source_tables = json.loads(args.source_tables.read_text(encoding="utf-8"))

    report = evaluate(args.pdf, args.ledger, args.typst, source_tables)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    for key, value in report.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k2, v2 in value.items():
                print(f"    {k2}: {v2}")
        else:
            print(f"{key:26} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
