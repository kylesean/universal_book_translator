#!/usr/bin/env python3
"""Batch baseline over the semantic-path gold corpus (RFC-2026-02 Sprint 1).

Runs ``scripts/eval_semantic.py`` over every case in
``corpus/semantic_gold/manifest.json`` that has a delivered artifact, and writes
a single JSON + a readable table.

The corpus ships *sources*, not artifacts: a full run needs the translation
pipeline (LLM calls). Point ``--artifacts`` at a directory of delivered PDFs
named ``<case-id>.pdf`` (or ``<case-id>_mono.pdf``); cases without an artifact
are reported ``missing`` rather than silently skipped, so a partial baseline is
visibly partial.

Usage:
    uv run python scripts/eval_gold_corpus.py --artifacts /path/to/outputs
    uv run python scripts/eval_gold_corpus.py --artifacts ~/Documents/UBT --json out.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_semantic import evaluate  # noqa: E402

MANIFEST = ROOT / "corpus" / "semantic_gold" / "manifest.json"
SOURCES = ROOT / "corpus" / "semantic_gold" / "documents"

#: Filenames an artifact may take inside ``--artifacts``, in preference order.
ARTIFACT_PATTERNS = ("{id}_mono.pdf", "{id}.pdf", "{id}_mono_v3.pdf")


def _find_artifact(directory: Path, case_id: str) -> Path | None:
    for pattern in ARTIFACT_PATTERNS:
        candidate = directory / pattern.format(id=case_id)
        if candidate.exists():
            return candidate
    return None


def _source_table_headers(case_id: str, markdown_dir: Path | None) -> list[list[str]]:
    """Header rows of the source document, from its Docling Markdown.

    The leak metric needs the source headers. Re-deriving them means running
    Docling (GPU, minutes); the semantic path already produces the Markdown, so
    we read that instead. Looked up as ``<markdown_dir>/<case-id>.md``. When the
    Markdown is absent the metric reports 0.0 and ``source_tables_available``
    is false, so an unmeasured value cannot be mistaken for a good one.
    """
    if markdown_dir is None:
        return []
    for candidate in (markdown_dir / f"{case_id}.md", markdown_dir / f"{case_id}_docling.md"):
        if candidate.exists():
            return _headers_from_markdown(candidate.read_text(encoding="utf-8", errors="replace"))
    return []


def _headers_from_markdown(md: str) -> list[list[str]]:
    headers: list[list[str]] = []
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("|"):
            block: list[str] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            if len(block) >= 3:
                headers.append([c.strip() for c in block[0].strip().strip("|").split("|")])
            continue
        i += 1
    return headers


def _find_ledger(ledger_dir: Path | None, case_id: str) -> Path | None:
    """The job ledger whose source document is this case.

    Matched on ``job_meta.source_path`` rather than by recency: a directory of
    ledgers holds every job ever run, and picking the newest would attribute one
    document's block statistics to another. A case whose ledger is absent yields
    ``None`` and the translatable-ratio metric is simply not reported, so an
    unmeasured value cannot be read as a good one.
    """
    if ledger_dir is None or not ledger_dir.exists():
        return None
    wanted = {f"{case_id}.pdf", f"{case_id}.PDF"}
    best: tuple[float, Path] | None = None
    for db in ledger_dir.glob("*.sqlite"):
        try:
            conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            try:
                row = next(conn.execute("SELECT source_path FROM job_meta LIMIT 1"), None)
            finally:
                conn.close()
        except sqlite3.Error:
            continue
        if row is None or not row[0]:
            continue
        if Path(str(row[0])).name not in wanted:
            continue
        stamp = db.stat().st_mtime
        if best is None or stamp > best[0]:
            best = (stamp, db)
    return best[1] if best else None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--artifacts", type=Path, required=True, help="directory holding delivered PDFs"
    )
    ap.add_argument(
        "--source-markdown",
        type=Path,
        default=None,
        help="directory holding <case-id>.md (Docling output) for source-header scoring",
    )
    ap.add_argument(
        "--ledgers",
        type=Path,
        default=ROOT / ".ubt" / "ledgers",
        help="directory holding job ledgers (for the translatable-content ratio)",
    )
    ap.add_argument(
        "--typst",
        type=Path,
        default=None,
        help="directory holding .<case>_typst/main.typ for math-residue scoring",
    )
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--json", type=Path, default=None, help="write the full report here")
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for case in manifest["cases"]:
        case_id = case["id"]
        artifact = _find_artifact(args.artifacts, case_id)
        if artifact is None:
            rows.append({"id": case_id, "status": "missing", "category": case["category"]})
            continue
        headers = _source_table_headers(case_id, args.source_markdown)
        ledger = _find_ledger(args.ledgers, case_id)
        typst = None
        if args.typst is not None:
            candidate = args.typst / f".{case_id}_typst" / "main.typ"
            typst = candidate if candidate.exists() else None
        report = evaluate(artifact, ledger, typst, headers or None)
        report["source_tables_available"] = bool(headers)
        report["ledger_found"] = ledger is not None
        rows.append({"id": case_id, "status": "measured", "category": case["category"], **report})

    measured = [r for r in rows if r["status"] == "measured"]
    missing = [r for r in rows if r["status"] == "missing"]

    if measured:
        keys = [
            "cjk_ratio",
            "cjk_ratio_translatable",
            "untranslated_marks",
            "p95_font_size_pt",
            "distinct_font_sizes",
            "source_header_term_leakage",
        ]
        print(f"{'case':<28} " + " ".join(f"{k[:11]:>11}" for k in keys))
        print("-" * (29 + 12 * len(keys)))
        for row in measured:
            ledger = row.get("ledger") or {}
            flat = dict(row)
            # The translatable ratio lives under the ledger subtree; lift it so
            # the table reads both numbers side by side, which is the point of
            # reporting it separately.
            flat["cjk_ratio_translatable"] = ledger.get("cjk_ratio_translatable") or 0.0
            print(f"{row['id']:<28} " + " ".join(f"{flat.get(k, 0):>11}" for k in keys))

    if missing:
        print(f"\nmissing ({len(missing)}): " + ", ".join(r["id"] for r in missing))
    print(f"\n{len(measured)} measured, {len(missing)} missing, {len(rows)} total")

    payload = {
        "artifacts_dir": str(args.artifacts),
        "measured": len(measured),
        "missing": len(missing),
        "rows": rows,
    }
    if args.json:
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
