#!/usr/bin/env python
"""Phase-0 acceptance: differential shadow check over the corpus (ADR-0001).

For every block of every corpus document, run the *legacy* check and the *new*
``ubt.verify`` verifier on identical inputs, and assert their three-valued
outcomes agree. A mismatch means the wrapper does not faithfully represent the
old check. Zero mismatches is the ADR's Phase-0 acceptance.

Two input sources:

- default (fast): parse each document with its adapter -- exercises the
  structural mapping on real blocks, but has no translations, so the text
  predicate and most pixel witnesses cannot run meaningfully.
- ``--run`` (faithful): translate each document dry-run into a temp ledger first,
  then check the *delivered* blocks (real targets, real reconstructions) --
  this is the acceptance run.

Exit code 0 when every compared case agrees, 1 otherwise.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from ubt.core.content.asset_verify import verify_asset_structure
from ubt.core.content.nodes import AssetKind
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.ir.models import BlockType, IRBlock
from ubt.core.policy.layout_policy import PROSE_BLOCK_TYPES
from ubt.core.qe.fast_pass import FastPassFilter
from ubt.verify import (
    RasterFormula,
    RasterTable,
    ShadowRun,
    StructuralAsset,
    TextPair,
    Verifiers,
    build_verifiers,
)

_ASSET_KIND: dict[BlockType, AssetKind] = {
    BlockType.FORMULA: AssetKind.FORMULA,
    BlockType.TABLE: AssetKind.TABLE,
}


def _parse_blocks(document: Path, engine: str) -> list[IRBlock]:
    from ubt.adapters.factory import get_adapter_for_path

    adapter = get_adapter_for_path(document, pdf_engine=engine)

    async def collect() -> list[IRBlock]:
        blocks: list[IRBlock] = []
        async for chapter in adapter.parse_stream(document):
            blocks.extend(chapter.blocks)
        return blocks

    return asyncio.run(collect())


def _run_and_read_ledger(document: Path, case_id: str) -> list[IRBlock]:
    """Dry-run translate one document and read back its delivered blocks."""
    tmp = Path(tempfile.mkdtemp(prefix=f"ubt-shadow-{case_id}-"))
    out_path = tmp / f"{case_id}_mono.pdf"
    db_dir = tmp / "ledgers"
    env = dict(os.environ)
    env["UBT_OUTPUT_DIR"] = str(tmp)
    cmd = [
        sys.executable,
        "-m",
        "ubt",
        "translate",
        str(document),
        "--fresh",
        "--dry-run",
        "--yes",
        "-o",
        str(out_path),
        "--db-dir",
        str(db_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600, check=False)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-6:]
        raise RuntimeError(f"translation failed (exit {proc.returncode}): {' / '.join(tail)}")
    ledgers = sorted(db_dir.glob("*.sqlite"))
    if not ledgers:
        raise RuntimeError(f"no ledger written under {db_dir}")
    ledger = SQLiteJobLedger(ledgers[0], read_only=True)
    try:
        return ledger.get_all_blocks(ledgers[0].stem)
    finally:
        ledger.close()


def _compare(blocks: list[IRBlock], document: Path, *, typst: str | None) -> ShadowRun:
    fast_pass = FastPassFilter(source_lang="en", target_lang="zh")
    verifiers = build_verifiers(fast_pass)
    shadow = ShadowRun()

    for block in blocks:
        _compare_structural(block, verifiers, shadow)
        if typst is not None:
            _compare_pixel(block, verifiers, shadow, document, typst)
        _compare_text(block, verifiers, shadow, fast_pass)

    return shadow


def _compare_structural(block: IRBlock, verifiers: Verifiers, shadow: ShadowRun) -> None:
    asset_kind = _ASSET_KIND.get(block.block_type)
    if asset_kind is None:
        return
    text = block.target_text or block.source_text or ""
    legacy = verify_asset_structure(asset_kind, text).verdict.value
    proof = verifiers.structural.verify(StructuralAsset(asset_kind, text))
    shadow.record(f"structural:{block.id}", legacy, proof)


def _compare_pixel(
    block: IRBlock, verifiers: Verifiers, shadow: ShadowRun, document: Path, typst: str
) -> None:
    if block.bbox is None or block.bbox.page <= 0:
        return
    emitted = block.target_text or block.source_text or ""
    if not emitted.strip():
        return
    if block.block_type is BlockType.FORMULA:
        from ubt.adapters.pdf.formula_witness import witness_formula

        legacy = witness_formula(emitted, block, document, typst).status
        proof = verifiers.formula.verify(RasterFormula(emitted, block, document, typst))
        shadow.record(f"formula:{block.id}", legacy, proof)
    elif block.block_type is BlockType.TABLE:
        from ubt.adapters.pdf.table_witness import witness_table

        legacy = witness_table(emitted, block, document, typst).status
        proof = verifiers.table.verify(RasterTable(emitted, block, document, typst))
        shadow.record(f"table:{block.id}", legacy, proof)


def _compare_text(
    block: IRBlock, verifiers: Verifiers, shadow: ShadowRun, fast_pass: FastPassFilter
) -> None:
    if block.block_type not in PROSE_BLOCK_TYPES:
        return
    source = block.source_text or ""
    target = block.target_text or ""
    if not source.strip() or not target.strip() or target == source:
        return
    passed = fast_pass.evaluate(source, target, block_type=block.block_type).passed
    legacy = "pass" if passed else "fail"
    proof = verifiers.text.verify(TextPair(source, target, block.block_type, block.skip_translate))
    shadow.record(f"text:{block.id}", legacy, proof)


def _load_cases(corpus_dir: Path) -> list[tuple[str, Path]]:
    from ubt.core.content.verify import load_corpus

    cases: list[tuple[str, Path]] = []
    for case in load_corpus(corpus_dir):
        if not case.document:
            continue
        document = Path(case.document)
        if not document.is_absolute():
            document = corpus_dir / document
        cases.append((case.id, document))
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="corpus", help="Directory containing cases.json")
    parser.add_argument("--engine", default="auto", help="PDF parser engine for parse-only mode")
    parser.add_argument(
        "--run", action="store_true", help="Dry-run translate each case first (faithful acceptance)"
    )
    parser.add_argument("--json", action="store_true", help="Machine-readable output")
    args = parser.parse_args()

    corpus_dir = Path(args.corpus)
    typst = shutil.which("typst")
    if typst is None:
        print("note: typst not on PATH; pixel witnesses will be skipped", file=sys.stderr)

    per_case: list[dict[str, object]] = []
    total = ShadowRun()
    for case_id, document in _load_cases(corpus_dir):
        if not document.exists():
            per_case.append({"id": case_id, "status": "missing", "document": str(document)})
            continue
        try:
            blocks = (
                _run_and_read_ledger(document, case_id)
                if args.run
                else _parse_blocks(document, args.engine)
            )
        except Exception as exc:  # one bad document must not sink the acceptance run
            per_case.append({"id": case_id, "status": "error", "reason": str(exc)})
            continue
        shadow = _compare(blocks, document, typst=typst)
        report = shadow.report
        total.report.total += report.total
        total.report.agreed += report.agreed
        total.report.skipped += report.skipped
        for key, value in report.by_outcome.items():
            total.report.by_outcome[key] = total.report.by_outcome.get(key, 0) + value
        total.report.mismatches.extend(report.mismatches)
        per_case.append(
            {
                "id": case_id,
                "status": "pass" if report.passed else "fail",
                "blocks": len(blocks),
                "compared": report.total,
                "agreed": report.agreed,
                "skipped": report.skipped,
                "by_outcome": report.by_outcome,
                "mismatches": [m.__dict__ for m in report.mismatches],
            }
        )

    final = total.report
    payload = {
        "status": "pass" if final.passed else "fail",
        "mode": "run" if args.run else "parse",
        "summary": final.summary(),
        "by_outcome": final.by_outcome,
        "cases": per_case,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"\nPhase-0 shadow acceptance ({payload['mode']} mode) — {corpus_dir}")
        for entry in per_case:
            print(
                f"  {entry['id']:<20} {entry['status']:<8} "
                f"compared={entry.get('compared', 0)} "
                f"agreed={entry.get('agreed', 0)} "
                f"skipped={entry.get('skipped', 0)}"
                + (f"  reason={entry['reason']}" if entry.get("reason") else "")
            )
        print(f"\n  TOTAL: {final.summary()}")
        print(f"  outcomes: {final.by_outcome}")
        for mismatch in final.mismatches[:20]:
            print(
                f"    MISMATCH {mismatch.label}: legacy={mismatch.legacy} modern={mismatch.modern}"
            )
    return 0 if final.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
