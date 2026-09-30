#!/usr/bin/env python
"""Phase-3 acceptance: the realize() attestations account for the delivery.

Runs a corpus document through the real pipeline (dry-run, mock translations, no
API key) and compares the attestation shadow the export stage writes against the
delivery contract written beside it:

- the shadow covers every element and records no violation;
- text the contract delivered (TRANSLATED) equals the text realized
  RECONSTRUCTED_ADAPTED;
- text the contract kept verbatim or source-kept equals the text realized
  PRESERVED_OPAQUE.

Asset reconstruction is not compared yet: the backends do not reconstruct
formulas/tables, so those attest PRESERVED_OPAQUE while the existing renderer may
reconstruct them. That gap is reported, not failed. Exit 0 iff the text account
agrees.

One document by default (~1-2 min); ``--all`` runs the corpus.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from ubt.core.job_options import companion_path, sidecar_path


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


def _run_one(
    case_id: str, document: Path
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[str]]:
    """Returns (contract, attestations, problems)."""
    out_dir = Path(tempfile.mkdtemp(prefix=f"ubt-attest-{case_id}-"))
    out_path = out_dir / f"{case_id}_mono.pdf"
    env = dict(os.environ)
    env["UBT_OUTPUT_DIR"] = str(out_dir)
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
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600, check=False)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-6:]
        return None, None, [f"translation failed (exit {proc.returncode}): {' / '.join(tail)}"]

    contract_path = sidecar_path(out_path, "contract.json")
    shadow_path = companion_path(out_path, "_attestations.json")
    problems: list[str] = []
    if not contract_path.exists():
        problems.append(f"no contract at {contract_path}")
        return None, None, problems
    if not shadow_path.exists():
        problems.append(f"no attestation shadow at {shadow_path}")
        return None, None, problems
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    shadow = json.loads(shadow_path.read_text(encoding="utf-8"))

    if shadow["violations"]:
        problems.append(f"{len(shadow['violations'])} violation(s): {shadow['violations'][:3]}")
    if not shadow["total"]:
        problems.append("shadow covers no elements")

    text = shadow["text"]
    realized = text.get("RECONSTRUCTED_ADAPTED", 0)
    opaque = text.get("PRESERVED_OPAQUE", 0)
    kept = contract["verbatim_text"] + contract["source_kept_text"]
    if realized != contract["delivered_text"]:
        problems.append(f"delivered text {contract['delivered_text']} != realized {realized}")
    if opaque != kept:
        problems.append(f"kept text {kept} != opaque {opaque}")
    rebuilt = shadow["assets"].get("RECONSTRUCTED_VERIFIED", 0)
    if rebuilt != contract["reconstructed_assets"]:
        problems.append(
            f"reconstructed assets {contract['reconstructed_assets']} != attested {rebuilt}"
        )
    if contract["missing_assets"]:
        problems.append(f"contract reports {contract['missing_assets']} missing asset(s)")
    if contract["pending_text"] or contract["skipped_text"]:
        problems.append(
            f"contract left text unaccounted (pending={contract['pending_text']}, "
            f"skipped={contract['skipped_text']})"
        )
    return contract, shadow, problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="corpus")
    parser.add_argument("--case", default=None, help="One case id (default: the first)")
    parser.add_argument("--all", action="store_true", help="Every case")
    args = parser.parse_args()

    corpus_dir = Path(args.corpus)
    cases = [(c, d) for c, d in _load_cases(corpus_dir) if d.exists()]
    if args.case:
        cases = [c for c in cases if c[0] == args.case]
    elif not args.all:
        cases = cases[:1]

    print(f"\nPhase-3 attestation-shadow acceptance — {corpus_dir} ({len(cases)} document(s))")
    problems: list[str] = []
    for case_id, document in cases:
        contract, shadow, issues = _run_one(case_id, document)
        problems.extend(f"[{case_id}] {p}" for p in issues)
        status = "pass" if not issues else "FAIL"
        if contract is not None and shadow is not None:
            asset_gap = contract["reconstructed_assets"] - shadow["assets"].get(
                "RECONSTRUCTED_VERIFIED", 0
            )
            print(
                f"  {case_id:<20} {status:<6} text={contract['total_text']} "
                f"delivered={contract['delivered_text']} kept={contract['verbatim_text'] + contract['source_kept_text']} "
                f"asset-reconstruction-gap={asset_gap}"
            )
        else:
            print(f"  {case_id:<20} {status:<6}")
    print(f"\n  problems={len(problems)} -> {'PASS' if not problems else 'FAIL'}")
    for line in problems[:15]:
        print(f"    {line}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
