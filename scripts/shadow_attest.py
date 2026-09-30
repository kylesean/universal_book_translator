#!/usr/bin/env python
"""Phase-3 acceptance: the attestations account for the delivery, artifact included.

Runs a corpus document through the real pipeline (dry-run, mock translations, no
API key) and checks the attestation companion the export stage writes:

- ``realize()`` covers every element and records no violation;
- the delivered artifact actually carries each text element's realization -- a
  translation where the element was reconstructed, the source where it was kept
  (the ADR's "产物保真度 ≥ 现路径" acceptance);
- the delivery contract beside it is the attestation *projection*: its delivered
  and reconstructed counts are exactly the verified realizations, and it reports
  no error.

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
        problems.append(f"no attestation account at {shadow_path}")
        return None, None, problems
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    shadow = json.loads(shadow_path.read_text(encoding="utf-8"))

    if shadow["violations"]:
        problems.append(f"{len(shadow['violations'])} violation(s): {shadow['violations'][:3]}")
    if not shadow["total"]:
        problems.append("attestations cover no elements")

    artifact = shadow.get("artifact")
    if artifact is None:
        problems.append("no artifact-level check in the attestation account")
    elif artifact["missing"]:
        problems.append(
            f"artifact missing {len(artifact['missing'])} text realization(s): "
            f"{artifact['missing'][:3]}"
        )

    # The contract is projected from these attestations, so the account must be
    # exactly the verified realizations.
    realized = shadow["text"].get("RECONSTRUCTED_ADAPTED", 0)
    if realized != contract["delivered_text"]:
        problems.append(f"contract delivered {contract['delivered_text']} != realized {realized}")
    rebuilt = shadow["assets"].get("RECONSTRUCTED_VERIFIED", 0)
    if rebuilt != contract["reconstructed_assets"]:
        problems.append(
            f"contract reconstructed {contract['reconstructed_assets']} != attested {rebuilt}"
        )
    errors = [v for v in contract["violations"] if v.get("severity") == "error"]
    if errors:
        problems.append(f"contract has {len(errors)} error(s): {errors[0]}")
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

    print(f"\nPhase-3 attestation acceptance — {corpus_dir} ({len(cases)} document(s))")
    problems: list[str] = []
    for case_id, document in cases:
        contract, shadow, issues = _run_one(case_id, document)
        problems.extend(f"[{case_id}] {p}" for p in issues)
        status = "pass" if not issues else "FAIL"
        if contract is not None and shadow is not None:
            artifact = shadow.get("artifact") or {}
            print(
                f"  {case_id:<20} {status:<6} text={contract['total_text']} "
                f"delivered={contract['delivered_text']} "
                f"kept={contract['verbatim_text'] + contract['source_kept_text']} "
                f"artifact-missing={len(artifact.get('missing', []))}"
            )
        else:
            print(f"  {case_id:<20} {status:<6}")
    print(f"\n  problems={len(problems)} -> {'PASS' if not problems else 'FAIL'}")
    for line in problems[:15]:
        print(f"    {line}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
