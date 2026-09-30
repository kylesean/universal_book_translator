#!/usr/bin/env python
"""Phase-2 acceptance: the export leaves a bilingual XLIFF beside the delivery.

Runs a corpus document through the real pipeline (dry-run, mock translations, no
API key) exactly as the corpus gate does, then inspects the companion the export
stage leaves beside the artifact:

- it exists at the name the deliverable implies and parses as XLIFF 2.1;
- every unit carries a source, and at least one carries the delivered target;
- protected spans survive as inline codes and the whole view round-trips through
  ``from_xliff`` / ``to_xliff`` without losing a field.

One document by default (~1-2 min); ``--all`` runs the whole corpus. Exit 0 iff
every check holds.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from ubt.core.job_options import companion_path
from ubt.model.segment import Segment
from ubt.segment.xliff import from_xliff, to_xliff


def _fingerprint(segment: Segment) -> tuple[object, ...]:
    return (
        segment.id,
        segment.source,
        segment.target,
        segment.state.value,
        tuple(sorted((p.token, p.kind, p.original) for p in segment.placeholders)),
    )


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


def _run_one(case_id: str, document: Path) -> tuple[int, int, int, list[str]]:
    """Returns (segments, targeted, placeholders, problems) for one document."""
    out_dir = Path(tempfile.mkdtemp(prefix=f"ubt-companion-{case_id}-"))
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
        return 0, 0, 0, [f"translation failed (exit {proc.returncode}): {' / '.join(tail)}"]

    companion = companion_path(out_path, ".xliff")
    if not companion.exists():
        return 0, 0, 0, [f"no companion at {companion}"]

    parsed = from_xliff(companion.read_text(encoding="utf-8"))
    segments = parsed.segments
    problems = [f"unit {s.id}: empty source" for s in segments if not s.source.strip()]
    targeted = sum(1 for s in segments if s.target)
    placeholders = sum(len(s.placeholders) for s in segments)
    if not segments:
        problems.append("companion carries no segments")
    elif targeted == 0:
        problems.append("companion carries no target (not bilingual)")

    again = to_xliff(
        segments, src_lang=parsed.src_lang, trg_lang=parsed.trg_lang, original=parsed.original
    )
    after = from_xliff(again)
    if len(after.segments) != len(segments):
        problems.append(f"round-trip count {len(after.segments)} != {len(segments)}")
    else:
        problems.extend(
            f"unit {a.id}: field mismatch on round-trip"
            for a, b in zip(segments, after.segments, strict=True)
            if _fingerprint(a) != _fingerprint(b)
        )
    return len(segments), targeted, placeholders, problems


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
    if not cases:
        print("no corpus cases selected")
        return 1

    print(f"\nPhase-2 XLIFF companion acceptance — {corpus_dir} ({len(cases)} document(s))")
    all_problems: list[str] = []
    for case_id, document in cases:
        segments, targeted, placeholders, problems = _run_one(case_id, document)
        all_problems.extend(f"[{case_id}] {p}" for p in problems)
        status = "pass" if not problems else "FAIL"
        print(
            f"  {case_id:<20} {status:<6} segments={segments} "
            f"targeted={targeted} placeholders={placeholders}"
        )
    print(f"\n  problems={len(all_problems)} -> {'PASS' if not all_problems else 'FAIL'}")
    for line in all_problems[:15]:
        print(f"    {line}")
    return 0 if not all_problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
