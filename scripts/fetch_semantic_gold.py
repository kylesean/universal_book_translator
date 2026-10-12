#!/usr/bin/env python3
"""Fetch the semantic-path gold corpus from arXiv (RFC-2026-02 Sprint 1).

Downloads the documents named in ``manifest.json`` into ``corpus/semantic_gold/
documents/``. Documents are deliberately not committed (size + licensing, same
policy as ``corpus/documents/``); this script is how a fresh checkout
reproduces the corpus.

Usage:
    uv run python scripts/fetch_semantic_gold.py [--manifest PATH] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "corpus" / "semantic_gold" / "manifest.json"
DOCS_DIR = ROOT / "corpus" / "semantic_gold" / "documents"

#: arXiv asks automated clients to identify themselves and to rate-limit.
USER_AGENT = "ubt-semantic-gold/1.0 (https://github.com/; local research corpus)"


def _arxiv_pdf_url(source: str) -> str | None:
    """'arxiv:2101.03961v3' -> the PDF URL, or None for a non-arXiv source."""
    if not source.startswith("arxiv:"):
        return None
    return f"https://arxiv.org/pdf/{source.split(':', 1)[1]}"


def fetch(url: str, dest: Path, *, timeout: int = 120) -> bool:
    if dest.exists() and dest.stat().st_size > 10_000:
        print(f"  = {dest.name} (already present, {dest.stat().st_size // 1024} KiB)")
        return True
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"  ! {dest.name}: {exc}", file=sys.stderr)
        return False
    # arXiv serves an HTML error page with a 200 for some malformed ids.
    if not data.startswith(b"%PDF"):
        print(f"  ! {dest.name}: response is not a PDF ({len(data)} bytes)", file=sys.stderr)
        return False
    tmp.write_bytes(data)
    tmp.replace(dest)
    print(f"  + {dest.name} ({len(data) // 1024} KiB)")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--dry-run", action="store_true", help="list what would be fetched")
    args = ap.parse_args()

    if not args.manifest.exists():
        print(f"error: manifest not found: {args.manifest}", file=sys.stderr)
        return 2
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    ok = failed = 0
    for case in manifest["cases"]:
        url = _arxiv_pdf_url(case["source"])
        if url is None:
            print(f"  - {case['id']}: no fetcher for source {case['source']!r}")
            continue
        dest = DOCS_DIR / f"{case['id']}.pdf"
        if args.dry_run:
            print(f"  would fetch {url} -> {dest.name}")
            continue
        if fetch(url, dest):
            ok += 1
        else:
            failed += 1

    if not args.dry_run:
        print(f"\n{ok} present, {failed} failed, {len(manifest['cases'])} total")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
