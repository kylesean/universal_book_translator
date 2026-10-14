"""The corpus gate, over a corpus the test builds itself.

The real-document corpus gate (``ubt verify --corpus --run`` over
``corpus/cases.json``) is the strongest check the product has -- the *real
pipeline* over real PDFs, each delivery contract reconciled against
thresholds -- but its documents are gitignored, so in CI the two corpus
harnesses skip and that gate guards nothing. This module closes the gap with
a corpus the test writes itself: synthetic PDFs built by
:mod:`tests.pdf_builders` (the same fixtures the render e2e uses), a
``cases.json`` with thresholds a rehearsal can meet, and the *same CLI
entrypoint* the real gate runs.

This is not the real-document gate (real pagination, real fonts, real scans
stay with the corpus harnesses); it is the guarantee that in an environment
with only the committed code, the gate's machinery -- dry-run translation,
contract writing, expectation reconciliation, exit codes -- executes and
holds. A regression that makes ``ubt verify`` unable to fail a bad delivery,
or unable to pass a good one, is caught in CI rather than discovered the next
time someone populates ``corpus/documents/``.

The negative case matters as much as the positive one: a case whose
``min_delivered_ratio`` no rehearsal can meet must exit non-zero, proving the
gate still fails closed.

Why ``slow``: each case shells out to ``python -m ubt translate --dry-run``
(Typst compile included), like the rest of the render e2e tier.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from pdf_builders import write_text_pdf

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        shutil.which("typst") is None,
        reason="the typst compiler is not installed; the render path cannot run",
    ),
]

_REPO = Path(__file__).resolve().parents[2]

_PROSE_PAGES = [
    (
        "The Attention Machine",
        "The machine relies on attention. Each layer runs a forward pass over the batch.",
    ),
    (
        "The Descendants",
        "The second chapter widens the lens to every descendant of the first model.",
    ),
]


def _write_corpus(root: Path) -> None:
    """A two-case corpus: one clean prose PDF, one deliberately empty PDF."""
    documents = root / "documents"
    documents.mkdir(parents=True)
    write_text_pdf(documents / "prose-book.pdf", _PROSE_PAGES)
    (documents / "empty-book.pdf").write_bytes(b"%PDF-1.4 broken \x00\x01")
    (root / "cases.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    {
                        "id": "synthetic-prose",
                        "description": "Synthetic prose PDF: the dry run must deliver every node.",
                        "document": "documents/prose-book.pdf",
                        "expect": {
                            "max_errors": 0,
                            "max_missing_assets": 0,
                            "max_source_kept": 0,
                            "min_accounted_ratio": 0.9,
                            "min_delivered_ratio": 0.5,
                        },
                    },
                    {
                        "id": "synthetic-broken",
                        "description": "An unparseable PDF must fail the run, never verify.",
                        "document": "documents/empty-book.pdf",
                        "expect": {
                            "max_errors": 0,
                            "min_delivered_ratio": 0.5,
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def _run_verify(corpus_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "ubt",
            "verify",
            "--corpus",
            str(corpus_dir),
            "--run",
            "--json",
        ],
        cwd=_REPO,
        capture_output=True,
        text=True,
        timeout=900,
    )


def test_the_verify_gate_passes_a_rehearsal_corpus(tmp_path: Path) -> None:
    """The same gate the real corpus uses, over a synthetic corpus, passes."""
    corpus = tmp_path / "corpus"
    _write_corpus(corpus)
    # Drop the case the dry run must refuse: the positive case is this test's
    # subject, the negative one is the next test's.
    cases = json.loads((corpus / "cases.json").read_text(encoding="utf-8"))
    cases["cases"] = [c for c in cases["cases"] if c["id"] == "synthetic-prose"]
    (corpus / "cases.json").write_text(json.dumps(cases), encoding="utf-8")
    cases = json.loads((corpus / "cases.json").read_text(encoding="utf-8"))

    proc = _run_verify(corpus)
    if proc.returncode != 0:
        pytest.fail(
            f"verify exited {proc.returncode}\n"
            f"--- stdout tail ---\n{proc.stdout[-2000:]}\n"
            f"--- stderr tail ---\n{proc.stderr[-1000:]}"
        )
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["status"] == "pass", payload
    (entry,) = payload["cases"]
    assert entry["id"] == "synthetic-prose"
    assert entry["status"] == "pass", entry
    # The rehearsal delivered every translatable node on this fixture.
    assert entry["total_text"] > 0
    assert entry["delivered_text"] == entry["total_text"]


def test_the_verify_gate_fails_a_case_that_cannot_deliver(tmp_path: Path) -> None:
    """A case whose run dies must make the gate exit non-zero (fail closed)."""
    corpus = tmp_path / "corpus"
    _write_corpus(corpus)
    cases = json.loads((corpus / "cases.json").read_text(encoding="utf-8"))
    cases["cases"] = [c for c in cases["cases"] if c["id"] == "synthetic-broken"]
    (corpus / "cases.json").write_text(json.dumps(cases), encoding="utf-8")

    proc = _run_verify(corpus)
    assert proc.returncode != 0, f"the gate passed an untranslatable document: {proc.stdout[-800:]}"
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["status"] == "fail"
    (entry,) = payload["cases"]
    assert entry["id"] == "synthetic-broken"
    assert entry["status"] == "fail", entry


__all__ = ["pytest"]
