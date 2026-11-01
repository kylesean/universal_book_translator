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
    corpus = tmp_path / "corpus"
    _write_corpus(corpus)
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
    assert entry["total_text"] > 0
    assert entry["delivered_text"] == entry["total_text"]


def test_the_verify_gate_fails_a_case_that_cannot_deliver(tmp_path: Path) -> None:
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
