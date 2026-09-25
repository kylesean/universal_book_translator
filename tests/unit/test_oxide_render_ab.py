"""Gated A/B of the pdf_oxide rasterizer against poppler pdftoppm.

Skipped unless BOTH the ``oxide`` extra and the poppler binary are present,
and deselected by default (``-m 'not slow'``); the CI job that installs
``--extra oxide`` runs it with ``-m slow``. Thresholds and their provenance
live in scripts/oxide_render_ab.py.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

pytest.importorskip("pdf_oxide")
pytest.importorskip("PIL")

if shutil.which("pdftoppm") is None:
    pytest.skip("pdftoppm unavailable", allow_module_level=True)

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from oxide_render_ab import DPIS, PAGES, compare_pair  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
# The original real documents were removed in the 2026-09 legal review; the
# gate now runs on the generated, copyright-safe synthetic corpus
# (scripts/make_sample_corpus.py), which tests/conftest.py seeds before
# collection.
CORPUS = [
    REPO_ROOT / "docs" / "synthetic-mono.pdf",
    REPO_ROOT / "docs" / "synthetic-duo.pdf",
]


def _require_corpus(pdf: Path) -> None:
    """Fail loudly when the A/B corpus is absent.

    The corpus (docs/synthetic-*.pdf) is gitignored and generated, not
    committed: ``tests/conftest.py`` rebuilds it at ``pytest_configure`` time
    (it needs the ``typst`` binary, which the pdf CI job installs). Skipping
    here — the previous behaviour — turned the stage-2 adoption gate into a
    silent no-op: the job stayed green while the rasterizer equivalence it
    exists to prove was never checked. A genuinely unbuildable corpus (no typst
    binary, or the generator failing) is a broken gate, not an inapplicable
    one, so it fails the job rather than letting the adoption gate no-op.
    """
    if not pdf.exists():
        pytest.fail(
            f"A/B corpus {pdf.name} is missing. docs/*.pdf is gitignored, so the "
            "corpus must be fetched from its Release attachment before this gate "
            "can run — the stage-2 pdf_oxide adoption gate cannot be skipped."
        )


@pytest.mark.slow
@pytest.mark.parametrize("pdf", CORPUS, ids=lambda p: p.name)
@pytest.mark.parametrize("dpi", DPIS)
def test_render_pair_matches(pdf: Path, dpi: int) -> None:
    _require_corpus(pdf)
    for page in PAGES:
        result = compare_pair(pdf, page, dpi)
        assert result.size_ok, (
            f"{pdf.name} p{page}@{dpi}: {result.oxide_size} vs {result.pdftoppm_size}"
        )
        assert result.mismatch_ratio <= result.threshold, (
            f"{pdf.name} p{page}@{dpi}: mismatch ratio {result.mismatch_ratio:.4f} "
            f"> {result.threshold}"
        )
