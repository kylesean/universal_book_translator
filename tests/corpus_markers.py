"""Skip guards for tests that need the generated synthetic PDF corpus.

``docs/synthetic-*.pdf`` are generated fixtures, not committed binaries (ROI-1
in .gitignore); ``tests/conftest.py`` rebuilds them at configure time via
``scripts/make_sample_corpus.py`` whenever they are missing. These markers turn
a test into an honest skip only when even that rebuild was impossible (no
``typst`` on PATH, or a failed build), so a machine without the toolchain
degrades visibly instead of silently going red.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_DOCS = Path(__file__).resolve().parents[1] / "docs"

requires_synthetic_mono = pytest.mark.skipif(
    not (_DOCS / "synthetic-mono.pdf").exists(),
    reason="docs/synthetic-mono.pdf not present (regenerate with scripts/make_sample_corpus.py)",
)

requires_synthetic_duo = pytest.mark.skipif(
    not (_DOCS / "synthetic-duo.pdf").exists(),
    reason="docs/synthetic-duo.pdf not present (regenerate with scripts/make_sample_corpus.py)",
)
