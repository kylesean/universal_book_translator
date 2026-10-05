"""The analyze cache must invalidate when any block-shaping module changes."""

from __future__ import annotations

from pathlib import Path

import pytest

from ubt.adapters.pdf import docling_crosscheck, textgeom
from ubt.adapters.pdf.analysis_cache import docling_identity

pytestmark = pytest.mark.fast


def _file_identity(module: object) -> str:
    path = Path(module.__file__)  # type: ignore[attr-defined]
    stat = path.stat()
    return f"{path}:{stat.st_size}:{stat.st_mtime_ns}"


def test_docling_identity_covers_the_style_probe_modules() -> None:
    # ``docling_crosscheck`` writes block style (inline runs, first-line indent)
    # and ``textgeom`` supplies the pdfium probes it reads; keying only the parser
    # served stale blocks whose inline_runs were empty.
    identity = docling_identity()
    assert _file_identity(docling_crosscheck) in identity
    assert _file_identity(textgeom) in identity
