"""The analyze cache must invalidate when any block-shaping module changes."""

from __future__ import annotations

from pathlib import Path

import pytest

from ubt.adapters.pdf import docling_crosscheck, textgeom
from ubt.adapters.pdf.analysis_cache import docling_identity
from ubt.core.cleaners import lnds_pruner

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


def test_docling_identity_covers_the_math_pruner() -> None:
    """P2-6: the parser runs every title/body through the LNDS math pruner.

    ``normalize_academic_pdf_math`` (``ubt.core.cleaners.lnds_pruner``) shapes
    the cached text, so a pruner fix must invalidate the cached blocks. It was
    absent from the identity, so a fix was invisible on a cached re-run.
    """
    assert _file_identity(lnds_pruner) in docling_identity()


def test_the_docling_cache_dir_is_under_the_shared_cache_root() -> None:
    """P2-6: the Docling cache must not be CWD-relative.

    ``.ubt/docling_cache`` forked the cache per working directory (re-converting
    the same PDF in each) and landed the cached manuscript text inside a shared
    checkout. It belongs under ``cache_root()`` like every other disk cache.
    """
    from ubt.adapters.pdf.docling_parser import DOCLING_CACHE_DIR
    from ubt.cache.dirs import cache_root

    expected = cache_root() / "docling_cache"
    assert expected == DOCLING_CACHE_DIR
    assert DOCLING_CACHE_DIR.is_absolute()
    assert not str(DOCLING_CACHE_DIR).startswith(".ubt")


def test_a_requested_enrichment_is_not_served_from_the_block_cache(tmp_path: Path) -> None:
    """P2-6: a degraded enrichment run must not be cached as a healthy one.

    When enrichment is requested and Docling's ladder degrades (VLM OOM -> CPU
    or plain extraction), the blocks carry the same ``enrich=True`` key as a
    healthy run. Caching them made the degradation permanent. The outer block
    cache is bypassed for a requested enrichment; the inner conversion cache
    (keyed on the *effective* options) still dedupes the expensive conversion.
    """
    from unittest.mock import patch

    from ubt.adapters.pdf.docling_adapter import DoclingPDFAdapter
    from ubt.core.ports import AdapterRuntimeConfig

    adapter = DoclingPDFAdapter()
    adapter.apply_config(
        AdapterRuntimeConfig(
            ocr_mode="off",
            ocr_endpoint=None,
            ocr_api_key="",
            ocr_model="",
            formula_enrichment="on",
            docling_device="cpu",
            font_family=None,
            allow_page_upload=False,
            cache_dir=str(tmp_path / "cache"),
        )
    )
    assert adapter.analysis_cache is not None

    calls = {"n": 0}

    def fake_extract(path: Path, page_range: object) -> list[object]:
        calls["n"] += 1
        return []

    with (
        patch.object(adapter, "is_docling_installed", return_value=True),
        patch.object(adapter, "_resolve_formula_enrichment", return_value=True),
        patch.object(adapter, "_extract_with_docling", side_effect=fake_extract),
        patch("ubt.adapters.pdf.analysis_cache.cached_blocks") as cached,
    ):
        adapter._extract_blocks_sync(Path("/tmp/x.pdf"), None)
    cached.assert_not_called()
    assert calls["n"] == 1
