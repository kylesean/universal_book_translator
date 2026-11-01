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
    identity = docling_identity()
    assert _file_identity(docling_crosscheck) in identity
    assert _file_identity(textgeom) in identity


def test_docling_identity_covers_the_math_pruner() -> None:
    assert _file_identity(lnds_pruner) in docling_identity()


def test_docling_identity_covers_the_toc_synthesis() -> None:
    import ubt.adapters.pdf.docling_outline as outline

    assert _file_identity(outline) in docling_identity()


def test_the_docling_cache_dir_is_under_the_shared_cache_root() -> None:
    from ubt.adapters.pdf.docling_parser import DOCLING_CACHE_DIR
    from ubt.cache.dirs import cache_root

    expected = cache_root() / "docling_cache"
    assert expected == DOCLING_CACHE_DIR
    assert DOCLING_CACHE_DIR.is_absolute()
    assert not str(DOCLING_CACHE_DIR).startswith(".ubt")


def test_a_requested_enrichment_is_not_served_from_the_block_cache(tmp_path: Path) -> None:
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
            formula_crop_margin=0.08,
            toc_mode="auto",
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
