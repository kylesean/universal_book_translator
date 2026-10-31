"""The Docling conversion cache key must move with everything that shapes output."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from ubt.adapters.pdf import docling_cache as cache

pytestmark = pytest.mark.fast

_VERSIONS = {
    "docling": "2.140.0",
    "docling-core": "2.101.1",
    "docling-parse": "7.22.3",
    "docling-ibm-models": "4.0.2",
}


class _Options:
    """Stand-in for ``PdfPipelineOptions`` (the key only reads ``model_dump``)."""

    def model_dump(self, **_: object) -> dict[str, object]:
        return {"do_formula_enrichment": False}


def _key(**overrides: str) -> str:
    versions = {**_VERSIONS, **overrides}
    with patch.object(cache, "_package_version", side_effect=lambda name: versions[name]):
        return cache.docling_cache_key("sha", None, _Options())


def test_an_unchanged_environment_replays_the_entry() -> None:
    assert _key() == _key()


@pytest.mark.parametrize(
    "package",
    ["docling", "docling-core", "docling-parse", "docling-ibm-models"],
)
def test_upgrading_any_output_shaping_package_re_converts(package: str) -> None:
    """``docling`` alone is not enough.

    The layout and table models live in ``docling-ibm-models`` and the PDF reader
    in ``docling-parse``; docling pins both only by *range*
    (``docling-parse<8.0.0,>=7.22.1``, ``docling-ibm-models<5,>=4.0.2``), so
    ``uv lock`` can move either without moving ``docling``. Keying on the
    distribution and ``docling-core`` alone replayed a stale layout/table
    structure after such an upgrade — the operator believed they had the new
    parser's fixes.
    """
    baseline = _key()
    bumped = _key(**{package: _VERSIONS[package] + ".post1"})
    assert bumped != baseline, f"a {package} upgrade must invalidate the cache entry"


def _write_entry(directory: Path, key: str, *, pages: int, texts: int) -> None:
    """Write a cached conversion directly, without running Docling."""
    import gzip
    import json

    payload = {
        "schema_name": "DoclingDocument",
        "version": "1.0.0",
        "name": "cached",
        "origin": {"mimetype": "application/pdf", "binary_hash": 1, "filename": "x.pdf"},
        "pages": {
            str(no): {
                "page_no": no,
                "size": {"width": 100.0, "height": 100.0},
            }
            for no in range(1, pages + 1)
        },
        "texts": [
            {
                "self_ref": f"#/texts/{i}",
                "label": "paragraph",
                "text": f"paragraph {i}",
                "orig": f"paragraph {i}",
            }
            for i in range(texts)
        ],
    }
    with gzip.open(directory / f"{key}.json.gz", "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)


def test_a_truncated_conversion_is_not_replayed(tmp_path: Path) -> None:
    """Six paragraphs for a 26-page chunk is a broken conversion, not a sparse one.

    The old guard demanded only "at least one text item anywhere", so a
    conversion that returned a fraction of the document replayed as the whole
    book on every later run. The caller knows how many pages the chunk covers.
    """
    _write_entry(tmp_path, "truncated", pages=6, texts=6)

    assert cache.read_docling_document("truncated", cache_dir=tmp_path, expected_pages=26) is None
    assert (
        cache.read_docling_document("truncated", cache_dir=tmp_path, expected_pages=6) is not None
    )
