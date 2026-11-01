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
    baseline = _key()
    bumped = _key(**{package: _VERSIONS[package] + ".post1"})
    assert bumped != baseline, f"a {package} upgrade must invalidate the cache entry"


def test_a_new_model_snapshot_re_converts(tmp_path: Path) -> None:
    hub = tmp_path / "hub"
    repo = hub / "models--docling-project--docling-layout-heron"
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text("a" * 40, encoding="utf-8")

    with (
        patch.object(cache, "_hf_hub_cache", return_value=hub),
        patch.object(cache, "_local_artifacts_identity", return_value=None),
    ):
        first = _key()
        (repo / "refs" / "main").write_text("b" * 40, encoding="utf-8")
        second = _key()

    assert first != second, "a new model snapshot must invalidate the cached conversion"


def test_a_missing_model_repo_is_a_stable_identity(tmp_path: Path) -> None:
    hub = tmp_path / "hub"
    hub.mkdir()

    with (
        patch.object(cache, "_hf_hub_cache", return_value=hub),
        patch.object(cache, "_local_artifacts_identity", return_value=None),
    ):
        assert _key() == _key()


def test_a_local_artifact_store_changes_the_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    from types import SimpleNamespace

    from ubt.adapters.pdf import docling_cache as module

    store = tmp_path / "artifacts"
    store.mkdir()
    fake = SimpleNamespace(settings=SimpleNamespace(artifacts_path=str(store)))
    monkeypatch.setitem(sys.modules, "docling.datamodel.settings", fake)

    first = _key()
    other = tmp_path / "other-artifacts"
    other.mkdir()
    fake.settings.artifacts_path = str(other)

    assert first != _key()
    assert module._local_artifacts_identity() == f"artifacts={other}:{other.stat().st_mtime_ns}"


def test_a_replaced_local_artifact_snapshot_changes_the_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os
    import sys
    from types import SimpleNamespace

    store = tmp_path / "artifacts"
    store.mkdir()
    fake = SimpleNamespace(settings=SimpleNamespace(artifacts_path=str(store)))
    monkeypatch.setitem(sys.modules, "docling.datamodel.settings", fake)

    first = _key()
    stat = store.stat()
    os.utime(store, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    assert first != _key(), "replacing the weights under the store must re-convert"


def _write_entry(directory: Path, key: str, *, pages: int, texts: int) -> None:
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
    _write_entry(tmp_path, "truncated", pages=6, texts=6)

    assert cache.read_docling_document("truncated", cache_dir=tmp_path, expected_pages=26) is None
    assert (
        cache.read_docling_document("truncated", cache_dir=tmp_path, expected_pages=6) is not None
    )
