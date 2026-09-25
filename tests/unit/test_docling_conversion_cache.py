"""The Docling conversion cache: key contract and real reuse through the parser.

The enriched pass is the single most expensive step of ingesting a technical
book (measured: 759 s of the 769 s a 26-page chapter took), so re-ingesting the
same file — ``--fresh``, another target language, or a crash mid-ingest — must
not pay for it again. What it must do instead is *miss* whenever anything that
shapes the layout changed, which is what the key tests below pin.
"""

import os
from pathlib import Path
from typing import Any

import pytest

from ubt.adapters.pdf import docling_parser


class _StubOptions:
    """Stand-in for ``PdfPipelineOptions``: only ``model_dump`` matters."""

    def __init__(self, **overrides: Any) -> None:
        self.payload: dict[str, Any] = {
            "do_ocr": False,
            "do_formula_enrichment": False,
            "do_table_structure": True,
            "accelerator_options": {"device": "auto"},
        }
        self.payload.update(overrides)

    def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
        return self.payload


def test_key_is_stable_for_identical_inputs() -> None:
    a = docling_parser.docling_cache_key("sha-a", None, _StubOptions())
    b = docling_parser.docling_cache_key("sha-a", None, _StubOptions())
    assert a == b


def test_key_follows_everything_that_shapes_the_layout() -> None:
    """A key that misses a real input would serve the wrong document back."""
    base = docling_parser.docling_cache_key("sha-a", None, _StubOptions())
    variants = {
        "different file": docling_parser.docling_cache_key("sha-b", None, _StubOptions()),
        "page range": docling_parser.docling_cache_key("sha-a", (1, 5), _StubOptions()),
        "ocr switched on": docling_parser.docling_cache_key(
            "sha-a", None, _StubOptions(do_ocr=True)
        ),
        "enrichment switched on": docling_parser.docling_cache_key(
            "sha-a", None, _StubOptions(do_formula_enrichment=True)
        ),
        "cpu fallback after a CUDA failure": docling_parser.docling_cache_key(
            "sha-a",
            None,
            _StubOptions(accelerator_options={"device": "cpu"}),
        ),
    }
    assert base not in variants.values()
    assert len(set(variants.values())) == len(variants)


def test_empty_conversion_is_not_cached(tmp_path: Path) -> None:
    """Caching a conversion with no pages would make a transient failure permanent."""

    class _Empty:
        pages: dict[int, Any] = {}

        def model_dump(self, *, mode: str = "python") -> dict[str, Any]:  # pragma: no cover
            raise AssertionError("must not serialize a page-less document")

    docling_parser.write_docling_document("k", _Empty(), cache_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_corrupt_entry_reads_as_a_miss(tmp_path: Path) -> None:
    (tmp_path / "k.json.gz").write_bytes(b"not gzip at all")
    assert docling_parser.read_docling_document("k", cache_dir=tmp_path) is None
    assert docling_parser.read_docling_document("absent", cache_dir=tmp_path) is None


@pytest.mark.network
@pytest.mark.slow
def test_second_ingest_reuses_the_cached_conversion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two parses of one file must cost one Docling run, not two."""
    pytest.importorskip("docling")

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pdf = Path("docs/synthetic-mono.pdf").resolve()
    monkeypatch.chdir(tmp_path)
    cache = tmp_path / "docling_cache"
    monkeypatch.setattr(docling_parser, "DOCLING_CACHE_DIR", cache)

    conversions: list[str] = []

    class _CountingConverter:
        def __init__(self, **kwargs: Any) -> None:
            self._inner = DocumentConverter(**kwargs)

        def convert(self, path: Any, **kwargs: Any) -> Any:
            conversions.append(str(path))
            return self._inner.convert(path, **kwargs)

    def symbols() -> tuple[Any, Any, Any, Any]:
        return (InputFormat, PdfPipelineOptions, _CountingConverter, PdfFormatOption)

    first = docling_parser.extract_with_docling(pdf, None, symbols=symbols, enrich=False)
    assert conversions, "the first pass must actually convert"
    assert first, "the sample document must yield blocks"

    conversions.clear()
    second = docling_parser.extract_with_docling(pdf, None, symbols=symbols, enrich=False)

    assert conversions == [], "the second pass re-ran Docling instead of reusing the cache"
    assert [b.model_dump() for b in second] == [b.model_dump() for b in first]


def test_asset_dir_lives_under_the_cache_root_and_follows_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§10.5-A6: ledger IMAGE blocks store these paths, so they must share the
    cache's home — deletable together, resumable together — and a redirected
    cache must redirect its assets too (read at call time, not import time)."""
    sha = "ab" * 32
    default = docling_parser.docling_asset_dir(sha)
    assert default.parts[-3:] == ("docling_cache", "assets", sha)

    cache = tmp_path / "elsewhere"
    monkeypatch.setattr(docling_parser, "DOCLING_CACHE_DIR", cache)
    assert docling_parser.docling_asset_dir(sha) == cache / "assets" / sha


def test_offline_retry_restores_hf_hub_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The online retry must not leave HF_HUB_OFFLINE popped process-wide.

    ``configure_hf_environment`` sets it for the whole process; one document's
    offline fallback silently disabling it would let every later job download
    models the operator explicitly turned off.
    """
    monkeypatch.setattr(docling_parser, "configure_hf_environment", lambda **_k: None)
    monkeypatch.setattr(docling_parser, "read_docling_document", lambda *_a, **_k: None)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_ENDPOINT", "https://mirror.invalid")  # setdefault becomes a no-op

    class OfflineModeError(Exception):
        pass

    class _Options:
        do_formula_enrichment = False

        def model_dump(self, *, mode: str = "json") -> dict[str, Any]:
            return {}

    class _Converter:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def convert(self, _path: Any, **_kwargs: Any) -> Any:
            raise OfflineModeError("offline mode: model is not cached")

    class _InputFormat:
        PDF = "pdf"

    def symbols() -> tuple[Any, Any, Any, Any]:
        return (_InputFormat, _Options, _Converter, lambda **_k: object())

    pdf = tmp_path / "book.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    with pytest.raises(OfflineModeError):
        docling_parser.extract_with_docling(pdf, None, symbols=symbols, enrich=False)

    assert os.environ.get("HF_HUB_OFFLINE") == "1"


def test_offline_retry_restores_hf_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The online retry points HF at a third-party mirror; it must not persist.

    Every later HF download in the process — including the DeepSeek worker's
    ``snapshot_download``, which inherits ``subprocess_env()`` — would otherwise
    silently use a mirror the operator never configured.
    """
    monkeypatch.setattr(docling_parser, "configure_hf_environment", lambda **_k: None)
    monkeypatch.setattr(docling_parser, "read_docling_document", lambda *_a, **_k: None)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("HF_ENDPOINT", raising=False)

    class OfflineModeError(Exception):
        pass

    class _Options:
        do_formula_enrichment = False

        def model_dump(self, *, mode: str = "json") -> dict[str, Any]:
            return {}

    class _Converter:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def convert(self, _path: Any, **_kwargs: Any) -> Any:
            raise OfflineModeError("offline mode: model is not cached")

    class _InputFormat:
        PDF = "pdf"

    def symbols() -> tuple[Any, Any, Any, Any]:
        return (_InputFormat, _Options, _Converter, lambda **_k: object())

    pdf = tmp_path / "book.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    with pytest.raises(OfflineModeError):
        docling_parser.extract_with_docling(pdf, None, symbols=symbols, enrich=False)

    assert "HF_ENDPOINT" not in os.environ, "the retry's mirror must not outlive the ladder"


class _StubDocWithPicture:
    """A converted document whose picture items carry base64 payloads."""

    def __init__(self) -> None:
        self.pages = {1: object()}

    def model_dump(self, *, mode: str = "python") -> dict[str, object]:
        return {
            "schema_name": "DoclingDocument",
            "texts": [
                {
                    "label": "Picture",
                    "image": {
                        "mimetype": "image/png",
                        "dpi": 96,
                        "size": {"width": 10, "height": 10},
                        "uri": "data:image/png;base64," + "A" * 400_000,
                    },
                }
            ],
        }


def test_cache_write_drops_embedded_picture_payloads(tmp_path: Path) -> None:
    """The docstring promised this; ``model_dump`` kept every data-URI anyway.

    A real cache entry measured 31 MB of picture base64, and reading it back held
    bytes + string + dict + validated model at once, so ingest peak RSS scaled
    about 5x with the book — the inverse of why the cache exists.
    """
    import gzip

    docling_parser.write_docling_document("pic", _StubDocWithPicture(), cache_dir=tmp_path)
    blob = gzip.decompress((tmp_path / "pic.json.gz").read_bytes()).decode("utf-8")

    assert "data:image" not in blob
    assert len(blob) < 2_000
    assert '"width": 10' in blob  # geometry survives; only the payload goes


@pytest.mark.fast
def test_docling_parser_cache_image_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    import types
    from unittest.mock import MagicMock

    from ubt.core.ir import BlockType

    mock_mod = types.ModuleType("docling_core.types.doc.labels")

    class MockDocItemLabel:
        PICTURE = "picture"
        EMPTY_VALUE = "empty_value"
        MARKER = "marker"
        CHECKBOX_SELECTED = "checkbox_selected"
        CHECKBOX_UNSELECTED = "checkbox_unselected"
        HANDWRITTEN_TEXT = "handwritten_text"
        PAGE_HEADER = "page_header"
        PAGE_FOOTER = "page_footer"

    for k in (
        "PARAGRAPH",
        "SECTION_HEADER",
        "TITLE",
        "CAPTION",
        "TABLE",
        "FORMULA",
        "CODE",
        "LIST_ITEM",
        "FOOTNOTE",
    ):
        setattr(MockDocItemLabel, k, k.lower())
    # __dict__ assignment (not attribute syntax) so the fake module attribute is
    # accepted whether or not docling is installed in the checking environment.
    mock_mod.__dict__["DocItemLabel"] = MockDocItemLabel
    monkeypatch.setitem(sys.modules, "docling_core", types.ModuleType("docling_core"))
    monkeypatch.setitem(sys.modules, "docling_core.types", types.ModuleType("docling_core.types"))
    monkeypatch.setitem(
        sys.modules, "docling_core.types.doc", types.ModuleType("docling_core.types.doc")
    )
    monkeypatch.setitem(sys.modules, "docling_core.types.doc.labels", mock_mod)

    from ubt.adapters.pdf.docling_parser import map_iterated_items

    class MockBbox:
        l = 10.0  # noqa: E741
        b = 20.0
        r = 100.0
        t = 120.0

    class MockProv:
        page_no = 1
        bbox = MockBbox()

    class MockPicItem:
        label = MockDocItemLabel.PICTURE
        image = None
        prov = [MockProv()]
        text = ""

    assets_dir = tmp_path / "assets"
    assets_dir.mkdir(parents=True)
    existing_asset = assets_dir / "pic_p1_1.png"
    existing_asset.write_bytes(b"dummy_png_data")

    doc = MagicMock()
    blocks = map_iterated_items([(MockPicItem(), 0)], doc, assets_dir)
    assert len(blocks) == 1, f"Expected 1 image block recovered from cache, got {len(blocks)}"
    assert blocks[0].block_type == BlockType.IMAGE
    assert blocks[0].source_text == str(existing_asset)
