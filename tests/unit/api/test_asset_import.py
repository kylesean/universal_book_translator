"""Language-asset import + glossary conflict detection (PRD §4.4).

The console's asset center imports TMX/JSON into the shared TM and surfaces
glossary sources configured with two competing renderings. Both are pure
helpers over explicit paths, pinned here plus a thin HTTP round-trip.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from ubt.api.app import create_app
from ubt.api.assets import (
    GlossaryFormatError,
    TMImportError,
    detect_glossary_conflicts,
    import_tm_entries,
    parse_glossary_payload,
    parse_tm_json,
    parse_tm_payload,
    parse_tmx,
    read_glossary_terms,
)
from ubt.core.config import UBTConfig
from ubt.core.memory.tm import PROVENANCE_HUMAN_PE, PROVENANCE_MACHINE, TranslationMemory

pytestmark = pytest.mark.fast

_API_KEY = "assets-test-key"
_AUTH = {"X-API-Key": _API_KEY}


def _config(tmp_path: Path, **overrides: Any) -> UBTConfig:
    return UBTConfig(
        db_dir=tmp_path / "db",
        allowed_dirs=str(tmp_path),
        service_api_key=SecretStr(_API_KEY),
        **overrides,
    )


# --------------------------------------------------------------------------- #
# Glossary conflicts
# --------------------------------------------------------------------------- #


def test_conflict_detection_flags_competing_targets() -> None:
    terms = [
        {"source": "Backpropagation", "target": "反向传播"},
        {"source": "Backpropagation", "target": "反向传递"},
        {"source": "Attention", "target": "注意力"},
    ]
    assert detect_glossary_conflicts(terms) == [
        {"source": "Backpropagation", "targets": ["反向传播", "反向传递"]}
    ]


def test_conflict_detection_ignores_repeated_identical_targets() -> None:
    terms = [
        {"source": "Attention", "target": "注意力"},
        {"source": "Attention", "target": "注意力"},
    ]
    assert detect_glossary_conflicts(terms) == []


# --------------------------------------------------------------------------- #
# TMX / JSON parsing
# --------------------------------------------------------------------------- #

_TMX = """<?xml version="1.0" encoding="UTF-8"?>
<tmx version="1.4">
  <body>
    <tu>
      <tuv xml:lang="en"><seg>Hello world</seg></tuv>
      <tuv xml:lang="zh"><seg>你好世界</seg></tuv>
    </tu>
    <tu>
      <tuv xml:lang="en"><seg>Goodbye</seg></tuv>
      <tuv xml:lang="zh"><seg>再见</seg></tuv>
    </tu>
  </body>
</tmx>
"""


def test_parse_tmx_pairs_segments_by_language() -> None:
    rows = parse_tmx(_TMX)
    assert rows == [
        {
            "src_lang": "en",
            "tgt_lang": "zh",
            "source_text": "Hello world",
            "target_text": "你好世界",
        },
        {"src_lang": "en", "tgt_lang": "zh", "source_text": "Goodbye", "target_text": "再见"},
    ]


def test_parse_tmx_rejects_invalid_xml() -> None:
    with pytest.raises(TMImportError):
        parse_tmx("<not-xml")


def test_parse_tm_json_accepts_short_keys() -> None:
    rows = parse_tm_json('[{"source": "Foo", "target": "福"}]')
    assert rows == [{"src_lang": "", "tgt_lang": "", "source_text": "Foo", "target_text": "福"}]


def test_parse_tm_payload_rejects_unknown_format() -> None:
    with pytest.raises(TMImportError):
        parse_tm_payload("[]", "yaml")


# --------------------------------------------------------------------------- #
# Import into the store
# --------------------------------------------------------------------------- #


def test_import_writes_rows_with_the_default_language_pair(tmp_path: Path) -> None:
    tm_path = tmp_path / "tm.sqlite"
    rows = [{"src_lang": "", "tgt_lang": "", "source_text": "Hello", "target_text": "你好"}]
    written = import_tm_entries(
        tm_path,
        rows,
        default_src_lang="en",
        default_tgt_lang="zh",
        provenance=PROVENANCE_MACHINE,
    )
    assert written == 1

    tm = TranslationMemory(tm_path)
    try:
        (entry,) = tm.scan()
    finally:
        tm.close()
    assert (entry.src_lang, entry.tgt_lang) == ("en", "zh")
    assert entry.source_text == "Hello" and entry.target_text == "你好"


def test_machine_import_skips_a_human_row(tmp_path: Path) -> None:
    tm_path = tmp_path / "tm.sqlite"
    import_tm_entries(
        tm_path,
        [{"src_lang": "en", "tgt_lang": "zh", "source_text": "Hello", "target_text": "你好"}],
        default_src_lang="en",
        default_tgt_lang="zh",
        provenance=PROVENANCE_HUMAN_PE,
    )
    # The store's writeback keeps the human label but lets latest text win, so a
    # bulk machine import must skip the pair entirely to preserve the human row.
    written = import_tm_entries(
        tm_path,
        [{"src_lang": "en", "tgt_lang": "zh", "source_text": "Hello", "target_text": "喂"}],
        default_src_lang="en",
        default_tgt_lang="zh",
        provenance=PROVENANCE_MACHINE,
    )
    assert written == 0
    tm = TranslationMemory(tm_path)
    try:
        (entry,) = tm.scan()
    finally:
        tm.close()
    assert entry.provenance == PROVENANCE_HUMAN_PE
    assert entry.target_text == "你好"


def test_import_rejects_an_unknown_provenance(tmp_path: Path) -> None:
    with pytest.raises(TMImportError):
        import_tm_entries(
            tmp_path / "tm.sqlite",
            [{"src_lang": "en", "tgt_lang": "zh", "source_text": "a", "target_text": "b"}],
            default_src_lang="en",
            default_tgt_lang="zh",
            provenance="invented",
        )


def test_an_imported_machine_row_is_actually_hittable(tmp_path: Path) -> None:
    """P2-11: an imported machine pair must be reachable by the pipeline.

    The import writes ``context_hash=''`` (no pipeline run produced it), while
    the draft stage looks up with a computed ``compute_tm_context`` hash. Before
    the fix the read gate required equality, so the row was written but never
    hit — the import reported success and did nothing.
    """
    from ubt.core.memory.tm import compute_tm_context

    tm_path = tmp_path / "tm.sqlite"
    written = import_tm_entries(
        tm_path,
        [
            {
                "src_lang": "en",
                "tgt_lang": "zh",
                "source_text": "Hello world",
                "target_text": "你好世界",
            }
        ],
        default_src_lang="en",
        default_tgt_lang="zh",
        provenance=PROVENANCE_MACHINE,
    )
    assert written == 1
    tm = TranslationMemory(tm_path)
    try:
        # A later run's context fingerprint — different prompt/model/glossary —
        # must still hit the imported pair.
        ctx = compute_tm_context("v9", "technical", "glossary-v2", "en", "zh")
        hit = tm.lookup_exact("en", "zh", "Hello world", ctx)
        assert hit is not None
        assert hit.target_text == "你好世界"
    finally:
        tm.close()


def test_a_pipeline_machine_row_stays_context_gated(tmp_path: Path) -> None:
    """The import exemption must not un-gate rows the pipeline wrote.

    A pipeline machine row carries a real context hash, so a changed
    prompt/glossary/model must still miss it.
    """
    from ubt.core.memory.tm import TMPendingEntry, compute_tm_context

    tm_path = tmp_path / "tm.sqlite"
    ctx = compute_tm_context("v1", "general", "", "en", "zh")
    tm = TranslationMemory(tm_path)
    try:
        tm.writeback(
            [
                TMPendingEntry(
                    src_lang="en",
                    tgt_lang="zh",
                    source_text="Pipeline row",
                    target_text="管道行",
                    provenance=PROVENANCE_MACHINE,
                    context_hash=ctx,
                )
            ]
        )
        assert tm.lookup_exact("en", "zh", "Pipeline row", ctx) is not None
        other = compute_tm_context("v2", "general", "", "en", "zh")
        assert tm.lookup_exact("en", "zh", "Pipeline row", other) is None
    finally:
        tm.close()


# --------------------------------------------------------------------------- #
# HTTP surface
# --------------------------------------------------------------------------- #


def test_conflicts_endpoint_reports_a_conflicting_glossary(tmp_path: Path) -> None:
    glossary = tmp_path / "terms.csv"
    glossary.write_text(
        "source,translation\nBackpropagation,反向传播\nBackpropagation,反向传递\n",
        encoding="utf-8",
    )
    client = TestClient(create_app(_config(tmp_path, glossary_path=glossary)))
    body = client.get("/assets/glossary/conflicts", headers=_AUTH).json()
    assert body["conflicts"] == [{"source": "Backpropagation", "targets": ["反向传播", "反向传递"]}]


def test_tm_import_endpoint_roundtrips_tmx_and_json(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.db_dir.mkdir(parents=True, exist_ok=True)
    client = TestClient(create_app(config))

    res = client.post(
        "/assets/tm/import",
        json={"format": "tmx", "content": _TMX, "src_lang": "en", "tgt_lang": "zh"},
        headers=_AUTH,
    )
    assert res.status_code == 200
    assert res.json() == {"parsed": 2, "imported": 2}

    res = client.post(
        "/assets/tm/import",
        json={
            "format": "json",
            "content": '[{"source_text": "Foo", "target_text": "福"}]',
            "src_lang": "en",
            "tgt_lang": "zh",
        },
        headers=_AUTH,
    )
    assert res.status_code == 200
    assert res.json() == {"parsed": 1, "imported": 1}

    listed = client.get("/assets/tm", headers=_AUTH).json()
    assert listed["total"] == 3


def test_tm_import_endpoint_rejects_a_malformed_payload(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.db_dir.mkdir(parents=True, exist_ok=True)
    client = TestClient(create_app(config))
    res = client.post(
        "/assets/tm/import",
        json={"format": "tmx", "content": "<broken", "src_lang": "en", "tgt_lang": "zh"},
        headers=_AUTH,
    )
    assert res.status_code == 422


# --------------------------------------------------------------------------- #
# Glossary bulk import
# --------------------------------------------------------------------------- #


def test_parse_glossary_payload_reads_header_and_headerless_csv() -> None:
    with_header = parse_glossary_payload("source,translation\nMOSFET,MOSFET\n", "csv")
    assert with_header == [("MOSFET", "MOSFET")]
    # Headerless two-column rows are what a hand-maintained file often is.
    assert parse_glossary_payload("threshold voltage,阈值电压\n", "csv") == [
        ("threshold voltage", "阈值电压")
    ]
    assert parse_glossary_payload("gate oxide\t栅氧化层\n", "tsv") == [("gate oxide", "栅氧化层")]


def test_parse_glossary_payload_accepts_both_json_shapes() -> None:
    as_dict = parse_glossary_payload('{"SPICE": "SPICE", "BSIM": "BSIM"}', "json")
    assert as_dict == [("SPICE", "SPICE"), ("BSIM", "BSIM")]
    as_list = parse_glossary_payload('[{"term": "GAA", "translation": "GAA"}]', "json")
    assert as_list == [("GAA", "GAA")]


def test_parse_glossary_payload_rejects_malformed_content() -> None:
    with pytest.raises(GlossaryFormatError):
        parse_glossary_payload("{not json", "json")
    with pytest.raises(GlossaryFormatError):
        parse_glossary_payload('["a string, not an object"]', "json")
    with pytest.raises(GlossaryFormatError):
        parse_glossary_payload("a,b", "xml")


def test_glossary_import_endpoint_roundtrips_and_preserves_format(tmp_path: Path) -> None:
    glossary = tmp_path / "terms.csv"
    glossary.write_text("source,translation\nMOSFET,MOSFET\n", encoding="utf-8")
    client = TestClient(create_app(_config(tmp_path, glossary_path=glossary)))

    res = client.post(
        "/assets/glossary/import",
        json={
            "format": "csv",
            "content": "source,translation\nsubthreshold swing,亚阈值摆幅\nMOSFET,MOSFET\n",
        },
        headers=_AUTH,
    )
    assert res.status_code == 200
    assert res.json() == {"parsed": 2, "imported": 2}

    listed = client.get("/assets/glossary", headers=_AUTH).json()
    assert {term["source"]: term["target"] for term in listed["terms"]} == {
        "MOSFET": "MOSFET",
        "subthreshold swing": "亚阈值摆幅",
    }
    # The file keeps its header + CSV shape: the engine's own loader must
    # still read it, which is the whole point of editing it in place.
    assert glossary.read_text(encoding="utf-8").startswith("source,translation")


def test_glossary_import_endpoint_rejects_a_malformed_payload(tmp_path: Path) -> None:
    glossary = tmp_path / "terms.json"
    glossary.write_text("{}\n", encoding="utf-8")
    client = TestClient(create_app(_config(tmp_path, glossary_path=glossary)))
    res = client.post(
        "/assets/glossary/import",
        json={"format": "json", "content": "{not json"},
        headers=_AUTH,
    )
    assert res.status_code == 422


def test_glossary_upload_endpoint_stages_a_path_jobs_can_use(tmp_path: Path) -> None:
    client = TestClient(create_app(_config(tmp_path)))
    res = client.post(
        "/assets/glossary/upload",
        files={"file": ("terms.csv", b"source,translation\nMOSFET,MOSFET\n", "text/csv")},
        headers=_AUTH,
    )
    assert res.status_code == 200
    body = res.json()
    assert body["name"] == "terms.csv"
    staged = Path(body["path"])
    assert staged.exists()
    # The staged path must sit inside the managed uploads dir the engine can
    # read, and the engine's own loader must be able to parse it back — the
    # job submit hands this path to the pipeline untouched.
    assert "uploads" in staged.parts
    assert read_glossary_terms(staged) == [{"source": "MOSFET", "target": "MOSFET"}]


def test_glossary_upload_endpoint_rejects_non_glossary_formats(tmp_path: Path) -> None:
    client = TestClient(create_app(_config(tmp_path)))
    res = client.post(
        "/assets/glossary/upload",
        files={"file": ("book.pdf", b"%PDF-1.4 fake", "application/pdf")},
        headers=_AUTH,
    )
    assert res.status_code == 415
