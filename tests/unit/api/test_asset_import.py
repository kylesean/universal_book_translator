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
    TMImportError,
    detect_glossary_conflicts,
    import_tm_entries,
    parse_tm_json,
    parse_tm_payload,
    parse_tmx,
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
