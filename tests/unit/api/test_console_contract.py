"""Console contract tripwire: the web client and the API must not drift.

The operator console's biggest failure mode is silent: a client that calls a
path the server does not serve (``/jobs/{id}/progress/stream`` vs the real
``/jobs/{id}/stream``) renders a dead UI rather than an error. These tests pin
the two halves together:

- every path ``web/src/api/client.ts`` calls exists in the live OpenAPI schema;
- the committed ``web/openapi.json`` is not stale against ``create_app()``;
- the deliverable catalog names files the way the export stage writes them;
- the glossary asset helpers round-trip a real file.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from ubt.api.app import DELIVERABLE_LABELS, _deliverable_paths, create_app
from ubt.api.assets import (
    add_glossary_term,
    read_glossary_terms,
    remove_glossary_term,
)
from ubt.core.config import UBTConfig
from ubt.core.job_options import companion_path, sidecar_path

pytestmark = pytest.mark.fast

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CLIENT_TS = _REPO_ROOT / "web" / "src" / "api" / "client.ts"
_OPENAPI_JSON = _REPO_ROOT / "web" / "openapi.json"

#: Matches ``${BASE_URL}/some/path?query`` inside a fetch/EventSource template.
_CLIENT_URL_RE = re.compile(r"\$\{BASE_URL\}(/[^`\"']*)")


def _normalize(path: str) -> str:
    """Collapse every path parameter to ``{}`` so the two sides compare by shape.

    Handles both the OpenAPI form (``{job_id}``) and the client's template form
    (``${encodeURIComponent(jobId)}``).
    """
    return re.sub(r"\$?\{[^}]+\}", "{}", path)


def _frontend_paths() -> set[str]:
    text = _CLIENT_TS.read_text(encoding="utf-8")
    paths: set[str] = set()
    for match in _CLIENT_URL_RE.finditer(text):
        raw = match.group(1).split("?")[0]
        if raw:
            paths.add(_normalize(raw))
    return paths


def test_client_calls_only_paths_the_api_serves() -> None:
    if not _CLIENT_TS.exists():
        pytest.skip("web/ frontend is not present in this checkout")
    schema_paths = {_normalize(p) for p in create_app().openapi()["paths"]}
    frontend_paths = _frontend_paths()
    # Sanity: the extractor found the client's calls, not an empty set.
    assert len(frontend_paths) >= 10
    missing = frontend_paths - schema_paths
    assert not missing, (
        "web/src/api/client.ts calls paths the API does not serve: "
        f"{sorted(missing)}; API serves: {sorted(schema_paths)}"
    )


def test_client_stream_path_is_the_real_sse_route() -> None:
    if not _CLIENT_TS.exists():
        pytest.skip("web/ frontend is not present in this checkout")
    # The exact regression this suite exists for: the SSE subscription path.
    assert "/jobs/{}/stream" in _frontend_paths()


def test_committed_openapi_schema_matches_the_app() -> None:
    if not _OPENAPI_JSON.exists():
        pytest.skip("web/ frontend is not present in this checkout")
    committed = json.loads(_OPENAPI_JSON.read_text(encoding="utf-8"))
    current = create_app().openapi()
    assert set(committed.get("paths", {})) == set(current.get("paths", {})), (
        "web/openapi.json is stale — rerun scripts/generate_api_types.py"
    )


def test_deliverable_paths_match_the_export_stage_naming(tmp_path: Path) -> None:
    output = tmp_path / "book_bilingual.pdf"
    paths = _deliverable_paths(output)
    assert paths["primary"] == output
    assert paths["rigid"] == output.with_name("book_bilingual_rigid.pdf")
    assert paths["epub"] == companion_path(output, ".epub")
    assert paths["contract"] == sidecar_path(output, "contract.json")
    assert paths["quality_report"] == sidecar_path(output, "quality_report.json")
    assert set(paths) <= set(DELIVERABLE_LABELS)


def test_glossary_helpers_roundtrip_a_json_file(tmp_path: Path) -> None:
    path = tmp_path / "terms.json"
    path.write_text(
        json.dumps({"Backpropagation": "反向传播"}, ensure_ascii=False), encoding="utf-8"
    )

    add_glossary_term(path, "Attention", "注意力")
    assert read_glossary_terms(path) == [
        {"source": "Backpropagation", "target": "反向传播"},
        {"source": "Attention", "target": "注意力"},
    ]

    # Upsert updates in place rather than duplicating.
    add_glossary_term(path, "Attention", "注意力机制")
    assert {"source": "Attention", "target": "注意力机制"} in read_glossary_terms(path)

    assert remove_glossary_term(path, "Backpropagation") is True
    assert read_glossary_terms(path) == [{"source": "Attention", "target": "注意力机制"}]
    assert remove_glossary_term(path, "Missing") is False


def test_glossary_helpers_roundtrip_a_csv_file(tmp_path: Path) -> None:
    path = tmp_path / "terms.csv"
    path.write_text("source,translation\nBackpropagation,反向传播\n", encoding="utf-8")

    add_glossary_term(path, "Attention", "注意力")
    assert read_glossary_terms(path) == [
        {"source": "Backpropagation", "target": "反向传播"},
        {"source": "Attention", "target": "注意力"},
    ]
    assert remove_glossary_term(path, "Backpropagation") is True
    assert read_glossary_terms(path) == [{"source": "Attention", "target": "注意力"}]


_API_KEY = "console-test-key"
_AUTH = {"X-API-Key": _API_KEY}


def _config(tmp_path: Path, **overrides: Any) -> UBTConfig:
    return UBTConfig(
        db_dir=tmp_path / "db",
        allowed_dirs=str(tmp_path),
        service_api_key=SecretStr(_API_KEY),
        **overrides,
    )


def test_asset_endpoints_roundtrip_over_http(tmp_path: Path) -> None:
    glossary = tmp_path / "terms.json"
    glossary.write_text(
        json.dumps({"Backpropagation": "反向传播"}, ensure_ascii=False), encoding="utf-8"
    )
    client = TestClient(create_app(_config(tmp_path, glossary_path=glossary)))

    listed = client.get("/assets/glossary", headers=_AUTH)
    assert listed.status_code == 200
    assert listed.json()["terms"] == [{"source": "Backpropagation", "target": "反向传播"}]

    added = client.post(
        "/assets/glossary", json={"source": "Attention", "target": "注意力"}, headers=_AUTH
    )
    assert added.status_code == 200
    assert {"source": "Attention", "target": "注意力"} in added.json()["terms"]

    removed = client.delete("/assets/glossary", params={"source": "Backpropagation"}, headers=_AUTH)
    assert removed.status_code == 200
    assert removed.json()["terms"] == [{"source": "Attention", "target": "注意力"}]


def test_glossary_endpoint_reports_when_unconfigured(tmp_path: Path) -> None:
    client = TestClient(create_app(_config(tmp_path)))
    assert client.get("/assets/glossary", headers=_AUTH).status_code == 409


def test_deliverables_endpoint_serves_only_existing_files(tmp_path: Path) -> None:
    from ubt.core.engine.ledger import SQLiteJobLedger
    from ubt.core.ir.models import BookManifest

    output = tmp_path / "book_bilingual.pdf"
    output.write_bytes(b"%PDF-1.4\n")
    contract = sidecar_path(output, "contract.json")
    contract.write_text("{}", encoding="utf-8")

    cfg = _config(tmp_path)
    job_id = "consolejob001"
    cfg.db_dir.mkdir(parents=True, exist_ok=True)
    # Seed the job's artifact pointer the same way a finished run would: an
    # initialized job_meta row plus the persisted output_file path.
    with SQLiteJobLedger(cfg.db_dir / f"{job_id}.sqlite") as ledger:
        ledger.init_job_from_manifest(
            job_id,
            BookManifest(doc_id=job_id, title="t", source_path=str(output)),
        )
        ledger.set_job_metadata_value(job_id, "output_file", str(output))

    client = TestClient(create_app(cfg))
    listed = client.get(f"/jobs/{job_id}/deliverables", headers=_AUTH)
    assert listed.status_code == 200
    keys = {item["key"] for item in listed.json()["deliverables"]}
    assert keys == {"primary", "contract"}

    downloaded = client.get(f"/jobs/{job_id}/download/contract", headers=_AUTH)
    assert downloaded.status_code == 200
    assert downloaded.content == b"{}"

    assert client.get(f"/jobs/{job_id}/download/bogus", headers=_AUTH).status_code == 404
