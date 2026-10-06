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


def test_system_info_reports_the_live_security_boundary(tmp_path: Path) -> None:
    config = _config(tmp_path)
    client = TestClient(create_app(config))
    body = client.get("/system/info", headers=_AUTH).json()

    # The panel must reflect the running config, not a hardcoded sample.
    assert body["auth_enabled"] is True
    assert body["allowed_bases"] == [str(tmp_path)]
    assert body["db_dir"] == str(config.db_dir)
    assert body["job_mode"] == "embedded"
    assert body["host"]  # the host the request reached
    assert isinstance(body["is_loopback"], bool)


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


def test_assess_response_is_typed_not_a_bare_object() -> None:
    # The wizard's pre-flight panel reads nested fields (document.pages,
    # cost.total_cost_usd, route.recommended_render_engine). While the schema was
    # ``additionalProperties: true`` the generated TS type was an opaque object,
    # so the panel silently read non-existent flat keys and showed fallbacks.
    schema = create_app().openapi()
    body = schema["paths"]["/jobs/assess"]["post"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"]
    assert body["$ref"] == "#/components/schemas/JobAssessResponse"
    components = schema["components"]["schemas"]
    assert {"document", "route", "cost", "runtime"} <= set(
        components["JobAssessResponse"]["properties"]
    )
    assert {"pages", "estimated_tokens", "math_density", "scan_page_share"} <= set(
        components["AssessDocumentFacts"]["properties"]
    )
    assert "total_cost_usd" in components["AssessCost"]["properties"]
    assert "recommended_render_engine" in components["AssessRoute"]["properties"]


def test_jobs_endpoint_lists_ledgers_and_ignores_sidecars(tmp_path: Path) -> None:
    from ubt.core.engine.ledger import SQLiteJobLedger
    from ubt.core.ir.models import (
        BlockStatus,
        BlockType,
        BookManifest,
        ChapterIR,
        IRBlock,
        make_element,
    )
    from ubt.core.memory.tm import TranslationMemory

    cfg = _config(tmp_path)
    cfg.db_dir.mkdir(parents=True, exist_ok=True)

    def _seed(job_id: str, name: str, statuses: list[BlockStatus]) -> None:
        blocks = []
        for index, status in enumerate(statuses):
            element = make_element(
                id=f"b{index:03d}",
                spine_index=index,
                block_type=BlockType.NARRATIVE,
                source_text="s",
            )
            block = IRBlock(element=element)
            block.target_text = "t"
            block.status = status
            blocks.append(block)
        with SQLiteJobLedger(cfg.db_dir / f"{job_id}.sqlite") as ledger:
            ledger.init_job_from_manifest(
                job_id, BookManifest(doc_id=job_id, title="t", source_path=f"/books/{name}")
            )
            ledger.append_chapter(
                job_id,
                ChapterIR(doc_id=job_id, chapter_id="c1", title="C1", spine_index=0, blocks=blocks),
            )
            ledger.set_job_metadata_value(job_id, "estimated_cost_usd", 1.5)
            ledger.finalize_job(job_id, "completed")

    _seed("jobaaa00001", "book_a.pdf", [BlockStatus.MTQE_PASSED, BlockStatus.MTQE_PASSED])
    _seed("jobbbb00002", "book_b.epub", [BlockStatus.MTQE_PASSED, BlockStatus.NEEDS_HUMAN])
    # A shared TM database lives in the same directory and must not be listed.
    TranslationMemory(cfg.db_dir / "tm.sqlite").close()

    client = TestClient(create_app(cfg))
    body = client.get("/jobs", headers=_AUTH).json()
    ids = {item["job_id"] for item in body["jobs"]}
    assert ids == {"jobaaa00001", "jobbbb00002"}

    by_id = {item["job_id"]: item for item in body["jobs"]}
    assert by_id["jobaaa00001"]["file_name"] == "book_a.pdf"
    assert by_id["jobaaa00001"]["progress_percent"] == 100.0
    assert by_id["jobaaa00001"]["estimated_cost_usd"] == 1.5
    assert by_id["jobbbb00002"]["needs_human_blocks"] == 1
    assert by_id["jobbbb00002"]["progress_percent"] == 100.0


def test_spa_deep_links_fall_back_to_index_html(tmp_path: Path) -> None:
    static_index = _REPO_ROOT / "ubt" / "api" / "static" / "index.html"
    if not static_index.exists():
        pytest.skip("the console's built assets are not present in this checkout")

    client = TestClient(create_app(_config(tmp_path)))
    # Client-side routes the server does not serve as API endpoints must land on
    # the SPA shell so a refresh / pasted link resolves, not 404.
    browser = {**_AUTH, "Accept": "text/html,application/xhtml+xml"}
    for path in ("/wizard", "/jobs/abc123/quality", "/jobs/abc123/review", "/assets", "/system"):
        res = client.get(path, headers=browser)
        assert res.status_code == 200, path
        assert res.headers["content-type"].startswith("text/html"), path

    # An API path still answers JSON (the mount never shadows registered routes).
    assert client.get("/health").headers["content-type"].startswith("application/json")
    # A non-browser client probing a missing path gets a JSON 404, not the shell.
    assert client.get("/wizard", headers=_AUTH).status_code == 404


def test_jobs_endpoint_serves_html_to_a_browser_and_json_to_api_clients(tmp_path: Path) -> None:
    # ``/jobs`` is both the API list and the Mission Control URL: a browser
    # navigation (Accept: text/html) gets the SPA shell, an API client the JSON.
    static_index = _REPO_ROOT / "ubt" / "api" / "static" / "index.html"
    if not static_index.exists():
        pytest.skip("the console's built assets are not present in this checkout")

    client = TestClient(create_app(_config(tmp_path)))
    browser = client.get("/jobs", headers={**_AUTH, "Accept": "text/html,application/xhtml+xml"})
    assert browser.status_code == 200
    assert browser.headers["content-type"].startswith("text/html")

    api = client.get("/jobs", headers={**_AUTH, "Accept": "application/json"})
    assert api.status_code == 200
    assert api.headers["content-type"].startswith("application/json")
    assert "jobs" in api.json()
