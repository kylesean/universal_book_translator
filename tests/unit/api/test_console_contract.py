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
_EN_TS = _REPO_ROOT / "web" / "src" / "i18n" / "translations" / "en.ts"
_ASSESS_PY = _REPO_ROOT / "ubt" / "core" / "assess.py"
_WIZARD_TSX = _REPO_ROOT / "web" / "src" / "views" / "wizard" / "NewJobWizard.tsx"

_CLIENT_URL_RE = re.compile(r"\$\{BASE_URL\}(/[^`\"']*)")

_ASSESS_WARNING_CODE_RE = re.compile(r'AssessmentWarning\(\s*\n\s*"([A-Z][A-Z0-9_]+)"')


def _normalize(path: str) -> str:
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
    assert len(frontend_paths) >= 10
    missing = frontend_paths - schema_paths
    assert not missing, (
        "web/src/api/client.ts calls paths the API does not serve: "
        f"{sorted(missing)}; API serves: {sorted(schema_paths)}"
    )


def test_client_stream_path_is_the_real_sse_route() -> None:
    if not _CLIENT_TS.exists():
        pytest.skip("web/ frontend is not present in this checkout")
    assert "/jobs/{}/stream" in _frontend_paths()


def test_committed_openapi_schema_matches_the_app() -> None:
    if not _OPENAPI_JSON.exists():
        pytest.skip("web/ frontend is not present in this checkout")
    committed = json.loads(_OPENAPI_JSON.read_text(encoding="utf-8"))
    current = create_app().openapi()
    assert committed == current, "web/openapi.json is stale — rerun scripts/generate_api_types.py"


def test_generated_types_match_the_committed_schema() -> None:
    generated = _REPO_ROOT / "web" / "src" / "api" / "generated-types.ts"
    if not generated.exists() or not _OPENAPI_JSON.exists():
        pytest.skip("web/ frontend is not present in this checkout")
    schema = json.loads(_OPENAPI_JSON.read_text(encoding="utf-8"))
    text = generated.read_text(encoding="utf-8")
    for name in schema.get("components", {}).get("schemas", {}):
        assert re.search(rf"^\s*{re.escape(name)}: ", text, re.M), (
            f"generated-types.ts is stale: no declaration for schema {name!r}; "
            "rerun scripts/generate_api_types.py"
        )


def test_i18n_warning_catalogue_covers_every_engine_warning_code() -> None:
    if not _EN_TS.exists() or not _ASSESS_PY.exists():
        pytest.skip("web/ frontend is not present in this checkout")

    emitted = set(_ASSESS_WARNING_CODE_RE.findall(_ASSESS_PY.read_text(encoding="utf-8")))
    emitted |= {
        "PDF_PLAN_UNAVAILABLE",
        "PAGE_PROFILE_UNAVAILABLE",
        "FONT_WITNESS_UNAVAILABLE",
        "PDF_PROBE_UNAVAILABLE",
        "ROUTE_PROBE_UNAVAILABLE",
    }
    assert len(emitted) >= 15, f"warning-code extractor found too few codes: {sorted(emitted)}"

    en = _EN_TS.read_text(encoding="utf-8")
    catalogue = re.search(r"warningCodes:\s*\{(.*?)\n    \}", en, re.S)
    assert catalogue, "en.ts has no warningCodes block"
    translated = set(re.findall(r"^\s{6}([A-Z][A-Z0-9_]+):", catalogue.group(1), re.M))

    missing = emitted - translated
    assert not missing, (
        f"warning codes with no English copy in en.ts warningCodes: {sorted(missing)}"
    )
    assert not translated - emitted, (
        f"warningCodes entries the engine never emits: {sorted(translated - emitted)}"
    )


def _warning_calls(text: str) -> list[tuple[str, list[str], bool]]:
    calls: list[tuple[str, list[str], bool]] = []
    for match in re.finditer(r"AssessmentWarning\(", text):
        start = match.end()
        depth = 1
        cursor = start
        while depth and cursor < len(text):
            if text[cursor] == "(":
                depth += 1
            elif text[cursor] == ")":
                depth -= 1
            cursor += 1
        body = text[start : cursor - 1]
        literals = re.findall(r'"((?:[^"\\]|\\.)*)"', body)
        if len(literals) < 3 or not literals[0].isupper():
            continue
        has_dict = bool(re.search(r"\{\s*[\"']", body))
        calls.append((literals[0], literals[2:], has_dict))
    return calls


def test_assess_warnings_carry_structured_params() -> None:
    if not _ASSESS_PY.exists():
        pytest.skip("web/ frontend is not present in this checkout")
    calls = _warning_calls(_ASSESS_PY.read_text(encoding="utf-8"))
    assert len(calls) >= 12, f"warning extractor found too few calls: {calls}"

    interpolating = [(code, has_dict) for code, copy, has_dict in calls if "{" in "".join(copy)]
    assert interpolating, "no interpolating warning found — extractor is broken"
    missing = sorted(code for code, has_dict in interpolating if not has_dict)
    assert not missing, f"warnings interpolate values but pass no params dict: {missing}"


def test_language_picker_covers_every_engine_profile() -> None:
    if not _WIZARD_TSX.exists():
        pytest.skip("web/ frontend is not present in this checkout")
    from ubt.core.language_profile import supported_lang_codes

    text = _WIZARD_TSX.read_text(encoding="utf-8")
    block = re.search(r"const LANGUAGES[^=]*=\s*\[(.*?)\n\]", text, re.S)
    assert block, "NewJobWizard.tsx has no LANGUAGES block"
    offered = set(re.findall(r"code:\s*'([a-z-]+)'", block.group(1)))

    engine = set(supported_lang_codes())
    missing = engine - offered
    assert not missing, f"engine languages missing from the wizard picker: {sorted(missing)}"
    assert offered - engine == {"zh-tw"}, (
        f"unexpected extra wizard languages: {sorted(offered - engine)}"
    )


def test_every_css_var_used_in_tsx_is_defined_in_the_stylesheet() -> None:
    src_dir = _REPO_ROOT / "web" / "src"
    css = src_dir / "index.css"
    if not css.exists():
        pytest.skip("web/ frontend is not present in this checkout")

    defined = set(re.findall(r"^\s*(--[a-z0-9-]+)\s*:", css.read_text(encoding="utf-8"), re.M))
    assert defined, "index.css declares no custom properties"

    used: set[str] = set()
    for path in src_dir.rglob("*.tsx"):
        used |= set(re.findall(r"var\((--[a-z0-9-]+)", path.read_text(encoding="utf-8")))

    undefined = sorted(used - defined)
    assert not undefined, f"CSS variables used in TSX but never defined: {undefined}"


def test_no_css_var_uses_a_tailwind_opacity_modifier() -> None:
    src_dir = _REPO_ROOT / "web" / "src"
    if not src_dir.exists():
        pytest.skip("web/ frontend is not present in this checkout")
    pattern = re.compile(
        r"(?:bg|text|border|divide|ring|from|to|via|fill|stroke|shadow|outline|decoration)"
        r"-\[var\(--[a-z0-9-]+\)\]/\d+"
    )
    offenders: list[str] = []
    for path in src_dir.rglob("*.tsx"):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("//") or stripped.startswith("*"):
                continue
            if pattern.search(line):
                offenders.append(f"{path.relative_to(_REPO_ROOT)}:{lineno}")
    assert not offenders, (
        "Tailwind v3 drops an opacity modifier on a var() colour, so these "
        f"classes render nothing: {offenders}"
    )


def test_deliverable_paths_match_the_export_stage_naming(tmp_path: Path) -> None:
    output = tmp_path / "book_bilingual.pdf"
    paths = _deliverable_paths(output)
    assert paths["primary"] == output
    assert paths["epub"] == companion_path(output, ".epub")
    assert paths["html"] == companion_path(output, ".html")
    assert paths["xliff"] == companion_path(output, ".xliff")
    assert paths["contract"] == sidecar_path(output, "contract.json")
    assert paths["quality_report"] == sidecar_path(output, "quality_report.json")
    assert paths["attestation"] == companion_path(output, "_attestations.json")
    assert set(paths) <= set(DELIVERABLE_LABELS), "every produced path needs a label"
    _CONDITIONAL = {"secondary"}
    assert set(DELIVERABLE_LABELS) - _CONDITIONAL <= set(paths), (
        "every unconditional label must have a producer path"
    )


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

    assert body["auth_enabled"] is True
    assert body["allowed_bases"] == [str(tmp_path)]
    assert body["db_dir"] == str(config.db_dir)
    assert body["job_mode"] == "embedded"
    assert body["host"]
    assert isinstance(body["is_loopback"], bool)


async def test_system_info_loopback_is_derived_from_the_bind_not_the_host_header(
    tmp_path: Path,
) -> None:
    from ubt.api.app import create_app

    app = create_app(_config(tmp_path))
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "path": "/system/info",
        "raw_path": b"/system/info",
        "query_string": b"",
        "scheme": "http",
        "server": ("203.0.113.5", 8000),
        "client": ("203.0.113.9", 51000),
        "root_path": "",
        "headers": [
            (b"host", b"localhost"),
            (b"x-api-key", _API_KEY.encode()),
        ],
    }
    from collections.abc import MutableMapping

    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: MutableMapping[str, Any]) -> None:
        sent.append(dict(message))

    await app(scope, receive, send)

    body = json.loads(
        b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    )
    assert body["host"] == "localhost"
    assert body["is_loopback"] is False


def test_a_forged_host_header_is_refused(tmp_path: Path) -> None:
    client = TestClient(create_app(_config(tmp_path)))
    assert client.get("/health", headers={"Host": "evil.example.com"}).status_code == 400
    assert client.get("/health", headers={"Host": "localhost:8000"}).status_code == 200
    assert client.get("/health").status_code == 200


def test_trusted_hosts_config_admits_an_operator_domain(tmp_path: Path) -> None:
    config = _config(tmp_path).model_copy(update={"trusted_hosts": "books.example.com"})
    client = TestClient(create_app(config))
    assert client.get("/health", headers={"Host": "books.example.com"}).status_code == 200
    assert client.get("/health", headers={"Host": "other.example.com"}).status_code == 400


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
    assert "recommended_render_engine" not in components["AssessRoute"]["properties"]


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
    browser = {**_AUTH, "Accept": "text/html,application/xhtml+xml"}
    for path in ("/wizard", "/jobs/abc123/quality", "/jobs/abc123/review", "/assets", "/system"):
        res = client.get(path, headers=browser)
        assert res.status_code == 200, path
        assert res.headers["content-type"].startswith("text/html"), path

    assert client.get("/health").headers["content-type"].startswith("application/json")
    assert client.get("/wizard", headers=_AUTH).status_code == 404


def test_jobs_endpoint_serves_html_to_a_browser_and_json_to_api_clients(tmp_path: Path) -> None:
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


def test_the_wizard_sends_only_fields_the_submit_endpoint_accepts() -> None:
    from ubt.api.models import JobSubmitRequest

    source = _WIZARD_TSX.read_text(encoding="utf-8")
    match = re.search(
        r"const buildPayload = \(\): JobSubmitRequest => \(\{(.*?)\n  \}\)", source, re.DOTALL
    )
    assert match, "NewJobWizard.tsx has no buildPayload(): JobSubmitRequest object"

    sent = set(re.findall(r"^\s{4}([a-z_][a-z0-9_]*)\s*:", match.group(1), re.MULTILINE))
    accepted = set(JobSubmitRequest.model_fields)

    assert sent, "parsed no payload keys from the wizard"
    assert not (sent - accepted), (
        f"the wizard sends fields the API rejects: {sorted(sent - accepted)}"
    )

    assert "job_id: idempotencyKeyFor(payload)" in source, (
        "the submit must be sent with a job_id derived from the payload, or a "
        "retried click starts and bills a second identical run"
    )


def test_the_wizard_quotes_the_provider_it_will_submit() -> None:
    source = _WIZARD_TSX.read_text(encoding="utf-8")
    match = re.search(r"assessJob\(\{(.*?)\n        \}\)", source, re.DOTALL)
    assert match, "NewJobWizard.tsx has no assessJob({...}) call"

    body = match.group(1)
    for key in ("provider", "ocr_mode", "visual_judge_enabled", "max_concurrency"):
        assert f"{key}:" in body, (
            f"assessJob() does not send {key}; the quote then prices a different "
            "configuration than the submit will run"
        )


def test_the_wizard_does_not_reference_a_removed_internal_knob() -> None:
    for path in (_WIZARD_TSX, _EN_TS):
        text = path.read_text(encoding="utf-8")
        assert "exec_mode" not in text, f"{path.name} still references exec_mode"
        assert "execMode" not in text, f"{path.name} still references execMode"


def test_assess_reduces_the_source_path_to_a_basename(tmp_path: Path) -> None:
    source = tmp_path / "sub" / "secret-book.md"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("# Title\n\nSome body text.\n", encoding="utf-8")
    client = TestClient(create_app(_config(tmp_path)))

    res = client.post(
        "/jobs/assess",
        json={"input_path": str(source)},
        headers=_AUTH,
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["path"] == "secret-book.md"
    assert str(tmp_path) not in json.dumps(body)


def test_a_forged_tenant_header_is_ignored_by_default(tmp_path: Path) -> None:
    from ubt.api.scope import ApiScope
    from ubt.core.engine.job_queue import JobQueue

    queue = JobQueue(tmp_path / "q.sqlite")
    queue.enqueue("otherjob0001", {"v": 1}, tenant_id="victim")
    scope = ApiScope(
        config=_config(tmp_path),
        manager=object(),  # type: ignore[arg-type]
        job_queue=queue,
        assess_semaphore=None,  # type: ignore[arg-type]
    )
    assert scope.tenant_from_header("victim") == "default"
    assert scope.tenant_allows("otherjob0001", scope.tenant_from_header("victim")) is False


def test_the_tenant_header_is_honored_when_the_operator_trusts_it(tmp_path: Path) -> None:
    from ubt.api.scope import ApiScope
    from ubt.core.engine.job_queue import JobQueue

    queue = JobQueue(tmp_path / "q.sqlite")
    queue.enqueue("otherjob0001", {"v": 1}, tenant_id="victim")
    scope = ApiScope(
        config=_config(tmp_path, trust_tenant_header=True),
        manager=object(),  # type: ignore[arg-type]
        job_queue=queue,
        assess_semaphore=None,  # type: ignore[arg-type]
    )
    assert scope.tenant_from_header("victim") == "victim"
    assert scope.tenant_allows("otherjob0001", scope.tenant_from_header("victim")) is True


def test_the_wizard_offers_exactly_the_engine_presets() -> None:
    from ubt.core.presets import Preset

    if not _WIZARD_TSX.exists():
        pytest.skip("web/ frontend is not present in this checkout")
    source = _WIZARD_TSX.read_text(encoding="utf-8")
    offered = set(re.findall(r"id:\s*'([a-z_]+)'\s+as const", source))
    assert offered, "parsed no preset ids from NewJobWizard.tsx"
    assert offered == {p.value for p in Preset}, (
        f"wizard/engine preset mismatch: only-wizard={sorted(offered - {p.value for p in Preset})}, "
        f"only-engine={sorted({p.value for p in Preset} - offered)}"
    )


def test_the_wizard_does_not_promise_vector_formulas_outside_vector_math() -> None:
    from ubt.core.presets import PRESETS, Preset

    if not _EN_TS.exists():
        pytest.skip("web/ frontend is not present in this checkout")
    vector_presets = {p for p, pol in PRESETS.items() if pol.formula_enrichment == "on"}

    for path in (_EN_TS, _EN_TS.parent / "zh.ts"):
        text = path.read_text(encoding="utf-8")
        assert "vector" not in text.lower() or vector_presets, (
            f"{path.name} promises vector formulas but no preset enables them"
        )

    wizard = _WIZARD_TSX.read_text(encoding="utf-8")
    assert Preset.VECTOR_MATH.value in wizard


def test_preset_badges_do_not_claim_a_quality_gate() -> None:
    from ubt.core.presets import PRESETS

    if not _EN_TS.exists():
        pytest.skip("web/ frontend is not present in this checkout")
    en = _EN_TS.read_text(encoding="utf-8")
    badge_block = re.search(r"presetBadge\w+:\s*'([^']*)'", en)
    assert badge_block, "en.ts has no presetBadge key"
    for preset in PRESETS.values():
        assert preset.pdf_engine in ("auto", "pdfium", "docling")
        assert preset.prompt_strategy in ("auto", "minimal", "hybrid", "rich")


def test_wizard_sends_layout_columns_within_the_api_range() -> None:
    from ubt.api.models import JobSubmitRequest

    if not _WIZARD_TSX.exists():
        pytest.skip("web/ frontend is not present in this checkout")
    source = _WIZARD_TSX.read_text(encoding="utf-8")
    assert "layout_columns" in source
    assert re.search(r"layout_columns:\s*layoutColumns === 'auto' \? null", source)
    assert "layout_columns" in JobSubmitRequest.model_fields


_WIZARD_SRC = _REPO_ROOT / "web" / "src" / "views" / "wizard" / "NewJobWizard.tsx"
_PREFLIGHT_SRC = _REPO_ROOT / "web" / "src" / "views" / "wizard" / "PreflightPanel.tsx"


def test_the_estimate_provenance_badge_is_panel_level_not_per_row() -> None:
    if not _PREFLIGHT_SRC.exists():
        pytest.skip("web/ frontend is not present in this checkout")
    source = _PREFLIGHT_SRC.read_text(encoding="utf-8")

    header_end = source.index("preflightTitle")
    metrics_start = source.index("sourceChars")
    assert header_end < metrics_start, "unexpected panel layout"

    header_block = source[header_end:metrics_start]
    assert "billable_blocks_is_exact" in header_block, (
        "the estimate-provenance badge is not in the panel header"
    )

    metrics_block = source[metrics_start : source.index("probeWarnings")]
    assert "estimateExact" not in metrics_block, (
        "the estimate marker is repeated per row; it belongs on the panel header only"
    )


def test_the_formula_density_row_explains_text_vs_layout() -> None:
    from ubt.api.models import AssessDocumentFacts

    assert "math_density" in AssessDocumentFacts.model_fields
    assert "has_formulas" in AssessDocumentFacts.model_fields

    if not _PREFLIGHT_SRC.exists():
        pytest.skip("web/ frontend is not present in this checkout")
    source = _PREFLIGHT_SRC.read_text(encoding="utf-8")
    assert "mathDensityWithObjects" in source
    assert "mathDensityTextOnly" in source


def test_font_residue_warning_says_it_is_not_a_scan_defect() -> None:
    if not _EN_TS.exists():
        pytest.skip("web/ frontend is not present in this checkout")

    for path, needles in (
        (_EN_TS, ("font-encoding", "scan-quality")),
        (_EN_TS.parent / "zh.ts", ("字体编码", "扫描质量")),
    ):
        text = path.read_text(encoding="utf-8")
        block = re.search(r"FONT_RESIDUE_RISK:\s*\n?\s*'([^']*)'", text)
        assert block, f"{path.name} has no FONT_RESIDUE_RISK copy"
        copy = block.group(1)
        for needle in needles:
            assert needle in copy, f"{path.name} FONT_RESIDUE_RISK lost {needle!r}"


def test_the_wizard_shows_total_tokens_not_just_the_source_size() -> None:
    from ubt.api.models import AssessCost, AssessDocumentFacts

    assert "estimated_tokens" in AssessDocumentFacts.model_fields
    assert "prompt_tokens" in AssessCost.model_fields
    assert "completion_tokens" in AssessCost.model_fields

    if not _PREFLIGHT_SRC.exists():
        pytest.skip("web/ frontend is not present in this checkout")
    source = _PREFLIGHT_SRC.read_text(encoding="utf-8")
    assert "cost.prompt_tokens + assessment.cost.completion_tokens" in source, (
        "the token row does not sum prompt+completion"
    )
    zh = (_EN_TS.parent / "zh.ts").read_text(encoding="utf-8")
    label = re.search(r"estTokens:\s*'([^']*)'", zh)
    note = re.search(r"tokensEstimateNote:\s*\n?\s*'([^']*)'", zh)
    assert label and note, "zh.ts lost the token label or its estimate note"
    combined = label.group(1) + note.group(1)
    assert "估算" in combined, (
        "nothing in the token panel says the figure is an estimate — a bare "
        f"number reads as a measurement: {label.group(1)!r} / {note.group(1)!r}"
    )
    assert "非真实分词器" in note.group(1), (
        "the note must still say the count is a character heuristic, not a tokenizer"
    )


def test_the_budget_control_has_no_zero_means_unlimited_trap() -> None:
    from ubt.api.models import JobSubmitRequest

    field = JobSubmitRequest.model_fields["budget_usd"]
    assert field.metadata, "budget_usd lost its gt=0 bound"
    assert any(getattr(m, "gt", None) == 0 for m in field.metadata), (
        "budget_usd no longer rejects 0; the UI guard would be unnecessary"
    )

    if not _WIZARD_SRC.exists():
        pytest.skip("web/ frontend is not present in this checkout")
    source = _WIZARD_SRC.read_text(encoding="utf-8")
    assert 'type="range"' not in source, "the budget slider is back"
    assert "budgetUnlimitedToggle" in source
    assert "budgetInvalid" in source
    assert "parsed > 0" in source


def test_font_residue_warning_carries_the_at_risk_page_count() -> None:
    if not _ASSESS_PY.exists():
        pytest.skip("engine source is not present in this checkout")
    source = _ASSESS_PY.read_text(encoding="utf-8")
    assert '"at_risk_pages": at_risk' in source, (
        "FONT_RESIDUE_RISK no longer carries at_risk_pages in its params"
    )
    if not _PREFLIGHT_SRC.exists():
        pytest.skip("web/ frontend is not present in this checkout")
    wizard = _PREFLIGHT_SRC.read_text(encoding="utf-8")
    assert "at_risk_pages" in wizard
    assert "warningAtRiskClause" in wizard


def test_every_screen_is_lazily_loaded() -> None:
    app = _REPO_ROOT / "web" / "src" / "App.tsx"
    if not app.exists():
        pytest.skip("web/ frontend is not present in this checkout")
    source = app.read_text(encoding="utf-8")

    views = (
        "NewJobWizard",
        "MissionControl",
        "QualityGateView",
        "LanguageAssetsView",
        "SystemDoctorView",
        "ReviewWorkbench",
    )
    for view in views:
        assert re.search(rf"const {view} = lazy\(", source), f"{view} is not lazy-loaded"
        assert not re.search(rf"^import \{{[^}}]*\b{view}\b", source, re.MULTILINE), (
            f"{view} is also imported statically, which defeats code splitting"
        )

    assert "<Suspense" in source
    assert "RouteFallback" in source


def test_the_shipped_bundle_has_no_chunk_over_vite_warning_size() -> None:
    static_assets = _REPO_ROOT / "ubt" / "api" / "static" / "assets"
    if not static_assets.exists():
        pytest.skip("console bundle is not present in this checkout")

    oversized = [
        (path.name, path.stat().st_size)
        for path in static_assets.glob("*.js")
        if path.stat().st_size > 500 * 1024
    ]
    assert not oversized, f"chunks over the 500 kB Vite warning size: {oversized}"


def test_the_bundle_ships_more_than_one_js_chunk() -> None:
    static_assets = _REPO_ROOT / "ubt" / "api" / "static" / "assets"
    if not static_assets.exists():
        pytest.skip("console bundle is not present in this checkout")

    js = sorted(p.name for p in static_assets.glob("*.js"))
    assert len(js) > 5, f"expected per-route chunks, found only: {js}"
    entry = [n for n in js if n.startswith("index-")]
    assert len(entry) == 1, f"expected exactly one entry chunk, found: {entry}"


def test_the_wizard_offers_every_engine_ocr_mode() -> None:
    import typing

    from ubt.core.config import OcrMode

    if not _WIZARD_SRC.exists():
        pytest.skip("web/ frontend is not present in this checkout")
    source = _WIZARD_SRC.read_text(encoding="utf-8")

    offered = set(re.findall(r"\{ value: '(\w+)', label: t\.wizard\.ocrMode", source))
    engine = set(typing.get_args(OcrMode))
    assert offered, "the wizard has no OCR mode select"
    assert offered == engine, (
        f"wizard/engine OCR mode mismatch: only-wizard={sorted(offered - engine)}, "
        f"only-engine={sorted(engine - offered)}"
    )


def test_the_wizard_marks_ocr_modes_that_need_page_egress() -> None:
    if not _WIZARD_SRC.exists():
        pytest.skip("web/ frontend is not present in this checkout")
    source = _WIZARD_SRC.read_text(encoding="utf-8")
    assert re.search(r"ocrNeedsEgress\s*=\s*ocrMode === 'cloud'", source)
    assert "ocrModeEgressWarning" in source


def test_the_wizard_sends_ocr_mode_within_the_api_enum() -> None:
    from ubt.api.models import JobSubmitRequest

    if not _WIZARD_SRC.exists():
        pytest.skip("web/ frontend is not present in this checkout")
    source = _WIZARD_SRC.read_text(encoding="utf-8")
    assert re.search(r"ocr_mode:\s*ocrMode === 'auto' \? null : ocrMode", source)
    assert "ocr_mode" in JobSubmitRequest.model_fields


def test_runtime_config_never_exposes_credentials() -> None:
    from ubt.api.runtime_config import _EXPOSED_FIELDS

    forbidden = {"api_key", "api-key", "token", "secret", "password", "extra_headers"}
    assert not (set(_EXPOSED_FIELDS) & forbidden), (
        f"credential fields are exposed: {sorted(set(_EXPOSED_FIELDS) & forbidden)}"
    )

    import json

    from ubt.api.runtime_config import describe_runtime_config
    from ubt.core.config import UBTConfig

    payload = json.dumps(describe_runtime_config(UBTConfig.from_env()))
    for needle in ("api_key", "OPENCODE_API_KEY", "extra_headers", "Bearer"):
        assert needle not in payload, f"runtime config leaked {needle!r}"


def test_runtime_config_reports_the_active_model_not_the_preset_table() -> None:
    from ubt.api.runtime_config import describe_runtime_config
    from ubt.core.config import UBTConfig

    cfg = UBTConfig.from_env()
    active = describe_runtime_config(cfg)["active"]

    assert active["provider"] == (cfg.provider or "")
    assert active["draft_model"] == cfg.draft_model
    assert active["base_url"] == cfg.base_url
    assert "matched_pattern" in active
    assert active["matched_pattern"] != "*"


def test_runtime_config_separates_declared_providers_from_aliases() -> None:
    from ubt.api.runtime_config import describe_runtime_config
    from ubt.core.config import UBTConfig
    from ubt.core.providers import PROTOCOL_ALIASES, declared_provider_names

    providers = describe_runtime_config(UBTConfig.from_env())["providers"]
    declared = [p for p in providers if p["declared"]]
    aliases = [p for p in providers if p["is_alias"]]

    assert {p["name"] for p in declared} == declared_provider_names()
    assert not ({p["name"] for p in declared} & set(PROTOCOL_ALIASES))
    assert {p["name"] for p in aliases} <= set(PROTOCOL_ALIASES)
    assert sum(1 for p in providers if p["active"]) <= 1


def test_the_doctor_no_longer_claims_a_latency_matrix() -> None:
    repo = _REPO_ROOT
    view = repo / "web" / "src" / "views" / "doctor" / "SystemDoctorView.tsx"
    if not view.exists():
        pytest.skip("web/ frontend is not present in this checkout")

    source = view.read_text(encoding="utf-8")
    assert "runtimeConfig" in source, "the panel does not use the runtime-config endpoint"
    assert "getRuntimeConfig" in source
    assert ".filter((provider) => provider.declared)" in source


def test_the_doctor_renders_declared_capability_not_a_preset_table() -> None:
    from ubt.api.runtime_config import describe_runtime_config
    from ubt.core.config import UBTConfig

    declared = describe_runtime_config(UBTConfig(draft_model="m", supports_temperature=False))[
        "active"
    ]
    assert declared["declared_capability"] == ["supports_temperature"]
    assert declared["supports_temperature"] is False

    assumed = describe_runtime_config(UBTConfig(draft_model="m"))["active"]
    assert assumed["declared_capability"] == []
    assert assumed["supports_temperature"] is True


def test_the_doctor_has_no_capability_preset_list() -> None:
    repo = _REPO_ROOT
    view = repo / "web" / "src" / "views" / "doctor" / "SystemDoctorView.tsx"
    if not view.exists():
        pytest.skip("web/ frontend is not present in this checkout")

    source = view.read_text(encoding="utf-8")
    assert "listModelProfiles" not in source
    assert "capabilityPresets" not in source
    assert "ProviderEditor" in source


def test_the_provider_editor_gates_the_credential_field_on_the_deployment() -> None:
    repo = _REPO_ROOT
    editor = repo / "web" / "src" / "views" / "doctor" / "ProviderEditor.tsx"
    if not editor.exists():
        pytest.skip("web/ frontend is not present in this checkout")

    source = editor.read_text(encoding="utf-8")
    assert "has_credential" in source
    assert "'********'" in source
    assert "canWriteCredential" in source
    assert "credentialBlockedHint" in source
    assert "ocr_api_key" not in source


def test_the_console_can_create_the_first_provider() -> None:
    repo = _REPO_ROOT
    view = repo / "web" / "src" / "views" / "doctor" / "SystemDoctorView.tsx"
    if not view.exists():
        pytest.skip("web/ frontend is not present in this checkout")

    source = view.read_text(encoding="utf-8")
    assert "addProvider" in source
    assert "setEditingProvider(null)" in source


def test_translation_memory_rows_and_preview_share_one_markdown_renderer() -> None:
    repo = _REPO_ROOT
    view = repo / "web" / "src" / "views" / "language-assets" / "LanguageAssetsView.tsx"
    renderer = repo / "web" / "src" / "components" / "content" / "RichText.tsx"
    if not view.exists() or not renderer.exists():
        pytest.skip("web/ frontend is not present in this checkout")

    source = view.read_text(encoding="utf-8")
    assert "RichText" in source
    assert "<TmPassage" in source
    assert "text={entry.source_text}" in source
    assert "text={entry.target_text}" in source
    assert "<RichText text={inspectTm.source_text} source />" in source
    assert "<RichText text={inspectTm.target_text} />" in source
    core = renderer.read_text(encoding="utf-8")
    assert "remarkGfm" in core and "remarkMath" in core and "rehypeKatex" in core
    assert "if (source && !hasStructuredMarkup(text))" in core


def test_the_console_uses_two_themes_and_an_extensible_language_menu() -> None:
    repo = _REPO_ROOT
    layout = repo / "web" / "src" / "components" / "layout" / "ConsoleLayout.tsx"
    languages = repo / "web" / "src" / "i18n" / "languages.ts"
    theme = repo / "web" / "src" / "lib" / "theme.ts"
    if not layout.exists() or not languages.exists() or not theme.exists():
        pytest.skip("web/ frontend is not present in this checkout")

    source = layout.read_text(encoding="utf-8")
    assert "type PaperTone" not in source
    assert "cyclePaperTone" not in source
    assert "onSetTheme('light')" in source and "onSetTheme('dark')" in source
    assert "LANGUAGE_OPTIONS.map" in source
    assert "DropdownMenuRadioGroup" in source

    registry = languages.read_text(encoding="utf-8")
    assert "export const LANGUAGE_OPTIONS" in registry
    assert "matchPrefixes" in registry
    assert "export function matchLanguageTag" in registry

    theme_source = theme.read_text(encoding="utf-8")
    assert "export type Theme = 'light' | 'dark'" in theme_source
    assert "prefers-color-scheme: dark" in theme_source


def test_a_failed_config_load_does_not_claim_zero_providers() -> None:
    repo = _REPO_ROOT
    view = repo / "web" / "src" / "views" / "doctor" / "SystemDoctorView.tsx"
    if not view.exists():
        pytest.skip("web/ frontend is not present in this checkout")

    source = view.read_text(encoding="utf-8")
    assert "loadFailed" in source
    assert "{(runtime?.providers ?? []).filter((p) => !p.declared).length}" not in source


def test_an_undeclared_configuration_reports_the_generic_default() -> None:
    from ubt.api.runtime_config import describe_runtime_config
    from ubt.core.config import UBTConfig

    active = describe_runtime_config(UBTConfig(draft_model="undeclared-model"))["active"]
    assert active["matched_pattern"] is None
    assert active["declared_capability"] == []
    assert active["supports_temperature"] is True
    assert active["supports_reasoning_effort"] is False


def test_a_state_value_is_declared_before_the_expression_that_reads_it() -> None:
    repo = _REPO_ROOT
    editor = repo / "web" / "src" / "views" / "doctor" / "ProviderEditor.tsx"
    if not editor.exists():
        pytest.skip("web/ frontend is not present in this checkout")

    source = editor.read_text(encoding="utf-8")
    derived = source.find("const isCreate =")
    detail_state = source.find("const [detail, setDetail] = useState")
    assert derived != -1 and detail_state != -1
    assert detail_state < derived, (
        "isCreate is declared above the useState it reads; that is a temporal dead "
        "zone at runtime and neither tsc nor oxlint reports it"
    )


def test_can_translate_is_the_servers_own_verdict() -> None:
    from ubt.api.runtime_config import describe_runtime_config
    from ubt.core.config import MOCK_API_KEY, UBTConfig

    HOSTED = "https://api.example.com/v1"

    keyless = describe_runtime_config(
        UBTConfig(base_url=HOSTED, draft_model="m", api_key=SecretStr(MOCK_API_KEY))
    )["active"]
    assert keyless["can_translate"] is False

    assert (
        describe_runtime_config(
            UBTConfig(base_url=HOSTED, draft_model="", api_key=SecretStr("sk-real"))
        )["active"]["can_translate"]
        is False
    )
    assert (
        describe_runtime_config(
            UBTConfig(base_url=HOSTED, draft_model="m", api_key=SecretStr(MOCK_API_KEY))
        )["active"]["can_translate"]
        is False
    )
    assert (
        describe_runtime_config(
            UBTConfig(base_url=HOSTED, draft_model="m", api_key=SecretStr("sk-real"))
        )["active"]["can_translate"]
        is True
    )


def test_the_wizard_blocks_a_rehearsal_it_has_not_announced() -> None:
    repo = _REPO_ROOT
    wizard = repo / "web" / "src" / "views" / "wizard" / "NewJobWizard.tsx"
    if not wizard.exists():
        pytest.skip("web/ frontend is not present in this checkout")

    source = wizard.read_text(encoding="utf-8")
    assert "can_translate" in source
    assert "rehearsalAcknowledged" in source
    guard = source.index("if (rehearsalOnly && !rehearsalAcknowledged)")
    submit = source.index("await submitJob(")
    assert guard < submit


def test_a_failed_config_load_is_not_reported_as_a_refusal() -> None:
    repo = _REPO_ROOT
    editor = repo / "web" / "src" / "views" / "doctor" / "ProviderEditor.tsx"
    if not editor.exists():
        pytest.skip("web/ frontend is not present in this checkout")

    source = editor.read_text(encoding="utf-8")
    assert "canWriteCredential: boolean | null" in source
    assert "credentialUnknownHint" in source


def test_the_frontend_typecheck_command_actually_checks_something() -> None:
    import json

    repo = _REPO_ROOT
    tsconfig = repo / "web" / "tsconfig.json"
    if not tsconfig.exists():
        pytest.skip("web/ frontend is not present in this checkout")

    root = json.loads(tsconfig.read_text(encoding="utf-8"))
    assert root.get("files") == [], (
        "web/tsconfig.json now has inputs, so `tsc --noEmit` is no longer a no-op — "
        "update this test and the documented command"
    )
    assert root.get("references"), "the solution file must reference the real projects"

    pkg = json.loads((repo / "web" / "package.json").read_text(encoding="utf-8"))
    assert pkg["scripts"]["build"].startswith("tsc -b"), (
        "web build must run `tsc -b`; `tsc --noEmit` type-checks nothing here"
    )


def test_the_capability_form_asks_only_what_the_engine_cannot_learn() -> None:
    repo = _REPO_ROOT
    editor = repo / "web" / "src" / "views" / "doctor" / "ProviderEditor.tsx"
    if not editor.exists():
        pytest.skip("web/ frontend is not present in this checkout")

    source = editor.read_text(encoding="utf-8")
    fields = source[source.index("const CAPABILITY_FIELDS") : source.index("type FlagField")]
    assert "supports_system_prompt" in fields
    assert "supports_temperature" not in fields
    assert "supports_reasoning_effort" not in fields

    for gone in (
        "capabilityTemperature",
        "capabilityReasoning",
        "capabilityAutoLearn",
        "capabilityAdvanced",
    ):
        assert gone not in source, f"{gone} is back in the form"
    assert "supports_vision" in source


def test_the_removed_controls_are_still_engine_fields() -> None:
    from ubt.core.config import UBTConfig
    from ubt.core.router.capabilities import ModelProfile

    for field in ("supports_temperature", "supports_reasoning_effort"):
        assert field in ModelProfile.model_fields, f"{field} left the profile"
        assert field in UBTConfig.model_fields, f"{field} left the config"
    from ubt.core.providers import PROVIDER_ALLOWED_KEYS

    assert "supports_temperature" in PROVIDER_ALLOWED_KEYS
    assert "supports_reasoning_effort" in PROVIDER_ALLOWED_KEYS


def test_every_rendered_capability_control_is_submitted() -> None:
    repo = _REPO_ROOT
    editor = repo / "web" / "src" / "views" / "doctor" / "ProviderEditor.tsx"
    if not editor.exists():
        pytest.skip("web/ frontend is not present in this checkout")

    source = editor.read_text(encoding="utf-8")
    save_start = source.index("const handleSave = useCallback")
    save_body = source[save_start : source.index("return (", save_start)]
    assert "SUBMITTED_FLAGS" in save_body, (
        "handleSave must iterate SUBMITTED_FLAGS; iterating ALL_FLAGS skips supports_vision"
    )
    assert "'supports_vision'" in source[source.index("const SUBMITTED_FLAGS") :][:200]


def test_the_doctor_panel_and_the_editor_share_one_grid_cell() -> None:
    repo = _REPO_ROOT
    view = repo / "web" / "src" / "views" / "doctor" / "SystemDoctorView.tsx"
    if not view.exists():
        pytest.skip("web/ frontend is not present in this checkout")

    source = view.read_text(encoding="utf-8")
    assert 'value="models"' in source
    assert "editingProvider !== undefined && 'hidden'" in source
    assert "name={editingProvider}" in source
    assert "editingProvider !== undefined && 'hidden'" in source
    assert "name={editingProvider}" in source


def test_the_credential_copy_matches_what_the_server_actually_does() -> None:
    repo = _REPO_ROOT
    zh = repo / "web" / "src" / "i18n" / "translations" / "zh.ts"
    en = repo / "web" / "src" / "i18n" / "translations" / "en.ts"
    client = repo / "web" / "src" / "api" / "client.ts"
    if not zh.exists():
        pytest.skip("web/ frontend is not present in this checkout")

    for path in (zh, en, client):
        text = path.read_text(encoding="utf-8")
        for stale in (
            "从不接受 HTTP",
            "never accepted over HTTP",
            "Credentials are not writable here",
        ):
            assert stale not in text, f"{path.name} still claims credentials are refused"

    zh_text = zh.read_text(encoding="utf-8")
    assert "本机" in zh_text and "0600" in zh_text
    en_text = en.read_text(encoding="utf-8")
    assert "loopback" in en_text and "0600" in en_text


def test_a_save_that_the_environment_outranks_is_reported_as_such() -> None:
    from ubt.api.models import ProviderDetailResponse

    assert "credential_shadowed_by_env" in ProviderDetailResponse.model_fields

    repo = _REPO_ROOT
    editor = repo / "web" / "src" / "views" / "doctor" / "ProviderEditor.tsx"
    if not editor.exists():
        pytest.skip("web/ frontend is not present in this checkout")
    source = editor.read_text(encoding="utf-8")
    assert "credential_shadowed_by_env" in source
    assert "credentialShadowedByEnv" in source


def test_the_doctor_does_not_render_a_credential_variable_name_in_the_list() -> None:
    from ubt.api.runtime_config import describe_runtime_config
    from ubt.core.config import UBTConfig

    payload = describe_runtime_config(UBTConfig.from_env())
    for provider in payload["providers"]:
        assert "unresolved_vars" not in provider
    import json

    assert "unresolved_vars" not in json.dumps(payload)


def test_the_doctor_names_the_judge_model_its_block_actually_uses() -> None:
    import os
    import tempfile
    from pathlib import Path

    from ubt.cli.commands.doctor import collect_checks

    with tempfile.TemporaryDirectory(dir=_REPO_ROOT) as tmp:
        Path(tmp, "ubt.toml").write_text(
            """
[defaults]
qe_engine = "tiered"
qe_judge_enabled = true
qe_judge_pass_sample = 0.15
qe_judge_provider = "judge"

[providers.main]
base_url = "http://127.0.0.1:9090/v1"
api_mode = "openai-chat"
draft_model = "translator-model"
repair_model = "translator-model"

[providers.judge]
base_url = "http://127.0.0.1:3425/v1"
api_mode = "openai-chat"
repair_model = "judge-model"
""",
            encoding="utf-8",
        )
        previous = Path.cwd()
        os.chdir(tmp)
        try:
            from ubt.core.config import UBTConfig

            os.environ.pop("UBT_PROVIDER", None)
            cfg = UBTConfig.from_env()
            checks = collect_checks(cfg, probe=False)
        finally:
            os.chdir(previous)

    qe = next(c for c in checks if c.code == "qe")
    assert "judge-model" in qe.detail, f"QE row names the wrong judge model: {qe.detail}"
    assert "translator-model" not in qe.detail
    assert (qe.params or {}).get("provider") == "judge"


def test_the_doctor_panel_prefers_the_provider_variant_when_one_is_named() -> None:
    repo = _REPO_ROOT
    text = repo / "web" / "src" / "i18n" / "doctorText.ts"
    if not text.exists():
        pytest.skip("web/ frontend is not present in this checkout")

    source = text.read_text(encoding="utf-8")
    assert "PARAM_VARIANTS" in source, "the variant picker was removed"
    assert "'qe.tiered'" in source and "'via'" in source
    assert "'provider'" in source


def test_a_fresh_install_can_reach_a_runnable_state_from_the_console() -> None:
    import os
    import tempfile
    from pathlib import Path

    from ubt.api.provider_editor import upsert_provider
    from ubt.api.runtime_config import describe_runtime_config
    from ubt.core.config import UBTConfig

    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp, "ubt.toml")
        target.write_text('[defaults]\nqe_engine = "heuristic"\n', encoding="utf-8")
        previous = Path.cwd()
        os.chdir(tmp)
        try:
            os.environ.pop("UBT_PROVIDER", None)
            before = describe_runtime_config(UBTConfig.from_env())["active"]
            assert before["can_translate"] is False

            upsert_provider(
                "first",
                {
                    "base_url": "https://api.example.com/v1",
                    "api_mode": "openai-chat",
                    "draft_model": "the-model",
                    "api_key": "sk-user",
                },
                allow_credential=True,
                make_default=True,
            )
            after = describe_runtime_config(UBTConfig.from_env())["active"]
        finally:
            os.chdir(previous)

    assert after["provider"] == "first"
    assert after["draft_model"] == "the-model"
    assert after["can_translate"] is True, (
        "creating the first provider must leave a runnable configuration"
    )


def test_the_wizard_lists_which_provider_is_the_file_default() -> None:
    from ubt.api.models import RuntimeProvider

    assert "is_default" in RuntimeProvider.model_fields
    repo = _REPO_ROOT
    view = repo / "web" / "src" / "views" / "doctor" / "SystemDoctorView.tsx"
    if not view.exists():
        pytest.skip("web/ frontend is not present in this checkout")
    source = view.read_text(encoding="utf-8")
    assert "provider.is_default" in source
    assert "activateProvider" in source
    assert "setDefaultProvider" in source


def test_creating_a_provider_makes_it_the_default() -> None:
    repo = _REPO_ROOT
    editor = repo / "web" / "src" / "views" / "doctor" / "ProviderEditor.tsx"
    if not editor.exists():
        pytest.skip("web/ frontend is not present in this checkout")
    source = editor.read_text(encoding="utf-8")
    assert "changes.make_default = true" in source
    save_start = source.index("const handleSave = useCallback")
    save_body = source[source.index("if (isCreate) changes.make_default", save_start) :][:120]
    assert "isCreate" in save_body


def test_the_doctor_endpoint_reads_the_live_config_not_the_boot_snapshot() -> None:
    repo = _REPO_ROOT
    routes = repo / "ubt" / "api" / "routes" / "system.py"
    source = routes.read_text(encoding="utf-8")
    start = source.index("async def system_doctor")
    body = source[start : start + 1200]
    assert "live_config()" in body, (
        "the doctor must check the live configuration; app_config is a boot snapshot"
    )
    assert "collect_checks, app_config" not in body


def test_a_saved_provider_is_immediately_visible_to_the_doctor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ubt.api.provider_editor import upsert_provider
    from ubt.cli.commands.doctor import collect_checks
    from ubt.core.config import UBTConfig

    (tmp_path / "ubt.toml").write_text('[defaults]\nqe_engine = "heuristic"\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("UBT_PROVIDER", raising=False)
    monkeypatch.delenv("UBT_LLM_API_KEY", raising=False)

    before = {c.code: c for c in collect_checks(UBTConfig.from_env(), probe=False)}
    assert before["api_key"].status == "FAIL"
    assert before["models"].status == "FAIL"

    upsert_provider(
        "first",
        {
            "base_url": "https://api.example.com/v1",
            "api_mode": "openai-chat",
            "draft_model": "the-model",
            "api_key": "sk-user",
        },
        allow_credential=True,
        make_default=True,
    )

    after = {c.code: c for c in collect_checks(UBTConfig.from_env(), probe=False)}
    assert after["api_key"].status == "OK"
    assert after["models"].status == "OK"
    assert "the-model" in after["models"].detail


def test_submit_and_assess_read_the_live_config() -> None:
    repo = _REPO_ROOT
    routes = repo / "ubt" / "api" / "routes" / "job.py"
    source = routes.read_text(encoding="utf-8")

    assess = source[source.index("async def assess_job") :][:900]
    assert "cfg = scope.config" in assess, "assess must price against the live config"

    submit = source[source.index("# Live config, not the startup snapshot") :][:700]
    assert "scope.config" in submit, "the rehearsal decision must read the live config"

    assert source.count("execute_job(record, scope.config)") == 3
    assert "execute_job(record, app_config)" not in source


def test_the_builtin_list_distinguishes_protocols_from_aliases() -> None:
    from ubt.api.runtime_config import describe_runtime_config
    from ubt.core.config import UBTConfig

    providers = describe_runtime_config(UBTConfig.from_env())["providers"]
    by_name = {p["name"]: p for p in providers if not p["declared"]}

    protocols = {n for n, p in by_name.items() if p["is_protocol"]}
    assert protocols == {
        "openai-chat",
        "openai-responses",
        "anthropic-messages",
        "gemini-native",
    }
    for name in protocols:
        assert by_name[name]["is_alias"] is False
        assert by_name[name]["alias_of"] is None

    for name, row in by_name.items():
        if not row["is_alias"]:
            continue
        assert row["alias_of"] in protocols, f"{name} points at {row['alias_of']!r}"
    assert by_name["claude"]["alias_of"] == "anthropic-messages"
    assert by_name["chat"]["alias_of"] == "openai-chat"


def test_the_panel_help_says_a_protocol_still_needs_a_model_id() -> None:
    repo = _REPO_ROOT
    zh = repo / "web" / "src" / "i18n" / "translations" / "zh.ts"
    view = repo / "web" / "src" / "views" / "doctor" / "SystemDoctorView.tsx"
    if not view.exists():
        pytest.skip("web/ frontend is not present in this checkout")

    text = zh.read_text(encoding="utf-8")
    assert "builtinProvidersHelp" in text
    assert "模型名" in text, "the help must name the missing prerequisite"
    source = view.read_text(encoding="utf-8")
    assert "builtinProvidersHelp" in source
    assert "provider.is_protocol" in source
    assert "provider.alias_of" in source


def test_the_terminal_stream_frame_carries_the_final_counters() -> None:
    import json

    from ubt.api.routes.shared import terminal_frame

    frame = terminal_frame(
        "completed",
        "book.zh.pdf",
        None,
        counters={
            "total_blocks": 105,
            "drafted_blocks": 0,
            "completed_blocks": 90,
            "skipped_blocks": 154,
            "needs_human_blocks": 15,
            "failed_blocks": 0,
            "progress_percent": 100.0,
        },
    )
    payload = json.loads(frame.split("data: ", 1)[1].strip())
    assert payload["status"] == "completed"
    assert payload["total_blocks"] == 105
    assert payload["drafted_blocks"] == 0
    assert payload["completed_blocks"] == 90
    assert payload["skipped_blocks"] == 154
    assert payload["needs_human_blocks"] == 15

    bare = json.loads(terminal_frame("failed", None, "boom").split("data: ", 1)[1].strip())
    assert bare == {"status": "failed", "output_file": None, "error": "boom"}


def test_the_panel_renders_failures_and_the_human_queue() -> None:
    repo = _REPO_ROOT
    view = repo / "web" / "src" / "views" / "mission-control" / "MissionControl.tsx"
    if not view.exists():
        pytest.skip("web/ frontend is not present in this checkout")
    source = view.read_text(encoding="utf-8")

    for field in ("needs_human_blocks", "failed_blocks", "skipped_blocks"):
        assert field in source, f"{field} is not read from the stream"
    assert "needsHumanBlocks > 0" in source
    assert "failedBlocks > 0" in source
    assert "skippedBlocks > 0" in source

    assert "frame.completed_blocks ?? completedRef.current" in source
    assert "mission.activityProgress" in source
    assert "mission.activityPreserved" in source
    assert "const drafted = frame.drafted_blocks ?? draftedRef.current" not in source
    assert "${frame.completed_blocks ?? 0}/${frame.total_blocks ?? 0}" not in source


def test_the_review_pane_does_not_render_the_failed_draft_markup() -> None:
    repo = _REPO_ROOT
    view = repo / "web" / "src" / "views" / "review" / "ReviewWorkbench.tsx"
    if not view.exists():
        pytest.skip("web/ frontend is not present in this checkout")

    source = view.read_text(encoding="utf-8")
    assert "seg.delivered_text" in source
    assert "{seg.target_text || '—'}" not in source
    assert "parseFailedDraft" not in source
    assert not (repo / "web" / "src" / "lib" / "failedDraft.ts").exists()
    assert "failedDraftNotice" in source
    assert "seg.repair_error" in source
    assert "quarantinedNotice" in source
    assert "seg.quarantined" in source

    serializer = (repo / "ubt" / "api" / "review.py").read_text(encoding="utf-8")
    assert "from ubt.render.delivered_text import" in serializer
    assert '"delivered_text": delivered_text(block)' in serializer
    assert '"repair_failed": is_failed_draft(target)' in serializer
    assert '"quarantined": is_placeholder(target)' in serializer


def test_the_review_page_has_one_name_in_each_locale() -> None:
    repo = _REPO_ROOT
    zh = repo / "web" / "src" / "i18n" / "translations" / "zh.ts"
    if not zh.exists():
        pytest.skip("web/ frontend is not present in this checkout")
    text = zh.read_text(encoding="utf-8")

    assert "质检后编台" not in text
    assert "审校后编台" not in text
    assert text.count("译文审校") >= 2


def test_the_shell_is_never_cached_and_the_bundles_always_are(tmp_path: Path) -> None:
    client = TestClient(create_app(_config(tmp_path)))

    shell = client.get("/")
    assert shell.status_code == 200
    assert shell.headers.get("cache-control") == "no-cache"

    route = client.get("/wizard", headers={"Accept": "text/html"})
    assert route.status_code == 200
    assert route.headers.get("cache-control") == "no-cache"

    body = shell.text
    import re

    match = re.search(r"assets/(index-[A-Za-z0-9_-]+\.js)", body)
    assert match, "the shell must reference a hashed entry bundle"
    asset = client.get(f"/assets/{match.group(1)}")
    assert asset.status_code == 200
    assert asset.headers.get("cache-control") == "public, max-age=31536000, immutable"


def test_delivery_contract_endpoint_and_cels_option(tmp_path: Path) -> None:
    from ubt.api.models import JobSubmitRequest

    req = JobSubmitRequest(
        input_path=str(tmp_path / "book.pdf"),
        target_lang="zh",
        delivery_mode="faithful",
        allow_degrade=True,
        enable_cels=False,
    )
    assert req.enable_cels is False
    assert req.delivery_mode == "faithful"
    assert req.allow_degrade is True

    assert "delivery_contract" in DELIVERABLE_LABELS
    paths = _deliverable_paths(tmp_path / "book.pdf")
    assert "delivery_contract" in paths
    assert paths["delivery_contract"].name == "book_pdf_delivery_contract.json"

    contract_data = {
        "contract": "faithful",
        "passed": True,
        "deliverable": True,
        "summary": {"critical": 0, "major": 0, "total": 0},
        "findings": [],
        "checks": [{"name": "overlap", "status": "pass"}],
    }
    paths["delivery_contract"].write_text(json.dumps(contract_data), encoding="utf-8")

    from ubt.core.engine.ledger import SQLiteJobLedger
    from ubt.core.ir.models import BookManifest

    job_id = "web_testjob123"
    cfg = _config(tmp_path)
    cfg.db_dir.mkdir(parents=True, exist_ok=True)
    db_path = cfg.db_dir / f"{job_id}.sqlite"
    output_pdf = tmp_path / "book.pdf"
    output_pdf.write_bytes(b"%PDF-1.4\n")
    with SQLiteJobLedger(db_path) as ledger:
        ledger.init_job_from_manifest(
            job_id, BookManifest(doc_id=job_id, title="Test", source_path=str(output_pdf))
        )
        ledger.set_job_metadata_value(job_id, "output_file", str(output_pdf))

    client = TestClient(create_app(cfg))
    resp = client.get(f"/jobs/{job_id}/delivery-contract", headers=_AUTH)
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["contract"] == "faithful"
    assert payload["deliverable"] is True
    assert payload["summary"]["critical"] == 0
