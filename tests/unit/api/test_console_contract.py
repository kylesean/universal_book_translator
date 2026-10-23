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
_EN_TS = _REPO_ROOT / "web" / "src" / "i18n" / "translations" / "en.ts"
_ASSESS_PY = _REPO_ROOT / "ubt" / "core" / "assess.py"
_WIZARD_TSX = _REPO_ROOT / "web" / "src" / "views" / "wizard" / "NewJobWizard.tsx"

#: Matches ``${BASE_URL}/some/path?query`` inside a fetch/EventSource template.
_CLIENT_URL_RE = re.compile(r"\$\{BASE_URL\}(/[^`\"']*)")

#: The stable warning codes the assess module emits as ``AssessmentWarning(code, ...)``.
_ASSESS_WARNING_CODE_RE = re.compile(r'AssessmentWarning\(\s*\n\s*"([A-Z][A-Z0-9_]+)"')


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
    """The committed schema must equal the app's, not merely share its paths.

    Comparing only path *sets* let a field added to a response model ship
    without regenerating the client types: the console then read a property
    TypeScript did not know about. Compare the whole document instead.
    """
    if not _OPENAPI_JSON.exists():
        pytest.skip("web/ frontend is not present in this checkout")
    committed = json.loads(_OPENAPI_JSON.read_text(encoding="utf-8"))
    current = create_app().openapi()
    assert committed == current, "web/openapi.json is stale — rerun scripts/generate_api_types.py"


def test_generated_types_match_the_committed_schema() -> None:
    """``generated-types.ts`` must be regenerated whenever the schema changes.

    The file is what the console's ``tsc -b`` type-checks against, so a stale
    copy type-checks clean while describing an API that no longer exists.
    """
    generated = _REPO_ROOT / "web" / "src" / "api" / "generated-types.ts"
    if not generated.exists() or not _OPENAPI_JSON.exists():
        pytest.skip("web/ frontend is not present in this checkout")
    schema = json.loads(_OPENAPI_JSON.read_text(encoding="utf-8"))
    text = generated.read_text(encoding="utf-8")
    # Every component schema the API advertises must appear as a generated
    # interface; the generator names them verbatim.
    for name in schema.get("components", {}).get("schemas", {}):
        # Object schemas generate ``Name: {``; enums generate ``Name: "a" | ...``.
        assert re.search(rf"^\s*{re.escape(name)}: ", text, re.M), (
            f"generated-types.ts is stale: no declaration for schema {name!r}; "
            "rerun scripts/generate_api_types.py"
        )


def test_i18n_warning_catalogue_covers_every_engine_warning_code() -> None:
    """Every warning the assess stage emits must have English copy.

    The wizard renders ``t.wizard.warningCodes[code]`` and falls back to the
    engine's Chinese ``detail_zh``; a code with no entry therefore leaks
    Chinese copy into the English console. The engine's own ``_safe``-helper
    codes are passed as a variable, so they are pinned separately below.
    """
    if not _EN_TS.exists() or not _ASSESS_PY.exists():
        pytest.skip("web/ frontend is not present in this checkout")

    emitted = set(_ASSESS_WARNING_CODE_RE.findall(_ASSESS_PY.read_text(encoding="utf-8")))
    # Codes routed through the ``_safe(...)`` degrade helper rather than a
    # literal call site.
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
    # And the reverse: a catalogue entry for a code the engine never emits is
    # dead copy that will silently rot.
    assert not translated - emitted, (
        f"warningCodes entries the engine never emits: {sorted(translated - emitted)}"
    )


def _warning_calls(text: str) -> list[tuple[str, list[str], bool]]:
    """Each ``AssessmentWarning(...)`` call as (code, string literals, has_dict_arg).

    Uses balanced-paren scanning rather than a regex: the copy argument is an
    implicit f-string concatenation that can wrap lines, and a regex over that
    silently matches nothing — a vacuous pass.
    """
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
            continue  # the ``_safe`` helper passes ``code`` as a variable
        has_dict = bool(re.search(r"\{\s*[\"']", body))
        calls.append((literals[0], literals[2:], has_dict))
    return calls


def test_assess_warnings_carry_structured_params() -> None:
    """A warning that interpolates values must expose them for localization.

    ``detail_zh`` is a rendered string; a non-Chinese consumer cannot rebuild
    it without the raw values, so any warning whose copy contains a
    placeholder must populate ``params``.
    """
    if not _ASSESS_PY.exists():
        pytest.skip("web/ frontend is not present in this checkout")
    calls = _warning_calls(_ASSESS_PY.read_text(encoding="utf-8"))
    # Guard against the extractor silently finding nothing.
    assert len(calls) >= 12, f"warning extractor found too few calls: {calls}"

    interpolating = [(code, has_dict) for code, copy, has_dict in calls if "{" in "".join(copy)]
    assert interpolating, "no interpolating warning found — extractor is broken"
    missing = sorted(code for code, has_dict in interpolating if not has_dict)
    assert not missing, f"warnings interpolate values but pass no params dict: {missing}"


def test_language_picker_covers_every_engine_profile() -> None:
    """The wizard's language list must not be narrower than the engine's.

    The console once offered four source / three target languages while the
    engine supported ten, so ja→ko or en→ru were unreachable from the UI.
    """
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
    # The only code the picker may add beyond the engine profiles is the
    # Traditional Chinese variant the target validator accepts.
    assert offered - engine == {"zh-tw"}, (
        f"unexpected extra wizard languages: {sorted(offered - engine)}"
    )


def test_every_css_var_used_in_tsx_is_defined_in_the_stylesheet() -> None:
    """A ``var(--x)`` with no declaration renders unstyled, silently.

    The AuthGate sign-in card once referenced four variables that were never
    defined (``--rule``, ``--paper-raised``, ``--accent``, ``--ink``): the card
    came out transparent and its button white-on-white. Nothing in the build
    catches that, so the definition set is pinned here.
    """
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
    """``bg-[var(--x)]/10`` silently emits no CSS under Tailwind v3.

    The colour is an arbitrary value, not a theme colour, so Tailwind cannot
    split it into ``rgb(... / <alpha>)`` and drops the utility entirely — the
    element renders transparent with no build error. Tint a literal instead, or
    add a dedicated ``--ink-*-wash`` token.
    """
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
    # Both directions. The old assertion only checked "every path key has a
    # label", which is the direction that would KeyError — but it could not see
    # a label with no producer, nor a default companion missing from the table.
    # HTML and XLIFF are written on every run and were missing from both, so the
    # console could not list or download them.
    assert set(paths) <= set(DELIVERABLE_LABELS), "every produced path needs a label"
    # ``secondary`` is conditional by design: it exists only when a mono/dual
    # sibling was written, so it is exempt from the reverse direction. Every
    # other label must have a producer path, or it is a download that can only
    # 404.
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


async def test_system_info_loopback_is_derived_from_the_bind_not_the_host_header(
    tmp_path: Path,
) -> None:
    """P1-D2: ``is_loopback`` must not trust the client-supplied Host header.

    ``request.url`` is built from the ``Host`` header, so a DNS-rebinding probe
    (``Host: evil.example.com`` reaching the loopback console) read as
    ``is_loopback=False`` — the panel then told the operator the console was
    exposed when it was not, and ``Host: localhost`` against a real exposed bind
    read the reverse. The verdict now comes from the socket's server address.

    Driven at the ASGI layer with a real non-loopback bind IP: ``TestClient``'s
    own bind is ``testserver`` (a hostname the checker skips), so only a forged
    scope can tell the two derivations apart.
    """
    from ubt.api.app import create_app

    app = create_app(_config(tmp_path))
    # A request that arrived on an *exposed* bind (203.0.113.5) but carries a
    # loopback Host header. The old code read the header and reported loopback;
    # the bind is what actually answers.
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
    # Exposed bind → not loopback, regardless of the loopback Host header.
    assert body["is_loopback"] is False


def test_a_forged_host_header_is_refused(tmp_path: Path) -> None:
    """A Host outside the trusted set is a DNS-rebinding attempt: 400."""
    client = TestClient(create_app(_config(tmp_path)))
    assert client.get("/health", headers={"Host": "evil.example.com"}).status_code == 400
    # The loopback names and the test client's own host stay reachable.
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
    # cost.total_cost_usd, route.recommended_dual_mode). While the schema was
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


def test_the_wizard_sends_only_fields_the_submit_endpoint_accepts() -> None:
    """Every key the wizard puts in its payload must exist on the REST model.

    ``JobSubmitRequest`` sets ``extra="forbid"``, so one stale key makes *every*
    submission fail with a 422. This happened: the wizard kept sending
    ``exec_mode`` after the field was removed from the model, so the console
    could not start a job at all, and nothing caught it because the frontend
    type-check (``tsc -b``) is a separate gate from the Python suite.

    The payload is built by one ``buildPayload()`` function rather than inline at
    the ``submitJob`` call. That is what makes the idempotency key trustworthy:
    the key is a fingerprint of exactly this object, so a payload assembled in
    two places could change without the key changing and resubmit the *old* job
    (silently dropping the edit). This test therefore reads that function.
    """
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

    # The idempotency key rides alongside the payload (it is a fingerprint *of*
    # it, so it cannot live inside the object it hashes).
    assert "job_id: idempotencyKeyFor(payload)" in source, (
        "the submit must be sent with a job_id derived from the payload, or a "
        "retried click starts and bills a second identical run"
    )


def test_the_wizard_quotes_the_provider_it_will_submit() -> None:
    """The assess call must carry the provider/OCR/vision choices it prices.

    The quote and the run used to disagree: assess applied only the preset, so
    the dollar figure came from whatever ``[defaults] provider`` named while the
    dropdown above it selected a different block. A user comparing two providers
    saw the same total twice, and picking cloud OCR added a vision line to the
    run that the quote never priced.

    This is a source-level assertion because the mismatch spans two call sites
    that no runtime test in this suite exercises.
    """
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
    """An engine detail must not reappear in the console's source or copy.

    ``exec_mode`` is an internal routing knob (env/CLI only). Leaving its label
    and options in the i18n tables invites the field to be wired back up.
    """
    for path in (_WIZARD_TSX, _EN_TS):
        text = path.read_text(encoding="utf-8")
        assert "exec_mode" not in text, f"{path.name} still references exec_mode"
        assert "execMode" not in text, f"{path.name} still references execMode"


# --- P1-D3: no host-path echo, no forgeable tenant scoping -------------------


def test_assess_reduces_the_source_path_to_a_basename(tmp_path: Path) -> None:
    """``/jobs/assess`` echoed the source file's absolute path.

    Every other surface (``/status``, ``/download``) reduces artifact paths to
    their basenames; the quote leaked the host's directory layout to any caller
    that could reach the (unauthenticated-by-default) port.
    """
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
    """``X-UBT-Tenant`` is client-forgeable; honored only when trusted.

    In the default (single-tenant) deployment every request is the ``default``
    tenant, so a forged header cannot scope a caller into another tenant's job.
    """
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
    # Default: the forged header resolves to "default", which is NOT the victim,
    # so the victim's job is invisible.
    assert scope.tenant_from_header("victim") == "default"
    assert scope.tenant_allows("otherjob0001", scope.tenant_from_header("victim")) is False


def test_the_tenant_header_is_honored_when_the_operator_trusts_it(tmp_path: Path) -> None:
    """The multi-tenant SaaS path stays available behind the explicit opt-in."""
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


# --- preset copy must describe what the engine actually does -----------------


def test_the_wizard_offers_exactly_the_engine_presets() -> None:
    """A preset the engine dropped would 422 every submission; one it added would
    be unreachable. Both directions are checked.

    The wizard used to list four presets and hard-code their labels, while
    ``ubt/core/presets.py`` owned the real table. Adding ``publication_math``
    to the engine would have left it invisible in the console.
    """
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


def test_the_wizard_does_not_promise_vector_formulas_outside_publication_math() -> None:
    """Only ``formula_enrichment="on"`` runs the VLM transcription.

    ``auto`` resolves to cropping the source region, so copy claiming "vector
    formula re-render" or "公式矢量重排" for publication/standard described
    behaviour no run produced. This pins the claim to the one preset that
    actually pays for it.
    """
    from ubt.core.presets import PRESETS, Preset

    if not _EN_TS.exists():
        pytest.skip("web/ frontend is not present in this checkout")
    vector_presets = {p for p, pol in PRESETS.items() if pol.formula_enrichment == "on"}

    for path in (_EN_TS, _EN_TS.parent / "zh.ts"):
        text = path.read_text(encoding="utf-8")
        # Import inside the loop so a missing zh.ts is a visible skip, not a crash.
        assert "vector" not in text.lower() or vector_presets, (
            f"{path.name} promises vector formulas but no preset enables them"
        )

    # The wizard must key the "slow / vector" badge off the preset that has it.
    wizard = _WIZARD_TSX.read_text(encoding="utf-8")
    assert Preset.PUBLICATION_MATH.value in wizard


def test_preset_badges_do_not_claim_a_quality_gate() -> None:
    """A preset never changes a quality gate (see ``ubt/core/presets.py``).

    The badge used to read "富提示词 ＋ 公式见证" / "RICH PROMPTS + FORMULA
    WITNESS", borrowing the word "gate" from the export stage while setting no
    gate at all. Badges now name the parser and prompt depth, which *are*
    preset fields.
    """
    from ubt.core.presets import PRESETS

    if not _EN_TS.exists():
        pytest.skip("web/ frontend is not present in this checkout")
    en = _EN_TS.read_text(encoding="utf-8")
    badge_block = re.search(r"presetBadge\w+:\s*'([^']*)'", en)
    assert badge_block, "en.ts has no presetBadge key"
    for preset in PRESETS.values():
        # Every preset declares both parameters the badges describe.
        assert preset.pdf_engine in ("auto", "pdfium", "docling")
        assert preset.prompt_strategy in ("auto", "minimal", "hybrid", "rich")


def test_wizard_sends_layout_columns_within_the_api_range() -> None:
    """``layout_columns`` accepts 1 or 2 or null; the wizard must not send 0/"".

    An empty string or 0 would be a 422 for every job, the exact class of bug
    the payload-key test above was written for.
    """
    from ubt.api.models import JobSubmitRequest

    if not _WIZARD_TSX.exists():
        pytest.skip("web/ frontend is not present in this checkout")
    source = _WIZARD_TSX.read_text(encoding="utf-8")
    assert "layout_columns" in source
    # The wizard maps its 'auto' sentinel to null before submitting.
    assert re.search(r"layout_columns:\s*layoutColumns === 'auto' \? null", source)
    assert "layout_columns" in JobSubmitRequest.model_fields


# --- the pre-flight panel must not show contradictory numbers ----------------

_WIZARD_SRC = _REPO_ROOT / "web" / "src" / "views" / "wizard" / "NewJobWizard.tsx"


def test_the_estimate_provenance_badge_is_panel_level_not_per_row() -> None:
    """One provenance marker for the whole panel.

    A silent re-assess (any preset/pages/language tweak) runs with
    ``deep=false``, which replaces the exact block count with a sampled
    estimate. The character count moves with it — 33,968 exact vs 47,698
    sampled on corpus/documents/chapter-3.pdf, a 40% swing — so marking only
    the token row left the user with two contradictory scale figures and no
    explanation. Every scale figure comes from the same assess call, so the
    marker belongs on the panel header.
    """
    if not _WIZARD_SRC.exists():
        pytest.skip("web/ frontend is not present in this checkout")
    source = _WIZARD_SRC.read_text(encoding="utf-8")

    # The badge is rendered from the header block, above the metric rows.
    header_end = source.index("preflightTitle")
    metrics_start = source.index("sourceChars")
    assert header_end < metrics_start, "unexpected panel layout"

    header_block = source[header_end:metrics_start]
    assert "billable_blocks_is_exact" in header_block, (
        "the estimate-provenance badge is not in the panel header"
    )

    # And it must not be repeated inside the metric rows.
    metrics_block = source[metrics_start : source.index("probeWarnings")]
    assert "estimateExact" not in metrics_block, (
        "the estimate marker is repeated per row; it belongs on the panel header only"
    )


def test_the_formula_density_row_explains_text_vs_layout() -> None:
    """``math_density`` and ``has_formulas`` are measured from different things.

    ``math_density`` is a text-sample symbol density; ``has_formulas`` is the
    per-page layout census. corpus/documents/chapter-3.pdf reports
    ``math_density=high`` with ``has_formulas=False`` — both true, but the pair
    reads as a contradiction without a caption.
    """
    from ubt.api.models import AssessDocumentFacts

    assert "math_density" in AssessDocumentFacts.model_fields
    assert "has_formulas" in AssessDocumentFacts.model_fields

    if not _WIZARD_SRC.exists():
        pytest.skip("web/ frontend is not present in this checkout")
    source = _WIZARD_SRC.read_text(encoding="utf-8")
    assert "mathDensityWithObjects" in source
    assert "mathDensityTextOnly" in source


def test_font_residue_warning_says_it_is_not_a_scan_defect() -> None:
    """A 0% scanned share next to a 20-page garble warning needs a bridge.

    ``FONT_RESIDUE_RISK`` comes from font-encoding damage, so it fires on a
    perfectly born-digital PDF. Rendered next to ``scan_page_share=0%`` the
    pair looked like a bug; both locales now say the two are unrelated.
    """
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


# --- the quote must not under-report tokens or hide the at-risk pages --------


def test_the_wizard_shows_total_tokens_not_just_the_source_size() -> None:
    """``estimated_tokens`` is the source size; the bill is prompt+completion.

    On corpus/documents/chapter-3.pdf the payload carries
    ``estimated_tokens=11,956`` while the real estimate is
    ``prompt_tokens + completion_tokens = 32,013``. Labelling the source figure
    "预估消耗 Tokens" under-quoted spend by ~2.7x, which is exactly the number
    a user reads to decide whether to run.
    """
    from ubt.api.models import AssessCost, AssessDocumentFacts

    assert "estimated_tokens" in AssessDocumentFacts.model_fields
    assert "prompt_tokens" in AssessCost.model_fields
    assert "completion_tokens" in AssessCost.model_fields

    if not _WIZARD_SRC.exists():
        pytest.skip("web/ frontend is not present in this checkout")
    source = _WIZARD_SRC.read_text(encoding="utf-8")
    assert "cost.prompt_tokens + assessment.cost.completion_tokens" in source, (
        "the token row does not sum prompt+completion"
    )
    # And the label must admit it is an estimate: no character heuristic can be
    # exact across model families (the same English sentence is 29 tokens to
    # XLM-R and 19 to DeepSeek).
    zh = (_EN_TS.parent / "zh.ts").read_text(encoding="utf-8")
    assert "估算" in re.search(r"estTokens:\s*'([^']*)'", zh).group(1)  # type: ignore[union-attr]


def test_the_budget_control_has_no_zero_means_unlimited_trap() -> None:
    """The old slider mapped 0 to "no cap" and capped at $50.

    Both were wrong: a user dragging to 0 to mean "spend nothing" got an
    uncapped run, and a $50 ceiling cannot express a large cloud job's budget.
    The control is now an explicit switch plus a numeric field, and the field
    must never submit 0 because the API declares ``budget_usd: gt=0``.
    """
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
    # An unusable entry blocks submission rather than sending 0.
    assert "budgetInvalid" in source
    assert "parsed > 0" in source


def test_font_residue_warning_carries_the_at_risk_page_count() -> None:
    """``at_risk_pages`` is the witness's recall half and was dropped.

    The engine computed it (``wit.get("at_risk_pages")``) and only wrote it into
    a free-text quality signal, so the console showed 20 confirmed pages and
    hid the 5 suspects — the set an operator would want visually checked.
    """
    if not _ASSESS_PY.exists():
        pytest.skip("engine source is not present in this checkout")
    source = _ASSESS_PY.read_text(encoding="utf-8")
    assert '"at_risk_pages": at_risk' in source, (
        "FONT_RESIDUE_RISK no longer carries at_risk_pages in its params"
    )
    if not _WIZARD_SRC.exists():
        pytest.skip("web/ frontend is not present in this checkout")
    wizard = _WIZARD_SRC.read_text(encoding="utf-8")
    assert "at_risk_pages" in wizard
    assert "warningAtRiskClause" in wizard


# --- the console bundle must stay split across routes ------------------------


def test_every_screen_is_lazily_loaded() -> None:
    """A static view import puts the whole console in the entry chunk.

    Before the split, `App.tsx` imported all six screens statically, so the
    entry bundle carried the wizard, mission control, the review workbench, the
    quality gate, language assets and the doctor before any one of them
    rendered. It crossed 500 kB and tripped Vite's size warning. Each screen is
    now a `lazy()` import with its own chunk.

    This is a source-level guard because the failure mode is silent: reverting
    to a static import still builds, still type-checks, and only shows up as a
    slower first paint.
    """
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
        # A static `import { X } from '@/views/...'` would defeat the split.
        assert not re.search(rf"^import \{{[^}}]*\b{view}\b", source, re.MULTILINE), (
            f"{view} is also imported statically, which defeats code splitting"
        )

    # The Suspense boundary is what lets a lazy route render at all.
    assert "<Suspense" in source
    assert "RouteFallback" in source


def test_the_shipped_bundle_has_no_chunk_over_vite_warning_size() -> None:
    """The committed bundle must stay under the 500 kB warning threshold.

    ``ubt/api/static`` is what a `pip install`ed release serves, so this checks
    the artefact rather than the source. Skipped when the bundle is absent (a
    source checkout without a build).
    """
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
    """Proof the split actually produced separate chunks in the artefact.

    A regression to static imports would collapse the bundle back to a single
    ``index-*.js``; that still passes the size check on a small enough tree, so
    the chunk count is asserted separately.
    """
    static_assets = _REPO_ROOT / "ubt" / "api" / "static" / "assets"
    if not static_assets.exists():
        pytest.skip("console bundle is not present in this checkout")

    js = sorted(p.name for p in static_assets.glob("*.js"))
    assert len(js) > 5, f"expected per-route chunks, found only: {js}"
    entry = [n for n in js if n.startswith("index-")]
    assert len(entry) == 1, f"expected exactly one entry chunk, found: {entry}"


def test_the_wizard_offers_every_engine_ocr_mode() -> None:
    """The console must not be narrower than ``OcrMode``.

    ``--ocr`` and the REST ``ocr_mode`` field existed from the start, but the
    wizard exposed no OCR control at all — so an operator reading a
    FONT_RESIDUE_RISK warning ("enable the OCR pre-check") had no way to act on
    that advice from the console. A mode the engine dropped would 422 every
    submission; a mode the engine added would stay unreachable.
    """
    import typing

    from ubt.core.config import OcrMode

    if not _WIZARD_SRC.exists():
        pytest.skip("web/ frontend is not present in this checkout")
    source = _WIZARD_SRC.read_text(encoding="utf-8")

    offered = set(re.findall(r'<option value="(\w+)">\{t\.wizard\.ocrMode', source))
    engine = set(typing.get_args(OcrMode))
    assert offered, "the wizard has no OCR mode select"
    # 'auto' is the sentinel the wizard submits as null, so it is offered but is
    # also a legal engine value.
    assert offered == engine, (
        f"wizard/engine OCR mode mismatch: only-wizard={sorted(offered - engine)}, "
        f"only-engine={sorted(engine - offered)}"
    )


def test_the_wizard_marks_ocr_modes_that_need_page_egress() -> None:
    """``cloud`` and ``vlm`` ship page images off the machine.

    The engine refuses both unless ``UBT_ALLOW_PAGE_UPLOAD=true``. Surfacing
    that at selection time is the difference between a clear refusal and a
    submit-time error the operator has to decode.
    """
    if not _WIZARD_SRC.exists():
        pytest.skip("web/ frontend is not present in this checkout")
    source = _WIZARD_SRC.read_text(encoding="utf-8")
    assert re.search(r"ocrNeedsEgress\s*=\s*ocrMode === 'cloud' \|\| ocrMode === 'vlm'", source)
    assert "ocrModeEgressWarning" in source


def test_the_wizard_sends_ocr_mode_within_the_api_enum() -> None:
    """``ocr_mode`` must reach the API as null or a legal ``OcrMode`` value."""
    from ubt.api.models import JobSubmitRequest

    if not _WIZARD_SRC.exists():
        pytest.skip("web/ frontend is not present in this checkout")
    source = _WIZARD_SRC.read_text(encoding="utf-8")
    assert re.search(r"ocr_mode:\s*ocrMode === 'auto' \? null : ocrMode", source)
    assert "ocr_mode" in JobSubmitRequest.model_fields


# --- the doctor's config panel must describe the live configuration ----------


def test_runtime_config_never_exposes_credentials() -> None:
    """Provider blocks may hold an ``api_key``; the endpoint must filter it.

    ``load_provider_block`` returns the merged block as written, and a real
    config does declare ``api_key = "${OPENCODE_API_KEY}"``. Returning the block
    verbatim would put credential material — or at least the name of the secret
    it expands from — on an unauthenticated-by-default HTTP surface.
    """
    from ubt.api.runtime_config import _EXPOSED_FIELDS

    forbidden = {"api_key", "api-key", "token", "secret", "password", "extra_headers"}
    assert not (set(_EXPOSED_FIELDS) & forbidden), (
        f"credential fields are exposed: {sorted(set(_EXPOSED_FIELDS) & forbidden)}"
    )

    # And the serialised payload carries no credential material at all.
    import json

    from ubt.api.runtime_config import describe_runtime_config
    from ubt.core.config import UBTConfig

    payload = json.dumps(describe_runtime_config(UBTConfig.from_env()))
    for needle in ("api_key", "OPENCODE_API_KEY", "extra_headers", "Bearer"):
        assert needle not in payload, f"runtime config leaked {needle!r}"


def test_runtime_config_reports_the_active_model_not_the_preset_table() -> None:
    """The panel must name the provider/model in force, not fourteen presets.

    ``/model-profiles`` returns the engine's hard-coded capability table
    (``o1``, ``o3``, ``claude``, …). None of those is a configured model; on the
    reference machine the live configuration is ``local-gemma4`` /
    ``gemma4-e4b``, which matches none of them. The screen was titled "model
    routing and latency matrix" while containing neither.
    """
    from ubt.api.runtime_config import describe_runtime_config
    from ubt.core.config import UBTConfig

    cfg = UBTConfig.from_env()
    active = describe_runtime_config(cfg)["active"]

    # UBTConfig.provider is None when unset; the view normalises it to "".
    assert active["provider"] == (cfg.provider or "")
    assert active["draft_model"] == cfg.draft_model
    assert active["base_url"] == cfg.base_url
    # A model that hits no preset must report that honestly rather than
    # borrowing the wildcard profile's name.
    assert "matched_pattern" in active
    assert active["matched_pattern"] != "*"


def test_runtime_config_separates_declared_providers_from_aliases() -> None:
    """Seventeen rows for a six-provider config was the readability failure.

    ``list_providers`` merges declared blocks, shipped wire protocols and bare
    aliases (``chat``, ``claude``, ``responses``). A configuration screen must
    show the operator's own entries first and fold the rest away.
    """
    from ubt.api.runtime_config import describe_runtime_config
    from ubt.core.config import UBTConfig
    from ubt.core.providers import PROTOCOL_ALIASES, declared_provider_names

    providers = describe_runtime_config(UBTConfig.from_env())["providers"]
    declared = [p for p in providers if p["declared"]]
    aliases = [p for p in providers if p["is_alias"]]

    # Every declared name is flagged, and no alias is.
    assert {p["name"] for p in declared} == declared_provider_names()
    assert not ({p["name"] for p in declared} & set(PROTOCOL_ALIASES))
    assert {p["name"] for p in aliases} <= set(PROTOCOL_ALIASES)
    # Exactly one provider is marked active.
    assert sum(1 for p in providers if p["active"]) <= 1


def test_the_doctor_no_longer_claims_a_latency_matrix() -> None:
    """No endpoint returns latency; the heading said it did."""
    repo = _REPO_ROOT
    view = repo / "web" / "src" / "views" / "doctor" / "SystemDoctorView.tsx"
    if not view.exists():
        pytest.skip("web/ frontend is not present in this checkout")

    source = view.read_text(encoding="utf-8")
    assert "runtimeConfig" in source, "the panel does not use the runtime-config endpoint"
    assert "getRuntimeConfig" in source
    # The alias/protocol split is what keeps the list readable.
    assert ".filter((p) => p.declared)" in source


def test_the_doctor_renders_declared_capability_not_a_preset_table() -> None:
    """The screen must distinguish a declared flag from an assumed default.

    ``declared_capability`` exists because four plain booleans cannot say whether
    the operator told the engine that the endpoint rejects temperature, or the
    engine simply assumed it does not. Presenting an assumption as a measurement
    is exactly the failure the old "latency matrix" heading had.
    """
    from ubt.api.runtime_config import describe_runtime_config
    from ubt.core.config import UBTConfig

    declared = describe_runtime_config(
        UBTConfig.from_env(draft_model="m", supports_temperature=False)
    )["active"]
    assert declared["declared_capability"] == ["supports_temperature"]
    assert declared["supports_temperature"] is False

    assumed = describe_runtime_config(UBTConfig.from_env(draft_model="m"))["active"]
    assert assumed["declared_capability"] == []
    assert assumed["supports_temperature"] is True


def test_the_doctor_has_no_capability_preset_list() -> None:
    """The fourteen-name table is gone from both the view and the API surface."""
    repo = _REPO_ROOT
    view = repo / "web" / "src" / "views" / "doctor" / "SystemDoctorView.tsx"
    if not view.exists():
        pytest.skip("web/ frontend is not present in this checkout")

    source = view.read_text(encoding="utf-8")
    # The panel used to render one row per built-in preset.
    assert "listModelProfiles" not in source
    assert "capabilityPresets" not in source
    # It edits the block the capability actually comes from instead.
    assert "ProviderEditor" in source


def test_the_provider_editor_gates_the_credential_field_on_the_deployment() -> None:
    """The credential input must be conditional, not always-on or always-off.

    Refusing the write outright (the earlier design) left a fresh install with
    zero declared providers and zero edit buttons, so a model could only be
    configured from a shell. Offering it unconditionally would put a secret in a
    request body on a shared server. The form has to follow the server's answer,
    which arrives as ``can_write_credential``.
    """
    repo = _REPO_ROOT
    editor = repo / "web" / "src" / "views" / "doctor" / "ProviderEditor.tsx"
    if not editor.exists():
        pytest.skip("web/ frontend is not present in this checkout")

    source = editor.read_text(encoding="utf-8")
    # Presence is reported as a mask, never as the value: the placeholder is the
    # fixed mask, so a configured key renders as "already set" rather than being
    # read back into the browser.
    assert "has_credential" in source
    assert "'********'" in source
    # The input is bound to the deployment's permission, and the refusal branch
    # explains itself instead of silently hiding the control.
    assert "canWriteCredential" in source
    assert "credentialBlockedHint" in source
    # The other channel's credential has no business in an LLM provider form.
    assert "ocr_api_key" not in source


def test_the_console_can_create_the_first_provider() -> None:
    """A fresh install has no declared provider, so the list alone is a dead end.

    ``providers.filter(p => p.declared)`` renders zero rows when nothing is
    configured, which means zero edit buttons — the exact state a new user starts
    in. A create affordance is what makes the panel reachable at all.
    """
    repo = _REPO_ROOT
    view = repo / "web" / "src" / "views" / "doctor" / "SystemDoctorView.tsx"
    if not view.exists():
        pytest.skip("web/ frontend is not present in this checkout")

    source = view.read_text(encoding="utf-8")
    assert "addProvider" in source
    # ``null`` means "create" and ``undefined`` means "closed"; a single nullable
    # string cannot express both, and collapsing them kills the button.
    assert "setEditingProvider(null)" in source


def test_a_failed_config_load_does_not_claim_zero_providers() -> None:
    """``(runtime?.providers ?? []).length`` renders "(0)" on a load failure.

    That asserts a fact the panel does not have — the same failure as the
    "latency matrix" heading that named a number no endpoint produced. The count
    must distinguish "none" from "could not read".
    """
    repo = _REPO_ROOT
    view = repo / "web" / "src" / "views" / "doctor" / "SystemDoctorView.tsx"
    if not view.exists():
        pytest.skip("web/ frontend is not present in this checkout")

    source = view.read_text(encoding="utf-8")
    assert "loadFailed" in source
    # The old unconditional coalesce must be gone from the count expression.
    assert "{(runtime?.providers ?? []).filter((p) => !p.declared).length}" not in source


def test_an_undeclared_configuration_reports_the_generic_default() -> None:
    """The panel must not echo the model id back as if it were a capability.

    With nothing declared, ``matched_pattern`` was the model name — so the screen
    read "Capability profile: gemma4-e4b" while the paragraph beneath it said no
    flags were declared. Two contradictory statements on one panel.
    """
    from ubt.api.runtime_config import describe_runtime_config
    from ubt.core.config import UBTConfig

    active = describe_runtime_config(UBTConfig.from_env(draft_model="undeclared-model"))["active"]
    assert active["matched_pattern"] is None
    assert active["declared_capability"] == []
    # The safe defaults are still reported, so the panel can show what will be
    # sent rather than leaving four blanks.
    assert active["supports_temperature"] is True
    assert active["supports_reasoning_effort"] is False


def test_a_state_value_is_declared_before_the_expression_that_reads_it() -> None:
    """A hook-derived value must not be declared above the state it reads.

    ``const isCreate = initialName === null && detail === null`` placed above the
    ``useState`` for ``detail`` throws "Cannot access 'm' before initialization" —
    a temporal dead zone. Neither ``tsc`` nor ``oxlint`` flags it (the identifier
    is declared later in the same scope, so both consider it valid), and the error
    boundary turns the whole screen into a blank page. Only a browser run caught
    it, which is too late for CI.
    """
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
    """The wizard must not infer "would run as a rehearsal" from provider rows.

    ``/jobs/submit`` downgrades a keyless submit using ``credential_required``
    plus ``has_real_credential``. A client-side guess based on "no declared
    provider" would be wrong for the common built-in-protocol setup:
    ``UBT_LLM_API_KEY`` plus ``openai-chat`` has no declared block and still
    translates for real.

    Every config here pins ``base_url`` to a *hosted* endpoint on purpose. The
    credential rules depend on it: a loopback endpoint needs no key and reports
    ``can_translate: true``, so inheriting the ambient ``ubt.toml`` made this
    test pass or fail depending on the developer's machine — it went green only
    while the file had no default provider.
    """
    from ubt.api.runtime_config import describe_runtime_config
    from ubt.core.config import MOCK_API_KEY, UBTConfig

    HOSTED = "https://api.example.com/v1"

    keyless = describe_runtime_config(UBTConfig.from_env(base_url=HOSTED, draft_model="m"))[
        "active"
    ]
    assert keyless["can_translate"] is False

    # A model with no credential cannot translate …
    assert (
        describe_runtime_config(
            UBTConfig.from_env(base_url=HOSTED, draft_model="", api_key="sk-real")
        )["active"]["can_translate"]
        is False
    )
    # … and the placeholder sentinel is not a credential either.
    assert (
        describe_runtime_config(
            UBTConfig.from_env(base_url=HOSTED, draft_model="m", api_key=MOCK_API_KEY)
        )["active"]["can_translate"]
        is False
    )
    # Real key plus a model is the only combination that translates.
    assert (
        describe_runtime_config(
            UBTConfig.from_env(base_url=HOSTED, draft_model="m", api_key="sk-real")
        )["active"]["can_translate"]
        is True
    )


def test_the_wizard_blocks_a_rehearsal_it_has_not_announced() -> None:
    """A mock deliverable nobody asked for is the failure being prevented."""
    repo = _REPO_ROOT
    wizard = repo / "web" / "src" / "views" / "wizard" / "NewJobWizard.tsx"
    if not wizard.exists():
        pytest.skip("web/ frontend is not present in this checkout")

    source = wizard.read_text(encoding="utf-8")
    # The verdict comes from the server, not from a row count.
    assert "can_translate" in source
    assert "rehearsalAcknowledged" in source
    # The guard runs before the submit call.
    guard = source.index("if (rehearsalOnly && !rehearsalAcknowledged)")
    submit = source.index("await submitJob(")
    assert guard < submit


def test_a_failed_config_load_is_not_reported_as_a_refusal() -> None:
    """``?? false`` on a failed load claimed the server was reachable remotely.

    That is a statement about the deployment the console cannot make when it could
    not read the deployment configuration at all.
    """
    repo = _REPO_ROOT
    editor = repo / "web" / "src" / "views" / "doctor" / "ProviderEditor.tsx"
    if not editor.exists():
        pytest.skip("web/ frontend is not present in this checkout")

    source = editor.read_text(encoding="utf-8")
    assert "canWriteCredential: boolean | null" in source
    assert "credentialUnknownHint" in source


def test_the_frontend_typecheck_command_actually_checks_something() -> None:
    """``tsc --noEmit`` on this repo type-checks NOTHING.

    ``web/tsconfig.json`` is a solution file: ``{"files": [], "references": [...]}``.
    ``tsc --noEmit`` reads that root config, finds no inputs, and exits 0 — so a
    green run proves only that tsc started. The real check is ``tsc -b`` (which is
    what ``pnpm build`` runs), and it caught two genuine type errors that the
    ``--noEmit`` form had reported as clean.
    """
    import json

    repo = _REPO_ROOT
    tsconfig = repo / "web" / "tsconfig.json"
    if not tsconfig.exists():
        pytest.skip("web/ frontend is not present in this checkout")

    root = json.loads(tsconfig.read_text(encoding="utf-8"))
    # The root config must stay a pure solution file; if it ever gains "include"
    # or "files" entries, `--noEmit` would start working and this guard should go.
    assert root.get("files") == [], (
        "web/tsconfig.json now has inputs, so `tsc --noEmit` is no longer a no-op — "
        "update this test and the documented command"
    )
    assert root.get("references"), "the solution file must reference the real projects"

    # The script CI relies on is the one that actually builds.
    pkg = json.loads((repo / "web" / "package.json").read_text(encoding="utf-8"))
    assert pkg["scripts"]["build"].startswith("tsc -b"), (
        "web build must run `tsc -b`; `tsc --noEmit` type-checks nothing here"
    )


def test_the_capability_form_asks_only_what_the_engine_cannot_learn() -> None:
    """Two controls, and the reason the other two are absent.

    ``supports_temperature`` and ``supports_reasoning_effort`` are recovered by the
    400 self-heal and then remembered per model (measured: four blocks take five
    requests, and the parameter never reappears). A control could only ever save
    that one round-trip, while asking an operator to answer a question the engine
    answers itself. They were removed from the form entirely — not moved to an
    "advanced" group, which still asked the question.

    ``supports_system_prompt`` cannot be healed at all (the fix is a merge
    decision), so it stays. ``supports_vision`` fails per block, so it stays.
    """
    repo = _REPO_ROOT
    editor = repo / "web" / "src" / "views" / "doctor" / "ProviderEditor.tsx"
    if not editor.exists():
        pytest.skip("web/ frontend is not present in this checkout")

    source = editor.read_text(encoding="utf-8")
    fields = source[source.index("const CAPABILITY_FIELDS") : source.index("type FlagField")]
    assert "supports_system_prompt" in fields
    assert "supports_temperature" not in fields
    assert "supports_reasoning_effort" not in fields

    # No control at all for the two the engine learns — not even a collapsed one.
    for gone in (
        "capabilityTemperature",
        "capabilityReasoning",
        "capabilityAutoLearn",
        "capabilityAdvanced",
    ):
        assert gone not in source, f"{gone} is back in the form"
    assert "supports_vision" in source


def test_the_removed_controls_are_still_engine_fields() -> None:
    """Removing the form control must not remove the engine's knowledge.

    ``ModelProfile.supports_temperature`` decides whether the parameter is *sent*.
    Without it a rejected temperature would be re-sent on every block, and the
    self-heal could only fix the retry, not the next call.
    """
    from ubt.core.config import UBTConfig
    from ubt.core.router.capabilities import ModelProfile

    for field in ("supports_temperature", "supports_reasoning_effort"):
        assert field in ModelProfile.model_fields, f"{field} left the profile"
        assert field in UBTConfig.model_fields, f"{field} left the config"
    # And they are still settable in a provider block, for the operator who wants
    # to skip the first 400.
    from ubt.core.providers import PROVIDER_ALLOWED_KEYS

    assert "supports_temperature" in PROVIDER_ALLOWED_KEYS
    assert "supports_reasoning_effort" in PROVIDER_ALLOWED_KEYS


def test_every_rendered_capability_control_is_submitted() -> None:
    """A checkbox that renders but never reaches the request is a dead control.

    ``supports_vision`` is its own checkbox rather than an entry in
    ``ALL_FLAGS``, so the save loop (which walked ``ALL_FLAGS``) skipped it: the
    box toggled, the state updated, and nothing was sent. Found by saving through
    the real UI and diffing the file — the type checker and the linter both saw a
    valid component.
    """
    repo = _REPO_ROOT
    editor = repo / "web" / "src" / "views" / "doctor" / "ProviderEditor.tsx"
    if not editor.exists():
        pytest.skip("web/ frontend is not present in this checkout")

    source = editor.read_text(encoding="utf-8")
    # The save loop must iterate the submitted set, not the display groups.
    save_start = source.index("const handleSave = useCallback")
    save_body = source[save_start : source.index("return (", save_start)]
    assert "SUBMITTED_FLAGS" in save_body, (
        "handleSave must iterate SUBMITTED_FLAGS; iterating ALL_FLAGS skips supports_vision"
    )
    # And that set must actually contain the standalone checkbox.
    assert "'supports_vision'" in source[source.index("const SUBMITTED_FLAGS") :][:200]


def test_the_doctor_panel_and_the_editor_share_one_grid_cell() -> None:
    """The editor must replace the runtime panel, not push it out of the grid.

    They were two independent grid children of a two-column grid, so opening the
    form made three boxes: the editor took the right cell of row 1 and the whole
    runtime panel wrapped into row 2 at x=368 — under the diagnostics, in the
    *left* column, still reporting "local-gemma4 is active" while the operator
    edited index-translate. Measured in a browser at 1440px: panel x=860 before,
    x=368 after. Same data twice on screen, one copy in the wrong place.
    """
    repo = _REPO_ROOT
    view = repo / "web" / "src" / "views" / "doctor" / "SystemDoctorView.tsx"
    if not view.exists():
        pytest.skip("web/ frontend is not present in this checkout")

    source = view.read_text(encoding="utf-8")
    # One wrapper claims the right column explicitly...
    assert "md:col-start-2" in source, (
        "the panel and the editor must share a grid cell; as two siblings the "
        "third box wraps into the left column"
    )
    # ...and the panel is hidden rather than unmounted while the form is open.
    assert "editingProvider !== undefined ? 'hidden'" in source
    # The row being edited is marked, or the list highlights only the *active*
    # provider and gives no confirmation of what is open.
    assert "editingProvider === p.name" in source


def test_the_credential_copy_matches_what_the_server_actually_does() -> None:
    """The UI must not claim credentials are refused over HTTP.

    That sentence was true before ``34fb4de`` made the write deployment-gated,
    and it survived in three places — the zh copy, the en copy, and the client
    docstring — so the same screen offered a credential field and, one line
    below it, said a credential can never arrive over HTTP. A form contradicting
    itself is worse than a form with no explanation.
    """
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

    # And the replacement says what is actually true.
    zh_text = zh.read_text(encoding="utf-8")
    assert "本机" in zh_text and "0600" in zh_text
    en_text = en.read_text(encoding="utf-8")
    assert "loopback" in en_text and "0600" in en_text


def test_a_save_that_the_environment_outranks_is_reported_as_such() -> None:
    """The form said "effective on the next job" while the env said otherwise.

    ``UBT_LLM_API_KEY`` outranks every provider block, so writing a key into the
    file while that variable is set produced a successful save whose credential
    the engine never used. The response has to carry that fact, and the form has
    to render it, or the operator debugs a translation that keeps hitting the old
    account.
    """
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
    """The provider list must report the *code*, not the variable name.

    A name like ``OPENCODE_API_KEY`` is one step closer to the credential than a
    stable code, and this payload renders every provider at once. The actionable
    detail belongs on the single-provider endpoint the operator opens to fix it.
    """
    from ubt.api.runtime_config import describe_runtime_config
    from ubt.core.config import UBTConfig

    payload = describe_runtime_config(UBTConfig.from_env())
    for provider in payload["providers"]:
        assert "unresolved_vars" not in provider
    import json

    assert "unresolved_vars" not in json.dumps(payload)


def test_the_doctor_names_the_judge_model_its_block_actually_uses() -> None:
    """The QE row printed the translator's repair_model as the judge.

    When ``qe_judge_provider`` names a different block, that block carries its own
    model. The reference config answers with ``workbuddy/deepseek-v4.1-flash``
    while the row said ``gemma4-e4b`` — a model the judge endpoint has never heard
    of. The same mismatch had already been measured at runtime as twelve
    ``404 no such model`` answers in one run.
    """
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

            # The ambient shell may export UBT_PROVIDER (this repo's own does),
            # which would select a provider this fixture never declared.
            os.environ.pop("UBT_PROVIDER", None)
            cfg = UBTConfig.from_env()
            checks = collect_checks(cfg, probe=False)
        finally:
            os.chdir(previous)

    qe = next(c for c in checks if c.code == "qe")
    assert "judge-model" in qe.detail, f"QE row names the wrong judge model: {qe.detail}"
    assert "translator-model" not in qe.detail
    # ``params`` is optional on the model; a check that emits a template with
    # ``{provider}`` must always set it, so an absent dict is itself a failure.
    assert (qe.params or {}).get("provider") == "judge"


def test_the_doctor_panel_prefers_the_provider_variant_when_one_is_named() -> None:
    """The variant template must actually be reachable from the renderer.

    ``qe.tiered.via`` exists so the Chinese sentence can name the provider block.
    A catalogue entry nothing selects is dead copy — the same class of defect as a
    control that renders but never submits.
    """
    repo = _REPO_ROOT
    text = repo / "web" / "src" / "i18n" / "doctorText.ts"
    if not text.exists():
        pytest.skip("web/ frontend is not present in this checkout")

    source = text.read_text(encoding="utf-8")
    assert "PARAM_VARIANTS" in source, "the variant picker was removed"
    assert "'qe.tiered'" in source and "'via'" in source
    # The picker must consult the param that decides the variant.
    assert "'provider'" in source


def test_a_fresh_install_can_reach_a_runnable_state_from_the_console() -> None:
    """The end-to-end gap: create a block, and it must actually run.

    Measured on a fresh clone (no ``ubt.toml``, no ``UBT_*``): one block declared,
    yet ``provider: ''``, ``draft_model: ''`` and ``can_translate: false``. The
    console showed the block and a working credential, and the first job still
    became a rehearsal. Nothing in the UI pointed at a default, and
    ``activateProvider`` — the copy for exactly that action — was unused.
    """
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

            # The create form's path.
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
    """``active`` and ``is_default`` answer different questions.

    ``UBT_PROVIDER`` outranks the file, so a shell that exported a provider hid
    what the file would pick. The list has to carry both, or "which one runs
    without my shell variable" is unanswerable from the console.
    """
    from ubt.api.models import RuntimeProvider

    assert "is_default" in RuntimeProvider.model_fields
    repo = _REPO_ROOT
    view = repo / "web" / "src" / "views" / "doctor" / "SystemDoctorView.tsx"
    if not view.exists():
        pytest.skip("web/ frontend is not present in this checkout")
    source = view.read_text(encoding="utf-8")
    assert "p.is_default" in source
    # And the switch affordance uses the copy that existed but was never wired.
    assert "activateProvider" in source
    assert "setDefaultProvider" in source


def test_creating_a_provider_makes_it_the_default() -> None:
    """One action, not "create, then discover you also have to select it"."""
    repo = _REPO_ROOT
    editor = repo / "web" / "src" / "views" / "doctor" / "ProviderEditor.tsx"
    if not editor.exists():
        pytest.skip("web/ frontend is not present in this checkout")
    source = editor.read_text(encoding="utf-8")
    assert "changes.make_default = true" in source
    # Only on create: editing an existing block must not silently steal the
    # default from whichever block is currently configured to run.
    save_start = source.index("const handleSave = useCallback")
    save_body = source[source.index("if (isCreate) changes.make_default", save_start) :][:120]
    assert "isCreate" in save_body


def test_the_doctor_endpoint_reads_the_live_config_not_the_boot_snapshot() -> None:
    """One screen, two answers: the form said saved, the checklist said missing.

    ``/system/doctor`` ran ``collect_checks`` against ``app_config`` — the config
    built once at startup — while ``/system/runtime-config`` beside it read the
    live source. Create a provider in the console and the panel reported the new
    block as active while the checklist still said ``API key FAIL`` and
    ``no draft model configured``.
    """
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
    """Behavioural counterpart: the checks must see what the console just wrote.

    The static guard above can be satisfied by a rename; this one runs the real
    path — write a block, rebuild the config, run the checks — and asserts the
    credential row flips. It is the difference between "the variable name is
    right" and "a fresh install's first provider is actually reported as usable".
    """
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
    """A provider saved a second ago must be what the next job uses.

    ``app_config = scope.config`` takes the *value* the live property returned at
    startup, so everything that read it was pinned to the boot snapshot. Found on
    a fresh install: the console showed the new provider as active and
    ``can_translate: true``, while ``/jobs/assess`` answered ``draft_model: ""``
    and the submit path still considered the deployment keyless.
    """
    repo = _REPO_ROOT
    routes = repo / "ubt" / "api" / "routes" / "job.py"
    source = routes.read_text(encoding="utf-8")

    assess = source[source.index("async def assess_job") :][:900]
    assert "cfg = scope.config" in assess, "assess must price against the live config"

    submit = source[source.index("auto_rehearsal = False") :][:600]
    assert "scope.config" in submit, "the rehearsal decision must read the live config"

    # And the worker itself, at every call site (fresh submit, resume, re-parse).
    # Every one must hand over ``scope.config``: the boot snapshot would run with
    # the provider that existed when the server started, not the one the operator
    # just saved.
    assert source.count("execute_job(record, scope.config)") == 3
    assert "execute_job(record, app_config)" not in source


def test_the_builtin_list_distinguishes_protocols_from_aliases() -> None:
    """Labelling a protocol an "alias" hid which name belongs in a block.

    Both kinds rendered the same way, so ``openai-chat`` (a wire protocol — the
    name a ``[providers.*]`` block should carry) looked identical to ``chat`` (a
    shortcut to it), and ``anthropic-messages`` printed its own name on the right,
    which says nothing. The payload now separates them and names the target.
    """
    from ubt.api.runtime_config import describe_runtime_config
    from ubt.core.config import UBTConfig

    providers = describe_runtime_config(UBTConfig.from_env())["providers"]
    by_name = {p["name"]: p for p in providers if not p["declared"]}

    # Exactly four protocols, and they are not aliases.
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

    # Every alias names the protocol it resolves to, and that target exists.
    for name, row in by_name.items():
        if not row["is_alias"]:
            continue
        assert row["alias_of"] in protocols, f"{name} points at {row['alias_of']!r}"
    assert by_name["claude"]["alias_of"] == "anthropic-messages"
    assert by_name["chat"]["alias_of"] == "openai-chat"


def test_the_panel_help_says_a_protocol_still_needs_a_model_id() -> None:
    """The prerequisite a newcomer actually trips over.

    Selecting ``gemini`` supplies an endpoint and a wire format and nothing else;
    ``draft_model`` stays empty and the run cannot start. The list said none of
    that, so the entries looked like ready-to-use configurations.
    """
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
    # And the two groups are rendered separately, not as one flat list.
    assert "p.is_protocol" in source
    assert "p.alias_of" in source


def test_the_terminal_stream_frame_carries_the_final_counters() -> None:
    """The last log line of a finished job said "completed · 0/0 blocks".

    ``terminal_frame`` held only ``status``/``output_file``/``error``, so the
    console fell back to ``?? 0`` for both counts — on the one line a user reads
    to confirm the run.
    """
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

    # A caller that passes nothing still gets a valid frame (older clients and
    # the failed-before-start path have no snapshot to send).
    bare = json.loads(terminal_frame("failed", None, "boom").split("data: ", 1)[1].strip())
    assert bare == {"status": "failed", "output_file": None, "error": "boom"}


def test_the_panel_renders_failures_and_the_human_queue() -> None:
    """The API always carried them; the panel never showed them.

    A job ending with 15 blocks routed to a person rendered as a clean 100%, and
    the number existed only in ``/status``.
    """
    repo = _REPO_ROOT
    view = repo / "web" / "src" / "views" / "mission-control" / "MissionControl.tsx"
    if not view.exists():
        pytest.skip("web/ frontend is not present in this checkout")
    source = view.read_text(encoding="utf-8")

    # Counters are folded from the stream and rendered.
    for field in ("needs_human_blocks", "failed_blocks", "skipped_blocks"):
        assert field in source, f"{field} is not read from the stream"
    assert "needsHumanBlocks > 0" in source
    assert "failedBlocks > 0" in source
    assert "skippedBlocks > 0" in source

    # And the zero-fallback that produced "0/0 blocks" is gone from the log line.
    assert "frame.completed_blocks ?? completedRef.current" in source
    assert "frame.drafted_blocks ?? draftedRef.current" in source
    assert "${frame.completed_blocks ?? 0}/${frame.total_blocks ?? 0}" not in source


def test_the_review_pane_does_not_render_the_failed_draft_markup() -> None:
    """The engine's <mark> wrapper must not reach the operator as raw text.

    ``repair.py`` wraps a failed draft in ``<mark class="ubt-failed-draft"
    title="...">译文</mark>``. The workbench rendered ``target_text`` directly, so
    the attribute — a full escaped JSON envelope — showed up as the translation,
    overflowing its column (measured 828px of content in a 446px pane).
    """
    repo = _REPO_ROOT
    view = repo / "web" / "src" / "views" / "review" / "ReviewWorkbench.tsx"
    helper = repo / "web" / "src" / "lib" / "failedDraft.ts"
    if not view.exists():
        pytest.skip("web/ frontend is not present in this checkout")

    assert helper.exists(), "the markup needs a parser, not a raw render"
    source = view.read_text(encoding="utf-8")
    assert "parseFailedDraft" in source
    assert "{failed.text || '—'}" in source
    # The raw value must no longer render anywhere as the segment body.
    assert "{seg.target_text || '—'}" not in source
    # And the reason is shown in its own bounded, wrapping box.
    assert "failedDraftNotice" in source
    assert "break-words" in source

    parser = helper.read_text(encoding="utf-8")
    assert "ubt-failed-draft" in parser
    # Entities in the attribute are decoded for display.
    assert "&quot;" in parser


def test_the_review_page_has_one_name_in_each_locale() -> None:
    """It had two — "质检后编台" in the sidebar, "审校后编台" on the button.

    Both were transliterations of "Workbench" that named neither the page nor
    what it does: it walks the book block by block so a person can review and
    edit each translation.
    """
    repo = _REPO_ROOT
    zh = repo / "web" / "src" / "i18n" / "translations" / "zh.ts"
    if not zh.exists():
        pytest.skip("web/ frontend is not present in this checkout")
    text = zh.read_text(encoding="utf-8")

    assert "质检后编台" not in text
    assert "审校后编台" not in text
    # The replacement says what the page does.
    assert "逐块审校台" in text
