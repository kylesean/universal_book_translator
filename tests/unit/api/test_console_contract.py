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
