"""Regression tests for API security, ledger isolation, pipeline statelessness, and adapter SPI."""

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from ubt.adapters.base import BaseDocumentAdapter
from ubt.adapters.factory import (
    _ADAPTER_REGISTRY,
    _PDF_ENGINE_REGISTRY,
    get_adapter_for_path,
    register_adapter,
    register_pdf_engine,
)
from ubt.api.app import (
    SYSTEM_DISALLOWED_PREFIXES,
    create_app,
    resolve_secure_path,
    validate_job_id,
)
from ubt.core.config import UBTConfig
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.pipeline import PipelineOrchestrator
from ubt.core.engine.repair_loop import RepairLoop
from ubt.core.exceptions import UnsupportedDocumentFormatError
from ubt.core.ir.models import (
    BlockStatus,
    BookManifest,
    ChapterIR,
    ChapterMeta,
    IRBlock,
)
from ubt.core.qe.comet_runner import MockQERunner
from ubt.core.qe.fast_pass import FastPassFilter
from ubt.core.router.provider import MockModelProvider
from ubt.core.router.router import ModelRouter

# =============================================================================
# 1. API Security Hardening Tests
# =============================================================================


def test_validate_safe_path_security_checks(tmp_path: Path) -> None:
    """Verify resolve_secure_path rejects system paths, traversals, and sensitive files."""
    # Disallowed system directories
    for bad in [
        "/etc/shadow",
        "/root/secrets.txt",
        "/proc/cpuinfo",
        "/sys/kernel",
        "/var/log/syslog",
    ]:
        with pytest.raises(HTTPException) as exc:
            resolve_secure_path(Path(bad), must_exist=False)
        assert exc.value.status_code == 403
        assert "restricted system directory" in exc.value.detail

    # Sensitive filenames / hidden files
    for sensitive in [".ssh/id_rsa", ".env", ".aws/credentials", ".bashrc", ".git/config"]:
        with pytest.raises(HTTPException) as exc:
            resolve_secure_path(tmp_path / sensitive, must_exist=False)
        assert exc.value.status_code == 403
        assert "sensitive configuration directory or file" in exc.value.detail

    # Directory traversal
    with pytest.raises(HTTPException) as exc:
        resolve_secure_path("../traversal/file.epub", must_exist=False)
    assert exc.value.status_code == 403
    assert "Directory traversal" in exc.value.detail

    # Safe path succeeds when it is inside an allowed base
    safe = tmp_path / "valid_doc.epub"
    safe.touch()
    assert (
        resolve_secure_path(safe, must_exist=True, allowed_bases=[tmp_path.resolve()])
        == safe.resolve()
    )


def test_operator_whitelist_unblocks_paths_inside_a_system_prefix(tmp_path: Path) -> None:
    """Risk: the hard-coded system deny list was checked before containment and
    overrode an explicit whitelist, so ``UBT_ALLOWED_DIRS=/var/lib/ubt/books``
    was unusable with 403 "restricted system directory" — the whitelist the
    non-loopback startup guardrail demands could not be satisfied with the
    conventional ``/var/lib/ubt`` data location."""
    config = UBTConfig(allowed_dirs="/var/lib/ubt/books", db_dir=tmp_path / "ledgers")

    allowed = resolve_secure_path("/var/lib/ubt/books/book.pdf", must_exist=False, config=config)
    assert allowed.name == "book.pdf"

    # Containment stays the authority: everything outside the whitelist is still
    # refused, including the system dirs the deny list covers.
    for denied in ("/etc/passwd", "/root/.bashrc", str(tmp_path / "outside.md")):
        with pytest.raises(HTTPException) as exc:
            resolve_secure_path(denied, must_exist=False, config=config)
        assert exc.value.status_code == 403


def test_system_deny_list_still_applies_without_a_whitelist(tmp_path: Path) -> None:
    """Risk: lifting the deny list for whitelisted paths must not weaken the
    default posture — with no ``UBT_ALLOWED_DIRS`` the system directories stay
    the fallback sandbox (fail closed)."""
    config = UBTConfig(db_dir=tmp_path / "ledgers")
    for denied in ("/etc/shadow", "/proc/cpuinfo", "/dev/null", "/usr/bin/env"):
        with pytest.raises(HTTPException) as exc:
            resolve_secure_path(denied, must_exist=False, config=config)
        assert exc.value.status_code == 403
        assert "restricted system directory" in exc.value.detail


def test_system_deny_prefixes_are_pre_resolved() -> None:
    """Risk (macOS): /etc, /var and /tmp are symlinks into /private, so a deny
    list built from unresolved paths never matched the resolved request path and
    the system-directory branch silently allowed those reads."""
    assert SYSTEM_DISALLOWED_PREFIXES
    assert all(prefix == prefix.resolve() for prefix in SYSTEM_DISALLOWED_PREFIXES)
    assert all(prefix.is_absolute() for prefix in SYSTEM_DISALLOWED_PREFIXES)


def test_containment_denial_does_not_leak_server_paths(tmp_path: Path) -> None:
    """Risk: the 403 body echoed the server's absolute allowed bases, disclosing
    the host's directory layout to any caller of a route that is unauthenticated
    by default."""
    base = tmp_path / "books"
    base.mkdir()
    config = UBTConfig(allowed_dirs=str(base), db_dir=tmp_path / "ledgers")

    with pytest.raises(HTTPException) as exc:
        resolve_secure_path(tmp_path / "outside_doc.md", must_exist=False, config=config)
    assert exc.value.status_code == 403
    detail = str(exc.value.detail)
    assert str(tmp_path) not in detail
    assert str(base) not in detail
    assert "outside_doc" not in detail


def test_sensitive_list_is_casefolded_and_complete(tmp_path: Path) -> None:
    """L1: the deny list must cover today's credential stores, case-insensitively.

    Two gaps: the comparison was case-sensitive (so ``.SSH``/``.ENV`` slipped
    through on case-insensitive filesystems), and the list predated
    ``.git-credentials``/``.gnupg``/``.password-store``/``.bash_history``/
    ``.env.local``/``secrets.env``/``credentials.json``.
    """
    from ubt.core.fs_perms import SENSITIVE_FILENAME_PARTS

    required = {
        ".git-credentials",
        ".gnupg",
        ".password-store",
        ".bash_history",
        ".env.local",
        "secrets.env",
        "credentials.json",
    }
    assert required <= set(SENSITIVE_FILENAME_PARTS), "review L1 additions went missing"

    probes = [
        ".SSH/id_rsa",
        ".GNUPG/pubring.kbx",
        ".password-store/.gpg-id",
        ".bash_history",
        ".env.local",
        "secrets.env",
        "credentials.json",
        "nested/.Git-Credentials",
    ]
    for name in probes:
        with pytest.raises(HTTPException) as exc:
            resolve_secure_path(tmp_path / name, must_exist=False)
        assert exc.value.status_code == 403, name
        assert "sensitive configuration directory or file" in exc.value.detail, name


def test_sensitive_deny_beats_an_explicit_whitelist(tmp_path: Path) -> None:
    """The precedence is deliberate, not a contradiction (review L1).

    ``resolve_secure_path`` docstring point 3 exempts a whitelisted path from
    the *system* deny list; point 4 says the *sensitive-name* deny list still
    wins. An allowlist is a scope decision ("these books"), not a licence to
    serve the ``credentials.json`` that happens to live beside them — and this
    route is reachable without credentials by default.
    """
    config = UBTConfig(allowed_dirs=str(tmp_path), db_dir=tmp_path / "ledgers")

    # The allowlist admits an ordinary file in the same directory...
    ok = tmp_path / "book.md"
    ok.write_text("# Title\n", encoding="utf-8")
    assert resolve_secure_path(ok, must_exist=True, config=config) == ok.resolve()

    # ...but not the credential file beside it, whitelist or no whitelist.
    secret = tmp_path / "credentials.json"
    secret.write_text("{}", encoding="utf-8")
    with pytest.raises(HTTPException) as exc:
        resolve_secure_path(secret, must_exist=True, config=config)
    assert exc.value.status_code == 403
    assert "sensitive configuration directory or file" in exc.value.detail


def test_validate_job_id_format() -> None:
    """Verify job_id validation prevents path traversal and SQL injection characters."""
    valid_ids = ["job_123", "job-abc-456", "ABC_xyz-001", "simple1"]
    for jid in valid_ids:
        assert validate_job_id(jid) == jid

    invalid_ids = [
        "../traversal",
        "job/123",
        "job\\123",
        "job;drop table",
        "job' or '1'='1",
        "job with space",
        "",
        "job@name",
    ]
    for jid in invalid_ids:
        with pytest.raises(HTTPException) as exc:
            validate_job_id(jid)
        assert exc.value.status_code == 400
        assert "Invalid job_id format" in exc.value.detail


def test_submit_job_persists_resolved_output_path(tmp_path: Path, system_probe_path: str) -> None:
    """Verify submit_job resolves output_path safely and captures it in the background task."""
    input_file = tmp_path / "book.md"
    input_file.write_text("# Test Title\n\nHello world", encoding="utf-8")
    out_file = tmp_path / "sub_dir" / "translated.md"

    db_dir = tmp_path / "ledgers"
    config = UBTConfig(db_dir=db_dir, allowed_dirs=str(tmp_path))
    app = create_app(config=config)
    client = TestClient(app)

    # 1. Traversal attempt in output_path
    resp = client.post(
        "/jobs/submit",
        json={
            "input_path": str(input_file),
            # Next to the platform's own system probe, so the deny list (not the
            # containment fallback) is what answers on every OS.
            "output_path": str(Path(system_probe_path).with_name("evil_output.md")),
            "target_lang": "zh",
        },
    )
    assert resp.status_code == 403
    assert "restricted system directory" in resp.text

    # 2. Sensitive file in output_path
    resp = client.post(
        "/jobs/submit",
        json={
            "input_path": str(input_file),
            "output_path": str(tmp_path / ".bashrc"),
            "target_lang": "zh",
        },
    )
    assert resp.status_code == 403
    assert "sensitive configuration" in resp.text

    # 3. Valid submission
    resp = client.post(
        "/jobs/submit",
        json={
            "input_path": str(input_file),
            "output_path": str(out_file),
            "target_lang": "zh",
        },
    )
    assert resp.status_code == 202
    resp.json()["job_id"]

    # Verify status endpoint rejects invalid job IDs
    resp_bad = client.get("/jobs/../bad_id/status")
    # FastAPI path routing or validate_job_id returns 400/404
    assert resp_bad.status_code in (400, 404)


# =============================================================================
# 2. Ledger Multi-Job Scope Isolation Tests
# =============================================================================


def test_ledger_strict_job_isolation_for_same_doc_id(tmp_path: Path) -> None:
    """Verify that multiple jobs for the same doc_id never mix blocks."""
    db_path = tmp_path / "multi_job_ledger.sqlite3"
    ledger = SQLiteJobLedger(db_path=db_path)

    doc_id = "shared_book_doc"
    job_1 = "job_v1_run"
    job_2 = "job_v2_run"

    # Manifest for shared book
    manifest = BookManifest(
        doc_id=doc_id,
        title="Shared Book",
        source_path="/dummy/book.epub",
        source_lang="en",
        target_lang="zh",
        chapters=[ChapterMeta(chapter_id="ch_1", title="Chapter One", spine_index=1)],
    )

    # Init both jobs
    ledger.init_job_from_manifest(job_1, manifest)
    ledger.init_job_from_manifest(job_2, manifest)

    # Job 1 blocks
    ch1_blocks = [
        IRBlock(id="j1_b1", spine_index=1, source_text="J1 Block 1", status=BlockStatus.PENDING),
        IRBlock(id="j1_b2", spine_index=2, source_text="J1 Block 2", status=BlockStatus.DRAFTED),
    ]
    ch1 = ChapterIR(
        doc_id=doc_id, chapter_id="ch_1", title="Chapter 1", spine_index=1, blocks=ch1_blocks
    )
    ledger.append_chapter(job_1, ch1)

    # Job 2 blocks
    ch2_blocks = [
        IRBlock(id="j2_b1", spine_index=1, source_text="J2 Block 1", status=BlockStatus.PENDING),
        IRBlock(id="j2_b2", spine_index=2, source_text="J2 Block 2", status=BlockStatus.DRAFTED),
        IRBlock(
            id="j2_b3", spine_index=3, source_text="J2 Block 3", status=BlockStatus.REPAIR_PENDING
        ),
    ]
    ch2 = ChapterIR(
        doc_id=doc_id, chapter_id="ch_1", title="Chapter 1", spine_index=1, blocks=ch2_blocks
    )
    ledger.append_chapter(job_2, ch2)

    # 1. Verify get_all_blocks isolation
    j1_all = ledger.get_all_blocks(job_1)
    j2_all = ledger.get_all_blocks(job_2)

    assert len(j1_all) == 2
    assert [b.id for b in j1_all] == ["j1_b1", "j1_b2"]

    assert len(j2_all) == 3
    assert [b.id for b in j2_all] == ["j2_b1", "j2_b2", "j2_b3"]

    # 2. Verify fetch_blocks_by_status isolation
    j1_drafted = ledger.fetch_blocks_by_status(job_1, BlockStatus.DRAFTED)
    assert len(j1_drafted) == 1
    assert j1_drafted[0].id == "j1_b2"

    j2_drafted = ledger.fetch_blocks_by_status(job_2, BlockStatus.DRAFTED)
    assert len(j2_drafted) == 1
    assert j2_drafted[0].id == "j2_b2"

    # 3. Verify fetch_repair_eligible_blocks isolation
    j1_repair = ledger.fetch_repair_eligible_blocks(job_1)
    assert len(j1_repair) == 2  # pending and drafted are non-terminal

    j2_repair = ledger.fetch_repair_eligible_blocks(job_2)
    assert len(j2_repair) == 3

    # 4. Verify resolving by doc_id resolves ONLY to the most recent job (not a union)
    doc_blocks = ledger.get_all_blocks(doc_id)
    # job_2 was initialized second, so doc_blocks must contain only job_2 blocks
    assert len(doc_blocks) == 3
    assert [b.id for b in doc_blocks] == ["j2_b1", "j2_b2", "j2_b3"]


# =============================================================================
# 3. Pipeline Statelessness & Local FastPass Tests
# =============================================================================


@pytest.mark.asyncio
async def test_pipeline_stateless_fast_pass_and_repair() -> None:
    """Verify PipelineOrchestrator does not mutate shared state during execution."""
    router = ModelRouter(provider=MockModelProvider())
    repair_loop = RepairLoop(router=router, qe_runner=MockQERunner())

    initial_fast_pass = FastPassFilter(source_lang="en", target_lang="zh")
    repair_loop.fast_pass = initial_fast_pass

    orchestrator = PipelineOrchestrator(
        router=router,
        repair_loop=repair_loop,
        qe_runner=MockQERunner(),
    )
    # The orchestrator no longer carries a language-blind
    # fast_pass attribute at all — filters are per-stage, per-language.
    assert not hasattr(orchestrator, "fast_pass")

    # Ensure repair_single_block accepts explicit fast_pass
    custom_fast_pass = FastPassFilter(source_lang="ja", target_lang="en")
    block = IRBlock(
        id="repair_b1",
        spine_index=1,
        source_text="こんにちは世界",
        target_text="Bonjour le monde",
        status=BlockStatus.REPAIR_PENDING,
        repair_rounds=0,
    )

    repaired = await repair_loop.repair_single_block(
        block=block,
        glossary_table="",
        target_lang="en",
        source_lang="ja",
        fast_pass=custom_fast_pass,
    )
    assert repaired.repair_rounds == 1

    # Verify instance fast_pass was not clobbered
    assert repair_loop.fast_pass is initial_fast_pass


# =============================================================================
# 4. Adapter SPI Registry Pattern Tests
# =============================================================================


def test_adapter_registry_extensibility() -> None:
    """Verify that new adapters and PDF engines can be registered without modifying core factory."""

    class CustomXYZAdapter(BaseDocumentAdapter):
        async def extract_manifest(self, input_path: Path) -> BookManifest:
            return BookManifest(
                doc_id="xyz_doc",
                title="XYZ",
                source_path=str(input_path),
                source_lang="en",
                target_lang="zh",
                chapters=[],
            )

        async def parse_stream(
            self, input_path: Path, pages: set[int] | None = None
        ) -> AsyncIterator[ChapterIR]:
            yield ChapterIR(
                doc_id="xyz_doc", chapter_id="xyz_ch1", title="XYZ", spine_index=1, blocks=[]
            )

        async def render_output(
            self,
            manifest: BookManifest,
            ledger: SQLiteJobLedger,
            target_lang: str,
            output_path: Path,
            job_id: str | None = None,
            bilingual_mode: str | None = None,
            **kwargs: Any,
        ) -> Path:
            return output_path

    # Register a custom mock adapter factory
    @register_adapter([".xyz", ".zyx"])
    def make_xyz_adapter(pdf_engine: str, path: Path) -> BaseDocumentAdapter:
        return CustomXYZAdapter()

    # Verify lookup succeeds for both extensions
    adapter_xyz = get_adapter_for_path(Path("sample.xyz"))
    assert isinstance(adapter_xyz, CustomXYZAdapter)

    adapter_zyx = get_adapter_for_path(Path("sample.zyx"))
    assert isinstance(adapter_zyx, CustomXYZAdapter)

    # Register a custom PDF engine
    @register_pdf_engine(["custom_pdf_engine"])
    def create_custom_engine() -> Any:
        return CustomXYZAdapter()

    custom_pdf = get_adapter_for_path(Path("sample.pdf"), pdf_engine="custom_pdf_engine")
    assert isinstance(custom_pdf, CustomXYZAdapter)

    # Verify unsupported extension raises UnsupportedDocumentFormatError
    with pytest.raises(UnsupportedDocumentFormatError) as exc:
        get_adapter_for_path(Path("file.unknown_ext"))
    assert "No adapter registered" in str(exc.value)

    # Verify unsupported PDF engine raises UnsupportedDocumentFormatError
    with pytest.raises(UnsupportedDocumentFormatError) as exc:
        get_adapter_for_path(Path("file.pdf"), pdf_engine="nonexistent_engine")
    assert "Unsupported or unregistered PDF engine" in str(exc.value)

    # Clean up registries
    _ADAPTER_REGISTRY.pop(".xyz", None)
    _ADAPTER_REGISTRY.pop(".zyx", None)
    _PDF_ENGINE_REGISTRY.pop("custom_pdf_engine", None)


def test_every_registered_pdf_engine_is_selectable_from_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The config layer must not keep its own copy of the engine names.

    ``PdfEngine`` used to be a Literal duplicating ``_PDF_ENGINE_REGISTRY`` and
    silently rejected ``typst`` and ``modern``, which the factory does serve.
    Iterating the registry keeps this guard from becoming a second drift source.
    """
    assert _PDF_ENGINE_REGISTRY, "registry is empty; the loop below would prove nothing"
    for name in sorted(_PDF_ENGINE_REGISTRY):
        monkeypatch.setenv("UBT_PDF_ENGINE", name)
        assert UBTConfig.from_env().pdf_engine == name


def test_unknown_pdf_engine_reports_the_live_registry(tmp_path: Path) -> None:
    """Typo-safety moved here from the Literal, so it has to list the real options."""
    with pytest.raises(UnsupportedDocumentFormatError) as exc:
        get_adapter_for_path(tmp_path / "book.pdf", pdf_engine="doclin")

    message = str(exc.value)
    assert "Available:" in message
    for name in _PDF_ENGINE_REGISTRY:
        assert name in message


def test_default_sandbox_confines_to_cwd_and_db_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty UBT_ALLOWED_DIRS must not disable the containment check."""
    monkeypatch.chdir(tmp_path)
    cfg = UBTConfig(db_dir=tmp_path / "ledgers")

    outside = tmp_path.parent / "ubt_outside_secret.txt"
    outside.write_text("secret", encoding="utf-8")
    try:
        with pytest.raises(HTTPException) as exc:
            resolve_secure_path(outside, must_exist=True, config=cfg)
        assert exc.value.status_code == 403
    finally:
        outside.unlink(missing_ok=True)

    (tmp_path / "ledgers").mkdir()
    inside = tmp_path / "ledgers" / "note.txt"
    inside.write_text("ok", encoding="utf-8")
    assert resolve_secure_path(inside, must_exist=True, config=cfg) == inside.resolve()


def test_non_loopback_bind_refused_without_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    app_module = importlib.import_module("ubt.api.app")
    for name in ("UBT_API_KEY", "UBT_ALLOWED_DIRS", "UBT_ALLOWED_DIR", "UBT_ALLOW_INSECURE_BIND"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(SystemExit, match="refusing to bind"):
        app_module.run_server(host="0.0.0.0", port=1)


def test_health_is_open_but_other_routes_are_gated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UBT_API_KEY", "secret-key")
    config = UBTConfig(db_dir=tmp_path / "ledgers")
    client = TestClient(create_app(config=config))
    # Liveness probes cannot carry credentials.
    assert client.get("/health").status_code == 200
    assert client.get("/jobs/job_x/status").status_code == 401
    assert client.get("/jobs/job_x/status", headers={"X-API-Key": "secret-key"}).status_code == 404


def test_job_id_length_is_bounded() -> None:
    with pytest.raises(HTTPException) as exc:
        validate_job_id("a" * 129)
    assert exc.value.status_code == 400


def test_page_range_size_is_bounded() -> None:
    from ubt.core.config import parse_page_ranges

    with pytest.raises(ValueError, match="too large"):
        parse_page_ranges("1-999999999")
    assert parse_page_ranges("1-3,5") == {1, 2, 3, 5}
