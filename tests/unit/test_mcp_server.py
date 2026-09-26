"""MCP server tools: direct-function tests (protocol framing covered by mcp lib)."""
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from tests.pdf_builders import text_pdf
from tests.unit.ir_seed import SeedDoc, seed_job
from ubt.mcp.server import ubt_doctor, ubt_inspect_book, ubt_job_status


async def test_inspect_book_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The manifest tool reads inside the sandbox (M4: paths go through it now)."""
    monkeypatch.setenv("UBT_ALLOWED_DIRS", str(tmp_path))
    pdf = text_pdf(tmp_path / "mcp.pdf", 2)
    out = await ubt_inspect_book(str(pdf))
    assert out["title"]
    assert out["doc_id"]
    assert len(out["chapters"]) >= 1


async def test_inspect_missing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UBT_ALLOWED_DIRS", str(tmp_path))
    with pytest.raises(Exception, match="not found"):
        await ubt_inspect_book(str(tmp_path / "does-not-exist-ubt.pdf"))


async def test_job_status_unknown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UBT_ALLOWED_DIRS", str(tmp_path))
    out = await ubt_job_status("job_nope_unknown", db_dir=str(tmp_path / "mcp_empty"))
    assert out["status"] == "unknown"


async def test_job_status_bad_id() -> None:
    with pytest.raises(Exception, match="Invalid job_id"):
        await ubt_job_status("../evil")


async def test_paths_outside_the_allowlist_are_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M4: the sandbox used to be a literal ``..`` check and nothing else.

    ``ubt_job_status(db_dir=...)`` therefore created and opened
    ``<job_id>.sqlite`` (read-write, schema init) at any path an agent named,
    and every other tool accepted any readable input. Default posture matches
    the REST allowlist: cwd + ``config.db_dir``, widened only by
    ``UBT_ALLOWED_DIRS`` (unset here so the default is what is asserted).
    """
    from ubt.mcp.server import ubt_translate_book

    monkeypatch.delenv("UBT_ALLOWED_DIRS", raising=False)
    with pytest.raises(Exception, match="outside the allowed directories"):
        await ubt_job_status("job_any", db_dir="/tmp/ubt_mcp_escape")
    with pytest.raises(Exception, match="outside the allowed directories"):
        await ubt_inspect_book("/etc/hosts")
    with pytest.raises(Exception, match="outside the allowed directories"):
        await ubt_translate_book(input_path="/etc/hosts")


async def test_sensitive_paths_are_refused_inside_the_allowlist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sensitive-name deny list beats the allowlist, exactly as in REST.

    An MCP server started in ``$HOME`` has the whole home directory inside the
    default allowlist; a prompt-injected agent must still not read ``.ssh``
    through these tools (review M4 + L1).
    """
    monkeypatch.setenv("UBT_ALLOWED_DIRS", str(tmp_path))
    secret_dir = tmp_path / ".ssh"
    secret_dir.mkdir()
    with pytest.raises(Exception, match="sensitive configuration"):
        await ubt_job_status("job_any", db_dir=str(secret_dir))


async def test_job_status_default_ledger_dir_is_still_served(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The *positive* default posture: cwd + ``config.db_dir`` must keep working.

    The sandbox tests above only prove refusals; this pins that a ledger in the
    configured default location is still reachable when no ``UBT_ALLOWED_DIRS``
    and no explicit ``db_dir`` argument are given — the common case for every
    polling agent.
    """
    from ubt.core.config import UBTConfig
    from ubt.core.engine.ledger import SQLiteJobLedger
    from ubt.core.ir.models import BookManifest, ChapterMeta

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("UBT_ALLOWED_DIRS", raising=False)

    job_id = "job_default_ledger"
    db_dir = UBTConfig.from_env().db_dir  # .ubt/ledgers, relative to the cwd
    Path(db_dir).mkdir(parents=True, exist_ok=True)
    ledger = SQLiteJobLedger(Path(db_dir) / f"{job_id}.sqlite")
    ledger.init_job_from_manifest(
        job_id,
        BookManifest(
            doc_id="mcp_default",
            title="t",
            source_path="book.md",
            chapters=[ChapterMeta(chapter_id="ch01", title="One", spine_index=1)],
        ),
    )
    ledger.close()

    out = await ubt_job_status(job_id)
    assert out["status"] in ("running", "completed", "empty")


async def test_deep_assess_is_capped_at_two_and_queues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M4: ``deep=True`` runs full adapter ingest, so it gets REST's ceiling too.

    REST gates the same work behind ``assess_semaphore``; MCP had none, and
    ``MCP_MAX_RUNNING_JOBS`` deliberately covers translations only. Two run at
    once, the rest queue — they must not be dropped or serialised to one.
    """
    import asyncio

    import ubt.core.assess as assess_mod
    import ubt.mcp.server as srv

    monkeypatch.setattr(srv, "_deep_assess_semaphore", asyncio.Semaphore(2))
    monkeypatch.setenv("UBT_ALLOWED_DIRS", str(tmp_path))

    active = 0
    peak = 0
    gate = asyncio.Event()

    class _Report:
        def to_dict(self) -> dict[str, Any]:
            return {"status": "ok"}

    async def _slow(_path: object, _cfg: object, *, deep: bool = False, **_kw: object) -> Any:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        if deep:
            await gate.wait()
        active -= 1
        return _Report()

    monkeypatch.setattr(assess_mod, "assess_document_async", _slow)
    doc = tmp_path / "assess_me.md"
    doc.write_text("Hello world.", encoding="utf-8")

    tasks = [asyncio.create_task(srv.ubt_assess_book(str(doc), deep=True)) for _ in range(4)]
    await asyncio.sleep(0.05)
    assert active == 2, f"deep assess must run 2 at a time, saw {active}"
    gate.set()
    results = await asyncio.gather(*tasks)
    assert len(results) == 4, "waiters queue, they are not rejected"
    assert peak == 2, f"concurrency ceiling breached: {peak}"

    # Shallow assess is a cheap profile read: not gated by the semaphore.
    assert (await srv.ubt_assess_book(str(doc), deep=False))["status"] == "ok"


def test_main_converges_env_file_permissions(monkeypatch: pytest.MonkeyPatch) -> None:
    """M1: ``ubt-mcp`` must chmod ``.env`` to 0600 before the tools read it."""
    import ubt.mcp.server as srv

    calls: list[object] = []
    monkeypatch.setattr(srv, "restrict_env_file", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(srv.mcp, "run", lambda **_k: None)
    srv.main()
    assert calls, "main() must converge .env to owner-only before serving"


async def test_doctor_shape() -> None:
    out = await ubt_doctor()
    assert isinstance(out["ok"], bool)
    names = {c["name"] for c in out["checks"]}
    assert {"api_key", "ledger_dir"} <= names


async def test_translate_book_rejects_bad_lang() -> None:
    """§10.3-#4: entry-layer language-code validation, same pattern as REST."""
    from ubt.mcp.server import ubt_translate_book

    with pytest.raises(Exception, match="Invalid target_lang"):
        await ubt_translate_book(input_path="/any/book.pdf", target_lang="zh; rm -rf")
    with pytest.raises(Exception, match="Invalid source_lang"):
        await ubt_translate_book(input_path="/any/book.pdf", source_lang="$(x)")


async def test_translate_book_caps_concurrency(monkeypatch: pytest.MonkeyPatch) -> None:
    """§10.3-#4: MCP honours a max_running_jobs ceiling like the REST API."""
    import ubt.mcp.server as srv
    from ubt.mcp.server import ubt_translate_book

    monkeypatch.setattr(
        srv,
        "_JOBS",
        {
            f"job_{i}": srv._JobRecord(job_id=f"job_{i}", status="running")
            for i in range(srv.MCP_MAX_RUNNING_JOBS)
        },
    )
    with pytest.raises(Exception, match="Too many concurrent jobs"):
        await ubt_translate_book(input_path="/any/book.pdf", target_lang="zh", source_lang="en")


async def test_cancelled_job_reaches_a_terminal_status(monkeypatch: pytest.MonkeyPatch) -> None:
    """A torn-down MCP task must not leave its job "running" (review P1-19).

    The record counts against MCP_MAX_RUNNING_JOBS and blocks a resubmit of the
    same id while it sits at "running", and CancelledError is a BaseException, so
    the generic failure handler never saw the cancellation.
    """
    import asyncio

    import ubt.mcp.server as srv

    class HangingOrchestrator:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def run(self, **_kwargs: object) -> AsyncIterator[None]:
            await asyncio.Event().wait()
            yield None  # pragma: no cover - makes this an async generator

    monkeypatch.setattr(srv, "PipelineOrchestrator", HangingOrchestrator)
    rec = srv._JobRecord(job_id="mcp_cancel")
    jobs = {"mcp_cancel": rec}
    monkeypatch.setattr(srv, "_JOBS", jobs)

    task = asyncio.create_task(srv._execute("mcp_cancel", {"input_path": "/any/book.pdf"}))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert rec.status == "cancelled"
    # A cancelled job no longer occupies the concurrency cap, and its id is reusable.
    running = sum(1 for r in jobs.values() if r.status in ("submitted", "running"))
    assert running == 0
    assert jobs["mcp_cancel"].status not in ("submitted", "running")


def test_prune_jobs_uses_shared_terminal_statuses(monkeypatch: pytest.MonkeyPatch) -> None:
    """MCP trimming must agree with the queue's definition of "done".

    It kept its own (\"completed\", \"failed\", \"cancelled\") literal set; when
    the queue's terminal set grows, a forgotten copy would retain (or evict)
    the wrong records.
    """
    import ubt.mcp.server as server
    from ubt.core.engine.job_queue import TERMINAL_JOB_STATUSES

    monkeypatch.setattr(server, "_MAX_RETAINED", 3)
    terminal = [str(status) for status in sorted(TERMINAL_JOB_STATUSES, key=str)]
    jobs = {f"j_{name}": server._JobRecord(job_id=f"j_{name}", status=name) for name in terminal}
    jobs["j_running"] = server._JobRecord(job_id="j_running", status="running")
    jobs["j_queued"] = server._JobRecord(job_id="j_queued", status="queued")
    monkeypatch.setattr(server, "_JOBS", jobs)
    server._prune_jobs()
    assert "j_running" in server._JOBS  # live records are never pruned
    assert "j_queued" in server._JOBS
    assert not any(rec.status in TERMINAL_JOB_STATUSES for rec in server._JOBS.values())


async def test_job_status_rejects_traversal_db_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ubt_translate_book guards db_dir with the path sandbox; the status
    disk-fallback used to take it raw and hand SQLiteJobLedger (read-write,
    schema-init) any path an injected agent names (L20/M4)."""
    monkeypatch.setenv("UBT_ALLOWED_DIRS", str(tmp_path))

    with pytest.raises(Exception, match=r"\.\."):
        await ubt_job_status("job_any", db_dir=str(Path(tmp_path) / ".." / "etc"))

    # A legitimate ledger dir still works after the guard.
    from ubt.core.engine.ledger import SQLiteJobLedger
    from ubt.core.ir.models import BookManifest, ChapterMeta

    manifest = BookManifest(
        doc_id="mcp_doc",
        title="t",
        source_path="book.md",
        chapters=[ChapterMeta(chapter_id="ch01", title="One", spine_index=1)],
    )
    ledger = SQLiteJobLedger(Path(tmp_path) / "job_mcp_ok.sqlite")
    ledger.init_job_from_manifest(
        "job_mcp_ok",
        manifest,
    )
    out = await ubt_job_status("job_mcp_ok", db_dir=str(tmp_path))
    assert out["status"] in ("running", "completed", "empty")


@pytest.mark.fast
async def test_mcp_server_job_status_disk_fallback_has_output_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ubt.core.engine.ledger import SQLiteJobLedger
    from ubt.mcp.server import ubt_job_status

    monkeypatch.setenv("UBT_ALLOWED_DIRS", str(tmp_path))
    db_path = tmp_path / "job_mcp_test.sqlite"
    doc_ir = SeedDoc(
        doc_id="doc1", source_path=str(tmp_path / "book.pdf"), format_type="pdf", blocks=[]
    )
    with SQLiteJobLedger(db_path) as ledger:
        seed_job(ledger, "job_mcp_test", doc_ir, "zh")
        ledger.set_job_metadata_value(
            "job_mcp_test", "output_file", str(tmp_path / "book_bilingual.pdf")
        )
        ledger.set_job_metadata_value("job_mcp_test", "report_file", str(tmp_path / "quality.json"))

    res = await ubt_job_status("job_mcp_test", db_dir=str(tmp_path))
    assert res["job_id"] == "job_mcp_test"
    assert res.get("output_file") == str(tmp_path / "book_bilingual.pdf")


@pytest.mark.fast
async def test_mcp_doctor_checks_pypdfium2() -> None:
    from ubt.mcp.server import ubt_doctor

    res = await ubt_doctor()
    check_names = [c["name"] for c in res["checks"]]
    assert "dep:pypdfium2" in check_names
    assert "dep:pypdf" not in check_names


@pytest.mark.fast
def test_mcp_in_memory_status_includes_report_fields() -> None:
    import asyncio

    from ubt.core.engine.progress import ProgressSnapshot
    from ubt.mcp.server import _JOBS, _JobRecord, ubt_job_status

    job_id = "test_mcp_job"
    progress = ProgressSnapshot(
        total_blocks=10,
        completed_blocks=10,
        current_avg_qe=0.95,
        output_file="/tmp/out.epub",
        report_file="/tmp/out_quality_report.json",
        visual_report_file="/tmp/out_visual_report.json",
    )
    _JOBS[job_id] = _JobRecord(
        job_id=job_id,
        status="completed",
        progress=progress,
    )
    try:
        status_res = asyncio.run(ubt_job_status(job_id))
        assert status_res.get("report_file") == "/tmp/out_quality_report.json"
        assert status_res.get("visual_report_file") == "/tmp/out_visual_report.json"
        assert status_res.get("avg_qe_score") == 0.95
    finally:
        _JOBS.pop(job_id, None)


# ---------------------------------------------------------------------------
# Engine-knob parity with the REST payload surface
# ---------------------------------------------------------------------------


#: The knob families an agent may set. Kept in step with the REST
#: ``JobSubmitRequest`` test (``tests/unit/test_api.py``) so the two shells
#: cannot drift apart again: the CLI could always set these, and only the
#: payload-gated surfaces lagged.
_MCP_ENGINE_KNOBS: dict[str, Any] = {
    # Resume control (a UBTConfig field, unlike the two below).
    "fresh": True,
    "budget_usd": 5.0,
    "max_concurrency": 8,
    "batch_limit": 3,
    "macro_chunk_size": 12,
    "short_max_pages": 12,
    "enable_rolling_summary": True,
    "chapter_streaming_enabled": True,
    "offline_batch_enabled": True,
    "qe_engine": "tiered",
    "visual_judge_enabled": True,
    "visual_judge_model": "gpt-4o-mini",
    "prompt_strategy": "rich",
    "translate_chrome": True,
    "facing_spread": True,
    "emit_both": True,
    "cover_mode": "never",
    "formula_enrichment": "on",
    "formula_render": "image",
    "math_backend": "mathjax",
    "ocr_mode": "rapidocr",
    "domain": "semiconductor",
}


async def test_translate_book_carries_the_engine_knobs_into_the_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tool signature *is* the payload, so a missing arg is a missing feature.

    ``_execute`` feeds the payload straight into the shared
    ``overrides_from_request`` mapping, so a knob absent from this signature is
    a knob an agent cannot ask for at all — the same defect the REST
    ``JobSubmitRequest`` had, where ``extra="forbid"`` turned it into a 422.

    Asserted on the config handed to the orchestrator, i.e. after
    payload -> overrides -> UBTConfig, so a knob that is accepted but then
    silently dropped fails here too.
    """
    import ubt.mcp.server as srv
    from ubt.core.engine.events import EventType, TranslationProgressEvent

    monkeypatch.setenv("UBT_ALLOWED_DIRS", str(tmp_path))
    doc = tmp_path / "mcp_knobs.md"
    doc.write_text("# C1\n\nBody text.\n", encoding="utf-8")

    captured: dict[str, Any] = {}

    class CapturingOrchestrator:
        def __init__(self, config: Any, **_kwargs: object) -> None:
            captured["config"] = config

        async def run(self, **kwargs: object) -> AsyncIterator[TranslationProgressEvent]:
            captured["run_kwargs"] = kwargs
            yield TranslationProgressEvent(
                event_type=EventType.JOB_STARTED,
                job_id="mcp_knobs",
                total_blocks=1,
                completed_blocks=0,
            )

    # dry_run keeps this off the real provider stack; the config build happens
    # before the dry-run branch, so the assertion still covers the real path.
    monkeypatch.setattr(srv, "create_dry_run_orchestrator", CapturingOrchestrator)

    result = await srv.ubt_translate_book(
        input_path=str(doc),
        dry_run=True,
        start_chapter=2,
        max_chapters=3,
        **_MCP_ENGINE_KNOBS,
    )
    job = srv._JOBS[result["job_id"]]
    assert job.task is not None, "the submit path must schedule the job task"
    await job.task

    config = captured["config"]
    for key, expected in _MCP_ENGINE_KNOBS.items():
        assert getattr(config, key) == expected, (
            f"{key!r}: tool arg {expected!r} did not reach UBTConfig (got {getattr(config, key)!r})"
        )

    # The chapter window is run-only: it must reach ``orchestrator.run``, which
    # ``run_kwargs_from_request`` is the single owner of.
    run_kwargs = captured["run_kwargs"]
    assert run_kwargs["start_chapter"] == 2
    assert run_kwargs["max_chapters"] == 3


def test_translate_book_signature_admits_no_credential_keys() -> None:
    """Widening the engine surface must not widen the credential surface.

    MCP has no request model with ``extra="forbid"``; its guard *is* the fixed
    signature, so the absence of these parameter names is the defence.
    ``overrides_from_request(allow_provider_keys=False)`` is the second,
    shared with REST — and it is what would fail loudly if one were added.
    """
    import inspect

    import ubt.mcp.server as srv

    params = set(inspect.signature(srv.ubt_translate_book).parameters)
    forbidden = {
        "api_key",
        "base_url",
        "api_mode",
        "ocr_api_key",
        "ocr_endpoint",
        "service_api_key",
        "provider_profile",
    }
    assert not (params & forbidden), sorted(params & forbidden)


def test_mcp_check_lang_rejects_unsupported_target() -> None:
    from ubt.core.exceptions import UBTError
    from ubt.mcp.server import _check_lang

    with pytest.raises(UBTError, match="[Uu]nsupported"):
        _check_lang("pt-BR", field="target_lang")
    # A supported region tag passes through untouched.
    assert _check_lang("zh-CN", field="target_lang") == "zh-CN"
    # Shape violations are still rejected by the regex guard.
    with pytest.raises(UBTError):
        _check_lang("not a lang!", field="target_lang")
