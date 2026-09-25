"""UBT MCP server: agent-facing tools over the translation engine (stdio).

The tools (no agent loop inside — the engine stays deterministic; the
calling agent is the orchestrator):

- ``ubt_translate_book`` — submit a translation, returns ``job_id`` immediately
  (long books run minutes~hours; the background asyncio task owns the run).
- ``ubt_job_status`` — poll ledger-backed progress (memory record + SQLite
  fallback, mirroring the REST ``GET /jobs/{id}/status`` disk fallback).
- ``ubt_inspect_book`` — manifest JSON (title/doc_id/chapters), no Rich text.
- ``ubt_doctor`` — preflight checks (API key, writable ledger dir, deps).

Transport: stdio (local-trust: same user, no X-API-Key gate — path inputs
still pass the shared allowlist sandbox in ``_sandbox_path``, the REST
``resolve_secure_path`` default re-implemented locally, and remote use should
front the REST API which enforces it plus auth).
Entry point: ``ubt-mcp`` (or ``python -m ubt.mcp``; ``python -m ubt.mcp.server``
runs the module directly). The console-script wrappers live in
``ubt.mcp._entrypoint`` so a missing ``mcp`` extra exits cleanly instead of
surfacing this module's import guard as a traceback.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from mcp.server.mcpserver import MCPServer
except ModuleNotFoundError as exc:  # pragma: no cover - install-shape guard
    # Only a missing ``mcp`` itself means the extra is absent. A transitive
    # import that fails *inside* an installed mcp (e.g. jsonschema) must keep
    # its own error, or the operator gets a wrong "install [mcp]" hint that
    # permanently hides the real cause.
    if exc.name != "mcp" and not (exc.name or "").startswith("mcp."):
        raise
    # ``[project.scripts]`` entry points are installed unconditionally, so a
    # base ``pip install universal-book-translator`` creates a ``ubt-mcp``
    # command whose module imports ``mcp`` at module scope.
    #
    # A normal, catchable exception — not ``SystemExit``: importing this module
    # is not the same act as running the ``ubt-mcp`` command, and library
    # callers (agent frameworks probing which tools exist) must be able to
    # handle the missing extra without their own process exiting.
    # ``ubt.mcp._entrypoint`` restores the clean CLI exit for the console script.
    from ubt.core.exceptions import OptionalDependencyError

    raise OptionalDependencyError(
        "The MCP server needs the optional 'mcp' extra:\n"
        "    pip install 'universal-book-translator[mcp]'"
    ) from exc

from ubt.adapters import get_adapter_for_path
from ubt.core.config import MOCK_API_KEY, UBTConfig
from ubt.core.engine.dry_run import create_dry_run_orchestrator
from ubt.core.engine.events import TranslationProgressEvent

# Re-exported so the ceiling the server enforces is the number the registry
# carries (``import as`` is the explicit-reexport spelling mypy wants).
from ubt.core.engine.job_queue import TERMINAL_JOB_STATUSES, JobStatus
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.pipeline import PipelineOrchestrator
from ubt.core.engine.progress import ARTIFACT_KEYS, ProgressSnapshot
from ubt.core.exceptions import UBTError
from ubt.core.fs_perms import is_sensitive_path_part, restrict_env_file
from ubt.core.job_options import (
    JOB_ID_MAX_LEN,
    LANG_CODE_RE,
    apply_config_overrides,
    job_id_is_valid,
    overrides_from_request,
    run_kwargs_from_request,
)
from ubt.core.job_options import (
    JOB_ID_RE as JOB_ID_RE,
)
from ubt.core.language_profile import is_supported_lang
from ubt.core.policy.layout_policy import MCP_MAX_RUNNING_JOBS as MCP_MAX_RUNNING_JOBS

mcp = MCPServer(name="ubt")

logger = logging.getLogger(__name__)


@dataclass
class _JobRecord:
    job_id: str
    status: str = JobStatus.SUBMITTED
    progress: ProgressSnapshot = field(default_factory=ProgressSnapshot)
    error: str | None = None
    rehearsal: bool = False
    task: asyncio.Task[None] | None = field(default=None, repr=False)


_JOBS: dict[str, _JobRecord] = {}
_MAX_RETAINED = 100

#: Slots for ``ubt_assess_book(deep=True)``.
#:
#: Deep assess runs the full adapter ingest (Docling model load — minutes of
#: GPU/CPU per document), and the REST ``POST /jobs/assess`` gates exactly the
#: same work behind ``assess_semaphore`` (``ubt/api/app.py``) while MCP had no
#: ceiling at all: a burst of deep assesses from an agent piled unbounded heavy
#: ingest onto this process. Two slots, waiters queue —
#: MCP tool calls are already long-running by design. ``MCP_MAX_RUNNING_JOBS``
#: keeps governing translations only; shallow assesses stay ungated (they are
#: a cheap manifest/profile read).
_DEEP_ASSESS_SLOTS = 2
_deep_assess_semaphore = asyncio.Semaphore(_DEEP_ASSESS_SLOTS)


def _check_job_id(job_id: str) -> str:
    if not job_id_is_valid(job_id):
        raise UBTError(
            f"Invalid job_id {job_id!r}: letters/digits/-/_ only, up to {JOB_ID_MAX_LEN} characters"
        )
    return job_id


def _check_lang(code: str, field: str) -> str:
    """Entry-layer language-code guard, same pattern the REST API enforces.

    Defense in depth: the render-side ``sanitize_lang_tag`` already seals every
    Typst interpolation, so this is not closing a live injection — it makes the
    four entry points validate consistently instead of only the API, so a future
    consumer that skips the render gate cannot reintroduce the class.
    """
    if not LANG_CODE_RE.fullmatch(code):
        raise UBTError(
            f"Invalid {field} {code!r}: expected an ISO-ish language code (e.g. zh, en, zh-CN)"
        )
    if field == "target_lang" and not is_supported_lang(code):
        raise UBTError(
            f"Unsupported {field} {code!r}. Supported base languages: "
            "zh, en, ja, ko, fr, de, es, ru (region tags such as 'zh-CN' are accepted)."
        )
    return code


def _sandbox_path(raw: str, *, must_exist: bool) -> Path:
    """Resolve a caller-supplied path inside the shared UBT path sandbox.

    The tools used to reject only a literal ``..``, so any writable path an
    agent named was accepted — and ``ubt_job_status(db_dir=...)`` would then
    create and open ``<job_id>.sqlite`` there read-write, schema init included
    without sandbox restriction.

    The allowlist is the REST ``resolve_secure_path`` default, re-implemented
    here instead of imported on purpose: ``ubt.api``'s package ``__init__``
    imports ``ubt.api.app``, whose module body *builds the FastAPI app and
    reconfigures root logging* — unacceptable side effects inside a stdio MCP
    server. The allowlist *source* is still shared: ``UBT_ALLOWED_DIRS`` when
    the operator sets it, else the working directory plus ``config.db_dir``,
    exactly like ``ubt/api/security.py``.
    """
    candidate = str(raw).strip()
    if not candidate:
        raise UBTError("Empty path provided")
    if ".." in Path(candidate).parts:
        raise UBTError(f"Refusing path with '..': {raw!r}")
    config = UBTConfig.from_env()
    bases = config.allowed_base_dirs() or [Path.cwd().resolve(), config.db_dir.resolve()]
    try:
        path = Path(candidate).expanduser()
        resolved = (bases[0] / path).resolve() if not path.is_absolute() else path.resolve()
    except (OSError, RuntimeError) as err:
        raise UBTError(f"Invalid path format: {err}") from err
    if not any(resolved == base or base in resolved.parents for base in bases):
        # Generic wording (mirrors the REST 403): the bases are server-side
        # paths and must not be echoed to a caller — echo the caller's own
        # path back instead, which is what the agent needs to correct course.
        raise UBTError(
            f"Access denied: path is outside the allowed directories "
            f"(set UBT_ALLOWED_DIRS to widen the sandbox): {raw!r}"
        )
    # Same sensitive-name rule as the REST sandbox, and it beats the allowlist
    # there too (``resolve_secure_path`` docstring point 4): an MCP server
    # started in $HOME has its whole home directory inside the default
    # allowlist, and a prompt-injected agent must not be able to read
    # ``.ssh``/``.env``/``credentials.json`` through these tools. Shared from
    # ``ubt.core.fs_perms`` — importing ``ubt.api.security`` would build the
    # FastAPI app as a side effect (see this function's docstring).
    for part in resolved.parts:
        if is_sensitive_path_part(part):
            raise UBTError(
                f"Access denied: accessing sensitive configuration directory or file {part!r} is prohibited"
            )
    if must_exist and not resolved.exists():
        raise UBTError(f"Source file not found: {raw}")
    return resolved


def _resolve_input(raw: str) -> Path:
    """Sandboxed input path: must be inside the allowlist *and* exist."""
    return _sandbox_path(raw, must_exist=True)


def _safe_output_path(raw: str) -> Path:
    """Sandboxed output/db path: inside the allowlist, existence not required.

    Outputs must not escape the sandbox (the caller's shell user is the trust
    boundary, so this blocks accidental or injected traversal); they are exempt
    from the existence check because the pipeline creates them.
    """
    return _sandbox_path(raw, must_exist=False)


def _prune_jobs() -> None:
    if len(_JOBS) < _MAX_RETAINED:
        return
    done = [jid for jid, r in _JOBS.items() if r.status in TERMINAL_JOB_STATUSES]
    for jid in done[: len(_JOBS) - _MAX_RETAINED + 1]:
        del _JOBS[jid]


async def _execute(job_id: str, payload: dict[str, Any]) -> None:
    """Background translation task: owns its orchestrator end to end."""
    rec = _JOBS[job_id]
    rec.status = JobStatus.RUNNING
    try:
        # Config construction and validation live inside the try: with
        # validate_assignment=True an invalid enum would otherwise raise out
        # of the background task and leave the job "running" forever. The
        # shared mapping is the same one the CLI/API use, so
        # preset/glossary/pages are carried here too.
        overrides = overrides_from_request(payload, allow_provider_keys=False)
        if payload.get("db_dir"):
            overrides["db_dir"] = _safe_output_path(str(payload["db_dir"]))
        config = apply_config_overrides(UBTConfig.from_env(), overrides)
        output_path = (
            _safe_output_path(str(payload["output_path"])) if payload.get("output_path") else None
        )

        def _persist_final(event: TranslationProgressEvent) -> None:
            progress = ProgressSnapshot.from_event(event)
            ledger_path = Path(config.db_dir) / f"{job_id}.sqlite"
            if not ledger_path.exists():
                return
            with SQLiteJobLedger(ledger_path) as ldg:
                for metadata_key in (*ARTIFACT_KEYS, "estimated_cost_usd"):
                    value = getattr(progress, metadata_key)
                    if value is not None:
                        ldg.set_job_metadata_value(job_id, metadata_key, value)

        if payload.get("dry_run"):
            orchestrator = create_dry_run_orchestrator(config, finalize_job=_persist_final)
        else:
            orchestrator = PipelineOrchestrator(config=config, finalize_job=_persist_final)
        run_kwargs = run_kwargs_from_request({**payload, "job_id": job_id})
        async for event in orchestrator.run(
            input_path=Path(str(payload["input_path"])),
            output_path=output_path,
            **run_kwargs,
        ):
            rec.progress = ProgressSnapshot.from_event(event)
        rec.status = JobStatus.COMPLETED
    except asyncio.CancelledError:
        # Mark cancelled task status explicitly so job slots are freed and resubmission succeeds.
        rec.status = JobStatus.CANCELLED
        logger.warning("MCP job %s task cancelled", job_id)
        raise
    except Exception as exc:
        logger.warning("MCP job %s failed: %s", job_id, exc)
        rec.status = JobStatus.FAILED
        # Avoid echoing raw exception text (may contain paths or source text).
        rec.error = f"{type(exc).__name__} (see server logs; job_id={job_id})"


@mcp.tool()
async def ubt_translate_book(
    input_path: str,
    target_lang: str = "zh",
    profile: str = "general",
    source_lang: str = "en",
    output_path: str | None = None,
    job_id: str | None = None,
    draft_model: str | None = None,
    repair_model: str | None = None,
    # None-defaults throughout: _run_translation_job only setattr()s non-None
    # payload values, so an unset tool arg never overwrites the operator's
    # UBT_EXEC_MODE / UBT_FORMULA_MODE environment (same convention as the CLI).
    exec_mode: str | None = None,
    formula_mode: str | None = None,
    render_engine: str | None = None,
    dual_mode: str | None = None,
    db_dir: str | None = None,
    preset: str | None = None,
    glossary: str | None = None,
    pages: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Translate a document end to end. Returns immediately with a job_id; poll ubt_job_status."""
    _check_lang(target_lang, "target_lang")
    _check_lang(source_lang, "source_lang")
    _prune_jobs()
    running = sum(
        1 for rec in _JOBS.values() if rec.status in (JobStatus.SUBMITTED, JobStatus.RUNNING)
    )
    if running >= MCP_MAX_RUNNING_JOBS:
        raise UBTError(
            f"Too many concurrent jobs ({running}/{MCP_MAX_RUNNING_JOBS}); "
            "wait for one to finish before submitting another."
        )
    resolved = _resolve_input(input_path)
    jid = _check_job_id(job_id) if job_id else f"job_{uuid.uuid4().hex[:12]}"
    if jid in _JOBS and _JOBS[jid].status in (JobStatus.SUBMITTED, JobStatus.RUNNING):
        raise UBTError(f"Job {jid} is already running")
    _JOBS[jid] = _JobRecord(job_id=jid)
    # Rehearsal when asked, or when no key is configured: a keyless stdio
    # server must label mock output instead of returning it as a delivery
    # (same rule as the REST intake).
    rehearsal = dry_run or UBTConfig.from_env().api_key.get_secret_value() == MOCK_API_KEY
    _JOBS[jid].rehearsal = rehearsal
    payload: dict[str, Any] = {
        "input_path": str(resolved),
        "output_path": output_path,
        "target_lang": target_lang,
        "profile": profile,
        "source_lang": source_lang,
        "draft_model": draft_model,
        "repair_model": repair_model,
        "exec_mode": exec_mode,
        "formula_mode": formula_mode,
        "render_engine": render_engine,
        "dual_mode": dual_mode,
        "db_dir": db_dir,
        "preset": preset,
        "glossary_path": str(_resolve_input(glossary)) if glossary else None,
        "pages": pages,
        "dry_run": rehearsal,
    }
    _JOBS[jid].task = asyncio.create_task(_execute(jid, payload))
    return {"job_id": jid, "status": JobStatus.SUBMITTED, "rehearsal": rehearsal}


@mcp.tool()
async def ubt_job_status(job_id: str, db_dir: str | None = None) -> dict[str, Any]:
    """Poll translation progress. Falls back to the SQLite ledger when the job is unknown in memory."""
    jid = _check_job_id(job_id)
    rec = _JOBS.get(jid)
    if rec is not None:
        live: dict[str, Any] = {
            "job_id": jid,
            "status": rec.status,
            "total": rec.progress.total_blocks,
            "completed": rec.progress.completed_blocks,
            "repaired": rec.progress.repaired_blocks,
            "failed": rec.progress.failed_blocks,
            "avg_qe_score": rec.progress.current_avg_qe,
            "output_file": rec.progress.output_file,
            "report_file": rec.progress.report_file,
            "visual_report_file": rec.progress.visual_report_file,
            "rehearsal": rec.rehearsal,
        }
        if rec.error:
            live["error"] = rec.error
        return live
    # Disk fallback (mirrors REST status fallback; never fabricates cost).
    # db_dir goes through the same traversal guard ubt_job_submit applies:
    # SQLiteJobLedger opens its target read-write with schema init, so a
    # raw caller path is exactly what the guard exists for.
    base = _safe_output_path(str(db_dir)) if db_dir else UBTConfig.from_env().db_dir
    db_path = base / f"{jid}.sqlite"
    if not db_path.exists():
        return {"job_id": jid, "status": "unknown", "error": f"no such job: {jid}"}
    ledger = SQLiteJobLedger(db_path, read_only=True)
    try:
        # Off-loop: a GB-scale ledger read blocks the event loop for 10-35 ms
        # per poll; MCP status is an async endpoint, so
        # unlike the REST sync endpoints it does not get a threadpool for free.
        stats = await asyncio.to_thread(ledger.get_job_stats, jid)
        if not stats:
            return {"job_id": jid, "status": "empty"}
        progress = await asyncio.to_thread(
            ProgressSnapshot.from_ledger,
            stats,
            metadata=lambda k: ledger.get_job_metadata_value(jid, k),
        )
        # Honor the persisted terminal status first — a job that
        # FAILED before the restart must not read as "running" forever to a
        # polling agent.
        persisted = await asyncio.to_thread(ledger.get_job_status, jid)
        if persisted == "failed":
            status = JobStatus.FAILED
        elif persisted == "cancelled":
            # The pipeline records a cancel as terminal; reading it as "running"
            # would leave a polling agent waiting on a job that already stopped.
            status = JobStatus.CANCELLED
        elif persisted == "completed" or (progress.completed_blocks >= progress.total_blocks > 0):
            status = JobStatus.COMPLETED
        else:
            status = JobStatus.RUNNING
        disk: dict[str, Any] = {
            "job_id": jid,
            "status": status,
            "total": progress.total_blocks,
            "completed": progress.completed_blocks,
            "repaired": progress.repaired_blocks,
            "failed": progress.failed_blocks,
            "avg_qe_score": progress.current_avg_qe,
            "output_file": progress.output_file,
            "report_file": progress.report_file,
            "visual_report_file": progress.visual_report_file,
        }
        if status == JobStatus.FAILED:
            disk["error"] = "job terminated with status=failed before the restart"
        return disk
    finally:
        ledger.close()


@mcp.tool()
async def ubt_inspect_book(input_path: str) -> dict[str, Any]:
    """Return a document manifest (title/doc_id/chapters) as JSON-serializable dict."""
    resolved = _resolve_input(input_path)
    # Same engine the pipeline resolves, so the manifest matches real parsing.
    adapter = get_adapter_for_path(resolved, pdf_engine=UBTConfig.from_env().pdf_engine)
    manifest = await adapter.extract_manifest(resolved)
    return {
        "title": manifest.title,
        "doc_id": manifest.doc_id,
        "source_path": str(resolved),
        "chapters": [
            {"chapter_id": c.chapter_id, "title": c.title, "spine_index": c.spine_index}
            for c in manifest.chapters
        ],
    }


@mcp.tool()
async def ubt_assess_book(
    input_path: str,
    target_lang: str = "zh",
    source_lang: str = "en",
    deep: bool = False,
) -> dict[str, Any]:
    """Profile a cold document into a quote (route, expected cost, risks) before translating."""
    _check_lang(target_lang, "target_lang")
    _check_lang(source_lang, "source_lang")
    from ubt.core.assess import assess_document_async

    resolved = _resolve_input(input_path)
    config = UBTConfig.from_env()

    async def _assess() -> dict[str, Any]:
        report = await assess_document_async(
            resolved,
            config,
            deep=deep,
            target_lang=target_lang,
            source_lang=source_lang,
        )
        return report.to_dict()

    if deep:
        # Queue behind the deep-ingest ceiling (see _DEEP_ASSESS_SLOTS).
        async with _deep_assess_semaphore:
            return await _assess()
    return await _assess()


@mcp.tool()
async def ubt_doctor() -> dict[str, Any]:
    """Preflight checks: API key, writable ledger dir, key optional deps."""
    config = UBTConfig.from_env()
    checks: list[dict[str, str]] = []

    def _record(name: str, status: str, detail: str = "") -> None:
        checks.append({"name": name, "status": status, "detail": detail})

    key = config.api_key.get_secret_value()
    if not key or key == MOCK_API_KEY:
        _record("api_key", "FAIL", "set UBT_LLM_API_KEY or OPENAI_API_KEY")
    else:
        _record("api_key", "OK", f"base={config.base_url}")
    try:
        config.db_dir.mkdir(parents=True, exist_ok=True)
        probe = config.db_dir / ".ubt_doctor_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        _record("ledger_dir", "OK", str(config.db_dir))
    except OSError as exc:
        _record("ledger_dir", "FAIL", str(exc))
    import importlib.util as _ilu

    for dep in ("pypdfium2", "docx", "bs4"):
        _record(
            "dep:" + dep,
            "OK" if _ilu.find_spec(dep) is not None else "WARN",
            "missing — install the feature extra" if _ilu.find_spec(dep) is None else "",
        )
    _record(
        "typst",
        "OK" if shutil.which("typst") else "WARN",
        "" if shutil.which("typst") else "PDF reflow falls back to markdown companion",
    )
    ok = all(c["status"] != "FAIL" for c in checks)
    return {"ok": ok, "checks": checks}


def main() -> None:
    """Stdio entry point (``ubt-mcp`` script + ``python -m ubt.mcp``)."""
    # Same ``.env`` convergence as the API bootstrap (review M1): every tool
    # builds ``UBTConfig.from_env()``, which reads ``.env`` from the cwd.
    restrict_env_file()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
