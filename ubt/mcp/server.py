"""UBT MCP server: agent-facing tools over the translation engine (stdio).

The tools (no agent loop inside — the engine stays deterministic; the
calling agent is the orchestrator):

- ``ubt_translate_book`` — submit a translation, returns ``job_id`` immediately
  (long books run minutes~hours; the background asyncio task owns the run).
- ``ubt_job_status`` — poll ledger-backed progress (memory record + SQLite
  fallback, mirroring the REST ``GET /jobs/{id}/status`` disk fallback).
- ``ubt_cancel_job`` — cancel an in-flight translation job (memory record + SQLite
  fallback, idempotent: terminal jobs return as-is).
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
import functools
import logging
import shutil
import time
from pathlib import Path
from typing import Any

try:
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
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
from ubt.api.manager import JobManager
from ubt.api.models import JobSubmitRequest
from ubt.core.config import MOCK_API_KEY, UBTConfig
from ubt.core.config import parse_page_ranges as parse_page_ranges

# Re-exported so the ceiling the server enforces is the number the registry
# carries (``import as`` is the explicit-reexport spelling mypy wants).
from ubt.core.engine.job_queue import TERMINAL_JOB_STATUSES, JobStatus
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.progress import ProgressSnapshot
from ubt.core.engine.writer_lock import LedgerWriterLock
from ubt.core.exceptions import LedgerWriterLockConflictError, UBTError
from ubt.core.fs_perms import (
    SYSTEM_DISALLOWED_PREFIXES,
    is_sensitive_path_part,
)
from ubt.core.job_options import (
    JOB_ID_MAX_LEN,
    LANG_CODE_RE,
    apply_config_overrides,
    default_output_path,
    job_id_is_valid,
    profile_name_is_valid,
    resolve_target_output,
    validate_request_enums,
)
from ubt.core.job_options import (
    JOB_ID_RE as JOB_ID_RE,
)
from ubt.core.language_profile import is_supported_lang, supported_lang_codes
from ubt.core.policy.layout_policy import MCP_MAX_RUNNING_JOBS as MCP_MAX_RUNNING_JOBS

mcp = MCPServer(name="ubt")


def _mcp_error_boundary(func: Any) -> Any:
    """Decorator converting anticipated domain errors into ToolError for clean MCP error responses."""

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await func(*args, **kwargs)
        except ToolError:
            raise
        except (UBTError, ValueError, FileNotFoundError) as err:
            raise ToolError(str(err)) from err

    return wrapper


logger = logging.getLogger(__name__)

#: How long a disk-path cancel waits for the pipeline's writer lock before
#: leaving the terminal write to the running pipeline (mirrors the REST wait).
_CANCEL_LOCK_WAIT_SEC = 10.0


_MAX_RETAINED = 100

#: The shared in-process job lifecycle: concurrency cap, retention pruning, the
#: orchestrator build + one-owner progress fold, and the cancel/abort handling.
#: MCP used to reimplement all of it; one implementation also means the
#: REST and agent surfaces cannot drift. ``ubt.api`` is a lazy package and
#: ``JobManager`` imports no FastAPI, so the MCP extra stays light.
_MANAGER = JobManager(
    max_running_jobs=MCP_MAX_RUNNING_JOBS,
    max_retained_jobs=_MAX_RETAINED,
)

#: Slots for ``ubt_assess_book(deep=True)``.
#:
#: Deep assess runs the full adapter ingest (Docling model load — minutes of
#: GPU/CPU per document), and the REST ``POST /jobs/assess`` gates exactly the
#: same work behind ``assess_semaphore`` (``ubt/api/app.py``) while MCP had no
#: ceiling at all: a burst of deep assesses from an agent piled unbounded heavy
#: ingest onto this process. Two slots, waiters queue —
#: MCP tool calls are already long-running by design. ``MCP_MAX_RUNNING_JOBS``
#: keeps governing translations only. Shallow assesses stay ungated but are
#: *not* free: they skip the Docling ingest yet still run the full pdfium page
#: census + font-encoding witness (``assess._pdf_facts``). That work is
#: serialized process-wide by ``PDFIUM_LOCK`` (ubt/adapters/pdf/pdfium_gate.py),
#: which is what keeps a shallow burst from exhausting threads.
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
            f"{', '.join(supported_lang_codes())} (region tags such as 'zh-CN' are accepted)."
        )
    return code


def _check_pages(pages: str | None) -> str | None:
    """Entry-layer page-range guard, matching the REST ``JobSubmitRequest`` 422.

    Without it a malformed range passed the tool and failed the job
    asynchronously (or 500'd the assess path); a multi-MB value was also
    materialized before the cap.
    """
    if pages is None:
        return None
    try:
        parse_page_ranges(pages)
    except ValueError as exc:
        raise UBTError(f"Invalid pages {pages!r}: {exc}") from exc
    return pages


def _check_profile(profile: str) -> str:
    """Entry-layer guard: a profile names a packaged resource dir, never a path.

    Same shape the REST ``JobSubmitRequest.profile`` enforces. Without it, a
    prompt-injected agent could pass ``/tmp/x`` and read ``/tmp/x/en-zh.json``
    through ``seed_entries_for_profile``, escaping the MCP path sandbox.
    """
    if not profile_name_is_valid(profile):
        raise UBTError(
            f"Invalid profile {profile!r}: letters/digits/-/_ only (e.g. general, textbook, paper)."
        )
    return profile


def _sandbox_path(raw: str, *, must_exist: bool) -> Path:
    """Resolve a caller-supplied path inside the shared UBT path sandbox.

    The tools used to reject only a literal ``..``, so any writable path an
    agent named was accepted — and ``ubt_job_status(db_dir=...)`` would then
    create and open ``<job_id>.sqlite`` there read-write, schema init included
    without sandbox restriction.

    The allowlist is the REST ``resolve_secure_path`` default, re-implemented
    here instead of imported on purpose: ``ubt.api.security`` imports
    ``fastapi`` at module scope, and the stdio MCP server must not require a
    web-framework dependency to run. The allowlist *primitives* are shared —
    ``SYSTEM_DISALLOWED_PREFIXES`` and ``is_sensitive_path_part`` come from
    ``ubt.core.fs_perms`` — and the base precedence (``UBT_ALLOWED_DIRS`` when
    the operator sets it, else the working directory plus ``config.db_dir``)
    matches ``ubt/api/security.py``.
    """
    candidate = str(raw).strip()
    if not candidate:
        raise UBTError("Empty path provided")
    if ".." in Path(candidate).parts:
        raise UBTError(f"Refusing path with '..': {raw!r}")
    config = UBTConfig.from_env()
    configured_bases = config.allowed_base_dirs()
    bases = configured_bases or [Path.cwd().resolve(), config.db_dir.resolve()]
    try:
        path = Path(candidate).expanduser()
        resolved = (bases[0] / path).resolve() if not path.is_absolute() else path.resolve()
    except (OSError, RuntimeError) as err:
        raise UBTError(f"Invalid path format: {err}") from err
    # System directories are off-limits unless the operator explicitly widened
    # the sandbox (same precedence as REST resolve_secure_path): an implicit
    # base such as cwd=/, or a db_dir under /var, must not expose /etc, /proc….
    if not configured_bases:
        for disallowed in SYSTEM_DISALLOWED_PREFIXES:
            if resolved == disallowed or disallowed in resolved.parents:
                raise UBTError(
                    f"Access denied: path accesses a restricted system directory: {raw!r}"
                )
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
    # ``ubt.core.fs_perms`` — importing ``ubt.api.security`` would drag in
    # fastapi, which the stdio MCP server does not require (see this
    # function's docstring).
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


@mcp.tool()
@_mcp_error_boundary
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
    # Run scoping and resume controls — the same keys the REST payload carries.
    # ``start_chapter``/``max_chapters`` are run-only: they ride into
    # ``orchestrator.run`` via the shared ``run_kwargs_from_request``. ``fresh``
    # is a UBTConfig field (discard prior ledger state instead of resuming).
    start_chapter: int | None = None,
    max_chapters: int | None = None,
    fresh: bool | None = None,
    # Engine knobs — parity with the REST ``JobSubmitRequest``, which carries
    # the same set. None-defaults for the same reason as above; note that a
    # ``False`` default on a bool would be a *silent override* of the
    # operator's ``UBT_*`` environment, so the optionals stay ``bool | None``.
    budget_usd: float | None = None,
    max_concurrency: int | None = None,
    batch_limit: int | None = None,
    macro_chunk_size: int | None = None,
    short_max_pages: int | None = None,
    enable_rolling_summary: bool | None = None,
    chapter_streaming_enabled: bool | None = None,
    offline_batch_enabled: bool | None = None,
    qe_engine: str | None = None,
    visual_judge_enabled: bool | None = None,
    visual_judge_model: str | None = None,
    prompt_strategy: str | None = None,
    translate_chrome: bool | None = None,
    facing_spread: bool | None = None,
    emit_both: bool | None = None,
    cover_mode: str | None = None,
    formula_enrichment: str | None = None,
    formula_render: str | None = None,
    math_backend: str | None = None,
    ocr_mode: str | None = None,
    domain: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Translate a document end to end. Returns immediately with a job_id; poll ubt_job_status."""
    _check_lang(target_lang, "target_lang")
    _check_lang(source_lang, "source_lang")
    profile = _check_profile(profile)
    pages = _check_pages(pages)
    resolved = _resolve_input(input_path)
    if output_path is not None:
        resolved_out = _safe_output_path(output_path)
        target_candidate = resolve_target_output(resolved_out, resolved)
        if target_candidate.exists() and not fresh:
            raise ToolError(f"output_path already exists; refusing to overwrite it: {output_path}")
        output_path = str(target_candidate)
    else:
        # The implicit deliverable defaults to ~/Documents/UBT, which is outside
        # the MCP sandbox; REST relocates it inside the allowlist, so MCP must
        # too instead of writing outside its own contract. Relocate into the
        # input book's directory (REST parity) or allowed bases with the same file name.
        candidate = default_output_path(resolved)
        try:
            _sandbox_path(str(candidate), must_exist=False)
            output_path = str(candidate)
        except UBTError:
            config_env = UBTConfig.from_env()
            bases = config_env.allowed_base_dirs()
            if any(resolved.parent == b or b in resolved.parent.parents for b in bases):
                relocated = resolved.parent / candidate.name
            elif bases:
                relocated = bases[0] / candidate.name
            else:
                relocated = Path(config_env.db_dir) / candidate.name
            output_path = str(_sandbox_path(str(relocated), must_exist=False))
        if Path(output_path).exists() and not fresh:
            raise ToolError(f"output_path already exists; refusing to overwrite it: {output_path}")
    jid = _check_job_id(job_id) if job_id else ""
    # A stable job_id that is still live is a resubmit, not a second run.
    existing = _MANAGER.get_job(jid) if jid else None
    if existing is not None and existing.status in (JobStatus.SUBMITTED, JobStatus.RUNNING):
        raise UBTError(f"Job {jid} is already running")

    base_config = UBTConfig.from_env()
    if db_dir:
        # A server-side path: it rides on the base config rather than the request
        # (whose model is ``extra="forbid"`` for REST parity). The shared
        # ``execute_job`` applies the request overrides on top of this config.
        base_config = apply_config_overrides(
            base_config, {"db_dir": str(_safe_output_path(db_dir))}
        )
    # Rehearsal when asked, or when no key is configured: a keyless stdio server
    # must label mock output instead of returning it as a delivery (same rule as
    # the REST intake).
    key = base_config.api_key.get_secret_value()
    rehearsal = dry_run or not key or key == MOCK_API_KEY
    payload: dict[str, Any] = {
        "input_path": str(resolved),
        "output_path": output_path,
        "target_lang": target_lang,
        "source_lang": source_lang,
        "profile": profile,
        "draft_model": draft_model,
        "repair_model": repair_model,
        "exec_mode": exec_mode,
        "formula_mode": formula_mode,
        "render_engine": render_engine,
        "dual_mode": dual_mode,
        "preset": preset,
        "glossary": str(_resolve_input(glossary)) if glossary else None,
        "pages": pages,
        "start_chapter": start_chapter,
        "max_chapters": max_chapters,
        "fresh": fresh,
        "budget_usd": budget_usd,
        "max_concurrency": max_concurrency,
        "batch_limit": batch_limit,
        "macro_chunk_size": macro_chunk_size,
        "short_max_pages": short_max_pages,
        "enable_rolling_summary": enable_rolling_summary,
        "chapter_streaming_enabled": chapter_streaming_enabled,
        "offline_batch_enabled": offline_batch_enabled,
        "qe_engine": qe_engine,
        "visual_judge_enabled": visual_judge_enabled,
        "visual_judge_model": visual_judge_model,
        "prompt_strategy": prompt_strategy,
        "translate_chrome": translate_chrome,
        "facing_spread": facing_spread,
        "emit_both": emit_both,
        "cover_mode": cover_mode,
        "formula_enrichment": formula_enrichment,
        "formula_render": formula_render,
        "math_backend": math_backend,
        "ocr_mode": ocr_mode,
        "domain": domain,
        "dry_run": rehearsal,
    }
    # Reject an out-of-vocabulary enum here, synchronously: otherwise the tool
    # returned a job_id and the job only failed once the background task tried to
    # apply the value (REST 422s upfront, so this restores parity).
    validate_request_enums(payload)
    request = JobSubmitRequest.model_validate(payload)
    record = _MANAGER.create_job(request, job_id=jid or None)
    record.task = asyncio.create_task(_MANAGER.execute_job(record, base_config))
    return {"job_id": record.job_id, "status": JobStatus.SUBMITTED, "rehearsal": rehearsal}


@mcp.tool()
@_mcp_error_boundary
async def ubt_job_status(job_id: str, db_dir: str | None = None) -> dict[str, Any]:
    """Poll translation progress. Falls back to the SQLite ledger when the job is unknown in memory."""
    jid = _check_job_id(job_id)
    rec = _MANAGER.get_job(jid)
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
            "rehearsal": rec.request.dry_run,
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
@_mcp_error_boundary
async def ubt_cancel_job(job_id: str, db_dir: str | None = None) -> dict[str, Any]:
    """Cancel an in-flight translation job. Idempotent: terminal jobs return their status as-is."""
    jid = _check_job_id(job_id)
    rec = _MANAGER.get_job(jid)
    if rec is not None:
        if rec.status not in (JobStatus.SUBMITTED, JobStatus.RUNNING):
            return {"job_id": jid, "status": rec.status}
        rec.status = JobStatus.CANCELLED
        task = rec.task
        if task is not None and not task.done():
            task.cancel()
        return {"job_id": jid, "status": JobStatus.CANCELLED}

    base = _safe_output_path(str(db_dir)) if db_dir else UBTConfig.from_env().db_dir
    db_path = base / f"{jid}.sqlite"
    if not db_path.exists():
        raise ToolError(f"no such job: {jid}")
    ledger = SQLiteJobLedger(db_path)
    try:
        current_status = await asyncio.to_thread(ledger.get_job_status, jid)
        if current_status is None:
            raise ToolError(f"no such job: {jid}")
        if current_status not in TERMINAL_JOB_STATUSES:
            # The running pipeline holds the job-level writer lock for the whole
            # run; taking it here too is what keeps a cancellation from
            # clobbering the pipeline's checkpoints (mirrors the REST surface).
            lock = LedgerWriterLock(db_path, jid)
            deadline = time.monotonic() + _CANCEL_LOCK_WAIT_SEC
            while True:
                try:
                    lock.acquire()
                    break
                except LedgerWriterLockConflictError:
                    if time.monotonic() >= deadline:
                        logger.info(
                            "MCP cancel for %s: writer lock still held; the running "
                            "pipeline will finalize it.",
                            jid,
                        )
                        return {"job_id": jid, "status": current_status}
                    await asyncio.sleep(0.1)
            try:
                # Re-check under the lock: the pipeline may have finalized while
                # we waited, and a finished job must not be rewritten.
                status_now = await asyncio.to_thread(ledger.get_job_status, jid)
                if status_now is not None and status_now not in TERMINAL_JOB_STATUSES:
                    await asyncio.to_thread(ledger.finalize_job, jid, status=JobStatus.CANCELLED)
                    return {"job_id": jid, "status": JobStatus.CANCELLED}
                return {"job_id": jid, "status": status_now or current_status}
            finally:
                lock.release()
        return {"job_id": jid, "status": current_status}
    finally:
        ledger.close()


@mcp.tool()
@_mcp_error_boundary
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
@_mcp_error_boundary
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
@_mcp_error_boundary
async def ubt_doctor() -> dict[str, Any]:
    """Preflight checks: API key, writable ledger dir, key optional deps."""
    config = UBTConfig.from_env()
    checks: list[dict[str, str]] = []

    def _record(name: str, status: str, detail: str = "") -> None:
        checks.append({"name": name, "status": status, "detail": detail})

    key = config.api_key.get_secret_value()
    if not key or key == MOCK_API_KEY:
        _record("api_key", "FAIL", "set UBT_LLM_API_KEY or select a provider (UBT_PROVIDER)")
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
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
