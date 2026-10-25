"""Self-check configuration, credentials, and local environment."""

import importlib
import importlib.util
import json
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import httpx
import typer
from pydantic import ValidationError
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from ubt.cache.dirs import cache_root
from ubt.core.config import UBTConfig, credential_required, has_real_credential
from ubt.core.exceptions import DocumentParseError, UBTError
from ubt.core.fs_perms import world_readable_files
from ubt.core.job_options import default_output_dir_for_scan
from ubt.core.router.pricing import (
    billing_enabled_for_local_endpoints,
    endpoint_is_local,
    price_is_known,
)

console = Console()

#: Render order for the human checklist. Every check is tagged with one of
#: these when recorded, so the flat probe sequence below still groups.
#: Render order for the human checklist, as ``(code, English label)``. The code is
#: what a localized console switches on; the label is what the CLI prints.
_GROUPS: tuple[tuple[str, str], ...] = (
    ("credentials", "Credentials & models"),
    ("storage", "Storage & privacy"),
    ("quality", "Quality & optional tiers"),
    ("rendering", "Rendering & toolchain"),
)

_STATUS_MARK = {"OK": "\u2713", "WARN": "\u26a0", "FAIL": "\u2717", "SKIP": "\u00b7"}

#: Every check code this module can emit, and the console's translation contract.
#:
#: Stated explicitly because a consumer cannot reliably recover the set from the
#: source: ``probe_writable`` builds its codes with an f-string (``f"{code}.writable"``),
#: which no scan of string literals can see. A test asserts the catalogue in
#: ``web/src/i18n/translations/doctorCodes.ts`` covers exactly these, so adding a
#: check without a translation fails CI instead of shipping an English line into
#: the Chinese console.
CHECK_CODES: dict[str, tuple[str, ...]] = {
    "names": (
        "api_key",
        "base_url",
        "live_probe",
        "models",
        "cost_pricing",
        "ledger_dir",
        "ledger_permissions",
        "translation_memory",
        "disk_space",
        "sqlite_wal",
        "page_images",
        "qe",
        "batch_api",
        "pdf_engine",
        "ocr_engine",
        "render",
        "typst",
        "typst_math",
        "cjk_fonts_render",
        "poppler",
        "pdf_raster",
        "pillow",
        "cjk_font_metrics",
    ),
    "details": (
        "api_key.missing",
        "api_key.ok",
        # A loopback endpoint needs no credential; reporting FAIL for one sent
        # operators looking for a key the endpoint never wanted.
        "api_key.local_not_needed",
        "base_url.ok",
        "live_probe.model_online",
        "live_probe.model_missing",
        "live_probe.reachable",
        "live_probe.auth_failed",
        "live_probe.models_404",
        "live_probe.http_error",
        "live_probe.unreachable",
        "models.missing",
        "models.ok",
        "cost_pricing.local_endpoint",
        "cost_pricing.unpriced",
        "cost_pricing.ok",
        "ledger_dir.writable",
        "ledger_dir.not_writable",
        "ledger_permissions.exposed",
        "ledger_permissions.ok",
        "ledger_permissions.windows_acl",
        "translation_memory.ok",
        "translation_memory.disabled",
        "disk_space.critical",
        "disk_space.low",
        "disk_space.ok",
        "disk_space.unknown",
        "sqlite_wal.ok",
        "sqlite_wal.failed",
        "page_images.disabled",
        "page_images.allowed",
        "qe.tiered",
        "qe.heuristic",
        "batch_api.enabled",
        "batch_api.disabled",
        "pdf_engine.docling_ok",
        "pdf_engine.docling_missing",
        "pdf_engine.pdfium",
        "pdf_engine.auto_docling",
        "pdf_engine.auto_missing",
        "pdf_engine.registered",
        "pdf_engine.unregistered",
        "ocr_engine.sidecar",
        "ocr_engine.rapidocr",
        "ocr_engine.rapidocr_no_backend",
        "ocr_engine.cloud_ok",
        "ocr_engine.cloud_blocked",
        "ocr_engine.none",
        "ocr_engine.off",
        "render.ok",
        "typst.ok",
        "typst.missing",
        "typst_math.ok",
        "typst_math.unavailable",
        "cjk_fonts_render.ok",
        "cjk_fonts_render.missing",
        "cjk_fonts_render.no_probe",
        "cjk_fonts_render.no_module",
        "poppler.ok",
        "poppler.missing",
        "pdf_raster.ok",
        "pdf_raster.missing",
        "pillow.ok",
        "pillow.missing",
        "cjk_font_metrics.ok",
        "cjk_font_metrics.missing",
        "cjk_font_metrics.no_module",
    ),
    "fixes": (
        "api_key.missing",
        "models.missing",
        "live_probe.model_missing",
        "live_probe.auth_failed",
        "live_probe.unreachable",
        "cost_pricing.local_endpoint",
        "cost_pricing.unpriced",
        "ledger_permissions.exposed",
        "disk_space.critical",
        "disk_space.low",
        "sqlite_wal.failed",
        "page_images.allowed",
        "pdf_engine.docling_missing",
        "pdf_engine.auto_missing",
        "ocr_engine.rapidocr_no_backend",
        "ocr_engine.cloud_blocked",
        "ocr_engine.none",
        "typst.missing",
        "cjk_fonts_render.missing",
        "poppler.missing",
        "pdf_raster.missing",
        "pillow.missing",
    ),
}


_STATUS_COLOR = {"OK": "green", "WARN": "yellow", "FAIL": "red", "SKIP": "dim"}


@dataclass(frozen=True)
class _Check:
    """One probe result: what was checked, its verdict, and how to fix it.

    ``detail`` is the diagnosis (what is true right now); ``fix`` is the action
    the operator must take. Keeping them apart is the point of the checklist —
    a remediation buried mid-sentence in the detail reads as one more wrapped
    line of prose.

    **Localization.** Every string here is English, because the CLI prints it
    verbatim and English is the engine's own language. A non-English console must
    not have to parse English prose, so each check also carries a stable
    ``code`` plus the values interpolated into the copy:

    * ``code`` names the check (``api_key``, ``disk_space``, …). It is stable
      across releases; the English ``name`` is not, because a wording fix must
      not break a translation.
    * ``detail_code`` / ``fix_code`` name the *variant* of the message (a probe
      that failed for two different reasons is two codes), and ``params`` carries
      the interpolated values so a consumer can render its own sentence from its
      own catalogue.
    * ``group_code`` names the section, for the same reason.

    A consumer that does not know a code falls back to the English copy — honest,
    if untranslated — rather than printing a raw key. This is the same contract
    :class:`ubt.core.assess.AssessmentWarning` already uses.
    """

    group: str
    name: str
    status: str
    detail: str
    fix: str | None = None
    #: Stable identifiers and interpolation values for a localized renderer.
    group_code: str = ""
    code: str = ""
    detail_code: str = ""
    fix_code: str = ""
    params: dict[str, Any] | None = None


def _summary(checks: list[_Check]) -> dict[str, int]:
    summary = {"fail": 0, "warn": 0, "ok": 0, "skip": 0}
    for check in checks:
        summary[check.status.lower()] += 1
    return summary


def _overall_status(summary: dict[str, int]) -> str:
    if summary["fail"]:
        return "fail"
    if summary["warn"]:
        return "warn"
    return "ok"


def _emit_json(checks: list[_Check]) -> None:
    """Emit the checks as exactly one JSON object on stdout (no Rich chrome)."""
    summary = _summary(checks)
    payload = {
        "status": _overall_status(summary),
        "summary": summary,
        "checks": [
            {
                "group": check.group,
                "name": check.name,
                "status": check.status,
                "detail": check.detail,
                "fix": check.fix,
            }
            for check in checks
        ],
    }
    print(json.dumps(payload, ensure_ascii=False))


def _emit_human(checks: list[_Check]) -> None:
    """Render the checks as a grouped, borderless checklist.

    A borderless table (not a boxed one) gives the wrapped detail a hanging
    indent under its check, so a long remediation path does not stretch one row
    across eight terminal lines and destroy the table's rhythm.
    """
    summary = _summary(checks)
    header = (
        "[bold]UBT Doctor[/] \u2014 "
        f"[red]{summary['fail']} FAIL[/] \u00b7 "
        f"[yellow]{summary['warn']} WARN[/] \u00b7 "
        f"[green]{summary['ok']} OK[/]"
    )
    if summary["skip"]:
        header += f" \u00b7 [dim]{summary['skip']} SKIP[/]"
    console.print(header)
    # One table for every group (not one per group): Rich sizes columns from
    # the whole table, so a wide name in one group cannot push that group's
    # Detail column out of alignment with its neighbours.
    table = Table(box=None, show_header=False, pad_edge=False, padding=(0, 1))
    table.add_column(no_wrap=True, width=1)
    table.add_column(no_wrap=True)
    table.add_column(no_wrap=True)
    table.add_column(overflow="fold")
    for index, (_code, group) in enumerate(_GROUPS):
        group_checks = [check for check in checks if check.group == group]
        if not group_checks:
            continue
        if index:
            table.add_section()
        table.add_row("", f"[bold]{group}[/]", "", "")
        for check in group_checks:
            color = _STATUS_COLOR[check.status]
            mark = f"[{color}]{_STATUS_MARK[check.status]}[/]"
            status = f"[{color}]{check.status}[/]"
            detail = escape(check.detail)
            if check.fix:
                detail += f"\n[dim]\u2192 {escape(check.fix)}[/]"
            table.add_row(mark, escape(check.name), status, detail)
    console.print(table)

    if summary["fail"]:
        console.print("[bold red]\u2717 doctor found FAIL items above.[/]")
    elif summary["warn"]:
        names = [check.name for check in checks if check.status == "WARN"]
        console.print(
            f"[bold green]\u2713 All checks passed[/] with [yellow]{len(names)} "
            f"warning(s)[/]: {', '.join(names)}"
        )
    else:
        console.print("[bold green]\u2713 All checks passed.[/]")


def doctor_command(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit the checks as a single JSON object on stdout."),
    ] = False,
    probe: Annotated[
        bool,
        typer.Option(
            "--probe",
            help="Live network probe: check endpoint connectivity, measure RTT, and verify models via /v1/models.",
        ),
    ] = False,
) -> None:
    """Self-check configuration, credentials, and local environment.

    Read-only and offline: validates env values, credential presence,
    directory writability, and optional-dependency availability. Exits 1 when
    any check FAILs (WARNs never fail the command). ``--json`` emits one object
    with ``status``/``summary``/``checks`` so a CI gate need not scrape the
    human checklist.
    """
    try:
        config = UBTConfig.from_env()
    except ValidationError as exc:
        errors = [
            {"loc": ".".join(str(part) for part in err["loc"]), "msg": err["msg"]}
            for err in exc.errors()
        ]
        if json_output:
            print(
                json.dumps(
                    {"status": "error", "code": "config_invalid", "errors": errors},
                    ensure_ascii=False,
                )
            )
        else:
            console.print("[bold red]Configuration invalid:[/]")
            for err in errors:
                console.print(f"  [red]\u2022[/] {err['loc']}: {err['msg']}")
        raise typer.Exit(code=2) from exc
    except UBTError as exc:
        # e.g. a malformed UBT_PROVIDER raises ProviderNotFoundError; the
        # diagnostic command must explain it, not die with a traceback.
        if json_output:
            print(
                json.dumps(
                    {
                        "status": "error",
                        "code": "config_invalid",
                        "errors": [{"loc": "", "msg": str(exc)}],
                    },
                    ensure_ascii=False,
                )
            )
        else:
            console.print(f"[bold red]Configuration invalid:[/] {exc}")
        raise typer.Exit(code=2) from exc

    checks = collect_checks(config, probe=probe)
    if json_output:
        _emit_json(checks)
    else:
        _emit_human(checks)
    if any(check.status == "FAIL" for check in checks):
        raise typer.Exit(code=1)


def _record_ledger_permissions(config: UBTConfig, record: Callable[..., None]) -> None:
    """The ledger-permissions probe, platform-gated.

    On Windows, POSIX mode bits are synthesized — a writable file always
    reports 0666 — so the group/other scan would flag every file, and the
    chmod remedy it prints would be noise. SKIP says honestly: not measurable
    on this platform; privacy there is an NTFS-ACL question.
    """
    if sys.platform == "win32":
        record(
            "Ledger permissions",
            "SKIP",
            "Windows enforces permissions through NTFS ACLs, not POSIX bits; "
            "keep the ledger under your user profile and it stays private "
            "to your account.",
            code="ledger_permissions",
            detail_code="ledger_permissions.windows_acl",
            params={"platform": "windows"},
        )
        return
    docling_cache = cache_root() / "docling_cache"
    output_dir = default_output_dir_for_scan()
    exposed = world_readable_files(config.db_dir, extra_dirs=(docling_cache, output_dir))
    if exposed:
        # Name the scan roots that hold an exposed file, not each file's
        # immediate parent: the docling cache keeps one hash directory per
        # asset, so listing parents printed five long paths where
        # ``docling_cache`` covers them all — and stays correct as new
        # hash directories appear.
        def _under(root: Path, file: Path) -> bool:
            try:
                file.resolve().relative_to(root.resolve())
                return True
            except ValueError:
                return False

        roots = [
            root
            for root in (config.db_dir, docling_cache, output_dir)
            if any(_under(root, path) for path in exposed)
        ]
        record(
            "Ledger permissions",
            "WARN",
            f"{len(exposed)} file(s) carrying book text are group/other-readable, "
            f"e.g. {exposed[0]}",
            fix="chmod -R go-rwx " + " ".join(f"'{root}'" for root in roots),
            code="ledger_permissions",
            detail_code="ledger_permissions.exposed",
            fix_code="ledger_permissions.exposed",
            params={
                "count": len(exposed),
                "example": str(exposed[0]),
                "roots": [str(r) for r in roots],
            },
        )
    else:
        record(
            "Ledger permissions",
            "OK",
            f"owner-only under {config.db_dir}",
            code="ledger_permissions",
            detail_code="ledger_permissions.ok",
            params={"db_dir": str(config.db_dir)},
        )


def collect_checks(config: UBTConfig, *, probe: bool = False) -> list[_Check]:
    """Run every read-only self-check and return the results.

    Shared by ``ubt doctor`` and the console's ``/system/doctor`` endpoint so the
    two surfaces cannot drift. Read-only and offline unless ``probe`` is set, in
    which case the provider endpoint is contacted once.
    """
    # Imported here, not at module scope: `ubt doctor` is not `ubt --help`, but
    # every command module is imported when the Typer app is built, so a
    # module-scope adapter import makes `ubt version` pay the whole
    # docling/pdf_oxide/pikepdf graph (~80 ms) before parsing an argument.
    from ubt.adapters import is_pdf_engine_registered

    checks: list[_Check] = []
    group_code = _GROUPS[0][0]
    group = _GROUPS[0][1]

    def record(
        name: str,
        status: str,
        detail: str,
        fix: str | None = None,
        *,
        code: str = "",
        detail_code: str = "",
        fix_code: str = "",
        params: dict[str, Any] | None = None,
    ) -> None:
        checks.append(
            _Check(
                group=group,
                name=name,
                status=status,
                detail=detail,
                fix=fix,
                group_code=group_code,
                code=code,
                detail_code=detail_code,
                fix_code=fix_code,
                params=params,
            )
        )

    def probe_writable(name: str, path: Path, code: str) -> None:
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".ubt_doctor_probe"
            probe.write_text("ok", encoding="utf-8")
            # missing_ok: two concurrent `ubt doctor` runs share this fixed
            # probe name, so one may already have removed the other's file —
            # a spurious FAIL that said the ledger dir was unwritable.
            probe.unlink(missing_ok=True)
            record(
                name,
                "OK",
                f"writable: {path}",
                code=code,
                detail_code=f"{code}.writable",
                params={"path": str(path)},
            )
        except OSError as exc:
            record(
                name,
                "FAIL",
                f"not writable: {path} ({exc})",
                code=code,
                detail_code=f"{code}.not_writable",
                params={"path": str(path), "error": str(exc)},
            )

    # -- credentials -----------------------------------------------------
    # A loopback endpoint needs no credential, so reporting FAIL for one was the
    # single most misleading line in this report: on the reference machine the
    # operator had a working local model at 127.0.0.1:9090 and the first row said
    # "API key: FAIL, not configured". The run was fine; the check was wrong.
    if has_real_credential(config):
        record("API key", "OK", "configured", code="api_key", detail_code="api_key.ok")
    elif not credential_required(config):
        record(
            "API key",
            "OK",
            "not needed for a local endpoint",
            code="api_key",
            detail_code="api_key.local_not_needed",
            params={"base_url": config.base_url},
        )
    else:
        record(
            "API key",
            "FAIL",
            "not configured",
            fix="export UBT_LLM_API_KEY=sk-... (or set api_key in ubt.toml / pass --api-key)",
            code="api_key",
            detail_code="api_key.missing",
            fix_code="api_key.missing",
        )
    record(
        "Base URL",
        "OK",
        config.base_url,
        code="base_url",
        detail_code="base_url.ok",
        params={"base_url": config.base_url},
    )

    if probe:
        key_str = config.api_key.get_secret_value()
        base_clean = config.base_url.strip().rstrip("/")
        headers: dict[str, str] = {}
        # Only send an Authorization header when there is a real credential —
        # the same rule the transport follows. Probing with `Bearer mock-key`
        # would test a wire shape no real job ever uses, and a local server that
        # rejects an unknown bearer would report a connectivity failure that
        # does not exist.
        if has_real_credential(config):
            if config.api_mode == "anthropic-messages":
                headers["x-api-key"] = key_str
                headers["anthropic-version"] = "2023-06-01"
            elif config.api_mode == "gemini-native":
                headers["x-goog-api-key"] = key_str
            else:
                headers["Authorization"] = f"Bearer {key_str}"
        headers.update(config.extra_headers)

        import time

        start_time = time.perf_counter()
        try:
            with httpx.Client(timeout=5.0) as client:
                models_url = f"{base_clean}/models"
                resp = client.get(models_url, headers=headers)
                rtt_ms = (time.perf_counter() - start_time) * 1000

                if resp.status_code == 200:
                    try:
                        resp_json = resp.json()
                        raw_models = resp_json.get("data") or resp_json.get("models") or []
                        available = []
                        for m in raw_models:
                            if isinstance(m, dict):
                                mid = m.get("id") or m.get("name")
                                if mid:
                                    available.append(str(mid).removeprefix("models/"))
                    except Exception:
                        available = []
                    if available:
                        if config.draft_model in available:
                            record(
                                "Live Endpoint Probe",
                                "OK",
                                f"{base_clean} reachable ({rtt_ms:.0f}ms); draft model '{config.draft_model}' confirmed online",
                                code="live_probe",
                                detail_code="live_probe.model_online",
                                params={
                                    "base_url": base_clean,
                                    "rtt_ms": f"{rtt_ms:.0f}",
                                    "model": config.draft_model,
                                },
                            )
                        else:
                            preview = ", ".join(available[:6])
                            record(
                                "Live Endpoint Probe",
                                "WARN",
                                f"{base_clean} reachable ({rtt_ms:.0f}ms), but draft model '{config.draft_model}' not found in {len(available)} model(s). Available: {preview}",
                                fix=f"choose from: {preview}",
                                code="live_probe",
                                detail_code="live_probe.model_missing",
                                fix_code="live_probe.model_missing",
                                params={
                                    "base_url": base_clean,
                                    "rtt_ms": f"{rtt_ms:.0f}",
                                    "model": config.draft_model,
                                    "count": len(available),
                                    "preview": preview,
                                },
                            )
                    else:
                        record(
                            "Live Endpoint Probe",
                            "OK",
                            f"{base_clean} reachable ({rtt_ms:.0f}ms)",
                            code="live_probe",
                            detail_code="live_probe.reachable",
                            params={"base_url": base_clean, "rtt_ms": f"{rtt_ms:.0f}"},
                        )
                elif resp.status_code in (401, 403):
                    record(
                        "Live Endpoint Probe",
                        "FAIL",
                        f"Endpoint {base_clean} returned {resp.status_code} (Authentication failed)",
                        fix=f"check credential for {config.provider or 'current provider'}",
                        code="live_probe",
                        detail_code="live_probe.auth_failed",
                        fix_code="live_probe.auth_failed",
                        params={
                            "base_url": base_clean,
                            "status": resp.status_code,
                            "provider": config.provider or "current provider",
                        },
                    )
                elif resp.status_code == 404:
                    record(
                        "Live Endpoint Probe",
                        "WARN",
                        f"Endpoint {base_clean} reachable ({rtt_ms:.0f}ms), but /models returned 404",
                        code="live_probe",
                        detail_code="live_probe.models_404",
                        params={"base_url": base_clean, "rtt_ms": f"{rtt_ms:.0f}"},
                    )
                else:
                    record(
                        "Live Endpoint Probe",
                        "WARN",
                        f"Endpoint {base_clean} returned HTTP {resp.status_code} ({rtt_ms:.0f}ms)",
                        code="live_probe",
                        detail_code="live_probe.http_error",
                        params={
                            "base_url": base_clean,
                            "status": resp.status_code,
                            "rtt_ms": f"{rtt_ms:.0f}",
                        },
                    )
        except Exception as exc:
            record(
                "Live Endpoint Probe",
                "FAIL",
                f"Cannot connect to {base_clean}: {exc}",
                fix="verify endpoint URL, network connection, or local model server status",
                code="live_probe",
                detail_code="live_probe.unreachable",
                fix_code="live_probe.unreachable",
                params={"base_url": base_clean, "error": str(exc)},
            )
    if not config.draft_model.strip():
        record(
            "Models",
            "FAIL",
            "no draft model configured",
            fix="set --draft-model/--repair-model, UBT_DRAFT_MODEL, or --provider",
            code="models",
            detail_code="models.missing",
            fix_code="models.missing",
        )
    else:
        record(
            "Models",
            "OK",
            f"draft={config.draft_model} repair={config.repair_model}",
            code="models",
            detail_code="models.ok",
            params={"draft": config.draft_model, "repair": config.repair_model},
        )
    local_endpoint = endpoint_is_local(config.base_url) and not (
        billing_enabled_for_local_endpoints()
    )
    unpriced = sorted(
        {
            model
            for model in (config.draft_model, config.repair_model)
            if model and not price_is_known(model, base_url=config.base_url)
        }
    )
    if local_endpoint:
        # A self-hosted endpoint cannot bill, so an absent price-table entry is
        # not a defect: WARNing about it told Ollama/llama.cpp users their money
        # features were broken when the run in fact costs $0.
        record(
            "Cost pricing",
            "OK",
            f"local endpoint {config.base_url} \u2014 nothing is billed, so model "
            "names need no price-table entry",
            fix="set UBT_BILL_LOCAL_ENDPOINT=1 only for a paid gateway behind the local URL",
            code="cost_pricing",
            detail_code="cost_pricing.local_endpoint",
            fix_code="cost_pricing.local_endpoint",
            params={"base_url": config.base_url},
        )
    elif unpriced:
        record(
            "Cost pricing",
            "WARN",
            f"no price-table entry for {', '.join(unpriced)} \u2014 cost reports as "
            "unknown and --budget-usd cannot trip",
            fix=(
                "add the per-MTok rate to ubt/resources/prices.toml "
                "(legacy fallback: MODEL_PRICES_USD_PER_MTOK in ubt/core/router/pricing.py)"
            ),
            code="cost_pricing",
            detail_code="cost_pricing.unpriced",
            fix_code="cost_pricing.unpriced",
            params={"models": ", ".join(unpriced)},
        )
    else:
        record(
            "Cost pricing",
            "OK",
            "draft and repair models are priced",
            code="cost_pricing",
            detail_code="cost_pricing.ok",
        )

    # -- storage -----------------------------------------------------------
    group_code = _GROUPS[1][0]
    group = _GROUPS[1][1]
    probe_writable("Ledger dir", config.db_dir, "ledger_dir")
    _record_ledger_permissions(config, record)
    if config.tm_enabled:
        record(
            "Translation memory",
            "OK",
            f"shared tm.sqlite, fuzzy threshold {config.tm_fuzzy_threshold}",
            code="translation_memory",
            detail_code="translation_memory.ok",
            params={"threshold": config.tm_fuzzy_threshold},
        )
    else:
        record(
            "Translation memory",
            "SKIP",
            "disabled",
            code="translation_memory",
            detail_code="translation_memory.disabled",
        )

    try:
        usage = shutil.disk_usage(config.db_dir)
        free_gb = usage.free / (1024**3)
        total_gb = usage.total / (1024**3)
        if free_gb < 1.0:
            record(
                "Disk space",
                "FAIL",
                f"only {free_gb:.2f} GB free of {total_gb:.1f} GB on {config.db_dir} — temporary render artifacts will fail",
                fix="free up disk space on the volume containing UBT_DB_DIR",
                code="disk_space",
                detail_code="disk_space.critical",
                fix_code="disk_space.critical",
                params={
                    "free_gb": f"{free_gb:.2f}",
                    "total_gb": f"{total_gb:.1f}",
                    "db_dir": str(config.db_dir),
                },
            )
        elif free_gb < 5.0:
            record(
                "Disk space",
                "WARN",
                f"{free_gb:.1f} GB free of {total_gb:.1f} GB on {config.db_dir} — low disk space for large PDF compilations",
                fix="ensure at least 5 GB free disk space for high-volume compiles",
                code="disk_space",
                detail_code="disk_space.low",
                fix_code="disk_space.low",
                params={
                    "free_gb": f"{free_gb:.1f}",
                    "total_gb": f"{total_gb:.1f}",
                    "db_dir": str(config.db_dir),
                },
            )
        else:
            record(
                "Disk space",
                "OK",
                f"{free_gb:.1f} GB free of {total_gb:.1f} GB on {config.db_dir}",
                code="disk_space",
                detail_code="disk_space.ok",
                params={
                    "free_gb": f"{free_gb:.1f}",
                    "total_gb": f"{total_gb:.1f}",
                    "db_dir": str(config.db_dir),
                },
            )
    except OSError as exc:
        record(
            "Disk space",
            "WARN",
            f"unable to query disk usage on {config.db_dir}: {exc}",
            code="disk_space",
            detail_code="disk_space.unknown",
            params={"db_dir": str(config.db_dir), "error": str(exc)},
        )

    try:
        import sqlite3

        probe_db = config.db_dir / ".ubt_doctor_wal_probe.db"
        config.db_dir.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(probe_db, timeout=2.0) as conn:
            cur = conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL;")
            wal_mode = cur.fetchone()
            mode_str = str(wal_mode[0]).upper() if wal_mode else "UNKNOWN"
            cur.execute("CREATE TABLE IF NOT EXISTS _probe (id INTEGER PRIMARY KEY, ts REAL);")
            cur.execute("INSERT OR REPLACE INTO _probe VALUES (1, 1.0);")
            conn.commit()
        for p in (
            probe_db,
            probe_db.with_name(probe_db.name + "-wal"),
            probe_db.with_name(probe_db.name + "-shm"),
        ):
            p.unlink(missing_ok=True)
        record(
            "SQLite WAL Ledger",
            "OK",
            f"journal_mode={mode_str}, write lock acquired and released cleanly",
            code="sqlite_wal",
            detail_code="sqlite_wal.ok",
            params={"mode": mode_str},
        )
    except Exception as exc:
        record(
            "SQLite WAL Ledger",
            "WARN",
            f"WAL lock probe failed on {config.db_dir}: {exc}",
            fix="check filesystem lock support (NFS/network mounts may fail SQLite WAL locks; use a local disk)",
            code="sqlite_wal",
            detail_code="sqlite_wal.failed",
            fix_code="sqlite_wal.failed",
            params={"db_dir": str(config.db_dir), "error": str(exc)},
        )

    # -- privacy: page-image egress disclosure -----------------------------
    if not config.allow_page_upload:
        record(
            "Page images",
            "OK",
            "egress disabled (UBT_ALLOW_PAGE_UPLOAD=false): book pages never leave "
            "this machine; cloud/vlm OCR refuses to start, visual repair degrades to text",
            code="page_images",
            detail_code="page_images.disabled",
        )
    else:
        routes: list[str] = []
        if config.ocr_mode in ("cloud", "vlm"):
            routes.append(
                f"OCR {config.ocr_mode} pages -> {config.ocr_endpoint or config.base_url}"
            )
        elif config.ocr_mode == "auto":
            routes.append("OCR auto: sidecar -> local rapidocr -> cloud last-resort")
        if config.visual_judge_enabled:
            routes.append(f"VLM page judge -> {config.visual_judge_model or config.repair_model}")
        routes.append("visual scalpel page crops -> repair model (when vision-capable)")
        record(
            "Page images",
            "WARN"
            if config.ocr_mode in ("cloud", "vlm", "auto") or config.visual_judge_enabled
            else "OK",
            "allowed; active routes: " + "; ".join(routes),
            fix="set UBT_ALLOW_PAGE_UPLOAD=false to keep pages local",
            code="page_images",
            detail_code="page_images.allowed",
            fix_code="page_images.allowed",
            params={"routes": routes},
        )

    # -- quality ------------------------------------------------------------
    group_code = _GROUPS[2][0]
    group = _GROUPS[2][1]
    if config.qe_engine == "tiered":
        # The judge may be a *different* provider block, and that block carries
        # its own model. Printing the translator's repair_model here named a model
        # the judge endpoint has never heard of — the reference config answers
        # with `workbuddy/deepseek-v4.1-flash` while this line said `gemma4-e4b`,
        # and the same mismatch had already been measured at runtime as twelve
        # `404 no such model` answers in one run.
        #
        # Resolution lives in `ubt.core.qe.settings` so this row, the console's
        # QE card and the pipeline cannot disagree about it again.
        from ubt.core.qe.settings import resolve_judge_model

        judge = resolve_judge_model(config)
        judge_provider = config.qe_judge_provider
        record(
            "QE (tiered)",
            "OK",
            (
                f"ambiguous-class + {config.qe_judge_pass_sample:.0%} pass sample, "
                f"judge={judge}" + (f" via {judge_provider}" if judge_provider else "")
            ),
            code="qe",
            detail_code="qe.tiered",
            params={
                "sample": f"{config.qe_judge_pass_sample:.0%}",
                "judge": judge,
                "provider": judge_provider or "",
            },
        )
    else:
        record(
            "QE (heuristic)",
            "OK",
            "zero-token rule scoring",
            code="qe",
            detail_code="qe.heuristic",
        )

    # -- optional tiers --------------------------------------------------------
    if config.batch_enabled:
        record(
            "Batch API",
            "WARN",
            "enabled \u2014 endpoint must implement /v1/files + /v1/batches (not probed offline)",
            code="batch_api",
            detail_code="batch_api.enabled",
        )
    else:
        record(
            "Batch API",
            "SKIP",
            "disabled",
            code="batch_api",
            detail_code="batch_api.disabled",
        )

    # -- rendering ------------------------------------------------------------
    group_code = _GROUPS[3][0]
    group = _GROUPS[3][1]
    if config.pdf_engine == "docling":
        if importlib.util.find_spec("docling") is not None:
            record(
                "PDF engine",
                "OK",
                "docling importable",
                code="pdf_engine",
                detail_code="pdf_engine.docling_ok",
            )
        else:
            record(
                "PDF engine",
                "WARN",
                "docling not installed \u2014 UBT_PDF_ENGINE=docling is an explicit "
                "request, so a PDF job fails with an actionable error instead of "
                "silently dropping tables, figures and formulas",
                fix="uv sync --extra pdf",
                code="pdf_engine",
                detail_code="pdf_engine.docling_missing",
                fix_code="pdf_engine.docling_missing",
            )
    elif config.pdf_engine == "pdfium":
        record(
            "PDF engine",
            "OK",
            "pypdfium2 fast path (bundled dependency); a scanned document "
            "still routes to docling automatically",
            code="pdf_engine",
            detail_code="pdf_engine.pdfium",
        )
    elif config.pdf_engine == "auto":
        if importlib.util.find_spec("docling") is not None:
            record(
                "PDF engine",
                "OK",
                "auto = docling layout (unconditional)",
                code="pdf_engine",
                detail_code="pdf_engine.auto_docling",
            )
        else:
            # ``auto`` is docling; without the package the adapter degrades to
            # the geometric pdfium reader (warned at runtime, and namespaced in
            # the job id). It does not fail the job \u2014 the old text claimed it
            # would, which sent operators looking for a crash that never came
            # while tables, figures and formulas were silently dropped.
            record(
                "PDF engine",
                "WARN",
                "auto selects docling, which is not installed \u2014 PDF jobs fall "
                "back to the geometric pdfium reader, dropping tables, figures and "
                "formulas from the rebuilt PDF",
                fix="uv sync --extra pdf, or set UBT_PDF_ENGINE=pdfium to accept the fast path",
                code="pdf_engine",
                detail_code="pdf_engine.auto_missing",
                fix_code="pdf_engine.auto_missing",
            )
    elif is_pdf_engine_registered(config.pdf_engine):
        record(
            "PDF engine",
            "OK",
            f"registered engine: {config.pdf_engine}",
            code="pdf_engine",
            detail_code="pdf_engine.registered",
            params={"engine": config.pdf_engine},
        )
    else:
        record(
            "PDF engine",
            "WARN",
            f"'{config.pdf_engine}' is not a registered engine",
            code="pdf_engine",
            detail_code="pdf_engine.unregistered",
            params={"engine": config.pdf_engine},
        )

    # -- OCR engine readiness ------------------------------------------------
    if config.ocr_mode != "off":
        from ubt.adapters.pdf.vlm.drivers.rapidocr_driver import RapidOcrDriver
        from ubt.adapters.pdf.vlm.drivers.sidecar_driver import SidecarOcrDriver

        has_sidecar = SidecarOcrDriver.is_healthy()
        has_rapidocr = RapidOcrDriver.is_available()
        rapidocr_spec = (
            importlib.util.find_spec("rapidocr") is not None
            or importlib.util.find_spec("rapidocr_onnxruntime") is not None
        )

        if has_sidecar:
            record(
                "OCR engine",
                "OK",
                "sidecar service online (http://localhost:8765)",
                code="ocr_engine",
                detail_code="ocr_engine.sidecar",
            )
        elif has_rapidocr:
            record(
                "OCR engine",
                "OK",
                "local rapidocr + onnxruntime operational",
                code="ocr_engine",
                detail_code="ocr_engine.rapidocr",
            )
        elif rapidocr_spec:
            record(
                "OCR engine",
                "WARN",
                "rapidocr installed but inference backend (onnxruntime) is missing — scanned PDFs cannot be OCRed",
                fix="uv sync --all-extras (or uv pip install onnxruntime)",
                code="ocr_engine",
                detail_code="ocr_engine.rapidocr_no_backend",
                fix_code="ocr_engine.rapidocr_no_backend",
            )
        elif config.ocr_mode in ("cloud", "vlm"):
            if config.allow_page_upload:
                record(
                    "OCR engine",
                    "OK",
                    f"cloud/vlm egress enabled ({config.ocr_mode})",
                    code="ocr_engine",
                    detail_code="ocr_engine.cloud_ok",
                    params={"mode": config.ocr_mode},
                )
            else:
                record(
                    "OCR engine",
                    "WARN",
                    f"ocr_mode='{config.ocr_mode}' but allow_page_upload=False — scanned pages cannot egress",
                    fix="enable page upload or install local rapidocr/sidecar",
                    code="ocr_engine",
                    detail_code="ocr_engine.cloud_blocked",
                    fix_code="ocr_engine.cloud_blocked",
                    params={"mode": config.ocr_mode},
                )
        else:
            record(
                "OCR engine",
                "WARN",
                "no local OCR engine operational (sidecar offline, rapidocr/onnxruntime not installed) — scanned PDFs will fail",
                fix="uv sync --all-extras (or docker run -d -p 8765:8765 ubt-ocr-sidecar)",
                code="ocr_engine",
                detail_code="ocr_engine.none",
                fix_code="ocr_engine.none",
            )
    else:
        record(
            "OCR engine",
            "SKIP",
            "ocr_mode=off",
            code="ocr_engine",
            detail_code="ocr_engine.off",
        )

    record(
        "Render",
        "OK",
        f"dual_mode={config.dual_mode}, pdf=typst-reflow",
        code="render",
        detail_code="render.ok",
        params={"dual_mode": config.dual_mode},
    )

    # -- local render toolchain -----------------------------------------------
    try:
        from ubt.adapters.pdf.typst_compile import typst_available

        has_typst = typst_available()
    except ImportError:
        has_typst = False
    if has_typst:
        record(
            "Typst",
            "OK",
            "PDF reflow rendering available",
            code="typst",
            detail_code="typst.ok",
        )
    else:
        record(
            "Typst",
            "WARN",
            "not found on PATH \u2014 a PDF render fails at stage 6",
            fix="install it (brew install typst / cargo install typst-cli), or choose "
            "a text output format (.md / .docx / .epub)",
            code="typst",
            detail_code="typst.missing",
            fix_code="typst.missing",
        )

    if has_typst:
        record(
            "Typst Math",
            "OK",
            "Typst native formula micro-typesetter available",
            code="typst_math",
            detail_code="typst_math.ok",
        )
    else:
        record(
            "Typst Math",
            "WARN",
            "Typst not available; formulas will remain as source graphics",
            code="typst_math",
            detail_code="typst_math.unavailable",
        )

    try:
        from ubt.adapters.pdf.font_probe import (
            available_font_families,
            is_cjk_capable,
            resolve_font_stack,
        )
        from ubt.core.language_profile import resolve_font_config

        families = available_font_families()
        if families is None:
            record(
                "CJK fonts (render)",
                "SKIP",
                "no probe answered (typst/fc-list absent)",
                code="cjk_fonts_render",
                detail_code="cjk_fonts_render.no_probe",
            )
        else:
            found = sorted({f for f in families if is_cjk_capable(f)})
            if found:
                dropped = resolve_font_stack(
                    list(resolve_font_config("zh").typst_fonts), available=families
                ).unavailable
                detail = f"{len(found)} resolvable CJK family/families (e.g. {found[0]})"
                if dropped:
                    detail += f"; profile names not installed: {', '.join(dropped)}"
                record(
                    "CJK fonts (render)",
                    "OK",
                    detail,
                    code="cjk_fonts_render",
                    detail_code="cjk_fonts_render.ok",
                    params={"count": len(found), "example": found[0], "dropped": list(dropped)},
                )
            else:
                record(
                    "CJK fonts (render)",
                    "WARN",
                    "no CJK-capable font the renderer can see \u2014 a zh/ja/ko PDF "
                    "render will produce tofu",
                    fix="install a CJK font (e.g. noto-fonts-cjk)",
                    code="cjk_fonts_render",
                    detail_code="cjk_fonts_render.missing",
                    fix_code="cjk_fonts_render.missing",
                )
    except ImportError:
        record(
            "CJK fonts (render)",
            "SKIP",
            "font probe module unavailable",
            code="cjk_fonts_render",
            detail_code="cjk_fonts_render.no_module",
        )

    poppler_missing = [b for b in ("pdftotext",) if shutil.which(b) is None]
    if not poppler_missing:
        record(
            "Poppler",
            "OK",
            "pdftotext available",
            code="poppler",
            detail_code="poppler.ok",
        )
    else:
        record(
            "Poppler",
            "WARN",
            f"missing: {', '.join(poppler_missing)} \u2014 this also disables the "
            "fail-closed artifact-parity check (target-language absence, page size, "
            "image count), so an untranslated book can finalize as completed",
            fix="install poppler-utils",
            code="poppler",
            detail_code="poppler.missing",
            fix_code="poppler.missing",
            params={"missing": list(poppler_missing)},
        )
    if importlib.util.find_spec("pdf_oxide") is not None:
        record(
            "PDF raster (oxide)",
            "OK",
            "pdf_oxide in-process page rasterizer",
            code="pdf_raster",
            detail_code="pdf_raster.ok",
        )
    else:
        record(
            "PDF raster (oxide)",
            "FAIL",
            "pdf_oxide is a base dependency but not importable \u2014 pixel gates dead",
            fix="reinstall the project environment (uv sync)",
            code="pdf_raster",
            detail_code="pdf_raster.missing",
            fix_code="pdf_raster.missing",
        )
    if importlib.util.find_spec("PIL") is not None:
        record(
            "Pillow",
            "OK",
            "pixel heuristics available",
            code="pillow",
            detail_code="pillow.ok",
        )
    else:
        record(
            "Pillow",
            "WARN",
            "not installed \u2014 pixel blank/black-block checks will skip",
            fix="install pillow",
            code="pillow",
            detail_code="pillow.missing",
            fix_code="pillow.missing",
        )

    try:
        from ubt.adapters.pdf.font_metrics import resolve_cjk_ttc

        record(
            "CJK font (metrics)",
            "OK",
            resolve_cjk_ttc(),
            code="cjk_font_metrics",
            detail_code="cjk_font_metrics.ok",
            params={"path": resolve_cjk_ttc()},
        )
    except DocumentParseError as exc:
        record(
            "CJK font (metrics)",
            "WARN",
            str(exc),
            code="cjk_font_metrics",
            detail_code="cjk_font_metrics.missing",
            params={"error": str(exc)},
        )
    except ImportError:
        record(
            "CJK font (metrics)",
            "SKIP",
            "font metrics module unavailable",
            code="cjk_font_metrics",
            detail_code="cjk_font_metrics.no_module",
        )

    return checks
