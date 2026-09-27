"""Self-check configuration, credentials, and local environment."""

import importlib
import importlib.util
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from ubt.core.config import MOCK_API_KEY, UBTConfig
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
_GROUPS = (
    "Credentials & models",
    "Storage & privacy",
    "Quality & optional tiers",
    "Rendering & toolchain",
)

_STATUS_MARK = {"OK": "\u2713", "WARN": "\u26a0", "FAIL": "\u2717", "SKIP": "\u00b7"}
_STATUS_COLOR = {"OK": "green", "WARN": "yellow", "FAIL": "red", "SKIP": "dim"}


@dataclass(frozen=True)
class _Check:
    """One probe result: what was checked, its verdict, and how to fix it.

    ``detail`` is the diagnosis (what is true right now); ``fix`` is the action
    the operator must take. Keeping them apart is the point of the checklist —
    the remediation used to be buried mid-sentence in the detail, where it read
    as one more wrapped line of prose.
    """

    group: str
    name: str
    status: str
    detail: str
    fix: str | None = None


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
    indent under its check, so a long remediation path no longer stretches one
    row across eight terminal lines and destroys the table's rhythm.
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
    for index, group in enumerate(_GROUPS):
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
) -> None:
    """Self-check configuration, credentials, and local environment.

    Read-only and offline: validates env values, credential presence,
    directory writability, and optional-dependency availability. Exits 1 when
    any check FAILs (WARNs never fail the command). ``--json`` emits one object
    with ``status``/``summary``/``checks`` so a CI gate need not scrape the
    human checklist.
    """
    # Imported here, not at module scope: `ubt doctor` is not `ubt --help`, but
    # every command module is imported when the Typer app is built, so a
    # module-scope adapter import makes `ubt version` pay the whole
    # docling/pdf_oxide/pikepdf graph (~80 ms) before parsing an argument.
    from ubt.adapters import is_pdf_engine_registered

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

    checks: list[_Check] = []
    group = _GROUPS[0]

    def record(name: str, status: str, detail: str, fix: str | None = None) -> None:
        checks.append(_Check(group=group, name=name, status=status, detail=detail, fix=fix))

    def probe_writable(name: str, path: Path) -> None:
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".ubt_doctor_probe"
            probe.write_text("ok", encoding="utf-8")
            # missing_ok: two concurrent `ubt doctor` runs share this fixed
            # probe name, so one may already have removed the other's file —
            # a spurious FAIL that said the ledger dir was unwritable.
            probe.unlink(missing_ok=True)
            record(name, "OK", f"writable: {path}")
        except OSError as exc:
            record(name, "FAIL", f"not writable: {path} ({exc})")

    # -- credentials -----------------------------------------------------
    key = config.api_key.get_secret_value()
    if not key or key == MOCK_API_KEY:
        record(
            "API key",
            "FAIL",
            "not configured",
            fix="export UBT_LLM_API_KEY=sk-... (or select a provider whose "
            "api_key_env names the variable holding the key)",
        )
    else:
        record("API key", "OK", "configured")
    record("Base URL", "OK", config.base_url)
    shipped_defaults = {
        UBTConfig.model_fields[name].default for name in ("draft_model", "repair_model")
    }
    shipped = sorted(
        {model for model in (config.draft_model, config.repair_model) if model in shipped_defaults}
    )
    # A selected provider supplies these models on purpose (the opencode provider's
    # default *is* the shipped one), so only an unselected run is a real oversight.
    if shipped and config.provider is None:
        record(
            "Models",
            "WARN",
            f"still the shipped benchmark default ({', '.join(shipped)})",
            fix="set --draft-model/--repair-model or --provider before translating",
        )
    else:
        record("Models", "OK", f"draft={config.draft_model} repair={config.repair_model}")
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
        )
    elif unpriced:
        record(
            "Cost pricing",
            "WARN",
            f"no price-table entry for {', '.join(unpriced)} \u2014 cost reports as "
            "unknown and --budget-usd cannot trip",
            fix="add the per-MTok rate to MODEL_PRICES_USD_PER_MTOK in ubt/core/router/pricing.py",
        )
    else:
        record("Cost pricing", "OK", "draft and repair models are priced")

    # -- storage -----------------------------------------------------------
    group = _GROUPS[1]
    probe_writable("Ledger dir", config.db_dir)
    docling_cache = Path(".ubt/docling_cache")
    output_dir = default_output_dir_for_scan()
    exposed = world_readable_files(config.db_dir, extra_dirs=(docling_cache, output_dir))
    if exposed:
        # Name the scan roots that hold an exposed file, not each file's
        # immediate parent: the docling cache keeps one hash directory per
        # asset, so listing parents printed five long paths where
        # ``.ubt/docling_cache`` covers them all — and stays correct as new
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
        )
    else:
        record("Ledger permissions", "OK", f"owner-only under {config.db_dir}")
    if config.tm_enabled:
        record(
            "Translation memory",
            "OK",
            f"shared tm.sqlite, fuzzy threshold {config.tm_fuzzy_threshold}",
        )
    else:
        record("Translation memory", "SKIP", "disabled")

    # -- privacy: page-image egress disclosure -----------------------------
    if not config.allow_page_upload:
        record(
            "Page images",
            "OK",
            "egress disabled (UBT_ALLOW_PAGE_UPLOAD=false): book pages never leave "
            "this machine; cloud/vlm OCR refuses to start, visual repair degrades to text",
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
        )

    # -- quality ------------------------------------------------------------
    group = _GROUPS[2]
    if config.qe_engine == "subprocess":
        script = config.comet_script_path
        if script is not None and script.exists():
            record("QE (neural)", "OK", f"scoring script: {script}")
        else:
            record(
                "QE (neural)",
                "FAIL",
                f"scoring script missing: {script}",
                fix="point UBT_COMET_SCRIPT_PATH at the scoring script, or set "
                "UBT_QE_ENGINE=tiered/heuristic",
            )
    elif config.qe_engine == "tiered":
        judge = config.qe_judge_model or config.repair_model
        record(
            "QE (tiered)",
            "OK",
            f"gray zone [{config.qe_judge_gray_low}, {config.qe_judge_gray_high}], judge={judge}",
        )
    else:
        record("QE (heuristic)", "OK", "zero-token rule scoring")

    # -- optional tiers --------------------------------------------------------
    if config.batch_enabled:
        record(
            "Batch API",
            "WARN",
            "enabled \u2014 endpoint must implement /v1/files + /v1/batches (not probed offline)",
        )
    else:
        record("Batch API", "SKIP", "disabled")
    if config.pe_queue_enabled:
        record("PE queue", "OK", f"export format: {config.pe_export_format}")
    else:
        record("PE queue", "SKIP", "disabled")

    # -- rendering ------------------------------------------------------------
    group = _GROUPS[3]
    if config.pdf_engine == "docling":
        if importlib.util.find_spec("docling") is not None:
            record("PDF engine", "OK", "docling importable")
        else:
            record(
                "PDF engine",
                "WARN",
                "docling not installed \u2014 PDF jobs will fail",
                fix="uv sync --extra pdf",
            )
    elif config.pdf_engine == "pdfium":
        record("PDF engine", "OK", "pypdfium2 fast path (bundled dependency)")
    elif config.pdf_engine == "auto":
        if importlib.util.find_spec("docling") is not None:
            record("PDF engine", "OK", "auto-select (pdfium fast path / docling layout)")
        else:
            record(
                "PDF engine",
                "WARN",
                "auto routing needs docling for layout analysis but it is not "
                "installed \u2014 complex PDFs silently degrade to the pypdf fast path "
                "and whole pages arrive as one untranslatable block",
                fix="uv sync --extra pdf, or set UBT_PDF_ENGINE=pdfium to force the fast path",
            )
    elif is_pdf_engine_registered(config.pdf_engine):
        record("PDF engine", "OK", f"registered engine: {config.pdf_engine}")
    else:
        record("PDF engine", "WARN", f"'{config.pdf_engine}' is not a registered engine")
    record(
        "Render",
        "OK",
        f"dual_mode={config.dual_mode}, pdf={config.render_engine}",
    )

    # -- local render toolchain -----------------------------------------------
    try:
        from ubt.adapters.pdf.typst_compile import typst_available

        has_typst = typst_available()
    except ImportError:
        has_typst = False
    if has_typst:
        record("Typst", "OK", "PDF reflow / rigid rendering available")
    else:
        record(
            "Typst",
            "WARN",
            "not found on PATH \u2014 a PDF render fails at stage 6",
            fix="install it (brew install typst / cargo install typst-cli), or use "
            "--render-engine with a text output format",
        )

    try:
        from ubt.adapters.pdf.math_renderer import MathjaxRenderer

        mathjax_ready = MathjaxRenderer().available()
    except Exception:
        mathjax_ready = False
    if config.math_backend != "mathjax":
        record("MathJax", "SKIP", f"math_backend={config.math_backend}")
    elif mathjax_ready:
        record(
            "MathJax", "OK", "node + scripts/mathjax/node_modules present (vector display formulas)"
        )
    else:
        record(
            "MathJax",
            "WARN",
            "math_backend='mathjax' but Node or scripts/mathjax/node_modules is "
            "missing \u2014 display formulas degrade to the typst backend",
            fix="run npm ci in scripts/mathjax (source checkout), or set "
            "UBT_MATH_BACKEND=typst/image to stop advertising vector math",
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
            record("CJK fonts (render)", "SKIP", "no probe answered (typst/fc-list absent)")
        else:
            found = sorted({f for f in families if is_cjk_capable(f)})
            if found:
                dropped = resolve_font_stack(
                    list(resolve_font_config("zh").typst_fonts), available=families
                ).unavailable
                detail = f"{len(found)} resolvable CJK family/families (e.g. {found[0]})"
                if dropped:
                    detail += f"; profile names not installed: {', '.join(dropped)}"
                record("CJK fonts (render)", "OK", detail)
            else:
                record(
                    "CJK fonts (render)",
                    "WARN",
                    "no CJK-capable font the renderer can see \u2014 a zh/ja/ko PDF "
                    "render will produce tofu",
                    fix="install a CJK font (e.g. noto-fonts-cjk)",
                )
    except ImportError:
        record("CJK fonts (render)", "SKIP", "font probe module unavailable")

    poppler_missing = [b for b in ("pdftotext", "pdftocairo") if shutil.which(b) is None]
    if not poppler_missing:
        record("Poppler", "OK", "pdftotext + pdftocairo available")
    else:
        record(
            "Poppler",
            "WARN",
            f"missing: {', '.join(poppler_missing)} \u2014 this also disables the "
            "fail-closed artifact-parity check (target-language absence, page size, "
            "image count), so an untranslated book can finalize as completed",
            fix="install poppler-utils",
        )
    if importlib.util.find_spec("pdf_oxide") is not None:
        record("PDF raster (oxide)", "OK", "pdf_oxide in-process page rasterizer")
    else:
        record(
            "PDF raster (oxide)",
            "FAIL",
            "pdf_oxide is a base dependency but not importable \u2014 pixel gates dead",
            fix="reinstall the project environment (uv sync)",
        )
    if shutil.which("pdftocairo") is not None and shutil.which("pdftotext") is not None:
        record("Diagram SVG", "OK", "pdftocairo vector route available")
    else:
        record(
            "Diagram SVG",
            "WARN",
            "vector SVG unavailable \u2014 diagrams fall back to raster PNG",
            fix="install poppler-utils for the vector route",
        )
    if importlib.util.find_spec("PIL") is not None:
        record("Pillow", "OK", "pixel heuristics available")
    else:
        record(
            "Pillow",
            "WARN",
            "not installed \u2014 pixel blank/black-block checks will skip",
            fix="install pillow",
        )

    try:
        from ubt.adapters.pdf.font_metrics import resolve_cjk_ttc

        record("CJK font (metrics)", "OK", resolve_cjk_ttc())
    except DocumentParseError as exc:
        record("CJK font (metrics)", "WARN", str(exc))
    except ImportError:
        record("CJK font (metrics)", "SKIP", "font metrics module unavailable")

    if json_output:
        _emit_json(checks)
    else:
        _emit_human(checks)
    if any(check.status == "FAIL" for check in checks):
        raise typer.Exit(code=1)
