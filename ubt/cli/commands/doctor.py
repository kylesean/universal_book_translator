"""Self-check configuration, credentials, and local environment."""

import importlib
import importlib.util
import shutil
from pathlib import Path

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from ubt.core.config import MOCK_API_KEY, UBTConfig
from ubt.core.exceptions import DocumentParseError
from ubt.core.fs_perms import world_readable_files
from ubt.core.job_options import default_output_dir_for_scan
from ubt.core.router.pricing import (
    billing_enabled_for_local_endpoints,
    endpoint_is_local,
    price_is_known,
)

console = Console()


def doctor_command() -> None:
    # Imported here, not at module scope: `ubt doctor` is not `ubt --help`, but
    # every command module is imported when the Typer app is built, so a
    # module-scope adapter import makes `ubt version` pay the whole
    # docling/pdf_oxide/pikepdf graph (~80 ms) before parsing an argument.
    from ubt.adapters import is_pdf_engine_registered

    """Self-check configuration, credentials, and local environment.

    Read-only and offline: validates env values, credential presence,
    directory writability, and optional-dependency availability. Exits 1 when
    any check FAILs (WARNs never fail the command).
    """
    try:
        config = UBTConfig.from_env()
    except ValidationError as exc:
        console.print("[bold red]Configuration invalid:[/]")
        for err in exc.errors():
            loc = ".".join(str(part) for part in err["loc"])
            console.print(f"  [red]\u2022[/] {loc}: {err['msg']}")
        raise typer.Exit(code=2) from exc

    rows: list[tuple[str, str, str]] = []
    failed = False
    warned: list[str] = []

    def record(name: str, status: str, detail: str) -> None:
        nonlocal failed
        if status == "FAIL":
            failed = True
        elif status == "WARN":
            warned.append(name)
        color = {"OK": "green", "WARN": "yellow", "FAIL": "red", "SKIP": "dim"}[status]
        rows.append((name, f"[{color}]{status}[/]", escape(detail)))

    def probe_writable(name: str, path: Path) -> None:
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".ubt_doctor_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            record(name, "OK", f"writable: {path}")
        except OSError as exc:
            record(name, "FAIL", f"not writable: {path} ({exc})")

    # -- credentials -----------------------------------------------------
    key = config.api_key.get_secret_value()
    if not key or key == MOCK_API_KEY:
        record("API key", "FAIL", "not configured \u2014 set UBT_LLM_API_KEY or OPENAI_API_KEY")
    else:
        record("API key", "OK", "configured")
    record("Base URL", "OK", config.base_url)
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
            f"local endpoint {config.base_url} — nothing is billed, so model "
            "names need no price-table entry (set UBT_BILL_LOCAL_ENDPOINT=1 only "
            "for a paid gateway behind it)",
        )
    elif unpriced:
        record(
            "Cost pricing",
            "WARN",
            f"no price-table entry for {', '.join(unpriced)} — cost reports as unknown "
            "and --budget-usd cannot trip; add the per-MTok rate to "
            "MODEL_PRICES_USD_PER_MTOK in ubt/core/router/pricing.py",
        )
    else:
        record("Cost pricing", "OK", "draft and repair models are priced")

    # -- storage -----------------------------------------------------------
    probe_writable("Ledger dir", config.db_dir)
    exposed = world_readable_files(
        config.db_dir,
        extra_dirs=(Path(".ubt/docling_cache"), default_output_dir_for_scan()),
    )
    if exposed:
        parents = sorted({str(path.parent) for path in exposed})
        target = " ".join(f"'{parent}'" for parent in parents)
        record(
            "Ledger permissions",
            "WARN",
            f"{len(exposed)} file(s) carrying book text are group/other-readable, "
            f"e.g. {exposed[0]} — tighten with: chmod -R go-rwx {target}",
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

    # -- quality ------------------------------------------------------------
    if config.qe_engine == "subprocess":
        script = config.comet_script_path
        if script is not None and script.exists():
            record("QE (neural)", "OK", f"scoring script: {script}")
        else:
            record("QE (neural)", "FAIL", f"scoring script missing: {script}")
    elif config.qe_engine == "tiered":
        judge = config.qe_judge_model or config.repair_model
        record(
            "QE (tiered)",
            "OK",
            f"gray zone [{config.qe_judge_gray_low}, {config.qe_judge_gray_high}], judge={judge}",
        )
    else:
        record("QE (heuristic)", "OK", "zero-token rule scoring")

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
            "allowed; active routes: "
            + "; ".join(routes)
            + " — set UBT_ALLOW_PAGE_UPLOAD=false to keep pages local",
        )

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
    if config.pdf_engine == "docling":
        if importlib.util.find_spec("docling") is not None:
            record("PDF engine", "OK", "docling importable")
        else:
            record("PDF engine", "WARN", "docling not installed \u2014 PDF jobs will fail")
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
                "installed — install the extra (`uv sync --extra pdf`), or complex "
                "PDFs silently degrade to the pypdf fast path and whole pages "
                "arrive as one untranslatable block",
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
            "not found on PATH — a PDF render fails at stage 6; install it "
            "(brew install typst / cargo install typst-cli) or use --render-engine "
            "with a text output format",
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
            "missing — display formulas degrade to the typst backend; run `npm ci` "
            "in scripts/mathjax (source checkout), or set UBT_MATH_BACKEND=typst "
            "or image to stop advertising vector math",
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
                    "no CJK-capable font the renderer can see — a zh/ja/ko PDF "
                    "render will produce tofu; install a CJK font (e.g. noto-fonts-cjk)",
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
            f"missing: {', '.join(poppler_missing)} — install poppler-utils. This "
            "also disables the fail-closed artifact-parity check (target-language "
            "absence, page size, image count): an untranslated book can then "
            "finalize as completed",
        )
    if importlib.util.find_spec("pdf_oxide") is not None:
        record("PDF raster (oxide)", "OK", "pdf_oxide in-process page rasterizer")
    else:
        record(
            "PDF raster (oxide)",
            "FAIL",
            "pdf_oxide is a base dependency but not importable — pixel gates dead",
        )
    if shutil.which("pdftocairo") is not None and shutil.which("pdftotext") is not None:
        record("Diagram SVG", "OK", "pdftocairo vector route available")
    else:
        record(
            "Diagram SVG",
            "WARN",
            "vector SVG unavailable — diagrams fall back to raster PNG",
        )
    if importlib.util.find_spec("PIL") is not None:
        record("Pillow", "OK", "pixel heuristics available")
    else:
        record(
            "Pillow",
            "WARN",
            "not installed — pixel blank/black-block checks will skip",
        )

    try:
        from ubt.adapters.pdf.font_metrics import resolve_cjk_ttc

        record("CJK font (metrics)", "OK", resolve_cjk_ttc())
    except DocumentParseError as exc:
        record("CJK font (metrics)", "WARN", str(exc))
    except ImportError:
        record("CJK font (metrics)", "SKIP", "font metrics module unavailable")

    table = Table(title="UBT Doctor")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    for name, status, detail in rows:
        table.add_row(name, status, detail)
    console.print(table)
    if failed:
        console.print("[bold red]\u2717 doctor found FAIL items above.[/]")
        raise typer.Exit(code=1)
    if warned:
        console.print(
            f"[bold green]\u2713 All checks passed[/] with [yellow]{len(warned)} "
            f"warning(s)[/]: {', '.join(warned)}"
        )
        return
    console.print("[bold green]\u2713 All checks passed.[/]")
