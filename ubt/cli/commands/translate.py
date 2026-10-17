"""CLI command for end-to-end document translation."""

import asyncio
import contextlib
import json
import logging
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from ubt.cli.ui import ui_msg
from ubt.core.config import (
    ApiMode,
    DualMode,
    OcrMode,
    PdfEngine,
    parse_page_ranges,
)
from ubt.core.job_options import (
    RUN_REPORT_KINDS,
    deliverable_candidates,
    delivery_verdict_from_summary,
    profile_name_is_valid,
    sidecar_path,
)
from ubt.core.log_config import setup_logging
from ubt.core.presets import Preset

#: The shared language switch (ubt.cli.ui owns the UBT_UI_LANG rule; assess and
#: translate both read it here rather than each keeping a copy).
_ui_msg = ui_msg


console = Console()


def _usage_error(message: str, json_output: bool) -> NoReturn:
    """Report a usage error the way this command already does, then exit 2.

    Usage errors exit 2 (typer's own convention for bad usage); runtime
    failures keep exit 1.
    """
    if json_output:
        print(json.dumps({"status": "failed", "error": message}))
    else:
        console.print(f"[bold red]Error:[/] {escape(message)}")
    raise typer.Exit(code=2)


def _get_run_translation() -> Any:
    # Deferred import of the shared entry helper: main.py imports this command
    # at module load, so importing main here (not at top level) avoids the
    # cycle while still resolving the *live* (monkeypatchable) module attribute.
    from ubt.cli.main import _run_translation

    return _run_translation


def _get_strict_failures() -> Any:
    from ubt.cli.main import _strict_failures

    return _strict_failures


#: Domain-profile names are not a closed set: ``seed_entries_for_profile`` and
#: the packaged glossary directories key on names like ``semiconductor`` /
#: ``semiconductor_paper``, and operators may add their own resource dirs. The
#: only invariant is the shared safe-name pattern, so the CLI validates with
#: ``profile_name_is_valid`` exactly like the API and MCP; a hardcoded allowlist
#: would reject any profile that carries glossary seeds.
_PROFILE_EXAMPLES = "general, textbook, paper, fiction, humanities, semiconductor"


def _clean_stale_companions(output: Path | None, *, input_path: Path | None = None) -> list[Path]:
    """Delete the sibling deliverables a ``--fresh`` run is about to regenerate.

    Only the names this pipeline actually writes for the mono/dual/bilingual
    family are considered; a name nothing produces is never swept, so no file
    is deleted for merely resembling a retired pattern. A deleted sibling takes
    its derived reports with it: leaving ``x_bilingual_md_quality_report.json``
    behind describes a document that does not exist, and the next run reads that
    report as the current one's.

    Returns what was deleted, so the caller can say so.
    """
    if output is None:
        return []
    stem = output.stem
    names: list[str] = []
    if stem.endswith("_mono"):
        base = stem[: -len("_mono")]
        names.extend([f"{base}_bilingual{output.suffix}", f"{base}_dual{output.suffix}"])
    elif stem.endswith(("_dual", "_bilingual")):
        base = stem.rsplit("_", 1)[0]
        names.append(f"{base}_mono{output.suffix}")
    deleted: list[Path] = []
    for n in names:
        p = output.with_name(n)
        if not p.exists() or p == output:
            continue
        if input_path is not None and _same_file(p, input_path):
            # A name-pattern match on the input document itself: -o X_mono.md
            # beside an input named X_bilingual.md must not delete the source.
            continue
        try:
            p.unlink()
        except OSError:
            continue
        deleted.append(p)
        for kind in RUN_REPORT_KINDS:
            with contextlib.suppress(OSError):
                sidecar_path(p, kind).unlink(missing_ok=True)
    return deleted


def _same_file(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return False


def _refuse_existing_output(
    output: Path | None,
    *,
    fresh: bool | None,
    input_path: Path,
    pages: str | None = None,
) -> Path | None:
    """Mirror the API 409 / MCP ToolError overwrite guard on the CLI surface.

    The other two entry points refuse to overwrite an existing deliverable
    unless the caller asks for a fresh run; the CLI holds the same line, and
    its default output name collides on a second run. The candidate set is
    :func:`~ubt.core.job_options.deliverable_candidates` — every name this
    run could actually write — so a suffix-less ``-o out/delivery`` (which
    resolves to ``out/delivery.md``) and the monolingual name inside a
    directory are both guarded, not just the bilingual one.

    ``pages`` must be the run's own page selection: a ``--pages`` run writes a
    scope-suffixed default name, and guarding the unscoped one would both
    protect a file this run will not touch and miss the one it will.
    """
    from ubt.core.config import parse_page_ranges

    try:
        page_set = parse_page_ranges(pages)
    except ValueError:
        # The range is validated later, with a better message; the guard must
        # not be the thing that reports it.
        page_set = None

    if output is not None and _same_file(output, input_path):
        raise typer.BadParameter(
            f"output path equals the input document; refusing to overwrite it: {output}"
        )
    if output is None:
        # The default deliverable is resolved inside the export stage, so guard
        # both default names here — a second run without --fresh must fail
        # loudly instead of overwriting the first delivery.
        if not fresh:
            for candidate in deliverable_candidates(None, input_path, pages=page_set):
                if candidate.exists():
                    raise typer.BadParameter(
                        f"default output already exists; refusing to overwrite it: {candidate} "
                        "(pass --fresh to overwrite or -o to choose another path)"
                    )
        return None
    if fresh:
        removed = _clean_stale_companions(output, input_path=input_path)
        if removed:
            # Say it, don't just do it: these are files this run did not write
            # and will not rewrite unless the run emits the same companion.
            console.print(
                "[dim]--fresh removed the previously delivered companion(s):[/] "
                + ", ".join(escape(p.name) for p in removed)
            )
        return output
    for candidate in deliverable_candidates(output, input_path, pages=page_set):
        if candidate.exists():
            raise typer.BadParameter(
                f"output path already exists; refusing to overwrite it: {candidate} "
                "(pass --fresh to overwrite)"
            )
    return output


def _delivery_verdict(summary: dict[str, Any]) -> tuple[str, str]:
    """The three-state delivery verdict from a quality-report summary.

    Thin CLI wrapper over the shared authority
    :func:`ubt.core.job_options.delivery_verdict_from_summary` — the web
    quality gate and the status surface consume the same rule, so four
    facades answer "is this delivered" identically.

    Returns ``(state, message)`` where ``state`` is ``"passed"``,
    ``"warning"``, or ``"blocked"`` -- the PASSED / WARNING / BLOCKED
    vocabulary of the operator console (``docs/UBT_OPERATOR_CONSOLE_PRD.md``).
    The artifact is delivered in all three cases; the state only says whether
    the reader will meet untranslated source.
    """
    failed = int(summary.get("failed_blocks", 0) or 0)
    blocked = int(summary.get("blocked_human_blocks", 0) or 0)
    needs_human = int(summary.get("needs_human_blocks", 0) or 0)
    state = delivery_verdict_from_summary(summary)
    if state == "blocked" and (failed or blocked):
        return (
            "blocked",
            _ui_msg(
                f"⚠ Delivered with {blocked + failed} block(s) shipped untranslated "
                "(source shown; quarantined for human post-editing) — inspect the "
                "quality report before publishing",
                f"⚠ 已交付但含 {blocked + failed} 个未译块（出厂为原文，已隔离待人工后编）"
                "— 发布前请查看质量报告",
            ),
        )
    if state == "blocked":
        return (
            "blocked",
            _ui_msg(
                "⚠ 0 blocks passed automated QA — inspect the quality report before delivery",
                "⚠ 0 块通过自动质检 — 请查看质量报告后再交付",
            ),
        )
    if state == "warning":
        return (
            "warning",
            _ui_msg(
                f"⚠ Delivered with {needs_human} block(s) awaiting human review "
                "(translations shipped) — verify the quality report",
                f"⚠ 已交付，含 {needs_human} 个待人工审校块（译文已出厂）— 请核对质量报告",
            ),
        )
    return "passed", ""


def _qe_measured_from_report(report_data: dict[str, Any]) -> bool:
    """Whether the quality report says any block was actually QE-scored.

    Under the default heuristic engine nothing is scored and the report's
    avg-QE fields are 0.0 sentinels; the CLI banner and JSON carry this flag
    so an unmeasured run is never read as a measured zero.
    """
    metrics = report_data.get("score_metrics", {})
    if not isinstance(metrics, dict):
        return False
    return bool(metrics.get("qe_measured", False))


def _report_degrades_bilingual_delivery(report_data: dict[str, Any]) -> bool:
    """Whether the quality report says the delivery is not a bilingual document.

    The reporter serializes the advisory under ``mode_advisory``; the CLI used
    to look up a key that never exists in the report, so degraded deliveries
    were announced as bilingual documents.
    """
    advisory = report_data.get("mode_advisory", {})
    if not isinstance(advisory, dict):
        return False
    rendered_modes = advisory.get("rendered_modes", [])
    effective = advisory.get("effective")
    return (
        rendered_modes == ["mono"] or effective == "mono" or "dual_mode_downgraded" in report_data
    )


def _find_companion_paths(
    result_path: Path, *, report_data: dict[str, Any] | None = None
) -> list[Path]:
    """Existing complementary artifacts the pipeline wrote beside the primary.

    When --emit-both is on, the pipeline emits a complementary mono/dual render
    (e.g. ``_bilingual`` or ``_dual`` when the primary is ``_mono``, or ``_mono``
    when the primary is bilingual).
    """
    stem = result_path.stem
    names: list[str] = []
    if stem.endswith("_mono"):
        base = stem[: -len("_mono")]
        names.extend([f"{base}_bilingual{result_path.suffix}", f"{base}_dual{result_path.suffix}"])
    elif stem.endswith("_bilingual"):
        base = stem[: -len("_bilingual")]
        names.append(f"{base}_mono{result_path.suffix}")
    elif stem.endswith("_dual"):
        base = stem[: -len("_dual")]
        names.append(f"{base}_mono{result_path.suffix}")
    else:
        names.extend(
            [
                f"{stem}_mono{result_path.suffix}",
                f"{stem}_dual{result_path.suffix}",
                f"{stem}_bilingual{result_path.suffix}",
            ]
        )

    res_mtime = result_path.stat().st_mtime if result_path.exists() else 0.0
    companions: list[Path] = []
    for p in (result_path.with_name(n) for n in names):
        if not p.exists() or p == result_path:
            continue
        # Stale artifact rejection: a companion must have been written in the same
        # run window as the primary deliverable (within 180s), not days ago.
        if res_mtime > 0 and abs(p.stat().st_mtime - res_mtime) > 180:
            continue
        companions.append(p)
    return companions


def translate(
    input_path: Annotated[Path, typer.Argument(help="Path to input document (.epub, .md, .pdf)")],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Target output bilingual document path (default: <documents>/UBT/<stem>_bilingual<suffix>)",
        ),
    ] = None,
    source_lang: Annotated[
        str, typer.Option("--source-lang", "-s", help="Source document language (default: en)")
    ] = "en",
    target_lang: Annotated[
        str, typer.Option("--target-lang", "-l", help="Target translation language")
    ] = "zh",
    profile: Annotated[
        str,
        typer.Option(
            "--profile",
            "--domain-profile",
            "-p",
            help="Domain profile name (e.g. auto, general, textbook, paper, fiction, "
            "humanities, semiconductor); 'auto' infers from document archetype",
        ),
    ] = "auto",
    columns: Annotated[
        int | None,
        typer.Option(
            "--columns",
            help="Physical column count for a reflowed PDF: 1 (single) or 2 (dual). "
            "Default (unset) aligns to the source layout — a two-column paper stays "
            "two-column, a single-column one stays single-column. This is the layout "
            "switch; --profile only styles the text and never forces a column count.",
        ),
    ] = None,
    glossary: Annotated[
        Path | None,
        typer.Option(
            "--glossary",
            "-g",
            help="Path to external user glossary file (.csv, .tsv, or .json) for domain terms",
        ),
    ] = None,
    domain: Annotated[
        str | None,
        typer.Option(
            "--domain",
            help="Domain or subject field descriptor (e.g. 'semiconductor device physics', 'biomedicine')",
        ),
    ] = None,
    preset: Annotated[
        Preset | None,
        typer.Option(
            "--preset",
            help="Quality tier (prompt depth + formula care; never the engine): 'publication' (rich context + term bible + lossless ingest), 'standard' (model-adaptive prompts + lossless ingest), 'preview' (fast pdfium ingest path, formula source images), or 'fast' (fast pdfium ingest path, short chain). An explicit flag always beats the preset; without this option the engine defaults apply",
            show_default=False,
        ),
    ] = None,
    api_key: Annotated[
        str | None,
        typer.Option(
            "--api-key",
            help=(
                "Outbound LLM API key (overrides UBT_LLM_API_KEY and provider's "
                "configured api_key). Prefer the env var: argv is visible in "
                "`ps` and shell history"
            ),
        ),
    ] = None,
    base_url: Annotated[
        str | None,
        typer.Option(
            "--base-url",
            help="Outbound LLM Base URL (e.g. https://api.openai.com/v1, https://generativelanguage.googleapis.com/v1beta/openai, https://api.anthropic.com)",
        ),
    ] = None,
    api_mode: Annotated[
        ApiMode | None,
        typer.Option(
            "--api-mode",
            help="API protocol mode: 'openai-chat', 'openai-responses', 'anthropic-messages', or 'gemini-native'",
        ),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider",
            help="Provider name — built-in protocol ('openai-chat'/'openai', 'anthropic-messages'/'anthropic', 'gemini-native'/'gemini', 'openai-responses') or declared in ubt.toml / ~/.ubt/config.toml (e.g. 'deepseek', 'openrouter', 'ollama')",
        ),
    ] = None,
    repair_provider: Annotated[
        str | None,
        typer.Option(
            "--repair-provider",
            help="Secondary provider name for repair tier (defaults to primary provider)",
        ),
    ] = None,
    visual_judge: Annotated[
        bool | None,
        typer.Option(
            "--visual-judge/--no-visual-judge",
            help="Send sampled rendered pages to a vision LLM judge (T2) for "
            "layout/overlap/blank verdicts; off by default (spends tokens). "
            "Delivery-blocking parity majors fire regardless.",
            show_default=False,
        ),
    ] = None,
    visual_judge_model: Annotated[
        str | None,
        typer.Option(
            "--visual-judge-model",
            help="Vision model for the judge (default: provider's vision-capable tier)",
            show_default=False,
        ),
    ] = None,
    draft_model: Annotated[
        str | None,
        typer.Option(
            "--draft-model",
            "--model",
            "-m",
            help="Override draft translation model (and repair model if not set)",
        ),
    ] = None,
    repair_model: Annotated[
        str | None, typer.Option("--repair-model", help="Override tier-2 repair model")
    ] = None,
    db_dir: Annotated[
        Path | None,
        typer.Option(
            "--db-dir",
            help="SQLite ledger storage directory (default: from config / UBT_DB_DIR)",
            show_default=False,
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Simulate translation using mock provider without external LLM API spend",
        ),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="Confirm pre-flight layout tradeoff warnings without interactive pause",
        ),
    ] = False,
    start_chapter: Annotated[
        int, typer.Option("--start-chapter", help="Starting chapter index (1-based)")
    ] = 1,
    max_chapters: Annotated[
        int | None, typer.Option("--max-chapters", help="Maximum number of chapters to process")
    ] = None,
    budget_usd: Annotated[
        float | None,
        typer.Option(
            "--budget-usd",
            help="Hard USD cap for this job, counting every resume of the same job id; "
            "the run fails once the priced cost exceeds it (drafted progress and spend "
            "stay in the ledger). Unset follows UBT_BUDGET_USD",
            show_default=False,
        ),
    ] = None,
    rolling_summary: Annotated[
        bool | None,
        typer.Option(
            "--rolling-summary/--no-rolling-summary",
            help="Enable cross-chapter rolling summary (default: from config, True)",
        ),
    ] = None,
    concurrency: Annotated[
        int | None,
        typer.Option(
            "--concurrency",
            "-c",
            help="Concurrent LLM requests limit (default: from config, e.g. 10)",
        ),
    ] = None,
    batch_limit: Annotated[
        int | None,
        typer.Option(
            "--batch-limit",
            "-b",
            help="Keyset pagination batch size (default: from config, e.g. 30)",
        ),
    ] = None,
    macro_chunk_size: Annotated[
        int | None,
        typer.Option(
            "--macro-chunk-size",
            help="Number of consecutive micro-blocks to pack per LLM request (e.g. 5) for ultra-fast long documents (default: 1)",
            show_default=False,
        ),
    ] = None,
    chapter_streaming: Annotated[
        bool | None,
        typer.Option(
            "--chapter-streaming/--no-chapter-streaming",
            help="Enable chapter-level streaming pipeline for multi-chapter documents (decouples stage barriers)",
            show_default=False,
        ),
    ] = None,
    offline_batch: Annotated[
        bool | None,
        typer.Option(
            "--offline-batch/--no-offline-batch",
            help="Draft entire book asynchronously using cloud Batch API (OpenAI/Anthropic 50% discount, high throughput)",
            show_default=False,
        ),
    ] = None,
    pdf_engine: Annotated[
        PdfEngine | None,
        typer.Option(
            "--pdf-engine",
            help="PDF 解析引擎：'auto'（默认，恒用无损 docling 引擎）、'pdfium'（强制快速文本提取——其读取器无法表示表格/图/公式，交付的书会缺这些内容，仅适用于已知纯散文的文档）、或 'docling'（强制无损引擎）",
            show_default=False,
        ),
    ] = None,
    dual_mode: Annotated[
        DualMode | None,
        typer.Option(
            "--dual-mode",
            help="What the reader receives: 'dual' (each block's source line above its translation), 'mono' (target only), or 'auto' (the advisor picks from the document's structure). Unset follows config/UBT_DUAL_MODE",
            show_default=False,
        ),
    ] = None,
    translate_chrome: Annotated[
        bool | None,
        typer.Option(
            "--translate-chrome/--no-translate-chrome",
            help="Translate running heads and footers (page numbers never); off by default so chrome stays source-visible",
            show_default=False,
        ),
    ] = None,
    emit_both: Annotated[
        bool | None,
        typer.Option(
            "--emit-both/--no-emit-both",
            help="Also render the complementary artifact (mono when primary is dual and vice versa)",
            show_default=False,
        ),
    ] = None,
    fresh: Annotated[
        bool | None,
        typer.Option(
            "--fresh",
            help="Discard existing ledger blocks and re-ingest from scratch (parse-stage fixes and changed sources need this; resume is the default)",
            show_default=False,
        ),
    ] = None,
    ocr: Annotated[
        OcrMode | None,
        typer.Option(
            "--ocr",
            help="Pluggable OCR mode: 'auto' (probe Docker sidecar -> local rapidocr -> Cloud/VLM last, egress gated by UBT_ALLOW_PAGE_UPLOAD), 'sidecar', 'cloud', 'vlm', 'rapidocr', or 'off'. Unset follows config/UBT_OCR_MODE",
            show_default=False,
        ),
    ] = None,
    ocr_endpoint: Annotated[
        str | None,
        typer.Option(
            "--ocr-endpoint",
            help="OCR server endpoint URL (e.g. http://localhost:8765 or https://api.openai.com/v1)",
        ),
    ] = None,
    ocr_api_key: Annotated[
        str | None,
        typer.Option(
            "--ocr-api-key",
            help="API key for Cloud OCR or Vision LLM provider (or UBT_OCR_API_KEY)",
        ),
    ] = None,
    short_max_pages: Annotated[
        int | None,
        typer.Option(
            "--short-max-pages",
            help="Short-chain page ceiling (default: from config, 30)",
        ),
    ] = None,
    pages: Annotated[
        str | None,
        typer.Option(
            "--pages",
            "--page-range",
            help="Page range to process for PDFs (e.g. '1-2', '1,3,5', '5-10')",
        ),
    ] = None,
    job_id: Annotated[
        str | None,
        typer.Option(
            "--job-id",
            help="Explicit job identity (letters/digits/-/_): reruns resume the same ledger instead of deriving job_<hash>",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Machine-readable mode: stdout carries exactly one JSON object (no Rich rendering)",
        ),
    ] = False,
    strict: Annotated[
        bool,
        typer.Option(
            "--strict",
            help="Exit non-zero unless the run is fully clean: no failed / needs-human / blocked blocks and no fail-closed render skips.",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable verbose DEBUG logging to stderr.",
        ),
    ] = False,
) -> None:
    """Translate an entire book end-to-end with 4-layer defense and live dual dashboard."""
    # ``--profile auto`` probes the document (paper/fiction/semiconductor/…);
    # the shared resolver is the same one every other surface goes through, so
    # the CLI no longer owns the only copy of this rule.
    from ubt.core.job_options import resolve_profile_name as _resolve_profile

    profile = _resolve_profile(profile, input_path)

    # The shared job-request surface (ubt.core.job_options): keys are
    # UBTConfig field names (plus the ``glossary`` alias), ``None`` means
    # "flag not passed" and falls through to env/preset. CLI flag names
    # that differ from the config field are mapped here, once, and the
    # orchestrator consumes the mapping as-is.
    request: dict[str, Any] = {
        "target_lang": target_lang,
        "source_lang": source_lang,
        "profile": profile,
        "job_id": job_id,
        "layout_columns": columns,
        "start_chapter": start_chapter,
        "max_chapters": max_chapters,
        "draft_model": draft_model,
        "repair_model": repair_model,
        "db_dir": db_dir,
        "budget_usd": budget_usd,
        "enable_rolling_summary": rolling_summary,
        "max_concurrency": concurrency,
        "batch_limit": batch_limit,
        "macro_chunk_size": macro_chunk_size,
        "chapter_streaming_enabled": chapter_streaming,
        "offline_batch_enabled": offline_batch,
        "dual_mode": dual_mode,
        "translate_chrome": translate_chrome,
        "preset": preset,
        "pdf_engine": pdf_engine,
        "emit_both": emit_both,
        "fresh": fresh,
        "ocr_mode": ocr,
        "ocr_endpoint": ocr_endpoint,
        "ocr_api_key": ocr_api_key,
        "short_max_pages": short_max_pages,
        "glossary": glossary,
        "domain": domain,
        "pages": pages,
        "api_key": api_key,
        "base_url": base_url,
        "api_mode": api_mode,
        "provider": provider,
        "repair_provider": repair_provider,
        "visual_judge_enabled": visual_judge,
        "visual_judge_model": visual_judge_model,
    }

    from ubt.core.job_options import lang_pair_validation_error

    # Usage errors exit 2 (typer's own convention for bad usage); runtime
    # failures keep exit 1. The language gate is the shared one, so the CLI,
    # the API and MCP word it identically.
    lang_error = lang_pair_validation_error(request.get("source_lang"), request.get("target_lang"))
    if lang_error is not None:
        if json_output:
            print(json.dumps({"status": "failed", "error": lang_error}))
        else:
            key_part, _, rest = lang_error.partition(":")
            console.print(f"[bold red]{key_part}:[/]{rest}")
        raise typer.Exit(code=2)

    # A well-formed but unsupported profile would otherwise ingest the whole
    # book and only then fail in the pipeline. Reject it here, before any
    # token is spent.
    if not profile_name_is_valid(profile):
        err_msg = (
            f"Invalid domain profile: {profile!r}. Use a name of letters, digits, "
            f"'-' or '_' (examples: {_PROFILE_EXAMPLES})."
        )
        if json_output:
            print(json.dumps({"status": "failed", "error": err_msg}))
        else:
            console.print(
                f"[bold red]Invalid domain profile:[/] {escape(str(profile))!r}. "
                f"Use a name of letters, digits, '-' or '_' "
                f"(examples: {_PROFILE_EXAMPLES})."
            )
        raise typer.Exit(code=2)

    # Chapter-window and page-range guards. A non-positive window is refused
    # here so a run cannot deliver a single-chapter book under a "success"
    # banner; a malformed --pages is refused before ingestion begins. Both are
    # usage errors: exit 2.
    # any work starts.
    if start_chapter < 1:
        _usage_error(f"--start-chapter must be >= 1 (got {start_chapter})", json_output)
    if max_chapters is not None and max_chapters < 1:
        _usage_error(f"--max-chapters must be >= 1 (got {max_chapters})", json_output)
    if columns is not None and columns not in (1, 2):
        _usage_error(f"--columns must be 1 or 2 (got {columns})", json_output)
    if pages is not None:
        try:
            parse_page_ranges(pages)
        except ValueError as exc:
            _usage_error(f"invalid --pages: {exc}", json_output)

    # ``verbose`` is this command's own flag, but the global ``-v`` (main
    # callback) also enables DEBUG; re-calling setup_logging here with
    # verbose=False would reset the level to INFO and discard it.
    if verbose or logging.getLogger().isEnabledFor(logging.DEBUG):
        setup_logging(verbose=True, console=None if json_output else console)
    elif json_output:
        setup_logging(level="WARNING")
    else:
        setup_logging(console=console)

    if not input_path.exists():
        # A bad path is bad usage, not a runtime failure: exit 2.
        if json_output:
            print(json.dumps({"status": "failed", "error": f"Input file not found: {input_path}"}))
            raise typer.Exit(code=2)
        console.print(f"[bold red]Error:[/] Input file not found: {input_path}")
        raise typer.Exit(code=2)

    if not json_output:
        console.print(
            Panel.fit(
                f"[bold cyan]UBT Universal Book Translator{' [yellow](DRY RUN)[/]' if dry_run else ''}[/]\n"
                f"[dim]Input:[/] {input_path.name}  |  [dim]Source:[/] {source_lang}  |  [dim]Target:[/] {target_lang}  |  [dim]Profile:[/] {profile}",
                border_style="yellow" if dry_run else "cyan",
            )
        )
    output = _refuse_existing_output(output, fresh=fresh, input_path=input_path, pages=pages)
    try:
        run_fn = _get_run_translation()
        result_path = asyncio.run(
            run_fn(
                input_path=input_path,
                output_path=output,
                dry_run=dry_run,
                quiet=json_output,
                **request,
            )
        )
        if result_path is None:
            if json_output:
                print(
                    json.dumps(
                        {
                            "status": "failed",
                            "error": "No output document was generated",
                        }
                    )
                )
            else:
                console.print(
                    "\n[bold red]Translation failed: no output document was generated.[/]"
                )
            raise typer.Exit(code=1)
        report_path = sidecar_path(result_path, "quality_report.json")
        visual_path = sidecar_path(result_path, "visual_report.json")
        report_data: dict[str, Any] = {}
        if report_path.exists():
            with contextlib.suppress(Exception):
                report_data = json.loads(report_path.read_text(encoding="utf-8"))
        companions = _find_companion_paths(result_path, report_data=report_data)
        if strict:
            strict_fn = _get_strict_failures()
            failures = strict_fn(report_path)
            if failures:
                if json_output:
                    print(
                        json.dumps(
                            {
                                "status": "strict_failed",
                                "output_file": str(result_path),
                                "strict_failures": failures,
                            }
                        )
                    )
                else:
                    console.print(
                        f"\n[bold red]Strict gate failed:[/] {escape('; '.join(failures))}"
                    )
                raise typer.Exit(code=1)
        # Delivery verdict: an exported artifact with quarantined or failed
        # blocks must not be announced as an unqualified success. The summary
        # and score metrics are read once, before either output path, so JSON
        # and the banner agree.
        summary: dict[str, Any] = {}
        if report_path.exists():
            try:
                report_root = json.loads(report_path.read_text(encoding="utf-8"))
                summary = report_root.get("summary", {}) or {}
                report_data = report_root
            except (OSError, json.JSONDecodeError):
                summary = {}
        verdict, verdict_msg = _delivery_verdict(summary) if summary else ("passed", "")
        # Was any block actually scored? Under the default heuristic QE engine
        # nothing is, and the avg-QE line then reads a 0.0 sentinel — say so
        # instead of letting "avg QE 0.000" masquerade as a measured zero.
        qe_measured = _qe_measured_from_report(report_data)
        if json_output:
            print(
                json.dumps(
                    {
                        "status": (
                            "completed" if verdict == "passed" else f"completed_with_{verdict}"
                        ),
                        "delivery_verdict": verdict,
                        "blocked_human_blocks": int(summary.get("blocked_human_blocks", 0) or 0),
                        "needs_human_blocks": int(summary.get("needs_human_blocks", 0) or 0),
                        "qe_measured": qe_measured,
                        "output_file": str(result_path),
                        "companion_file": str(companions[0]) if companions else None,
                        "companion_files": [str(p) for p in companions],
                        "quality_report": str(report_path) if report_path.exists() else None,
                        "visual_report": str(visual_path) if visual_path.exists() else None,
                    }
                )
            )
            return
        if verdict != "passed":
            color = "bold red" if verdict == "blocked" else "bold yellow"
            console.print(f"\n[{color}]{verdict_msg}[/]")
        is_bilingual = True
        if report_data:
            if _report_degrades_bilingual_delivery(report_data):
                is_bilingual = False
        elif dual_mode == "mono":
            is_bilingual = False

        doc_label = (
            _ui_msg("Bilingual Document:", "双语对照文档:")
            if is_bilingual
            else _ui_msg("Translated Document:", "译文交付文档:")
        )
        if verdict == "passed":
            console.print("\n[bold green]✓ Translation Completed Successfully![/]")
        else:
            console.print("\n[bold]✓ Translation completed[/] [yellow](with warnings)[/]")
        console.print(f"  [cyan]{doc_label}[/] [link=file://{result_path}]{result_path}[/]")
        if not qe_measured and report_path.exists():
            # Three different reasons produce an unscored run, and telling the
            # operator to enable something they already enabled is worse than
            # saying nothing: it hides a real failure behind a config hint.
            judge_calls = int(report_data.get("score_metrics", {}).get("qe_judge_calls") or 0)
            judge_errors = int(report_data.get("score_metrics", {}).get("qe_judge_errors") or 0)
            if judge_calls and judge_errors >= judge_calls:
                console.print(
                    _ui_msg(
                        f"  [yellow]Quality: the judge was called {judge_calls} time(s) "
                        f"and every call failed[/] [dim]— the scores fall back to the "
                        "heuristic. Check the judge endpoint's credentials, model name "
                        "and reachability (see SEMANTIC_JUDGE.md).[/]",
                        f"  [yellow]质量：judge 被调用 {judge_calls} 次，全部失败[/] "
                        "[dim]— 分数退回启发式。检查 judge 端点的凭证、模型名与可达性"
                        "（见 SEMANTIC_JUDGE.md）。[/]",
                    )
                )
            elif judge_calls:
                console.print(
                    _ui_msg(
                        "  [dim]Quality: sampled audit only — no judged block scored below "
                        f"the line ({judge_calls} call(s), {judge_errors} failed). "
                        "Raise qe_judge_pass_sample for wider coverage.[/]",
                        "  [dim]质量：仅抽样审计 — 没有被判定的块低于阈值"
                        f"（调用 {judge_calls} 次，失败 {judge_errors} 次）。"
                        "提高 qe_judge_pass_sample 可扩大覆盖。[/]",
                    )
                )
            else:
                console.print(
                    _ui_msg(
                        "  [dim]Quality: not machine-scored this run (heuristic QE engine; "
                        "avg-QE figures in the report are placeholders). "
                        "Set UBT_QE_JUDGE_ENABLED=true (with UBT_QE_ENGINE=tiered) to measure.[/]",
                        "  [dim]质量：本次运行未做机器评分（启发式 QE 引擎，报告中的 avg-QE "
                        "为占位值）。设置 UBT_QE_JUDGE_ENABLED=true（配合 UBT_QE_ENGINE=tiered）"
                        "以启用测量。[/]",
                    )
                )
        for companion in companions:
            console.print(
                f"  [green]Companion Artifact:[/] [link=file://{companion}]{companion}[/]"
            )
        if report_path.exists():
            console.print(
                f"  [yellow]Quality Report:[/]     [link=file://{report_path}]{report_path}[/]"
            )
        if visual_path.exists():
            console.print(
                f"  [magenta]Visual Report:[/]      [link=file://{visual_path}]{visual_path}[/]"
            )
    except typer.Exit:
        raise
    except KeyboardInterrupt:
        # Click would print a bare "Aborted!" and exit 1. The checkpoints are
        # on disk and the job is resumable, but only if the operator knows the
        # id — say so instead of leaving them to re-run (and re-pay) from
        # scratch. The id line itself is printed at JOB_STARTED.
        if json_output:
            print(json.dumps({"status": "cancelled", "error": "interrupted by user"}))
        else:
            console.print(
                _ui_msg(
                    "\n[yellow]Interrupted — completed blocks are checkpointed to the ledger.[/]",
                    "\n[yellow]已中断 — 已完成的块已写入账本检查点。[/]",
                )
            )
            console.print(
                _ui_msg(
                    "  [dim]Resume with: [bold]ubt translate <input> --job-id <id>[/] "
                    "(the Job ID line above lists it; `ubt status` also lists recent jobs).[/]",
                    "  [dim]续跑：[bold]ubt translate <input> --job-id <id>[/]"
                    "（即上方 Job ID 行；`ubt status` 也会列出最近的 job）。[/]",
                )
            )
        raise typer.Exit(code=130) from None
    except PermissionError as exc:
        if json_output:
            print(json.dumps({"status": "failed", "error": str(exc)}))
        else:
            console.print(
                _ui_msg(
                    f"\n[bold red]Output path is not writable:[/] {escape(str(exc))}",
                    f"\n[bold red]输出路径不可写:[/] {escape(str(exc))}",
                )
            )
            console.print(
                _ui_msg(
                    "  [dim]Hint: verify that the -o target directory exists and has write permissions.[/]",
                    "  [dim]提示：检查 -o 目标目录是否存在、当前用户是否有写权限[/]",
                )
            )
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        if json_output:
            print(json.dumps({"status": "failed", "error": str(exc)}))
            raise typer.Exit(code=1) from exc
        console.print(f"\n[bold red]Pipeline Execution Failed:[/] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc
