"""CLI command for end-to-end document translation."""

import asyncio
import contextlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Annotated, Any, Literal

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from ubt.core.config import (
    ApiMode,
    CoverMode,
    DualMode,
    ExecMode,
    FormulaEnrichment,
    FormulaMode,
    FormulaRender,
    MathBackend,
    OcrMode,
    PromptStrategyName,
    QeEngine,
    canonical_render_engine,
)
from ubt.core.job_options import profile_name_is_valid, sidecar_path
from ubt.core.log_config import setup_logging
from ubt.core.presets import Preset


def _ui_msg(en: str, zh: str) -> str:
    """Return English CLI copy by default, or Chinese when UBT_UI_LANG=zh."""
    return zh if os.environ.get("UBT_UI_LANG", "en").strip().lower().startswith("zh") else en


UserRenderEngine = Literal["rigid", "reflow", "auto"]

console = Console()


def _get_run_translation() -> Any:
    # Deferred import of the shared entry helper: main.py imports this command
    # at module load, so importing main here (not at top level) avoids the
    # cycle while still resolving the *live* (monkeypatchable) module attribute.
    from ubt.cli.main import _run_translation

    return _run_translation


def _get_strict_failures() -> Any:
    from ubt.cli.main import _strict_failures

    return _strict_failures


def _is_interactive() -> bool:
    """Return True if running in an interactive terminal session."""
    return sys.stdin.isatty()


#: Domain-profile names are not a closed set: ``seed_entries_for_profile`` and
#: the packaged glossary directories key on names like ``semiconductor`` /
#: ``semiconductor_paper``, and operators may add their own resource dirs. The
#: only invariant is the shared safe-name pattern, so the CLI validates with
#: ``profile_name_is_valid`` exactly like the API and MCP — the old hardcoded
#: allowlist rejected every profile that actually had glossary seeds.
_PROFILE_EXAMPLES = "general, textbook, paper, fiction, humanities, semiconductor"


def translate(
    input_path: Annotated[Path, typer.Argument(help="Path to input document (.epub, .md, .pdf)")],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Target output bilingual document path (default: tmp/output/<stem>_bilingual<suffix>)",
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
            help="Domain profile name (e.g. general, textbook, paper, fiction, "
            "humanities, semiconductor); also names a packaged glossary directory",
        ),
    ] = "general",
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
            help="Quality tier (prompt depth + formula care; never the render route): 'publication' (rich context + term bible), 'standard' (model-adaptive prompts), 'preview' (formula source images), or 'fast' (zero-model fast paper). An explicit flag always beats the preset; without this option the engine defaults apply",
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
    qe_engine: Annotated[
        QeEngine | None,
        typer.Option(
            "--qe-engine",
            help="Quality Estimation engine: 'heuristic' (default fast) or 'comet' (neural CometKiwi)",
        ),
    ] = None,
    render_engine: Annotated[
        UserRenderEngine | None,
        typer.Option(
            "--render-engine",
            help="PDF 渲染引擎：'auto'（默认，智能路由：公式/表格密集→rigid 保真，纯正文→reflow 重排）、'rigid'（原位覆盖底板，单语输出，图形表格零丢失）、'reflow'（Typst 从零重排，支持双语，适合结构抽取可靠的正文文档）",
            show_default=False,
        ),
    ] = None,
    dual_mode: Annotated[
        DualMode | None,
        typer.Option(
            "--dual-mode",
            help="Bilingual render mode: 'inline' (interleaved, default), 'alternating' (page zipper), 'facing' (spread with flyleaf), 'monolingual' (target only), or 'auto' (advisor decides). Unset follows config/UBT_DUAL_MODE",
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
    facing_spread: Annotated[
        bool | None,
        typer.Option(
            "--facing-spread/--no-facing-spread",
            help="Pad recto flyleaf in alternating PDF mode so facing pages align properly",
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
    cover_mode: Annotated[
        CoverMode | None,
        typer.Option(
            "--cover-mode",
            help="Cover policy: 'auto' (cover only when page 1 has no body text), 'always', or 'never'. Unset follows config",
            show_default=False,
        ),
    ] = None,
    prompt_strategy: Annotated[
        PromptStrategyName | None,
        typer.Option(
            "--prompt-strategy",
            help="Prompt assembly: 'auto' (per-model registry), or force 'minimal'/'hybrid'/'rich'",
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
    exec_mode: Annotated[
        ExecMode | None,
        typer.Option(
            "--mode",
            "--exec-mode",
            help="Execution router: 'auto' (≤short-max born-digital pages → short chain, else long 6-stage), or force 'short'/'long'",
            show_default=False,
        ),
    ] = None,
    formula_mode: Annotated[
        FormulaMode | None,
        typer.Option(
            "--formula-mode",
            help="Short-chain formula policy: 'readable' (Unicode近似, default) or 'strict' (byte-identical gates). Unset follows config",
            show_default=False,
        ),
    ] = None,
    formula_enrichment: Annotated[
        FormulaEnrichment | None,
        typer.Option(
            "--formula-enrichment",
            help="Docling math formula enrichment (CodeFormulaV2 VLM): 'auto' (GPU-enabled, degrades on failure), 'on' (force VLM), or 'off' (bypass VLM, 10x faster ingest)",
            show_default=False,
        ),
    ] = None,
    formula_render: Annotated[
        FormulaRender | None,
        typer.Option(
            "--formula-render",
            help="Display-formula fidelity: 'witness' (default; verify converted math against source pixels, fall back to the source graphic on mismatch), 'image' (always render display formulas from their source graphic), 'native' (converted Typst math only)",
            show_default=False,
        ),
    ] = None,
    math_backend: Annotated[
        MathBackend | None,
        typer.Option(
            "--math-backend",
            help="Display-formula renderer: 'mathjax' (default; typeset the OCR LaTeX with MathJax into an SVG vector, verify it against the source and fall back to the source graphic; degrades to 'typst' when Node is absent), 'image' (source crop for every display formula), 'typst' (legacy PDF-formula fidelity modes)",
            show_default=False,
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
        "qe_engine": qe_engine,
        "dual_mode": dual_mode,
        "translate_chrome": translate_chrome,
        "preset": preset,
        "facing_spread": facing_spread,
        "render_engine": render_engine,
        "emit_both": emit_both,
        "cover_mode": cover_mode,
        "prompt_strategy": prompt_strategy,
        "fresh": fresh,
        "ocr_mode": ocr,
        "ocr_endpoint": ocr_endpoint,
        "ocr_api_key": ocr_api_key,
        "exec_mode": exec_mode,
        "formula_mode": formula_mode,
        "formula_enrichment": formula_enrichment,
        "formula_render": formula_render,
        "math_backend": math_backend,
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

    from ubt.core.job_options import LANG_CODE_RE
    from ubt.core.language_profile import is_supported_lang

    for key in ("source_lang", "target_lang"):
        value = request.get(key)
        if value is not None and LANG_CODE_RE.fullmatch(str(value)) is None:
            if json_output:
                print(
                    json.dumps(
                        {
                            "status": "failed",
                            "error": (
                                f"Invalid {key.replace('_', '-')}: {str(value)!r}. "
                                "Use an ISO-ish language code such as 'en', 'zh' or 'zh-CN'."
                            ),
                        }
                    )
                )
            else:
                console.print(
                    f"[bold red]Invalid {key.replace('_', '-')}:[/] {escape(str(value))!r}. "
                    "Use an ISO-ish language code such as 'en', 'zh' or 'zh-CN'."
                )
            raise typer.Exit(code=1)

    # A well-formed but unsupported target (e.g. 'pt-BR') would otherwise ingest
    # the whole book and only then raise "Unknown language profile" in the
    # pipeline. Reject it here, before any token is spent.
    target_lang_value = request.get("target_lang")
    if target_lang_value is not None and not is_supported_lang(str(target_lang_value)):
        err_msg = (
            f"Unsupported target-lang: {str(target_lang_value)!r}. Supported base "
            "languages: zh, en, ja, ko, fr, de, es, ru (region tags such as "
            "'zh-CN' are accepted)."
        )
        if json_output:
            print(json.dumps({"status": "failed", "error": err_msg}))
        else:
            console.print(
                f"[bold red]Unsupported target-lang:[/] "
                f"{escape(str(target_lang_value))!r}. Supported base languages: "
                "zh, en, ja, ko, fr, de, es, ru (region tags such as 'zh-CN' are accepted)."
            )
        raise typer.Exit(code=1)

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
        raise typer.Exit(code=1)

    # ``verbose`` is this command's own flag, but the global ``-v`` (main
    # callback) also enables DEBUG; re-calling setup_logging here with
    # verbose=False used to reset the level to INFO, silently discarding it.
    if verbose or logging.getLogger().isEnabledFor(logging.DEBUG):
        setup_logging(verbose=True, console=None if json_output else console)
    elif json_output:
        setup_logging(level="WARNING")
    else:
        setup_logging(console=console)

    if not input_path.exists():
        if json_output:
            print(json.dumps({"status": "failed", "error": f"Input file not found: {input_path}"}))
            raise typer.Exit(code=1)
        console.print(f"[bold red]Error:[/] Input file not found: {input_path}")
        raise typer.Exit(code=1)

    if not json_output:
        console.print(
            Panel.fit(
                f"[bold cyan]UBT Universal Book Translator{' [yellow](DRY RUN)[/]' if dry_run else ''}[/]\n"
                f"[dim]Input:[/] {input_path.name}  |  [dim]Source:[/] {source_lang}  |  [dim]Target:[/] {target_lang}  |  [dim]Profile:[/] {profile}",
                border_style="yellow" if dry_run else "cyan",
            )
        )
        if input_path.suffix.lower() == ".pdf" and render_engine in ("reflow", "publication"):
            pre_adv = None
            try:
                from ubt.core.advisor import DocumentAdvisor

                pre_adv = DocumentAdvisor.analyze(input_path)
            except Exception:
                pre_adv = None

            if pre_adv is not None and (
                getattr(pre_adv, "recommended_render_engine", None) == "rigid"
                or str(getattr(pre_adv, "math_density", "")).lower() == "high"
            ):
                from ubt.cli.main import resolve_cli_adaptive_dual_mode

                math_level = str(getattr(pre_adv, "math_density", "high"))
                adaptive_mode = (
                    resolve_cli_adaptive_dual_mode(dual_mode, profile, render_engine)
                    or "monolingual"
                )
                if dual_mode is not None:
                    requested_desc_en = (
                        f"--render-engine {render_engine} --dual-mode {dual_mode} (explicit)"
                    )
                    requested_desc_zh = (
                        f"--render-engine {render_engine} --dual-mode {dual_mode} (显式指定)"
                    )
                else:
                    requested_desc_en = (
                        f"--render-engine {render_engine} (adaptive dual-mode: {adaptive_mode})"
                    )
                    requested_desc_zh = (
                        f"--render-engine {render_engine} (自适应双语模式: {adaptive_mode})"
                    )

                panel_title = _ui_msg(
                    "[bold yellow]⚠ Pre-Flight Layout Tradeoff[/]",
                    "[bold yellow]⚠ 排版与双语模式风险预警 (Pre-Flight Layout Tradeoff)[/]",
                )
                panel_body = _ui_msg(
                    f"Detected dense math/2D structure ([cyan]math_density={math_level}[/], [cyan]profile={profile}[/]).\n"
                    f"• [dim]Recommended :[/] [bold green]--render-engine rigid[/]  [dim](preserves 2D equations/diagrams)[/]\n"
                    f"• [dim]Requested   :[/] [bold yellow]{requested_desc_en}[/]\n"
                    "  [dim]Note: reflow disassembles 2D page geometry and may fragment dense math.[/]\n\n"
                    "  [bold green][1][/] Switch to [bold green]'rigid'[/] (Recommended — 100% faithful 2D layout, mono)\n"
                    "  [bold yellow][2][/] Keep [bold yellow]'reflow'[/] + auto-emit [bold green]'*_rigid.pdf'[/] companion (0 extra tokens)\n"
                    "  [bold red][3][/] Abort",
                    f"检测到当前 PDF 为高密度公式/二维结构文档 ([cyan]math_density={math_level}[/], [cyan]profile={profile}[/])：\n"
                    f"• [dim]系统推荐 :[/] [bold green]--render-engine rigid[/]  [dim](100% 原位保留交换图与公式几何坐标)[/]\n"
                    f"• [dim]当前指定 :[/] [bold yellow]{requested_desc_zh}[/]\n"
                    "  [dim]注意：流式重排 (reflow) 会拆解二维页面坐标，可能触发复杂公式截图回退或正文穿插顿挫。[/]\n\n"
                    "  [bold green][1][/] 切换为推荐的 [bold green]'rigid'[/] 原位引擎 (100% 保真二维几何版式，单语输出)\n"
                    "  [bold yellow][2][/] 继续 [bold yellow]'reflow'[/] 重排 + 零 Token 成本自动附赠 [bold green]'*_rigid.pdf'[/] 保真对照版\n"
                    "  [bold red][3][/] 取消并退出 (Abort)",
                )
                console.print(
                    Panel.fit(
                        panel_body, title=panel_title, title_align="left", border_style="yellow"
                    )
                )
                if _is_interactive() and not yes:
                    try:
                        console.print(
                            _ui_msg(
                                "[bold yellow]Select action [1=rigid / 2=reflow+companion (default) / 3=abort]: [/]",
                                "[bold yellow]请选择执行策略 [1=切换rigid / 2=继续reflow+双交付(默认) / 3=中断退出]: [/]",
                            ),
                            end="",
                        )
                        user_choice = input().strip().lower()
                    except (EOFError, KeyboardInterrupt) as exc:
                        console.print(
                            _ui_msg(
                                "\n[bold red]✕ Aborted by user.[/]",
                                "\n[bold red]✕ 用户已中断执行。[/]",
                            )
                        )
                        raise typer.Exit(code=130) from exc
                    if user_choice in ("3", "n", "no", "q", "quit", "abort"):
                        console.print(
                            _ui_msg(
                                "[bold red]✕ Execution aborted by user.[/]",
                                "[bold red]✕ 已根据您的选择中止任务。[/]",
                            )
                        )
                        raise typer.Exit(code=130)
                    if user_choice in ("1", "rigid"):
                        render_engine = "rigid"
                        dual_mode = "monolingual"
                        request["render_engine"] = "rigid"
                        request["dual_mode"] = "monolingual"
                        if output is not None:
                            orig_name = output.name
                            new_name = (
                                orig_name.replace("_reflow_bilingual", "_rigid_zh")
                                .replace("_reflow", "_rigid")
                                .replace("_bilingual", "_mono")
                            )
                            if new_name != orig_name:
                                output = output.with_name(new_name)
                                console.print(
                                    _ui_msg(
                                        f"[dim]Output target updated to '{output.name}' to match rigid monolingual delivery.[/]\n",
                                        f"[dim]输出目标文件名已自愈调整为 '{output.name}'（以匹配 rigid 单语保真交付）。[/]\n",
                                    )
                                )
                        console.print(
                            _ui_msg(
                                "[bold green]✓ Switched to '--render-engine rigid --dual-mode monolingual'.[/]\n",
                                "[bold green]✓ 已切换为推荐的 rigid 原位引擎 (--render-engine rigid --dual-mode monolingual)。[/]\n",
                            )
                        )
                    else:
                        request["emit_companion_rigid"] = True
                        console.print(
                            _ui_msg(
                                "[bold yellow]⚡ Proceeding with 'reflow' + zero-cost '*_rigid.pdf' companion delivery.[/]\n",
                                "[bold yellow]⚡ 已确认继续 reflow 重排，并在导出阶段自动零成本额外生成 *_rigid.pdf 保真对照文档。[/]\n",
                            )
                        )
                else:
                    request["emit_companion_rigid"] = True

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
        companion_rigid = result_path.with_name(f"{result_path.stem}_rigid.pdf")
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
        if json_output:
            print(
                json.dumps(
                    {
                        "status": "completed",
                        "output_file": str(result_path),
                        "companion_file": str(companion_rigid)
                        if companion_rigid.exists()
                        else None,
                        "quality_report": str(report_path) if report_path.exists() else None,
                        "visual_report": str(visual_path) if visual_path.exists() else None,
                    }
                )
            )
            return
        # Transparent banner: a run that exported but passed zero blocks through
        # automated QA must not be displayed as an unqualified success.
        summary: dict[str, Any] = {}
        if report_path.exists():
            try:
                summary = (
                    json.loads(report_path.read_text(encoding="utf-8")).get("summary", {}) or {}
                )
            except (OSError, json.JSONDecodeError):
                summary = {}
        if summary and not summary.get("completed_blocks"):
            awaiting = int(summary.get("needs_human_blocks", 0)) + int(
                summary.get("blocked_human_blocks", 0)
            )
            console.print(
                _ui_msg(
                    f"[bold yellow]⚠ 0 blocks passed automated QA[/] ({awaiting} pending human review) — inspect quality report before delivery",
                    f"[bold yellow]⚠ 0 块通过自动质检[/]（{awaiting} 块待人工审校）— 请查看质量报告后再交付",
                )
            )
        is_bilingual = True
        if report_path.exists():
            with contextlib.suppress(Exception):
                report_data = json.loads(report_path.read_text(encoding="utf-8"))
                bilingual_adv = report_data.get("bilingual_advisory", {})
                rendered_modes = bilingual_adv.get("rendered_modes", [])
                effective = bilingual_adv.get("effective")
                if (
                    rendered_modes == ["monolingual"]
                    or effective == "monolingual"
                    or "dual_mode_downgraded" in report_data
                    or "overlay engine" in report_data.get("delivery_status", "")
                    or "overlay engine" in report_data.get("delivery_warning", "")
                ):
                    is_bilingual = False
        elif dual_mode == "monolingual" or canonical_render_engine(render_engine) == "rigid":
            is_bilingual = False

        doc_label = (
            _ui_msg("Bilingual Document:", "双语对照文档:")
            if is_bilingual
            else _ui_msg("Translated Document:", "译文交付文档:")
        )
        console.print("\n[bold green]✓ Translation Completed Successfully![/]")
        console.print(f"  [cyan]{doc_label}[/] [link=file://{result_path}]{result_path}[/]")
        if companion_rigid.exists():
            console.print(
                f"  [green]Companion Faithful:[/] [link=file://{companion_rigid}]{companion_rigid}[/]"
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
