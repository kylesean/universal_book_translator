"""CLI command for document pre-translation assessment and quote generation."""

import json
import shlex
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from ubt.core.assess import SCHEMA_VERSION as ASSESS_SCHEMA_VERSION
from ubt.core.assess import AssessmentError, assess_document
from ubt.core.engine.pipeline import derive_job_id
from ubt.core.exceptions import UBTError
from ubt.core.ir.serializer import compute_file_sha256
from ubt.core.presets import Preset, resolve_engine_params

console = Console()


def _assess_money(value: float | None) -> str:
    """Unknown money is shown as 未知 — never a fabricated $0."""
    return "未知" if value is None else f"${value:.4f}"


def assess_cmd(
    input_path: Annotated[
        Path,
        typer.Argument(help="待评估文档 (.pdf / .epub / .docx / .md / .txt / .html)"),
    ],
    preset: Annotated[
        Preset | None,
        typer.Option(
            "--preset",
            help="按该预设报价（缺省时报告给出推荐预设）；语义与 translate --preset 完全一致。",
            show_default=False,
        ),
    ] = None,
    provider_profile: Annotated[
        str | None,
        typer.Option(
            "--provider-profile",
            help="按该 provider 档案（模型/端点）报价",
            show_default=False,
        ),
    ] = None,
    target_lang: Annotated[
        str, typer.Option("--target-lang", "-l", help="目标语言（影响报价与 job-id）")
    ] = "zh",
    source_lang: Annotated[str, typer.Option("--source-lang", "-s", help="源语言")] = "en",
    deep: Annotated[
        bool,
        typer.Option(
            "--deep",
            help="运行真实解析得到精确分块数（大 PDF 会加载 docling，可能耗时数分钟）",
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json", help="Machine-readable mode: stdout carries the assessment JSON object"
        ),
    ] = False,
) -> None:
    """译前报价与体检：零 token、零账本写入，给出判型/路线/预期成本/风险信号。

    不预测质量分数（MTQE 只存在于译文之上）；费用为「预期非保证」。
    """
    from ubt.core.job_options import LANG_CODE_RE
    from ubt.core.language_profile import is_supported_lang

    for key, val in (("source-lang", source_lang), ("target-lang", target_lang)):
        if val is not None and LANG_CODE_RE.fullmatch(str(val)) is None:
            if json_output:
                print(
                    json.dumps(
                        {
                            "schema_version": ASSESS_SCHEMA_VERSION,
                            "status": "error",
                            "code": "INVALID_LANG",
                            "error": f"Invalid {key}: {val!r}",
                        }
                    )
                )
                raise typer.Exit(code=2)
            console.print(f"[bold red]语言代码非法:[/] {escape(str(val))}")
            raise typer.Exit(code=2)

    # A well-formed but unsupported target (e.g. 'pt-BR') would otherwise fail
    # deep in the pipeline after ingest; reject it at the entry point.
    if target_lang is not None and not is_supported_lang(str(target_lang)):
        if json_output:
            print(
                json.dumps(
                    {
                        "schema_version": ASSESS_SCHEMA_VERSION,
                        "status": "error",
                        "code": "UNSUPPORTED_LANG",
                        "error": (
                            f"Unsupported target-lang: {target_lang!r}. Supported base "
                            "languages: zh, en, ja, ko, fr, de, es, ru"
                        ),
                    }
                )
            )
            raise typer.Exit(code=2)
        console.print(
            f"[bold red]不支持的目标语言:[/] {escape(str(target_lang))} "
            "(支持 zh, en, ja, ko, fr, de, es, ru；接受 zh-CN 等区域码)"
        )
        raise typer.Exit(code=2)

    overrides: dict[str, Any] = {}
    if provider_profile is not None:
        overrides["provider_profile"] = provider_profile
    # assess sets no explicit engine flag: quote the preset, or — with no
    # preset — the engine default (an all-None table was just {} in disguise).
    overrides.update(resolve_engine_params(preset, {}))
    from ubt.cli.main import _build_config

    try:
        config = _build_config(overrides)
    except (ValidationError, UBTError) as exc:
        # A bad --provider-profile raises ProfileNotFoundError (a UBTError), not
        # a ValidationError; without this the --json contract broke with a raw
        # traceback and empty stdout.
        if json_output:
            print(
                json.dumps(
                    {
                        "schema_version": ASSESS_SCHEMA_VERSION,
                        "status": "error",
                        "code": "INVALID_CONFIG",
                        "error": str(exc),
                    }
                )
            )
            raise typer.Exit(code=2) from exc
        console.print(f"[bold red]配置非法:[/] {escape(str(exc))}")
        raise typer.Exit(code=2) from exc

    try:
        report = assess_document(
            input_path, config, deep=deep, target_lang=target_lang, source_lang=source_lang
        )
    except AssessmentError as exc:
        if json_output:
            print(
                json.dumps(
                    {
                        "schema_version": ASSESS_SCHEMA_VERSION,
                        "status": "error",
                        "code": exc.code,
                        "error": str(exc),
                    }
                )
            )
            raise typer.Exit(code=1) from exc
        console.print(f"[bold red]无法评估:[/] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc

    effective_preset = preset or Preset(report.route.recommended_preset)
    command_parts = [
        "ubt",
        "translate",
        shlex.quote(str(input_path)),
        "--preset",
        effective_preset.value,
        "--target-lang",
        shlex.quote(target_lang),
        "--source-lang",
        shlex.quote(source_lang),
        "--job-id",
        derive_job_id(
            doc_id=compute_file_sha256(input_path),
            target_lang=target_lang,
            pages=None,
            start_chapter=1,
            max_chapters=None,
        ),
    ]
    if report.route.recommended_render_engine != "auto":
        command_parts += ["--render-engine", report.route.recommended_render_engine]
    if report.route.recommended_dual_mode != "inline":
        command_parts += ["--dual-mode", report.route.recommended_dual_mode]
    # The profile decides prompt assembly, glossary seeding and whether
    # chapter rolling summaries happen at all, so a quote taken under the
    # recommended one is only reproducible if the command carries it.
    if report.route.recommended_profile != "general":
        command_parts += ["--profile", shlex.quote(report.route.recommended_profile)]
    if provider_profile:
        command_parts += ["--provider-profile", shlex.quote(provider_profile)]
    report = replace(report, next_step_command=" ".join(command_parts))

    if json_output:
        print(json.dumps(report.to_dict(), ensure_ascii=False))
        return

    doc, rec, cost = report.document, report.route, report.cost
    console.print(
        Panel(
            f"[bold]{escape(doc.file_name)}[/]  ({doc.file_size_bytes / 1024:.0f} KB, .{doc.format_ext})\n"
            f"判型: [cyan]{escape(doc.category)}[/] · 领域 [magenta]{escape(doc.detected_domain)}"
            f"[/] (置信 {doc.domain_confidence}) · 数学密度 [yellow]{escape(doc.math_density)}[/]\n"
            f"规模: {doc.pages} 页 · {doc.chapters} 章 · {doc.source_chars} 字符 (~{doc.estimated_tokens} tok)\n"
            + (
                f"结构: 主引擎 {escape(doc.primary_engine or '-')} · 文本层覆盖 "
                f"{doc.text_layer_coverage:.0%} · 扫描页占比 {doc.scan_page_share:.0%}"
                if doc.text_layer_coverage is not None
                else ""
            ),
            title="UBT 译前报价 (assess)",
            border_style="cyan",
        )
    )

    route_table = Table(title="推荐路线", border_style="blue")
    route_table.add_column("项", style="bold cyan")
    route_table.add_column("值", style="bold green")
    route_table.add_row("链路", f"{rec.mode} — {escape(rec.reason)}")
    preset_note = (
        f"{rec.recommended_preset}（本次报价基于显式指定的预设 {preset.value}）"
        if preset is not None
        else f"{rec.recommended_preset}（本次报价基于默认配置 standard；如需按推荐预设报价请加 --preset {rec.recommended_preset}）"
    )
    route_table.add_row("预设", preset_note)
    route_table.add_row(
        "渲染引擎/版式", f"{rec.recommended_render_engine} / {rec.recommended_dual_mode}"
    )
    route_table.add_row("领域档案", rec.recommended_profile)
    route_table.add_row(
        "路由置信度", f"{rec.confidence:.2f} — {escape(rec.confidence_basis)}（非质量分数）"
    )
    console.print(route_table)

    cost_table = Table(
        title=f"预期成本 (draft={escape(cost.draft_model)}, repair={escape(cost.repair_model)})",
        border_style="magenta",
    )
    cost_table.add_column("项", style="bold cyan")
    cost_table.add_column("值", style="bold green")
    cost_table.add_row(
        "计费分块",
        f"{cost.billable_blocks}"
        + ("" if cost.billable_blocks_is_exact else "（估算，--deep 可精确）"),
    )
    cost_table.add_row(
        "token (prompt+completion)",
        f"{cost.prompt_tokens} + {cost.completion_tokens}（前缀 {cost.prefix_tokens_per_call}/次）",
    )
    cost_table.add_row(
        "draft",
        f"{_assess_money(cost.draft_cost_usd_cached)} ~ {_assess_money(cost.draft_cost_usd_uncached)} (缓存⇄未缓存)",
    )
    cost_table.add_row(f"修复 (≤{cost.repair_blocks} 块)", _assess_money(cost.repair_cost_usd))
    cost_table.add_row(
        f"QE 评审 ({cost.qe_calls} 次)",
        "不触发（QE 引擎为 heuristic，零 token 评审）"
        if not cost.qe_calls
        else _assess_money(cost.qe_cost_usd),
    )
    cost_table.add_row(
        f"视觉 (VLM {cost.vlm_page_calls} + OCR {cost.ocr_page_calls} 页)",
        "不触发"
        if not (cost.vlm_page_calls or cost.ocr_page_calls)
        else _assess_money(cost.vision_cost_usd),
    )
    cost_table.add_row(
        f"章节滚动摘要 ({cost.rollup_calls} 次)",
        "不触发" if not cost.rollup_calls else _assess_money(cost.rollup_cost_usd),
    )
    money_line = _assess_money(cost.total_cost_usd)
    if cost.money_is_unknown:
        money_line += " [yellow](模型无价格表条目，绝非 $0)[/]"
    cost_table.add_row("[bold]合计（预期非保证）[/]", f"[bold]{money_line}[/]")
    console.print(cost_table)

    console.print(
        f"[dim]预计耗时（启发式）: {report.runtime.est_seconds_low:.0f}s ~ "
        f"{report.runtime.est_seconds_high:.0f}s — {escape(report.runtime.basis)}[/]"
    )
    for signal in report.quality_signals:
        console.print(f"  · [cyan]信号[/] {escape(signal)}")
    if report.warnings:
        warn_table = Table(title="警告", border_style="yellow")
        warn_table.add_column("码", style="bold red")
        warn_table.add_column("级别")
        warn_table.add_column("说明", style="green")
        for w in report.warnings:
            warn_table.add_row(w.code, w.level, escape(w.detail_zh))
        console.print(warn_table)

    console.print(
        Panel.fit(
            f"[bold green]{report.next_step_command}[/]",
            title="确认报价后执行（可照抄）",
            border_style="green",
        )
    )
