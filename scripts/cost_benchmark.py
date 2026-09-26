#!/usr/bin/env python3
"""Real-LLM end-to-end cost benchmark for the UBT pipeline.

Runs the full 6-stage pipeline on a real book against any OpenAI-compatible
API (default: DeepSeek) and reports per-stage call counts, latency, token
usage, and estimated cost — plus the 0-token defense net's actual savings.

Usage:
    export DEEPSEEK_API_KEY=sk-...
    .venv/bin/python scripts/cost_benchmark.py path/to/book.md [options]

Options:
    --target-lang zh        target language (default zh)
    --profile general       genre profile (general/fiction/textbook/paper)
    --max-chapters N        md/txt only: keep the first N chapters (cost cap)
    --draft-model NAME      default: deepseek-chat
    --repair-model NAME     default: deepseek-chat (deepseek-reasoner costs more)
    --base-url URL          default: https://api.deepseek.com
    --price-input USD       USD per 1M input tokens, cache-miss (default 0.27)
    --price-cache-hit USD   USD per 1M input tokens, cache-hit (default 0.07)
    --price-output USD      USD per 1M output tokens (default 1.10)
    --job-id ID             ledger job id (default: benchmark_<timestamp>)

Prices default to DeepSeek's published deepseek-chat rates; override them for
other providers. Cost figures are estimates — always cross-check the provider
dashboard.
"""

import argparse
import asyncio
import atexit
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ubt.core.config import UBTConfig, require_api_key  # noqa: E402
from ubt.core.engine.pipeline import PipelineOrchestrator  # noqa: E402
from ubt.core.qe.comet_runner import MockQERunner  # noqa: E402
from ubt.core.router.provider import OpenAICompatibleProvider  # noqa: E402
from ubt.core.router.rate_limiter import AdaptiveTokenBucket  # noqa: E402
from ubt.core.router.router import ModelRouter  # noqa: E402

# Prompt fingerprints for classifying calls by pipeline stage
_STAGE_MARKERS = [
    ("backfill", "terminology curator"),
    ("summary", "continuation summary"),
    ("repair", "### Problematic Draft"),
    ("draft", "### Source Paragraph to Translate"),
]


def classify_call(prompt: str, system_prompt: str) -> str:
    for kind, marker in _STAGE_MARKERS:
        if marker in prompt or marker in system_prompt:
            return kind
    return "other"


class BenchmarkProvider(OpenAICompatibleProvider):
    """Provider that records per-call stage, latency, and token usage."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.calls: list[dict] = []

    async def generate_with_finish_reason(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = 0.3,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> tuple[str, str | None]:
        """Record around the method the router actually calls.

        ``ModelRouter`` only ever enters a provider through
        ``generate_with_finish_reason``; ``generate`` is a sibling, not a
        delegate, so counting there recorded nothing and the whole report —
        stage table, cost, per-call JSON — came out empty.
        """
        t0 = time.perf_counter()
        text, finish_reason = await super().generate_with_finish_reason(
            prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
        entry = {
            "kind": classify_call(prompt, system_prompt or ""),
            "latency_s": round(time.perf_counter() - t0, 3),
            "prompt_chars": len(prompt),
        }
        if self.usage_log:
            entry.update(self.usage_log[-1])
        self.calls.append(entry)
        return text, finish_reason


def truncate_markdown_chapters(path: Path, max_chapters: int, out: Path) -> Path:
    """Keep only the first `max_chapters` '# ' chapters of a md/txt file."""
    lines = path.read_text(encoding="utf-8", errors="replace").split("\n")
    chapter_starts = [i for i, ln in enumerate(lines) if ln.strip().startswith("# ")]
    if len(chapter_starts) <= max_chapters:
        return path
    cut = chapter_starts[max_chapters]
    out.write_text("\n".join(lines[:cut]), encoding="utf-8")
    return out


def estimate_cost(calls: list[dict], p_in: float, p_hit: float, p_out: float) -> float:
    total = 0.0
    for c in calls:
        pt = float(c.get("prompt_tokens", 0) or 0)
        hit = float(c.get("prompt_cache_hit_tokens", 0) or 0)
        miss = max(pt - hit, 0.0)
        ct = float(c.get("completion_tokens", 0) or 0)
        total += (miss * p_in + hit * p_hit + ct * p_out) / 1_000_000
    return total


def _default_rate_covers(model: str) -> bool:
    """True when the built-in rate table (DeepSeek's) applies to ``model``.

    The built-in defaults ARE DeepSeek's published rates, so they are only a
    valid stand-in for a DeepSeek model; a gpt/gemini run that supplies no
    ``--price-*`` has no knowable rate and must write null, not DeepSeek numbers.
    """
    return (model or "").strip().lower().startswith("deepseek")


def _resolve_prices(args: argparse.Namespace) -> tuple[float, float, float, bool]:
    """``(input, cache_hit, output, priced)`` from args.

    The operator-supplied rates win; otherwise the built-in DeepSeek rates apply
    but the run is "priced" only when the model is actually a DeepSeek one. That
    keeps the default DeepSeek benchmark priced (so the committed artifact
    carries the reproducible bill) while a model the built-in table cannot price
    writes ``null`` rather than a wrong DeepSeek-rate number.
    """
    price_input = args.price_input if args.price_input is not None else 0.27
    price_cache_hit = args.price_cache_hit if args.price_cache_hit is not None else 0.07
    price_output = args.price_output if args.price_output is not None else 1.10
    explicit = any(
        value is not None
        for value in (args.price_input, args.price_cache_hit, args.price_output)
    )
    priced = explicit or (
        _default_rate_covers(args.draft_model) and _default_rate_covers(args.repair_model)
    )
    return price_input, price_cache_hit, price_output, priced


async def run(args: argparse.Namespace) -> None:
    api_key = require_api_key()

    price_input, price_cache_hit, price_output, priced = _resolve_prices(args)

    src = Path(args.input)
    if not src.exists():
        sys.exit(f"error: input not found: {src}")

    tmp_truncated: Path | None = None
    if args.max_chapters and src.suffix.lower() in (".md", ".txt"):
        # Unique path (concurrent runs must not share one) cleaned up at exit
        # (the script has no single try/finally around the whole run).
        fd, tmp_name = tempfile.mkstemp(
            prefix=f"ubt_benchmark_{src.stem}_c{args.max_chapters}_", suffix=src.suffix
        )
        os.close(fd)
        tmp_truncated = Path(tmp_name)
        atexit.register(tmp_truncated.unlink, missing_ok=True)
        src = truncate_markdown_chapters(src, args.max_chapters, tmp_truncated)
        print(f"[benchmark] truncated to first {args.max_chapters} chapters -> {src}")

    config = UBTConfig.from_env(
        api_key=api_key,
        base_url=args.base_url,
        draft_model=args.draft_model,
        repair_model=args.repair_model,
        db_dir=Path("/tmp/ubt_benchmark_ledgers"),
        rate_limit_rpm=3000,
        # A page window is the only cost cap that works for PDF input
        # (--max-chapters is md/txt only), so a first run can stay in pennies.
        pages=args.pages,
    )
    provider = BenchmarkProvider(
        api_key=api_key,
        base_url=args.base_url,
        default_model=args.draft_model,
        timeout=120.0,
        provider_name="opencode_zen"
        if "opencode.ai" in args.base_url
        else ("deepseek" if "deepseek" in args.base_url else "openai_compatible"),
        api_mode=args.api_mode,
        # opencode /go/ rejects every request without x-opencode-session, and
        # this script builds its own provider instead of going through the
        # pipeline's router, so the session id has to be forwarded by hand.
        opencode_session_id=config.opencode_session_id,
        # Forward the same wire knobs the pipeline's provider gets, or the
        # benchmark measures a different configuration than production (custom
        # gateway headers, vLLM chat-template overrides, prompt caching).
        extra_headers=config.extra_headers,
        chat_template_kwargs=config.chat_template_kwargs,
        prompt_caching=config.prompt_caching_enabled,
    )
    router = ModelRouter(
        provider=provider,
        draft_model=args.draft_model,
        repair_model=args.repair_model,
        rate_limiter=AdaptiveTokenBucket(initial_rpm=3000, max_rpm=3000),
    )
    # Mock QE keeps the benchmark focused on LLM call cost; swap in a real
    # COMET runner here when measuring the full quality-gate latency too.
    orchestrator = PipelineOrchestrator(
        config=config, router=router, qe_runner=MockQERunner(default_score=0.9)
    )

    job_id = args.job_id or f"benchmark_{int(time.time())}"
    out_path = Path(f"/tmp/ubt_benchmark_out_{job_id}{src.suffix}")
    print(
        f"[benchmark] job={job_id} model(draft)={args.draft_model} model(repair)={args.repair_model}"
    )
    t0 = time.perf_counter()
    events = 0
    async for _ in orchestrator.run(
        input_path=src,
        output_path=out_path,
        target_lang=args.target_lang,
        profile_name=args.profile,
        job_id=job_id,
    ):
        events += 1
    wall = time.perf_counter() - t0

    calls = provider.calls
    by_kind: dict[str, list[dict]] = {}
    for c in calls:
        by_kind.setdefault(c["kind"], []).append(c)

    print("\n=== UBT cost benchmark ===")
    print(f"input: {src}  target_lang={args.target_lang} profile={args.profile}")
    print(f"wall time: {wall:.1f}s   pipeline events: {events}")

    print(f"\n{'stage':<10} {'calls':>6} {'latency avg':>12} {'prompt tok':>11} {'compl tok':>10}")
    order = ["draft", "repair", "backfill", "summary", "other"]
    for kind in order + sorted(set(by_kind) - set(order)):
        cs = by_kind.get(kind, [])
        if not cs:
            continue
        avg_lat = sum(c["latency_s"] for c in cs) / len(cs)
        pt = sum(c.get("prompt_tokens", 0) for c in cs)
        ct = sum(c.get("completion_tokens", 0) for c in cs)
        print(f"{kind:<10} {len(cs):>6} {avg_lat:>10.2f}s {pt:>11} {ct:>10}")

    totals = provider.usage_totals
    cost = estimate_cost(calls, price_input, price_cache_hit, price_output)
    cache_hit = sum(c.get("prompt_cache_hit_tokens", 0) or 0 for c in calls)
    print(
        f"\ntotals: {totals['calls']} calls, {totals['prompt_tokens']} prompt tok "
        f"(cache-hit {cache_hit}), {totals['completion_tokens']} completion tok"
    )
    if priced:
        print(
            f"estimated cost: ${cost:.4f}  (in ${price_input}/M, "
            f"hit ${price_cache_hit}/M, out ${price_output}/M)"
        )
    else:
        print(
            f"estimated cost: unknown (the built-in DeepSeek rates do not cover "
            f"model {args.draft_model!r}; pass --price-* to price the run)"
        )

    report_path = out_path.with_name(f"{out_path.stem}_quality_report.json")
    if report_path.exists():
        rep = json.loads(report_path.read_text(encoding="utf-8"))
        s = rep.get("summary", {})
        print(
            f"\npipeline: blocks={s.get('total_blocks')} completed={s.get('completed_blocks')} "
            f"failed={s.get('failed_blocks')} pass_rate={s.get('pass_rate')} avg_qe={rep.get('score_metrics', {}).get('avg_qe')}"
        )
    print(f"output: {out_path}")

    detail_path = out_path.with_name(f"{out_path.stem}_benchmark_calls.json")
    detail_path.write_text(json.dumps(calls, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"per-call detail: {detail_path}")

    if args.metrics_dir:
        _write_committed_metrics(
            args, job_id, out_path, wall, events, totals, cache_hit, cost, by_kind
        )


def _write_committed_metrics(
    args: argparse.Namespace,
    job_id: str,
    out_path: Path,
    wall: float,
    events: int,
    totals: dict,
    cache_hit: int,
    cost: float,
    by_kind: dict,
) -> None:
    """Write the text-free cost record where it can be committed.

    Everything else this script emits lands under ``/tmp`` and dies there, which
    is why the repository holds zero reproducible evidence of what a real book
    costs: no ``--budget-usd`` value could be calibrated, and a wrong price-table
    entry had no invoice to be checked against. Only counters go in here — the
    per-call detail above carries the prompts, i.e. the book itself, and stays
    out of the repository on purpose.
    """
    from datetime import UTC, datetime

    from ubt.core.ir.serializer import compute_file_sha256

    pipeline: dict = {}
    quality = out_path.with_name(f"{out_path.stem}_quality_report.json")
    if quality.exists():
        summary = json.loads(quality.read_text(encoding="utf-8")).get("summary", {})
        pipeline = {
            key: summary.get(key)
            for key in ("total_blocks", "completed_blocks", "failed_blocks", "pass_rate")
        }

    metrics_dir = Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out = metrics_dir / f"{job_id}.json"
    # A run against a model this tool has no rate for must write null, not 0.0:
    # the pipeline's own cost KPI already collapses to a fake zero for exactly
    # this case, and a committed benchmark must not repeat it.
    price_input, price_cache_hit, price_output, priced = _resolve_prices(args)
    out.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "job_id": job_id,
                "corpus": {
                    "name": Path(args.input).name,
                    "sha256": compute_file_sha256(Path(args.input)),
                    "max_chapters_cap": args.max_chapters or None,
                    "pages_window": args.pages or None,
                },
                "target_lang": args.target_lang,
                "profile": args.profile,
                "models": {"draft": args.draft_model, "repair": args.repair_model},
                "prices_usd_per_mtok": (
                    {
                        "input": price_input,
                        "cache_hit": price_cache_hit,
                        "output": price_output,
                    }
                    if priced
                    else None
                ),
                "wall_seconds": round(wall, 2),
                "pipeline_events": events,
                "usage_totals": totals,
                "cache_hit_prompt_tokens": cache_hit,
                "estimated_cost_usd": round(cost, 4) if priced else None,
                "cost_note": (
                    None
                    if priced
                    else "no rate supplied for this model: the token counts are the "
                    "evidence, and null is not zero"
                ),
                "by_stage": {
                    kind: {
                        "calls": len(cs),
                        "avg_latency_s": round(sum(c["latency_s"] for c in cs) / len(cs), 2),
                        "prompt_tokens": sum(c.get("prompt_tokens", 0) for c in cs),
                        "completion_tokens": sum(c.get("completion_tokens", 0) for c in cs),
                    }
                    for kind, cs in sorted(by_kind.items())
                },
                "pipeline_summary": pipeline,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"committed metrics: {out}")


def build_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("input")
    ap.add_argument("--target-lang", default="zh")
    ap.add_argument("--profile", default="general")
    ap.add_argument("--max-chapters", type=int, default=0)
    ap.add_argument(
        "--pages",
        default=None,
        help="page window for the cost cap on PDF input, e.g. '1-6' or '3,5,7-9' "
        "(--max-chapters only truncates md/txt)",
    )
    ap.add_argument("--draft-model", default="deepseek-chat")
    ap.add_argument("--repair-model", default="deepseek-chat")
    ap.add_argument("--base-url", default="https://api.deepseek.com")
    ap.add_argument("--api-mode", default="chat", choices=["chat", "responses"])
    ap.add_argument("--price-input", type=float, default=None)
    ap.add_argument("--price-cache-hit", type=float, default=None)
    ap.add_argument("--price-output", type=float, default=None)
    ap.add_argument("--job-id", default=None)
    ap.add_argument(
        "--metrics-dir",
        default="docs/benchmarks",
        help="where to write the text-free cost record so it can be committed "
        "(default: docs/benchmarks; pass '' to skip). Prompts and translations "
        "never land here.",
    )
    return ap.parse_args(argv)


def main() -> None:
    asyncio.run(run(build_args(sys.argv[1:])))


if __name__ == "__main__":
    main()
