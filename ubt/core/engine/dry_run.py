"""Zero-token rehearsal runs, shared by every surface that offers ``--dry-run``.

A dry run exercises the whole pipeline (ingest, bible, draft, QE, repair,
render) against a deterministic echo provider so a book can be validated
without spending tokens or needing an API key. Every surface assembles its
router/QE runner pair through this module, so the echo format cannot drift
between the CLI, the API and the MCP server.
"""

from __future__ import annotations

from typing import Any

from ubt.core.config import UBTConfig
from ubt.core.engine.pipeline import PipelineOrchestrator
from ubt.core.qe.fast_pass import REHEARSAL_MARKER as _DRY_RUN_PREFIX
from ubt.core.qe.heuristic import MockQERunner
from ubt.core.router.prompts import draft_source_from_prompt
from ubt.core.router.provider import MockModelProvider
from ubt.core.router.rate_limiter import AdaptiveTokenBucket
from ubt.core.router.registry import registry_from_config
from ubt.core.router.router import ModelRouter


class DryRunModelProvider(MockModelProvider):
    """Echoes the prompt's own source span behind a 模拟翻译 marker.

    The echo is deliberately bare (no ``<translation>`` wrapper): every
    built-in model profile resolves to ``ExtractionStrategy.AUTO``, which
    falls back to fence/prefix cleaning when no tag is present, so the bare
    form survives routing unchanged while a wrapped one would leak through a
    ``RAW`` profile. Staying a :class:`MockModelProvider` subclass is what the
    pipeline's ``_is_mock_run`` check keys on to namespace the ledger and skip
    TM writeback, so a rehearsal run cannot poison a real job's state.
    """

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = 0.3,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        # The chapter-level (macro) path sends several blocks in one prompt
        # wrapped in ``<block id="...">`` markup and expects the same structure
        # back. Echo each block behind the marker inside the XML envelope, or the
        # macro extractor finds nothing and every block ships empty.
        if "<blocks>" in prompt and "<block id=" in prompt:
            import re

            matches = re.findall(
                r'<\s*block\s+id=["\'](.*?)["\']\s*>([\s\S]*?)<\s*/\s*block\s*>', prompt
            )
            if matches:
                items = []
                for bid, bsrc in matches:
                    body = self._echo(str(bsrc))
                    items.append(f'<block id="{bid}">{body}</block>')
                return "<blocks>\n" + "\n".join(items) + "\n</blocks>"

        return self._echo(draft_source_from_prompt(prompt))

    @staticmethod
    def _echo(src: str) -> str:
        if src.startswith("#"):
            # Keep the markdown heading sigil attached, or the renderer
            # loses the block's level and the rehearsal stops resembling a book.
            hashes, _, rest = src.partition(" ")
            return f"{hashes} {_DRY_RUN_PREFIX} {rest}".strip()
        return f"{_DRY_RUN_PREFIX} {src}"


def create_dry_run_orchestrator(
    config: UBTConfig, **orchestrator_kwargs: Any
) -> PipelineOrchestrator:
    """Build the rehearsal orchestrator: echo provider, mock QE, no rate limit.

    ``orchestrator_kwargs`` forwards ``PipelineOrchestrator`` seams (e.g. the
    API manager's ``finalize_job`` completion hook) so a rehearsal can run
    through the same persistence path as a real job instead of the caller
    having to rebuild the echo wiring itself.

    QE is mocked at 0.92 (above every gate threshold) on purpose: a rehearsal
    should prove the *plumbing* reaches a rendered artifact, not re-litigate
    translation quality that a dry run cannot measure.

    The echo provider only covers the LLM hop. OCR and the visual judge run off
    the adapter's config, so an operator config with ``ocr_mode`` enabled and
    page upload allowed would still make real cloud calls with their key and
    upload manuscript pages during a "zero-spend" rehearsal. The rehearsal runs
    on a copy with every page-image egress path closed.
    """
    rehearsal_config = config.model_copy(
        update={
            "ocr_mode": "off",
            "allow_page_upload": False,
            "visual_judge_enabled": False,
        }
    )
    router = ModelRouter(
        provider=DryRunModelProvider(),
        rate_limiter=AdaptiveTokenBucket(initial_rpm=100_000, max_rpm=100_000),
        # The rehearsal must exercise the *configured* prompt strategy, or it
        # proves the plumbing of a run the operator never asked for: with
        # ``prompt_strategy=minimal`` a real run sends a different prompt shape
        # than the one the rehearsal rendered, and the rehearsal's "it works"
        # says nothing about the run it is standing in for. The model tiers are
        # passed for the same reason even though the echo provider ignores them
        # -- routing them here keeps the two routers' surfaces identical, so a
        # later change cannot make the rehearsal silently skip a tier.
        draft_model=rehearsal_config.draft_model,
        repair_model=rehearsal_config.repair_model,
        draft_reasoning_effort=rehearsal_config.draft_reasoning_effort,
        repair_reasoning_effort=rehearsal_config.repair_reasoning_effort,
        fallback_models=rehearsal_config.fallback_models,
        prompt_strategy_override=rehearsal_config.prompt_strategy,
        # The rehearsal closes page-image egress above; the router's master gate
        # must agree, or visual-scalpel repair would still try to upload pages.
        allow_page_upload=False,
        # Same reason as the strategy: a rehearsal that resolved capabilities
        # differently from the real run would rehearse a prompt shape and an
        # output budget the run never sends.
        registry=registry_from_config(rehearsal_config),
    )
    return PipelineOrchestrator(
        config=rehearsal_config,
        router=router,
        qe_runner=MockQERunner(default_score=0.92),
        **orchestrator_kwargs,
    )


__all__ = ["DryRunModelProvider", "create_dry_run_orchestrator"]
