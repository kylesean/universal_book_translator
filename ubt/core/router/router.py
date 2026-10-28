"""Intelligent ModelRouter governing model tier dispatch, retries and rate limiting.

Prompt text lives in :mod:`ubt.core.router.prompts`; this file resolves a model
name to a capability profile and executes against the provider.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from collections.abc import Callable
from contextlib import suppress
from contextvars import ContextVar
from typing import TYPE_CHECKING, Literal, overload

if TYPE_CHECKING:
    pass

from ubt.core.exceptions import (
    BudgetExceededError,
    JobInterruptedError,
    LedgerWriterLeaseLostError,
    ModelProviderError,
)
from ubt.core.ir.models import IRBlock
from ubt.core.router.capabilities import (
    ModelProfile,
    PromptStrategy,
)
from ubt.core.router.extractor import TranslationOutputExtractor
from ubt.core.router.pricing import estimate_cost_usd
from ubt.core.router.prompts import (
    build_hybrid_draft_prompt,
    build_hybrid_repair_prompt,
    build_macro_chunk_draft_prompt,
    build_minimal_draft_prompt,
    build_minimal_repair_prompt,
    build_rich_draft_prompt,
    build_rich_repair_prompt,
)
from ubt.core.router.provider import BaseModelProvider, OpenAICompatibleProvider
from ubt.core.router.rate_limiter import AdaptiveTokenBucket, NullRateLimiter
from ubt.core.router.registry import ModelCapabilityRegistry, fallback_profile
from ubt.core.router.router_errors import (  # noqa: F401 -- re-exported
    _CONTINUATION_TAIL_CHARS,
    _FAIL_FAST_STATUS,
    _LOCAL_FALLBACK_BASE_URL_DEFAULT,
    _LOCAL_FALLBACK_BASE_URL_ENV,
    _MAX_CONTINUATION_ROUNDS,
    _MIN_MERGE_OVERLAP,
    _MODEL_IDENTITY_PHRASES,
    _MODEL_REJECTION_STATUSES,
    _QUOTA_EXHAUSTED_PHRASES,
    _RETRY_AFTER_MAX_SECONDS,
    _RETRYABLE_STATUS,
    _SELF_HOSTED_BACKENDS,
    _VISION_TOKENS_PER_PAGE_IMAGE,
    ProviderErrorAction,
    _estimate_prompt_tokens,
    _is_chain_fail_fast,
    _is_rate_limit_error,
    _merge_continuation,
    _parse_retry_after,
    classify_provider_error,
)

logger = logging.getLogger(__name__)

#: Floor for the one-shot budget widening when a reasoning model returns no
#: visible content because hidden reasoning consumed ``max_tokens``.
_REASONING_MIN_OUTPUT_TOKENS = 8192

#: Reasoning tokens are not proportional to the visible translation alone: the
#: model may spend several times the source length before writing anything. A
#: macro-chunk of eight blocks therefore needs more headroom than one block.
_REASONING_OUTPUT_HEADROOM = 8

#: Protocols whose transport already translates reasoning effort into a token
#: budget and raises ``max_tokens`` above it. The router must not pre-inflate
#: those: Anthropic would then send ``budget + 1024`` over an already-scaled
#: value, multiplying a routine repair call's request budget.
_TRANSPORT_OWNS_REASONING_BUDGET = frozenset({"anthropic-messages"})

#: Cooperative-cancel token for the CURRENT run's provider retry backoff.
#: A ContextVar, not a router field, because one process can drive several
#: jobs over ONE shared router (JobWorker with concurrency>1, the API
#: server): a field is per-instance state, and a second job overwrote the
#: first job's token — cancelling job B aborted job A's backoff (and a
#: cancel for A could not interrupt A's own sleep). Context vars are
#: inherited per asyncio task, so each run's context carries its own token.
_cancel_token_ctx: ContextVar[asyncio.Event | None] = ContextVar("ubt_cancel_token", default=None)
_lease_lost_ctx: ContextVar[asyncio.Event | None] = ContextVar("ubt_lease_lost", default=None)


def set_run_cancel_token(
    token: asyncio.Event | None,
    lease_lost: asyncio.Event | None = None,
) -> None:
    """Bind the current run's cooperative-cancel and lease-lost tokens for provider backoff."""
    _cancel_token_ctx.set(token)
    _lease_lost_ctx.set(lease_lost)


def _raise_if_cancelled(fallback: asyncio.Event | None = None) -> None:
    """Raise when this run's lease is lost or cancel token is set.

    Raises :class:`LedgerWriterLeaseLostError` if the writer lease was lost,
    or :class:`JobInterruptedError` when this run's cancel token is set.
    The ContextVars are the per-task tokens; ``fallback`` is the router's
    constructor-time field, for library callers that never bind one.
    """
    lease_lost = _lease_lost_ctx.get()
    if lease_lost is not None and lease_lost.is_set():
        raise LedgerWriterLeaseLostError("Writer lease was reclaimed by another process")
    token = _cancel_token_ctx.get() or fallback
    if token is not None and token.is_set():
        raise JobInterruptedError("Job cancelled before the next provider call")


class ModelRouter:
    """Routes translation and repair tasks across LLM tiers with capability strategy dispatch."""

    def __init__(
        self,
        provider: BaseModelProvider,
        draft_model: str = "",
        repair_model: str = "",
        rate_limiter: AdaptiveTokenBucket | None = None,
        max_retries: int = 3,
        draft_reasoning_effort: str = "low",
        repair_reasoning_effort: str = "high",
        registry: ModelCapabilityRegistry | None = None,
        fallback_models: list[str] | None = None,
        prompt_strategy_override: str | None = None,
        allow_page_upload: bool = False,
        # AIMD oscillation guard for the limiter this router builds when the
        # caller does not inject one. Mirrors the UBTConfig default (3.0); a
        # bare ``AdaptiveTokenBucket()`` defaulted it to 0.0, so a router built
        # outside the pipeline (library/MCP use) lost the guard silently.
        backoff_cooldown_sec: float = 3.0,
        repair_provider: BaseModelProvider | None = None,
        cancel_token: asyncio.Event | None = None,
        request_timeout: float = 120.0,
    ) -> None:
        self.provider = provider
        self.repair_provider = repair_provider or provider
        self.request_timeout = request_timeout
        # Master gate for shipping rendered PAGE IMAGES of the book to model
        # endpoints (visual-scalpel repair, VLM judge passthrough). Text
        # prompts stay unaffected. Off unless the caller (the pipeline, from
        # UBTConfig.allow_page_upload) opts in.
        self.allow_page_upload = allow_page_upload
        self.draft_model = draft_model
        self.repair_model = repair_model
        # Forced prompt strategy for every model ("minimal"/"hybrid"/"rich";
        # None/"auto" keeps the registry decision). Only prompt assembly is
        # overridden — extraction and output-format profiles stay per-model.
        normalized = (prompt_strategy_override or "").strip().lower()
        self.prompt_strategy_override: PromptStrategy | None = None
        if normalized and normalized != "auto":
            try:
                self.prompt_strategy_override = PromptStrategy(normalized)
            except ValueError:
                logger.warning(
                    "Ignoring unknown prompt_strategy_override=%r (expected minimal/hybrid/rich)",
                    prompt_strategy_override,
                )
        # Ordered model-level fallback chain. Duplicates of the
        # primary model are dropped when the chain is walked.
        self.fallback_models = list(fallback_models or [])
        # Local self-hosted retry provider, built lazily: models whose registry
        # profile declares a self-hosted deployment_backend (Ollama, llama.cpp,
        # llama-swap, vLLM, LM Studio) run against a remote gateway by default,
        # and gateways can drop them with 401/404 "model not supported". One
        # retry against the local stack then succeeds where the gateway refused.
        self._fallback_provider: OpenAICompatibleProvider | None = None
        # An explicit limiter always wins — the retry/AIMD tests hand one in.
        # Otherwise mock traffic gets the limiter that never gates: a real
        # bucket protects a credential mock calls cannot exhaust.
        if rate_limiter is not None:
            self.rate_limiter = rate_limiter
        elif provider.is_mock:
            self.rate_limiter = NullRateLimiter()
        else:
            self.rate_limiter = AdaptiveTokenBucket(backoff_cooldown_sec=backoff_cooldown_sec)
        self.max_retries = max_retries
        # Cooperative-cancel token (the job worker's). A long Retry-After sleep
        # must not outlive a cancellation: the router aborts the wait and raises
        # JobInterruptedError, the same stop the stages raise, instead of
        # pinning the worker to a 24h sleep it will never finish.
        # NOTE: this field is the CONSTRUCTION default only. A shared router
        # (JobWorker with concurrency>1) must not carry a per-job token here —
        # two concurrent jobs would overwrite each other's token and a cancel
        # for one would abort the other's backoff. The per-run value lives in
        # the cancel_token ContextVar below; the pipeline sets it in run().
        self.cancel_token = cancel_token
        self.draft_reasoning_effort = draft_reasoning_effort
        self.repair_reasoning_effort = repair_reasoning_effort
        self.registry = registry or ModelCapabilityRegistry()
        # In-memory circuit breaker for fallback candidates: model -> (failure_count, cooldown_until_monotonic)
        self._model_circuit: dict[str, tuple[int, float]] = {}

    def _provider_for(
        self, model: str | None = None, *, is_repair: bool = False
    ) -> BaseModelProvider:
        """Select primary provider or specialized repair provider based on target model or repair context."""
        if (
            is_repair or (model is not None and model == self.repair_model)
        ) and self.repair_provider is not None:
            return self.repair_provider
        return self.provider

    def _reasoning_output_tokens(
        self,
        source_tokens: int,
        *,
        provider: BaseModelProvider | None = None,
    ) -> int:
        """Output budget for a reasoning call, sized for hidden + visible tokens.

        Chat-completions gateways (the common proxy case) apply their own tiny
        default when ``max_tokens`` is omitted, so the router must send a value.
        Gemini counts ``thinkingBudget`` inside ``maxOutputTokens`` and needs the
        same headroom. Anthropic's transport raises ``max_tokens`` above its
        ``budget_tokens`` itself, so the router sends only the ordinary cap and
        lets that single owner do the addition once.
        """
        active = provider or self.provider
        api_mode = str(getattr(active, "_api_mode", "") or "")
        if api_mode in _TRANSPORT_OWNS_REASONING_BUDGET:
            return max(4096, int(source_tokens * 4.0))
        return max(_REASONING_MIN_OUTPUT_TOKENS, source_tokens * _REASONING_OUTPUT_HEADROOM)

    # ------------------------------------------------------------------
    # Real usage / cost accounting
    # ------------------------------------------------------------------
    def _usage_totals_of(self, provider: object) -> dict[str, int]:
        totals = getattr(provider, "usage_totals", None)
        if callable(totals):
            totals = totals()
        return dict(totals) if isinstance(totals, dict) else {}

    def _usage_by_model_of(self, provider: object) -> dict[str, dict[str, int]]:
        by_model = getattr(provider, "usage_totals_by_model", None)
        if callable(by_model):
            by_model = by_model()
        if not isinstance(by_model, dict):
            return {}
        return {str(model): dict(totals) for model, totals in by_model.items()}

    def usage_totals(self) -> dict[str, int]:
        """Aggregate token usage across all provider calls (empty when unsupported).

        Includes the lazily built self-hosted fallback provider: its calls are
        real spend, so a router-level read must not under-report them.
        """
        totals = self._usage_totals_of(self.provider)
        if self.repair_provider is not None and self.repair_provider is not self.provider:
            for key, value in self._usage_totals_of(self.repair_provider).items():
                totals[key] = totals.get(key, 0) + value
        if self._fallback_provider is not None:
            for key, value in self._usage_totals_of(self._fallback_provider).items():
                totals[key] = totals.get(key, 0) + value
        return totals

    def usage_totals_by_model(self) -> dict[str, dict[str, int]]:
        """Per-model token usage (empty when the provider does not track it)."""
        merged = self._usage_by_model_of(self.provider)
        if self.repair_provider is not None and self.repair_provider is not self.provider:
            for model, totals in self._usage_by_model_of(self.repair_provider).items():
                bucket = merged.setdefault(model, {})
                for key, value in totals.items():
                    bucket[key] = bucket.get(key, 0) + value
        if self._fallback_provider is not None:
            for model, totals in self._usage_by_model_of(self._fallback_provider).items():
                bucket = merged.setdefault(model, {})
                for key, value in totals.items():
                    bucket[key] = bucket.get(key, 0) + value
        return merged

    def begin_usage_sink(self) -> dict[str, dict[str, int]] | None:
        """Attach a per-run usage bucket, or None when the provider can't attribute.

        A shared provider (API server: one provider, N concurrent jobs) reports
        only process-wide totals, which would bill every job the same blended
        spend; a sink gives the calling run its own numbers.
        """
        begin = getattr(self.provider, "begin_usage_sink", None)
        if not callable(begin):
            return None
        sink = begin()
        return sink if isinstance(sink, dict) else None

    def cache_hit_rate(self) -> float:
        """Fraction of prompt tokens served from the provider cache.

        Verifies the README's static-prefix TCO lever. Returns 0.0 when the
        provider does not measure cache hits or reports no usage yet.
        """
        rate = getattr(self.provider, "cache_hit_rate", None)
        return float(rate) if isinstance(rate, (int, float)) else 0.0

    def estimate_cost_usd(self) -> float | None:
        """Cumulative USD cost derived from real token usage and the price table.

        Returns 0.0 when the provider reports no usage, 0.0 for a self-hosted
        endpoint (there is no provider bill to estimate), and None when a model
        that consumed tokens has no knowable price (unknown cost — never a
        fabricated free).

        The self-hosted fallback is a *different* endpoint and may be $0 by
        construction; pricing its models at the primary's ``base_url`` would
        either over-bill them (cloud primary) or raise the whole run to
        "unknown" for a model the table cannot name. Models the fallback served
        *exclusively* are therefore billed against the fallback's endpoint.
        """
        primary_url = getattr(self.provider, "base_url", "") or None
        return estimate_cost_usd(
            self.usage_totals_by_model(),
            base_url=primary_url,
            endpoint_map=self.billing_endpoint_map() or None,
        )

    def billing_endpoint_map(self) -> dict[str, str]:
        """Models billed through an endpoint other than the primary ``base_url``.

        Only models the self-hosted fallback served *exclusively* are attributed
        to the fallback URL (a name served by both channels can't be split, so it
        keeps the primary endpoint). The budget ledger consumes this so a
        fallback-served model is not priced at the primary endpoint.
        """
        endpoint_map: dict[str, str] = {}
        if self.repair_provider is not None and self.repair_provider is not self.provider:
            repair_url = getattr(self.repair_provider, "base_url", "") or None
            if repair_url:
                endpoint_map[self.repair_model] = repair_url
        if self._fallback_provider is not None:
            fallback_url = getattr(self._fallback_provider, "base_url", "") or None
            if fallback_url:
                primary_models = self._usage_by_model_of(self.provider)
                for model in self._usage_by_model_of(self._fallback_provider):
                    if model not in primary_models:
                        endpoint_map[model] = fallback_url
        return endpoint_map

    async def aclose(self) -> None:
        """Release the underlying provider's HTTP connection pool."""
        closer = getattr(self.provider, "aclose", None)
        if callable(closer):
            with suppress(Exception):
                await closer()
        if self.repair_provider is not None and self.repair_provider is not self.provider:
            repair_closer = getattr(self.repair_provider, "aclose", None)
            if callable(repair_closer):
                with suppress(Exception):
                    await repair_closer()
        if self._fallback_provider is not None:
            fallback_closer = getattr(self._fallback_provider, "aclose", None)
            if callable(fallback_closer):
                with suppress(Exception):
                    await fallback_closer()
            self._fallback_provider = None
        if self.rate_limiter is not None:
            rl_closer = getattr(self.rate_limiter, "close", None)
            if callable(rl_closer):
                with suppress(Exception):
                    rl_closer()

    def _get_profile(self, model_name: str | None = None) -> ModelProfile:
        """The capability profile for ``model_name``: declared, else the fallback.

        The router never guesses from the model name. A model whose provider block
        declares nothing resolves to
        :func:`~ubt.core.router.registry.fallback_profile`, which asserts only what
        is safe for an unknown OpenAI-compatible endpoint. That is why the
        ``registry is None`` branch uses the *same* profile rather than a second,
        subtly different set of defaults: one definition, both paths.
        """
        target = model_name or self.draft_model
        profile = (
            self.registry.resolve(target) if self.registry is not None else fallback_profile(target)
        )
        if self.prompt_strategy_override is not None:
            profile = profile.model_copy(update={"prompt_strategy": self.prompt_strategy_override})
        return profile

    def build_draft_prompt(
        self,
        source_text: str,
        glossary_table: str = "",
        neighbor_context: str = "",
        target_lang: str = "zh",
        source_lang: str = "en",
        genre_profile: str = "general",
        rolling_summary: str = "",
        global_glossary: str = "",
        few_shot_reference: str = "",
        epoch_summary: str = "",
        model: str | None = None,
        domain: str | None = None,
        block_type: str | None = None,
    ) -> tuple[str, str]:
        """Construct draft prompt using strategy configured on the model capability profile."""
        profile = self._get_profile(model or self.draft_model)
        if profile.prompt_strategy == PromptStrategy.MINIMAL:
            return build_minimal_draft_prompt(
                source_text=source_text,
                glossary_table=glossary_table,
                target_lang=target_lang,
                source_lang=source_lang,
                global_glossary=global_glossary,
                few_shot_reference=few_shot_reference,
                genre_profile=genre_profile,
                domain=domain,
                block_type=block_type,
            )
        if profile.prompt_strategy == PromptStrategy.HYBRID:
            return build_hybrid_draft_prompt(
                source_text=source_text,
                glossary_table=glossary_table,
                neighbor_context=neighbor_context,
                target_lang=target_lang,
                source_lang=source_lang,
                genre_profile=genre_profile,
                rolling_summary=rolling_summary,
                global_glossary=global_glossary,
                few_shot_reference=few_shot_reference,
                epoch_summary=epoch_summary,
                domain=domain,
                block_type=block_type,
            )
        return build_rich_draft_prompt(
            source_text=source_text,
            glossary_table=glossary_table,
            neighbor_context=neighbor_context,
            target_lang=target_lang,
            source_lang=source_lang,
            genre_profile=genre_profile,
            rolling_summary=rolling_summary,
            global_glossary=global_glossary,
            few_shot_reference=few_shot_reference,
            epoch_summary=epoch_summary,
            domain=domain,
            block_type=block_type,
        )

    def build_repair_prompt(
        self,
        source_text: str,
        draft_text: str,
        error_flags: list[str],
        glossary_table: str = "",
        target_lang: str = "zh",
        source_lang: str = "en",
        annotated_draft: str = "",
        has_error_spans: bool = False,
        model: str | None = None,
        global_glossary: str = "",
    ) -> tuple[str, str]:
        """Construct repair prompt using strategy configured on the model capability profile."""
        profile = self._get_profile(model or self.repair_model)
        if profile.prompt_strategy == PromptStrategy.MINIMAL:
            return build_minimal_repair_prompt(
                source_text=source_text,
                draft_text=draft_text,
                error_flags=error_flags,
                glossary_table=glossary_table,
                target_lang=target_lang,
                source_lang=source_lang,
                global_glossary=global_glossary,
            )
        if profile.prompt_strategy == PromptStrategy.HYBRID:
            return build_hybrid_repair_prompt(
                source_text=source_text,
                draft_text=draft_text,
                error_flags=error_flags,
                glossary_table=glossary_table,
                target_lang=target_lang,
                source_lang=source_lang,
                annotated_draft=annotated_draft,
                has_error_spans=has_error_spans,
                global_glossary=global_glossary,
            )
        return build_rich_repair_prompt(
            source_text=source_text,
            draft_text=draft_text,
            error_flags=error_flags,
            glossary_table=glossary_table,
            target_lang=target_lang,
            source_lang=source_lang,
            annotated_draft=annotated_draft,
            has_error_spans=has_error_spans,
            global_glossary=global_glossary,
        )

    def _local_fallback_base_url(self) -> str:
        """The local endpoint to retry against (env-overridable, Ollama default)."""
        return os.environ.get(_LOCAL_FALLBACK_BASE_URL_ENV, "").strip() or (
            _LOCAL_FALLBACK_BASE_URL_DEFAULT
        )

    def _local_fallback_provider(self) -> OpenAICompatibleProvider:
        """Lazily build (and cache) the local self-hosted provider."""
        if self._fallback_provider is None:
            # Any non-empty key: self-hosted servers usually ignore auth, and the
            # OpenAI-compatible clients require a non-empty one.
            self._fallback_provider = OpenAICompatibleProvider(
                api_key="local",
                base_url=self._local_fallback_base_url(),
                default_model=self.draft_model,
                provider_name="self_hosted_local",
            )
        return self._fallback_provider

    def _is_self_hosted_rejection(self, exc: ModelProviderError, model: str) -> bool:
        """True when a gateway rejection deserves one local self-hosted retry."""
        backend = (self._get_profile(model).deployment_backend or "").strip().lower()
        if backend not in _SELF_HOSTED_BACKENDS:
            return False
        status = (exc.details or {}).get("status_code")
        if status not in _MODEL_REJECTION_STATUSES:
            return False
        if not any(phrase in str(exc).lower() for phrase in _MODEL_IDENTITY_PHRASES):
            return False
        primary_base = getattr(self.provider, "_base_url", "") or ""
        from ubt.core.router.transports.base import hostname_of

        # Host compare, not substring: ``"localhost" in base_url`` also matched
        # e.g. ``my-localhost-proxy.example.com``.
        return hostname_of(primary_base) not in ("localhost", "127.0.0.1", "::1", "0.0.0.0")

    async def _generate_via_local_fallback(
        self,
        exc: ModelProviderError,
        model: str,
        user_prompt: str,
        system_prompt: str | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> tuple[str, str | None]:
        """Retry one gateway-rejected self-hosted-model call against localhost.

        Raises the original error when the model is not declared self-hosted,
        the primary already is localhost, or the rejection shape does not
        match; raises a chained error when localhost itself fails.
        """
        if not self._is_self_hosted_rejection(exc, model):
            raise exc
        logger.warning(
            "Model %s rejected by gateway (%s); retrying once via the local "
            "self-hosted endpoint %s",
            model,
            exc,
            self._local_fallback_base_url(),
        )
        try:
            return await self._local_fallback_provider().generate_with_finish_reason(
                prompt=user_prompt,
                system_prompt=system_prompt,
                model=model,
                temperature=temperature if temperature is not None else 0.3,
                max_tokens=max_tokens,
            )
        except Exception as local_exc:
            # Forward the local failure's structured details so
            # ``classify_provider_error`` sees the real status (e.g. a terminal
            # 404) instead of an unknown error it would keep retrying. Without
            # this a terminal local rejection re-ran the gateway + local call
            # max_retries+1 times and surfaced only the local message.
            local_details = getattr(local_exc, "details", None) or {}
            raise ModelProviderError(
                f"Local self-hosted fallback failed for {model}: {local_exc}",
                details=dict(local_details),
            ) from exc

    @overload
    async def _execute_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float,
        reasoning_effort: str | None = ...,
        max_tokens: int | None = ...,
        *,
        prompt_builder: Callable[[str], tuple[str, str]] | None = ...,
        max_tokens_fn: Callable[[str, ModelProfile], int | None] | None = ...,
        return_model: Literal[False] = ...,
        is_repair: bool = ...,
    ) -> str: ...

    @overload
    async def _execute_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float,
        reasoning_effort: str | None = ...,
        max_tokens: int | None = ...,
        *,
        prompt_builder: Callable[[str], tuple[str, str]] | None = ...,
        max_tokens_fn: Callable[[str, ModelProfile], int | None] | None = ...,
        return_model: Literal[True],
        is_repair: bool = ...,
    ) -> tuple[str, str]: ...

    async def _execute_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float,
        reasoning_effort: str | None = None,
        max_tokens: int | None = None,
        *,
        prompt_builder: Callable[[str], tuple[str, str]] | None = None,
        max_tokens_fn: Callable[[str, ModelProfile], int | None] | None = None,
        return_model: bool = False,
        is_repair: bool = False,
    ) -> str | tuple[str, str]:
        """Execute with per-model retry, then model-level fallback.

        When the primary model fails — fail-fast client errors like 401/402
        (model not deployed / account issue) or retries exhausted under 429/
        5xx — the request is retried on the next entry of ``fallback_models``
        instead of failing every block in the book. Capability profiles are
        re-resolved per candidate, and prompt builders re-adapt per candidate
        automatically.
        """
        # Order-preserving dedup: a repeated fallback entry (e.g. the primary
        # also listed in fallback_models) would retry the same failing model an
        # extra time per block with no chance of success.
        chain = list(dict.fromkeys([model, *self.fallback_models]))
        now = time.monotonic()
        # Circuit breaker: prune expired entries when table exceeds bound
        if len(self._model_circuit) > 100:
            self._model_circuit = {
                m: (cnt, exp) for m, (cnt, exp) in self._model_circuit.items() if exp > now
            }
        # Circuit breaker: prioritize candidates whose circuit is healthy
        # (cooldown expired). When *every* candidate is cooling the full chain is
        # retried deliberately rather than failing the block: the cooldown is a
        # 60s probe window, so refusing outright would sink a whole book during a
        # transient provider-wide outage. The breaker still spares the healthy
        # path from probing a known-bad model.
        healthy = [c for c in chain if self._model_circuit.get(c, (0, 0.0))[1] <= now]
        effective_chain = healthy or chain

        # Chain-level attempt budget. ``max_retries`` is per model, so a chain of
        # N models could otherwise make ``N * (max_retries + 1)`` provider calls
        # for ONE block, each with its own backoff (up to a clamped 375s). With
        # the default 3 retries and a 3-model chain that is 12 paid calls and
        # over an hour of sleeping for a block that is going to fail anyway.
        # The per-model budget still does its job inside the primary model; this
        # cap bounds the *total* so a bad chain fails in bounded time.
        budget = [self.max_retries + 1]
        for index, candidate in enumerate(effective_chain):
            cand_profile = self._get_profile(candidate)
            cand_effort = reasoning_effort if cand_profile.supports_reasoning_effort else None
            if prompt_builder is not None:
                cand_sys, cand_user = prompt_builder(candidate)
            else:
                cand_sys, cand_user = system_prompt, user_prompt

            cand_max_tokens = max_tokens
            if max_tokens_fn is not None:
                cand_max_tokens = max_tokens_fn(candidate, cand_profile)

            try:
                res = await self._execute_single_model(
                    system_prompt=cand_sys,
                    user_prompt=cand_user,
                    model=candidate,
                    temperature=temperature,
                    reasoning_effort=cand_effort,
                    max_tokens=cand_max_tokens,
                    is_repair=is_repair,
                    attempt_budget=budget,
                )
                self._model_circuit[candidate] = (0, 0.0)
                if return_model:
                    return res, candidate
                return res
            except ModelProviderError as exc:
                # Content refusal is a prompt/content policy violation, not an infrastructure
                # failure or model outage: do not trip the model circuit breaker.
                is_refusal = (exc.details or {}).get("reason") == "content_refusal"
                if not is_refusal:
                    # Re-read the clock: ``now`` was taken at entry, and the retries
                    # and fallback walk above can burn minutes. A cooldown computed
                    # from the entry timestamp would already be expired when
                    # written, leaving the breaker permanently closed under a
                    # slow-failing chain.
                    failure_now = time.monotonic()
                    failures, exp = self._model_circuit.get(candidate, (0, 0.0))
                    # Half-open: if cooldown expired, treat as single probe failure
                    if exp > 0 and exp <= failure_now:
                        failures = 2
                    failures += 1
                    cooldown = (failure_now + 60.0) if failures >= 3 else 0.0
                    self._model_circuit[candidate] = (failures, cooldown)
                    if cooldown > 0:
                        logger.warning(
                            "Model %s tripped circuit breaker (%d consecutive failures); cooling down for 60s",
                            candidate,
                            failures,
                        )
                # Billing and credential-side failures cannot resolve by
                # renaming the model: fail the chain fast instead of burning
                # one call per entry on every block of the book.
                if _is_chain_fail_fast(exc):
                    raise
                if index >= len(effective_chain) - 1:
                    raise
                logger.warning(
                    "Model %s failed (%s), falling back to %s",
                    candidate,
                    exc,
                    effective_chain[index + 1],
                )
        raise ModelProviderError(f"All models in the fallback chain failed: {effective_chain}")

    async def _interruptible_sleep(self, seconds: float) -> None:
        """Sleep ``seconds``, aborting early on writer lease loss or cooperative-cancel token.

        A rate-limit backoff can be many minutes (a clamped ``Retry-After``);
        sleeping it in one ``asyncio.sleep`` gives the retry loop no cancel point
        until it wakes, so a job cancelled or whose lease was reclaimed mid-backoff
        kept a worker pinned. When a cancellable token or lease-lost event is present,
        the wait is raced against them.
        """
        token = _cancel_token_ctx.get() or self.cancel_token
        lease_lost = _lease_lost_ctx.get()

        if lease_lost is not None and lease_lost.is_set():
            raise LedgerWriterLeaseLostError(
                "Writer lease was reclaimed during provider retry backoff"
            )
        if token is not None and token.is_set():
            raise JobInterruptedError("Job cancelled during provider retry backoff")

        if token is None and lease_lost is None:
            await asyncio.sleep(seconds)
            return

        sleep_task = asyncio.ensure_future(asyncio.sleep(seconds))
        waiters: list[asyncio.Future[object]] = [sleep_task]
        token_waiter: asyncio.Future[object] | None = None
        lease_waiter: asyncio.Future[object] | None = None

        if token is not None:
            token_waiter = asyncio.ensure_future(token.wait())
            waiters.append(token_waiter)
        if lease_lost is not None:
            lease_waiter = asyncio.ensure_future(lease_lost.wait())
            waiters.append(lease_waiter)

        try:
            done, _pending = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in waiters:
                if not task.done():
                    task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

        if lease_lost is not None and lease_lost.is_set():
            raise LedgerWriterLeaseLostError(
                "Writer lease was reclaimed during provider retry backoff"
            )
        if token is not None and token.is_set():
            raise JobInterruptedError("Job cancelled during provider retry backoff")

    async def _execute_single_model(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float,
        reasoning_effort: str | None = None,
        max_tokens: int | None = None,
        *,
        is_repair: bool = False,
        attempt_budget: list[int] | None = None,
    ) -> str:
        """Execute inference with rate limiter acquisition, parameter pre-cleaning, and backoff retry on 429."""
        profile = self._get_profile(model)

        # 1. System prompt handling: merge into user prompt if backend doesn't support system turn
        if not profile.supports_system_prompt and system_prompt.strip():
            effective_user_prompt = f"{system_prompt.strip()}\n\n{user_prompt.strip()}"
            effective_system_prompt: str | None = None
        else:
            effective_user_prompt = user_prompt
            effective_system_prompt = system_prompt if system_prompt.strip() else None

        # 2. Reasoning effort parameter pre-cleaning
        effective_effort = reasoning_effort if profile.supports_reasoning_effort else None

        # 3. Temperature parameter pre-cleaning: unsupported models (o1/o3)
        # get None and the provider omits the key entirely, instead of
        # sending a value and eating one 400 per block.
        effective_temp = temperature if profile.supports_temperature else None

        retries = 0
        # A reasoning model can return null content with finish_reason=length
        # because the hidden reasoning consumed the whole output budget. Retrying
        # the identical max_tokens cannot help; widen it once before the normal
        # provider retry ladder takes over.
        reasoning_budget_retried = False

        while True:
            _raise_if_cancelled(self.cancel_token)
            if attempt_budget is not None:
                if attempt_budget[0] <= 0:
                    raise ModelProviderError(
                        f"Attempt budget exhausted on {model}: the fallback chain used "
                        "every call this block is allowed"
                    )
                attempt_budget[0] -= 1
            # The TPM bucket consumes the estimated request size. Count it with
            # the script-aware estimator, NOT ``chars/4``: the flat heuristic
            # under-reserves CJK ~3.4x (zh averages ~0.85 tok/char), so the
            # bucket believed it was idle while the provider returned 429 — and
            # AIMD then halved capacity for a self-inflicted breach.
            # Reserve the completion too: provider TPM counts prompt+output, and
            # for translation the output is comparable to the input. Reserving
            # only the prompt under-enforced TPM by ~2x.
            prompt_tokens = _estimate_prompt_tokens(system_prompt, user_prompt)
            completion_tokens = max_tokens if max_tokens else prompt_tokens
            estimated_tokens = prompt_tokens + max(completion_tokens, 0)
            await self.rate_limiter.acquire(estimated_tokens=estimated_tokens)
            active_provider = self._provider_for(model, is_repair=is_repair)
            try:
                try:

                    async def _invoke_provider(
                        prov: BaseModelProvider = active_provider,
                        output_limit: int | None = max_tokens,
                    ) -> tuple[str, str | None]:
                        try:
                            return await prov.generate_with_finish_reason(
                                prompt=effective_user_prompt,
                                system_prompt=effective_system_prompt,
                                model=model,
                                temperature=effective_temp,
                                max_tokens=output_limit,
                                reasoning_effort=effective_effort,
                            )
                        except TypeError as type_err:
                            signature_missing = (
                                "reasoning_effort" in str(type_err)
                                and "unexpected keyword argument" in str(type_err).lower()
                            )
                            if signature_missing:
                                return await prov.generate_with_finish_reason(
                                    prompt=effective_user_prompt,
                                    system_prompt=effective_system_prompt,
                                    model=model,
                                    temperature=effective_temp,
                                    max_tokens=output_limit,
                                )
                            raise

                    if self.request_timeout and self.request_timeout > 0:
                        try:
                            result, finish_reason = await asyncio.wait_for(
                                _invoke_provider(), timeout=self.request_timeout
                            )
                        except TimeoutError as timeout_err:
                            raise ModelProviderError(
                                f"Model request timed out after {self.request_timeout}s for model {model}",
                                details={"status_code": 408, "timeout": self.request_timeout},
                            ) from timeout_err
                    else:
                        result, finish_reason = await _invoke_provider()
                except ModelProviderError as exc:
                    # Gateway dropped a declared self-hosted model (401/404
                    # "model not supported"): one localhost retry before the
                    # budget/retry handling below.
                    if self._is_self_hosted_rejection(exc, model):
                        result, finish_reason = await self._generate_via_local_fallback(
                            exc,
                            model,
                            effective_user_prompt,
                            effective_system_prompt,
                            effective_temp,
                            max_tokens,
                        )
                        exc = None  # type: ignore[assignment]
                    if exc is not None and (
                        (exc.details or {}).get("reasoning_budget_exhausted")
                        and not reasoning_budget_retried
                    ):
                        reasoning_budget_retried = True
                        # A reasoning-capable provider may leave ``max_tokens``
                        # unset (reasoning tokens share the output budget) and
                        # the gateway may then impose its own tiny default. The
                        # exhaustion response is the only evidence of that
                        # default, so widen from the reported reasoning spend.
                        observed = int((exc.details or {}).get("reasoning_tokens", 0) or 0)
                        current = max_tokens or observed
                        widened = max(current * 4, _REASONING_MIN_OUTPUT_TOKENS)
                        logger.warning(
                            "Model %s exhausted %s output tokens on reasoning; "
                            "retrying once with max_tokens=%d",
                            model,
                            (
                                f"max_tokens={max_tokens}"
                                if max_tokens is not None
                                else "the gateway default"
                            ),
                            widened,
                        )
                        max_tokens = widened
                        retries += 1
                        if retries > self.max_retries:
                            raise
                        continue
                    elif exc is not None:
                        raise
                # Rate-limiter bookkeeping must never be misread as a provider
                # failure: letting an exception here reach the broad handler
                # below would re-run the already-billed request and relabel the
                # cause as "Unexpected provider error". Log and continue.
                try:
                    if hasattr(self.rate_limiter, "report_success_async"):
                        await self.rate_limiter.report_success_async()
                    else:
                        self.rate_limiter.report_success()
                except Exception:
                    logger.warning("rate limiter success bookkeeping failed", exc_info=True)
                # "length" means the output hit the token limit — continue
                # tail-anchored instead of silently shipping half a block.
                if finish_reason == "length":
                    result = await self._continue_truncated_output(
                        system_prompt=effective_system_prompt or "",
                        user_prompt=effective_user_prompt,
                        model=model,
                        temperature=effective_temp,
                        partial=result,
                        is_repair=is_repair,
                        reasoning_effort=effective_effort,
                        max_tokens=max_tokens,
                    )
                return result
            except ModelProviderError as exc:
                details = exc.details or {}
                retry_after_header = details.get("retry_after")
                # Fail-fast: 400/401/402/403 raise immediately instead of
                # burning max_retries+1 calls that can never succeed.
                action = classify_provider_error(exc)
                if not action.retryable:
                    if action.top_up_hint:
                        raise ModelProviderError(
                            f"{exc} [fail-fast: billing/quota exhausted, "
                            "top up the provider account before retrying]",
                            doc_id=exc.doc_id,
                            details=details,
                        ) from exc
                    raise
                # Detect 429 through the one shared rule (see
                # _is_rate_limit_error).
                retries += 1
                if retries > self.max_retries:
                    raise
                is_429 = _is_rate_limit_error(exc)
                if is_429:
                    if hasattr(self.rate_limiter, "report_429_async"):
                        await self.rate_limiter.report_429_async()
                    else:
                        self.rate_limiter.report_429()
                    wait = 0.5 * (2**retries)
                    if retry_after_header:
                        parsed_wait = _parse_retry_after(retry_after_header)
                        if parsed_wait is not None:
                            # Clamp an absurd header (86400 s, or an HTTP-date
                            # days out) so one 429 cannot park every worker for
                            # the rest of the day.
                            wait = min(max(wait, parsed_wait), _RETRY_AFTER_MAX_SECONDS)
                    # Full jitter: without it, N concurrent workers retry in
                    # lockstep and re-create the thundering herd we just escaped.
                    wait += random.uniform(0, min(1.0, wait * 0.25))
                    await self._interruptible_sleep(wait)
                else:
                    # Jittered linear backoff: without jitter concurrent workers
                    # retry in lockstep and re-create the herd. Interruptible for
                    # the same reason the 429 path is: a cancel that has to wait
                    # out a 5xx backoff still bills the retry that follows it.
                    base = 0.2 * retries
                    await self._interruptible_sleep(base + random.uniform(0, base))
            except ValueError as exc:
                # A 200 whose body is not the declared JSON (a proxy error page,
                # a truncated stream) is deterministic: retrying the same request
                # only burns quota and then rewraps the real cause as "unexpected".
                # Fail this model fast; the fallback chain still gets its turn.
                # Narrow on purpose: a ``TypeError`` re-raised by the inner
                # signature-missing guard (or any other programming error) must
                # still reach the generic retry handler below, not be relabelled
                # as an unparseable body.
                raise ModelProviderError(
                    f"Provider response could not be parsed: {type(exc).__name__}: {exc}"
                ) from exc
            except (BudgetExceededError, JobInterruptedError, LedgerWriterLeaseLostError):
                # A hard stop (spend cap / cooperative cancel) is not a transient
                # provider error: retrying it burns the budget the cap exists to
                # protect, and rewrapping it as "Unexpected provider error" hides
                # the real cause from the caller.
                raise
            except (
                AssertionError,
                TypeError,
                AttributeError,
                NameError,
                NotImplementedError,
            ) as err:
                logger.critical(
                    "Programming defect encountered in ModelRouter: %s", err, exc_info=True
                )
                raise
            except Exception as exc:
                retries += 1
                if retries > self.max_retries:
                    raise ModelProviderError(f"Unexpected provider error: {exc}") from exc
                base = 0.2 * retries
                await self._interruptible_sleep(base + random.uniform(0, base))

    async def _continue_truncated_output(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float | None,
        partial: str,
        is_repair: bool = False,
        reasoning_effort: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Continue a token-limit-truncated generation.

        Tail-anchored: the last ~200 characters of the partial output ride
        along so the model resumes at the right spot; the merge step drops
        the echoed overlap. A failed continuation call keeps the partial
        output (half a block beats no block — downstream gates still run).
        """
        text = partial
        for round_idx in range(_MAX_CONTINUATION_ROUNDS):
            # Continuation is up to 3 more paid calls. Without a check here a
            # cooperative cancel only took effect after the whole loop, so a
            # cancelled job kept buying completions.
            _raise_if_cancelled(self.cancel_token)
            tail = text[-_CONTINUATION_TAIL_CHARS:]
            logger.info(
                "Model %s generation truncated (length limit hit, %d chars); auto-continuing (round %d/%d)...",
                model,
                len(text),
                round_idx + 1,
                _MAX_CONTINUATION_ROUNDS,
            )
            continuation_prompt = (
                f"{user_prompt}\n\n"
                "---\n"
                "Your previous reply was cut off by the output token limit. "
                "Continue it from exactly where it stopped. Do NOT repeat "
                "already-generated content and do NOT add commentary. The "
                "final characters of your previous reply, for anchoring:\n"
                f"{tail}"
            )
            try:
                # Reserve like the primary call does: the continuation prompt
                # re-sends the full user_prompt, and a fixed 500 under-counts
                # an order of magnitude on long macro-chunks — the provider
                # 429s while the bucket believes it is idle, and AIMD halves
                # capacity for a self-inflicted breach.
                prompt_tokens = _estimate_prompt_tokens(system_prompt, continuation_prompt)
                await self.rate_limiter.acquire(estimated_tokens=prompt_tokens * 2)
                active_provider = self._provider_for(model, is_repair=is_repair)

                async def _invoke_continuation(
                    prov: BaseModelProvider = active_provider,
                    c_prompt: str = continuation_prompt,
                ) -> tuple[str, str | None]:
                    try:
                        return await prov.generate_with_finish_reason(
                            prompt=c_prompt,
                            system_prompt=system_prompt,
                            model=model,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            reasoning_effort=reasoning_effort,
                        )
                    except TypeError as type_err:
                        if (
                            "reasoning_effort" in str(type_err)
                            and "unexpected keyword argument" in str(type_err).lower()
                        ):
                            return await prov.generate_with_finish_reason(
                                prompt=c_prompt,
                                system_prompt=system_prompt,
                                model=model,
                                temperature=temperature,
                                max_tokens=max_tokens,
                            )
                        raise

                if self.request_timeout and self.request_timeout > 0:
                    continuation, finish_reason = await asyncio.wait_for(
                        _invoke_continuation(),
                        timeout=self.request_timeout,
                    )
                else:
                    continuation, finish_reason = await _invoke_continuation()
                try:
                    if hasattr(self.rate_limiter, "report_success_async"):
                        await self.rate_limiter.report_success_async()
                    else:
                        self.rate_limiter.report_success()
                except Exception:
                    logger.warning("continuation limiter bookkeeping failed", exc_info=True)
            except (BudgetExceededError, JobInterruptedError, LedgerWriterLeaseLostError):
                # Keeping the partial output is right for a transient provider
                # failure, but wrong for a hard stop: swallowing a spend cap or a
                # cooperative cancel here would let a still-billing generation
                # continue as if nothing happened.
                raise
            except (
                AssertionError,
                TypeError,
                AttributeError,
                NameError,
                NotImplementedError,
            ) as err:
                logger.critical(
                    "Programming defect encountered during continuation: %s", err, exc_info=True
                )
                raise
            except Exception as exc:
                # Any failure of the *continuation* keeps the already-produced
                # partial: the primary call succeeded and was billed, and half a
                # block beats re-running (and re-billing) the whole generation.
                # ``asyncio.CancelledError`` is a BaseException, so a real cancel
                # still propagates.
                if _is_rate_limit_error(exc):
                    if hasattr(self.rate_limiter, "report_429_async"):
                        await self.rate_limiter.report_429_async()
                    else:
                        self.rate_limiter.report_429()
                logger.warning(
                    "Continuation call failed, keeping partial output: %s", exc, exc_info=True
                )
                return text
            text = _merge_continuation(text, continuation)
            if finish_reason != "length":
                break
        else:
            logger.warning(
                "Output still truncated after %d continuation rounds",
                _MAX_CONTINUATION_ROUNDS,
            )
        return text

    async def complete_raw(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        reasoning_effort: str | None = None,
        model: str | None = None,
    ) -> str:
        """Free-form completion on the draft tier (e.g. bulk terminology backfill)."""
        target_model = model or self.draft_model
        effort = reasoning_effort or self.draft_reasoning_effort
        return await self._execute_with_retry(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=target_model,
            temperature=temperature,
            reasoning_effort=effort,
        )

    async def complete_with_images(
        self,
        prompt: str,
        images_b64_png: list[str],
        system_prompt: str | None = None,
        temperature: float = 0.0,
        model: str | None = None,
    ) -> str:
        """Vision completion passthrough (raises when the provider lacks support)."""
        if not self.allow_page_upload:
            raise ModelProviderError(
                "page-image egress disabled (UBT_ALLOW_PAGE_UPLOAD=false): "
                "re-enable it or use a text-only route"
            )
        active_provider = self._provider_for(model or self.repair_model)
        generate_vision = getattr(active_provider, "generate_with_images", None)
        if not callable(generate_vision):
            raise ModelProviderError("Bound provider does not support vision input")
        # This channel carries the run's heaviest requests (a 150 dpi page crop
        # per call), so it must go through the limiter like every other call:
        # reserve tokens, report 429, and let the AIMD loop learn the provider
        # is throttling it. Base64 characters are not text tokens, so a page
        # image is charged at the tile count a vision encoder spends on it.
        estimated_tokens = _estimate_prompt_tokens(prompt, system_prompt or "") + (
            _VISION_TOKENS_PER_PAGE_IMAGE * max(1, len(images_b64_png))
        )
        await self.rate_limiter.acquire(estimated_tokens=estimated_tokens)
        try:
            result = await generate_vision(
                prompt,
                images_b64_png,
                system_prompt=system_prompt,
                model=model or self.repair_model,
                temperature=temperature,
            )
        except Exception as exc:
            if _is_rate_limit_error(exc):
                if hasattr(self.rate_limiter, "report_429_async"):
                    await self.rate_limiter.report_429_async()
                else:
                    self.rate_limiter.report_429()
            raise
        if hasattr(self.rate_limiter, "report_success_async"):
            await self.rate_limiter.report_success_async()
        else:
            self.rate_limiter.report_success()
        return str(result)

    async def draft(
        self,
        block: IRBlock,
        glossary_table: str = "",
        neighbor_context: str = "",
        target_lang: str = "zh",
        source_lang: str = "en",
        genre_profile: str = "general",
        rolling_summary: str = "",
        temperature: float = 0.3,
        reasoning_effort: str | None = None,
        global_glossary: str = "",
        few_shot_reference: str = "",
        epoch_summary: str = "",
        model: str | None = None,
        domain: str | None = None,
    ) -> str:
        """Route to draft model tier with strategy-driven prompt construction and extraction.

        ``model`` allows overriding the default draft model for specific blocks.
        ``domain`` is the operator's explicit subject descriptor (``--domain``).
        """
        effective_model = model or self.draft_model
        effort = reasoning_effort or self.draft_reasoning_effort

        def _build_prompts(cand_model: str) -> tuple[str, str]:
            return self.build_draft_prompt(
                source_text=block.source_text,
                glossary_table=glossary_table,
                neighbor_context=neighbor_context,
                target_lang=target_lang,
                source_lang=source_lang,
                genre_profile=genre_profile,
                rolling_summary=rolling_summary,
                global_glossary=global_glossary,
                few_shot_reference=few_shot_reference,
                epoch_summary=epoch_summary,
                model=cand_model,
                domain=domain,
                block_type=block.block_type,
            )

        def _calc_max_tokens(cand_model: str, cand_profile: ModelProfile) -> int | None:
            # Bound the output budget from the source size: ~4 chars per
            # source token, times a generous 2x headroom => len/2 tokens. The
            # provider's 400 recovery does not shrink max_tokens, so an
            # over-large request draws "max_tokens too large" from models capped
            # at 8k/16k output — the budget must be right the first time.
            if cand_profile.supports_reasoning_effort:
                # Reasoning tokens also count against the output cap. Leaving
                # it unset let a gateway apply its own tiny default (observed:
                # hy3 spent all 1024 on hidden reasoning and returned null
                # content). Size the budget from the source instead.
                src_tokens = _estimate_prompt_tokens(block.source_text)
                return self._reasoning_output_tokens(
                    src_tokens,
                    provider=self._provider_for(cand_model, is_repair=False),
                )
            src_tokens = _estimate_prompt_tokens(block.source_text)
            return max(1024, int(src_tokens * 2.0))

        raw_res, actual_model = await self._execute_with_retry(
            system_prompt="",
            user_prompt="",
            model=effective_model,
            temperature=temperature,
            reasoning_effort=effort,
            prompt_builder=_build_prompts,
            max_tokens_fn=_calc_max_tokens,
            return_model=True,
        )
        actual_profile = self._get_profile(actual_model)
        return TranslationOutputExtractor.extract(
            raw_res,
            strategy=actual_profile.extraction_strategy,
        )

    async def draft_macro_chunk(
        self,
        blocks: list[IRBlock],
        glossary_table: str = "",
        neighbor_context: str = "",
        target_lang: str = "zh",
        source_lang: str = "en",
        genre_profile: str = "general",
        rolling_summary: str = "",
        temperature: float = 0.3,
        reasoning_effort: str | None = None,
        global_glossary: str = "",
        few_shot_reference: str = "",
        epoch_summary: str = "",
        model: str | None = None,
        domain: str | None = None,
        section_context: str = "",
    ) -> dict[str, str]:
        """Draft a group of blocks in a single structured XML prompt.

        Returns a dictionary mapping block_id -> translated_text.
        """
        if not blocks:
            return {}
        effective_model = model or self.draft_model
        effort = reasoning_effort or self.draft_reasoning_effort
        block_pairs = [(b.id, b.source_text) for b in blocks]

        def _build_prompts(cand_model: str) -> tuple[str, str]:
            return build_macro_chunk_draft_prompt(
                blocks=block_pairs,
                glossary_table=glossary_table,
                neighbor_context=neighbor_context,
                target_lang=target_lang,
                source_lang=source_lang,
                genre_profile=genre_profile,
                rolling_summary=rolling_summary,
                global_glossary=global_glossary,
                few_shot_reference=few_shot_reference,
                epoch_summary=epoch_summary,
                domain=domain,
                section_context=section_context,
            )

        def _calc_max_tokens(cand_model: str, cand_profile: ModelProfile) -> int | None:
            if cand_profile.supports_reasoning_effort:
                total_src_tokens = sum(_estimate_prompt_tokens(b.source_text) for b in blocks)
                return self._reasoning_output_tokens(
                    total_src_tokens,
                    provider=self._provider_for(cand_model, is_repair=False),
                )
            total_src_tokens = sum(_estimate_prompt_tokens(b.source_text) for b in blocks)
            return max(2048, int(total_src_tokens * 2.0) + 512)

        raw_res, actual_model = await self._execute_with_retry(
            system_prompt="",
            user_prompt="",
            model=effective_model,
            temperature=temperature,
            reasoning_effort=effort,
            prompt_builder=_build_prompts,
            max_tokens_fn=_calc_max_tokens,
            return_model=True,
        )
        return TranslationOutputExtractor.extract_macro_blocks(raw_res)

    async def repair(
        self,
        block: IRBlock,
        draft_text: str,
        error_flags: list[str],
        glossary_table: str = "",
        target_lang: str = "zh",
        source_lang: str = "en",
        temperature: float = 0.2,
        reasoning_effort: str | None = None,
        annotated_draft: str = "",
        has_error_spans: bool = False,
        image_b64: str | None = None,
        global_glossary: str = "",
    ) -> str:
        """Route to repair model tier with strategy-driven prompt construction and extraction."""
        effort = reasoning_effort or self.repair_reasoning_effort

        # L4 Visual Scalpel precision repair: if high-res visual crop is available,
        # try multimodal repair first to resolve formula, subscript, or layout defects.
        repair_profile = self._get_profile(self.repair_model)
        # Declared capability only. This used to fall back to a list of name
        # substrings (``vl``, ``vision``, ``4o``, ``gemini``, ``claude-3``, …),
        # which both over- and under-claimed: ``gemini-embedding-001`` matched
        # ``gemini`` and was sent a page image it cannot read, while
        # ``claude-sonnet-4-5`` matched nothing and silently lost visual repair
        # altogether. The provider block is where the answer is actually known, so
        # an operator who wants the visual path declares ``supports_vision = true``.
        model_is_vision = repair_profile.supports_vision
        if image_b64 and model_is_vision and not self.allow_page_upload:
            logger.warning(
                "Visual repair skipped (crop ready): page-image egress disabled by "
                "allow_page_upload=false; falling back to text-only repair."
            )
        elif image_b64 and model_is_vision:
            try:
                system_prompt, user_prompt = self.build_repair_prompt(
                    source_text=block.source_text,
                    draft_text=draft_text,
                    error_flags=error_flags,
                    glossary_table=glossary_table,
                    target_lang=target_lang,
                    source_lang=source_lang,
                    annotated_draft=annotated_draft,
                    has_error_spans=has_error_spans,
                    model=self.repair_model,
                    global_glossary=global_glossary,
                )
                vision_prompt = (
                    f"{user_prompt}\n\n"
                    "Attached is the high-resolution visual crop of the original block from the document. "
                    "Use this visual evidence to precisely resolve layout, subscript/superscript, formula, "
                    "or symbol ambiguities."
                )
                raw_res = await self.complete_with_images(
                    prompt=vision_prompt,
                    images_b64_png=[image_b64],
                    system_prompt=system_prompt,
                    temperature=temperature,
                    model=self.repair_model,
                )
                actual_profile = self._get_profile(self.repair_model)
                return TranslationOutputExtractor.extract(
                    raw_res,
                    strategy=actual_profile.extraction_strategy,
                )
            except (
                asyncio.CancelledError,
                JobInterruptedError,
                BudgetExceededError,
                LedgerWriterLeaseLostError,
            ):
                raise
            except Exception as exc:
                logger.warning(
                    "Visual repair failed for block %s; falling back to text repair: %s",
                    block.id,
                    exc,
                )

        def _build_repair_prompts(cand_model: str) -> tuple[str, str]:
            return self.build_repair_prompt(
                source_text=block.source_text,
                draft_text=draft_text,
                error_flags=error_flags,
                glossary_table=glossary_table,
                target_lang=target_lang,
                source_lang=source_lang,
                annotated_draft=annotated_draft,
                has_error_spans=has_error_spans,
                model=cand_model,
                global_glossary=global_glossary,
            )

        def _calc_repair_max_tokens(cand_model: str, cand_profile: ModelProfile) -> int | None:
            if cand_profile.supports_reasoning_effort:
                src_tokens = _estimate_prompt_tokens(block.source_text)
                return self._reasoning_output_tokens(
                    src_tokens,
                    provider=self._provider_for(cand_model, is_repair=True),
                )
            src_tokens = _estimate_prompt_tokens(block.source_text)
            return max(4096, int(src_tokens * 4.0))

        raw_res, actual_model = await self._execute_with_retry(
            system_prompt="",
            user_prompt="",
            model=self.repair_model,
            temperature=temperature,
            reasoning_effort=effort,
            prompt_builder=_build_repair_prompts,
            max_tokens_fn=_calc_repair_max_tokens,
            return_model=True,
            is_repair=True,
        )
        actual_profile = self._get_profile(actual_model)
        return TranslationOutputExtractor.extract(
            raw_res,
            strategy=actual_profile.extraction_strategy,
        )


#: Names this module re-exports from :mod:`ubt.core.router.router_errors` for
#: its consumers (the draft stage imports the batch types and the error
#: classifier from here; the retry-cap test reads the cap constant).
__all__ = [
    "ModelRouter",
    "ProviderErrorAction",
    "_CONTINUATION_TAIL_CHARS",
    "_FAIL_FAST_STATUS",
    "_LOCAL_FALLBACK_BASE_URL_DEFAULT",
    "_LOCAL_FALLBACK_BASE_URL_ENV",
    "_MAX_CONTINUATION_ROUNDS",
    "_MIN_MERGE_OVERLAP",
    "_MODEL_IDENTITY_PHRASES",
    "_MODEL_REJECTION_STATUSES",
    "_QUOTA_EXHAUSTED_PHRASES",
    "_RETRYABLE_STATUS",
    "_RETRY_AFTER_MAX_SECONDS",
    "_SELF_HOSTED_BACKENDS",
    "_VISION_TOKENS_PER_PAGE_IMAGE",
    "_estimate_prompt_tokens",
    "_is_chain_fail_fast",
    "_is_rate_limit_error",
    "_merge_continuation",
    "_parse_retry_after",
    "classify_provider_error",
]
