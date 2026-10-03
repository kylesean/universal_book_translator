"""Intelligent ModelRouter governing model tier dispatch, retries and rate limiting.

Prompt text lives in :mod:`ubt.core.router.prompts`; this file resolves a model
name to a capability profile and executes against the provider.
"""

import asyncio
import hashlib
import json
import logging
import os
import random
import re
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Literal, Protocol, overload

from ubt.core.exceptions import BudgetExceededError, JobInterruptedError, ModelProviderError
from ubt.core.ir.models import IRBlock
from ubt.core.router.capabilities import (
    ExtractionStrategy,
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
from ubt.core.router.registry import ModelCapabilityRegistry

logger = logging.getLogger(__name__)

# Max tail-anchored continuation rounds per generation before
# giving up (the fast-pass length-ratio gate still guards the result).
_MAX_CONTINUATION_ROUNDS = 3
# Characters of the already-generated tail carried into the continuation
# prompt, and the max suffix/prefix overlap removed when merging.
_CONTINUATION_TAIL_CHARS = 200
_MIN_MERGE_OVERLAP = 2


def _merge_continuation(text: str, continuation: str) -> str:
    """Merge a continuation, dropping the repeated overlap at the seam.

    LLM continuations often echo the last characters of the previous reply
    even when told not to; the largest suffix/prefix match (>= 2 chars so a
    coincidental single character cannot corrupt the join) is removed.
    """
    if not continuation:
        return text
    max_overlap = min(len(text), len(continuation), _CONTINUATION_TAIL_CHARS)
    for k in range(max_overlap, _MIN_MERGE_OVERLAP - 1, -1):
        if text.endswith(continuation[:k]):
            return text + continuation[k:]
    return text + continuation


@dataclass(frozen=True)
class ProviderErrorAction:
    """Fail-fast vs retryable taxonomy for provider errors."""

    retryable: bool
    top_up_hint: bool = False
    reason: str = ""


# Client errors that no retry will ever fix (auth/billing/malformed/not-found).
# 404/422 are model-rejection shapes: the declared self-hosted local retry runs
# before this classification, so a genuinely-missing model fails fast instead
# of burning max_retries.
_FAIL_FAST_STATUS = frozenset({400, 401, 402, 403, 404, 422})
# Transient errors worth a backoff retry.
_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
# A completed batch's result file is one GET away. Ride out a short transport
# blip rather than surfacing "batch is dead": the caller answers a batch failure
# by cancelling the already-paid job and re-drafting every block interactively,
# which bills the same work twice. Bounded, so a persistent outage still fails.
_BATCH_RESULT_FETCH_GRACE_SECONDS = 5.0
_BATCH_RESULT_FETCH_BACKOFF_SECONDS = 1.0
# Default local endpoint for the one retry a self-hosted model gets when the
# primary gateway drops it. Ollama is the default because it is the most common
# one-line install; llama-swap (:9090), llama-server (:8081) and vLLM (:8000)
# users set UBT_LOCAL_FALLBACK_BASE_URL.
_LOCAL_FALLBACK_BASE_URL_DEFAULT = "http://localhost:11434/v1"
_LOCAL_FALLBACK_BASE_URL_ENV = "UBT_LOCAL_FALLBACK_BASE_URL"
# ``deployment_backend`` values that mean "this model lives on a self-hosted
# server" (Ollama, llama.cpp/llama-swap, vLLM, LM Studio, ...). The gateway may
# not carry such a model; the local stack usually does.
_SELF_HOSTED_BACKENDS = frozenset(
    {
        "self_hosted",
        "self-hosted",
        "local",
        "ollama",
        "llama.cpp",
        "llama-cpp",
        "llama_server",
        "llama-server",
        "llama_swap",
        "llama-swap",
        "llamaswap",
        "vllm",
        "lmstudio",
        "lm-studio",
        "sglang",
    }
)

# Fallback name hints for multimodal models whose capability profile does not
# declare ``supports_vision=True``. The profile flag (settable per deployment
# via UBT_MODEL_PROFILES_JSON/FILE) always wins; this list only covers shipped
# families so visual-scalpel repair works out of the box.
_VISION_NAME_HINTS: tuple[str, ...] = (
    "vl",
    "vision",
    "4o",
    "gemini",
    "claude-3",
    "claude-4",
    "llava",
)
# Gateway rejections worth exactly one local retry (model unknown there).
_MODEL_REJECTION_STATUSES = frozenset({401, 404})
# Message shapes identifying a *model-identity* rejection (unknown / not
# deployed / not entitled model) as opposed to a credential-side auth failure
# or a malformed request. Identity rejections may resolve on another chain
# Entry; credential-side ones cannot, so the chain fails fast
# instead of burning a call per model.
_MODEL_IDENTITY_PHRASES = (
    "not supported",
    "not available",
    "not found",
    "no such model",
    "does not exist",
    "unknown model",
    "unsupported",
    "not deployed",
)


def _is_chain_fail_fast(exc: ModelProviderError) -> bool:
    """True when trying the next fallback model cannot help.

    402 (billing) always fails fast. Other fail-fast statuses (400/401/403/
    404/422) fail fast too unless the message reads as model-identity — only
    then is the model name (not the credential or the request) the problem
    and the next chain entry worth a call. Retryable/unknown errors proceed
    to the next fallback model in the chain.
    """
    action = classify_provider_error(exc)
    if action.top_up_hint:
        return True
    if (exc.details or {}).get("fail_fast"):
        # Locally-tagged unrecoverable (e.g. missing CLI binary): no other
        # entry can succeed either.
        return True
    status = (exc.details or {}).get("status_code")
    if status is None:
        match = re.search(r"\b([45]\d\d)\b", str(exc))
        if match:
            with suppress(ValueError, TypeError):
                status = int(match.group(1))
    if status in _FAIL_FAST_STATUS:
        return not any(phrase in str(exc).lower() for phrase in _MODEL_IDENTITY_PHRASES)
    return False


# A page crop sent to a vision model is billed as tiles, not as text: ~1k
# tokens for a full page is the order of magnitude the common encoders spend.
_VISION_TOKENS_PER_PAGE_IMAGE = 1024


def _estimate_prompt_tokens(*texts: str) -> int:
    """Script-aware token count for TPM reservation, imported lazily.

    ``ubt.core.engine.cost_estimate`` imports ``ubt.core.router.pricing``, which
    triggers the ``ubt.core.router`` package initializer (and thus this module),
    so a module-level import here closes a cycle at CLI startup. Deferring it to
    call time is safe: every module is initialized before the first request.
    """
    from ubt.core.engine.cost_estimate import count_text_tokens

    return sum(count_text_tokens(text) for text in texts)


def _is_rate_limit_error(exc: BaseException) -> bool:
    """True when the provider said "slow down" (HTTP 429, or its text equivalent).

    Detection is by structured status code, or an explicit rate-limit phrase from
    providers that omit ``details`` — never the bare substring "429", which
    matched unrelated messages and fed bogus backoff/AIMD signals.
    """
    details = getattr(exc, "details", None) or {}
    if details.get("status_code") == 429:
        return True
    # Word-boundary phrases only: a 5xx/408 body that merely mentions a
    # ``rate_limit`` request field must not trigger the 429 AIMD/backoff path.
    return re.search(r"\brate[- ]?limit(?:ed|ing)?\b", str(exc), re.IGNORECASE) is not None


def _parse_retry_after(value: str | None) -> float | None:
    """Seconds to wait from a ``Retry-After`` header, or None when unparseable.

    RFC 7231 allows ``delta-seconds`` *or* an HTTP-date. ``float()`` alone only
    handled the first form, so a server that sent a date (``Wed, 21 Oct 2026
    07:28:00 GMT``) was silently retried after the ~1s jittered backoff instead
    of the requested pause.
    """
    if not value:
        return None
    text = value.strip()
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    try:
        from email.utils import parsedate_to_datetime

        when = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    import datetime as _dt

    now = _dt.datetime.now(tz=when.tzinfo) if when.tzinfo else _dt.datetime.now()
    return max(0.0, (when - now).total_seconds())


_QUOTA_EXHAUSTED_PHRASES: tuple[str, ...] = (
    "insufficient_quota",
    "quota_exceeded",
    "exceeded your current quota",
    "credit balance",
    "balance is too low",
    "billing details",
)


def classify_provider_error(exc: BaseException) -> ProviderErrorAction:
    """Classify a provider failure as retryable or fail-fast (pure function).

    - 402 or quota exhausted → fail-fast with a top-up hint (billing, not a bug);
    - 400/401/403 → fail-fast (retrying burns quota for nothing);
    - 429/5xx/408/timeouts/connection errors → retryable with backoff;
    - anything unrecognized → retryable (retrying an unknown failure is safer
      than risking a false fail-fast).
    """
    if isinstance(exc, ModelProviderError):
        details = exc.details or {}
        if details.get("fail_fast"):
            return ProviderErrorAction(False, reason="tagged_fail_fast")
        status = details.get("status_code")
        if status is None:
            match = re.search(r"\b([45]\d\d)\b", str(exc))
            if match:
                with suppress(ValueError, TypeError):
                    status = int(match.group(1))
        if status == 402:
            return ProviderErrorAction(False, top_up_hint=True, reason="payment_required")
        body_text = str(details.get("body", "")).lower()
        msg_text = str(exc).lower()
        combined = f"{msg_text} {body_text}"
        if any(phrase in combined for phrase in _QUOTA_EXHAUSTED_PHRASES):
            return ProviderErrorAction(False, top_up_hint=True, reason="quota_exhausted")
        if status in _FAIL_FAST_STATUS:
            return ProviderErrorAction(False, reason=f"http_{status}")
        if status in _RETRYABLE_STATUS or status is None:
            message = str(exc).lower()
            if "timed out" in message or "timeout" in message:
                return ProviderErrorAction(True, reason="timeout")
            if status is not None or "rate limit" in message:
                return ProviderErrorAction(True, reason="transient")
            # No status and no recognizable message: provider bug or network
            # failure below the transport — retry, as before.
            return ProviderErrorAction(True, reason="unknown_transient")
        return ProviderErrorAction(True, reason="unknown_status")
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError, ConnectionError)):
        return ProviderErrorAction(True, reason="transport")
    return ProviderErrorAction(True, reason="unexpected")


class BatchTranslationError(RuntimeError):
    """Raised when an OpenAI-compatible batch job cannot be completed.

    Callers treat it as a signal to fall back to the interactive path; it is    never surfaced as a block failure on its own.

    ``batch_id`` is set when the error escapes *after* a batch was already
    submitted, so the caller that decides to abandon batch mode (and re-draft
    the same blocks interactively) can cancel that still-running job instead of
    paying twice. A pre-submission failure carries no id and must not cancel.
    """

    def __init__(self, *args: object, batch_id: str | None = None) -> None:
        super().__init__(*args)
        self.batch_id = batch_id


@dataclass
class BatchDraftRequest:
    """One block to be translated through the Batch API in a single job."""

    custom_id: str
    source_text: str
    glossary_table: str = ""
    neighbor_context: str = ""
    target_lang: str = "zh"
    source_lang: str = "en"
    genre_profile: str = "general"
    rolling_summary: str = ""
    global_glossary: str = ""
    few_shot_reference: str = ""
    epoch_summary: str = ""
    temperature: float = 0.3
    domain: str | None = None


@dataclass
class BatchDraftResult:
    """Outcome for one :class:`BatchDraftRequest` line."""

    custom_id: str
    text: str | None = None
    error: str | None = None


class BatchJobStore(Protocol):
    """Duck-typed ledger surface for Batch API persistence.

    Satisfied by :class:`ubt.core.engine.ledger.SQLiteJobLedger`; declared
    structurally so the router does not import the engine layer.
    """

    def register_batch_job(
        self, batch_id: str, job_id: str, idempotency_key: str, status: str = ...
    ) -> None: ...

    def find_live_batch_by_idempotency_key(self, idempotency_key: str) -> str | None: ...

    def find_live_batch_for_job(self, job_id: str, *, exclude_key: str) -> str | None: ...

    def reserve_batch_job(self, idempotency_key: str, job_id: str) -> tuple[str, str | None]: ...

    def finalize_batch_job(
        self, idempotency_key: str, batch_id: str, status: str = ...
    ) -> None: ...

    def update_batch_job_status(self, batch_id: str, status: str) -> None: ...

    def is_batch_live(self, batch_id: str) -> bool: ...


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
    ) -> None:
        self.provider = provider
        self.repair_provider = repair_provider or provider
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
        """Resolve model capability profile from registry, or default to standard LLM."""
        target = model_name or self.draft_model
        if self.registry is not None:
            profile = self.registry.resolve(target)
        else:
            profile = ModelProfile(
                model_pattern="*",
                prompt_strategy=PromptStrategy.RICH,
                extraction_strategy=ExtractionStrategy.AUTO,
                supports_reasoning_effort=False,
                supports_system_prompt=True,
                supports_temperature=True,
                display_name=f"Standard LLM ({target or 'default'})",
            )
        if self.prompt_strategy_override is not None:
            profile = profile.model_copy(update={"prompt_strategy": self.prompt_strategy_override})
        return profile

    @property
    def supports_batch_api(self) -> bool:
        """Whether the bound provider can execute OpenAI-compatible batch jobs."""
        return bool(getattr(self.provider, "supports_batch_api", False))

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
        # Circuit breaker: prioritize candidates whose circuit is healthy (cooldown expired)
        healthy = [c for c in chain if self._model_circuit.get(c, (0, 0.0))[1] <= now]
        effective_chain = healthy or chain

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
                )
                self._model_circuit[candidate] = (0, 0.0)
                if return_model:
                    return res, candidate
                return res
            except ModelProviderError as exc:
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

        while True:
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
                    result, finish_reason = await active_provider.generate_with_finish_reason(
                        prompt=effective_user_prompt,
                        system_prompt=effective_system_prompt,
                        model=model,
                        temperature=effective_temp,
                        max_tokens=max_tokens,
                        reasoning_effort=effective_effort,
                    )
                except TypeError as type_err:
                    # Only retry when the provider signature genuinely lacks the
                    # reasoning_effort parameter: requiring both the keyword name
                    # and "unexpected keyword argument" keeps an unrelated
                    # TypeError raised inside provider code from being swallowed.
                    signature_missing = (
                        "reasoning_effort" in str(type_err)
                        and "unexpected keyword argument" in str(type_err).lower()
                    )
                    if signature_missing:
                        result, finish_reason = await active_provider.generate_with_finish_reason(
                            prompt=effective_user_prompt,
                            system_prompt=effective_system_prompt,
                            model=model,
                            temperature=effective_temp,
                            max_tokens=max_tokens,
                        )
                    else:
                        raise
                except ModelProviderError as gateway_exc:
                    # Gateway dropped a declared self-hosted model (401/404
                    # "model not supported"): one localhost retry before the outer
                    # fail-fast/retry classification runs.
                    result, finish_reason = await self._generate_via_local_fallback(
                        gateway_exc,
                        model,
                        effective_user_prompt,
                        effective_system_prompt,
                        effective_temp,
                        max_tokens,
                    )
                # Rate-limiter bookkeeping must never be misread as a provider
                # failure: an exception here used to fall into the broad handler
                # below, re-run the already-billed request, and relabel the cause
                # as "Unexpected provider error". Log and continue.
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
                            wait = max(wait, parsed_wait)
                    # Full jitter: without it, N concurrent workers retry in
                    # lockstep and re-create the thundering herd we just escaped.
                    wait += random.uniform(0, min(1.0, wait * 0.25))
                    await asyncio.sleep(wait)
                else:
                    # Jittered linear backoff: without jitter concurrent workers
                    # retry in lockstep and re-create the herd.
                    base = 0.2 * retries
                    await asyncio.sleep(base + random.uniform(0, base))
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
            except (BudgetExceededError, JobInterruptedError):
                # A hard stop (spend cap / cooperative cancel) is not a transient
                # provider error: retrying it burns the budget the cap exists to
                # protect, and rewrapping it as "Unexpected provider error" hides
                # the real cause from the caller.
                raise
            except Exception as exc:
                retries += 1
                if retries > self.max_retries:
                    raise ModelProviderError(f"Unexpected provider error: {exc}") from exc
                base = 0.2 * retries
                await asyncio.sleep(base + random.uniform(0, base))

    async def _continue_truncated_output(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float | None,
        partial: str,
        is_repair: bool = False,
    ) -> str:
        """Continue a token-limit-truncated generation.

        Tail-anchored: the last ~200 characters of the partial output ride
        along so the model resumes at the right spot; the merge step drops
        the echoed overlap. A failed continuation call keeps the partial
        output (half a block beats no block — downstream gates still run).
        """
        text = partial
        for _ in range(_MAX_CONTINUATION_ROUNDS):
            tail = text[-_CONTINUATION_TAIL_CHARS:]
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
                continuation, finish_reason = await active_provider.generate_with_finish_reason(
                    prompt=continuation_prompt,
                    system_prompt=system_prompt,
                    model=model,
                    temperature=temperature,
                )
                try:
                    if hasattr(self.rate_limiter, "report_success_async"):
                        await self.rate_limiter.report_success_async()
                    else:
                        self.rate_limiter.report_success()
                except Exception:
                    logger.warning("continuation limiter bookkeeping failed", exc_info=True)
            except (BudgetExceededError, JobInterruptedError):
                # Keeping the partial output is right for a transient provider
                # failure, but wrong for a hard stop: swallowing a spend cap or a
                # cooperative cancel here would let a still-billing generation
                # continue as if nothing happened.
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
                logger.warning("Continuation call failed, keeping partial output: %s", exc)
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
            )

        def _calc_max_tokens(cand_model: str, cand_profile: ModelProfile) -> int | None:
            # Bound the output budget from the source size: ~4 chars per
            # source token, times a generous 2x headroom => len/2 tokens. The
            # provider's 400 recovery does not shrink max_tokens, so an
            # over-large request draws "max_tokens too large" from models capped
            # at 8k/16k output — the budget must be right the first time.
            # Reasoning models are exempt — their reasoning tokens also count
            # against max_tokens, and capping those corrupts the visible
            # output. Truncation at the cap is handled by the continuation
            # loop in the funnel.
            if cand_profile.supports_reasoning_effort:
                return None
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
            )

        def _calc_max_tokens(cand_model: str, cand_profile: ModelProfile) -> int | None:
            if cand_profile.supports_reasoning_effort:
                return None
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

    async def _await_batch_read[T](
        self,
        batch_id: str,
        call: Callable[[], Awaitable[T]],
        *,
        deadline: float,
        interval: float,
        what: str,
    ) -> T:
        """Await a read-only Batch API call, riding out retryable failures.

        A transient 5xx on the status endpoint *or* on the result-file fetch
        must not surface as a dead batch: the caller cancels the submitted job
        and re-drafts every block interactively, paying for the same work twice.
        Retry what the router's own taxonomy calls retryable until ``deadline``;
        a terminal failure or a passed deadline raises ``BatchTranslationError``
        carrying the batch id so the caller can abandon the job first.
        """
        loop = asyncio.get_running_loop()
        while True:
            try:
                return await call()
            except ModelProviderError as exc:
                if not classify_provider_error(exc).retryable or loop.time() >= deadline:
                    raise BatchTranslationError(
                        f"Batch {what} failed: {exc}", batch_id=batch_id
                    ) from exc
                logger.warning(
                    "Batch %s %s failed (%s); retrying inside the deadline",
                    batch_id,
                    what,
                    exc,
                )
                await asyncio.sleep(interval)

    async def draft_batch(
        self,
        requests: list[BatchDraftRequest],
        *,
        poll_interval: float = 30.0,
        poll_timeout: float = 3600.0,
        ledger: BatchJobStore | None = None,
        job_id: str | None = None,
        cleanup_files: bool = True,
        status_callback: Callable[[str, dict[str, Any]], Any] | None = None,
    ) -> list[BatchDraftResult]:
        """Translate a batch of blocks through the provider's Batch API.

        Prompts are built with the exact same topology as :meth:`draft` so
        batch and interactive outputs stay consistent. A *read* failure (the
        status endpoint or the result-file fetch) that the router's own taxonomy
        calls retryable is ridden out inside ``poll_timeout``; a non-retryable
        transport failure, that timeout, or a non-completed terminal status
        raises
        :class:`BatchTranslationError` with the provider job attached, and
        callers fall back to interactive drafting for the same blocks. A
        :class:`BudgetExceededError` from ``status_callback`` cancels the batch
        first: the cap governs spend, not just this loop.

        when ``ledger`` is given, the job is persisted under an
        idempotency key derived from the submitted custom_id set. A restart
        that proposes the same batch again resumes polling the original
        provider job instead of double-billing a duplicate submission.

        Known cost, deliberately not fixed: submitting holds the whole book in
        memory several times over at once -- the pending blocks, their prepared
        draft inputs, ``jsonl_requests``, and the JSONL string the provider
        upload joins out of it. Measured on 2,000 blocks of ~576 characters
        (1.15 MB of text): an 11 MB payload and an 88 MB peak, because every
        line repeats the system prefix. A 10,000-block book therefore costs
        hundreds of MB. Streaming each line as it is built and dropping that
        block's prepared input would fix it; it is not on the hot path for the
        document sizes this mode targets today, so the copies stay.
        """
        if not requests:
            return []
        if not self.supports_batch_api:
            raise BatchTranslationError("Provider does not support the Batch API")
        profile = self._get_profile(self.draft_model)
        effort = self.draft_reasoning_effort

        jsonl_requests: list[dict[str, Any]] = []
        for req in requests:
            system_prompt, user_prompt = self.build_draft_prompt(
                source_text=req.source_text,
                glossary_table=req.glossary_table,
                neighbor_context=req.neighbor_context,
                target_lang=req.target_lang,
                source_lang=req.source_lang,
                genre_profile=req.genre_profile,
                rolling_summary=req.rolling_summary,
                global_glossary=req.global_glossary,
                few_shot_reference=req.few_shot_reference,
                epoch_summary=req.epoch_summary,
                model=self.draft_model,
                domain=req.domain,
            )
            messages: list[dict[str, str]] = []
            if system_prompt.strip():
                if profile.supports_system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                else:
                    user_prompt = f"{system_prompt.strip()}\n\n{user_prompt}"
            messages.append({"role": "user", "content": user_prompt})

            body: dict[str, Any] = {"model": self.draft_model, "messages": messages}
            if profile.supports_temperature:
                body["temperature"] = req.temperature
            if profile.supports_reasoning_effort and effort:
                body["reasoning_effort"] = effort
            if not profile.supports_reasoning_effort:
                # Mirror the interactive path's output budget. Without a cap the
                # provider's default truncates a long block, and the Batch path
                # has no continuation loop — a half sentence would ship.
                src_tokens = _estimate_prompt_tokens(req.source_text)
                body["max_tokens"] = max(1024, int(src_tokens * 2.0))

            # Match the interactive path: a server relying on
            # {"enable_thinking": false} must not start emitting thinking
            # traces in batch mode only.
            chat_template_kwargs = self.provider.chat_template_kwargs
            if chat_template_kwargs:
                body["chat_template_kwargs"] = chat_template_kwargs
            jsonl_requests.append(
                {
                    "custom_id": req.custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }
            )

        try:
            # Idempotency over the exact submitted payload (model +
            # custom_ids + fully built prompts), not just model+custom_id: a
            # glossary/context change on restart must submit a fresh batch
            # instead of resuming one built with the old terminology.
            idempotency_key = hashlib.sha1(
                json.dumps(jsonl_requests, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest()

            batch_id: str | None = None
            if ledger is not None and job_id is not None:
                # Synchronous SQLite: off the loop -- a WAL lock under
                # ``busy_timeout`` would stall every concurrent job and SSE
                # subscriber sharing it.
                outcome, existing = await asyncio.to_thread(
                    ledger.reserve_batch_job, idempotency_key, job_id
                )
                if outcome == "resume" and existing:
                    logger.info(
                        "Resuming live batch job %s (idempotency key %s…)",
                        existing,
                        idempotency_key[:12],
                    )
                    batch_id = existing
                elif outcome == "pending":
                    # A concurrent worker holds the create reservation for this
                    # exact payload. Submitting anyway would double-bill the same
                    # blocks, so bail and let the caller fall back to interactive
                    # drafting rather than block on the other worker.
                    raise BatchTranslationError(
                        f"batch {idempotency_key[:12]}… is being submitted by another "
                        "worker; skipping batch to avoid a duplicate submission"
                    )
            if batch_id is None:
                # Own the create (fresh reservation, or adopted a stale one).
                # A changed payload yields a new idempotency key, so a
                # prior live batch for this job would otherwise keep billing and
                # never be cancelled. Abandon it before creating the new one.
                if ledger is not None and job_id is not None:
                    superseded = await asyncio.to_thread(
                        ledger.find_live_batch_for_job, job_id, exclude_key=idempotency_key
                    )
                    if superseded:
                        logger.warning(
                            "Abandoning superseded live batch %s for job %s", superseded, job_id
                        )
                        await self.abandon_batch(superseded, ledger=ledger, job_id=job_id)
                batch_id = await self.provider.create_batch_job(jsonl_requests)
                if ledger is not None and job_id is not None:
                    await asyncio.to_thread(ledger.finalize_batch_job, idempotency_key, batch_id)
            loop = asyncio.get_running_loop()
            deadline = loop.time() + max(poll_timeout, 1.0)
            interval = max(poll_interval, 0.5)
            job: dict[str, Any] = {}
            while True:
                job = await self._await_batch_read(
                    batch_id,
                    lambda: self.provider.get_batch_job(batch_id),
                    deadline=deadline,
                    interval=interval,
                    what="status poll",
                )
                status = str(job.get("status") or "")
                if ledger is not None and job_id is not None:
                    await asyncio.to_thread(
                        ledger.update_batch_job_status, batch_id, status or "unknown"
                    )
                if status_callback is not None:
                    # Not swallowed: the callback is the caller's cooperative
                    # interrupt channel (cancel check, budget cap). Passing
                    # exceptions on let a still-billing batch run past the cap.
                    try:
                        res = status_callback(status, job)
                        if asyncio.iscoroutine(res):
                            await res
                    except (BudgetExceededError, JobInterruptedError):
                        # The cap is a hard stop on spend and this batch *is*
                        # spend: leaving it running makes the limit a
                        # suggestion. The same holds for a cooperative cancel —
                        # the caller re-raises without falling back, so without
                        # this the batch keeps billing after the job is gone.
                        # Cancel it here; the caller has no cancel point of its
                        # own.
                        await self.abandon_batch(batch_id, ledger=ledger, job_id=job_id)
                        raise
                if status in ("completed", "failed", "expired", "cancelled"):
                    break
                if loop.time() >= deadline:
                    raise BatchTranslationError(
                        f"Batch job {batch_id} timed out after {poll_timeout:.0f}s "
                        f"(status={status!r})",
                        batch_id=batch_id,
                    )
                await asyncio.sleep(interval)
            if str(job.get("status")) != "completed":
                raise BatchTranslationError(
                    f"Batch job {batch_id} ended with status {str(job.get('status'))!r}",
                    batch_id=batch_id,
                )
            raw_results = await self._await_batch_read(
                batch_id,
                lambda: self.provider.fetch_batch_results(batch_id),
                deadline=loop.time() + _BATCH_RESULT_FETCH_GRACE_SECONDS,
                interval=_BATCH_RESULT_FETCH_BACKOFF_SECONDS,
                what="result fetch",
            )
        except ModelProviderError as exc:
            # Carry batch_id through the rewrap so the caller can still
            # abandon_batch a job that was already created (otherwise it keeps
            # billing while interactive drafting re-translates the same blocks).
            raise BatchTranslationError(
                f"Batch transport failure: {exc}",
                batch_id=batch_id,
            ) from exc

        # Reap the Files-API objects now that the results are in hand: the
        # input file is the whole batch's source text and the output file its
        # translation, neither of which should outlive the job on the provider.
        # Best-effort (cleanup_batch_files never raises), and outside the
        # transport try/except so a failed delete never masquerades as a
        # batch failure that would re-bill the blocks via interactive fallback.
        if cleanup_files:
            await self.provider.cleanup_batch_files(batch_id)
        if ledger is not None and job_id is not None:
            await asyncio.to_thread(ledger.update_batch_job_status, batch_id, "consumed")

        results: list[BatchDraftResult] = []
        for req in requests:
            entry = raw_results.get(req.custom_id)
            if entry is None:
                results.append(
                    BatchDraftResult(custom_id=req.custom_id, error="Missing from batch output")
                )
                continue
            if entry.get("error") or entry.get("content") is None:
                results.append(
                    BatchDraftResult(
                        custom_id=req.custom_id,
                        error=str(entry.get("error") or "Empty batch response"),
                    )
                )
                continue
            text = TranslationOutputExtractor.extract(
                str(entry["content"]),
                strategy=profile.extraction_strategy,
            )
            results.append(BatchDraftResult(custom_id=req.custom_id, text=text))
        return results

    async def abandon_batch(
        self, batch_id: str, ledger: BatchJobStore | None = None, job_id: str | None = None
    ) -> None:
        """Cancel a still-running batch we are about to abandon to interactive.

        Only a live batch is cancelled and marked ``cancelled``: a batch that
        already finished — results harvested or not — is terminal, so cancelling
        is a no-op at the provider and overwriting its status would drop the
        resumable state of an already-paid run. The caller invokes this at the
        decision point; cleanup must never raise over the fallback it annotates.
        """
        if (
            ledger is not None
            and job_id is not None
            and not await asyncio.to_thread(ledger.is_batch_live, batch_id)
        ):
            return
        try:
            await self.provider.cancel_batch_job(batch_id)
        except Exception as exc:  # never mask the fallback
            logger.debug("Batch abandon cancel failed for %s: %s", batch_id, exc)
        if ledger is not None and job_id is not None:
            try:
                await asyncio.to_thread(ledger.update_batch_job_status, batch_id, "cancelled")
            except Exception as exc:  # bookkeeping only
                logger.debug("Batch abandon status update failed for %s: %s", batch_id, exc)

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
    ) -> str:
        """Route to repair model tier with strategy-driven prompt construction and extraction."""
        effort = reasoning_effort or self.repair_reasoning_effort

        # L4 Visual Scalpel precision repair: if high-res visual crop is available,
        # try multimodal repair first to resolve formula, subscript, or layout defects.
        repair_profile = self._get_profile(self.repair_model)
        # Prefer the declared capability (ModelProfile.supports_vision, which an
        # operator can set on any model via UBT_MODEL_PROFILES_*); fall back to
        # a named hint list for the common multimodal families that ship
        # without a vision flag. Kept out of the call site so it reads as data.
        model_is_vision = repair_profile.supports_vision or any(
            hint in self.repair_model.lower() for hint in _VISION_NAME_HINTS
        )
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
            except (asyncio.CancelledError, JobInterruptedError, BudgetExceededError):
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
            )

        raw_res, actual_model = await self._execute_with_retry(
            system_prompt="",
            user_prompt="",
            model=self.repair_model,
            temperature=temperature,
            reasoning_effort=effort,
            prompt_builder=_build_repair_prompts,
            return_model=True,
            is_repair=True,
        )
        actual_profile = self._get_profile(actual_model)
        return TranslationOutputExtractor.extract(
            raw_res,
            strategy=actual_profile.extraction_strategy,
        )
