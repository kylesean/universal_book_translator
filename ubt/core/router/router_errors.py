"""Provider-error taxonomy, retry/parse helpers, and the Batch API request types.

Split out of :mod:`ubt.core.router.router` (which still re-exports every name
here): these are the pure pieces of the router -- classifying a provider
failure, parsing ``Retry-After``, merging a continuation, the batch request
dataclasses -- that the draft stage and the transports consume directly. The
:class:`~ubt.core.router.router.ModelRouter` class stays where it is; this
module only removes the pure prefix that made every read of the router start
with 300 lines of taxonomy.
"""

from __future__ import annotations

import asyncio
import re
from contextlib import suppress
from dataclasses import dataclass

from ubt.core.exceptions import ModelProviderError

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
# Upper bound on a single ``Retry-After`` pause. A header may carry a far-future
# value (``86400`` = 24h, or an HTTP-date days out); honouring it verbatim makes
# every block in every worker sleep the rest of the day with no log line, and
# leaves the retry loop un-cancellable for that whole window. Clamp to a sane
# ceiling — a longer outage is better surfaced as a failure than ridden out by
# every concurrent worker.
_RETRY_AFTER_MAX_SECONDS = 300.0
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
    if isinstance(
        exc,
        (
            TypeError,
            AttributeError,
            NameError,
            UnboundLocalError,
            KeyError,
            IndexError,
            AssertionError,
        ),
    ):
        return ProviderErrorAction(False, reason="programming_defect")
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
    block_type: str = ""


@dataclass
class BatchDraftResult:
    """Outcome for one :class:`BatchDraftRequest` line."""

    custom_id: str
    text: str | None = None
    error: str | None = None
