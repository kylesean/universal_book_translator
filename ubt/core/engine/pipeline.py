"""End-to-end translation pipeline orchestrator (resources and run lifecycle).

This module owns a run's *resources*: the writer-lock lease, the ledger, the
adapter, the router and the per-run collaborators. :mod:`ubt.core.engine.plan`
owns *what runs in what order* -- it invokes 14 stages, from ingest through
export.

The "six-stage" label this docstring used to carry described an earlier shape of
the plan and outlived it by nine stages; it is removed rather than corrected to
another number, because the order belongs to ``plan.py`` and a count here can
only drift again.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import AsyncGenerator, Callable
from contextlib import aclosing, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ubt.cache.dirs import cache_root
from ubt.core.config import UBTConfig
from ubt.core.engine.blocks import BlockReader
from ubt.core.engine.events import EventType, TranslationProgressEvent
from ubt.core.engine.facts import RunFacts
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.plan import RunGates, run_stages
from ubt.core.engine.repair_loop import RepairLoop
from ubt.core.engine.services import RunServices
from ubt.core.engine.source_document import SourceDocumentIndex
from ubt.core.engine.stage_context import StageContext
from ubt.core.engine.stages import (
    run_tm_writeback_stage,
)
from ubt.core.engine.usage import JobBill, bill_job_run, budget_violation
from ubt.core.engine.writer_lock import LedgerWriterLock, keep_lease_fresh
from ubt.core.exceptions import (
    BudgetExceededError,
    DocumentParseError,
    JobInterruptedError,
    LedgerWriterLeaseLostError,
    UBTError,
    UnsupportedDocumentFormatError,
    public_job_error,
)
from ubt.core.fs_perms import warn_world_readable
from ubt.core.ir.models import BookManifest
from ubt.core.job_options import default_output_dir_for_scan
from ubt.core.memory.tm import (
    TranslationMemory,
)
from ubt.core.policy.adaptive_policy import (
    resolve_adaptive_policy,
)
from ubt.core.ports import (
    AdapterRuntimeConfig,
    DocumentAdapter,
    apply_runtime_config,
    resolve_adapter,
)
from ubt.core.qe.base import BaseQERunner
from ubt.core.qe.fast_pass import FastPassFilter
from ubt.core.qe.heuristic import HeuristicQERunner
from ubt.core.router.pricing import price_is_known
from ubt.core.router.provider import create_model_provider

if TYPE_CHECKING:
    from ubt.core.qe.llm_judge import JudgeFn
from ubt.core.router.rate_limiter import AdaptiveTokenBucket
from ubt.core.router.registry import registry_from_config
from ubt.core.router.router import ModelRouter, set_run_cancel_token
from ubt.core.router_mode import decide

logger = logging.getLogger(__name__)

# Ledger namespace for runs driven by a mock provider (``--dry-run``). See the
# suffix application in ``run``.
_MOCK_JOB_SUFFIX = "_mock"


def derive_job_id(
    *,
    doc_id: str,
    target_lang: str,
    pages: str | None,
    start_chapter: int,
    max_chapters: int | None,
    mock_run: bool = False,
    profile_name: str = "general",
    engine_signature: str = "",
    source_lang: str = "en",
) -> str:
    """Ledger job id derived from the run's full identity.

    The ledger *is* the resume state, and ingest skips parsing entirely whenever
    it already holds blocks — so anything that changes which part of a document
    a run covers must namespace the ledger. Otherwise a narrower run leaves a
    partial ledger behind, and the wider run resumes into it, drafts nothing
    new, exports the narrow subset and still reports "completed" (e.g.
    `--max-chapters 2` followed by a full-book run).

    Language and page selection already namespace it; the chapter window is
    included here for the same reason. The *default* window adds no suffix, so
    ledgers written without one stay resumable. ``profile_name`` (the genre
    profile) and ``engine_signature`` (the non-default preset/engine knobs, see
    :func:`engine_signature`) namespace it too: a run under a different profile
    or ``--preset`` drafts differently and must not resume the other's drafts.

    ``mock_run`` (a ``--dry-run`` / 演练模式 pass) namespaces it too: the mock
    provider's drafts are not translations, and without this a simulated run
    left ``mtqe_passed`` rows behind that the next real run resumed, exported
    and reported as success without a single API call.

    ``source_lang`` namespaces it only when it is NOT the default ``en``: a
    mistyped ``-s`` was persisted in ``job_meta`` but never compared on resume,
    so correcting it and re-running printed "Translation Completed Successfully"
    over the previous run's translations — every ``mtqe_passed`` row was
    exported untouched. The non-default suffix keeps every existing ledger
    resumable (they were all created with the default), while a corrected
    source language starts its own ledger instead of replaying the wrong one.
    """
    base = f"job_{doc_id[:12]}_{target_lang}"
    suffixes: list[str] = []
    if pages:
        suffixes.append(f"p{re.sub(r'[^A-Za-z0-9]', '_', pages.strip())}")
    if start_chapter != 1 or max_chapters is not None:
        window_end = "end" if max_chapters is None else str(start_chapter + max_chapters - 1)
        suffixes.append(f"c{start_chapter}-{window_end}")
    if mock_run:
        suffixes.append(_MOCK_JOB_SUFFIX.lstrip("_"))
    if profile_name and profile_name.strip() != "general":
        suffixes.append(f"pr{re.sub(r'[^A-Za-z0-9]', '', profile_name.strip())}")
    if engine_signature:
        suffixes.append(f"eng{re.sub(r'[^A-Za-z0-9]', '', engine_signature)}")
    normalized_source = (source_lang or "en").strip().lower().replace("_", "-")
    if normalized_source and normalized_source != "en":
        suffixes.append(f"src{re.sub(r'[^A-Za-z0-9]', '', normalized_source)}")
    return f"{base}_{'_'.join(suffixes)}" if suffixes else base


#: Engine knobs, besides the profile/target/chapter window, that change what a
#: run drafts. A preset (or an explicit flag) that moves any of them must get
#: its own ledger, or the resume would skip drafting under the new settings.
_ENGINE_SIGNATURE_FIELDS: tuple[tuple[str, str], ...] = (
    ("draft_model", "dm"),
    ("repair_model", "rm"),
    ("prompt_strategy", "ps"),
    ("exec_mode", "em"),
    ("formula_enrichment", "fe"),
)


def degraded_pdf_engine(config: Any, manifest_metadata: dict[str, Any] | None) -> str | None:
    """The engine that actually ran when the configured one could not.

    ``auto`` (and an explicit ``docling`` that the factory let through) selects
    docling; without the package the adapter silently falls back to the
    geometric pdfium reader. The manifest records whichever parser produced the
    blocks, so this reads it back. Returns ``None`` for a healthy run — a
    deliberate ``--pdf-engine pdfium`` is not a degradation and must not
    re-namespace existing ledgers.

    Both the derived job id and the resume identity guard call this: two
    copies of the rule would drift, and then the guard would refuse to resume
    the very ledger the id was derived from.
    """
    requested = str(getattr(config, "pdf_engine", "") or "")
    actual = str((manifest_metadata or {}).get("pdf_parser_engine", "") or "")
    if requested in ("auto", "docling") and actual and actual != "docling":
        return actual
    return None


def engine_signature(config: Any, *, pdf_engine_fallback: str | None = None) -> str:
    """Compact suffix for the engine knobs a config overrides from the default.

    The default config yields ``""`` (historical ledgers stay resumable); any
    ``--preset`` or single overridden knob (flag OR env) contributes a readable
    token, so a re-run under different draft settings cannot resume the wrong
    ledger. Compared against the declared field default, not ``UBTConfig()``:
    that constructor reads the environment itself and would hide an env-only
    override.

    ``pdf_engine_fallback`` names the engine that actually ran when the
    configured one could not (``auto`` → docling, docling missing → pdfium).
    It is namespaced only in that case: the degraded ledger holds no tables,
    figures or formulas, and a later run *with* docling installed must not
    resume it as though it were a full ingest. A normal run — including an
    explicit ``--pdf-engine pdfium`` — adds nothing, so existing ids stay
    resumable.
    """
    fields = getattr(type(config), "model_fields", {})
    parts: list[str] = []
    for fname, tag in _ENGINE_SIGNATURE_FIELDS:
        value = getattr(config, fname, None)
        field_info = fields.get(fname)
        default = getattr(field_info, "default", None)
        if value is not None and value != default:
            parts.append(f"{tag}{re.sub(r'[^A-Za-z0-9]', '', str(value))}")
    if pdf_engine_fallback:
        parts.append(f"fb{re.sub(r'[^A-Za-z0-9]', '', pdf_engine_fallback)}")
    return "-".join(parts)


async def _close_adapter_off_loop(adapter: Any, job_id: str) -> None:
    """Close the adapter off the event loop.

    Adapter close may wait on subprocess termination (``proc.wait(timeout=...)``);
    running it inline in run()'s finally
    stalls the loop shared with SSE fan-out for every connected client.
    """
    close_adapter = getattr(adapter, "close", None)
    if not callable(close_adapter):
        return
    try:
        await asyncio.to_thread(close_adapter)
    except Exception as adapter_exc:
        logger.debug("Error closing adapter for job %s: %s", job_id, adapter_exc)


async def _mark_failed_unless_completed(
    ledger: SQLiteJobLedger | None,
    job_id: str,
    status: str = "failed",
    *,
    error: str | None = None,
) -> None:
    """Finalize a job as failed/cancelled on abort, but never flip a completed job.

    Export finalizes ``completed`` before its terminal event is yielded, so a
    consumer that stops iterating right after EXPORT_COMPLETED (GeneratorExit)
    must not rewrite the completed status to failed.
    """
    if ledger is None:
        return
    try:
        current = await asyncio.to_thread(ledger.get_job_status, job_id)
        if current in ("completed", "cancelled"):
            return
        if error is not None:
            await asyncio.to_thread(ledger.set_job_metadata_value, job_id, "last_error", error)
        await asyncio.to_thread(ledger.finalize_job, job_id, status=status)
    except Exception as fin_exc:
        logger.warning("Failed to finalize aborted job %s in ledger: %s", job_id, fin_exc)


@dataclass
class _RunBillingSession:
    """Encapsulates per-job billing counters and locks for orchestrator re-entrancy."""

    sink: dict[str, dict[str, int]] | None = None
    baseline: dict[str, dict[str, int]] = field(default_factory=dict)
    billed_usage: dict[str, dict[str, int]] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass(frozen=True)
class _RunPreparation:
    """Everything one run needs resolved before the resource lifecycle starts."""

    adapter: DocumentAdapter
    manifest: BookManifest
    source_lang: str
    qe_runner: BaseQERunner
    repair_loop: RepairLoop
    actual_job_id: str
    run_session: _RunBillingSession
    #: The run's single FastPassFilter — the object the qe_runner was bound to
    #: and every stage reads, so the gate and the scorer cannot diverge.
    fast_pass: FastPassFilter


class PipelineOrchestrator:
    """Orchestrates the multi-stage book translation lifecycle.

    ``ubt.core.engine.plan.run_stages`` is the authoritative, gate-dependent
    stage list; do not duplicate the stage count here — it has gone stale twice.
    """

    def __init__(
        self,
        config: UBTConfig | None = None,
        router: ModelRouter | None = None,
        qe_runner: BaseQERunner | None = None,
        repair_loop: RepairLoop | None = None,
        adapter: DocumentAdapter | None = None,
        rate_limiter: AdaptiveTokenBucket | None = None,
        finalize_job: Callable[[TranslationProgressEvent, SQLiteJobLedger], None] | None = None,
    ) -> None:
        self.config = config or UBTConfig.from_env()
        self.config.bootstrap_runtime_environment()

        if router is None:
            if not self.config.draft_model:
                raise UBTError(
                    "No draft model configured: set UBT_DRAFT_MODEL, select a provider "
                    "(UBT_PROVIDER=...), or declare [providers.<name>]."
                )
            # A long-lived caller (API server, worker) that runs several jobs
            # passes one bucket in, because `rate_limit_rpm` is a per-credential
            # budget: one bucket per job multiplied it by the job count.
            rate_limiter = rate_limiter or AdaptiveTokenBucket(
                initial_rpm=self.config.rate_limit_rpm,
                initial_tpm=self.config.rate_limit_tpm,
                max_rpm=self.config.rate_limit_max_rpm,
                max_tpm=self.config.rate_limit_max_tpm,
                backoff_cooldown_sec=self.config.rate_limit_backoff_cooldown_sec,
            )
            provider = create_model_provider(
                api_key=self.config.api_key.get_secret_value(),
                base_url=self.config.base_url,
                default_model=self.config.draft_model,
                api_mode=self.config.api_mode,
                timeout=self.config.api_timeout,
                extra_headers=self.config.extra_headers,
                chat_template_kwargs=self.config.chat_template_kwargs,
                reasoning_dialect=self.config.reasoning_dialect,
                prompt_caching=self.config.prompt_caching_enabled,
            )
            repair_provider = provider
            if self.config.repair_provider and self.config.repair_provider != self.config.provider:
                from pydantic import SecretStr

                from ubt.core.providers import load_layer_config

                r_cfg = load_layer_config(self.config.repair_provider)
                raw_key = r_cfg.api_key
                r_key = (
                    raw_key.get_secret_value()
                    if isinstance(raw_key, SecretStr)
                    else str(raw_key or "")
                ) or self.config.api_key.get_secret_value()
                repair_provider = create_model_provider(
                    api_key=r_key,
                    base_url=str(r_cfg.base_url or self.config.base_url),
                    default_model=str(r_cfg.repair_model or self.config.repair_model),
                    api_mode=str(r_cfg.api_mode or self.config.api_mode),
                    timeout=float(
                        r_cfg.api_timeout
                        if r_cfg.api_timeout is not None
                        else self.config.api_timeout
                    ),
                    extra_headers=dict(r_cfg.extra_headers or self.config.extra_headers),
                    chat_template_kwargs=dict(
                        r_cfg.chat_template_kwargs or self.config.chat_template_kwargs
                    ),
                    reasoning_dialect=str(r_cfg.reasoning_dialect or self.config.reasoning_dialect),
                    prompt_caching=bool(
                        r_cfg.prompt_caching_enabled
                        if r_cfg.prompt_caching_enabled is not None
                        else self.config.prompt_caching_enabled
                    ),
                )
            self.router = ModelRouter(
                provider=provider,
                repair_provider=repair_provider,
                draft_model=self.config.draft_model,
                repair_model=self.config.repair_model,
                rate_limiter=rate_limiter,
                draft_reasoning_effort=self.config.draft_reasoning_effort,
                repair_reasoning_effort=self.config.repair_reasoning_effort,
                fallback_models=self.config.fallback_models,
                prompt_strategy_override=self.config.prompt_strategy,
                allow_page_upload=self.config.allow_page_upload,
                # Per-configuration, never the process singleton: this config's
                # capability declarations belong to this job. Registering them
                # globally let job A's override become job B's capability for the
                # same model id, and grew the registry without bound.
                registry=registry_from_config(self.config),
                request_timeout=self.config.api_timeout,
            )
        else:
            self.router = router

        # Completion hook the orchestrator invokes *inside* its writer-lock block
        # to ensure all post-completion writes remain under the single-writer guard.
        self.finalize_job = finalize_job

        #: Set by the lease-renewal task when another process reclaimed this
        #: run's writer lease. Read by :meth:`run` after the stage plan ends, so
        #: a stalled-then-reclaimed run reports why it stopped instead of
        #: finalizing itself as a success over the new owner's ledger.
        self._lease_lost_reason: str | None = None
        #: Event the detached renewal task sets on lease loss. Kept separate
        #: from the caller's ``cancel_token`` so a reclaim never reads as a user
        #: cancellation and — unlike the cancel token, which the REST embedding
        #: path never passes — is always wired.
        self._lease_lost_event: asyncio.Event | None = None

        # The per-run protective collaborators are owned elsewhere and handed in
        # per stage, so they are not reconstructed here: the HTML validator is a
        # ``RunServices`` field built by ``field(default_factory=...)``, and the
        # maskers belong to ``ubt.segment.PlaceholderEngine`` (the neighbor
        # builder to ``HierarchicalMemory``). The billing
        # state below IS per-run instance state: every entry point (CLI, API
        # manager, job worker) builds one orchestrator per job, so it is not
        # shared — sharing a single instance across concurrent jobs would
        # cross-contaminate usage accounting and budget enforcement.

        # NOTE: the orchestrator is built before the job's
        # languages are known, so we do NOT bake a FastPassFilter() here — its
        # no-arg form silently falls back to ZH thresholds. Every stage receives
        # the per-run filter (built in run() with source/target langs), and the
        # repair loop's fallback stays None rather than a wrong-language one.

        # The judge's own provider, when ``qe_judge_provider``/``qe_judge_base_url``
        # pointed it somewhere other than the translator. ``None`` means the main
        # router answers and there is nothing extra to close or attribute.
        self._judge_provider: Any = None
        self._judge_model: str | None = None
        self._judge_base_url: str | None = None

        if qe_runner is not None:
            self.qe_runner: BaseQERunner = qe_runner
        else:
            # The neural CometKiwi subprocess runner (CC-BY-NC-SA weights) was
            # retired for commercial-license compliance. The heuristic runner is
            # the zero-token Tier-1 default; a heavier judge is opt-in via
            # ``qe_engine=tiered`` (L3 gray-zone LLM rescoring) below.
            self.qe_runner = HeuristicQERunner()
        # L3 gray-zone judge: wrap heuristic with suspect-only LLM rescoring.
        # Explicit opt-in (UBT_QE_JUDGE_ENABLED); ``qe_engine=tiered`` alone
        # keeps the zero-token heuristic path, matching the documented opt-in.
        # RepairLoop shares the same runner so the closed-loop rescore after
        # repair uses the identical policy.
        if self.config.qe_judge_enabled and isinstance(self.qe_runner, HeuristicQERunner):
            from ubt.core.qe.llm_judge import LLMJudgeQERunner, TieredQERunner

            judge_fn, judge_model, judge_base_url, judge_provider = self._build_judge_fn(
                self.config.qe_judge_model or self.config.repair_model
            )
            # Keep the provider and its endpoint on the orchestrator: the run's
            # finally closes the pool, and billing attributes the judge's tokens
            # to the endpoint they were actually spent at. Both were impossible
            # while the provider lived only inside the closure.
            self._judge_provider = judge_provider
            self._judge_model = judge_model
            self._judge_base_url = judge_base_url
            judge = LLMJudgeQERunner(
                judge_fn=judge_fn,
                model=judge_model,
                compare_sentences=self.config.qe_judge_sentences,
            )
            self.qe_runner = TieredQERunner(
                heuristic=self.qe_runner,
                judge=judge,
                pass_sample=self.config.qe_judge_pass_sample,
            )
        self.repair_loop: RepairLoop = repair_loop or RepairLoop(
            router=self.router,
            qe_runner=self.qe_runner,
            max_rounds=self.config.max_repair_rounds,
            qe_threshold=self.config.qe_threshold,
        )
        # Explicitly injected collaborators are kept verbatim; the default ones
        # are language-agnostic at construction time and get a per-run language
        # binding in run() (see _runtime_qe_runner / _runtime_repair_loop).
        self._qe_runner_explicit = qe_runner is not None
        self._owns_qe_runner = qe_runner is None
        self._repair_loop_explicit = repair_loop is not None
        self.custom_adapter = adapter
        # Only close a router we constructed ourselves. An injected/shared
        # router (API JobManager) would otherwise be closed by whichever job
        # finishes first, killing in-flight requests of concurrent jobs.
        self._owns_router = router is None
        # Per-run billing sessions keyed by job_id for concurrent multi-job safety.
        self._billing_sessions: dict[str, _RunBillingSession] = {}

    def _get_billing_session(self, job_id: str) -> _RunBillingSession:
        """The run's billing session. Every entry point builds one orchestrator
        per job, so a job_id always has at most one live session here."""
        try:
            return self._billing_sessions[job_id]
        except KeyError:
            raise KeyError(f"No active billing session for job '{job_id}'") from None

    @property
    def _is_mock_run(self) -> bool:
        """Whether the injected provider produces simulated text.

        Detected on the provider, not the config: ``--dry-run`` keeps a real
        ``api_key`` in config and only swaps the router's provider.
        """
        return self.router.provider.is_mock

    def _preflight_budget_pricing(self) -> None:
        """Refuse to start a capped run whose spend cannot be measured.

        ``budget_violation`` can only fire when the price table yields a cost;
        with an unpriced model every cost is None, which the cap reads as
        "never exceeded" -- the one control the user asked for, silently
        inert. The warning-once behavior stays for UNCAPPED runs; a run that
        sets a budget gets a startup refusal instead (UBT_ALLOW_UNPRICED_BUDGET=1
        downgrades it back to the warning deliberately).

        Self-hosted endpoints are exempt (measurable at $0): the exemption is
        per model (``UBTConfig.remote_billing_models``), so a local LLM with
        cloud OCR is still capped on the OCR channel.
        """
        if self.config.budget_usd is None or self.config.allow_unpriced_budget or self._is_mock_run:
            return
        # Every model the run can bill through -- not just the two headline
        # ones. A fallback entry is reached after a 429/5xx retry
        # (``router._execute_with_retry``), and the L3 judge is enabled by
        # ``qe_judge_enabled`` (see the runner wiring above), so checking only
        # ``qe_engine == "tiered"`` would price a judge that is explicitly off
        # and refuse a zero-token tiered run unnecessarily.
        models: list[str | None] = [
            self.router.draft_model,
            self.router.repair_model,
            *self.router.fallback_models,
        ]
        if self.config.visual_judge_enabled:
            models.append(self.config.visual_judge_model)
        # The OCR channel bills a model of its own through its own httpx client
        # (see cloud_driver). Leaving it out of this enumeration let a capped
        # run select an unpriced vision model, pass pre-flight, and then spend
        # on a channel the cap could not see at all. Which channels bill
        # elsewhere (and whether 'auto' may reach a cloud) is one rule now, in
        # ``UBTConfig.remote_billing_models``.
        endpoint_map = self.config.remote_billing_models()
        if endpoint_map:
            models.append(self.config.ocr_model)
        # Once wrapped, ``self.qe_runner`` *is* a ``TieredQERunner``; testing
        # the pre-wrap ``HeuristicQERunner`` type never matched, leaving the
        # documented ``UBT_QE_JUDGE_ENABLED=true`` path unpriced.
        from ubt.core.qe.llm_judge import TieredQERunner

        if isinstance(self.qe_runner, TieredQERunner):
            # The judge's own resolved model name, not the raw config value: a
            # ``[providers.<qe_judge_provider>]`` block carries ``repair_model``,
            # and checking only ``qe_judge_model or repair_model`` priced a
            # different name than the one the run would actually bill. A local
            # judge reached that way also stays exempt, which is correct — the
            # endpoint, not the name, decides that.
            models.append(
                self._judge_model or self.config.qe_judge_model or self.config.repair_model
            )
        # The judge's endpoint joins the map so a self-hosted translator with a
        # cloud judge is capped on the judge channel, and a cloud translator
        # with a local judge is not refused for an unpriced local model.
        endpoint_map = {**endpoint_map, **self._judge_endpoint_map()}
        unpriced = sorted(
            m
            for m in models
            if m and not price_is_known(m, base_url=self.config.base_url, endpoint_map=endpoint_map)
        )
        if unpriced:
            raise UBTError(
                f"--budget-usd/UBT_BUDGET_USD is set but these models have no "
                f"known price: {', '.join(unpriced)}. Their spend would "
                "report as unknown and the cap could never fire. Add the rate to "
                "ubt/resources/prices.toml (legacy: MODEL_PRICES_USD_PER_MTOK; see "
                "`ubt doctor`), switch to a priced model, or set "
                "UBT_ALLOW_UNPRICED_BUDGET=1 to run uncapped-by-warning anyway. "
                "Models reached through a self-hosted endpoint are exempt (they cost $0)."
            )

    def _judge_endpoint_map(self) -> dict[str, str]:
        """The judge model's billing endpoint, when it differs from the primary.

        Empty for the common single-endpoint setup (the judge answers through
        the main router), so nothing is overridden there.
        """
        if self._judge_base_url and self._judge_model:
            return {self._judge_model: self._judge_base_url}
        return {}

    def _build_judge_fn(self, judge_model: str) -> tuple[JudgeFn, str, str | None, Any]:
        """The judge's completion call, resolved in a standard way.

        Three sources, in this order — each one something a user already knows
        how to write:

        1. ``qe_judge_provider`` — the name of a ``[providers.<name>]`` block.
           Preferred, because it reuses the registry's own credential handling
           (``${VAR}`` expansion, ``UBT_LLM_API_KEY``), wire-protocol selection
           and header rules instead of duplicating them here.
        2. explicit ``qe_judge_base_url`` / ``qe_judge_api_key`` /
           ``qe_judge_api_mode`` — for a judge endpoint that is not declared as
           a provider (one-off, or a key that only lives in the environment).
        3. nothing set — the main router answers, so an existing single-endpoint
           configuration keeps working unchanged.

        The judge may run somewhere other than the translator on purpose: a model
        large enough to notice a dropped noun is slow locally (measured 35-85 s
        per block on a 27B) and quick hosted (1.6-3.5 s for the same decisions).

        Returns the callable, **the model name it resolves to**, **the endpoint
        it bills through** (``None`` when the main router answers), and **the
        provider it owns** (also ``None`` for the router case). The model name
        matters because a provider block carries its own: returning only the
        callable left the caller passing the translation provider's
        ``repair_model`` — a name the judge's endpoint has never heard of.
        Measured: a local name (``gemma4-e4b``) sent to a different gateway
        answered ``404 no such model`` on every judge call, twelve times in a
        run, and the report read "not machine-scored" with no indication why.

        The endpoint and the provider are returned for the two things the
        caller must do with them and could not before: attribute the judge's
        tokens to *its* endpoint (a local translator with a cloud judge was
        billed $0.00, and a cloud translator with a local judge reported an
        unknown cost that made ``--budget-usd`` inert) and close the HTTP pool
        the closure owns (it was created per run and never released, so a
        long-lived API/worker leaked one ``httpx.AsyncClient`` per judged run).
        """
        if self.config.qe_judge_provider:
            from pydantic import SecretStr

            from ubt.core.providers import load_layer_config
            from ubt.core.router.provider import create_model_provider

            judge_cfg = load_layer_config(self.config.qe_judge_provider)
            raw_key = judge_cfg.api_key
            j_key = (
                raw_key.get_secret_value() if isinstance(raw_key, SecretStr) else str(raw_key or "")
            ) or self.config.api_key.get_secret_value()
            judge_base_url = str(judge_cfg.base_url or self.config.base_url)
            judge_provider = create_model_provider(
                api_key=j_key,
                base_url=judge_base_url,
                default_model=str(judge_cfg.repair_model or judge_model),
                api_mode=str(judge_cfg.api_mode or self.config.api_mode),
                timeout=float(
                    judge_cfg.api_timeout
                    if judge_cfg.api_timeout is not None
                    else self.config.api_timeout
                ),
                extra_headers=dict(judge_cfg.extra_headers or {}),
            )
            judge_model = str(judge_cfg.repair_model or judge_model)
        elif self.config.qe_judge_base_url:
            from ubt.core.router.provider import create_model_provider

            judge_base_url = self.config.qe_judge_base_url
            judge_provider = create_model_provider(
                api_key=self.config.qe_judge_api_key or self.config.api_key,
                base_url=judge_base_url,
                default_model=judge_model,
                # ApiMode is a Literal of wire-protocol strings, not an enum.
                api_mode=str(self.config.qe_judge_api_mode or self.config.api_mode),
                timeout=self.config.api_timeout,
                # Headers belong to the endpoint they were configured for; reuse
                # them only when the judge shares the translator's endpoint.
                extra_headers=(
                    dict(self.config.extra_headers)
                    if self.config.qe_judge_base_url == self.config.base_url
                    else {}
                ),
            )
        else:
            return self.router.complete_raw, judge_model, None, None

        async def _judge_complete(
            *,
            system_prompt: str,
            user_prompt: str,
            temperature: float = 0.0,
            model: str | None = None,
        ) -> str:
            return await judge_provider.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                model=model or judge_model,
                temperature=temperature,
            )

        return _judge_complete, judge_model, judge_base_url, judge_provider

    def _runtime_qe_runner(
        self, source_lang: str, target_lang: str, *, fast_pass: FastPassFilter | None = None
    ) -> BaseQERunner:
        """Bind the run's language pair — and its single filter — for the QE runner.

        Injected runners pass through ``with_languages`` too: every runner's
        implementation returns the same instance when the pair already
        matches, and a fresh/rebound instance otherwise — so a shared runner
        (the API's one instance, concurrent jobs) never scores another job
        with this job's language-pair-dependent thresholds, and an injected
        runner no longer keeps its construction-time en→zh
        state for every pair.

        ``fast_pass`` is the run's filter (created by the caller with the same
        pair): the runner scores through the very object the draft stage binds
        the glossary and document context onto, so the inner/outer gate split
        cannot come back. Bind-after-languages is the required order — the
        handed filter already carries the pair.
        """
        # Fresh run, fresh residency attempt (the API shares one runner).
        if self._qe_runner_explicit:
            self.qe_runner.reset_residency()
        runner = self.qe_runner.with_languages(source_lang=source_lang, target_lang=target_lang)
        bind = getattr(runner, "with_fast_pass", None)
        if fast_pass is not None and callable(bind):
            runner = bind(fast_pass)
        return runner

    def _runtime_repair_loop(self, qe_runner: BaseQERunner) -> RepairLoop:
        """Build the per-run repair loop so no language-bound state is shared."""
        if self._repair_loop_explicit:
            return self.repair_loop
        return RepairLoop(
            router=self.router,
            qe_runner=qe_runner,
            max_rounds=self.config.max_repair_rounds,
            qe_threshold=self.config.qe_threshold,
        )

    @staticmethod
    def _usage_delta(
        baseline: dict[str, dict[str, int]], current: dict[str, dict[str, int]]
    ) -> dict[str, dict[str, int]]:
        """Per-model token delta between two provider usage snapshots.

        An empty baseline (no snapshot taken) returns ``current`` unchanged.
        """
        if not baseline:
            return current
        delta: dict[str, dict[str, int]] = {}
        for model, totals in current.items():
            base = baseline.get(model, {})
            diff = {key: value - base.get(key, 0) for key, value in totals.items()}
            if any(value != 0 for value in diff.values()):
                delta[model] = diff
        return delta

    def _session_usage(self, session: _RunBillingSession) -> dict[str, dict[str, int]]:
        """Tokens this run spent, per model.

        Prefer the provider's per-run sink: it is exact even when several jobs
        share one provider. Only when the provider cannot attribute falls back
        to diffing the process-wide counters against the start-of-run snapshot
        (correct for a solo CLI run, blended for a shared one).
        """
        if session.sink is not None:
            # Copy: the provider keeps accumulating into this dict concurrently.
            return {model: dict(totals) for model, totals in list(session.sink.items())}
        return self._usage_delta(session.baseline, self.router.usage_totals_by_model())

    async def _bill_session(
        self,
        session: _RunBillingSession,
        job_id: str,
        ledger: SQLiteJobLedger,
        *,
        raise_on_budget: bool,
    ) -> JobBill:
        async with session.lock:
            run_usage = self._session_usage(session)
            newly_spent = self._usage_delta(session.billed_usage, run_usage)
            bill = await bill_job_run(
                ledger,
                job_id,
                run_usage,
                newly_spent=newly_spent,
                base_url=self.config.base_url,
                # Merge the OCR-channel map with the router's fallback attribution
                # so a model served only by the local fallback is billed at the
                # fallback endpoint, not the primary — plus the judge's own
                # endpoint when it was configured separately. Without the last
                # one a mixed deployment priced the judge's tokens at the
                # translator's URL: local translator + cloud judge reported
                # $0.00, and cloud translator + local judge reported "unknown",
                # which made --budget-usd inert.
                endpoint_map={
                    **self.config.remote_billing_models(),
                    **self.router.billing_endpoint_map(),
                    **self._judge_endpoint_map(),
                },
            )
            session.billed_usage = run_usage
            if raise_on_budget:
                violation = budget_violation(bill, self.config.budget_usd, job_id)
                if violation is not None:
                    raise BudgetExceededError(violation)
        return bill

    async def _bill_run(
        self, job_id: str, ledger: SQLiteJobLedger, *, raise_on_budget: bool
    ) -> JobBill:
        """Fold this run's usage into the job's absolute ledger figure once.

        Shared by the progress-event path and the export-stage hook that persists
        post-visual-gate spend before the report reads the ledger. The budget only
        stops the job at the progress-event call site; the export event is let
        through so the artifact and report still say what the job spent.
        """
        session = self._get_billing_session(job_id)
        return await self._bill_session(session, job_id, ledger, raise_on_budget=raise_on_budget)

    async def _create_progress_event(
        self,
        event_type: EventType,
        job_id: str,
        ledger: SQLiteJobLedger,
        message: str = "",
        active_block_id: str | None = None,
        artifact_path: str | None = None,
    ) -> TranslationProgressEvent:
        """Query real-time statistics from ledger and build progress event.

        the stats query (including an ``ORDER BY mtqe_score`` over all
        scored blocks) is synchronous SQLite; it must run in a worker thread, not
        on the event loop where every SSE subscriber and concurrent job shares
        the same thread. All stage call sites therefore ``await`` this method.
        """
        stats = await asyncio.to_thread(ledger.get_job_stats, job_id)
        total = int(stats.get("total", 0))
        skipped = int(stats.get("skipped", 0))
        completed = int(stats.get("completed", 0))
        drafted = int(stats.get("drafted", 0))
        repaired = int(stats.get("repaired", 0))
        failed = int(stats.get("failed", 0))
        repair_pending = int(stats.get("repair_pending", 0))
        needs_human = int(stats.get("needs_human", 0))
        blocked_human = int(stats.get("blocked_human", 0))
        avg_qe = float(stats.get("avg_qe_score", 0.0))
        bottom_15_avg = float(stats.get("bottom_15_avg_qe", 0.0))

        # Real per-model token cost from the provider's usage
        # accounting (None when the model has no price-table entry or made calls
        # whose response carried no usage block — both are "unknown", never a
        # fake 0.0). This run's own spend, never a shared provider's
        # lifetime totals, folded into the bill the job carries across resumes.
        # Only the part not billed by an earlier event is written: ``run_usage``
        # is cumulative, so passing the whole of it to an absolute write would
        # re-add every previous event's tokens.
        bill = await self._bill_run(
            job_id, ledger, raise_on_budget=event_type is not EventType.EXPORT_COMPLETED
        )

        return TranslationProgressEvent(
            event_type=event_type,
            job_id=job_id,
            total_blocks=total,
            skipped_blocks=skipped,
            completed_blocks=completed,
            drafted_blocks=drafted,
            repair_pending_blocks=repair_pending,
            repaired_blocks=repaired,
            failed_blocks=failed,
            needs_human_blocks=needs_human,
            blocked_human_blocks=blocked_human,
            current_avg_qe=avg_qe,
            bottom_15_avg_qe=bottom_15_avg,
            estimated_cost_usd=bill.cost_usd,
            cache_hit_rate=bill.cache_hit_rate,
            message=message,
            active_block_id=active_block_id,
            artifact_path=artifact_path,
        )

    async def _renew_writer_lock(self, writer_lock: LedgerWriterLock) -> None:
        """Keep the writer-lock lease fresh for the life of the run.

        Delegates to :func:`ubt.core.engine.writer_lock.keep_lease_fresh` — the
        one renewal implementation, shared with ``re_export_job`` so the two
        cannot drift about the renewal interval or the lost-lease rule.

        A *lost* lease is different from a transient renewal failure and must
        stop the run. The renewal task is detached from the stage generator, so
        there is no return value to inspect: it records the loss on the
        orchestrator and sets the run's dedicated ``lease_lost`` event, which
        every stage checks at its next boundary through
        ``StageContext.check_cancelled``. Continuing would put two writers on one
        ledger (both paying the provider, both overwriting checkpoints), and the
        work already paid for is durable — the reclaiming process resumes from it.

        The signal is *not* the cancel token: a lost lease must not read as a user
        cancellation, and the REST embedding path passes no ``cancel_token`` at
        all (only the queue worker does), so a token-based signal fired on some
        paths and did nothing on others.
        """

        def _record_loss(reason: str) -> None:
            self._lease_lost_reason = reason
            if self._lease_lost_event is not None:
                self._lease_lost_event.set()

        await keep_lease_fresh(writer_lock, on_lost=_record_loss)

    async def _run_finalize_hook(
        self, event: TranslationProgressEvent, ledger: SQLiteJobLedger
    ) -> None:
        """Best-effort post-finalize hook; a failure is logged, never fatal.

        The hook persists the artifact/report pointers callers download through,
        so a silent failure would strand delivered artifacts with no signal —
        unlike the abort-finalize path this is the only terminal hook that had
        no log at all.

        The run's own ``ledger`` is handed over rather than a path: the hook used
        to reopen the file by path, which opened a *second* connection to the
        same database outside the writer lock. Under contention that surfaced as
        ``database is locked``, and it defeated the connection's WAL/busy-timeout
        configuration by building a fresh one. The caller passes the live ledger
        so the write shares the run's single connection.
        """
        if self.finalize_job is None:
            return
        try:
            # Off-loop: the hook writes metadata to SQLite.
            await asyncio.to_thread(self.finalize_job, event, ledger)
        except Exception as exc:
            logger.warning("finalize_job hook failed for job %s: %s", event.job_id, exc)

    async def _prepare_run(
        self,
        input_path: Path,
        output_path: Path | None,
        target_lang: str,
        profile_name: str,
        job_id: str | None,
        start_chapter: int,
        max_chapters: int | None,
        source_lang: str,
    ) -> _RunPreparation:
        """Resolve the run's pre-flight: disclosures, adapter contract, manifest, billing.

        Kept out of :meth:`run` so that method is the resource lifecycle alone
        (writer lock, ledger, stage plan, teardown) rather than pre-flight plus
        lifecycle in one body.
        """
        self._preflight_budget_pricing()
        # Disclose, once per run, whether book page images may leave this
        # machine (visual repair / cloud OCR / VLM judge) — this must never
        # happen silently.
        logger.info(
            "Page-image egress policy: allow_page_upload=%s (ocr_mode=%s, "
            "visual_judge=%s). UBT_ALLOW_PAGE_UPLOAD=false keeps every page image local.",
            self.config.allow_page_upload,
            self.config.ocr_mode,
            self.config.visual_judge_enabled,
        )
        # The local half of the same disclosure: ledgers store the full source
        # and target text, and files created before UBT restricted its own
        # artifacts are still group/other-readable. New runs cannot fix them
        # silently (that would be an unasked-for chmod), so say it out loud.
        # Runs off-loop: the recursive scan would otherwise stall heartbeats.
        scan_dirs = [cache_root() / "docling_cache", default_output_dir_for_scan()]
        if output_path is not None:
            out_p = Path(output_path)
            scan_dirs.append(out_p if out_p.is_dir() else out_p.parent)
        await asyncio.to_thread(warn_world_readable, self.config.db_dir, scan_dirs)
        # Bill this run alone: take a provider-side sink when the provider can
        # attribute per run, otherwise a snapshot to diff the shared counters in.
        sink = self.router.begin_usage_sink()
        baseline = {} if sink is not None else self.router.usage_totals_by_model()
        run_session = _RunBillingSession(sink=sink, baseline=baseline, billed_usage={})
        if self.config.qe_engine == "tiered" and not self.config.qe_judge_enabled:
            logger.warning(
                "qe_engine='tiered' selected without UBT_QE_JUDGE_ENABLED=true: "
                "every pair keeps its heuristic score and no LLM judge runs — "
                "'tiered' is then identical to the zero-token default. "
                "Set UBT_QE_JUDGE_ENABLED=true to enable TieredQERunner."
            )

        adapter = self.custom_adapter or resolve_adapter(
            input_path, pdf_engine=self.config.pdf_engine
        )
        # Everything below can refuse the run (a contradicting ``--output``
        # suffix, a manifest that cannot be read), and an adapter may own
        # subprocesses. ``run()``'s ``finally`` cannot close it because it never
        # received the object, so a rejected request in a long-lived API/worker
        # leaked one Docling/PDF child each. Close on any failure here.
        try:
            return await self._prepare_run_with_adapter(
                adapter,
                input_path=input_path,
                output_path=output_path,
                target_lang=target_lang,
                profile_name=profile_name,
                job_id=job_id,
                start_chapter=start_chapter,
                max_chapters=max_chapters,
                source_lang=source_lang,
                run_session=run_session,
            )
        except BaseException:
            if self.custom_adapter is None:
                await _close_adapter_off_loop(adapter, "prepare-failed")
            raise

    async def _prepare_run_with_adapter(
        self,
        adapter: DocumentAdapter,
        *,
        input_path: Path,
        output_path: Path | None,
        target_lang: str,
        profile_name: str,
        job_id: str | None,
        start_chapter: int,
        max_chapters: int | None,
        source_lang: str,
        run_session: _RunBillingSession,
    ) -> _RunPreparation:
        """The rest of ``_prepare_run`` once the adapter exists (see the caller)."""
        # The adapter decides the artifact format, so a contradicting ``--output``
        # is refused here -- before a single token is spent. A Markdown adapter
        # must never write Markdown into ``book.pdf`` and still report success.
        # A suffix-less name stays allowed: the caller just gets the format
        # without saying so.
        allowed_suffixes: frozenset[str] = getattr(type(adapter), "output_suffixes", frozenset())
        if allowed_suffixes and output_path is not None:
            out_suffix = output_path.suffix.lower()
            if out_suffix and out_suffix not in allowed_suffixes:
                wanted = "/".join(sorted(allowed_suffixes))
                raise UnsupportedDocumentFormatError(
                    f"{type(adapter).__name__} writes {wanted}, so --output "
                    f"{output_path.name} would not be a readable "
                    f"{out_suffix.lstrip('.')}. Drop --output (the default keeps the "
                    f"input format) or choose one of {wanted}."
                )
        # Hand the engine-level runtime knobs to the adapter through its SPI
        # hook: the pipeline resolves the config once and the adapter that owns
        # each field (the Docling PDF family) applies the subset it consumes, so
        # no per-field branch is added here when a config field grows.
        apply_runtime_config(
            adapter,
            AdapterRuntimeConfig(
                ocr_mode=self.config.ocr_mode,
                ocr_endpoint=self.config.ocr_endpoint,
                ocr_api_key=self.config.ocr_api_key.get_secret_value(),
                ocr_model=self.config.ocr_model,
                formula_enrichment=self.config.formula_enrichment,
                docling_device=self.config.docling_device,
                formula_crop_margin=self.config.formula_crop_margin,
                toc_mode=self.config.toc_mode,
                font_family=self.config.font_family,
                allow_page_upload=self.config.allow_page_upload,
                cache_dir=str(self.config.cache_dir) if self.config.cache_enabled else "",
                docling_isolate_process=self.config.docling_isolate_process,
                enable_layout_auto_heal=self.config.enable_layout_auto_heal,
                layout_auto_heal_max_passes=self.config.layout_auto_heal_max_passes,
                delivery_mode=self.config.delivery_mode,
                enable_pdf_optimization=self.config.enable_pdf_optimization,
                pdf_optimize_profile=self.config.pdf_optimize_profile,
            ),
        )

        manifest = await adapter.extract_manifest(input_path)
        if source_lang:
            manifest.source_lang = source_lang
        else:
            source_lang = getattr(manifest, "source_lang", "en") or "en"
        # Persist the requested target language with the manifest so the job
        # ledger records it (resume language guard in the ingest stage).
        manifest.target_lang = target_lang

        # The run's single FastPassFilter, created with the run's pair and
        # handed to every consumer: the draft stage binds the glossary and the
        # document context onto it, the repair/triage/consistency re-checks
        # evaluate through it, and the QE runner scores through the same
        # object (``_runtime_qe_runner``). One filter, one owner — the
        # inner/outer gate split that required hand-carried rebinds is gone.
        runtime_fast_pass = FastPassFilter(source_lang=source_lang, target_lang=target_lang)
        runtime_qe_runner = self._runtime_qe_runner(
            source_lang, target_lang, fast_pass=runtime_fast_pass
        )
        runtime_repair_loop = self._runtime_repair_loop(runtime_qe_runner)

        # What the run actually ingested with, when that is not what it asked
        # for. The manifest records the parser that produced these blocks, so
        # this reads the real answer instead of re-deriving the fallback.
        # ``oxide_fallback`` is the docling adapter's own degraded path — the
        # one that silently drops tables/figures/formulas.
        degraded_engine = degraded_pdf_engine(self.config, manifest.metadata)

        if job_id:
            # A caller-chosen id is the caller's to name: ``ubt status <id>`` and
            # the API's job contract must find exactly what was asked for.
            actual_job_id = job_id
        else:
            actual_job_id = derive_job_id(
                doc_id=manifest.doc_id,
                target_lang=target_lang,
                pages=self.config.pages,
                start_chapter=start_chapter,
                max_chapters=max_chapters,
                mock_run=self._is_mock_run,
                profile_name=profile_name,
                engine_signature=engine_signature(self.config, pdf_engine_fallback=degraded_engine),
                source_lang=source_lang,
            )
        self._billing_sessions[actual_job_id] = run_session
        return _RunPreparation(
            adapter=adapter,
            manifest=manifest,
            source_lang=source_lang,
            qe_runner=runtime_qe_runner,
            repair_loop=runtime_repair_loop,
            actual_job_id=actual_job_id,
            run_session=run_session,
            fast_pass=runtime_fast_pass,
        )

    async def run(
        self,
        input_path: Path,
        output_path: Path | None = None,
        target_lang: str = "zh",
        profile_name: str = "general",
        job_id: str | None = None,
        start_chapter: int = 1,
        max_chapters: int | None = None,
        source_lang: str = "",
        cancel_token: asyncio.Event | None = None,
    ) -> AsyncGenerator[TranslationProgressEvent, None]:
        """Execute the staged translation pipeline, yielding real-time progress events."""
        if not input_path.exists():
            raise DocumentParseError(f"Input document does not exist: {input_path}")
        prep = await self._prepare_run(
            input_path=input_path,
            output_path=output_path,
            target_lang=target_lang,
            profile_name=profile_name,
            job_id=job_id,
            start_chapter=start_chapter,
            max_chapters=max_chapters,
            source_lang=source_lang,
        )
        adapter = prep.adapter
        manifest = prep.manifest
        source_lang = prep.source_lang
        runtime_qe_runner = prep.qe_runner
        runtime_repair_loop = prep.repair_loop
        runtime_fast_pass = prep.fast_pass
        actual_job_id = prep.actual_job_id
        run_session = prep.run_session
        # The renewal task signals a lost lease through its own event; reset the
        # per-run flags so a reused orchestrator cannot inherit a previous run's
        # verdict. A fresh Event per run is required: the old one may already be
        # set from a previous run whose lease was reclaimed.
        self._lease_lost_event = asyncio.Event()
        self._lease_lost_reason = None
        # Make a provider Retry-After backoff cancellable: the router sleeps the
        # clamped header, and a job cancelled mid-wait or whose lease was reclaimed
        # must stop there rather than pin the worker to the rest of the pause.
        # The tokens ride ContextVars (per asyncio task), NOT router fields.
        set_run_cancel_token(cancel_token, lease_lost=self._lease_lost_event)
        ledger: SQLiteJobLedger | None = None
        tm: TranslationMemory | None = None
        writer_lock: LedgerWriterLock | None = None

        try:
            # Adaptive short/long chain router (heuristic features per carrier;
            # see ubt.core.router_mode.decide).
            route_decision = decide(
                input_path,
                short_max_pages=self.config.short_max_pages,
                exec_mode=self.config.exec_mode,
            )
            short_chain = route_decision.mode == "short"
            if self.config.exec_mode == "short" and not short_chain:
                raise DocumentParseError(
                    f"exec_mode='short' forced but {input_path.name} routed "
                    f"{route_decision.mode} ({route_decision.reason}) — refusing to "
                    "burn chapter-scale context on a long book; use --mode auto/long"
                )
            manifest.run.route_decision = route_decision.to_dict()
            manifest.run.route_mode = route_decision.mode
            # Explicit ``--columns`` override (a run decision, so it rides on
            # ``run``). ``None`` keeps the default: align the delivered layout to
            # the source's own column count, read from the route decision above.
            manifest.run.layout_columns = self.config.layout_columns
            manifest.run.config_snapshot = {
                "draft_model": self.config.draft_model,
                "repair_model": self.config.repair_model,
                "qe_engine": self.config.qe_engine,
                # Without this the report can say "nothing was scored" but not
                # *why*, and the console could only offer the generic "QE not
                # measured". The judge flag is the whole answer: the heuristic
                # engine emits discrete defect bands (never a continuous score)
                # unless the LLM judge is switched on, and that is a separate
                # opt-in (UBT_QE_JUDGE_ENABLED).
                "qe_judge_enabled": self.config.qe_judge_enabled,
                "qe_threshold": self.config.qe_threshold,
                "max_repair_rounds": self.config.max_repair_rounds,
                "prompt_strategy": self.config.prompt_strategy,
            }
            if short_chain:
                logger.info(
                    "Short chain for job %s: %s",
                    actual_job_id,
                    route_decision.reason,
                )

            # Shared translation memory across jobs. Created lazily per run inside
            # the guarded block so construction failures are handled cleanly.
            db_path = self.config.db_dir / f"{actual_job_id}.sqlite"
            # Single-writer guard, taken *before* the failure-marking try/finally:
            # a losing second writer must not write job state into the winner's
            # ledger (the engine claims no per-block leases; this lock is the only
            # cross-process job mutual exclusion).
            writer_lock = LedgerWriterLock(db_path, actual_job_id)
            # Off-loop, like every other sqlite call: a losing claimant parks on
            # the lock's busy_timeout (10 s) and a fresh ledger runs its
            # migrations, so taking either inline stalls every slot's loop —
            # the missed-heartbeat path that reclaims a still-running job and
            # bills the book twice (see job_worker's ``_q``).
            await asyncio.to_thread(writer_lock.acquire)

            # Staged execution lifecycle with guaranteed resource cleanup and failure handling.
            ledger = await asyncio.to_thread(
                SQLiteJobLedger, db_path, fresh=bool(self.config.fresh)
            )
            if self.config.tm_enabled:
                self.config.db_dir.mkdir(parents=True, exist_ok=True)
                tm = TranslationMemory(self.config.db_dir / "tm.sqlite")

            # -----------------------------------------------------------------
            # Adaptive Policy Resolution (single pipeline)
            # Granularity is always micro; the render engine stays a separate
            # decision (see adaptive_policy).
            # -----------------------------------------------------------------
            adaptive_policy = resolve_adaptive_policy(
                manifest=manifest,
                route_decision=route_decision,
                config=self.config,
            )
            manifest.run.adaptive_policy = adaptive_policy.to_dict()

            # One object carries the run's shared state into each stage, so a
            # stage's signature says what is *its own* (the visual-gate knobs,
            # the repair-round cap) rather than restating ledger, job id,
            # languages, event factory and semaphore every time. The per-run
            # language-bound collaborators are built here rather than at the
            # stage that uses them so the whole set is visible in one place.
            concurrency_sem = asyncio.Semaphore(self.config.max_concurrency)
            # The per-run services the stages read (document compiler architecture): built here, where
            # the whole set is visible in one place, and handed to the plan
            # instead of being fields on the shared context. ``fast_pass`` is
            # the run's single filter, created above next to the runner it is
            # bound to.
            services = RunServices(
                fast_pass=runtime_fast_pass,
                qe_runner=runtime_qe_runner,
                repair_loop=runtime_repair_loop,
                adaptive_policy=adaptive_policy,
                concurrency_sem=concurrency_sem,
                tm=tm,
                # The book's source text is immutable after ingest, so it is read
                # and indexed once per run and shared by the draft stage and the
                # quality gate (see SourceDocumentIndex). Loaded lazily, after
                # ingest has produced the blocks.
                source_document=SourceDocumentIndex(),
                qe_runner_explicit=self._qe_runner_explicit,
                repair_loop_explicit=self._repair_loop_explicit,
            )
            ctx = StageContext(
                config=self.config,
                router=self.router,
                ledger=ledger,
                manifest=manifest,
                adapter=adapter,
                job_id=actual_job_id,
                input_path=input_path,
                output_path=output_path,
                source_lang=source_lang,
                target_lang=target_lang,
                profile_name=profile_name,
                create_event=self._create_progress_event,
                short_chain=short_chain,
                start_chapter=start_chapter,
                max_chapters=max_chapters,
                # The fast lane may still be refused by the file probe the bible
                # stage runs; this is the policy's wish, not the verdict.
                fast_lane=adaptive_policy.fast_lane_bible,
                # A callable, so the export stage reads the provider's counters
                # when it renders rather than when the loop started.
                measure_run_usage=lambda: self._session_usage(run_session),
                bill_run_usage=lambda: self._bill_session(
                    run_session, actual_job_id, ledger, raise_on_budget=False
                ),
                cancel_token=cancel_token,
                # Separate stop channel: fires when another process reclaims this
                # run's writer lease. Present on every path, including the REST
                # embedding path that passes no ``cancel_token``.
                lease_lost=self._lease_lost_event,
            )
            logger.info(
                "Adaptive execution policy for %s: %s",
                actual_job_id,
                adaptive_policy.reason,
            )

            # -----------------------------------------------------------------
            # The stage plan (ubt.core.engine.plan) owns order and gating: this
            # class owns the run's resources -- the writer lock, ledger, adapter,
            # router -- and the plan owns what runs in what order. The terminal
            # export event triggers TM writeback and the finalize hook BEFORE it
            # is yielded, so a caller that breaks immediately does not lose
            # those writes to GeneratorExit.
            # -----------------------------------------------------------------
            # The values a stage produces and a later one consumes live here for
            # the whole run (explicit stage execution context): the plan threads each to
            # the stage that reads it, and the terminal hook reads the same facts
            # the stages filled.
            facts = RunFacts()
            # The run's one mutable state: the block snapshot, owned by the plan
            # and read explicitly by the analyze-adjacent stages (unified
            # pipeline stage context).
            blocks = BlockReader(ledger, actual_job_id)

            async def _on_export_completed(event: TranslationProgressEvent) -> None:
                await run_tm_writeback_stage(ctx, services, facts.terminology)
                await self._run_finalize_hook(event, ledger)

            gates = RunGates(
                chapter_streaming=self.config.chapter_streaming_enabled
                and len(manifest.chapters) > 1,
                c_text=self.config.c_text_enabled,
                consistency=self.config.consistency_enforce != "off",
            )
            # Renew the writer-lock lease for the life of the run: the lock is an
            # *expiring* lease (so a crashed writer can be reclaimed), and a real
            # book translation routinely runs longer than the lease window. Left
            # unrenewed, the lease would expire mid-run and a second process could
            # take over the job — both would then draft and pay the provider, and
            # their checkpoints would interleave. The renewer touches the row every
            # ``renew_interval`` seconds, well inside the lease window.
            renew_task = asyncio.create_task(
                self._renew_writer_lock(writer_lock), name=f"renew-lock:{actual_job_id}"
            )
            # ``aclosing`` owns the stage generator: its own ``finally`` (stage
            # teardown, checkpoint flush) must run *here*, inside the try, so it
            # finishes before this method's ``finally`` closes the ledger and
            # releases the writer lock. Without it the inner generator finalized
            # only when the outer one was GC'd — after the lock release — and
            # its writes silently reopened the closed ledger (see
            # ``LedgerBase._get_conn``).
            #
            # The renewal task is stopped in this block's own ``finally``, not
            # after the loop: every ``except`` arm below re-raises, so cancelling
            # after the ``async for`` leaked one never-ending task per failed or
            # cancelled job in the long-lived API/worker process. Once the lock
            # is released ``renew()`` returns silently (``_held`` is False), so
            # the loop never self-terminates — it would hold the whole
            # orchestrator alive for the life of the process.
            try:
                async with aclosing(
                    run_stages(
                        ctx,
                        gates,
                        facts,
                        services,
                        blocks=blocks,
                        on_export_completed=_on_export_completed,
                    )
                ) as stage_events:
                    async for event in stage_events:
                        yield event
                # The renewal task is detached from the stage generator, so a
                # lease lost mid-run may surface here rather than as a stage
                # exception. The stages stop at their next boundary check; this is
                # the one place that can turn that into the operator-facing
                # reason. Without it a reclaimed run would finalize itself as a
                # success over a ledger another process now owns.
                if self._lease_lost_reason is not None:
                    raise LedgerWriterLeaseLostError(self._lease_lost_reason)
            finally:
                renew_task.cancel()
                with suppress(asyncio.CancelledError):
                    await renew_task

        except GeneratorExit:
            # Consumer closed the generator early (job_worker's ``aclosing`` on a
            # lost lease, or any caller that stops iterating). NOT proof the user
            # cancelled — a lease loss lands here too — so mark failed, not
            # cancelled; the worker rewrites it with the real reason.
            logger.warning("Pipeline generator closed early for job %s", actual_job_id)
            await asyncio.shield(
                _mark_failed_unless_completed(ledger, actual_job_id, status="failed")
            )
            raise
        except LedgerWriterLeaseLostError as exc:
            # Another process owns this ledger now. Writing a terminal state here
            # is exactly what the exception exists to prevent: the new owner is
            # still running the job, and a ``failed`` row would make every status
            # surface report a failure for work that is actively progressing. The
            # write would also happen through a connection whose lease is gone.
            #
            # The work this process paid for is already durable in the ledger, so
            # the new owner resumes from it. Only the queue row needs a decision,
            # and the worker's own ``LeaseLostError`` branch makes it.
            logger.warning("Pipeline stopped for job %s: %s", actual_job_id, exc)
            raise
        except (
            asyncio.CancelledError,
            JobInterruptedError,
            KeyboardInterrupt,
        ) as exc:
            # BaseExceptions the `except Exception` branch never sees; each must
            # still land the terminal write or the job reads as "running"
            # forever and JobManager keeps counting it against max_running_jobs.
            # Shielded so a second cancel cannot abandon it (job_worker pattern).
            logger.warning("Pipeline aborted for job %s (%s)", actual_job_id, type(exc).__name__)
            await asyncio.shield(
                _mark_failed_unless_completed(ledger, actual_job_id, status="cancelled")
            )
            raise
        except Exception as exc:
            # UBTError messages are operator-facing; real bugs keep the traceback.
            if isinstance(exc, UBTError):
                logger.error("Pipeline failed for job %s: %s", actual_job_id, exc)
            else:
                logger.exception("Pipeline failed for job %s: %s", actual_job_id, exc)
            # Shielded like the GeneratorExit / cancel branches above: a second
            # cancellation arriving while this terminal write is in flight must
            # not abandon it, or the job reads as "running" forever and keeps
            # counting against max_running_jobs.
            await asyncio.shield(
                _mark_failed_unless_completed(
                    ledger,
                    actual_job_id,
                    status="failed",
                    error=public_job_error(exc, actual_job_id),
                )
            )
            # Report the job's real progress on the failure event: a run that
            # paid for 900/1000 blocks must not read as 0% on /status, SSE, or
            # the queue row just because it ended in failure.
            failure_stats = (
                await asyncio.to_thread(ledger.get_job_stats, actual_job_id)
                if ledger is not None
                else None
            )
            failure_event = TranslationProgressEvent(
                event_type=EventType.PIPELINE_FAILED,
                job_id=actual_job_id,
                total_blocks=int((failure_stats or {}).get("total", 0) or 0),
                drafted_blocks=int((failure_stats or {}).get("drafted", 0) or 0),
                repair_pending_blocks=int((failure_stats or {}).get("repair_pending", 0) or 0),
                completed_blocks=int((failure_stats or {}).get("completed", 0) or 0),
                failed_blocks=int((failure_stats or {}).get("failed", 0) or 0),
                needs_human_blocks=int((failure_stats or {}).get("needs_human", 0) or 0),
                blocked_human_blocks=int((failure_stats or {}).get("blocked_human", 0) or 0),
                message=public_job_error(exc, actual_job_id),
            )
            try:
                yield failure_event
            except Exception as report_exc:
                logger.warning(
                    "Failed reporting failure event for job %s: %s", actual_job_id, report_exc
                )
            raise
        finally:
            # Unbind this run's cooperative-cancel token. A ContextVar outlives
            # the coroutine that set it whenever the task keeps running, and
            # ``JobWorker`` runs job after job on the same task: leaving a set
            # token bound made the next job's first backoff raise
            # JobInterruptedError for a cancel that belonged to its predecessor.
            set_run_cancel_token(None, None)
            # Release the WAL + connection per run, and close the
            # provider's HTTP pool (it is recreated lazily on the next call, so
            # reusing this orchestrator for another job stays correct).
            if ledger is not None:
                try:
                    ledger.close()
                except Exception as ledger_exc:
                    logger.debug("Error closing ledger for job %s: %s", actual_job_id, ledger_exc)
            if tm is not None:
                try:
                    tm.close()
                except Exception as tm_exc:
                    logger.debug("Error closing TM for job %s: %s", actual_job_id, tm_exc)
            # Adapters may own subprocesses; close
            # them per run so a long-lived server does not accumulate children.
            if adapter is not None and self.custom_adapter is None:
                await _close_adapter_off_loop(adapter, actual_job_id)
            if self._owns_router:
                try:
                    await self.router.aclose()
                except Exception as router_exc:
                    logger.debug("Error closing router for job %s: %s", actual_job_id, router_exc)
            if self._owns_qe_runner:
                try:
                    await self.qe_runner.aclose()
                except Exception as qe_exc:
                    logger.debug("Error closing QE runner for job %s: %s", actual_job_id, qe_exc)
            # The judge's own provider is created per orchestrator and owns an
            # httpx.AsyncClient; nothing else ever closed it, so a long-lived
            # API/worker leaked one connection pool per judged run. It is closed
            # after the QE runner (which may still be unwinding a call).
            if self._judge_provider is not None:
                try:
                    await self._judge_provider.aclose()
                except Exception as judge_exc:
                    logger.debug(
                        "Error closing judge provider for job %s: %s", actual_job_id, judge_exc
                    )
            if writer_lock is not None:
                try:
                    writer_lock.release()
                except Exception as lock_exc:
                    logger.debug(
                        "Error releasing writer lock for job %s: %s", actual_job_id, lock_exc
                    )
            # Nothing reads the session after the run ends (one orchestrator per
            # job), so the pop is pure teardown.
            self._billing_sessions.pop(actual_job_id, None)
