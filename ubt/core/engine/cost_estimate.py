"""What this run is about to cost — said before any of it is spent.

The pipeline can already stop the moment a job passes ``--budget-usd``, and a
quality report now prices the job's whole life. Neither answers the question a
user actually asks first: *is this book worth the money it will cost?* That
question has to be answered before the first request, when the only inputs are
the block count, the source lengths, and the prompt the draft stage is about to
send — all of which exist the moment ingest finishes.

Two deliberate limits on the estimate:

* It prices **draft calls only**. Repair passes, QE judging, backfills and
  summaries are real but variable spend, and pretending to fold them in with a
  made-up percentage would make the number look derived when it is guessed. The
  printed line says so.
* The per-block prompt is *measured*, not assumed: the fixed scaffolding comes
  from the same prompt builder the draft stage uses. Earlier calibrations of
  "about 1,000 prefix tokens per block" were true of one prompt version and
  silently wrong for every later one.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ubt.core.router.pricing import (
    billing_enabled_for_local_endpoints,
    endpoint_is_local,
    price_is_known,
    resolve_cached_input_price,
    resolve_model_prices,
)

if TYPE_CHECKING:
    from ubt.core.ir.models import IRBlock
    from ubt.core.router.router import ModelRouter

logger = logging.getLogger(__name__)

# The router reserves TPM with the same heuristic, so estimate and reservation
# cannot disagree about how many tokens a block of text is worth.
_CHARS_PER_TOKEN = 4

# Default fallback ratio when source_lang/target_lang are not provided.
_OUTPUT_TOKEN_RATIO = 1.5

_CJK_LANGS = frozenset({"zh", "ja", "ko", "yue", "wuu"})
_HIGH_EXPANSION_LANGS = frozenset(
    {"ru", "uk", "be", "bg", "sr", "ar", "fa", "hi", "bn", "ta", "th", "el", "de"}
)


def count_text_tokens(text: str) -> int:
    """Script-aware BPE token count across Latin, CJK, Cyrillic, Arabic, and Indic scripts.

    Pure ASCII prose averages ~4 chars/token in BPE tokenizers (cl100k/o200k/DeepSeek),
    whereas CJK ideographs average ~0.85 tokens/char and Cyrillic/Arabic/Indic scripts
    average ~0.55 tokens/char.
    """
    if not text:
        return 0
    if text.isascii():
        return len(text) // _CHARS_PER_TOKEN
    ascii_chars = 0
    cjk_chars = 0
    other_non_ascii = 0
    for ch in text:
        cp = ord(ch)
        if cp < 128:
            ascii_chars += 1
        elif (
            0x3000 <= cp <= 0x9FFF
            or 0xAC00 <= cp <= 0xD7AF
            or 0xF900 <= cp <= 0xFAFF
            or 0x20000 <= cp <= 0x2FA1F
        ):
            cjk_chars += 1
        else:
            other_non_ascii += 1
    return int(
        ascii_chars // _CHARS_PER_TOKEN
        + math.ceil(cjk_chars * 0.85)
        + math.ceil(other_non_ascii * 0.55)
    )


def resolve_output_token_ratio(
    source_lang: str | None = None,
    target_lang: str | None = None,
) -> float:
    """Directional language-pair token fertility ratio (completion_tokens / source_tokens)."""
    if not source_lang or not target_lang:
        return _OUTPUT_TOKEN_RATIO
    src = source_lang.split("-")[0].split("_")[0].lower()
    tgt = target_lang.split("-")[0].split("_")[0].lower()
    if src == tgt:
        return 1.0
    src_cjk = src in _CJK_LANGS
    tgt_cjk = tgt in _CJK_LANGS
    if src_cjk and not tgt_cjk:
        return 1.05 if tgt in _HIGH_EXPANSION_LANGS else 0.85
    if not src_cjk and tgt_cjk:
        return 1.25
    if src_cjk and tgt_cjk:
        return 1.05
    if tgt in _HIGH_EXPANSION_LANGS:
        return 1.40
    return 1.15


@dataclass(frozen=True)
class RunEstimate:
    """The forecast for one run's draft calls."""

    billable_blocks: int
    prefix_tokens: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd_uncached: float | None
    cost_usd_cached: float | None
    draft_model: str

    @property
    def money_is_unknown(self) -> bool:
        """True when the draft model has no price entry, so money cannot be stated."""
        return self.cost_usd_uncached is None

    def describe(self) -> str:
        """One log line: the range, and what is excluded from it."""
        money = (
            "unknown (no price-table entry for this model)"
            if self.money_is_unknown
            else f"${self.cost_usd_cached:.4f}-${self.cost_usd_uncached:.4f}"
        )
        return (
            f"Pre-flight estimate: {self.billable_blocks} draft call(s) over "
            f"~{self.prompt_tokens} prompt + ~{self.completion_tokens} completion "
            f"tokens on '{self.draft_model}' (static prefix "
            f"{self.prefix_tokens} tok/call) = {money}. Draft calls only: repair, "
            "QE judging and summaries are not included."
        )


def measure_prefix_tokens(
    router: ModelRouter,
    *,
    target_lang: str,
    source_lang: str,
    genre_profile: str = "general",
    glossary_table: str = "",
) -> int | None:
    """Tokens of scaffolding every draft request carries, measured not assumed.

    Assembled by the router's own prompt builder with an empty source span, so
    the result follows the active prompt strategy, language profile and glossary
    size instead of a constant that goes stale the next time the prompt changes.

    ``None`` when the router hands back no usable ``(system, user)`` pair — a
    test double or a plugin router with a different contract. The estimate is
    information; losing it must never be what breaks a run.
    """
    try:
        system_prompt, user_prompt = router.build_draft_prompt(
            "",
            glossary_table=glossary_table,
            target_lang=target_lang,
            source_lang=source_lang,
            genre_profile=genre_profile,
        )
    except (TypeError, ValueError) as exc:
        logger.debug("Pre-flight estimate skipped: prompt probe failed (%s)", exc)
        return None
    if not isinstance(system_prompt, str) or not isinstance(user_prompt, str):
        logger.debug("Pre-flight estimate skipped: prompt probe was not a text pair")
        return None
    return count_text_tokens(system_prompt) + count_text_tokens(user_prompt)


def estimate_draft_cost(
    blocks: Sequence[IRBlock],
    *,
    draft_model: str,
    prefix_tokens: int,
    macro_chunk_size: int = 1,
    batch_discount: float = 1.0,
    base_url: str | None = None,
    source_lang: str | None = None,
    target_lang: str | None = None,
) -> RunEstimate:
    """Project the cost of drafting every untranslated block exactly once."""
    billable_list = [
        block for block in blocks if not block.skip_translate and not block.target_text
    ]
    source_chars = sum(len(block.source_text or "") for block in billable_list)
    measured_tokens = sum(count_text_tokens(block.source_text or "") for block in billable_list)
    billable = len(billable_list)
    return estimate_draft_cost_from_totals(
        billable_blocks=billable,
        source_chars=source_chars,
        source_tokens_override=measured_tokens,
        draft_model=draft_model,
        prefix_tokens=prefix_tokens,
        macro_chunk_size=macro_chunk_size,
        batch_discount=batch_discount,
        base_url=base_url,
        source_lang=source_lang,
        target_lang=target_lang,
    )


def estimate_draft_cost_from_totals(
    *,
    billable_blocks: int,
    source_chars: int,
    draft_model: str,
    prefix_tokens: int,
    macro_chunk_size: int = 1,
    batch_discount: float = 1.0,
    base_url: str | None = None,
    source_tokens_override: int | None = None,
    source_lang: str | None = None,
    target_lang: str | None = None,
) -> RunEstimate:
    """Same math as :func:`estimate_draft_cost`, but on pre-aggregated totals.

    ``ubt assess`` quotes a document before blocks exist, so it arrives with
    counts measured by the probe instead of an ``IRBlock`` list. One formula
    keeps the pre-run quote and the in-flight preflight from drifting apart.

    ``base_url`` decides free-vs-unknown for a model absent from the price
    table: self-hosted endpoints cost $0 (a quotable number), remote ones are
    unknown.
    """
    chunk_size = max(1, macro_chunk_size)
    total_calls = math.ceil(billable_blocks / chunk_size) if billable_blocks else 0
    source_tokens = (
        source_tokens_override
        if source_tokens_override is not None
        else source_chars // _CHARS_PER_TOKEN
    )
    prompt_tokens = total_calls * prefix_tokens + source_tokens
    ratio = resolve_output_token_ratio(source_lang, target_lang)
    completion_tokens = int(source_tokens * ratio)

    input_price, output_price = resolve_model_prices(draft_model)
    cached_input_price = resolve_cached_input_price(draft_model)
    uncached: float | None = None
    cached: float | None = None
    if billable_blocks and price_is_known(draft_model, base_url=base_url):
        if endpoint_is_local(base_url) and not billing_enabled_for_local_endpoints():
            # Self-hosted: $0 by construction. Pricing by NAME here would charge
            # a local `qwen3:8b` at the cloud Qwen rate — the exact mistake
            # ``pricing.py`` documents that it avoids — and, because
            # ``stages/preflight.py`` refuses a run whose floor exceeds the
            # budget, it would refuse a genuinely free run.
            uncached = 0.0
            cached = 0.0
        else:
            uncached = (
                (prompt_tokens * input_price + completion_tokens * output_price)
                / 1_000_000
                * batch_discount
            )
            prefix_prompt_tokens = total_calls * prefix_tokens
            cached = (
                (
                    prefix_prompt_tokens * cached_input_price
                    + (prompt_tokens - prefix_prompt_tokens) * input_price
                    + completion_tokens * output_price
                )
                / 1_000_000
                * batch_discount
            )

    return RunEstimate(
        billable_blocks=billable_blocks,
        prefix_tokens=prefix_tokens,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd_uncached=None if uncached is None else round(uncached, 5),
        cost_usd_cached=None if cached is None else round(cached, 5),
        draft_model=draft_model,
    )
