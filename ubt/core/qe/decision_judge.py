"""Zero-output-token Decision Model QE runner (Gemini Decisions API / Jev).

Replaces generative JSON-emitting LLM-as-Judge calls with batched, calibrated
decision-model scoring when ``UBT_QE_JUDGE_ENABLED=true`` and
``UBT_QE_JUDGE_BACKEND`` is ``"gemini_decision"`` or ``"jev"``.

Local offline runs (``qe_engine="heuristic"`` or ``qe_judge_enabled=false``)
never import or invoke this module's HTTP transport.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Literal

import httpx

from ubt.core.qe.base import BaseQERunner

logger = logging.getLogger(__name__)

DecisionBackend = Literal["gemini_decision", "jev"]

DEFAULT_GEMINI_DECISION_MODEL = "gemini-decision-1"
DEFAULT_GEMINI_DECISION_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_JEV_URL = "https://api.typesafe.ai/v1"

# Decision rubric choices mapped to [0.0, 1.0] quality scores.
_QUALITY_CHOICES: tuple[tuple[str, float], ...] = (
    ("accurate_faithful_fluent", 0.95),
    ("minor_style_or_fluency_issue", 0.78),
    ("major_mistranslation_or_omission", 0.45),
    ("critical_corruption_or_untranslated", 0.15),
)

#: Injectable async transport for testing without live network calls:
#: ``(url, headers, payload) -> parsed_json_dict``.
DecisionPostFn = Callable[[str, dict[str, str], dict[str, Any]], Awaitable[dict[str, Any]]]


def _expected_score_from_probs(probs: dict[str, float]) -> float:
    """Compute calibrated expected quality score in [0.0, 1.0] from choice probabilities."""
    total_p = 0.0
    weighted = 0.0
    for label, value in _QUALITY_CHOICES:
        p = float(probs.get(label, 0.0))
        if p > 0.0:
            total_p += p
            weighted += p * value
    if total_p <= 0.0:
        return 0.5
    return max(0.0, min(1.0, weighted / total_p))


def _extract_decision_scores(payload: dict[str, Any], expected_count: int) -> list[float]:
    """Parse calibrated scores from Gemini Decisions or Jev JSON response."""
    items: list[Any] = (
        payload.get("decisions") or payload.get("results") or payload.get("scores") or []
    )
    if not isinstance(items, list) or not items:
        return [0.5] * expected_count

    out: list[float] = []
    for item in items[:expected_count]:
        if isinstance(item, (int, float)):
            val = float(item)
            out.append(max(0.0, min(1.0, val / 100.0 if val > 1.0 else val)))
        elif isinstance(item, dict):
            if "score" in item and isinstance(item["score"], (int, float)):
                val = float(item["score"])
                out.append(max(0.0, min(1.0, val / 100.0 if val > 1.0 else val)))
            elif isinstance(item.get("probabilities"), dict):
                out.append(_expected_score_from_probs(item["probabilities"]))
            elif isinstance(item.get("choices"), list):
                probs = {
                    str(c.get("label") or c.get("choice") or ""): float(c.get("probability", 0.0))
                    for c in item["choices"]
                    if isinstance(c, dict)
                }
                out.append(_expected_score_from_probs(probs))
            else:
                out.append(0.5)
        else:
            out.append(0.5)

    while len(out) < expected_count:
        out.append(0.5)
    return out


class DecisionQERunner(BaseQERunner):
    """Batched zero-output-token calibrated QE judge using a Decision Model API."""

    def __init__(
        self,
        *,
        backend: DecisionBackend = "gemini_decision",
        api_key: str = "",
        model: str | None = None,
        base_url: str | None = None,
        source_lang: str = "en",
        target_lang: str = "zh",
        batch_size: int = 32,
        timeout_s: float = 30.0,
        max_concurrency: int = 4,
        post_fn: DecisionPostFn | None = None,
        on_call: Callable[[], None] | None = None,
    ) -> None:
        self._backend: DecisionBackend = backend
        self._api_key = api_key
        self._model = model or (
            DEFAULT_GEMINI_DECISION_MODEL if backend == "gemini_decision" else "jev-1"
        )
        self._base_url = (
            base_url
            or (DEFAULT_GEMINI_DECISION_URL if backend == "gemini_decision" else DEFAULT_JEV_URL)
        ).rstrip("/")
        self._source_lang = source_lang
        self._target_lang = target_lang
        self._batch_size = max(1, batch_size)
        self._timeout_s = timeout_s
        self._max_concurrency = max(1, max_concurrency)
        self._post_fn = post_fn
        self._on_call = on_call
        self.calls_made = 0
        self._sem = asyncio.Semaphore(self._max_concurrency)
        self._client: httpx.AsyncClient | None = None

    def with_languages(self, source_lang: str, target_lang: str) -> DecisionQERunner:
        if source_lang == self._source_lang and target_lang == self._target_lang:
            return self
        return DecisionQERunner(
            backend=self._backend,
            api_key=self._api_key,
            model=self._model,
            base_url=self._base_url,
            source_lang=source_lang,
            target_lang=target_lang,
            batch_size=self._batch_size,
            timeout_s=self._timeout_s,
            max_concurrency=self._max_concurrency,
            post_fn=self._post_fn,
            on_call=self._on_call,
        )

    def is_calibrated(self) -> bool:
        """Decision models emit calibrated probabilities suitable for candidate ranking."""
        return True

    def score_source(self) -> str:
        return f"decision_judge:{self._backend}"

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _build_request(
        self, batch: list[dict[str, str]]
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        choices = [label for label, _ in _QUALITY_CHOICES]
        questions = [
            {
                "id": str(idx),
                "prompt": (
                    f"Evaluate this {self._source_lang}->{self._target_lang} translation "
                    "for accuracy, completeness, and fluency.\n"
                    f"SOURCE: {p.get('src', '')}\n"
                    f"TRANSLATION: {p.get('mt', '')}"
                ),
                "choices": choices,
            }
            for idx, p in enumerate(batch)
        ]

        if self._backend == "gemini_decision":
            url = f"{self._base_url}/models/{self._model}:decide"
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self._api_key,
            }
            payload: dict[str, Any] = {
                "model": self._model,
                "questions": questions,
            }
            return url, headers, payload

        # Jev decision endpoint
        url = f"{self._base_url}/decide"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        payload = {
            "model": self._model,
            "items": questions,
        }
        return url, headers, payload

    async def _post_json(
        self, url: str, headers: dict[str, str], payload: dict[str, Any]
    ) -> dict[str, Any]:
        if self._post_fn is not None:
            return await self._post_fn(url, headers, payload)
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout_s)
        resp = await self._client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, dict) else {}

    async def _score_batch(self, batch: list[dict[str, str]]) -> list[float]:
        async with self._sem:
            self.calls_made += 1
            if self._on_call is not None:
                self._on_call()
            url, headers, payload = self._build_request(batch)
            try:
                data = await self._post_json(url, headers, payload)
                return _extract_decision_scores(data, len(batch))
            except Exception as exc:
                logger.warning(
                    "DecisionQERunner (%s) batch scoring failed (%s); returning neutral 0.5",
                    self._backend,
                    exc,
                )
                return [0.5] * len(batch)

    async def score_pairs(self, pairs: list[dict[str, str]]) -> list[float]:
        if not pairs:
            return []
        batches = [pairs[i : i + self._batch_size] for i in range(0, len(pairs), self._batch_size)]
        batch_results = await asyncio.gather(*[self._score_batch(b) for b in batches])
        return [score for sub in batch_results for score in sub]
