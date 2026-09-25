"""L3 LLM-as-Judge QE (suspect-only) + tiered gray-zone routing.

Industry mapping (2026 consensus):
- L1 rules/heuristic, full volume: ``FastPassFilter`` + ``HeuristicQERunner``.
- L2 learned QE small model, full volume / routing: ``SubprocessQERunner``
  (CometKiwi, optional, heavy dep).
- L3 LLM judge, selected pairs only: this module. Only the pairs a second
  opinion can actually inform get one cheap LLM call (score-only, ~20 output
  tokens, temperature 0). Deterministic span annotation for repair stays
  rule-based (``MQMSpanAnnotator``) — the judge only decides routing, not spans.

ROI: with the default configuration the judge sees only the single unclassified
structural class the heuristic emits (the gray band [0.70, 0.80) contains exactly
one reachable score), so the paid pass is a fraction of blocks, not a quality
audit of them. ``UBT_QE_JUDGE_PASS_SAMPLE`` opts into auditing a sample of clean
passes, which is the one place the heuristic genuinely cannot tell a good
translation from a merely well-formed one.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import zlib
from collections.abc import Awaitable, Callable
from typing import Any

from ubt.core.qe.base import BaseQERunner
from ubt.core.qe.comet_runner import (
    QE_SCORE_PASS,
    QE_SCORE_STRUCTURAL_OTHER,
    HeuristicQERunner,
)

#: Heuristic classes worth a paid second opinion. Each discrete score names its
#: own defect class, so this set is the routing table; hard defects (omission,
#: numeric drift, leaks) are deliberately absent because they route to repair on
#: their flag and no judge can undo them.
JUDGED_CLASSES: frozenset[float] = frozenset({QE_SCORE_STRUCTURAL_OTHER})

logger = logging.getLogger(__name__)

JudgeFn = Callable[..., Awaitable[str]]

# Sentinel returned by the judge when it cannot produce a usable score (call
# failure or unparsable reply). It is outside the normalized [0, 1] range by
# construction, so the tiered runner can detect "no opinion" and preserve the
# Heuristic score instead of silently adopting a fallback.
_JUDGE_NO_OPINION = -1.0

JUDGE_SYSTEM_PROMPT = (
    "You are a translation quality estimator. Score the translation 0-100 "
    "(100 = perfect, 0 = unusable). Consider adequacy, fluency, terminology "
    "consistency, and numeric fidelity. Reply with exactly one line: "
    "`score: <number>`. No explanation."
)

_SCORE_RE = re.compile(
    r"\*{0,2}score\*{0,2}\s*[:=]\*{0,2}\s*(\d+(?:\.\d+)?)(?:\s*/\s*(\d+(?:\.\d+)?))?",
    re.IGNORECASE,
)
_BARE_NUMBER_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)(?:\s*/\s*(\d+(?:\.\d+)?))?\s*$")


def parse_judge_score(raw: str) -> float | None:
    """Parse an LLM judge reply to 0.0~1.0. Returns None when unparsable."""
    text = (raw or "").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "score" in data:
            return _normalize(float(data["score"]))
    except (ValueError, TypeError):
        pass
    m = _SCORE_RE.search(text)
    if m:
        try:
            num = float(m.group(1))
            if m.group(2):
                den = float(m.group(2))
                return min(1.0, max(0.0, num / den)) if den > 0 else 0.0
            return _normalize(num)
        except ValueError:
            return None
    m = _BARE_NUMBER_RE.match(text)
    if m:
        try:
            num = float(m.group(1))
            if m.group(2):
                den = float(m.group(2))
                return min(1.0, max(0.0, num / den)) if den > 0 else 0.0
            return _normalize(num)
        except ValueError:
            return None
    return None


def _normalize(score: float) -> float:
    """Map a judge reply onto 0.0~1.0.

    The prompt asks for a 0-100 score, so any value at or above 1 is a
    percentage. The boundary value 1 is genuinely ambiguous — "1 out of 100"
    or the fraction "1.0" (perfect) — and a judge explicitly told to answer
    0-100 means the former, so it is read as 1%. Treating it as perfect let a
    near-failing gray-zone block auto-pass under ``qe_judge_allow_upgrade``.
    Strictly fractional replies in (0, 1) (e.g. 0.95, 0.73) are still read as
    fractions, since models occasionally answer on the 0-1 scale regardless.
    """
    if score >= 1.0:
        score = score / 100.0
    return min(1.0, max(0.0, score))


def build_judge_prompts(
    src: str, mt: str, target_lang: str = "zh", source_lang: str = "en"
) -> tuple[str, str]:
    """Build (system, user) prompts for one judge call.

    Both language tags are explicit: a run translating e.g. fr→ja must not see
    its source mislabeled ``Source (en)``, which biases the judge's rubric.
    """
    user = (
        f"Source ({source_lang}):\n{src}\n\nTranslation ({target_lang}):\n{mt}\n\n"
        "Reply with exactly one line: `score: <0-100>`."
    )
    return JUDGE_SYSTEM_PROMPT, user


class LLMJudgeQERunner(BaseQERunner):
    """Score-only LLM judge. One short call per pair, temperature 0."""

    def __init__(
        self,
        judge_fn: JudgeFn,
        model: str | None = None,
        target_lang: str = "zh",
        source_lang: str = "en",
        temperature: float = 0.0,
        max_concurrency: int = 5,
    ) -> None:
        self._judge_fn = judge_fn
        self._model = model
        self._target_lang = target_lang
        self._source_lang = source_lang
        self._temperature = temperature
        self._sem = asyncio.Semaphore(max_concurrency)

    def with_languages(self, source_lang: str, target_lang: str) -> LLMJudgeQERunner:
        """Rebind the judge's language labels for this run's pair.

        Shared across concurrent runs would leak one book's pair into another,
        so return a fresh runner unless the pair already matches.
        """
        if source_lang == self._source_lang and target_lang == self._target_lang:
            return self
        return LLMJudgeQERunner(
            judge_fn=self._judge_fn,
            model=self._model,
            target_lang=target_lang,
            source_lang=source_lang,
            temperature=self._temperature,
        )

    def is_calibrated(self) -> bool:
        """LLM opinions are not calibrated enough to rank repair candidates."""
        return False

    async def _score_one(self, src: str, mt: str) -> float:
        system, user = build_judge_prompts(src, mt, self._target_lang, self._source_lang)
        try:
            async with self._sem:
                kwargs: dict[str, Any] = {
                    "system_prompt": system,
                    "user_prompt": user,
                    "temperature": self._temperature,
                }
                if self._model is not None:
                    kwargs["model"] = self._model
                raw = await self._judge_fn(**kwargs)
        except Exception as exc:
            logger.warning("LLM judge call failed, preserving heuristic score: %s", exc)
            return _JUDGE_NO_OPINION
        parsed = parse_judge_score(raw)
        if parsed is None:
            logger.warning("LLM judge unparsable reply %.80r, preserving heuristic score", raw)
            return _JUDGE_NO_OPINION
        return round(parsed, 4)

    async def score_pairs(self, pairs: list[dict[str, str]]) -> list[float]:
        if not pairs:
            return []
        results = await asyncio.gather(
            *[self._score_one(p.get("src", ""), p.get("mt", "")) for p in pairs]
        )
        return list(results)


class TieredQERunner(BaseQERunner):
    """L1 heuristic for all pairs, L3 judge only where a second opinion informs.

    The heuristic emits twelve discrete values, and each one *is* a defect class
    (:data:`QE_DEFECT_CLASS_LEGEND`), so routing on the score routes on the
    class. Two groups get the judge:

    * the unclassified structural class (and anything else inside the configured
      gray band) — by definition ambiguous;
    * a sample of clean passes (``pass_sample``), because "no deterministic
      invariant failed" is explicitly not a quality estimate — a faithful,
      fluent and a merely well-formed translation both score 0.92.

    Everything below the gray band is a hard defect (leaked prompt, dropped
    number, omission, broken formula) that routes to repair on its own flag; a
    judge cannot un-drop a number, so those pairs cost tokens for nothing.
    ``judge_calls`` records how many pairs actually hit the LLM (ROI observability).
    """

    def __init__(
        self,
        heuristic: HeuristicQERunner,
        judge: LLMJudgeQERunner | None = None,
        gray_low: float = 0.7,
        gray_high: float = 0.8,
        allow_upgrade: bool = False,
        pass_sample: float = 0.0,
    ) -> None:
        self.heuristic = heuristic
        self.judge = judge
        self.gray_low = gray_low
        self.gray_high = gray_high
        self.allow_upgrade = allow_upgrade
        self.pass_sample = min(1.0, max(0.0, pass_sample))
        self.judge_calls = 0
        # Count of gray-zone pairs where the judge produced no usable score
        # (call failure / unparsable). The heuristic score is kept for those
        # Pairs; this counter makes the silent fallback observable.
        self.judge_errors = 0

    def _needs_judge(self, score: float, src: str) -> bool:
        """Whether this pair is worth a paid second opinion."""
        if score in JUDGED_CLASSES or self.gray_low <= score < self.gray_high:
            return True
        if self.pass_sample <= 0.0 or score != QE_SCORE_PASS:
            return False
        # Stable sampling: the same segment is always or never judged, so a
        # rerun of one chapter does not audit a different set of blocks.
        return (zlib.crc32(src.encode("utf-8")) % 10_000) < int(self.pass_sample * 10_000)

    def is_glossary_aware(self) -> bool:
        """True when the underlying heuristic can see terminology drift.

        The judge never scores terminology, so the tier is only glossary-aware
        through its heuristic leg — which caps a violation below the gray band
        and therefore keeps the judge from ever seeing it.
        """
        return self.heuristic.is_glossary_aware()

    def is_calibrated(self) -> bool:
        """A tier is only ever as calibrated as its heuristic leg.

        The judge leg is an uncalibrated LLM opinion, and in the default
        configuration (``pass_sample`` below 1.0) clean-pass candidates score
        the heuristic's flat 0.92 without ever reaching it — best-of-n rerank
        would pick between heuristic bands and present the pick as measured.
        Same delegation shape as :meth:`is_glossary_aware`.
        """
        return self.heuristic.is_calibrated()

    def with_languages(self, source_lang: str, target_lang: str) -> TieredQERunner:
        heuristic = self.heuristic.with_languages(source_lang, target_lang)
        judge = self.judge.with_languages(source_lang, target_lang) if self.judge else None
        if heuristic is self.heuristic and judge is self.judge:
            return self
        rebound = TieredQERunner(
            heuristic=heuristic,
            judge=judge,
            gray_low=self.gray_low,
            gray_high=self.gray_high,
            allow_upgrade=self.allow_upgrade,
            pass_sample=self.pass_sample,
        )
        # Counters are per-run ROI observability, and the pipeline rebinds the
        # runner per run to set language labels: dropping them here made the
        # paid-judge call/error counts the report reads always zero.
        rebound.judge_calls = self.judge_calls
        rebound.judge_errors = self.judge_errors
        return rebound

    def with_glossary(self, glossary: list[dict[str, Any]]) -> TieredQERunner:
        """Bind the run's terminology to the heuristic leg.

        The judge never scores terminology (see :meth:`is_glossary_aware`), so
        glossary-awareness lives entirely in the heuristic. Without this the
        quality gate could not hand the glossary to a tiered runner: a dropped
        term scored the flat 0.92 pass band, and the repair loop -- seeing
        ``is_glossary_aware()`` false -- kept the violation marker forever, so
        a genuinely fixed term still ended FAILED.
        """
        heuristic = self.heuristic.with_glossary(glossary)
        if heuristic is self.heuristic:
            return self
        rebound = TieredQERunner(
            heuristic=heuristic,
            judge=self.judge,
            gray_low=self.gray_low,
            gray_high=self.gray_high,
            allow_upgrade=self.allow_upgrade,
            pass_sample=self.pass_sample,
        )
        rebound.judge_calls = self.judge_calls
        rebound.judge_errors = self.judge_errors
        return rebound

    async def score_pairs(self, pairs: list[dict[str, str]]) -> list[float]:
        if not pairs:
            return []
        base = await self.heuristic.score_pairs(pairs)
        if self.judge is None:
            return base
        gray_idx = [
            i
            for i, (s, p) in enumerate(zip(base, pairs, strict=True))
            if self._needs_judge(s, p.get("src", ""))
        ]
        if not gray_idx:
            return base
        subset = [pairs[i] for i in gray_idx]
        judged = await self.judge.score_pairs(subset)
        self.judge_calls += len(subset)
        merged = list(base)
        for i, s in zip(gray_idx, judged, strict=True):
            if s == _JUDGE_NO_OPINION:
                # Judge failed for this pair: preserve the original heuristic
                # Score rather than silently adopting a fallback.
                self.judge_errors += 1
                continue
            if self.allow_upgrade:
                merged[i] = s
            else:
                # Conservative mode: the judge may only LOWER the heuristic score
                merged[i] = min(base[i], s)
        return merged
