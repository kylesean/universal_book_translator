"""L3 LLM-as-Judge QE (suspect-only) + tiered routing.

Industry mapping (2026 consensus):
- L1 rules/heuristic, full volume: ``FastPassFilter`` + ``HeuristicQERunner``.
- L2/L3 LLM judge, selected suspect pairs only: this module. Only the pairs a second
  opinion can actually inform get one cheap LLM call (score-only, ~20 output
  tokens, temperature 0). Deterministic span annotation for repair stays
  rule-based (``MQMSpanAnnotator``) — the judge only decides routing, not spans.

ROI: with the default configuration the judge sees only the single unclassified
structural class the heuristic emits (the ambiguous class contains exactly
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
from ubt.core.qe.heuristic import (
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
    "`score: <number>`. No explanation.\n"
    "\n"
    "Adequacy is judged clause by clause against the source, not by how "
    "fluent the target reads. Fluent compression still loses content: ask "
    "whether every meaningful element of the source — each noun, modifier, "
    "quantifier and relation — has a counterpart in the translation. Two "
    "losses are easy to miss because the result reads well, and both are "
    "severe:\n"
    "  * a noun or modifier dropped while the sentence stays grammatical "
    '("publicly available" -> "获取" loses 公开; "content interference '
    'from background noise" -> "背景噪声的干扰" loses 内容);\n'
    "  * a phrase reduced to a shorter equivalent that no longer states what "
    "the source said.\n"
    "Score a translation that drops source content in proportion to what is "
    "lost, even when every remaining word is correct."
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
    0-100 means the former, so it is read as 1%. Reading it as perfect would
    score a failing block 1.0, which the tier then cannot lower to anything
    meaningful. Strictly fractional replies in (0, 1) (e.g. 0.95, 0.73) are
    still read as fractions, since models occasionally answer on the 0-1 scale
    regardless.
    """
    if score >= 1.0:
        score = score / 100.0
    return min(1.0, max(0.0, score))


#: Sentence end: CJK terminal punctuation splits with or without a following
#: space (Chinese prose puts none), while a Latin terminal needs whitespace —
#: otherwise "0.85" splits at the decimal point and "Fig. 3" at the
#: abbreviation. Only ``.`` ``!`` ``?`` ``。`` ``！`` ``？`` are terminals.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？])[\"'”’）)\]]*\s*|(?<=[.!?])[\"'”’）)\]]*\s+")

#: Abbreviations whose trailing period never ends a sentence. Lower-case only:
#: the split itself is case-sensitive where it matters.
_SENTENCE_ABBREVIATIONS = frozenset(
    {
        "fig",
        "figs",
        "eq",
        "eqs",
        "sec",
        "al",
        "e.g",
        "i.e",
        "etc",
        "vs",
        "cf",
        "no",
        "resp",
        "approx",
        "ref",
        "refs",
        "tab",
        "pp",
        "vol",
    }
)


def split_sentences(text: str) -> list[str]:
    """Split a paragraph into sentences, tolerant of the shapes these documents use.

    Used to align a source sentence with its translation. The pairing is
    positional, so a split that is even slightly wrong would compare one
    sentence's source against another's target and *invent* a defect — worse
    than not checking. The rules are therefore conservative: a break needs
    terminal punctuation followed by whitespace, and the word before the period
    must not be a known abbreviation (``Fig. 3``) or a bare numeral (``0.85``).
    When the two sides end up with different sentence counts the caller falls
    back to whole-block scoring rather than trusting a misaligned pair.
    """
    text = (text or "").strip()
    if not text:
        return []
    pieces = _SENTENCE_SPLIT_RE.split(text)
    out: list[str] = []
    buffer = ""
    for piece in pieces:
        candidate = f"{buffer} {piece}".strip() if buffer else piece.strip()
        if not candidate:
            continue
        last_word = re.split(r"[\s(\"']", candidate.rstrip(".!?。！？\"'”’）)]"))[-1].lower()
        if last_word in _SENTENCE_ABBREVIATIONS or last_word.isdigit():
            buffer = candidate
            continue
        out.append(candidate)
        buffer = ""
    if buffer:
        out.append(buffer)
    return out


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
        compare_sentences: bool = True,
        on_call: Callable[[], None] | None = None,
    ) -> None:
        self._judge_fn = judge_fn
        #: Invoked once per *paid* model call, at the only place a call is
        #: actually made (:meth:`_score_whole`). ``TieredQERunner`` counts
        #: through this instead of guessing per block: with sentence-level
        #: judging one 8-sentence block is eight calls, so the old
        #: ``judge_calls += len(subset)`` under-reported both the quoted price
        #: and the ROI reading by the sentence count.
        self._on_call = on_call
        self._model = model
        self._target_lang = target_lang
        self._source_lang = source_lang
        self._temperature = temperature
        self._max_concurrency = max_concurrency
        self._compare_sentences = compare_sentences
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
            max_concurrency=self._max_concurrency,
            compare_sentences=self._compare_sentences,
            on_call=self._on_call,
        )

    def is_calibrated(self) -> bool:
        """LLM opinions are not calibrated enough to rank repair candidates."""
        return False

    def score_source(self) -> str:
        """A graded judge opinion, distinct from the heuristic's discrete bands."""
        return "llm_judge"

    async def score_pairs(self, pairs: list[dict[str, str]]) -> list[float]:
        if not pairs:
            return []
        results = await asyncio.gather(
            *[self._score_one(p.get("src", ""), p.get("mt", "")) for p in pairs]
        )
        return list(results)

    async def _score_one(self, src: str, mt: str) -> float:
        if not self._compare_sentences:
            return await self._score_whole(src, mt)
        return await self._score_by_sentence(src, mt)

    async def _score_by_sentence(self, src: str, mt: str) -> float:
        """Score a block by scoring its sentences, keeping the worst.

        A whole-block score dilutes a one-word loss: the same defect measured
        0.85 against a 790-character block and below 0.50 when the clause it
        corrupts is scored on its own, because one wrong clause out of ten reads
        as a small proportional deduction. A quality gate is not asking for the
        average — it is asking whether anything in this block is wrong — so the
        block takes its **minimum** sentence score.

        The pairing is positional within the block: the translator works
        sentence by sentence, so sentence *i* of the source corresponds to
        sentence *i* of the target. When the two sides have different sentence
        counts (a merged or split sentence), the pairing cannot be trusted and
        the block falls back to a single whole-block call rather than scoring a
        misaligned pair, which would invent a defect.
        """
        src_sentences = split_sentences(src)
        mt_sentences = split_sentences(mt)
        if not src_sentences or len(src_sentences) != len(mt_sentences):
            return await self._score_whole(src, mt)
        if len(src_sentences) == 1:
            return await self._score_whole(src, mt)
        scores = await asyncio.gather(
            *[
                self._score_whole(src_sentence, mt_sentence)
                for src_sentence, mt_sentence in zip(src_sentences, mt_sentences, strict=True)
            ]
        )
        # A sentence the judge could not score (call failure) contributes no
        # opinion; if none could be scored, the block has no opinion either.
        usable = [score for score in scores if score != _JUDGE_NO_OPINION]
        if not usable:
            return _JUDGE_NO_OPINION
        return min(usable)

    async def _score_whole(self, src: str, mt: str) -> float:
        system, user = build_judge_prompts(src, mt, self._target_lang, self._source_lang)
        if self._on_call is not None:
            self._on_call()
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


class TieredQERunner(BaseQERunner):
    """L1 heuristic for all pairs, L3 judge only where a second opinion informs.

    The heuristic emits twelve discrete values, and each one *is* a defect class
    (:data:`QE_DEFECT_CLASS_LEGEND`), so routing on the score routes on the
    class. Two groups get the judge:

    * the unclassified structural class (and anything else inside the configured
      ambiguous class) — by definition ambiguous;
    * a sample of clean passes (``pass_sample``), because "no deterministic
      invariant failed" is explicitly not a quality estimate — a faithful,
      fluent and a merely well-formed translation both score 0.92.

    Everything below it is a hard defect (leaked prompt, dropped
    number, omission, broken formula) that routes to repair on its own flag; a
    judge cannot un-drop a number, so those pairs cost tokens for nothing.
    ``judge_calls`` records how many pairs actually hit the LLM (ROI observability).
    """

    def __init__(
        self,
        heuristic: HeuristicQERunner,
        judge: LLMJudgeQERunner | None = None,
        pass_sample: float = 0.0,
    ) -> None:
        self.heuristic = heuristic
        self.judge = judge
        self.pass_sample = min(1.0, max(0.0, pass_sample))
        self.judge_calls = 0
        # Count of ambiguous pairs where the judge produced no usable score
        # (call failure / unparsable). The heuristic score is kept for those
        # Pairs; this counter makes the silent fallback observable.
        self.judge_errors = 0
        #: Whether the judge returned a usable opinion this run. ``is_calibrated``
        #: reads it to say "these numbers are graded judgements, keep them" —
        #: without it the quality gate discarded every judge verdict.
        self._judged_any = False
        #: Indexes *within the last ``score_pairs`` batch* the judge actually
        #: produced an opinion for. ``judged_this_run`` says a batch contained
        #: at least one opinion; this says which entries carry one. Without it a
        #: partly-failed batch read as fully judged: the gate stored the
        #: heuristic bands of the entries the judge never answered as if they
        #: were measurements.
        self._judged_indices: frozenset[int] = frozenset()
        # Count every paid model call, not every block. Attached to the judge so
        # a sentence-level block contributes its real number of calls.
        if self.judge is not None:
            self.judge._on_call = self._count_judge_call

    def _count_judge_call(self) -> None:
        """One paid judge call finished (or was attempted)."""
        self.judge_calls += 1

    def _needs_judge(self, score: float, src: str) -> bool:
        """Whether this pair is worth a paid second opinion.

        Two cases only:

        * the heuristic rejected the block but no specific classifier matched
          (``JUDGED_CLASSES``) — the one deterministic verdict that is genuinely
          ambiguous, so a second opinion can separate "a real defect we cannot
          name" from "a rule with a false positive";
        * a stable sample of clean passes (``pass_sample``), because "no
          deterministic invariant failed" is not the same as "good".

        A numeric-fidelity failure is deliberately never sent: the judge cannot
        un-drop a number, so its opinion could only whitewash a real defect.
        """
        if score in JUDGED_CLASSES:
            return True
        if self.pass_sample <= 0.0 or score != QE_SCORE_PASS:
            return False
        # Stable sampling: the same segment is always or never judged, so a
        # rerun of one chapter does not audit a different set of blocks.
        return (zlib.crc32(src.encode("utf-8")) % 10_000) < int(self.pass_sample * 10_000)

    def is_glossary_aware(self) -> bool:
        """True when the underlying heuristic can see terminology drift.

        The judge never scores terminology, so the tier is only glossary-aware
        through its heuristic leg — which caps a violation below the ambiguous class
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

        Kept conservative on purpose: ``RepairLoop`` reads this to decide which
        repair lifecycle to run, and widening it would change repair behaviour
        rather than the reporting of a score. A judge verdict is kept by the
        quality gate through :attr:`judged_this_run` instead.
        """
        return self.heuristic.is_calibrated()

    @property
    def judged_this_run(self) -> bool:
        """Whether a judge opinion was actually obtained this run.

        The quality gate reads ``is_calibrated()`` to decide whether to keep the
        scores it computed, and ``TieredQERunner`` answers it conservatively —
        so every judge verdict was discarded: measured on arXiv 2606.23344, the
        judge said 0.78 for a block that had dropped a word and the gate stored
        ``None`` instead, leaving the block unable to reach repair on the score
        it had just been given.

        This says only "these numbers are judgement, not one of twelve discrete
        bands" — enough to keep them and route a block by them, not enough to
        rank candidates against each other.
        """
        return self._judged_any

    def judged_indices(self, count: int) -> frozenset[int]:
        """Indexes of the last batch the judge actually scored (``count`` entries).

        The gate reads this to keep a judgement for the blocks that got one and
        ``None`` for the rest, instead of applying one batch-wide verdict.
        """
        return frozenset(i for i in self._judged_indices if i < count)

    def may_be_calibrated(self) -> bool:
        """A tier has a judge worth calling even though the heuristic does not.

        ``HeuristicQERunner`` leaves this False so the gate never asks it; a tier
        must ask, or ``pass_sample`` and the judge would never run at all.
        """
        return True

    def score_source(self) -> str:
        """Name the leg that actually produced this run's scores.

        A tiered run is two engines: when the judge answered at least one pair
        the stored numbers include graded opinions, otherwise every number is a
        heuristic band. Reporting a flat "tiered" would hide that distinction
        from the audit artifact, which is the whole point of the label.
        """
        return "llm_judge" if self._judged_any else self.heuristic.score_source()

    def with_languages(self, source_lang: str, target_lang: str) -> TieredQERunner:
        heuristic = self.heuristic.with_languages(source_lang, target_lang)
        judge = self.judge.with_languages(source_lang, target_lang) if self.judge else None
        if heuristic is self.heuristic and judge is self.judge:
            return self
        rebound = TieredQERunner(
            heuristic=heuristic,
            judge=judge,
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
            pass_sample=self.pass_sample,
        )
        rebound.judge_calls = self.judge_calls
        rebound.judge_errors = self.judge_errors
        return rebound

    def with_document_context(self, document_text: str) -> TieredQERunner:
        """Forward the run's source text to the heuristic leg's omission gate.

        The judge scores semantics; the omission gate that decides "was this
        term dropped?" lives in the heuristic, so the context must reach it the
        same way the glossary does. Without this a tiered run kept the
        inner/outer gate split that already bit the heuristic path: the gate's
        external filter knew the document, ``score_pairs`` did not.
        """
        heuristic = self.heuristic.with_document_context(document_text)
        if heuristic is self.heuristic:
            return self
        rebound = TieredQERunner(
            heuristic=heuristic,
            judge=self.judge,
            pass_sample=self.pass_sample,
        )
        rebound.judge_calls = self.judge_calls
        rebound.judge_errors = self.judge_errors
        rebound._judged_any = self._judged_any
        rebound._judged_indices = self._judged_indices
        return rebound

    async def score_pairs(self, pairs: list[dict[str, str]]) -> list[float]:
        if not pairs:
            return []
        base = await self.heuristic.score_pairs(pairs)
        # ``_judged_any`` answers "did the judge produce a usable opinion for
        # *this* batch", and the gate reads it to decide whether the scores it
        # is about to store are judgements or heuristic bands. Leaving the
        # previous batch's value in place made a batch nobody judged inherit the
        # earlier one's answer: the gate then kept twelve discrete defect bands
        # as if they were graded opinions, and ``score_source()`` labelled them
        # ``llm_judge``.
        self._judged_any = False
        self._judged_indices = frozenset()
        if self.judge is None:
            return base
        ambiguous_idx = [
            i
            for i, (s, p) in enumerate(zip(base, pairs, strict=True))
            if self._needs_judge(s, p.get("src", ""))
        ]
        if not ambiguous_idx:
            return base
        subset = [pairs[i] for i in ambiguous_idx]
        judged = await self.judge.score_pairs(subset)
        # ``judge_calls`` is incremented by the judge itself, once per paid
        # model call (see ``LLMJudgeQERunner._on_call``): a block is not a call.
        #
        # ``_judged_any`` means "the judge returned at least one usable opinion",
        # not "a judge was asked". Setting it before inspecting the results made
        # a wholly failed judge batch read as graded: the quality gate then kept
        # the whole batch's heuristic bands as ``mtqe_score``, so blocks no judge
        # had ever scored entered the reported distribution as 0.10-0.60
        # "judgements". Also mark which blocks were actually judged, so the gate
        # can keep only those.
        self._judged_any = any(score != _JUDGE_NO_OPINION for score in judged)
        self._judged_indices = frozenset(
            i for i, s in zip(ambiguous_idx, judged, strict=True) if s != _JUDGE_NO_OPINION
        )
        merged = list(base)
        for i, s in zip(ambiguous_idx, judged, strict=True):
            if s == _JUDGE_NO_OPINION:
                # Judge failed for this pair: preserve the original heuristic
                # Score rather than silently adopting a fallback.
                self.judge_errors += 1
                continue
            # The judge may only LOWER a score. A judge verdict is a
            # probabilistic opinion, so it can legitimately say "this is worse
            # than the invariants suggest" but never "this is better": letting
            # it raise a block over the pass line would release a draft on an
            # opinion alone, which is the one thing this gate exists to prevent.
            # (``qe_judge_allow_upgrade`` used to permit exactly that and could
            # lift a real STRUCTURAL_OTHER failure to a pass; it is gone.)
            merged[i] = min(base[i], s)
        return merged
