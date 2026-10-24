"""Zero-token heuristic/QE runners (no PyTorch, no network).

Honesty note: the heuristic runner does NOT produce a calibrated quality score.
It maps a deterministic fast-pass rejection reason — plus the terminology signal
below — to a **discrete defect class** (see ``QE_DEFECT_CLASS_LEGEND``): a routing
signal, not a measurement. The ``0.92`` "pass" value carries no quality
information beyond "no deterministic invariant failed". Downstream surfaces
(report, progress events) must present it as such.

A fluent translation can use the WRONG enforced glossary term and pass every
structural check. The heuristic path therefore also checks the translation
against the run's glossary and classifies a violation as
``QE_SCORE_GLOSSARY_VIOLATION`` (:data:`GLOSSARY_VIOLATION_MARKER`), which the
quality gate refuses to auto-pass via ``STRUCTURAL_DEFECT_MARKERS``.
"""

from typing import Any

from ubt.core.qe.base import BaseQERunner
from ubt.core.qe.defect_taxonomy import ECHO_MARKERS
from ubt.core.qe.fast_pass import FastPassFilter
from ubt.core.qe.omission import DocumentVocabulary
from ubt.core.validators.consistency import GlossaryConsistencyValidator

# The exact score values the heuristic emits, and what each band
# actually means. Twelve discrete values — nothing between them is reachable.
# Single source of truth: the named constants below feed BOTH
# this legend and ``HeuristicQERunner.score_from_decision_reason`` so the two
# can never drift apart.
#
# IMPORTANT: these numbers are DIAGNOSTIC LABELS, not routing weights. The
# pipeline decides what to do with a block from its ``error_flags`` (see
# :func:`ubt.core.qe.defect_taxonomy.classify_defect`) and, for the terminal
# call, from ``qe_threshold``. The numeric value exists so a report can sort and
# display a distribution; it never selects a repair candidate on its own. An
# earlier design read these as severity weights and grew a second decision
# surface (``threshold * 0.8``) that no configuration could meaningfully move.
QE_SCORE_PASS = 0.92
QE_SCORE_STRUCTURAL_OTHER = 0.70
QE_SCORE_LENGTH = 0.60
QE_SCORE_NUMERIC = 0.55
QE_SCORE_SCRIPT_DENSITY = 0.40
QE_SCORE_OMISSION = 0.35
QE_SCORE_HTML_DELTA = 0.30
QE_SCORE_GLOSSARY_VIOLATION = 0.25
QE_SCORE_REPETITION = 0.20
QE_SCORE_FABRICATED = 0.15
QE_SCORE_LEAK = 0.10
QE_SCORE_EMPTY = 0.0

#: Error-flag fragment for a terminology violation. Registered in
#: ``ubt.core.qe.defect_taxonomy.STRUCTURAL_DEFECT_MARKERS`` so the quality
#: gate's existing ``has_structural_defect`` check keeps such a block out of
#: auto-pass even when a neural runner returns a high score.
GLOSSARY_VIOLATION_MARKER = "Glossary term violation"

#: The three outcomes the pipeline actually has, and the only three a consumer
#: should branch on. Everything else is a label on a report.
#:
#: ``PASS``                  — no deterministic invariant failed; ship it.
#: ``DETERMINISTIC_DEFECT``  — a named, re-checkable invariant failed (numeric
#:                             loss, dropped term, markup corruption, empty
#:                             target). Repair is *verifiable*: re-running the
#:                             same check decides whether the repair worked.
#: ``SEMANTIC_DOUBT``        — only a probabilistic judge disliked it. Repair is
#:                             *not* verifiable (the judge has its own noise), so
#:                             the block goes to a human queue instead of an
#:                             automatic rewrite.
DEFECT_OUTCOMES: tuple[str, ...] = ("PASS", "DETERMINISTIC_DEFECT", "SEMANTIC_DOUBT")

QE_DEFECT_CLASS_LEGEND: tuple[tuple[float, str], ...] = (
    (QE_SCORE_PASS, "pass — no deterministic invariant failed (NOT a quality estimate)"),
    (QE_SCORE_STRUCTURAL_OTHER, "other structural rejection (no specific classifier matched)"),
    (QE_SCORE_LENGTH, "suspicious length ratio (truncated or inflated target)"),
    (QE_SCORE_NUMERIC, "numeric fidelity failure (missing/altered figures)"),
    (QE_SCORE_SCRIPT_DENSITY, "script density anomaly (target not in expected script)"),
    (
        QE_SCORE_OMISSION,
        "omission suspected (dropped sentences / missing identifier terms / truncated residue)",
    ),
    (QE_SCORE_HTML_DELTA, "HTML tag delta failure (markup corrupted or dropped)"),
    (
        QE_SCORE_GLOSSARY_VIOLATION,
        "enforced glossary term dropped or altered in the target (terminology violation)",
    ),
    (QE_SCORE_REPETITION, "repetition-loop hallucination"),
    (QE_SCORE_FABRICATED, "untranslated residue / fabricated content vs empty source"),
    (QE_SCORE_LEAK, "prompt-template leak / empty-target via rejection path"),
    (QE_SCORE_EMPTY, "empty target (short-circuit in score_pairs)"),
)


def glossary_violation_flag(
    source_text: str,
    target_text: str,
    validator: GlossaryConsistencyValidator | None,
) -> str | None:
    """Terminology defect flag for a target that drops/alters an enforced term.

    Returns ``None`` when the check cannot run (no glossary) or the target
    carries every expected rendering. Detection is delegated to the
    export-side validator so QE and export can never disagree about what a
    violation is; building the validator once and passing it in keeps the
    per-block cost a single pass over the sorted glossary.
    """
    if validator is None:
        return None
    result = validator.validate(source_text, target_text)
    if result.is_valid:
        return None
    return f"{GLOSSARY_VIOLATION_MARKER}: {result.message}"


class MockQERunner(BaseQERunner):
    """Deterministic fast mock runner for unit and integration testing without PyTorch."""

    def __init__(self, default_score: float = 0.85) -> None:
        self.default_score = default_score

    def is_calibrated(self) -> bool:
        """False: a rehearsal's length-ratio number is not a quality estimate.

        Inheriting the base class's ``True`` made the dry-run path the only
        caller of ``RepairLoop``'s calibrated track — a "dead" branch that
        production never took but every ``--dry-run`` did, with score-comparison
        adoption semantics no real engine could reproduce. The mock scores a
        block by its length ratio; ranking repair candidates on that would be
        the same "presented as measured" problem the base class docstring warns
        about.
        """
        return False

    def score_source(self) -> str:
        """Label a rehearsal's synthetic scores so no report reads them as measured."""
        return "mock"

    async def score_pairs(self, pairs: list[dict[str, str]]) -> list[float]:
        scores: list[float] = []
        for p in pairs:
            src = p.get("src", "")
            mt = p.get("mt", "")
            # Deterministic variation based on length matching
            if not mt:
                scores.append(0.0)
            elif "<issues>" in mt:
                scores.append(0.35)
            else:
                ratio = min(len(mt), len(src)) / max(1, max(len(mt), len(src)))
                score = min(1.0, max(0.5, self.default_score * (0.8 + 0.2 * ratio)))
                scores.append(round(score, 4))
        return scores


class HeuristicQERunner(BaseQERunner):
    """Deterministic zero-token heuristic QE runner based on structural, lexical, and script invariants.

    Provides a production-grade default score (0.0 ~ 1.0) without requiring PyTorch or HuggingFace models.

    the score is a **discrete defect-class proxy** (see
    ``QE_DEFECT_CLASS_LEGEND``), not a calibrated quality estimate — there
    are only twelve reachable values and the gaps between them are
    meaningless. Use it for routing/repair triggering; use the tiered
    LLM judge (``LLMJudgeQERunner``) when an assessment of gray-zone candidates is
    needed.

    When constructed with the run's ``glossary``, a target that drops or
    alters an enforced term scores ``QE_SCORE_GLOSSARY_VIOLATION`` instead of
    the pass band.
    """

    def __init__(
        self,
        target_lang: str = "zh",
        source_lang: str = "en",
        glossary: list[dict[str, Any]] | None = None,
    ) -> None:
        self.target_lang = target_lang
        self.source_lang = source_lang
        self.fast_pass = FastPassFilter(source_lang=source_lang, target_lang=target_lang)
        self._glossary = glossary or None
        #: The run's source text, once the gate binds it (see
        #: :meth:`with_document_context`). Empty means "not bound yet".
        self._document_text: str = ""
        #: The indexed form of that text, when the run shares one (see
        #: :meth:`with_document_vocabulary`). Preferred over re-indexing.
        self._document_vocabulary: DocumentVocabulary | None = None
        self._glossary_validator = (
            GlossaryConsistencyValidator(glossary=glossary) if glossary else None
        )

    def with_languages(self, source_lang: str, target_lang: str) -> "HeuristicQERunner":
        if (self.source_lang, self.target_lang) == (source_lang, target_lang):
            return self
        rebound = HeuristicQERunner(
            target_lang=target_lang,
            source_lang=source_lang,
            glossary=self._glossary,
        )
        # Carry the bound context across a language rebind, like the glossary:
        # a runner that forgot it would reproduce the inner/outer gate split.
        if self._document_vocabulary is not None:
            rebound.with_document_vocabulary(self._document_vocabulary)
        elif self._document_text:
            rebound.with_document_context(self._document_text)
        return rebound

    def with_glossary(self, glossary: list[dict[str, Any]] | None) -> "HeuristicQERunner":
        """Bind the run's enforced terminology once the bible stage has it.

        The orchestrator constructs the runner before the glossary exists;
        this rebinds it so the repair loop's re-score sees a term violation
        instead of passing the repaired text at 0.92.

        The omission gate inside :attr:`fast_pass` is taught the same glossary.
        The draft stage does this for its own filter, but the quality gate scores
        through *this* runner's filter, which had never been told — so the two
        gates contradicted each other in the gate path even though the draft
        path was correct. Measured on arXiv 2606.23344 b0006: the glossary asks
        ``state-of-the-art -> 当前最先进的``, the repair obeyed it and wrote
        ``当前最先进的准确性``, and the omission gate then reported "identifier
        term(s) missing from target: ['SOTA']" and capped a correct repair at
        0.35 — the very failure the draft stage's comment describes.
        """
        if not glossary or glossary is self._glossary:
            return self
        bound = HeuristicQERunner(
            target_lang=self.target_lang,
            source_lang=self.source_lang,
            glossary=glossary,
        )
        bound.fast_pass.set_glossary(glossary)
        # Re-bind the document context this runner may already carry: rebinding
        # the glossary must not silently drop it (the quality gate calls this
        # before scoring, so the two would disagree again).
        if self._document_vocabulary is not None:
            bound.with_document_vocabulary(self._document_vocabulary)
        elif self._document_text:
            bound.fast_pass.set_document_context(self._document_text)
            bound._document_text = self._document_text
        return bound

    def with_document_context(self, document_text: str) -> "HeuristicQERunner":
        """Teach the omission gate the run's source text, as the draft stage does.

        The draft stage binds ``set_document_context`` onto its own filter, and
        the gate screens blocks with the *external* filter, but ``score_pairs``
        runs through this runner's internal ``fast_pass`` -- which was told the
        glossary and never the document. An uppercase table header, an
        abbreviation expanded elsewhere in the book, or any identifier the
        source states once then refers to by another form was judged "missing
        from target" by the inner gate while the outer one accepted the block.
        Measured: a block that passed the gate came back at 0.35 from
        ``score_pairs``, and ``is_shippable`` saw the wrong number.

        Idempotent, and returns ``self`` (the runner is rebound in place by the
        gate, which holds the reference).
        """
        if not document_text or document_text == self._document_text:
            return self
        self._document_text = document_text
        self._document_vocabulary = None
        self.fast_pass.set_document_context(document_text)
        return self

    def with_document_vocabulary(
        self, vocabulary: DocumentVocabulary | None
    ) -> "HeuristicQERunner":
        """Bind an already-indexed document (see ``DocumentVocabulary``).

        The run indexes its source text once (``SourceDocumentIndex``) and hands
        the same object to the draft filter and to this runner, so neither
        re-indexes the book per chapter. Returns ``self``.
        """
        if vocabulary is None:
            return self
        self._document_vocabulary = vocabulary
        self.fast_pass.set_document_vocabulary(vocabulary)
        return self

    def is_glossary_aware(self) -> bool:
        """True once a glossary is bound, so a term violation caps the score."""
        return self._glossary_validator is not None

    def is_calibrated(self) -> bool:
        """Always False: the twelve legend bands are a routing class, not a
        graded quality estimate, so they must never rank repair candidates."""
        return False

    def score_source(self) -> str:
        """The discrete defect-band engine, named as the report has always called it."""
        return "heuristic"

    def glossary_violation(self, source_text: str, target_text: str) -> str | None:
        """Terminology defect flag for this pair, or ``None`` (granularity-free)."""
        return glossary_violation_flag(source_text, target_text, self._glossary_validator)

    @staticmethod
    def score_from_decision_reason(reason: str) -> float:
        """Map a fast-pass structural rejection reason to its defect class.

        despite the "score" name this is a classification, not a
        graded measurement — each rejection reason maps to one of the fixed
        bands in ``QE_DEFECT_CLASS_LEGEND``.
        """
        r = reason.lower()
        if "empty" in r or "leak" in r:
            return QE_SCORE_LEAK
        if any(marker.lower() in r for marker in ECHO_MARKERS):
            # Same class as the fabricated-content band. Scoring a
            # near-verbatim untranslated target as QE_SCORE_STRUCTURAL_OTHER
            # (0.70) would leave it close enough to the default threshold to be
            # auto-passed, so it maps to the fabricated band instead.
            return QE_SCORE_FABRICATED
        if "hallucination" in r:
            return QE_SCORE_REPETITION
        if "added reference" in r or "fabricated" in r or "prompt scaffold" in r:
            # Fabricated content the source never had, including fabricated
            # citations and equation numbers. Without this branch the reason falls through
            # to 0.70 ("no specific classifier matched"), which is only 0.05
            # below the default threshold and reads as a generic structural
            # problem in the report's score distribution.
            # NOTE: scaffold *headings* ("Prompt scaffold leaked into the
            # target...") intentionally stay in the leak band above — see
            # test_d2_echo_regression. Only non-"leak" scaffold phrasings
            # reach this branch.
            return QE_SCORE_FABRICATED
        if "html" in r or "table" in r or "math" in r or "latex" in r:
            return QE_SCORE_HTML_DELTA
        if "omission" in r:
            return QE_SCORE_OMISSION
        if "script density" in r or "carries no letters at all" in r:
            return QE_SCORE_SCRIPT_DENSITY
        if "numeric" in r:
            return QE_SCORE_NUMERIC
        if "truncated" in r or "inflated" in r:
            return QE_SCORE_LENGTH
        return QE_SCORE_STRUCTURAL_OTHER

    async def score_pairs(self, pairs: list[dict[str, str]]) -> list[float]:
        scores: list[float] = []
        for p in pairs:
            src = p.get("src", "").strip()
            mt = p.get("mt", "").strip()
            if not mt:
                scores.append(QE_SCORE_EMPTY)
                continue
            if not src:
                # Non-empty MT against an empty source is fabricated content,
                # not a flawless translation — score it like untranslated residue.
                scores.append(QE_SCORE_FABRICATED)
                continue

            # Untranslated identical prose is decided in one place:
            # ``FastPassFilter.evaluate`` below rejects the echo and
            # ``score_from_decision_reason`` maps it to the fabricated band.
            decision = self.fast_pass.evaluate(src, mt, block_type=p.get("block_type"))
            if not decision.passed:
                scores.append(self.score_from_decision_reason(decision.reason))
                continue
            # Passing every structural gate is not enough — the target must
            # also carry the enforced glossary renderings.
            if self.glossary_violation(src, mt) is not None:
                scores.append(QE_SCORE_GLOSSARY_VIOLATION)
            else:
                scores.append(QE_SCORE_PASS)
        return scores
