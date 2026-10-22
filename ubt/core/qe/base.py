"""The MTQE scoring abstraction, kept apart from any concrete engine.

Every collaborator that needs "a QE runner" (the pipeline, the repair loop, the
stage context, the API and worker entry points) depends on this interface, not on
a realization of it. Keeping the interface here means an importer of the
abstraction does not drag in the COMET/subprocess implementation module — and
with it the subprocess runner's licensing warning and its IPC assumptions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseQERunner(ABC):
    """Abstract interface for local MTQE evaluation engines."""

    @abstractmethod
    async def score_pairs(self, pairs: list[dict[str, str]]) -> list[float]:
        """Score a list of ``{'src': ..., 'mt': ...}`` pairs → floats (0.0~1.0).

        A pair may carry an optional ``'block_type'`` (the IR block type's
        string value). The heuristic runner gates its prose-only invariants on
        it — the omission and script-density checks weigh a ``table`` or
        ``code`` block differently — so a caller that drops the key scores a
        table as prose and rejects repairs the prose rules should not touch.
        Neural runners ignore the key.
        """
        raise NotImplementedError

    async def score(self, src: str, mt: str, block_type: object | None = None) -> float:
        """Score a single (src, mt) pair returning a float score (0.0~1.0)."""
        pair = {"src": src, "mt": mt}
        if block_type is not None:
            pair["block_type"] = str(getattr(block_type, "value", block_type))
        scores = await self.score_pairs([pair])
        return scores[0] if scores else 0.0

    def with_languages(self, source_lang: str, target_lang: str) -> BaseQERunner:
        """Return a runner bound to the given language pair.

        Language-agnostic runners return themselves; the
        heuristic runner rebuilds its ``FastPassFilter`` so per-run language
        thresholds are never shared across concurrent runs.
        """
        return self

    def is_glossary_aware(self) -> bool:
        """Whether this runner's score already accounts for terminology drift.

        Neural and LLM-judge runners score fluency/adequacy, not whether an
        enforced glossary term survived, so a high re-score from them is no
        evidence a dropped term came back. Only the glossary-bound heuristic
        runner can answer for terminology; callers that clear a
        ``Glossary term violation`` flag on a clean re-score must consult this
        first, or they wash the defect into a shippable state.
        """
        return False

    def is_calibrated(self) -> bool:
        """Whether this runner's scores measure translation quality.

        Best-of-n reranking picks the candidate with the highest score, which
        is only sound when the score is a graded quality estimate. The
        heuristic runner emits discrete defect classes (see
        ``QE_DEFECT_CLASS_LEGEND``) — a tie among candidates that all cleared
        the same gate would be broken at random and presented as a measured
        preference. Runners that cannot prove their calibrated
        model scored must flip this to ``False``.
        """
        return True

    def score_source(self) -> str:
        """The engine label the quality report records for this run's scores.

        A run's audit artifact has to say *what* produced its ``mtqe_score``
        values, because the same number means different things depending on the
        engine: a graded judge opinion, a discrete heuristic defect band, or a
        rehearsal echo. The quality gate persists this on the job and the
        reporter publishes it as ``qe_score_source``.

        Concrete runners override this with a stable literal: deriving it from
        the class name would let a rename silently change an audit string that
        old reports already carry.
        """
        return "unknown"

    def may_be_calibrated(self) -> bool:
        """Whether it is worth *attempting* a scoring call to find out.

        Distinguishes "cannot score" from "has not proven it scored yet". A
        deferred runner may report ``is_calibrated() == False`` until its
        first batch answers, so a caller that gates the
        scoring call on ``is_calibrated()`` would never make that first call —
        the runner stayed permanently uncalibrated and every block got
        ``mtqe_score=None``. Such runners override this to ``True`` so the gate
        attempts a batch and re-reads ``is_calibrated()`` afterwards; a runner
        that can never be calibrated (the heuristic) leaves it ``False`` and is
        never asked.
        """
        return self.is_calibrated()

    def reset_residency(self) -> None:
        """Drop latched "fall back to the slow path" state at the start of a run.

        A runner that keeps resident state latches the resident path off after
        a protocol desync. That latch is per-run state, but the API keeps one
        runner per process across jobs: a fresh run deserves a fresh attempt
        instead of inheriting the previous job's degradation. No-op for runners
        with nothing resident.
        """
        return None

    async def aclose(self) -> None:
        """Release runner-owned subprocesses and other asynchronous resources."""
        return None
