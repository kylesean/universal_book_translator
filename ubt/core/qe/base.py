"""The MTQE scoring abstraction, kept apart from any concrete engine.

Every collaborator that needs "a QE runner" (the pipeline, the repair loop, the
stage context, the API and worker entry points) depends on this interface, not on
a realization of it. It used to live in ``comet_runner``, so importers of the
abstraction had to import the COMET/subprocess implementation module — and along
with it the subprocess runner's licensing warning and its IPC assumptions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseQERunner(ABC):
    """Abstract interface for local MTQE evaluation engines."""

    @abstractmethod
    async def score_pairs(self, pairs: list[dict[str, str]]) -> list[float]:
        """Score a list of {'src': ..., 'mt': ...} pairs returning float scores (0.0~1.0)."""
        pass

    async def score(self, src: str, mt: str) -> float:
        """Score a single (src, mt) pair returning a float score (0.0~1.0)."""
        scores = await self.score_pairs([{"src": src, "mt": mt}])
        return scores[0] if scores else 0.0

    def with_languages(self, source_lang: str, target_lang: str) -> BaseQERunner:
        """Return a runner bound to the given language pair.

        Language-agnostic runners (subprocess/neural) return themselves; the
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
        preference. Runners that cannot prove they scored with their neural
        model (e.g. a COMET subprocess whose weights failed to import and fell
        back to the heuristic) must flip this to ``False``.
        """
        return True

    def reset_residency(self) -> None:
        """Drop latched "fall back to the slow path" state at the start of a run.

        A runner that keeps a heavy model process alive (the COMET subprocess)
        latches the resident path off after a protocol desync. That latch is
        per-run state, but the API keeps one runner per process across jobs: a
        fresh run deserves a fresh attempt instead of inheriting the previous
        job's degradation. No-op for runners with nothing resident.
        """
        return None

    async def aclose(self) -> None:
        """Release runner-owned subprocesses and other asynchronous resources."""
        return None
