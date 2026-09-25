"""Unit tests for best-of-n repair reranking (MBR-lite)."""

from __future__ import annotations

from typing import Any

import pytest

from ubt.core.engine.repair_loop import RepairLoop
from ubt.core.ir.models import BlockStatus, BlockType, FlowID, IRBlock
from ubt.core.qe.base import BaseQERunner
from ubt.core.qe.comet_runner import HeuristicQERunner
from ubt.core.router.provider import MockModelProvider
from ubt.core.router.router import ModelRouter


class _RotatingProvider(MockModelProvider):
    """Mock provider returning successive responses, one per generate() call."""

    def __init__(self, responses: list[str]) -> None:
        super().__init__(default_response=responses[0])
        self._responses = list(responses)
        self._i = 0

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = 0.3,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        self.call_history.append({"prompt": prompt, "temperature": temperature})
        response = self._responses[self._i % len(self._responses)]
        self._i += 1
        return response

    async def generate_with_images(
        self,
        prompt: str,
        images_b64_png: list[str],
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        return await self.generate(prompt, system_prompt, model, temperature)


class _ByTextQERunner(BaseQERunner):
    """Ranks candidates by a marker in the text (deterministic utility)."""

    async def score_pairs(self, pairs: list[dict[str, str]]) -> list[float]:
        return [1.0 if "BEST" in pair["mt"] else 0.4 for pair in pairs]

    async def score(self, src: str, mt: str) -> float:
        return (await self.score_pairs([{"src": src, "mt": mt}]))[0]


class _AlwaysFailFilter:
    """Structural filter stub that rejects every candidate.

    Implements both halves of the ``FastPassFilter`` seam the repair loop calls
    (``validate_structural_invariants`` for structure, ``evaluate`` for factual
    invariants); a stub that provides only one would exercise a different code
    path than production.
    """

    class _Decision:
        passed = False
        reason = "stub rejection"

    def validate_structural_invariants(self, source: str, target: str, **kwargs: Any) -> Any:
        return self._Decision()

    def evaluate(self, source: str, target: str, **kwargs: Any) -> Any:
        return self._Decision()


def _block(score: float = 0.2) -> IRBlock:
    return IRBlock(
        id="b1",
        flow_id=FlowID.MAIN_STORY,
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        source_text="A source sentence.",
        draft_text="草稿",
        target_text="草稿",
        status=BlockStatus.REPAIR_PENDING,
        mtqe_score=score,
    )


def _router(provider: MockModelProvider) -> ModelRouter:
    return ModelRouter(provider=provider, max_retries=0)


@pytest.mark.asyncio
async def test_rerank_picks_best_candidate() -> None:
    provider = _RotatingProvider(["candidate A", "BEST candidate", "candidate C"])
    loop = RepairLoop(router=_router(provider), qe_runner=_ByTextQERunner(), rerank_k=3)
    block = _block()
    out = await loop.repair_single_block(block=block, target_lang="zh", source_lang="en")
    assert len(provider.call_history) == 3
    assert out.target_text == "BEST candidate"
    assert out.status == BlockStatus.REPAIRED
    assert out.error_flags == []


@pytest.mark.asyncio
async def test_rerank_disabled_at_k1_uses_single_candidate() -> None:
    provider = _RotatingProvider(["candidate A", "BEST candidate"])
    loop = RepairLoop(router=_router(provider), qe_runner=_ByTextQERunner(), rerank_k=1)
    out = await loop.repair_single_block(block=_block(), target_lang="zh", source_lang="en")
    assert len(provider.call_history) == 1
    assert out.target_text == "candidate A"


@pytest.mark.asyncio
async def test_heuristic_runner_never_reranks() -> None:
    """The heuristic runner emits defect classes, not a ranking signal."""
    provider = _RotatingProvider(["candidate A", "BEST candidate", "candidate C"])
    loop = RepairLoop(router=_router(provider), qe_runner=HeuristicQERunner(), rerank_k=3)
    await loop.repair_single_block(block=_block(), target_lang="zh", source_lang="en")
    assert len(provider.call_history) == 1


@pytest.mark.asyncio
async def test_all_candidates_structurally_invalid_keeps_flags() -> None:
    provider = _RotatingProvider(["bad A", "bad B", "bad C"])
    loop = RepairLoop(
        router=_router(provider),
        qe_runner=_ByTextQERunner(),
        rerank_k=3,
        fast_pass=_AlwaysFailFilter(),  # type: ignore[arg-type]
    )
    out = await loop.repair_single_block(block=_block(), target_lang="zh", source_lang="en")
    assert any("repair_structural_failure" in flag for flag in out.error_flags)
    assert out.target_text == "草稿"  # original draft preserved


@pytest.mark.asyncio
async def test_rerank_falls_back_when_all_calls_fail() -> None:
    class _BoomProvider(MockModelProvider):
        async def generate(
            self,
            prompt: str,
            system_prompt: str | None = None,
            model: str | None = None,
            temperature: float | None = 0.3,
            max_tokens: int | None = None,
            reasoning_effort: str | None = None,
        ) -> str:
            raise RuntimeError("provider down")

    loop = RepairLoop(router=_router(_BoomProvider()), qe_runner=_ByTextQERunner(), rerank_k=3)
    # The first failure is re-raised so the repair stage's "Repair error:" path
    # still records it (reranking must not swallow a provider outage).
    with pytest.raises(Exception):  # noqa: B017 (wrapped provider error)
        await loop.repair_single_block(block=_block(), target_lang="zh", source_lang="en")
