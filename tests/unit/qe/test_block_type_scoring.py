"""QE scoring must know a block's type, or it grades a table as prose.

``FastPassFilter.evaluate`` branches on ``block_type``: the prose-only echo and
residue checks and the omission gate's table mode all key off it. The
``HeuristicQERunner`` — the default zero-token scorer — called ``evaluate``
without it, so every block scored as prose at the QE/repair re-scoring step
while the quality-gate stage scored the same block with its real type. Two
answers for one block, and a table row judged by rules written for paragraphs.

These tests pin the plumbing: the runner reads ``block_type`` from the pair, and
the callers that build pairs supply it.
"""

from __future__ import annotations

import asyncio

import pytest

from ubt.core.engine.stages.quality_gate import _audit_pass_sample
from ubt.core.ir.models import BlockType, IRBlock
from ubt.core.qe.base import BaseQERunner
from ubt.core.qe.comet_runner import (
    QE_SCORE_FABRICATED,
    QE_SCORE_SCRIPT_DENSITY,
    HeuristicQERunner,
)
from ubt.model.ast import Table
from ubt.model.span import Span

pytestmark = pytest.mark.fast

#: An untranslated echo: rejected as fabricated residue on a prose block, but
#: the prose residue gate does not apply to a code/table block, which then
#: trips the (script-density) gate instead — a different band, same text.
_ECHO = "def compute(x): return x + 1"


def test_the_runner_gates_on_the_pairs_block_type() -> None:
    runner = HeuristicQERunner()

    prose = asyncio.run(runner.score_pairs([{"src": _ECHO, "mt": _ECHO}]))
    code = asyncio.run(runner.score_pairs([{"src": _ECHO, "mt": _ECHO, "block_type": "code"}]))
    table = asyncio.run(runner.score_pairs([{"src": _ECHO, "mt": _ECHO, "block_type": "table"}]))

    assert prose == [QE_SCORE_FABRICATED]
    assert code == [QE_SCORE_SCRIPT_DENSITY]
    assert table == code, "code and table must both leave the prose-only gates"


def test_a_prose_block_type_is_the_same_as_no_block_type() -> None:
    runner = HeuristicQERunner()
    omitted = asyncio.run(runner.score_pairs([{"src": _ECHO, "mt": _ECHO}]))
    explicit = asyncio.run(
        runner.score_pairs([{"src": _ECHO, "mt": _ECHO, "block_type": "narrative"}])
    )
    assert omitted == explicit


class _CapturingRunner(BaseQERunner):
    """Records the pairs it is handed and returns a fixed score."""

    def __init__(self, score: float = 0.5) -> None:
        self.seen: list[dict[str, str]] = []
        self._score = score

    async def score_pairs(self, pairs: list[dict[str, str]]) -> list[float]:
        self.seen.extend(pairs)
        return [self._score] * len(pairs)


def test_the_single_pair_helper_forwards_the_block_type() -> None:
    runner = _CapturingRunner()
    assert asyncio.run(runner.score("a", "b", block_type=BlockType.TABLE)) == 0.5
    assert runner.seen == [{"src": "a", "mt": "b", "block_type": "table"}]

    asyncio.run(runner.score("a", "b"))
    assert runner.seen[1] == {"src": "a", "mt": "b"}, "no block type, no key"


def test_batch_callers_send_the_block_type(monkeypatch: pytest.MonkeyPatch) -> None:
    element = Table(id="b1", spine_index=0, span=Span(page=1), markup="| a | b |")
    block = IRBlock(element=element, block_type=BlockType.TABLE, target_text="| 甲 | 乙 |")
    runner = _CapturingRunner()
    monkeypatch.setattr(runner, "pass_sample", 0.25, raising=False)

    asyncio.run(_audit_pass_sample(runner, [block], threshold=0.9))

    assert runner.seen == [
        {"src": "| a | b |", "mt": "| 甲 | 乙 |", "block_type": "table"},
    ]
