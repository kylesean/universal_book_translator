"""The translatable-content ratio: measuring translation over the right blocks.

``cjk_ratio`` (the whole-document CJK fraction) is a *delivery* measure: it counts
the bibliography, the author affiliations and every code listing that the policy
deliberately keeps in the source language. That is correct as a delivery figure
and misleading as a translation figure — measured on the gold corpus the two
diverge by ~0.3 (one gpt3-sized job reads 0.475 whole-document against 0.792 over
the blocks that were meant to be translated), enough to make a working pipeline
look like it barely translated anything.

``ledger.cjk_ratio_translatable`` is the second measure, over exactly the blocks
the renderer treated as translated. These tests pin the population selection —
the part that can silently drift from
:func:`ubt.render.typst.document_renderer._keeps_source`.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from scripts.eval_semantic import _ledger_stats

pytestmark = pytest.mark.fast


def _ledger(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    """A minimal ledger with the columns ``_ledger_stats`` reads."""
    path = tmp_path / "job_test.sqlite"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE blocks ("
        " status TEXT, error_flags_json TEXT, target_text TEXT, draft_text TEXT,"
        " source_text TEXT, skip_translate INTEGER, policy_translate INTEGER)"
    )
    for row in rows:
        conn.execute(
            "INSERT INTO blocks VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                row.get("status", "delivered"),
                row.get("error_flags_json", "[]"),
                row.get("target_text"),
                row.get("draft_text"),
                row.get("source_text", ""),
                row.get("skip_translate", 0),
                row.get("policy_translate"),
            ),
        )
    conn.commit()
    conn.close()
    return path


def test_kept_blocks_are_excluded_from_the_translatable_ratio(tmp_path: Path) -> None:
    """A bibliography entry must not depress the translation measure.

    The block was *supposed* to stay English, so counting it says nothing about
    whether translation happened — which is the whole reason the second metric
    exists.
    """
    ledger = _ledger(
        tmp_path,
        [
            {"target_text": "这是一段翻译得很好的中文段落内容", "skip_translate": 0},
            {"target_text": "这是一段翻译得很好的中文段落内容", "skip_translate": 0},
            # Deliberately kept in source: an English bibliography entry.
            {
                "target_text": "Vaswani, A. Attention is all you need. NeurIPS 2017.",
                "skip_translate": 1,
            },
        ],
    )
    stats = _ledger_stats(ledger)

    assert stats["translatable_blocks"] == 2
    assert stats["source_kept_blocks"] == 1
    assert stats["cjk_ratio_translatable"] == pytest.approx(1.0)


def test_a_policy_verdict_outranks_the_extractor_default(tmp_path: Path) -> None:
    """``policy_translate=False`` means the same as ``skip_translate``.

    Mirrors ``_keeps_source`` in the renderer: the policy verdict is authoritative
    when it exists, so a block the policy keeps is excluded even when the
    extractor's own flag was not set.
    """
    ledger = _ledger(
        tmp_path,
        [
            {"target_text": "中文译文", "skip_translate": 0, "policy_translate": 0},
            {"target_text": "另一段中文译文", "skip_translate": 0, "policy_translate": 1},
        ],
    )
    stats = _ledger_stats(ledger)

    assert stats["source_kept_blocks"] == 1
    assert stats["translatable_blocks"] == 1


def test_a_policy_verdict_of_true_does_not_clear_a_skip_flag(tmp_path: Path) -> None:
    """The rule is the renderer's exactly: policy ``False`` keeps; else ``skip``.

    ``_keeps_source`` returns ``True`` for a block the policy rejected and, for
    every other block, defers to the extractor's ``skip_translate``. A policy
    verdict of ``True`` therefore does *not* clear a skip flag — asserting the
    two implementations agree on that asymmetry is what stops the metric and the
    renderer from measuring different populations.
    """
    ledger = _ledger(
        tmp_path,
        [
            {
                "target_text": "这是译文内容",
                "skip_translate": 1,
                "policy_translate": 1,
            },
        ],
    )
    stats = _ledger_stats(ledger)

    from ubt.core.ir.models import BlockType, IRBlock, make_element
    from ubt.render.typst.document_renderer import _keeps_source

    element = make_element(
        id="b1",
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        source_text="x",
        level=1,
        skip_translate=True,
    )
    renderer_says_kept = _keeps_source(IRBlock(element=element, policy_translate=True))
    assert renderer_says_kept is True

    # The metric must agree with the renderer, block for block.
    assert stats["source_kept_blocks"] == 1
    assert stats["translatable_blocks"] == 0


def test_a_draft_substitutes_for_a_missing_target(tmp_path: Path) -> None:
    """An untargeted block's draft is what shipped, so it is what is measured."""
    ledger = _ledger(
        tmp_path,
        [{"target_text": None, "draft_text": "草稿中的中文内容", "skip_translate": 0}],
    )
    stats = _ledger_stats(ledger)

    assert stats["cjk_ratio_translatable"] == pytest.approx(1.0)


def test_no_translatable_block_yields_none_not_zero(tmp_path: Path) -> None:
    """An unmeasurable ratio is ``None``, never 0.0.

    ``0.0`` reads as "nothing was translated"; ``None`` reads as "not measured".
    A document of pure references must not look like a total failure.
    """
    ledger = _ledger(
        tmp_path,
        [{"target_text": "Vaswani, A. NeurIPS 2017.", "skip_translate": 1}],
    )
    stats = _ledger_stats(ledger)

    assert stats["cjk_ratio_translatable"] is None


def test_both_ratios_are_reported_so_the_gap_is_visible(tmp_path: Path) -> None:
    """The whole-document figure stays available; the two are read together."""
    ledger = _ledger(
        tmp_path,
        [
            {"target_text": "中文", "skip_translate": 0},
            {"target_text": "averyverylongenglishbibliographyentry", "skip_translate": 1},
        ],
    )
    stats = _ledger_stats(ledger)

    # Whole-document lives in the PDF-derived ``cjk_ratio``; the ledger reports
    # the translatable one plus the block accounting that explains the gap.
    assert stats["cjk_ratio_translatable"] == pytest.approx(1.0)
    assert stats["total_blocks"] == 2
    assert stats["source_kept_blocks"] == 1
