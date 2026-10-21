"""Whether a delivered file still matches the ledger it was rendered from.

A human edit lands in the ledger. The PDF/EPUB/HTML/XLIFF on disk were written
by an earlier export and nothing rewrites them. The download endpoint used to
serve that pre-edit file with no warning; the console tracked the mismatch in a
React `useState`, so reloading the page was enough to lose the warning while
keeping the staleness.

These tests pin the ledger-side answer, which is the part every surface shares.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ubt.core.engine.delivery_freshness import (
    RENDERED_TEXT_DIGEST_KEY,
    is_delivery_stale,
    record_rendered_digest,
    text_digest,
)
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.ir.models import (
    BlockStatus,
    BlockType,
    BookManifest,
    BoundingBox,
    ChapterIR,
    IRBlock,
    make_element,
)

pytestmark = pytest.mark.fast

_JOB = "job_freshness"


def _block(index: int, target: str, status: BlockStatus = BlockStatus.REPAIRED) -> IRBlock:
    """A real IRBlock: the ledger stores blocks through the IR, not a stub."""
    element = make_element(
        id=f"b{index:03d}",
        spine_index=index,
        block_type=BlockType.NARRATIVE,
        source_text=f"source {index}",
        bbox=BoundingBox(page=1, x0=50.0, y0=700.0 - index * 20, x1=350.0, y1=712.0 - index * 20),
    )
    block = IRBlock(element=element)
    block.target_text = target
    block.status = status
    return block


def _ledger(tmp_path: Path, blocks: list[IRBlock]) -> SQLiteJobLedger:
    ledger = SQLiteJobLedger(tmp_path / f"{_JOB}.sqlite")
    ledger.init_job_from_manifest(
        _JOB,
        BookManifest(
            doc_id="doc",
            title="book",
            source_path=str(tmp_path / "book.pdf"),
            target_lang="zh",
        ),
    )
    ledger.append_chapter(
        _JOB,
        ChapterIR(doc_id="doc", chapter_id="c0", title="c", spine_index=0, blocks=blocks),
    )
    return ledger


def test_a_job_with_no_recorded_digest_is_not_stale(tmp_path: Path) -> None:
    """Jobs finished before this marker existed must not be blocked.

    Calling them stale would refuse every download of work that is in fact
    fine — the marker only earns its keep once a render has recorded one.
    """
    ledger = _ledger(tmp_path, [_block(1, "译文")])
    assert ledger.get_job_metadata_value(_JOB, RENDERED_TEXT_DIGEST_KEY) is None
    assert is_delivery_stale(ledger, _JOB) is False


def test_a_render_settles_the_job(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path, [_block(1, "译文")])
    digest = record_rendered_digest(ledger, _JOB)
    assert ledger.get_job_metadata_value(_JOB, RENDERED_TEXT_DIGEST_KEY) == digest
    assert is_delivery_stale(ledger, _JOB) is False


def test_a_human_edit_makes_the_delivery_stale(tmp_path: Path) -> None:
    """The whole point: editing without re-rendering must be detectable."""
    ledger = _ledger(tmp_path, [_block(1, "机器译文")])
    record_rendered_digest(ledger, _JOB)
    ledger.save_checkpoints_batch(
        [{"block_id": "b001", "status": "repaired", "target_text": "人工校订译文"}], job_id=_JOB
    )
    assert is_delivery_stale(ledger, _JOB) is True


def test_a_re_render_clears_the_staleness(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path, [_block(1, "机器译文")])
    record_rendered_digest(ledger, _JOB)
    ledger.save_checkpoints_batch(
        [{"block_id": "b001", "status": "repaired", "target_text": "人工校订译文"}], job_id=_JOB
    )
    record_rendered_digest(ledger, _JOB)
    assert is_delivery_stale(ledger, _JOB) is False


def test_a_score_only_change_does_not_make_the_delivery_stale(tmp_path: Path) -> None:
    """Scores go to the quality report, not to the delivered text.

    Marking a job stale for a rescored block would send the operator through a
    full re-render to change nothing in the file.
    """
    ledger = _ledger(tmp_path, [_block(1, "译文")])
    record_rendered_digest(ledger, _JOB)
    ledger.save_checkpoints_batch(
        [{"block_id": "b001", "status": "repaired", "target_text": "译文", "mtqe_score": 0.42}],
        job_id=_JOB,
    )
    assert is_delivery_stale(ledger, _JOB) is False


def test_a_status_flip_alone_makes_the_delivery_stale(tmp_path: Path) -> None:
    """Status decides which text ships.

    A block that moves to NEEDS_HUMAN after a failed repair renders its source
    rather than its draft, so the delivered file changes even though
    ``target_text`` does not.
    """
    ledger = _ledger(tmp_path, [_block(1, "译文", status=BlockStatus.REPAIRED)])
    record_rendered_digest(ledger, _JOB)
    ledger.save_checkpoints_batch(
        [{"block_id": "b001", "status": "needs_human", "target_text": "译文"}], job_id=_JOB
    )
    assert is_delivery_stale(ledger, _JOB) is True


def test_the_digest_is_order_independent(tmp_path: Path) -> None:
    """A re-read that returns rows in another order is not a content change."""
    forward = text_digest([_block(1, "1"), _block(2, "2")])
    backward = text_digest([_block(2, "2"), _block(1, "1")])
    assert forward == backward
    assert text_digest([_block(1, "1"), _block(2, "3")]) != forward


def test_the_digest_survives_a_process_restart(tmp_path: Path) -> None:
    """The marker is ledger data, which is why it beats the old UI flag."""
    ledger = _ledger(tmp_path, [_block(1, "机器译文")])
    record_rendered_digest(ledger, _JOB)
    ledger.save_checkpoints_batch(
        [{"block_id": "b001", "status": "repaired", "target_text": "人工校订"}], job_id=_JOB
    )

    # A brand-new ledger object, as a restarted server or another surface sees it.
    reopened = SQLiteJobLedger(tmp_path / f"{_JOB}.sqlite")
    assert is_delivery_stale(reopened, _JOB) is True
