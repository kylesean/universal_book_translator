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
    ledger = _ledger(tmp_path, [_block(1, "译文")])
    assert ledger.get_job_metadata_value(_JOB, RENDERED_TEXT_DIGEST_KEY) is None
    assert is_delivery_stale(ledger, _JOB) is False


def test_a_render_settles_the_job(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path, [_block(1, "译文")])
    digest = record_rendered_digest(ledger, _JOB)
    assert ledger.get_job_metadata_value(_JOB, RENDERED_TEXT_DIGEST_KEY) == digest
    assert is_delivery_stale(ledger, _JOB) is False


def test_a_human_edit_makes_the_delivery_stale(tmp_path: Path) -> None:
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
    ledger = _ledger(tmp_path, [_block(1, "译文")])
    record_rendered_digest(ledger, _JOB)
    ledger.save_checkpoints_batch(
        [{"block_id": "b001", "status": "repaired", "target_text": "译文", "mtqe_score": 0.42}],
        job_id=_JOB,
    )
    assert is_delivery_stale(ledger, _JOB) is False


def test_a_status_flip_alone_makes_the_delivery_stale(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path, [_block(1, "译文", status=BlockStatus.REPAIRED)])
    record_rendered_digest(ledger, _JOB)
    ledger.save_checkpoints_batch(
        [{"block_id": "b001", "status": "needs_human", "target_text": "译文"}], job_id=_JOB
    )
    assert is_delivery_stale(ledger, _JOB) is True


def test_the_digest_is_order_independent(tmp_path: Path) -> None:
    forward = text_digest([_block(1, "1"), _block(2, "2")])
    backward = text_digest([_block(2, "2"), _block(1, "1")])
    assert forward == backward
    assert text_digest([_block(1, "1"), _block(2, "3")]) != forward


def test_the_digest_survives_a_process_restart(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path, [_block(1, "机器译文")])
    record_rendered_digest(ledger, _JOB)
    ledger.save_checkpoints_batch(
        [{"block_id": "b001", "status": "repaired", "target_text": "人工校订"}], job_id=_JOB
    )

    reopened = SQLiteJobLedger(tmp_path / f"{_JOB}.sqlite")
    assert is_delivery_stale(reopened, _JOB) is True


def test_finalize_and_digest_land_in_one_transaction(tmp_path: Path) -> None:
    from ubt.core.engine.delivery_freshness import rendered_digest_for

    ledger = _ledger(tmp_path, [_block(1, "译文")])
    digest = rendered_digest_for(ledger, _JOB)

    ledger.finalize_job(_JOB, status="completed", rendered_digest=digest)

    snapshot = ledger.get_job_snapshot(_JOB)
    assert snapshot is not None
    assert snapshot["status"] == "completed"
    assert ledger.get_job_metadata_value(_JOB, RENDERED_TEXT_DIGEST_KEY) == digest
    assert is_delivery_stale(ledger, _JOB) is False


def test_a_rejected_finalize_writes_no_digest(tmp_path: Path) -> None:
    from ubt.core.engine.delivery_freshness import rendered_digest_for
    from ubt.core.exceptions import LedgerError

    ledger = _ledger(tmp_path, [_block(1, "译文", status=BlockStatus.PENDING)])
    digest = rendered_digest_for(ledger, _JOB)

    with pytest.raises(LedgerError):
        ledger.finalize_job(_JOB, status="completed", rendered_digest=digest)

    assert ledger.get_job_metadata_value(_JOB, RENDERED_TEXT_DIGEST_KEY) is None


def test_a_refused_digest_write_rolls_the_completion_back(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path, [_block(1, "译文")])
    with ledger._get_conn() as conn:
        conn.execute("UPDATE job_meta SET metadata_json = ? WHERE job_id = ?", ("not json", _JOB))
        conn.commit()

    from ubt.core.exceptions import LedgerError

    with pytest.raises(LedgerError, match="malformed JSON"):
        ledger.finalize_job(_JOB, status="completed", rendered_digest="deadbeef")

    snapshot = ledger.get_job_snapshot(_JOB)
    assert snapshot is not None
    assert snapshot["status"] != "completed"
