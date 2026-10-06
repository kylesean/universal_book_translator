"""L3 review-workbench contract: segments, fault ribbon, and human edits.

These pin the workbench's backend contract:

- ``/segments`` projects blocks with their grouped issue kinds and filters by
  status/issues;
- ``/issues`` counts by the engine's own defect taxonomy;
- ``POST /segments/{id}`` writes a human revision through the same path the
  file-based PE import uses (ledger + shared TM, ``human_pe`` provenance).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from ubt.api.app import create_app
from ubt.api.review import segment_issue_kinds
from ubt.core.config import UBTConfig
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
from ubt.core.memory.tm import TranslationMemory

pytestmark = pytest.mark.fast

_API_KEY = "review-test-key"
_AUTH = {"X-API-Key": _API_KEY}
_JOB = "reviewjob0001"


def _block(index: int, source: str, target: str, flags: list[str], status: BlockStatus) -> IRBlock:
    element = make_element(
        id=f"b{index:03d}",
        spine_index=index,
        block_type=BlockType.NARRATIVE,
        source_text=source,
        bbox=BoundingBox(page=1, x0=50.0, y0=700.0 - index * 20, x1=350.0, y1=712.0 - index * 20),
    )
    block = IRBlock(element=element)
    block.target_text = target
    block.error_flags = flags
    block.status = status
    block.mtqe_score = 0.9
    return block


def _seed(tmp_path: Path) -> UBTConfig:
    config = UBTConfig(
        db_dir=tmp_path / "db",
        allowed_dirs=str(tmp_path),
        service_api_key=SecretStr(_API_KEY),
    )
    config.db_dir.mkdir(parents=True, exist_ok=True)
    blocks = [
        _block(1, "Hello world", "你好世界", [], BlockStatus.MTQE_PASSED),
        _block(
            2,
            "Backpropagation is key",
            "反向传递很关键",
            ["Glossary term violation: expected 反向传播"],
            BlockStatus.NEEDS_HUMAN,
        ),
        _block(
            3, "A formula here", "公式在此", ["render_skip:overflow(base=1)"], BlockStatus.REPAIRED
        ),
        _block(4, "Kept chrome", "保留页眉", ["inplace_skip:chrome"], BlockStatus.MTQE_PASSED),
    ]
    with SQLiteJobLedger(config.db_dir / f"{_JOB}.sqlite") as ledger:
        ledger.init_job_from_manifest(
            _JOB, BookManifest(doc_id=_JOB, title="t", source_path=str(tmp_path / "in.md"))
        )
        ledger.set_job_metadata_value(_JOB, "source_lang", "en")
        ledger.append_chapter(
            _JOB,
            ChapterIR(doc_id=_JOB, chapter_id="c1", title="C1", spine_index=0, blocks=blocks),
        )
    return config


def test_issue_classifier_ignores_intentional_skips() -> None:
    intentional = _block(9, "s", "t", ["inplace_skip:chrome"], BlockStatus.MTQE_PASSED)
    assert segment_issue_kinds(intentional) == []
    overflow = _block(9, "s", "t", ["render_skip:overflow(base=1)"], BlockStatus.REPAIRED)
    assert segment_issue_kinds(overflow) == ["render"]
    term = _block(9, "s", "t", ["Glossary term violation: x"], BlockStatus.NEEDS_HUMAN)
    assert segment_issue_kinds(term) == ["terminology"]


def test_issues_endpoint_counts_by_kind(tmp_path: Path) -> None:
    client = TestClient(create_app(_seed(tmp_path)))
    body = client.get(f"/jobs/{_JOB}/issues", headers=_AUTH).json()
    assert body["counts"]["terminology"] == 1
    assert body["counts"]["render"] == 1
    assert body["counts"]["formula"] == 0
    assert body["status"]["needs_human"] == 1
    assert body["total_issues"] == 2


def test_segments_filter_issues_excludes_clean_and_intentional(tmp_path: Path) -> None:
    client = TestClient(create_app(_seed(tmp_path)))
    body = client.get(f"/jobs/{_JOB}/segments", params={"status": "issues"}, headers=_AUTH).json()
    ids = {segment["block_id"] for segment in body["segments"]}
    # b002 (term violation + needs_human) and b003 (overflow) are issues;
    # b001 is clean and b004 is an intentional chrome skip.
    assert ids == {"b002", "b003"}
    assert body["total"] == 2


def test_segments_filter_status_all_returns_every_block(tmp_path: Path) -> None:
    client = TestClient(create_app(_seed(tmp_path)))
    body = client.get(f"/jobs/{_JOB}/segments", params={"status": "all"}, headers=_AUTH).json()
    assert body["total"] == 4


def test_edit_segment_writes_ledger_and_tm(tmp_path: Path) -> None:
    config = _seed(tmp_path)
    client = TestClient(create_app(config))
    res = client.post(
        f"/jobs/{_JOB}/segments/b002",
        json={"target_text": "反向传播很关键"},
        headers=_AUTH,
    )
    assert res.status_code == 200
    body = res.json()
    assert body["changed"] is True
    assert body["tm_written"] == 1
    assert body["segment"]["status"] == "repaired"
    assert body["segment"]["human_verified"] is True
    assert body["segment"]["mtqe_score"] is None  # stale machine verdict cleared

    tm = TranslationMemory(config.db_dir / "tm.sqlite")
    try:
        entries = tm.scan()
    finally:
        tm.close()
    assert [(e.source_text, e.target_text, e.provenance) for e in entries] == [
        ("Backpropagation is key", "反向传播很关键", "human_pe")
    ]


def test_edit_segment_is_a_noop_when_unchanged(tmp_path: Path) -> None:
    config = _seed(tmp_path)
    client = TestClient(create_app(config))
    res = client.post(
        f"/jobs/{_JOB}/segments/b001", json={"target_text": "你好世界"}, headers=_AUTH
    )
    assert res.status_code == 200
    assert res.json()["changed"] is False
    assert res.json()["tm_written"] == 0


def test_edit_segment_rejects_unknown_block(tmp_path: Path) -> None:
    client = TestClient(create_app(_seed(tmp_path)))
    res = client.post(f"/jobs/{_JOB}/segments/nope", json={"target_text": "x"}, headers=_AUTH)
    assert res.status_code == 404


def test_edit_segment_rejects_empty_target(tmp_path: Path) -> None:
    client = TestClient(create_app(_seed(tmp_path)))
    res = client.post(f"/jobs/{_JOB}/segments/b001", json={"target_text": ""}, headers=_AUTH)
    assert res.status_code == 422


def test_page_preview_unknown_job_is_404(tmp_path: Path) -> None:
    config = UBTConfig(
        db_dir=tmp_path / "db",
        allowed_dirs=str(tmp_path),
        service_api_key=SecretStr(_API_KEY),
    )
    client = TestClient(create_app(config))
    assert client.get("/jobs/nosuchjob00/pages/1/preview", headers=_AUTH).status_code == 404


def test_page_preview_rejects_page_zero(tmp_path: Path) -> None:
    client = TestClient(create_app(_seed(tmp_path)))
    assert client.get(f"/jobs/{_JOB}/pages/0/preview", headers=_AUTH).status_code == 422


def test_segments_unknown_job_is_404(tmp_path: Path) -> None:
    config = UBTConfig(
        db_dir=tmp_path / "db",
        allowed_dirs=str(tmp_path),
        service_api_key=SecretStr(_API_KEY),
    )
    client = TestClient(create_app(config))
    assert client.get("/jobs/nosuchjob00/segments", headers=_AUTH).status_code == 404
