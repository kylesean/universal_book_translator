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


# --------------------------------------------------------------------------- #
# Global term propagation (PRD §5.2.2)
# --------------------------------------------------------------------------- #

_TERM_JOB = "termjob00001"
_TERM_GLOSSARY = [{"source": "Backpropagation", "translation": "反向传播", "aliases": ["反向传递"]}]


def _seed_terms(tmp_path: Path) -> UBTConfig:
    config = UBTConfig(
        db_dir=tmp_path / "db",
        allowed_dirs=str(tmp_path),
        service_api_key=SecretStr(_API_KEY),
    )
    config.db_dir.mkdir(parents=True, exist_ok=True)
    blocks = [
        _block(
            1,
            "Backpropagation is key",
            "反向传递很关键",
            ["Glossary term violation: ..."],
            BlockStatus.NEEDS_HUMAN,
        ),
        _block(2, "Backpropagation again", "反向传递再次出现", [], BlockStatus.MTQE_PASSED),
        _block(3, "Backpropagation later", "这里也用了反向传递", [], BlockStatus.MTQE_PASSED),
        _block(4, "No term here", "无关内容", [], BlockStatus.MTQE_PASSED),
    ]
    with SQLiteJobLedger(config.db_dir / f"{_TERM_JOB}.sqlite") as ledger:
        ledger.init_job_from_manifest(
            _TERM_JOB,
            BookManifest(doc_id=_TERM_JOB, title="t", source_path=str(tmp_path / "in.md")),
        )
        ledger.set_job_metadata_value(_TERM_JOB, "source_lang", "en")
        ledger.set_job_metadata_value(
            _TERM_JOB,
            "bible_cache",
            {"glossary_dicts": _TERM_GLOSSARY, "abbreviation_entries": []},
        )
        ledger.append_chapter(
            _TERM_JOB,
            ChapterIR(doc_id=_TERM_JOB, chapter_id="c1", title="C1", spine_index=0, blocks=blocks),
        )
    return config


def test_block_terms_reports_cascade_counts(tmp_path: Path) -> None:
    client = TestClient(create_app(_seed_terms(tmp_path)))
    body = client.get(f"/jobs/{_TERM_JOB}/segments/b001/terms", headers=_AUTH).json()
    assert body["glossary_size"] == 1
    (violation,) = body["violations"]
    assert violation["surface"] == "反向传递"
    assert violation["expected"] == "反向传播"
    assert violation["kind"] == "alias"
    assert violation["occurrences"] == 1
    # b002 and b003 carry the same error; b001 itself is excluded.
    assert violation["cascade_all"] == 2
    assert violation["cascade_subsequent"] == 2


def test_block_terms_is_empty_for_an_unknown_block(tmp_path: Path) -> None:
    client = TestClient(create_app(_seed_terms(tmp_path)))
    assert client.get(f"/jobs/{_TERM_JOB}/segments/nope/terms", headers=_AUTH).status_code == 404


def test_term_propagation_scope_block_touches_only_the_pivot(tmp_path: Path) -> None:
    config = _seed_terms(tmp_path)
    client = TestClient(create_app(config))
    res = client.post(
        f"/jobs/{_TERM_JOB}/term-propagation",
        json={
            "block_id": "b001",
            "surface": "反向传递",
            "expected": "反向传播",
            "scope": "block",
        },
        headers=_AUTH,
    )
    assert res.status_code == 200
    body = res.json()
    assert body["block_ids"] == ["b001"]
    assert body["replacements"] == 1
    assert body["tm_written"] == 1

    segments = {
        s["block_id"]: s
        for s in client.get(
            f"/jobs/{_TERM_JOB}/segments", params={"status": "all"}, headers=_AUTH
        ).json()["segments"]
    }
    assert segments["b001"]["target_text"] == "反向传播很关键"
    assert segments["b001"]["human_verified"] is True
    assert segments["b002"]["target_text"] == "反向传递再次出现"  # untouched
    assert segments["b003"]["target_text"] == "这里也用了反向传递"


def test_term_propagation_scope_all_rewrites_every_match_and_feeds_tm(tmp_path: Path) -> None:
    config = _seed_terms(tmp_path)
    client = TestClient(create_app(config))
    res = client.post(
        f"/jobs/{_TERM_JOB}/term-propagation",
        json={"block_id": "b001", "surface": "反向传递", "expected": "反向传播", "scope": "all"},
        headers=_AUTH,
    )
    assert res.status_code == 200
    body = res.json()
    assert body["block_ids"] == ["b001", "b002", "b003"]
    assert body["replacements"] == 3
    assert body["tm_written"] == 3

    tm = TranslationMemory(config.db_dir / "tm.sqlite")
    try:
        entries = tm.scan()
    finally:
        tm.close()
    assert {(e.source_text, e.target_text, e.provenance) for e in entries} == {
        ("Backpropagation is key", "反向传播很关键", "human_pe"),
        ("Backpropagation again", "反向传播再次出现", "human_pe"),
        ("Backpropagation later", "这里也用了反向传播", "human_pe"),
    }


def test_term_propagation_scope_subsequent_skips_earlier_blocks(tmp_path: Path) -> None:
    client = TestClient(create_app(_seed_terms(tmp_path)))
    res = client.post(
        f"/jobs/{_TERM_JOB}/term-propagation",
        json={
            "block_id": "b002",
            "surface": "反向传递",
            "expected": "反向传播",
            "scope": "subsequent",
        },
        headers=_AUTH,
    )
    assert res.status_code == 200
    # b001 is before the pivot and untouched; b002 (pivot) and b003 are rewritten.
    assert res.json()["block_ids"] == ["b002", "b003"]


def test_term_propagation_unknown_term_is_422(tmp_path: Path) -> None:
    client = TestClient(create_app(_seed_terms(tmp_path)))
    res = client.post(
        f"/jobs/{_TERM_JOB}/term-propagation",
        json={"block_id": "b001", "surface": "不存在", "expected": "x", "scope": "all"},
        headers=_AUTH,
    )
    assert res.status_code == 422


def test_term_propagation_unknown_block_is_404(tmp_path: Path) -> None:
    client = TestClient(create_app(_seed_terms(tmp_path)))
    res = client.post(
        f"/jobs/{_TERM_JOB}/term-propagation",
        json={"block_id": "nope", "surface": "反向传递", "expected": "反向传播", "scope": "all"},
        headers=_AUTH,
    )
    assert res.status_code == 404


def test_term_propagation_without_a_glossary_is_422(tmp_path: Path) -> None:
    # The plain seed has no bible_cache, so there is nothing to propagate from.
    client = TestClient(create_app(_seed(tmp_path)))
    res = client.post(
        f"/jobs/{_JOB}/term-propagation",
        json={"block_id": "b002", "surface": "反向传递", "expected": "反向传播", "scope": "all"},
        headers=_AUTH,
    )
    assert res.status_code == 422
