"""The artifact routes: report / visual-report / deliverables / download.

These four endpoints were 25% covered — the console's Report, Quality and
Download surfaces all read them, so a wrong status code or a leaked host path
was invisible until an operator clicked. The tests pin the three-state
responses each route distinguishes: 404 (nothing known about the job), 409
(the job exists but its artifact is not ready), 200 (the artifact).

NOTE for maintainers: running THIS file alone with a *single-module* coverage
flag (``--cov=ubt.api.routes.artifact``) makes Starlette's router treat every
request as unmatched and answer 307 to a trailing-slash URL, so the tests
fail with TooManyRedirects. The gate's own invocation
(``--cov=ubt``, as pre-commit and CI run it) is unaffected, and so is
production; the module-scoped flag is the only shape that triggers it. If you
need per-module coverage here, run the whole ``tests/unit/api`` directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from ubt.api.app import create_app
from ubt.core.config import UBTConfig
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.ir.models import (
    BlockStatus,
    BlockType,
    BookManifest,
    ChapterIR,
    IRBlock,
    make_element,
)

pytestmark = pytest.mark.fast

_API_KEY = "artifact-test-key"
_AUTH = {"X-API-Key": _API_KEY}
_JOB = "artifactjob001"


def _config(tmp_path: Path) -> UBTConfig:
    config = UBTConfig(
        db_dir=tmp_path / "db",
        allowed_dirs=str(tmp_path),
        service_api_key=SecretStr(_API_KEY),
    )
    config.db_dir.mkdir(parents=True, exist_ok=True)
    return config


def _seed_job(config: UBTConfig, tmp_path: Path) -> None:
    """A completed job with one block and its artifact paths on disk."""
    out = tmp_path / "out.pdf"
    out.write_bytes(b"%PDF-1.4\nfake")
    report = tmp_path / "out_quality_report.json"
    report.write_text(
        json.dumps(
            {
                "summary": {
                    "completed_blocks": 10,
                    "failed_blocks": 0,
                    "blocked_human_blocks": 2,
                    "needs_human_blocks": 0,
                    "pass_rate": 0.83,
                },
                "score_metrics": {"avg_qe": 0.9, "qe_measured": True},
                "source_path": str(tmp_path / "secret-host-path.pdf"),
            }
        ),
        encoding="utf-8",
    )
    visual = tmp_path / "out_visual_report.json"
    visual.write_text(json.dumps({"findings": [], "total_pages": 3}), encoding="utf-8")

    with SQLiteJobLedger(config.db_dir / f"{_JOB}.sqlite") as ledger:
        ledger.init_job_from_manifest(
            _JOB,
            BookManifest(doc_id=_JOB, title="t", source_path=str(tmp_path / "in.md")),
        )
        block = IRBlock(
            element=make_element(
                id="b1",
                spine_index=1,
                block_type=BlockType.NARRATIVE,
                source_text="Hello",
            )
        )
        block.status = BlockStatus.MTQE_PASSED
        block.target_text = "你好"
        ledger.append_chapter(
            _JOB,
            ChapterIR(doc_id=_JOB, chapter_id="c1", title="C1", spine_index=0, blocks=[block]),
        )
        ledger.set_job_metadata_value(_JOB, "output_file", str(out))
        ledger.set_job_metadata_value(_JOB, "report_file", str(report))
        ledger.set_job_metadata_value(_JOB, "visual_report_file", str(visual))
        ledger.finalize_job(_JOB, status="completed")


def test_the_report_carries_the_shared_delivery_verdict(tmp_path: Path) -> None:
    """The web gate reads this field; it must be the same rule the CLI applies.

    The summary says 2 quarantined blocks (which ship source), so the verdict
    is BLOCKED even with zero failed blocks — the exact case the console used
    to announce as PRODUCTION READY.
    """
    config = _config(tmp_path)
    _seed_job(config, tmp_path)
    client = TestClient(create_app(config))

    res = client.get(f"/jobs/{_JOB}/report", headers=_AUTH)
    assert res.status_code == 200
    body = res.json()
    assert body["delivery_verdict"] == "blocked"
    assert body["score_metrics"]["qe_measured"] is True


def test_the_report_does_not_leak_host_paths(tmp_path: Path) -> None:
    """``public_report`` reduces absolute paths to basenames."""
    config = _config(tmp_path)
    _seed_job(config, tmp_path)
    client = TestClient(create_app(config))

    body = client.get(f"/jobs/{_JOB}/report", headers=_AUTH).json()
    assert str(tmp_path) not in json.dumps(body)
    assert body["source_path"] == "secret-host-path.pdf"


def test_a_known_job_without_a_report_is_409_not_404(tmp_path: Path) -> None:
    """The distinction the console's polling depends on: keep waiting vs stop."""
    config = _config(tmp_path)
    # A ledger exists (the job is known) but no report sidecar does.
    with SQLiteJobLedger(config.db_dir / f"{_JOB}.sqlite") as ledger:
        ledger.init_job_from_manifest(
            _JOB, BookManifest(doc_id=_JOB, title="t", source_path=str(tmp_path / "in.md"))
        )
    client = TestClient(create_app(config))

    res = client.get(f"/jobs/{_JOB}/report", headers=_AUTH)
    assert res.status_code == 409


def test_an_unknown_job_is_404(tmp_path: Path) -> None:
    client = TestClient(create_app(_config(tmp_path)))
    assert client.get("/jobs/unknownjob01/report", headers=_AUTH).status_code == 404
    assert client.get("/jobs/unknownjob01/visual-report", headers=_AUTH).status_code == 404
    assert client.get("/jobs/unknownjob01/deliverables", headers=_AUTH).status_code == 404


def test_a_corrupt_report_is_a_readable_400_not_a_crash(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _seed_job(config, tmp_path)
    (tmp_path / "out_quality_report.json").write_text("{not json", encoding="utf-8")
    client = TestClient(create_app(config))

    res = client.get(f"/jobs/{_JOB}/report", headers=_AUTH)
    assert res.status_code == 400


def test_the_visual_report_is_served_verbatim(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _seed_job(config, tmp_path)
    client = TestClient(create_app(config))

    res = client.get(f"/jobs/{_JOB}/visual-report", headers=_AUTH)
    assert res.status_code == 200
    assert res.json()["total_pages"] == 3


def test_deliverables_list_what_exists_on_disk(tmp_path: Path) -> None:
    """Only files that are really there; a missing companion is omitted."""
    config = _config(tmp_path)
    _seed_job(config, tmp_path)
    client = TestClient(create_app(config))

    res = client.get(f"/jobs/{_JOB}/deliverables", headers=_AUTH)
    assert res.status_code == 200
    body = res.json()
    assert body["job_id"] == _JOB
    keys = {item["key"] for item in body["deliverables"]}
    assert "primary" in keys
    # The EPUB was never written, so it must not be offered.
    assert "epub" not in keys
    for item in body["deliverables"]:
        assert item["size_bytes"] > 0
        assert "/" not in item["filename"], "only the basename is exposed"


def test_download_serves_the_primary_artifact(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _seed_job(config, tmp_path)
    client = TestClient(create_app(config))

    res = client.get(f"/jobs/{_JOB}/download/primary", headers=_AUTH)
    assert res.status_code == 200
    assert res.content.startswith(b"%PDF")


def test_download_of_an_unknown_key_is_404(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _seed_job(config, tmp_path)
    client = TestClient(create_app(config))

    assert client.get(f"/jobs/{_JOB}/download/nonsense", headers=_AUTH).status_code == 404


def test_a_bad_job_id_is_refused_before_any_lookup(tmp_path: Path) -> None:
    """The id is a SQLite file name; the validator is the path boundary."""
    client = TestClient(create_app(_config(tmp_path)))
    res = client.get("/jobs/..%2Fetc/report", headers=_AUTH)
    assert res.status_code in (400, 404, 422)
