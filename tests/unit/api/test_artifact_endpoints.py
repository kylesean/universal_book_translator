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
from ubt.core.job_options import sidecar_path

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


def _seed_job(config: UBTConfig, tmp_path: Path, *, blocked_human_blocks: int = 2) -> None:
    """A completed job with one block and its artifact paths on disk.

    ``blocked_human_blocks`` defaults to 2, so the seeded report's verdict is
    ``blocked`` — the state most of these tests care about, because it is the
    one the console used to misreport. Tests that are about *readiness* or
    *staleness* rather than the gate pass 0, so the download guard stays out of
    their way.
    """
    out = tmp_path / "out.pdf"
    out.write_bytes(b"%PDF-1.4\nfake")
    report_data = json.dumps(
        {
            "summary": {
                "completed_blocks": 10,
                "failed_blocks": 0,
                "blocked_human_blocks": blocked_human_blocks,
                "needs_human_blocks": 0,
                "pass_rate": 0.83,
            },
            "score_metrics": {"avg_qe": 0.9, "qe_measured": True},
            "source_path": str(tmp_path / "secret-host-path.pdf"),
        }
    )
    report = tmp_path / "out_quality_report.json"
    report.write_text(report_data, encoding="utf-8")
    sidecar_path(out, "quality_report.json").write_text(report_data, encoding="utf-8")
    visual = tmp_path / "out_visual_report.json"
    visual_data = json.dumps({"findings": [], "total_pages": 3})
    visual.write_text(visual_data, encoding="utf-8")
    sidecar_path(out, "visual_report.json").write_text(visual_data, encoding="utf-8")

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
    _seed_job(config, tmp_path, blocked_human_blocks=0)
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


def test_in_flight_download_and_report_return_409(tmp_path: Path) -> None:
    """Verify in-flight polling of /download and /report returns HTTP 409 Conflict instead of 400."""
    config = UBTConfig(
        provider="openai",
        api_key=SecretStr("test-provider-key"),
        service_api_key=SecretStr(_API_KEY),
        db_dir=tmp_path / "ledgers",
        allowed_dirs=str(tmp_path),
    )
    app = create_app(config=config)
    client = TestClient(app)

    # Submit a job
    input_file = tmp_path / "test.md"
    input_file.write_text("# Chapter 1\nHello", encoding="utf-8")

    res = client.post(
        "/jobs/submit",
        json={"input_path": str(input_file), "job_id": "test-inflight-409"},
        headers=_AUTH,
    )
    assert res.status_code == 202

    # Polling download before file is ready must return 409 Conflict
    dl_res = client.get("/jobs/test-inflight-409/download", headers=_AUTH)
    assert dl_res.status_code == 409
    assert "not ready" in dl_res.json()["detail"]

    # Polling report before report is generated must return 409 Conflict
    rep_res = client.get("/jobs/test-inflight-409/report", headers=_AUTH)
    assert rep_res.status_code == 409
    assert "not yet generated" in rep_res.json()["detail"]


def test_download_refuses_a_delivery_that_predates_the_ledger(tmp_path: Path) -> None:
    """The worst failure this endpoint has: shipping the pre-edit file.

    A human edit lands in the ledger; the PDF on disk was written by an earlier
    export. Serving it silently means the customer gets the draft the reviewer
    just corrected, and nothing anywhere reports a problem. Fail closed instead.
    """
    from ubt.core.engine.delivery_freshness import record_rendered_digest

    config = _config(tmp_path)
    _seed_job(config, tmp_path, blocked_human_blocks=0)
    client = TestClient(create_app(config))

    # A render happened, so the artifact is current.
    with SQLiteJobLedger(config.db_dir / f"{_JOB}.sqlite") as ledger:
        record_rendered_digest(ledger, _JOB)
    fresh = client.get(f"/jobs/{_JOB}/download/primary", headers=_AUTH)
    assert fresh.status_code == 200

    # Now a reviewer edits a segment and never re-renders.
    edit = client.post(
        f"/jobs/{_JOB}/segments/b1", json={"target_text": "人工校订的译文"}, headers=_AUTH
    )
    assert edit.status_code == 200, edit.text

    stale = client.get(f"/jobs/{_JOB}/download/primary", headers=_AUTH)
    assert stale.status_code == 409, stale.text
    assert "re-rendered" in stale.json()["detail"]

    # The listing says so up front, so the console can offer "re-render first"
    # instead of opening a download that would only fail.
    listing = client.get(f"/jobs/{_JOB}/deliverables", headers=_AUTH)
    assert listing.status_code == 200
    assert listing.json()["stale"] is True


def test_a_job_that_never_recorded_a_render_is_still_downloadable(tmp_path: Path) -> None:
    """Jobs finished before this marker existed must not be blocked.

    They have no baseline to compare against, and refusing them would break
    every pre-existing deliverable — the marker only earns its keep once a
    render has recorded one.
    """
    config = _config(tmp_path)
    _seed_job(config, tmp_path, blocked_human_blocks=0)
    client = TestClient(create_app(config))
    res = client.get(f"/jobs/{_JOB}/download/primary", headers=_AUTH)
    assert res.status_code == 200
    assert client.get(f"/jobs/{_JOB}/deliverables", headers=_AUTH).json()["stale"] is False


def test_a_blocked_delivery_is_not_served_by_default(tmp_path: Path) -> None:
    """The verdict is no longer advisory at the one moment it matters.

    ``blocked`` means quarantined or failed blocks ship the *source* in place of
    a translation, so the reader meets English in a Chinese book. Every other
    surface said so (CLI banner, gate screen, Mission Control) while this
    endpoint handed the file over regardless.
    """
    config = _config(tmp_path)
    _seed_job(config, tmp_path, blocked_human_blocks=2)
    client = TestClient(create_app(config))

    res = client.get(f"/jobs/{_JOB}/download/primary", headers=_AUTH)
    assert res.status_code == 409, res.text
    assert "blocked" in res.json()["detail"]
    assert "force=1" in res.json()["detail"]


def test_force_takes_a_blocked_delivery_knowingly(tmp_path: Path) -> None:
    """The escape hatch mirrors ``strict_contract``: safe by default, loosen on ask."""
    config = _config(tmp_path)
    _seed_job(config, tmp_path, blocked_human_blocks=2)
    client = TestClient(create_app(config))

    res = client.get(f"/jobs/{_JOB}/download/primary?force=1", headers=_AUTH)
    assert res.status_code == 200
    assert res.content.startswith(b"%PDF")


def test_the_audit_sidecars_stay_downloadable_from_a_blocked_job(tmp_path: Path) -> None:
    """Refusing these would make the refusal unexplainable.

    The quality report is the only place an operator can read *why* the gate
    blocked the delivery; the contract is the other half of that audit. Neither
    is reader-facing, so neither is gated.
    """
    config = _config(tmp_path)
    _seed_job(config, tmp_path, blocked_human_blocks=2)
    client = TestClient(create_app(config))

    # The deliverable path for ``quality_report`` is derived from the primary
    # artifact's stem (``out_pdf_quality_report.json``), not from the ledger's
    # ``report_file`` key, so write one there for this test.
    sidecar = tmp_path / "out_pdf_quality_report.json"
    sidecar.write_text(json.dumps({"summary": {"blocked_human_blocks": 2}}), encoding="utf-8")

    report = client.get(f"/jobs/{_JOB}/download/quality_report", headers=_AUTH)
    assert report.status_code == 200, report.text
    assert report.json()["summary"]["blocked_human_blocks"] == 2


def test_the_console_preview_of_a_blocked_delivery_still_renders(tmp_path: Path) -> None:
    """``?inline=1`` is the preview pane, not a delivery.

    The operator needs to look at the artifact to decide whether to fix it or
    take it with ``?force=1``; refusing the preview would leave them deciding
    blind.
    """
    config = _config(tmp_path)
    _seed_job(config, tmp_path, blocked_human_blocks=2)
    client = TestClient(create_app(config))

    res = client.get(f"/jobs/{_JOB}/download/primary?inline=1", headers=_AUTH)
    assert res.status_code == 200, res.text
    assert "attachment" not in res.headers.get("content-disposition", "")


def test_a_warning_delivery_downloads_without_force(tmp_path: Path) -> None:
    """``warning`` is deliverable-by-design: blocks await human review, they are
    not quarantined, so nothing ships source in place of a translation."""
    config = _config(tmp_path)
    _seed_job(config, tmp_path, blocked_human_blocks=0)
    report = tmp_path / "out_quality_report.json"
    data = json.loads(report.read_text(encoding="utf-8"))
    data["summary"]["needs_human_blocks"] = 3
    report.write_text(json.dumps(data), encoding="utf-8")
    client = TestClient(create_app(config))

    listing = client.get(f"/jobs/{_JOB}/deliverables", headers=_AUTH)
    assert listing.json()["delivery_verdict"] == "warning"
    assert client.get(f"/jobs/{_JOB}/download/primary", headers=_AUTH).status_code == 200


def test_the_listing_reports_the_verdict_before_the_click(tmp_path: Path) -> None:
    """So the console can warn instead of opening a download that would 409."""
    config = _config(tmp_path)
    _seed_job(config, tmp_path, blocked_human_blocks=2)
    client = TestClient(create_app(config))

    body = client.get(f"/jobs/{_JOB}/deliverables", headers=_AUTH).json()
    assert body["delivery_verdict"] == "blocked"


def test_a_job_with_no_report_is_not_treated_as_blocked(tmp_path: Path) -> None:
    """``None`` means "nothing to judge", not "refused".

    An in-flight job polls this endpoint; a missing report must not turn every
    poll into a 409.
    """
    config = _config(tmp_path)
    _seed_job(config, tmp_path, blocked_human_blocks=0)
    (tmp_path / "out_quality_report.json").unlink()
    client = TestClient(create_app(config))

    body = client.get(f"/jobs/{_JOB}/deliverables", headers=_AUTH).json()
    assert body["delivery_verdict"] is None
    assert client.get(f"/jobs/{_JOB}/download/primary", headers=_AUTH).status_code == 200
