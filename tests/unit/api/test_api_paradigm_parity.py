"""Contract and parity tests for dynamic routing, delivery plan assess, re-export override, and page geometry APIs."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError

from ubt.api.app import create_app
from ubt.app.spec import JobSpec
from ubt.core.config import UBTConfig
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.exceptions import UBTError
from ubt.core.ir.models import BookManifest
from ubt.core.job_options import overrides_from_request, validate_request_enums

pytestmark = pytest.mark.fast

_API_KEY = "parity-test-key"
_AUTH = {"X-API-Key": _API_KEY}


def _seed_ledger(tmp_path: Path, job_id: str = "parityjob0001") -> UBTConfig:
    config = UBTConfig(
        db_dir=tmp_path / "db",
        allowed_dirs=str(tmp_path),
        service_api_key=SecretStr(_API_KEY),
    )
    config.db_dir.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "book.md"
    source.write_text("# Chapter 1\n\nSome content for testing.\n", encoding="utf-8")
    out = tmp_path / "book_zh_mono.md"
    out.write_text("# 第一章\n\n测试内容。\n", encoding="utf-8")
    manifest = BookManifest(doc_id=job_id, title="Book", source_path=str(source))
    with SQLiteJobLedger(config.db_dir / f"{job_id}.sqlite") as ledger:
        ledger.init_job_from_manifest(job_id, manifest)
        ledger.set_job_metadata_value(job_id, "output_file", str(out))
    return config


def test_jobspec_exposes_routing_fallback_and_spill_knobs() -> None:
    spec = JobSpec(
        input_path="book.pdf",
        enable_dynamic_routing=True,
        dynamic_routing_policy="quality_first",
        ocr_fallback="weak",
        spill_policy="appendix",
    )
    overrides = overrides_from_request(spec.model_dump())
    assert overrides["enable_dynamic_routing"] is True
    assert overrides["dynamic_routing_policy"] == "quality_first"
    assert overrides["ocr_fallback"] == "weak"
    assert overrides["spill_policy"] == "appendix"

    with pytest.raises(ValidationError):
        JobSpec(input_path="book.pdf", ocr_fallback="bogus")  # type: ignore[arg-type]

    with pytest.raises(UBTError):
        validate_request_enums({"dynamic_routing_policy": "invalid_policy"})


def test_assess_returns_structured_delivery_plan_and_accepts_delivery_knobs(
    tmp_path: Path,
) -> None:
    doc = tmp_path / "sample.md"
    doc.write_text("# Title\n\n" + ("Hello world. " * 120), encoding="utf-8")
    client = TestClient(create_app(_seed_ledger(tmp_path)))

    res = client.post(
        "/jobs/assess",
        json={
            "input_path": str(doc),
            "delivery_mode": "readable",
            "document_mode": "book",
            "allow_degrade": True,
            "pdf_engine": "auto",
        },
        headers=_AUTH,
    )
    assert res.status_code == 200, res.text
    plan = res.json()["meta"].get("delivery_plan")
    assert isinstance(plan, dict)
    assert plan["contract"] == "readable"
    assert plan["feasible"] is True
    assert plan["effective_mode"] == "readable"


def test_re_export_forwards_optional_delivery_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ubt.api.routes.review as review_routes

    seen: list[str | None] = []

    async def _fake_re_export_job(**kwargs: Any) -> Any:
        seen.append(kwargs.get("delivery"))
        return SimpleNamespace(
            artifact_path=str(tmp_path / "book_zh_mono.md"),
            message="ok",
        )

    monkeypatch.setattr(review_routes, "re_export_job", _fake_re_export_job)
    client = TestClient(create_app(_seed_ledger(tmp_path)))

    r_default = client.post("/jobs/parityjob0001/re-export", headers=_AUTH)
    assert r_default.status_code == 200, r_default.text
    r_readable = client.post(
        "/jobs/parityjob0001/re-export",
        json={"delivery": "readable"},
        headers=_AUTH,
    )
    assert r_readable.status_code == 200, r_readable.text
    r_bad = client.post(
        "/jobs/parityjob0001/re-export",
        json={"delivery": "bogus"},
        headers=_AUTH,
    )
    assert r_bad.status_code == 422
    assert seen == [None, "readable"]


def test_page_geometry_endpoint_reports_measured_and_unmeasured_ledgers(tmp_path: Path) -> None:
    cfg = _seed_ledger(tmp_path, job_id="geomempty0001")
    cfg = _seed_ledger(tmp_path, job_id="geommeas0001")
    with SQLiteJobLedger(cfg.db_dir / "geommeas0001.sqlite") as ledger:
        ledger.record_page_geometry(
            "geommeas0001",
            [
                {
                    "page_no": 1,
                    "page_w_pt": 595.0,
                    "page_h_pt": 842.0,
                    "measured_size_pt": 10.0,
                    "measured_leading_pt": 12.0,
                    "delivered_size_pt": 10.0,
                    "delivered_leading_em": 0.3,
                    "tracking_em": 0.0,
                    "tier": "L0",
                    "fits": True,
                },
                {
                    "page_no": 2,
                    "page_w_pt": 595.0,
                    "page_h_pt": 842.0,
                    "measured_size_pt": 10.0,
                    "measured_leading_pt": 12.0,
                    "delivered_size_pt": 7.5,
                    "delivered_leading_em": 0.25,
                    "tracking_em": -0.02,
                    "tier": "L2",
                    "fits": False,
                },
            ],
        )

    client = TestClient(create_app(cfg))
    assert client.get("/jobs/geommeas0001/page-geometry").status_code == 401
    assert client.get("/jobs/nosuchjob0001/page-geometry", headers=_AUTH).status_code == 404

    empty = client.get("/jobs/geomempty0001/page-geometry", headers=_AUTH)
    assert empty.status_code == 200, empty.text
    assert empty.json()["measured"] is False
    assert empty.json()["pages"] == []

    measured = client.get("/jobs/geommeas0001/page-geometry", headers=_AUTH)
    assert measured.status_code == 200, measured.text
    body = measured.json()
    assert body["measured"] is True
    assert body["total_pages"] == 2
    assert body["scaled_pages"] == 1
    assert body["overflow_pages"] == 1
    assert body["min_delivered_size_pt"] == 7.5
    assert [p["page_no"] for p in body["pages"]] == [1, 2]
