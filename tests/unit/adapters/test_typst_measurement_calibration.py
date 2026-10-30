"""G2 Measurement Calibration suite: 80 cases across body/heading/caption/table.

Specification: ``docs/design/adaptive_delivery_prd.md`` §8.1 &
``docs/guides/ADAPTIVE_DELIVERY_TECH_DESIGN.md`` §5.2.
"""

from __future__ import annotations

import pytest

from ubt.adapters.pdf.typst_compile import typst_available
from ubt.adapters.pdf.typst_measurement import TypstMeasurementService
from ubt.delivery.calibration_corpus import get_calibration_corpus

pytestmark = [
    pytest.mark.fast,
    pytest.mark.skipif(not typst_available(), reason="typst compiler is not installed"),
]


def test_all_80_calibration_cases_measure_cleanly() -> None:
    cases = get_calibration_corpus()
    assert len(cases) == 80

    # Ensure equal 20 cases per role
    role_counts = {
        role: sum(1 for c in cases if c.role == role)
        for role in ("body", "heading", "caption", "table")
    }
    assert role_counts == {"body": 20, "heading": 20, "caption": 20, "table": 20}

    svc = TypstMeasurementService()
    reqs = [c.to_request(font_version=svc.typst_version or "") for c in cases]
    results = svc.measure_many(reqs)

    assert len(results) == 80
    for c, m in zip(cases, results, strict=True):
        assert m.line_count >= c.expected_min_lines, (
            f"[{c.case_id}] lines {m.line_count} < min {c.expected_min_lines}"
        )
        assert m.line_count <= c.expected_max_lines, (
            f"[{c.case_id}] lines {m.line_count} > max {c.expected_max_lines}"
        )
        assert m.height_pt > 0.0, f"[{c.case_id}] height {m.height_pt} <= 0"
        assert len(m.line_boxes) == m.line_count, (
            f"[{c.case_id}] box count {len(m.line_boxes)} != lines {m.line_count}"
        )
        assert not m.overflow, f"[{c.case_id}] unexpected overflow on width {c.width_pt}"


def test_batch_vs_single_measurement_parity() -> None:
    cases = get_calibration_corpus()
    # Sample 2 from each role
    sample_indices = [0, 10, 20, 30, 40, 50, 60, 70]
    sample_cases = [cases[i] for i in sample_indices]

    svc = TypstMeasurementService()
    reqs = [c.to_request(font_version=svc.typst_version or "") for c in sample_cases]

    batch_results = svc.measure_many(reqs)
    for c, batch_m in zip(sample_cases, batch_results, strict=True):
        single_m = svc.measure_block(c.text, c.to_request().style, c.width_pt)
        assert abs(batch_m.height_pt - single_m.height_pt) < 0.02, (
            f"[{c.case_id}] batch height {batch_m.height_pt} != single {single_m.height_pt}"
        )
        assert batch_m.line_count == single_m.line_count, (
            f"[{c.case_id}] batch line count {batch_m.line_count} != single {single_m.line_count}"
        )
        assert len(batch_m.line_boxes) == len(single_m.line_boxes)


def test_width_reduction_monotonicity() -> None:
    cases = get_calibration_corpus()
    body_case = next(c for c in cases if c.case_id == "body_01")
    caption_case = next(c for c in cases if c.case_id == "cap_01")

    svc = TypstMeasurementService()
    for case in (body_case, caption_case):
        w_wide = case.width_pt
        w_mid = case.width_pt * 0.75
        w_narrow = case.width_pt * 0.50

        req_wide = case.to_request()
        req_mid = case.to_request()
        req_narrow = case.to_request()

        m_wide = svc.measure_block(case.text, req_wide.style, w_wide)
        m_mid = svc.measure_block(case.text, req_mid.style, w_mid)
        m_narrow = svc.measure_block(case.text, req_narrow.style, w_narrow)

        assert m_mid.line_count >= m_wide.line_count, (
            f"[{case.case_id}] lines at mid {m_mid.line_count} < wide {m_wide.line_count}"
        )
        assert m_narrow.line_count >= m_mid.line_count, (
            f"[{case.case_id}] lines at narrow {m_narrow.line_count} < mid {m_mid.line_count}"
        )
