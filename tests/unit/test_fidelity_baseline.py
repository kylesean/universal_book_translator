"""Unit tests for the fidelity baseline harness pure helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "fidelity_baseline.py"
_spec = importlib.util.spec_from_file_location("fidelity_baseline", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
fb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fb)


def test_pair_parallel_dirs(tmp_path: Path) -> None:
    src = tmp_path / "src"
    art = tmp_path / "out"
    src.mkdir()
    art.mkdir()
    (src / "a.pdf").write_bytes(b"%PDF")
    (src / "b.pdf").write_bytes(b"%PDF")
    (art / "a.pdf").write_bytes(b"%PDF")  # only a has an artifact
    pairs = fb.pair_sources_artifacts(src, art)
    assert [s.name for s, _ in pairs] == ["a.pdf"]


def test_pair_single_dir_translated_suffix(tmp_path: Path) -> None:
    (tmp_path / "a.pdf").write_bytes(b"%PDF")
    (tmp_path / "a.translated.pdf").write_bytes(b"%PDF")
    (tmp_path / "orphan.pdf").write_bytes(b"%PDF")  # no artifact → skipped
    pairs = fb.pair_sources_artifacts(tmp_path, None)
    assert [(s.name, a.name) for s, a in pairs] == [("a.pdf", "a.translated.pdf")]


def test_format_summary_aggregates_measured_only() -> None:
    results = [
        {"pages_measured": 3, "non_text_diff_ratio": 0.01, "masked_coverage_ratio": 0.5},
        {"pages_measured": 0, "non_text_diff_ratio": 0.0, "masked_coverage_ratio": 0.0},
        {"pages_measured": 3, "non_text_diff_ratio": 0.03, "masked_coverage_ratio": 0.7},
    ]
    s = fb.format_summary(results)
    assert s["documents"] == 3
    assert s["documents_measured"] == 2
    assert abs(float(s["non_text_residual_mean"]) - 0.02) < 1e-9
    assert s["non_text_residual_max"] == 0.03
    assert s["painted_coverage_min"] == 0.5


def test_format_summary_empty_is_safe() -> None:
    s = fb.format_summary([])
    assert s["documents_measured"] == 0
    assert s["non_text_residual_mean"] is None


def test_main_fails_when_no_page_is_measurable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A document that measured nothing must not print a perfect 0.0 and exit 0.

    compute_render_fidelity selects pages from the block list; with none it
    returns pages_measured=0 and residual 0.0 for a measurement that never ran.
    """
    src = tmp_path / "src"
    art = tmp_path / "out"
    src.mkdir()
    art.mkdir()
    (src / "a.pdf").write_bytes(b"%PDF")
    (art / "a.pdf").write_bytes(b"%PDF")

    monkeypatch.setattr(
        "ubt.adapters.pdf.render_fidelity.compute_render_fidelity",
        lambda *_a, **_k: {"pages_measured": 0, "skipped_reason": "no_measurable_pages"},
    )
    rc = fb.main(["--source-dir", str(src), "--artifact-dir", str(art)])
    assert rc == 1
