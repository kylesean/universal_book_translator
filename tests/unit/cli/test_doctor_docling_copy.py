"""``ubt doctor`` tells the truth about a missing docling.

The check used to say PDF jobs "will fail" for both engine settings. Neither
was right: ``auto`` degrades to the geometric pdfium reader (tables, figures
and formulas silently dropped from the rebuilt PDF), while an explicit
``--pdf-engine docling`` now fails with an actionable error instead of
degrading. An operator reading the old text went looking for a crash that
never came.
"""

from __future__ import annotations

import importlib.util

import pytest

from ubt.cli.commands.doctor import _Check

pytestmark = pytest.mark.fast

_DOCLING_INSTALLED = importlib.util.find_spec("docling") is not None


def _check_for(engine: str) -> _Check:
    from ubt.cli.commands.doctor import collect_checks
    from ubt.core.config import UBTConfig

    checks = collect_checks(UBTConfig.from_env(pdf_engine=engine))
    matches = [c for c in checks if c.code == "pdf_engine"]
    assert matches, f"no pdf_engine check was emitted for {engine!r}"
    return matches[0]


@pytest.mark.skipif(_DOCLING_INSTALLED, reason="this machine has docling installed")
def test_auto_without_docling_warns_about_what_is_lost() -> None:
    check = _check_for("auto")
    assert check.status == "WARN", "auto degrades; it does not fail"
    assert "fall" in check.detail.lower()
    assert "tables" in check.detail.lower()
    assert check.detail_code == "pdf_engine.auto_missing"


@pytest.mark.skipif(_DOCLING_INSTALLED, reason="this machine has docling installed")
def test_explicit_docling_without_the_package_is_not_described_as_a_fallback() -> None:
    check = _check_for("docling")
    assert check.status == "WARN"
    assert check.detail_code == "pdf_engine.docling_missing"
    detail = check.detail.lower()
    assert "fail" in detail, "an explicit request raises rather than degrading"
    assert "fall back" not in detail and "falls back" not in detail


def test_the_two_engine_checks_do_not_share_copy() -> None:
    """The old text was one sentence used for both paths, and true for neither."""
    from ubt.cli.commands.doctor import collect_checks
    from ubt.core.config import UBTConfig

    auto = [
        c for c in collect_checks(UBTConfig.from_env(pdf_engine="auto")) if c.code == "pdf_engine"
    ]
    docling = [
        c
        for c in collect_checks(UBTConfig.from_env(pdf_engine="docling"))
        if c.code == "pdf_engine"
    ]
    assert auto and docling
    assert auto[0].detail != docling[0].detail
