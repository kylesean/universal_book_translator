from __future__ import annotations

from pathlib import Path

import pytest

from ubt.core.job_options import overrides_from_request

pytestmark = pytest.mark.fast


@pytest.fixture()
def paper_pdf(tmp_path: Path) -> Path:
    from pdf_builders import write_text_pdf

    path = tmp_path / "2401.12345v1.pdf"
    write_text_pdf(
        path,
        [("Abstract", "We propose a compact model for FinFET devices.", "References")],
    )
    return path


def test_auto_is_resolved_instead_of_matched_literally(paper_pdf: Path) -> None:
    from ubt.core.job_options import resolve_profile_name

    base = {"input_path": str(paper_pdf), "target_lang": "zh", "source_lang": "en"}
    auto = overrides_from_request({**base, "profile": "auto"})
    resolved_name = resolve_profile_name("auto", paper_pdf)
    by_name = overrides_from_request({**base, "profile": resolved_name})
    assert auto == by_name


def test_the_web_and_cli_forms_of_a_real_paper_agree(paper_pdf: Path) -> None:
    base = {"input_path": str(paper_pdf), "target_lang": "zh", "source_lang": "en"}
    auto = overrides_from_request({**base, "profile": "auto"})
    named = overrides_from_request({**base, "profile": "paper"})
    default_req = overrides_from_request(base)
    assert auto == named
    assert default_req == auto
    assert auto.get("dual_mode") == "mono"


def test_the_preset_bundle_is_identical_on_both_paths(paper_pdf: Path) -> None:
    base = {"input_path": str(paper_pdf), "target_lang": "zh", "source_lang": "en"}
    from ubt.core.job_options import resolve_profile_name

    resolved = resolve_profile_name("auto", paper_pdf)
    for preset in ("high_fidelity", "standard", "preview"):
        auto = overrides_from_request({**base, "profile": "auto", "preset": preset})
        named = overrides_from_request({**base, "profile": resolved, "preset": preset})
        assert auto == named, preset
    assert "dual_mode" in overrides_from_request(
        {**base, "profile": "paper", "preset": "high_fidelity"}
    )


def test_an_explicit_dual_mode_still_wins(paper_pdf: Path) -> None:
    base = {"input_path": str(paper_pdf), "target_lang": "zh", "source_lang": "en"}
    explicit = overrides_from_request({**base, "profile": "auto", "dual_mode": "dual"})
    assert explicit.get("dual_mode") == "dual"


def test_a_missing_file_does_not_crash_the_mapping(tmp_path: Path) -> None:
    overrides = overrides_from_request(
        {
            "input_path": str(tmp_path / "nope.pdf"),
            "target_lang": "zh",
            "source_lang": "en",
            "profile": "auto",
        }
    )
    assert isinstance(overrides, dict)
