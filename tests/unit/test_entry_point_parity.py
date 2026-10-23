"""Every entry point must derive the same engine overrides from the same document.

The CLI resolves ``--profile auto`` to a concrete name *before* it builds its
request; the web console submits the literal string ``"auto"`` from the browser
and lets the server resolve it. ``overrides_from_request`` then read the raw
string, so ``adaptive_dual_mode`` — which recognises only concrete profile
names — saw "auto" and returned nothing for the web path while returning
``mono`` for the CLI:

    web  -> {}
    cli  -> {"dual_mode": "mono"}

The same PDF therefore rendered monolingual through one surface and bilingual
through the other, with a different output filename, and nothing said so. The
fix is that the *document* decides, not the door the request came through.

Pinned here for both the profile-aware default and the preset bundle.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ubt.core.job_options import overrides_from_request

pytestmark = pytest.mark.fast


@pytest.fixture()
def paper_pdf(tmp_path: Path) -> Path:
    """A real one-page PDF, so the archetype probe has something to read."""
    from pdf_builders import write_text_pdf

    path = tmp_path / "2401.12345v1.pdf"
    write_text_pdf(
        path,
        [("Abstract", "We propose a compact model for FinFET devices.", "References")],
    )
    return path


def test_auto_is_resolved_instead_of_matched_literally(paper_pdf: Path) -> None:
    """``auto`` must be resolved, not passed through as a profile name.

    The web console sends the literal string; the CLI resolves it first. What
    the mapping must never do is treat "auto" as a concrete name, because
    ``adaptive_dual_mode`` recognises only concrete names and would silently
    return nothing.
    """
    from ubt.core.job_options import resolve_profile_name

    base = {"input_path": str(paper_pdf), "target_lang": "zh", "source_lang": "en"}
    auto = overrides_from_request({**base, "profile": "auto"})
    resolved_name = resolve_profile_name("auto", paper_pdf)
    by_name = overrides_from_request({**base, "profile": resolved_name})
    assert auto == by_name


def test_the_web_and_cli_forms_of_a_real_paper_agree(paper_pdf: Path) -> None:
    """The concrete case: a paper submitted either way gets the same mode.

    Uses a paper PDF, which the probe classifies as ``paper``, so the
    profile-aware default is exercised rather than a synthetic ``general``.
    """
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
    for preset in ("publication", "standard", "preview"):
        auto = overrides_from_request({**base, "profile": "auto", "preset": preset})
        named = overrides_from_request({**base, "profile": resolved, "preset": preset})
        assert auto == named, preset
    # A profile with a dual_mode default proves the shared default still fires.
    assert "dual_mode" in overrides_from_request(
        {**base, "profile": "paper", "preset": "publication"}
    )


def test_an_explicit_dual_mode_still_wins(paper_pdf: Path) -> None:
    """The adaptive default is a default: an operator choice is not overridden."""
    base = {"input_path": str(paper_pdf), "target_lang": "zh", "source_lang": "en"}
    explicit = overrides_from_request({**base, "profile": "auto", "dual_mode": "dual"})
    assert explicit.get("dual_mode") == "dual"


def test_a_missing_file_does_not_crash_the_mapping(tmp_path: Path) -> None:
    """The probe tolerates a path that is not there (the adapter reports later)."""
    overrides = overrides_from_request(
        {
            "input_path": str(tmp_path / "nope.pdf"),
            "target_lang": "zh",
            "source_lang": "en",
            "profile": "auto",
        }
    )
    assert isinstance(overrides, dict)
