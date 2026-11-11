"""Closure regressions for the remaining deep-review findings.

Covers: Latin-script parity (no script range to measure), recorded coordinate
clamping, the contract evidence no longer exempting short/small text, and the
document-measured ink/em ratio.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ubt.adapters.pdf.artifact_parity import check_artifact_parity, latin_stopword_share
from ubt.adapters.pdf.source_metrics import (
    INK_TO_EM_RATIO,
    measure_ink_to_em_ratio,
    source_body_size,
)

pytestmark = pytest.mark.fast


# -- Latin-script parity ---------------------------------------------------


def test_function_words_identify_the_delivered_language() -> None:
    english = "The trial was run and the result was that the model is of the kind."
    french = "Le procès a été lancé et le résultat est que le modèle est de la sorte."
    en_share, en_tokens = latin_stopword_share(english, "en")
    fr_share, _ = latin_stopword_share(french, "fr")
    assert en_share > 0.4 and fr_share > 0.4
    # Cross-language: English text scores near zero against French function words.
    assert latin_stopword_share(english, "fr")[0] < 0.05


def test_an_untranslated_latin_target_is_critical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The old code emitted only an info note for Latin targets, so an
    untranslated en->fr artifact passed fail-closed review."""
    from ubt.adapters.pdf import artifact_parity as ap

    english = (
        "The model was trained on the corpus and the results are reported in the "
        "table with the baseline for the comparison of the methods. "
    ) * 20
    monkeypatch.setattr(ap, "_page_text", lambda *_a, **_k: english)

    findings = check_artifact_parity(
        artifact_pdf=tmp_path / "out.pdf", target_lang="fr", source_lang="en"
    )
    assert any(f.code == "target_language_absent" and f.severity == "critical" for f in findings)


def test_a_translated_latin_target_is_not_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ubt.adapters.pdf import artifact_parity as ap

    french = (
        "Le modèle a été entraîné sur le corpus et les résultats sont rapportés dans "
        "le tableau avec la référence pour la comparaison des méthodes. "
    ) * 20
    monkeypatch.setattr(ap, "_page_text", lambda *_a, **_k: french)

    findings = check_artifact_parity(
        artifact_pdf=tmp_path / "out.pdf", target_lang="fr", source_lang="en"
    )
    assert not any(f.code == "target_language_absent" for f in findings)


# -- coordinate clamping ---------------------------------------------------


def test_an_off_page_box_is_clamped_and_the_clamp_is_recorded() -> None:
    from ubt.adapters.pdf.coordinate_resolver import PageBBoxResolver

    resolver = PageBBoxResolver(page_width=600.0, page_height=800.0)
    notes: list[str] = []
    box = resolver.resolve_bbox(
        [-50.0, 10.0, 700.0, 790.0],
        origin="bottom-left",
        coord_system="points",
        notes=notes,
    )
    assert box is not None
    assert box[0] == 0.0 and box[2] == 600.0
    assert notes and "clamped" in notes[0]


def test_an_inside_box_produces_no_note() -> None:
    from ubt.adapters.pdf.coordinate_resolver import PageBBoxResolver

    resolver = PageBBoxResolver(page_width=600.0, page_height=800.0)
    notes: list[str] = []
    assert (
        resolver.resolve_bbox([10.0, 10.0, 590.0, 790.0], coord_system="points", notes=notes)
        is not None
    )
    assert notes == []


# -- ink/em ratio ----------------------------------------------------------


class _Line:
    def __init__(self, size: float, height: float) -> None:
        self.font_size = size
        self.text = "body text here"
        self.rect = (50.0, 700.0, 350.0, 700.0 + height)


def test_the_document_ratio_is_measured_from_its_own_lines() -> None:
    # A face whose em inks 0.90 of it: 10pt reported, 9.0pt ink.
    pages = [[_Line(10.0, 9.0) for _ in range(30)]]
    ratio = measure_ink_to_em_ratio(pages)
    assert ratio is not None and ratio == pytest.approx(0.9, abs=0.01)


def test_a_tiny_sample_falls_back_to_the_documented_constant() -> None:
    pages = [[_Line(10.0, 9.0)]]
    assert measure_ink_to_em_ratio(pages) is None
    size = source_body_size(pages[0], ink_to_em_ratio=None)
    # Reported sizes are usable here, so the size comes from them regardless.
    assert size == pytest.approx(10.0)


def test_the_measured_ratio_drives_the_ink_fallback() -> None:
    # No reported size at all -> ink fallback scaled by the measured ratio.
    measured = [_Line(0.0, 8.0) for _ in range(5)]
    default = source_body_size(measured)
    custom = source_body_size(measured, ink_to_em_ratio=1.0)
    assert default == pytest.approx(8.0 * INK_TO_EM_RATIO, abs=0.05)
    assert custom == pytest.approx(8.0, abs=0.05)


# -- contract evidence -----------------------------------------------------


def test_contract_evidence_keeps_single_character_boxes() -> None:
    """A one-glyph box is exactly what a clipped line or table cell reduces to."""
    from ubt.adapters.pdf.delivery_evidence import _normalise_text

    assert _normalise_text("7") == "7"
