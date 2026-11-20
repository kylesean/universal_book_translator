"""Closure regressions for the remaining deep-review findings.

Covers: Latin-script parity (no script range to measure), recorded coordinate
clamping, the contract evidence no longer exempting short/small text, and the
document-measured ink/em ratio.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ubt.adapters.pdf.artifact_parity import check_artifact_parity, latin_stopword_share

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


# -- contract evidence -----------------------------------------------------


def test_contract_evidence_keeps_single_character_boxes() -> None:
    """A one-glyph box is exactly what a clipped line or table cell reduces to."""
    from ubt.adapters.pdf.delivery_evidence import _normalise_text

    assert _normalise_text("7") == "7"
