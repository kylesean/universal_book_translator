from __future__ import annotations

import pytest

from ubt.core.presets import PRESET_ENGINE_FIELDS, PRESETS, Preset, resolve_engine_params

pytestmark = pytest.mark.fast


def _resolve(preset: Preset | None, **explicit: object) -> dict[str, object]:
    payload: dict[str, object] = dict.fromkeys(PRESET_ENGINE_FIELDS)
    payload.update(explicit)
    return resolve_engine_params(preset, payload)


def test_default_engine_is_auto() -> None:
    from ubt.core.config import UBTConfig

    assert UBTConfig.model_fields["pdf_engine"].default == "auto"


def test_pdf_engine_is_a_preset_field() -> None:
    assert "pdf_engine" in PRESET_ENGINE_FIELDS


@pytest.mark.parametrize("preset", [Preset.HIGH_FIDELITY, Preset.STANDARD])
def test_quality_presets_route_by_probe(preset: Preset) -> None:
    assert _resolve(preset)["pdf_engine"] == "auto"


@pytest.mark.parametrize("preset", [Preset.PREVIEW, Preset.FAST])
def test_fast_presets_pin_the_fast_path(preset: Preset) -> None:
    assert _resolve(preset)["pdf_engine"] == "pdfium"


def test_explicit_engine_beats_the_preset() -> None:
    assert _resolve(Preset.FAST, pdf_engine="docling")["pdf_engine"] == "docling"
    assert _resolve(Preset.HIGH_FIDELITY, pdf_engine="pdfium")["pdf_engine"] == "pdfium"


def test_no_preset_leaves_the_engine_to_the_config_default() -> None:
    assert "pdf_engine" not in _resolve(None)


def test_every_preset_declares_an_engine() -> None:
    for preset, policy in PRESETS.items():
        assert policy.pdf_engine in ("auto", "pdfium", "docling"), preset


def test_every_preset_key_matches_its_enum_member() -> None:
    assert set(PRESETS) == set(Preset)
    for preset, policy in PRESETS.items():
        assert policy.key == preset


def test_vector_math_is_the_only_preset_that_transcribes_formulas() -> None:
    vector = {p for p, pol in PRESETS.items() if pol.formula_enrichment == "on"}
    assert vector == {Preset.VECTOR_MATH}

    assert PRESETS[Preset.HIGH_FIDELITY].formula_enrichment == (
        PRESETS[Preset.STANDARD].formula_enrichment
    )


def test_auto_formula_enrichment_resolves_to_the_crop_path() -> None:
    from ubt.adapters.pdf.docling_parser import resolve_formula_enrichment

    assert resolve_formula_enrichment("auto") is False
    assert resolve_formula_enrichment("on") is True


def test_vector_math_only_changes_the_formula_policy() -> None:
    pub = PRESETS[Preset.HIGH_FIDELITY].engine_overrides()
    math = PRESETS[Preset.VECTOR_MATH].engine_overrides()
    differing = {k for k in pub if pub[k] != math[k]}
    assert differing == {"formula_enrichment"}


def test_each_preset_has_a_distinct_parameter_bundle() -> None:
    bundles: dict[tuple[tuple[str, object], ...], Preset] = {}
    for preset, policy in PRESETS.items():
        signature = tuple(sorted(policy.engine_overrides().items()))
        assert signature not in bundles, f"{preset} duplicates {bundles[signature]}"
        bundles[signature] = preset


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("publication", "high_fidelity"),
        ("publication_math", "vector_math"),
        ("full_reflow", "high_fidelity"),
    ],
)
def test_a_retired_preset_name_names_its_replacement(old: str, new: str) -> None:
    from ubt.core.exceptions import UBTError
    from ubt.core.job_options import validate_request_enums

    with pytest.raises(UBTError) as excinfo:
        validate_request_enums({"preset": old})

    message = str(excinfo.value)
    assert f"was renamed to '{new}'" in message
    assert "no longer accepted" in message


def test_an_unknown_preset_gets_no_migration_hint() -> None:
    from ubt.core.exceptions import UBTError
    from ubt.core.job_options import validate_request_enums

    with pytest.raises(UBTError) as excinfo:
        validate_request_enums({"preset": "publikation"})
    assert "was renamed" not in str(excinfo.value)


def test_the_old_names_are_not_enum_members() -> None:
    for retired in ("publication", "publication_math", "full_reflow"):
        with pytest.raises(ValueError):
            Preset(retired)
    assert {p.value for p in Preset} == {
        "high_fidelity",
        "standard",
        "vector_math",
        "preview",
        "fast",
    }
