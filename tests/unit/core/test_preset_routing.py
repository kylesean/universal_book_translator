"""Contract tests for preset-driven PDF engine routing.

The routing model (docs/guides/PDF_ENGINE_ROUTING_REDESIGN.md):

* the default engine is ``auto`` — the probe picks the engine that can read
  the document, so an unconfigured user gets the best delivery rather than the
  fastest run (pdfium shatters a table/formula-heavy paper into fragments);
* ``high_fidelity`` / ``standard`` select ``auto`` as well;
* ``preview`` / ``fast`` select ``pdfium``;
* an explicit ``--pdf-engine`` always beats the preset.
"""

from __future__ import annotations

import pytest

from ubt.core.presets import PRESET_ENGINE_FIELDS, PRESETS, Preset, resolve_engine_params

pytestmark = pytest.mark.fast


def _resolve(preset: Preset | None, **explicit: object) -> dict[str, object]:
    payload: dict[str, object] = dict.fromkeys(PRESET_ENGINE_FIELDS)
    payload.update(explicit)
    return resolve_engine_params(preset, payload)


def test_default_engine_is_auto() -> None:
    # ``auto`` is the default because it picks the engine that can actually read
    # the document. A fast default (pdfium) shipped a shattered paper to the
    # user on 2026-10-10 (1801 fragments, 0 tables) where docling gave 350
    # structured blocks — the default must optimise delivery, not speed.
    from ubt.core.config import UBTConfig

    assert UBTConfig.model_fields["pdf_engine"].default == "auto"


def test_pdf_engine_is_a_preset_field() -> None:
    assert "pdf_engine" in PRESET_ENGINE_FIELDS


@pytest.mark.parametrize("preset", [Preset.HIGH_FIDELITY, Preset.STANDARD])
def test_quality_presets_route_by_probe(preset: Preset) -> None:
    # auto = the probe may pick docling on a layout/scan signal.
    assert _resolve(preset)["pdf_engine"] == "auto"


@pytest.mark.parametrize("preset", [Preset.PREVIEW, Preset.FAST])
def test_fast_presets_pin_the_fast_path(preset: Preset) -> None:
    assert _resolve(preset)["pdf_engine"] == "pdfium"


def test_explicit_engine_beats_the_preset() -> None:
    # An explicit --pdf-engine docling must win over --preset fast.
    assert _resolve(Preset.FAST, pdf_engine="docling")["pdf_engine"] == "docling"
    assert _resolve(Preset.HIGH_FIDELITY, pdf_engine="pdfium")["pdf_engine"] == "pdfium"


def test_no_preset_leaves_the_engine_to_the_config_default() -> None:
    # An un-preset run must not force a value: the config default applies.
    assert "pdf_engine" not in _resolve(None)


def test_every_preset_declares_an_engine() -> None:
    for preset, policy in PRESETS.items():
        assert policy.pdf_engine in ("auto", "pdfium", "docling"), preset


def test_every_preset_key_matches_its_enum_member() -> None:
    # ``PresetPolicy.key`` is what ``engine_overrides`` and the CLI layer read;
    # a preset added to the enum but missing from the table (or keyed with a
    # mismatched member) would silently resolve to no overrides at all.
    assert set(PRESETS) == set(Preset)
    for preset, policy in PRESETS.items():
        assert policy.key == preset


def test_vector_math_is_the_only_preset_that_transcribes_formulas() -> None:
    # ``formula_enrichment="auto"`` resolves to False (crop the source region),
    # so any copy promising "vector formula re-typesetting" for high_fidelity or
    # standard describes behaviour no preset delivers. Only ``on`` runs the VLM.
    vector = {p for p, pol in PRESETS.items() if pol.formula_enrichment == "on"}
    assert vector == {Preset.VECTOR_MATH}

    # And the pair that the old UI copy conflated must stay identical on math.
    assert PRESETS[Preset.HIGH_FIDELITY].formula_enrichment == (
        PRESETS[Preset.STANDARD].formula_enrichment
    )


def test_auto_formula_enrichment_resolves_to_the_crop_path() -> None:
    # Pins the engine fact the preset copy depends on. If ``auto`` ever starts
    # meaning "ask the VLM", the high_fidelity/standard copy above goes stale.
    from ubt.adapters.pdf.docling_parser import resolve_formula_enrichment

    assert resolve_formula_enrichment("auto", lambda: True) is False
    assert resolve_formula_enrichment("auto", lambda: False) is False
    assert resolve_formula_enrichment("on", lambda: False) is True


def test_vector_math_only_changes_the_formula_policy() -> None:
    # The preset must differ from high_fidelity in exactly one knob: otherwise its
    # label ("high_fidelity, but vector math") would under-describe the change.
    pub = PRESETS[Preset.HIGH_FIDELITY].engine_overrides()
    math = PRESETS[Preset.VECTOR_MATH].engine_overrides()
    differing = {k for k in pub if pub[k] != math[k]}
    assert differing == {"formula_enrichment"}


def test_each_preset_has_a_distinct_parameter_bundle() -> None:
    # The wizard badges claim the presets are distinguishable; two presets that
    # resolve to identical engine parameters would make the choice a no-op.
    bundles: dict[tuple[tuple[str, object], ...], Preset] = {}
    for preset, policy in PRESETS.items():
        signature = tuple(sorted(policy.engine_overrides().items()))
        assert signature not in bundles, f"{preset} duplicates {bundles[signature]}"
        bundles[signature] = preset
