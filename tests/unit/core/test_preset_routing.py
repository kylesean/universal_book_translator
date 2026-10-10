"""Contract tests for preset-driven PDF engine routing.

The routing model (docs/guides/PDF_ENGINE_ROUTING_REDESIGN.md):

* the default engine is ``auto`` — the probe picks the engine that can read
  the document, so an unconfigured user gets the best delivery rather than the
  fastest run (pdfium shatters a table/formula-heavy paper into fragments);
* ``publication`` / ``standard`` select ``auto`` as well;
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


@pytest.mark.parametrize("preset", [Preset.PUBLICATION, Preset.STANDARD])
def test_quality_presets_route_by_probe(preset: Preset) -> None:
    # auto = the probe may pick docling on a layout/scan signal.
    assert _resolve(preset)["pdf_engine"] == "auto"


@pytest.mark.parametrize("preset", [Preset.PREVIEW, Preset.FAST])
def test_fast_presets_pin_the_fast_path(preset: Preset) -> None:
    assert _resolve(preset)["pdf_engine"] == "pdfium"


def test_explicit_engine_beats_the_preset() -> None:
    # An explicit --pdf-engine docling must win over --preset fast.
    assert _resolve(Preset.FAST, pdf_engine="docling")["pdf_engine"] == "docling"
    assert _resolve(Preset.PUBLICATION, pdf_engine="pdfium")["pdf_engine"] == "pdfium"


def test_no_preset_leaves_the_engine_to_the_config_default() -> None:
    # An un-preset run must not force a value: the config default applies.
    assert "pdf_engine" not in _resolve(None)


def test_every_preset_declares_an_engine() -> None:
    for preset, policy in PRESETS.items():
        assert policy.pdf_engine in ("auto", "pdfium", "docling"), preset
