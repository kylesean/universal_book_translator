"""The calibration registry stays complete (discipline lock)."""

import inspect

import pytest

import ubt.core.policy.layout_policy as policy
from ubt.core.policy.layout_policy import CALIBRATION, calibration_summary


def test_every_knob_has_calibration() -> None:
    summary = calibration_summary()
    assert sum(summary.values()) == len(CALIBRATION)
    assert set(summary) == {"proven", "single_doc", "hypothesis"}
    for name, meta in CALIBRATION.items():
        # `isinstance(meta.status, Calibration)` and `isinstance(x, object)`
        # could not fail for any input, so they guarded nothing. What does break
        # is a knob registered without evidence, or registered under a name the
        # policy module does not export.
        assert meta.rationale.strip(), f"{name} registers no calibration evidence"
        assert hasattr(policy, name), f"{name} is registered but not a policy attribute"


def test_status_counts_are_sane() -> None:
    summary = calibration_summary()
    # Most knobs are single-doc: honest about the n=1 calibration debt. Kept at
    # ">=" so registering a genuinely mechanical knob (a POSIX permission mode)
    # cannot fail the suite — a red light here would push people to mislabel
    # proven values as single_doc just to stay green.
    assert summary["single_doc"] >= summary["proven"]


def test_anchored_floor_is_registered_and_in_force() -> None:
    """The anchored typesetter must consume the registry floor.

    The registry advertised 6.5 as the validated floor while the anchored
    engine silently ran its own inline 7.5, so the documented "single source
    of truth" was neither the documented value nor the one in force.
    """
    from ubt.adapters.pdf.rigid.typesetter import RigidTypesetter

    anchored_default = inspect.signature(RigidTypesetter.__init__).parameters["min_font_pt"].default
    assert anchored_default == policy.RIGID_MIN_FONT_PT == 7.0
    # Registered (name -> calibration note) and distinguishable from the
    # FlowFitter default the anchored engine deliberately overrides.
    assert "RIGID_MIN_FONT_PT" in CALIBRATION
    assert policy.RIGID_MIN_FONT_PT != policy.FIT_MIN_FONT_PT
    assert "FlowFitter default" in CALIBRATION["FIT_MIN_FONT_PT"].rationale


def test_cjk_metrics_font_is_findable_on_every_supported_os() -> None:
    """The width-metrics font must resolve on Windows and macOS too.

    ``CJK_FONT_CANDIDATES`` was two ``/usr/share`` paths, so an anchored render on
    any other OS aborted with ``DocumentParseError`` unless the user happened to
    know about ``UBT_CJK_FONT`` — and ``ubt doctor``, which delegates to the same
    resolver, could only WARN. Typst is told to find fonts among the *system*
    fonts (no ``--font-path``), so there is no bundle-to-fall-back-on; the
    candidate list itself has to cover the platform.

    Entries must stay ``.ttc``: the resolver reads them with fontTools'
    ``TTCollection``, which raises ``TTLibError: Not a Font Collection`` on a bare
    ``.ttf``/``.otf`` instead of degrading.
    """
    candidates = policy.CJK_FONT_CANDIDATES
    assert candidates and all(c.endswith(".ttc") for c in candidates), candidates
    assert any(c.startswith("/usr/share/fonts") for c in candidates)
    assert any(c.startswith("C:/Windows/Fonts/") for c in candidates)
    assert any(c.startswith("/System/Library/Fonts/") for c in candidates)
    # Linux keeps measuring the face it always measured: Noto stays first.
    assert candidates[0].startswith("/usr/share/fonts")


def test_anchored_fitter_takes_registered_precision(monkeypatch: pytest.MonkeyPatch) -> None:
    """Precision_pt is no longer re-typed inline as 0.1."""
    import ubt.adapters.pdf.font_metrics as font_metrics
    from ubt.adapters.pdf.rigid.typesetter import RigidTypesetter

    monkeypatch.setattr(font_metrics, "load_width_font", lambda *_a: object())
    monkeypatch.setattr(font_metrics, "resolve_cjk_ttc", lambda: "/nonexistent/font.ttc")

    fitter = RigidTypesetter()._fitter_obj()
    assert fitter.min_font_pt == policy.RIGID_MIN_FONT_PT
    assert fitter.precision_pt == policy.FIT_PRECISION_PT
