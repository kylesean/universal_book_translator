"""The K-1 sweep harness must read the registry correctly and classify honestly.

Two distinct failure modes are being guarded here. The scanner can drift from the
source (report a knob as sweepable when it has no override point, so a "band" gets
measured against a value nothing reads), and the classifier can call an
unmeasurable knob clean — the exact mistake recorded in
``docs/design/knob-calibration-protocol.md`` as what
the current defences make.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from ubt.core.policy.layout_policy import (
    FIT_MIN_FONT_PT,
    PUNCT_SQUEEZE_CAP,
    PUNCT_SQUEEZE_PER_PUNCT,
    RIGID_CAPTION_MIN_FONT_PT,
    RIGID_FOOTNOTE_MIN_FONT_PT,
    RIGID_MIN_FONT_PT,
    ROW_MERGE_GAP_PT,
    ROW_MERGE_Y_TOL,
    SHORT_CHAIN_MAX_PAGES,
)

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "knob_sweep.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ubt_knob_sweep", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # No sys.modules registration: the script is written so its dataclasses do not
    # need one, and tests/unit/test_sys_modules_guard.py bans the mutation anyway.
    spec.loader.exec_module(module)
    return module


sweep = _load()


def test_scanner_finds_every_override_point_and_agrees_with_the_module() -> None:
    """A knob listed as sweepable must hold the same value the live module exports."""
    live = {
        "FIT_MIN_FONT_PT": FIT_MIN_FONT_PT,
        "PUNCT_SQUEEZE_CAP": PUNCT_SQUEEZE_CAP,
        "PUNCT_SQUEEZE_PER_PUNCT": PUNCT_SQUEEZE_PER_PUNCT,
        "RIGID_CAPTION_MIN_FONT_PT": RIGID_CAPTION_MIN_FONT_PT,
        "RIGID_FOOTNOTE_MIN_FONT_PT": RIGID_FOOTNOTE_MIN_FONT_PT,
        "RIGID_MIN_FONT_PT": RIGID_MIN_FONT_PT,
        "ROW_MERGE_GAP_PT": ROW_MERGE_GAP_PT,
        "ROW_MERGE_Y_TOL": ROW_MERGE_Y_TOL,
        "SHORT_CHAIN_MAX_PAGES": SHORT_CHAIN_MAX_PAGES,
    }
    found = {k.name: k for k in sweep.sweepable_knobs()}
    assert found.keys() == live.keys(), (
        "the protocol's sweepable set drifted; see docs/design/knob-calibration-protocol.md "
        "states this list and its count"
    )
    for name, value in live.items():
        knob = found[name]
        assert knob.env_var.startswith("UBT_"), name
        # Not `UBT_{name}`: SHORT_CHAIN_MAX_PAGES' override predates the constant
        # rename and is still UBT_SHORT_MAX_PAGES, which is user-facing. Asserting
        # uniqueness keeps a second knob from silently sharing one env var.
        assert knob.env_var != name, name
        assert knob.default == pytest.approx(float(value))
        assert knob.is_int is (name == "SHORT_CHAIN_MAX_PAGES")
    assert len({k.env_var for k in sweep.sweepable_knobs()}) == len(live)


def test_short_chain_override_keeps_its_legacy_env_name() -> None:
    """The constant was renamed; its env var was not, and must stay that way.

    ``UBT_SHORT_MAX_PAGES`` is user-facing. Renaming it to match
    ``SHORT_CHAIN_MAX_PAGES`` would break a working setting for a naming tidy-up —
    recorded here so the next reader does not "fix" it.
    """
    knob = next(k for k in sweep.sweepable_knobs() if k.name == "SHORT_CHAIN_MAX_PAGES")
    assert knob.env_var == "UBT_SHORT_MAX_PAGES"


def test_int_knobs_do_not_sweep_as_floats() -> None:
    """UBT_SHORT_MAX_PAGES feeds an int comparison; 29.999 is not a page count."""
    pages = next(k for k in sweep.sweepable_knobs() if k.name == "SHORT_CHAIN_MAX_PAGES")
    assert isinstance(pages.value_at(1.5), int)
    assert pages.value_at(1.5) == int(SHORT_CHAIN_MAX_PAGES * 1.5)
    ratio = next(k for k in sweep.sweepable_knobs() if k.name == "ROW_MERGE_Y_TOL")
    assert ratio.value_at(1 / 1.5) == pytest.approx(ROW_MERGE_Y_TOL / 1.5)


def test_parse_failures_extracts_failed_and_error_nodeids() -> None:
    """ERROR lines are real measurements: a knob that breaks collection must be
    seen, not read as "no reaction" (`INVISIBLE`)."""
    stdout = (
        ".....................\n"
        "FAILED tests/unit/test_text_fit.py::test_squeezed_width - assert 0.0 > 0\n"
        "FAILED tests/baselines/test_baselines.py::test_baseline_call_of_the_wild\n"
        "ERROR tests/unit/test_broken.py\n"
        "1 failed, 1 error, 21 passed in 3.45s\n"
    )
    assert sweep.parse_failures(stdout) == {
        "tests/unit/test_text_fit.py::test_squeezed_width",
        "tests/baselines/test_baselines.py::test_baseline_call_of_the_wild",
        "tests/unit/test_broken.py",
    }


def _cell(label: str, *failures: str) -> object:
    cell = sweep.Cell(label=label)
    cell.failures = set(failures)
    return cell


@pytest.mark.parametrize(
    ("low_failures", "high_failures", "expected_prefix"),
    [
        (("a",), ("b",), "MEASURED both"),
        (("a",), (), "MEASURED one-sided (only low"),
        ((), ("b",), "MEASURED one-sided (only high"),
        ((), (), "INVISIBLE"),
    ],
)
def test_verdict_does_not_call_an_unmeasured_knob_clean(
    low_failures: tuple[str, ...], high_failures: tuple[str, ...], expected_prefix: str
) -> None:
    """Zero movement in both directions is "the corpus cannot see this", not a pass."""
    baseline = _cell("control")
    outcome = sweep.verdict(baseline, _cell("low", *low_failures), _cell("high", *high_failures))
    assert outcome.startswith(expected_prefix), outcome


def test_a_control_failure_is_not_attributed_to_the_knob() -> None:
    """A test already red at the centre must not read as sensitivity."""
    baseline = _cell("control", "tests/unit/test_already_red.py::x")
    outcome = sweep.verdict(
        baseline,
        _cell("low", "tests/unit/test_already_red.py::x"),
        _cell("high", "tests/unit/test_already_red.py::x"),
    )
    assert outcome.startswith("INVISIBLE")


def test_an_errored_cell_is_never_reported_as_insensitive() -> None:
    """A timeout says nothing about the knob; saying "INVISIBLE" would be a lie."""
    control = _cell("control")
    low = _cell("low")
    low.error = "timeout after 1s"  # type: ignore[attr-defined]
    assert sweep.verdict(control, low, _cell("high")) == "ERROR"
