"""Every doctor code the engine can emit must have a Chinese rendering.

The panel is the CLI checklist, and the engine's copy is English. Each check ships
a stable code plus the values interpolated into its English sentence, so the console
renders its own text from a catalogue (`web/src/i18n/translations/doctorCodes.ts`).

A code with no entry falls back to English — deliberate, so a console that lags the
engine shows one untranslated line instead of a raw key. That fallback is also how a
new check silently ships untranslated, which is what these tests prevent: the engine
side is enumerated from the source, so adding a check without a translation fails CI.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DOCTOR = _REPO_ROOT / "ubt" / "cli" / "commands" / "doctor.py"
_CATALOGUE = _REPO_ROOT / "web" / "src" / "i18n" / "translations" / "doctorCodes.ts"


def _engine_codes() -> dict[str, set[str]]:
    """The codes the engine declares it can emit.

    Read from ``CHECK_CODES`` rather than scraped from the source: ``probe_writable``
    builds its codes with an f-string, so no literal scan can see
    ``ledger_dir.not_writable``, and a scan that silently misses a code would let
    an untranslated check ship.
    """
    from ubt.cli.commands.doctor import CHECK_CODES

    return {section: set(codes) for section, codes in CHECK_CODES.items()}


def _catalogue_zh() -> dict[str, set[str]]:
    """The `zh` table's keys, split by section."""
    source = _CATALOGUE.read_text(encoding="utf-8")
    zh = source[source.index("doctorCodesZh") : source.index("doctorCodesEn")]
    out: dict[str, set[str]] = {}
    for section in ("groups", "names", "details", "fixes"):
        marker = f"  {section}: {{"
        start = zh.index(marker) + len(marker)
        end = zh.index("\n  },", start)
        body = zh[start:end]
        out[section] = set(re.findall(r"^\s+'?([\w.]+)'?:", body, re.MULTILINE))
    return out


def _group_codes() -> set[str]:
    """Section identifiers, which the engine keeps in its own ordered tuple."""
    from ubt.cli.commands.doctor import _GROUPS

    return {code for code, _label in _GROUPS}


@pytest.mark.parametrize("section", ["groups", "names", "details", "fixes"])
def test_every_engine_code_has_a_chinese_rendering(section: str) -> None:
    """A missing entry is an untranslated line in the console, not an error.

    That is the right runtime behaviour and the wrong build behaviour: without this
    guard, adding a check to the engine ships an English sentence in the Chinese UI
    and nothing reports it.
    """
    if not _CATALOGUE.exists():
        pytest.skip("web/ frontend is not present in this checkout")

    engine = _group_codes() if section == "groups" else _engine_codes()[section]
    translated = _catalogue_zh()[section]

    missing = sorted(engine - translated)
    assert not missing, f"{section} codes with no Chinese copy: {missing}"


def test_the_catalogue_has_no_stale_codes() -> None:
    """A code the engine no longer emits is dead weight that hides drift.

    Renaming a code in the engine would otherwise leave its translation behind,
    looking maintained while nothing reaches it.
    """
    if not _CATALOGUE.exists():
        pytest.skip("web/ frontend is not present in this checkout")

    declared = _engine_codes()
    translated = _catalogue_zh()

    stale = sorted(translated["groups"] - _group_codes())
    assert not stale, f"groups codes the engine never emits: {stale}"
    for section, codes in declared.items():
        stale = sorted(translated[section] - codes)
        assert not stale, f"{section} codes the engine never emits: {stale}"


def test_the_declared_codes_match_what_a_run_actually_produces() -> None:
    """The declaration must not drift from the code.

    ``CHECK_CODES`` is a promise to the console; a ``record()`` call using a code
    absent from it would ship untranslated, and a code listed but never emitted
    would keep a dead translation alive.
    """
    from ubt.cli.commands.doctor import collect_checks
    from ubt.core.config import UBTConfig

    checks = collect_checks(UBTConfig.from_env())
    emitted_names = {check.code for check in checks}
    emitted_details = {check.detail_code for check in checks if check.detail_code}
    emitted_fixes = {check.fix_code for check in checks if check.fix_code}

    declared = _engine_codes()
    # Every code a run produces must be declared (the reverse cannot be asserted
    # here: a branch this machine does not take still legitimately declares codes).
    assert emitted_names <= declared["names"], emitted_names - declared["names"]
    assert emitted_details <= declared["details"], emitted_details - declared["details"]
    assert emitted_fixes <= declared["fixes"], emitted_fixes - declared["fixes"]


def test_every_check_carries_the_codes_a_console_needs() -> None:
    """Structural: the dataclass fields are populated, not left at defaults.

    A `record()` call that omits `code` still renders in English, so nothing fails
    visibly — the panel just silently stops translating that row.
    """
    from ubt.cli.commands.doctor import collect_checks
    from ubt.core.config import UBTConfig

    checks = collect_checks(UBTConfig.from_env())
    assert checks, "collect_checks returned nothing"
    for check in checks:
        assert check.group_code, f"{check.name}: no group_code"
        assert check.code, f"{check.name}: no code"
        assert check.detail_code, f"{check.name}: no detail_code"
        # A fix without a fix_code would render in English inside a translated row.
        if check.fix:
            assert check.fix_code, f"{check.name}: has a fix but no fix_code"


def test_the_api_exposes_the_codes() -> None:
    """The console cannot localize what the API strips."""
    from fastapi.testclient import TestClient

    from ubt.api.app import create_app
    from ubt.core.config import UBTConfig

    client = TestClient(create_app(UBTConfig.from_env()))
    payload = client.get("/system/doctor").json()
    first = payload["checks"][0]
    for key in ("group_code", "code", "detail_code", "fix_code", "params"):
        assert key in first, f"the doctor payload dropped {key}"
