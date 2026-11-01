from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DOCTOR = _REPO_ROOT / "ubt" / "cli" / "commands" / "doctor.py"
_CATALOGUE = _REPO_ROOT / "web" / "src" / "i18n" / "translations" / "doctorCodes.ts"


def _engine_codes() -> dict[str, set[str]]:
    from ubt.cli.commands.doctor import CHECK_CODES

    return {section: set(codes) for section, codes in CHECK_CODES.items()}


def _catalogue_zh() -> dict[str, set[str]]:
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
    from ubt.cli.commands.doctor import _GROUPS

    return {code for code, _label in _GROUPS}


@pytest.mark.parametrize("section", ["groups", "names", "details", "fixes"])
def test_every_engine_code_has_a_chinese_rendering(section: str) -> None:
    if not _CATALOGUE.exists():
        pytest.skip("web/ frontend is not present in this checkout")

    engine = _group_codes() if section == "groups" else _engine_codes()[section]
    translated = _catalogue_zh()[section]

    missing = sorted(engine - translated)
    assert not missing, f"{section} codes with no Chinese copy: {missing}"


def _base_code(code: str) -> str:
    head, _, tail = code.rpartition(".")
    return head if tail in _VARIANT_SUFFIXES and head else code


_VARIANT_SUFFIXES = frozenset({"via"})


def test_the_catalogue_has_no_stale_codes() -> None:
    if not _CATALOGUE.exists():
        pytest.skip("web/ frontend is not present in this checkout")

    declared = _engine_codes()
    translated = _catalogue_zh()

    stale = sorted(translated["groups"] - _group_codes())
    assert not stale, f"groups codes the engine never emits: {stale}"
    for section, codes in declared.items():
        stale = sorted(code for code in translated[section] if _base_code(code) not in codes)
        assert not stale, f"{section} codes the engine never emits: {stale}"


def test_the_declared_codes_match_what_a_run_actually_produces() -> None:
    from ubt.cli.commands.doctor import collect_checks
    from ubt.core.config import UBTConfig

    checks = collect_checks(UBTConfig.from_env())
    emitted_names = {check.code for check in checks}
    emitted_details = {check.detail_code for check in checks if check.detail_code}
    emitted_fixes = {check.fix_code for check in checks if check.fix_code}

    declared = _engine_codes()
    assert emitted_names <= declared["names"], emitted_names - declared["names"]
    assert emitted_details <= declared["details"], emitted_details - declared["details"]
    assert emitted_fixes <= declared["fixes"], emitted_fixes - declared["fixes"]


def test_every_check_carries_the_codes_a_console_needs() -> None:
    from ubt.cli.commands.doctor import collect_checks
    from ubt.core.config import UBTConfig

    checks = collect_checks(UBTConfig.from_env())
    assert checks, "collect_checks returned nothing"
    for check in checks:
        assert check.group_code, f"{check.name}: no group_code"
        assert check.code, f"{check.name}: no code"
        assert check.detail_code, f"{check.name}: no detail_code"
        if check.fix:
            assert check.fix_code, f"{check.name}: has a fix but no fix_code"


def test_the_api_exposes_the_codes() -> None:
    from fastapi.testclient import TestClient

    from ubt.api.app import create_app
    from ubt.core.config import UBTConfig

    client = TestClient(create_app(UBTConfig.from_env()))
    payload = client.get("/system/doctor").json()
    first = payload["checks"][0]
    for key in ("group_code", "code", "detail_code", "fix_code", "params"):
        assert key in first, f"the doctor payload dropped {key}"
