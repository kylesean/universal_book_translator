from __future__ import annotations

import shutil
import subprocess

import pytest

from ubt.render.typst.document_renderer import typst_lang_arguments

pytestmark = pytest.mark.fast


def test_a_region_goes_into_region_not_lang() -> None:
    assert typst_lang_arguments("zh-CN") == ', lang: "zh", region: "CN"'
    assert typst_lang_arguments("pt-BR") == ', lang: "pt", region: "BR"'
    assert typst_lang_arguments("en-US") == ', lang: "en", region: "US"'


def test_a_bare_language_is_passed_through() -> None:
    assert typst_lang_arguments("zh") == ', lang: "zh"'
    assert typst_lang_arguments("ja") == ', lang: "ja"'


def test_a_script_subtag_is_dropped_not_sent_as_a_region() -> None:
    assert typst_lang_arguments("zh-Hant-TW") == ', lang: "zh", region: "TW"'
    assert "HANT" not in typst_lang_arguments("zh-Hant-TW").upper().replace("HANT", "")


def test_an_underscore_separator_is_normalized() -> None:
    assert typst_lang_arguments("zh_CN") == ', lang: "zh", region: "CN"'


@pytest.mark.parametrize("value", ["", None, "  "])
def test_an_absent_tag_emits_nothing(value: str | None) -> None:
    assert typst_lang_arguments(value) == ""


def test_the_generated_argument_actually_compiles() -> None:
    typst = shutil.which("typst")
    if typst is None:
        pytest.skip("typst is not installed")

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp, "t.typ")
        src.write_text(
            "#set text(size: 10.5pt" + typst_lang_arguments("zh-CN") + ")\n中文测试\n",
            encoding="utf-8",
        )
        out = Path(tmp, "t.pdf")
        result = subprocess.run(
            [typst, "compile", str(src), str(out)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert out.is_file()
