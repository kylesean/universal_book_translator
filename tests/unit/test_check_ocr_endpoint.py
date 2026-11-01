from __future__ import annotations

import sys
from pathlib import Path
from typing import TypedDict

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.check_ocr_endpoint import main  # noqa: E402

pytestmark = pytest.mark.fast


class _FakeState(TypedDict):
    reply: str
    calls: int
    image_bytes: int


_STATE: dict[str, str] = {"reply": ""}


@pytest.fixture()
def fake_post(monkeypatch: pytest.MonkeyPatch) -> _FakeState:
    state: _FakeState = {"reply": _STATE["reply"], "calls": 0, "image_bytes": 0}

    def _fake_post(endpoint: str, model: str, api_key: str, png: bytes, timeout: float) -> str:
        state["calls"] += 1
        state["image_bytes"] = len(png)
        return state["reply"]

    monkeypatch.setattr("scripts.check_ocr_endpoint._post", _fake_post)
    return state


@pytest.fixture()
def pdf(tmp_path: Path) -> Path:
    from pdf_builders import write_text_pdf

    return write_text_pdf(tmp_path / "page.pdf", [["Hello world test page content"]] * 3)


@pytest.mark.parametrize(
    "reply, expect_ok",
    [
        ("# 4. 实验\n\n$M = S \\cdot R \\cdot P$", True),
        ("", False),
        ("OCR:", False),
        ("OCR: OCR: OCR:", False),
    ],
)
def test_probe_verdicts(
    fake_post: dict[str, object], pdf: Path, reply: str, expect_ok: bool
) -> None:
    fake_post["reply"] = reply
    code = main(
        [
            "--pdf",
            str(pdf),
            "--page",
            "1",
            "--endpoint",
            "http://fake/v1/chat/completions",
            "--timeout",
            "10",
        ]
    )
    assert (code == 0) is expect_ok


def test_the_request_carries_the_page_image(fake_post: _FakeState, pdf: Path) -> None:
    fake_post["reply"] = "这是识别出的整页正文内容"
    code = main(["--pdf", str(pdf), "--page", "1", "--endpoint", "http://fake/v1/chat/completions"])
    assert code == 0
    assert fake_post["calls"] == 1
    assert fake_post["image_bytes"] > 0


def test_an_unreachable_endpoint_fails_fast(pdf: Path) -> None:
    code = main(
        [
            "--pdf",
            str(pdf),
            "--endpoint",
            "http://127.0.0.1:9/v1/chat/completions",
            "--timeout",
            "5",
        ]
    )
    assert code == 1


def test_a_missing_pdf_is_a_usage_error(tmp_path: Path) -> None:
    assert main(["--pdf", str(tmp_path / "nope.pdf")]) == 2
