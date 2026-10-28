from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import scripts.verify_sample_book as verifier  # noqa: E402


def test_build_parser_defaults() -> None:
    parser = verifier.build_parser()
    args = parser.parse_args([])
    assert args.target_lang == "zh"
    assert args.sample == ROOT / "examples" / "sample_book" / "deep_learning_primer.md"
    assert args.output_dir == ROOT / "output" / "sample_verify"
    assert not args.strict
    assert not args.quiet


def test_build_parser_custom_args(tmp_path: Path) -> None:
    parser = verifier.build_parser()
    custom_sample = tmp_path / "custom.md"
    custom_out = tmp_path / "out"
    args = parser.parse_args(
        [
            "--sample",
            str(custom_sample),
            "--output-dir",
            str(custom_out),
            "--target-lang",
            "ja",
            "--strict",
            "--quiet",
        ]
    )
    assert args.sample == custom_sample
    assert args.output_dir == custom_out
    assert args.target_lang == "ja"
    assert args.strict
    assert args.quiet


@pytest.mark.asyncio
async def test_run_verification_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "nonexistent.md"
    out = tmp_path / "out"
    result = await verifier.run_verification(missing, out)
    assert not result.passed
    assert any("Sample book file not found" in err for err in result.errors)


@pytest.mark.asyncio
async def test_run_verification_e2e(tmp_path: Path) -> None:
    sample = ROOT / "examples" / "sample_book" / "deep_learning_primer.md"
    assert sample.exists(), "Reference sample book must exist"

    out = tmp_path / "verify_run"
    result = await verifier.run_verification(sample, out, target_lang="zh")

    assert result.passed
    assert result.total_chapters == 4
    assert result.total_blocks >= 20
    assert result.heading_count >= 5
    assert result.code_block_count >= 1
    assert result.math_formula_count >= 2
    assert result.table_count >= 1
    assert result.typst_markup_bytes > 0
    assert (out / "sample_verified.typ").exists()


def test_verification_result_passed_logic() -> None:
    res = verifier.VerificationResult(sample_path="dummy.md")
    assert not res.passed

    res.total_blocks = 10
    res.heading_count = 3
    res.code_block_count = 1
    res.math_formula_count = 1
    res.table_count = 1
    res.typst_markup_bytes = 100
    res.contract_errors = 0
    assert res.passed

    res.errors.append("Something went wrong")
    assert not res.passed


def test_print_report_smoke() -> None:
    res = verifier.VerificationResult(
        sample_path="dummy.md",
        total_chapters=2,
        total_blocks=15,
        heading_count=4,
        math_formula_count=2,
        code_block_count=1,
        table_count=1,
        typst_markup_bytes=1000,
        pdf_compiled=True,
        pdf_bytes=2000,
        pdf_page_count=2,
        spill_count=0,
    )
    # Smoke test: formatting does not raise
    verifier.print_report(res)
