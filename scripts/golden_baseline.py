#!/usr/bin/env python3
"""Freeze and check a golden baseline for the delivery corpus.

The corpus gate answers "did this document lose content". It does not answer
"did this change make the document *worse*": the numbers are recomputed every
run and never compared to an accepted state, so a fix that repairs one defect
while regressing three others passes silently.

This tool closes that gap with four layers, over the same ``corpus/cases.json``:

1. **Contract** — ``violations == 0`` and ``missing_assets == 0`` (hard gates;
   these are already the corpus's own assertions, restated so a baseline
   comparison can never be looser than the corpus).
2. **Structure** — delivered/total text blocks, reconstructed and preserved
   assets. A drop beyond tolerance means the renderer stopped placing content.
3. **Visible text** — the PDF's own text layer, normalised and compared to the
   frozen copy. Catches the case the counters cannot: same block count, different
   words (a mistranslation, a swapped block, a lost TOC row).
4. **Visual** — SSIM on a sample of pages. Catches layout drift that changes no
   character (an image moved, a table squeezed, a page that now overflows).

Usage::

    # Accept the current artifacts as the baseline (after reviewing them).
    uv run python scripts/golden_baseline.py --accept --artifacts ~/Documents/UBT

    # Check the current artifacts against the frozen baseline (pre-push / CI).
    uv run python scripts/golden_baseline.py --check --artifacts ~/Documents/UBT

Exit code 1 on any regression, so it composes with a shell gate.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"
BASELINE_DIR = CORPUS / "baseline"

#: Filenames an artifact may take inside ``--artifacts``, in preference order.
ARTIFACT_PATTERNS = ("{id}_mono.pdf", "{id}.pdf", "{id}_mono_v3.pdf")

#: Pages sampled for the visual layer. Three spread across the document catch a
#: moved figure or a reflowed opening page without paying a full-page raster of
#: every document; the sample is deterministic so a baseline and a check compare
#: the same pages.
VISUAL_SAMPLE_PAGES = 3

#: SSIM below this is a visual regression. 0.985 is tight enough to catch a
#: figure that moved or a table that reflowed (measured: an unchanged page is
#: 1.000, a page with one moved block ~0.93–0.97).
#:
#: This only holds for a *deterministic* artifact. On a document translated by a
#: real model the text itself changes between runs, so the raster changes too:
#: measured on arXiv 2606.23344, two runs of identical code differed at
#: SSIM 0.64 / 0.23 / 0.23 on the three sampled pages — 30x below this floor,
#: while the baseline-vs-run gap was the same size (0.63 / 0.25 / …). The visual
#: layer therefore cannot gate a model-translated case; it gates the
#: deterministic (``--dry-run``) cases, where the render really is reproducible.
MIN_SSIM = 0.985

#: Allowed drift in the text layer, as a share of the baseline's characters.
#: Prose gets re-ordered by a translation change; a content regression (a
#: dropped paragraph, a lost table) moves this far more.
#:
#: Also only meaningful for a deterministic artifact. Two identical runs of a
#: model-translated document share just 0.83 word-sequence similarity, so a 2%
#: budget flags every run and a 20% budget would stop flagging anything. The
#: model-translated cases are gated by the *content invariants* below instead,
#: which are stable across sampling (measured A-vs-B: numbers 0.9987,
#: identifiers 0.9912, citations 1.0000).
MAX_TEXT_DIFF_SHARE = 0.02

#: Allowed drop in reconstructed-asset coverage. A repair raises it; nothing in
#: the pipeline should lower it, so the ratchet is zero.
MAX_RECONSTRUCTED_DROP = 0

#: Content invariants: facts a translation must carry regardless of how it words
#: them. Measured across two independent runs of the same code (arXiv
#: 2606.23344), reproduction drops by at most 0.88% for identifiers and 0.13%
#: for numbers, and not at all for citations — three orders of magnitude more
#: stable than the word sequence, which moves 17%. These are the layers that can
#: gate a model-translated document; a dropped paragraph or a lost table shows up
#: here as missing numbers/citations/identifiers, which is exactly the regression
#: the corpus exists to catch.
MAX_NUMBER_LOSS = 0.02
MAX_IDENTIFIER_LOSS = 0.02
MAX_CITATION_LOSS = 0.02

#: CJK share may move a little when a re-translation picks different words, but
#: it collapses toward zero on an untranslated page.
MAX_CJK_RATIO_SHIFT = 0.05

#: Word-sequence similarity below this is reported as ADVISORY only: it cannot
#: distinguish a regression from sampling, so it never fails a run. Set below the
#: measured same-code floor (0.83) with room for a differently-worded but correct
#: translation; it catches gross divergence (a page swapped for another's text)
#: without flagging mere rewording.
ADVISORY_TEXT_SIMILARITY = 0.55

#: A run that shares no exact sentence with the baseline is suspicious even when
#: its content invariants hold — it may be a wholesale re-translation or a
#: different document. Reported, never failed, for the same reason as above.
ADVISORY_SENTENCE_OVERLAP = 0.05

_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
_IDENTIFIER_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*")
_CITATION_RE = re.compile(r"\[\d+(?:\s*[,–—-]\s*\d+)*\]")


def _is_fused_number(token: str) -> bool:
    """Whether a numeric token is several values the text layer glued together.

    The PDF text layer joins adjacent table cells with no separator, so a
    results row can arrive as ``0.08290.30`` — two values, nothing between them.
    Counted literally, that token appears in a run whose table reflowed and not
    in the next, which reads as a lost number.

    The discriminator is the decimal points: a real number carries at most one,
    so two or more is proof of fusion. This is deliberately the *only* signal —
    a length or digit-run heuristic also matches real values (``132.1`` fps,
    arXiv id ``2606.23344``), and splitting those corrupts the count far more
    than the fusion it would fix. Such a token is dropped rather than split:
    its component values are counted where they appear on their own, and
    guessing the split points invents numbers that may never have existed.
    """
    return token.count(".") >= 2


def content_features(text: str) -> dict[str, Any]:
    """The facts a translation must carry, independent of how it words them.

    Word sequence is a poor regression signal for a model-translated document:
    two runs of identical code share only 0.83 of it, because the model re-words
    freely. What does *not* move is the content — the numbers, the identifiers,
    the citations. Measured across two such runs (arXiv 2606.23344): numbers
    0.9987, identifiers 0.9912, citations 1.0000, against 0.8305 for the sequence.
    A dropped paragraph, a lost table row or a swallowed bibliography entry shows
    up here immediately; rewording does not.
    """
    from collections import Counter

    numbers: Counter[str] = Counter()
    for raw in _NUMBER_RE.findall(text):
        # Drop a cell-fusion artefact rather than guess where its values split
        # (see ``_is_fused_number``). The values themselves are counted where a
        # table cell kept them separate, which is the layout this run renders.
        if _is_fused_number(raw):
            continue
        numbers[raw] += 1

    return {
        "numbers": dict(numbers),
        "identifiers": dict(Counter(_IDENTIFIER_RE.findall(text))),
        "citations": dict(Counter(_CITATION_RE.findall(text))),
        "chars": len(text),
        "cjk": sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff"),
    }


def _reproduction(reference: dict[str, int], current: dict[str, int]) -> float:
    """Share of the baseline's occurrences that survive in ``current``.

    Counted per token, capped at the baseline count, so a duplicated occurrence
    cannot mask a dropped one elsewhere.
    """
    total = sum(reference.values())
    if total == 0:
        return 1.0
    kept = sum(min(count, current.get(token, 0)) for token, count in reference.items())
    return kept / total


def _deterministic(case: dict[str, Any]) -> bool:
    """Whether this case's artifact is reproducible byte-for-byte.

    ``--dry-run`` echoes the source, so its render really is stable and the
    byte-level layers (word sequence, SSIM) are meaningful. A case translated by
    a model is not stable and must be judged on content invariants instead.
    """
    return bool(case.get("deterministic"))


def _find_artifact(directory: Path, case: dict[str, Any]) -> Path | None:
    """The delivered PDF for a case, matched on its SOURCE stem.

    ``cases.json`` names cases semantically (``twocol-paper-2609``) while the
    pipeline names its output after the source file (``2609.20519v1_mono.pdf``),
    so matching on the case id finds nothing. The source stem is the join key
    the pipeline itself uses.
    """
    source = str(case.get("document") or "")
    stem = Path(source).stem if source else ""
    candidates: list[str] = []
    if stem:
        candidates.extend(pattern.format(id=stem) for pattern in ARTIFACT_PATTERNS)
    candidates.extend(pattern.format(id=str(case.get("id", ""))) for pattern in ARTIFACT_PATTERNS)
    for name in candidates:
        candidate = directory / name
        if candidate.exists():
            return candidate
    return None


def _normalise_text(text: str) -> str:
    """Collapse whitespace so a reflow does not read as a text change.

    Only whitespace is normalised: a digit, a CJK glyph or a punctuation mark
    that appears or disappears is a real content change and must survive.
    """
    return re.sub(r"\s+", " ", text).strip()


def _pdf_text(pdf: Path) -> str:
    import logging

    import pypdf

    # pypdf logs a warning per CFF Type1 font it cannot decode without
    # fontTools; a delivered book has hundreds. They do not affect the text
    # layer this reads, and they bury the report.
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    reader = pypdf.PdfReader(str(pdf))
    return _normalise_text("\n".join((page.extract_text() or "") for page in reader.pages))


def _page_count(pdf: Path) -> int:
    import pypdf

    return len(pypdf.PdfReader(str(pdf)).pages)


def _sample_pages(total: int, wanted: int = VISUAL_SAMPLE_PAGES) -> list[int]:
    """Evenly spread 1-based page numbers, first page always included."""
    if total <= 0:
        return []
    if total <= wanted:
        return list(range(1, total + 1))
    step = (total - 1) / (wanted - 1)
    return sorted({1, *(round(1 + step * i) for i in range(1, wanted))})


def _render_page_png(pdf: Path, page_no: int, resolution: int = 110) -> bytes | None:
    """Rasterise one page via pdftoppm (the same tool the manual reviews use)."""
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory(prefix="ubt-baseline-") as tmp:
        stem = Path(tmp) / "page"
        try:
            subprocess.run(
                [
                    "pdftoppm",
                    "-png",
                    "-r",
                    str(resolution),
                    "-f",
                    str(page_no),
                    "-l",
                    str(page_no),
                    str(pdf),
                    str(stem),
                ],
                check=True,
                capture_output=True,
                timeout=120,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        produced = sorted(Path(tmp).glob("page*.png"))
        return produced[0].read_bytes() if produced else None


def _ssim(a_png: bytes, b_png: bytes) -> float | None:
    """Structural similarity of two page rasters (grayscale), or None."""
    try:
        from io import BytesIO

        import numpy as np
        from PIL import Image
    except ImportError:
        return None

    def gray(png: bytes) -> Any:
        img = Image.open(BytesIO(png)).convert("L")
        return np.asarray(img, dtype=np.float64)

    first, second = gray(a_png), gray(b_png)
    if first.shape != second.shape:
        return 0.0
    mu1, mu2 = first.mean(), second.mean()
    var1, var2 = first.var(), second.var()
    cov = ((first - mu1) * (second - mu2)).mean()
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    denominator = (mu1**2 + mu2**2 + c1) * (var1 + var2 + c2)
    if denominator == 0:
        return 1.0
    return float(((2 * mu1 * mu2 + c1) * (2 * cov + c2)) / denominator)


def _load_cases() -> list[dict[str, Any]]:
    manifest = CORPUS / "cases.json"
    if not manifest.exists():
        return []
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    return list(payload.get("cases", []))


def _contract_for(artifact: Path) -> dict[str, Any] | None:
    """The delivery contract written beside a delivered PDF, if present."""
    candidate = artifact.with_name(f"{artifact.stem}_contract.json")
    if not candidate.exists():
        return None
    try:
        return json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def snapshot(artifacts: Path, case: dict[str, Any]) -> dict[str, Any] | None:
    """Measure one case: structure, text and sampled page rasters."""
    case_id = str(case.get("id", ""))
    artifact = _find_artifact(artifacts, case)
    if artifact is None:
        return None
    contract = _contract_for(artifact) or {}
    text = _pdf_text(artifact)
    total = _page_count(artifact)
    pages = _sample_pages(total)

    rasters: dict[str, str] = {}
    visual_missing: list[int] = []
    for page_no in pages:
        png = _render_page_png(artifact, page_no)
        if png is None:
            visual_missing.append(page_no)
            continue
        rasters[str(page_no)] = base64.b64encode(png).decode("ascii")

    return {
        "case_id": case_id,
        "artifact": artifact.name,
        "pages": total,
        "structure": {
            "total_text": contract.get("total_text"),
            "delivered_text": contract.get("delivered_text"),
            "total_assets": contract.get("total_assets"),
            "reconstructed_assets": contract.get("reconstructed_assets"),
            "preserved_assets": contract.get("preserved_assets"),
            "missing_assets": contract.get("missing_assets", 0),
            "violations": len(contract.get("violations", []) or []),
        },
        "text": text,
        "text_chars": len(text),
        "features": content_features(text),
        "visual": {"pages": pages, "rasters": rasters, "unavailable": visual_missing},
    }


def compare(
    baseline: dict[str, Any], current: dict[str, Any], *, deterministic: bool = True
) -> list[str]:
    """Regressions of ``current`` against ``baseline`` (empty means clean).

    ``deterministic`` says whether the artifact is reproducible byte-for-byte
    (see :func:`_deterministic`). It selects which layers are gates: a
    model-translated document is judged on content invariants, because its word
    sequence and raster both move between runs of identical code.
    """
    problems: list[str] = []

    base_struct = baseline.get("structure", {})
    cur_struct = current.get("structure", {})

    # Layer 1: contract — absolute, not relative to the baseline.
    if cur_struct.get("violations"):
        problems.append(f"contract violations: {cur_struct['violations']}")
    if cur_struct.get("missing_assets"):
        problems.append(f"missing assets: {cur_struct['missing_assets']}")

    # Layer 2: structure, ratcheted.
    for key, label in (
        ("delivered_text", "delivered text blocks"),
        ("total_assets", "total assets"),
    ):
        before, after = base_struct.get(key), cur_struct.get(key)
        if isinstance(before, int) and isinstance(after, int) and after < before:
            problems.append(f"{label} dropped {before} -> {after}")
    # Asset *mix* may legitimately shift: a repair that turns a source crop into
    # real math moves one asset from preserved to reconstructed with the total
    # unchanged, which is the goal, not a regression. Only the total (above) and
    # a loss of reconstructed coverage (below) are ratchets.
    before_rec = base_struct.get("reconstructed_assets")
    after_rec = cur_struct.get("reconstructed_assets")
    if (
        isinstance(before_rec, int)
        and isinstance(after_rec, int)
        and after_rec < before_rec - MAX_RECONSTRUCTED_DROP
    ):
        problems.append(f"reconstructed assets dropped {before_rec} -> {after_rec}")

    # Layer 2b: page count. A lost page is an absolute signal, not a share:
    # on a 15-page paper one missing page is only 1.35% of the characters, which
    # slipped under the text-share threshold and passed (measured).
    base_pages = baseline.get("pages")
    cur_pages = current.get("pages")
    if isinstance(base_pages, int) and isinstance(cur_pages, int) and cur_pages < base_pages:
        problems.append(f"page count dropped {base_pages} -> {cur_pages}")

    # Layer 3: visible text.
    #
    # A character-count drop is checked FIRST and directly: a lost page or an
    # empty render is the regression that matters, and SequenceMatcher on a
    # truncated text still reports a high similarity when the cut is a small
    # share of a long document.
    base_chars = baseline.get("text_chars") or 0
    cur_chars = current.get("text_chars") or 0
    if base_chars and cur_chars < base_chars * (1.0 - MAX_TEXT_DIFF_SHARE):
        problems.append(
            f"visible text shrank {base_chars} -> {cur_chars} chars "
            f"(-{1 - cur_chars / base_chars:.1%}, limit {MAX_TEXT_DIFF_SHARE:.0%})"
        )

    # Layer 3a: content invariants — the layers that hold across sampling.
    # A baseline frozen before this layer existed has no ``features``; derive
    # them from the text it does carry rather than silently skipping the gate.
    base_features = baseline.get("features") or (
        content_features(baseline.get("text", "")) if baseline.get("text") else {}
    )
    cur_features = current.get("features") or (
        content_features(current.get("text", "")) if current.get("text") else {}
    )
    if base_features and cur_features:
        for key, limit, label in (
            ("numbers", MAX_NUMBER_LOSS, "numbers"),
            ("identifiers", MAX_IDENTIFIER_LOSS, "identifiers"),
            ("citations", MAX_CITATION_LOSS, "citations"),
        ):
            reproduced = _reproduction(base_features.get(key) or {}, cur_features.get(key) or {})
            lost = 1.0 - reproduced
            if lost > limit:
                problems.append(
                    f"{label} lost {lost:.1%} of the baseline's occurrences (limit {limit:.0%})"
                )
        # CJK share collapsing means a page came out untranslated (or in the
        # wrong language); a small shift is just different word choice.
        base_cjk, base_n = base_features.get("cjk", 0), base_features.get("chars", 0) or 0
        cur_cjk, cur_n = cur_features.get("cjk", 0), cur_features.get("chars", 0) or 0
        if base_n and cur_n:
            base_ratio, cur_ratio = base_cjk / base_n, cur_cjk / cur_n
            if abs(cur_ratio - base_ratio) > MAX_CJK_RATIO_SHIFT:
                problems.append(
                    f"CJK share moved {base_ratio:.1%} -> {cur_ratio:.1%} "
                    f"(limit {MAX_CJK_RATIO_SHIFT:.0%}, suggests untranslated text)"
                )

    # Layer 3b: word sequence — advisory for a model-translated case, a real
    # gate for a deterministic one. See MAX_TEXT_DIFF_SHARE.
    base_text = baseline.get("text", "")
    cur_text = current.get("text", "")
    notes: list[str] = []
    if base_text != cur_text:
        import difflib

        ratio = difflib.SequenceMatcher(None, base_text, cur_text).ratio()
        changed = 1.0 - ratio
        if deterministic:
            if changed > MAX_TEXT_DIFF_SHARE:
                problems.append(
                    f"visible text changed {changed:.1%} (limit {MAX_TEXT_DIFF_SHARE:.0%})"
                )
        elif changed > 0:
            # Cannot distinguish a regression from sampling: report, never fail.
            if ratio < ADVISORY_TEXT_SIMILARITY:
                notes.append(
                    f"visible text diverged {changed:.1%} from the baseline "
                    f"(similarity {ratio:.2f}); too far to compare word-for-word — "
                    f"content invariants above are the gate"
                )
            else:
                notes.append(
                    f"visible text differs {changed:.1%} (similarity {ratio:.2f}); "
                    f"expected when a model re-words, not evidence of a regression"
                )

    # Layer 4: visual. Only meaningful where the render is reproducible.
    base_visual = baseline.get("visual", {}).get("rasters", {})
    cur_visual = current.get("visual", {}).get("rasters", {})

    if deterministic:
        for page, encoded in base_visual.items():
            if page not in cur_visual:
                problems.append(f"page {page}: no raster to compare (was sampled before)")
                continue
            score = _ssim(base64.b64decode(encoded), base64.b64decode(cur_visual[page]))
            if score is None:
                continue
            if score < MIN_SSIM:
                problems.append(f"page {page}: visual drift (SSIM {score:.3f} < {MIN_SSIM})")
    else:
        missing = [p for p in base_visual if p not in cur_visual]
        if missing:
            notes.append(
                f"no raster for page(s) {', '.join(missing)}; visual layer skipped "
                f"(a model-translated render is not reproducible — measured same-code "
                f"SSIM 0.23–0.64, far below the {MIN_SSIM} floor)"
            )

    current["_notes"] = notes
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--accept", action="store_true", help="freeze current artifacts as baseline")
    mode.add_argument("--check", action="store_true", help="compare artifacts to the baseline")
    ap.add_argument(
        "--artifacts", type=Path, required=True, help="directory holding delivered PDFs"
    )
    ap.add_argument("--cases", nargs="*", default=None, help="only these case ids")
    ap.add_argument("--json", type=Path, default=None, help="write the report here")
    args = ap.parse_args(argv)

    cases = _load_cases()
    if args.cases:
        wanted = set(args.cases)
        cases = [c for c in cases if str(c.get("id")) in wanted]
    if not cases:
        print("no cases to process", file=sys.stderr)
        return 2

    report: dict[str, Any] = {"mode": "accept" if args.accept else "check", "cases": {}}
    failures = 0
    skipped: list[str] = []

    for case in cases:
        case_id = str(case.get("id"))
        current = snapshot(args.artifacts, case)
        if current is None:
            skipped.append(case_id)
            report["cases"][case_id] = {"status": "missing-artifact"}
            continue

        if args.accept:
            target = BASELINE_DIR / case_id
            target.mkdir(parents=True, exist_ok=True)
            (target / "baseline.json").write_text(
                json.dumps(current, ensure_ascii=False), encoding="utf-8"
            )
            report["cases"][case_id] = {
                "status": "accepted",
                "pages": current["pages"],
                "text_chars": current["text_chars"],
            }
            print(f"[accept] {case_id}: {current['pages']} pages, {current['text_chars']} chars")
            continue

        baseline_path = BASELINE_DIR / case_id / "baseline.json"
        if not baseline_path.exists():
            report["cases"][case_id] = {"status": "no-baseline"}
            print(f"[skip]   {case_id}: no frozen baseline (run --accept first)")
            continue
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        problems = compare(baseline, current, deterministic=_deterministic(case))
        if problems:
            failures += 1
            report["cases"][case_id] = {"status": "regressed", "problems": problems}
            print(f"[FAIL]   {case_id}")
            for problem in problems:
                print(f"           - {problem}")
        else:
            notes = current.get("_notes") or []
            report["cases"][case_id] = {"status": "ok", "notes": notes}
            print(f"[ok]     {case_id}")
            for note in notes:
                print(f"           ~ {note}")

    report["failures"] = failures
    report["skipped"] = skipped
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")

    if args.check and failures:
        print(f"\n{failures} case(s) regressed against the frozen baseline", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
