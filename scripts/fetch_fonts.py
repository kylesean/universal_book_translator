#!/usr/bin/env python3
"""Fetch the redistributable font set the renderer typesets against.

The fonts are deliberately not committed: a subset covering the CJK unified
ideographs measures 6.5–15 MB per language (measured with ``pyftsubset``), and
every clone would pay for fonts most checkouts never render. The precedent is
``corpus/semantic_gold`` — fetched on demand, pinned by digest.

Usage:
    uv run python scripts/fetch_fonts.py              # fetch what is missing
    uv run python scripts/fetch_fonts.py --dry-run    # show the plan only
    uv run python scripts/fetch_fonts.py --force      # re-download everything
    uv run python scripts/fetch_fonts.py --update-digests  # record new hashes

Normally you do not need this: the renderer falls back to system fonts and the
quality report says ``bundled: false``. Fetch the set when you want delivery
output to be independent of the host's font packages — CI, containers, and any
release artifact you intend to reproduce.

**Why the source is a pinned upstream URL rather than the local system.** A
system font is whatever the distribution shipped; using it would make the
"reproducible" claim depend on which Linux release built the bundle. The
subset is generated here from the upstream Noto release, so the digest pins
what was fetched *and* how it was cut.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ubt.render.typst.fonts import (  # noqa: E402
    BUNDLED_FONTS,
    DEFAULT_FONT_DIR,
    FontDigestMismatch,
    digest,
    font_dir,
)

#: Upstream releases. ``Noto CJK`` ships one ``.ttc`` holding SC/TC/JP/KR/...
#: and the subset step below extracts the face this project renders with
#: (Simplified Chinese; the others share the ideograph block).
NOTO_CJK_URL = (
    "https://github.com/notofonts/noto-cjk/releases/download/Serif2.003/09_NotoSerifCJKsc.zip"
)
#: The JP and KR faces are not published as their own zips at this release, so
#: they are extracted by face index from the multi-face collection. One archive
#: holds every regional face; ``pyftsubset --font-number`` picks one.
NOTO_CJK_OTC_URL = (
    "https://github.com/notofonts/noto-cjk/releases/download/Serif2.003/04_NotoSerifCJKOTC.zip"
)
#: Face indices inside the Serif CJK collection, measured with fontTools on the
#: 2.003 release. Pinned as data: a reordered release would silently typeset
#: Korean with Japanese glyph forms — legible, and wrong.
NOTO_CJK_FACE_INDEX = {"jp": 0, "kr": 1, "sc": 2, "tc": 3, "hk": 4}
NOTO_SANS_URL = (
    "https://github.com/notofonts/noto-cjk/releases/download/Sans2.004/08_NotoSansCJKsc.zip"
)
#: Latin face. Noto Serif shares the OFL and the design language with the CJK
#: release above, so a mixed page does not jump between two type traditions.
#: (Liberation was the first choice; its GitHub releases carry no assets, so a
#: digest could not be pinned to a version.)
#: Arabic and Hebrew: the pan-CJK faces carry neither block (measured 0% each),
#: so a delivery in those scripts needs their own files.
NOTO_NASKH_ARABIC_URL = (
    "https://github.com/notofonts/arabic/releases/download/"
    "NotoNaskhArabic-v2.021/NotoNaskhArabic-v2.021.zip"
)
NOTO_SERIF_HEBREW_URL = (
    "https://github.com/notofonts/hebrew/releases/download/"
    "NotoSerifHebrew-v2.004/NotoSerifHebrew-v2.004.zip"
)
NOTO_SERIF_LATIN_URL = (
    "https://github.com/notofonts/latin-greek-cyrillic/releases/download/"
    "NotoSerif-v2.015/NotoSerif-v2.015.zip"
)
#: Thai and Devanagari. Every Noto release of these scripts ships four builds of
#: the same family: ``full`` (the complete repertoire, ASCII included),
#: ``googlefonts`` (identical to full), ``hinted`` and ``unhinted`` (subset to
#: the script, carrying 2/95 and 39/95 ASCII). Only ``full`` can be isolated:
#: with system fonts sealed, the hinted build typesets every Latin word and
#: digit as tofu. The build is named in ``NOTO_VARIANT`` rather than globbed,
#: for the same reason the CJK weight is named.
NOTO_SERIF_THAI_URL = (
    "https://github.com/notofonts/thai/releases/download/"
    "NotoSerifThai-v2.002/NotoSerifThai-v2.002.zip"
)
NOTO_SERIF_DEVANAGARI_URL = (
    "https://github.com/notofonts/devanagari/releases/download/"
    "NotoSerifDevanagari-v2.006/NotoSerifDevanagari-v2.006.zip"
)
#: Which build of a script release to take. See above; ``hinted``/``unhinted``
#: are deliberately unreachable.
NOTO_VARIANT = "full"

#: Coverage for the CJK subsets. Every CJK unified ideograph plus the
#: punctuation a book actually uses.
#:
#: **The SC cut deliberately excludes kana and hangul.** Upstream ships one
#: pan-CJK face carrying every script, so the first version of this script cut
#: a single file serving Chinese, Japanese and Korean. That file is what broke:
#: measured with Typst 0.15.1 under ``--ignore-system-fonts``, a subset holding
#: the ideographs *plus* both kana and hangul (33,624 glyphs) typesets ``：``,
#: ``；`` and ``？`` as CID 0 — the glyphs exist in the font and the page looks
#: right, but the PDF's CIDSet declares only CID 0, so the characters are gone
#: from the text layer. Copy, search and the artifact check all fail on text a
#: reader can see. Neither block alone triggers it (ideographs+hangul = 32,597
#: glyphs is clean; ideographs+kana = 21,799 is clean), and the un-subsetted
#: face at 65,535 glyphs is clean, so this is a defect in the subsetting path,
#: not in the font.
#:
#: Since the JP and KR cuts already exist as their own files, the SC cut has no
#: reason to carry kana or hangul: dropping both removes the trigger *and* the
#: 2.7 MB they cost. Chinese coverage is unchanged (ideographs 100%, CJK
#: punctuation 100%, fullwidth forms 93.8%, ASCII 100%).
CJK_SUBSET_UNICODES = ",".join(
    [
        "U+0020-007E",  # ASCII
        "U+00A0-00FF",  # Latin-1 supplement (accents, currency)
        "U+2000-206F",  # general punctuation (quotes, dashes, ellipsis)
        "U+20A0-20CF",  # currency symbols
        "U+2100-214F",  # letterlike symbols (№, ™)
        "U+2190-21FF",  # arrows (common in technical prose)
        "U+2200-22FF",  # mathematical operators
        "U+3000-303F",  # CJK punctuation
        "U+4E00-9FFF",  # CJK unified ideographs
        "U+FE30-FE4F",  # CJK compatibility forms
        "U+FF00-FFEF",  # fullwidth forms
    ]
)

#: Kana and hangul, cut for the regional faces that own them. Kept as a
#: separate spec so the SC cut above stays free of the combination that breaks
#: Typst's subsetting; see the note there.
KANA_HANGUL_UNICODES = ",".join(
    [
        "U+1100-11FF",  # Hangul Jamo
        "U+3040-30FF",  # Hiragana + Katakana
        "U+3130-318F",  # Hangul compatibility Jamo
        "U+A960-A97F",  # Hangul Jamo Extended-A
        "U+AC00-D7A3",  # Hangul syllables
        "U+D7B0-D7FF",  # Hangul Jamo Extended-B
    ]
)

#: Layout features worth keeping: vertical typesetting and kerning are what a
#: book needs. Keeping ``*`` cost 2.4× the bytes for features no delivery uses.
SUBSET_LAYOUT = "kern,liga,vert,vrt2,halt,palt"


def _download(url: str, dest: Path, *, timeout: int = 180) -> None:
    print(f"  ↓ {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "ubt-fonts/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            dest.write_bytes(response.read())
    except urllib.error.URLError as exc:
        raise SystemExit(f"download failed: {url}\n  {exc}") from exc


def _subset(source: Path, dest: Path, *, font_number: int = 0, unicodes: str | None = None) -> None:
    """Cut ``source`` down to the delivery charset.

    ``unicodes`` defaults to :data:`CJK_SUBSET_UNICODES`, the Chinese cut. A
    face that owns kana or hangul passes the spec that includes them; keeping
    the two specs apart is what stops the SC cut from carrying the combination
    that breaks Typst's subsetting (see :data:`CJK_SUBSET_UNICODES`).
    """
    if shutil.which("pyftsubset") is None:
        raise SystemExit(
            "pyftsubset is required to build the font subset.\n"
            "  uv pip install fonttools   (or: pip install fonttools)"
        )
    cmd = [
        "pyftsubset",
        str(source),
        f"--unicodes={unicodes or CJK_SUBSET_UNICODES}",
        f"--output-file={dest}",
        f"--font-number={font_number}",
        f"--layout-features={SUBSET_LAYOUT}",
        "--drop-tables+=DSIG",
        "--no-hinting",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"pyftsubset failed for {source.name}:\n{proc.stderr[-800:]}")


def _obtain(filename: str, dest: Path) -> None:
    """Build ``dest`` from the pinned upstream release."""
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        # Named before the broad `NotoSerifCJK` prefix: `NotoSerifCJKjp`
        # starts with `NotoSerifCJK` too, and the first match wins.
        if filename.startswith("NotoSerifCJKjp") or filename.startswith("NotoSerifCJKkr"):
            # JP/KR glyph forms differ from SC for many shared ideographs
            # (直, 骨, 角 ...). Rendering Japanese with the SC face is legible
            # but reads as Chinese typesetting, so each region gets its own cut.
            # These faces own kana/hangul, so they take the wider spec: it is
            # the SC cut that must stay free of that combination.
            face = "jp" if "CJKjp" in filename else "kr"
            archive = work / "noto-otc.zip"
            _download(NOTO_CJK_OTC_URL, archive)
            shutil.unpack_archive(str(archive), str(work))
            # The archive holds one collection *per weight* (Bold, Light, …).
            # Picking the first match silently typesets the book in whichever
            # weight sorted first — measured: every file came out SemiBold.
            # The weight is part of the contract, so it is named explicitly.
            candidates = [
                p
                for p in [*work.rglob("*.ttc"), *work.rglob("*.otc")]
                if p.stem.endswith("-Regular")
            ]
            if not candidates:
                raise SystemExit("no *-Regular collection in the OTC archive")
            regional = ",".join([CJK_SUBSET_UNICODES, KANA_HANGUL_UNICODES])
            _subset(
                candidates[0],
                dest,
                font_number=NOTO_CJK_FACE_INDEX[face],
                unicodes=regional,
            )
        elif filename.startswith("NotoSerifCJK"):
            archive = work / "noto-serif.zip"
            _download(NOTO_CJK_URL, archive)
            shutil.unpack_archive(str(archive), str(work))
            # The zip carries every weight; taking the first ``*.otf`` picked
            # SemiBold on this release (measured: the delivered family name came
            # out "Noto Serif CJK SC SemiBold"). The weight is part of the
            # contract, so it is named.
            regular = [p for p in work.rglob("*.otf") if p.stem.endswith("-Regular")]
            if not regular:
                raise SystemExit("no *-Regular face in the Noto Serif CJK zip")
            _subset(regular[0], dest)
        elif filename.startswith("NotoSansCJK"):
            archive = work / "noto-sans.zip"
            _download(NOTO_SANS_URL, archive)
            shutil.unpack_archive(str(archive), str(work))
            # Same trap as the Serif zip: every weight is present and rglob
            # order is not weight order. Name the weight.
            regular = [p for p in work.rglob("*.otf") if p.stem.endswith("-Regular")]
            if not regular:
                raise SystemExit("no *-Regular face in the Noto Sans CJK zip")
            _subset(regular[0], dest)
        elif filename.startswith("NotoNaskhArabic"):
            archive = work / "naskh.zip"
            _download(NOTO_NASKH_ARABIC_URL, archive)
            shutil.unpack_archive(str(archive), str(work))
            candidates = [
                p for p in work.rglob("NotoNaskhArabic-Regular.ttf") if "VF" not in p.name
            ]
            if not candidates:
                raise SystemExit("NotoNaskhArabic-Regular.ttf not found in the release")
            shutil.copyfile(candidates[0], dest)
        elif filename.startswith("NotoSerifHebrew"):
            archive = work / "hebrew.zip"
            _download(NOTO_SERIF_HEBREW_URL, archive)
            shutil.unpack_archive(str(archive), str(work))
            candidates = [
                p for p in work.rglob("NotoSerifHebrew-Regular.ttf") if "VF" not in p.name
            ]
            if not candidates:
                raise SystemExit("NotoSerifHebrew-Regular.ttf not found in the release")
            shutil.copyfile(candidates[0], dest)
        elif filename.startswith("NotoSerifThai") or filename.startswith("NotoSerifDevanagari"):
            # Named before the generic `NotoSerif-Regular` branch, which would
            # otherwise never match these anyway — but keeping the pattern
            # explicit means a future face is added by name, not by luck.
            is_thai = filename.startswith("NotoSerifThai")
            url = NOTO_SERIF_THAI_URL if is_thai else NOTO_SERIF_DEVANAGARI_URL
            archive = work / ("thai.zip" if is_thai else "deva.zip")
            _download(url, archive)
            shutil.unpack_archive(str(archive), str(work))
            # Explicit path: the archive holds `full`, `googlefonts`, `hinted`
            # and `unhinted` copies of the same filename, and `rglob` order is
            # not a preference. Only `full` carries ASCII, so picking another
            # build silently ships tofu for every digit under isolation.
            found = sorted(work.rglob(f"{NOTO_VARIANT}/ttf/{filename}"))
            if not found:
                raise SystemExit(f"{filename} not found under {NOTO_VARIANT}/ttf in the release")
            shutil.copyfile(found[0], dest)
        elif filename.startswith("NotoSerif-Regular"):
            archive = work / "noto-serif-latin.zip"
            _download(NOTO_SERIF_LATIN_URL, archive)
            shutil.unpack_archive(str(archive), str(work))
            # The release ships many weights/widths; take the Regular upright,
            # not a variable or italic file.
            candidates = [
                p
                for p in work.rglob("NotoSerif-Regular.ttf")
                if "Italic" not in p.name and "VF" not in p.name
            ]
            if not candidates:
                raise SystemExit("NotoSerif-Regular.ttf not found in the release archive")
            shutil.copyfile(candidates[0], dest)
        else:  # pragma: no cover - only reachable after BUNDLED_FONTS changes
            raise SystemExit(f"no download recipe for {filename}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="show the plan only")
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    parser.add_argument(
        "--update-digests",
        action="store_true",
        help="print the sha256 to paste into BUNDLED_FONTS after a rebuild",
    )
    args = parser.parse_args()

    target = font_dir()
    print(f"font directory: {target}")
    if target != DEFAULT_FONT_DIR:
        print(f"  (from UBT_FONT_DIR; the repo default is {DEFAULT_FONT_DIR})")
    if not args.dry_run:
        target.mkdir(parents=True, exist_ok=True)

    plan: list[tuple[str, Path]] = []
    for face in BUNDLED_FONTS:
        dest = target / face.filename
        if dest.is_file() and dest.stat().st_size > 0 and not args.force:
            actual = digest(dest)
            state = "ok" if not face.sha256 or actual == face.sha256 else "DIGEST MISMATCH"
            print(f"  = {face.filename} ({dest.stat().st_size // 1024} KiB, {state})")
            if state != "ok":
                plan.append((face.filename, dest))
            continue
        print(f"  - {face.filename} (missing)")
        plan.append((face.filename, dest))

    if not plan:
        print("\nfont set complete.")
        return 0
    if args.dry_run:
        print(f"\nwould fetch {len(plan)} file(s); re-run without --dry-run")
        return 0

    for filename, dest in plan:
        print(f"\nbuilding {filename}")
        _obtain(filename, dest)
        actual = digest(dest)
        expected = next(f.sha256 for f in BUNDLED_FONTS if f.filename == filename)
        print(f"  wrote {dest.stat().st_size // 1024} KiB, sha256 {actual}")
        if args.update_digests:
            print(f'  -> set sha256="{actual}" for {filename} in ubt/render/typst/fonts.py')
        elif expected and actual != expected:
            raise FontDigestMismatch(filename, expected, actual)

    print(
        "\nfont set ready. Verify with: uv run python -c "
        "'from ubt.render.typst.fonts import describe; print(describe())'"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
