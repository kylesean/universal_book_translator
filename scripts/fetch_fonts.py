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
NOTO_SANS_URL = (
    "https://github.com/notofonts/noto-cjk/releases/download/Sans2.004/08_NotoSansCJKsc.zip"
)
#: Latin face. Noto Serif shares the OFL and the design language with the CJK
#: release above, so a mixed page does not jump between two type traditions.
#: (Liberation was the first choice; its GitHub releases carry no assets, so a
#: digest could not be pinned to a version.)
NOTO_SERIF_LATIN_URL = (
    "https://github.com/notofonts/latin-greek-cyrillic/releases/download/"
    "NotoSerif-v2.015/NotoSerif-v2.015.zip"
)

#: Coverage for the subset: every CJK unified ideograph plus the punctuation a
#: book actually uses. Cutting to GB2312 (7,445 chars) is smaller but ships
#: tofu the moment a book uses a name or technical term outside it — an
#: unacceptable failure mode for a translation engine, so the full block stays.
SUBSET_UNICODES = ",".join(
    [
        "U+0020-007E",  # ASCII
        "U+00A0-00FF",  # Latin-1 supplement (accents, currency)
        "U+2000-206F",  # general punctuation (quotes, dashes, ellipsis)
        "U+20A0-20CF",  # currency symbols
        "U+2100-214F",  # letterlike symbols (№, ™)
        "U+2190-21FF",  # arrows (common in technical prose)
        "U+2200-22FF",  # mathematical operators
        "U+3000-303F",  # CJK punctuation
        "U+FE30-FE4F",  # CJK compatibility forms
        "U+FF00-FFEF",  # fullwidth forms
        "U+4E00-9FFF",  # CJK unified ideographs
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


def _subset(source: Path, dest: Path, *, font_number: int = 0) -> None:
    """Cut ``source`` down to the delivery charset."""
    if shutil.which("pyftsubset") is None:
        raise SystemExit(
            "pyftsubset is required to build the font subset.\n"
            "  uv pip install fonttools   (or: pip install fonttools)"
        )
    cmd = [
        "pyftsubset",
        str(source),
        f"--unicodes={SUBSET_UNICODES}",
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
        if filename.startswith("NotoSerifCJK"):
            archive = work / "noto-serif.zip"
            _download(NOTO_CJK_URL, archive)
            shutil.unpack_archive(str(archive), str(work))
            ttc = (
                next(work.rglob("*.otf"))
                if list(work.rglob("*.otf"))
                else next(work.rglob("*.ttc"))
            )
            _subset(ttc, dest)
        elif filename.startswith("NotoSansCJK"):
            archive = work / "noto-sans.zip"
            _download(NOTO_SANS_URL, archive)
            shutil.unpack_archive(str(archive), str(work))
            otf = next(work.rglob("*.otf"))
            _subset(otf, dest)
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
