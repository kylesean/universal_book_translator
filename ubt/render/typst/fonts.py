"""Fonts the deliverable is typeset with: bundled, fetched, or honestly absent.

**Why this module exists.** Until now the renderer typeset every book with
whatever families the *machine* happened to have. The same input produced a
good PDF on a workstation with Noto CJK installed and a page of tofu on an
Alpine container, a clean Windows Server, or a Mac without the Noto release —
a product-level nondeterminism for an engine whose whole claim is reproducible
delivery. Typst supports the fix directly (``--font-path`` and
``--ignore-system-fonts``); what was missing was an owned font set to point at.

**Why the fonts are not committed.** A subset covering the full CJK unified
ideograph block with vertical-layout features measures 6.5–15 MB *per*
language (`Noto Serif CJK`, measured with `pyftsubset`). Four scripts plus the
Latin faces land between 26 and 60 MB in a repository that is 76 MB total, and
every clone pays for fonts most checkouts never render. The precedent in this
repository is `corpus/semantic_gold`: fetch it with a script, keep it out of
git. That is what :mod:`scripts.fetch_fonts` does, and the pinned digests in
:data:`BUNDLED_FONTS` make the result reproducible.

**The honesty rule.** A missing font set is *not* an error the pipeline can
fix by itself, and it is not silently equivalent to a good one. :func:`font_dir`
returns ``None`` when nothing is present, callers then fall back to the system
probe (previous behaviour), and the run reports ``bundled=False`` in the
quality report. What must never happen is claiming a hermetic font set while
rendering with ambient ones.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

#: Where fetched fonts land. Under the repo so ``uv run`` and an editable
#: install both find it, and ignored by git (see ``.gitignore``).
#: ``fonts.py`` is ``ubt/render/typst/fonts.py``: three parents reach the root.
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FONT_DIR = REPO_ROOT / "fonts"

#: Environment override, for an operator who ships fonts with an image.
FONT_DIR_ENV = "UBT_FONT_DIR"


@dataclass(frozen=True, slots=True)
class BundledFont:
    """One redistributable face the renderer can rely on.

    Every entry is SIL OFL 1.1 or a permissive license that allows
    redistribution with the license text alongside; ``license`` names the file
    the fetch script writes next to the family. Nothing here is shareware,
    "free for personal use", or a system-vendor font — those cannot be
    redistributed, and a delivery engine must not depend on them.
    """

    #: File name inside the font directory.
    filename: str
    #: Family name exactly as Typst reports it (``typst fonts``), not as a
    #: human writes it: a mismatch re-creates "unknown font family".
    family: str
    #: SHA-256 of the file, so a fetch cannot silently change what a build
    #: renders (same reasoning as the checksum-pinned Typst in CI).
    sha256: str
    #: Redistribution license identifier (all OFL 1.1 today).
    license: str
    #: Scripts this face serves, used by the `covers` routing in
    #: :mod:`ubt.render.typst.font_routing`.
    scripts: tuple[str, ...]


#: The set an offline delivery needs. Kept small on purpose: each CJK face is
#: a subset limited to the unified ideographs plus punctuation, so the total is
#: tens of megabytes rather than the 25 MB-per-file upstream ``.ttc``.
#:
#: **Digests are placeholders until the first fetch records them**; the fetch
#: script refuses to write a file whose digest does not match, and a mismatch
#: is how a corrupted or substituted download is caught. Regenerate with
#: ``python scripts/fetch_fonts.py --update-digests``.
BUNDLED_FONTS: tuple[BundledFont, ...] = (
    BundledFont(
        filename="NotoSerifCJKsc-Regular-subset.otf",
        family="Noto Serif CJK SC",
        sha256="887cc02eced4fe97f62ef9e0e4bcc201736761f56c9cae6802b4a9767c2b70a9",
        license="OFL-1.1",
        # `hant` too: the traditional ideographs live in the same unified block
        # the subset covers (measured against a traditional-only sample).
        # Every CJK character is here (kana 96.9%, hangul 100%, ideographs
        # 100%), but the *regional forms* belong to the JP/KR cuts below: an
        # SC 直 does not look like a Japanese 直, so those scripts are not
        # claimed here even though the glyphs exist.
        scripts=("hans", "hant", "cjk"),
    ),
    BundledFont(
        filename="NotoSansCJKsc-Regular-subset.otf",
        family="Noto Sans CJK SC",
        sha256="e1b256cfecfd8633146f0569045b43bcedf5a607f5dd07310d987df6633faeab",
        license="OFL-1.1",
        scripts=("hans", "hant", "cjk"),
    ),
    BundledFont(
        filename="NotoSerif-Regular.ttf",
        family="Noto Serif",
        sha256="19e72cd8d595fae5bd74a5206f5d938512e1183d4fed7abb1ec1be1d7efa5f88",
        license="OFL-1.1",
        # Measured on the fetched file, not assumed: 100% of assigned Latin,
        # Latin-1, Cyrillic (incl. ext-B, so Ukrainian/Bulgarian/Serbian
        # letters are present) and Greek-extended; the Greek block's only gaps
        # are the 14 Coptic letters that Unicode files inside it. So the
        # Cyrillic and Greek targets need no separate face — they were simply
        # not declared as covered, which kept them out of the reproducible
        # path for no reason.
        scripts=("latin", "latin-ext", "cyrillic", "greek"),
    ),
    # Regional CJK cuts. The SC face carries every CJK script's *characters*,
    # but shared ideographs are drawn differently per region (直, 骨, 角 …), so
    # typesetting Japanese with the SC face reads as Chinese. Each regional
    # face is cut from the same upstream collection (same glyphs, different
    # regional forms) and owns its own script token.
    BundledFont(
        filename="NotoSerifCJKjp-Regular-subset.otf",
        family="Noto Serif CJK JP",
        sha256="11fe500a9cdc0320819426720e339e60b68d5a9ecb013a9836310d7c6b34db78",
        license="OFL-1.1",
        scripts=("jpan",),
    ),
    BundledFont(
        filename="NotoSerifCJKkr-Regular-subset.otf",
        family="Noto Serif CJK KR",
        sha256="beba6811a5b588b1e40c11a8e64f6a83e1f8b7c4c798755067fed2e9b6c5fc29",
        license="OFL-1.1",
        scripts=("kore",),
    ),
    # Arabic and Hebrew are absent from the pan-CJK faces (measured: 0% of
    # both blocks), so they need their own files. Naskh is the conventional
    # book face for Arabic prose; Hebrew uses the serif release for the same
    # reason. Both are small (a few hundred KB) — unlike CJK, these scripts
    # have no ideograph block to carry.
    BundledFont(
        filename="NotoNaskhArabic-Regular.ttf",
        family="Noto Naskh Arabic",
        sha256="c34fdbd98af4dbc45ca192a23d2eeb77032add83086f5fe32957d23b3f36b221",
        license="OFL-1.1",
        scripts=("arabic",),
    ),
    BundledFont(
        filename="NotoSerifHebrew-Regular.ttf",
        family="Noto Serif Hebrew",
        sha256="dfd5a6aefe97a99f68fe43388342913d50bb9fbf6d3afc4d2c7725661bc4a2b1",
        license="OFL-1.1",
        scripts=("hebrew",),
    ),
    # Thai and Devanagari. Neither block is in the Latin face (measured 0%), and
    # neither can be served by a CJK face, so each needs its own file. Both are
    # small because neither script has an ideograph block to carry — 88 KB and
    # 312 KB, versus 9 MB per CJK face. Taken from the ``full`` variant of the
    # upstream release: the ``hinted``/``unhinted`` builds are subset-per-script
    # and carry 2/95 and 39/95 ASCII respectively, which would typeset embedded
    # Latin and digits as tofu the moment isolation is on.
    BundledFont(
        filename="NotoSerifThai-Regular.ttf",
        family="Noto Serif Thai",
        sha256="428afb46af2c025ed2b9fe39bda2fffce9475fa5d2e7ae7911771633014b91b0",
        license="OFL-1.1",
        scripts=("thai",),
    ),
    BundledFont(
        filename="NotoSerifDevanagari-Regular.ttf",
        family="Noto Serif Devanagari",
        sha256="e3aafe3baf7d8601b2c895fd10ee74ad6b108dad0f7617c3211f6fc7d7d2aada",
        license="OFL-1.1",
        scripts=("devanagari",),
    ),
)


def font_dir() -> Path:
    """The font directory in use: ``UBT_FONT_DIR`` or the repo default.

    Returned even when it does not exist, so callers can report *where* they
    looked. Use :func:`bundled_font_dir` when the question is "is it usable?".
    """
    override = os.environ.get(FONT_DIR_ENV)
    return Path(override).expanduser() if override else DEFAULT_FONT_DIR


def digest(path: Path) -> str:
    """SHA-256 of a file, streamed so a 15 MB font does not sit in memory."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def bundled_font_dir(*, verify_digests: bool = False) -> Path | None:
    """The font directory when it holds a usable set, else ``None``.

    "Usable" means the directory exists and every declared face is present. A
    *partial* set is treated as absent: rendering a Chinese book with only the
    Latin faces present is exactly the silent degradation this module exists
    to prevent, and a caller that gets a path back is entitled to assume the
    hermetic guarantees hold.

    ``verify_digests`` re-hashes the files. That costs a few hundred
    milliseconds and is what the doctor and CI use; the render path trusts the
    directory after the one-time check below.
    """
    directory = font_dir()
    if not directory.is_dir():
        return None
    for face in BUNDLED_FONTS:
        path = directory / face.filename
        if not path.is_file() or path.stat().st_size == 0:
            logger.debug("font set incomplete: %s is missing", path)
            return None
    if verify_digests:
        for face in BUNDLED_FONTS:
            if not face.sha256:
                # A placeholder digest cannot verify anything. Treat the set as
                # unverified rather than pretending it matched.
                logger.debug("no pinned digest for %s; skipping verification", face.filename)
                continue
            actual = digest(directory / face.filename)
            if actual != face.sha256:
                raise FontDigestMismatch(face.filename, face.sha256, actual)
    return directory


#: Script coverage the bundled set actually provides. Derived from
#: ``BUNDLED_FONTS[*].scripts`` so the declaration lives in one place.
def bundled_script_coverage() -> frozenset[str]:
    """Scripts this font set can typeset without any system font."""
    covered: set[str] = set()
    for face in BUNDLED_FONTS:
        covered.update(face.scripts)
    return frozenset(covered)


#: Target-language script -> the coverage token that must be present before
#: system fonts may be ignored. A language's script is *not* implied by another
#: language's: the SC subset measured 0% kana and 0% hangul coverage, so
#: isolating a Japanese book against it renders an entire page of tofu. Only a
#: language whose script the set actually carries gets the hermetic treatment.
_LANG_SCRIPT_TOKEN: dict[str, str] = {
    "zh": "hans",
    "zh-cn": "hans",
    "zh-hans": "hans",
    "zh-sg": "hans",
    # Traditional Chinese shares the ideograph block the SC subset carries, so
    # the SC face renders it (measured: no missing glyphs in a sample of
    # traditional-only characters). It is still its own token because a future
    # TC-specific face would declare it.
    "zh-tw": "hant",
    "zh-hant": "hant",
    "zh-hk": "hant",
    "zh-mo": "hant",
    "ja": "jpan",
    "ko": "kore",
    "ar": "arabic",
    "he": "hebrew",
    "en": "latin",
    "fr": "latin",
    "de": "latin",
    "es": "latin",
    "it": "latin",
    "pt": "latin",
    "nl": "latin",
    "vi": "latin-ext",
    "pl": "latin-ext",
    "cs": "latin-ext",
    "tr": "latin-ext",
    # Cyrillic and Greek need no face of their own: the bundled Noto Serif
    # carries 100% of assigned Cyrillic (through ext-B) and Greek. They were
    # absent from this table, which kept them out of the reproducible path for
    # no reason other than an unmeasured assumption.
    "ru": "cyrillic",
    "uk": "cyrillic",
    "bg": "cyrillic",
    "sr": "cyrillic",
    "mk": "cyrillic",
    "be": "cyrillic",
    "kk": "cyrillic",
    "el": "greek",
    "th": "thai",
    # Devanagari only. Bengali, Gurmukhi, Gujarati and Myanmar are *different
    # scripts* with their own blocks and cannot be set from a Devanagari face;
    # Lao is not Thai. Claiming otherwise here would isolate a Bengali book
    # against a Devanagari face and ship a page of tofu, which is exactly the
    # failure isolation must never introduce.
    "hi": "devanagari",
    "mr": "devanagari",
    "ne": "devanagari",
}


def _lang_token(target_lang: str | None) -> str:
    tag = (target_lang or "zh").strip().lower().replace("_", "-")
    return _LANG_SCRIPT_TOKEN.get(tag) or _LANG_SCRIPT_TOKEN.get(tag.split("-")[0], "")


def can_isolate(target_lang: str | None, *, verify_digests: bool = False) -> bool:
    """Whether system fonts may be ignored for this target without losing glyphs.

    Both conditions must hold:

    1. the bundled set is installed and complete, and
    2. it declares coverage for this target's script.

    ``False`` is not a failure — the caller compiles with system fonts
    available, which is the previous behaviour, and reports ``bundled: false``
    because the output is no longer host-independent.

    **Why the second condition carries the weight.** ``--ignore-system-fonts``
    is the only flag that makes the render reproducible, but it is absolute:
    Typst then sees nothing outside ``--font-path``. The SC subset measures 0%
    kana and 0% hangul, so applying it to a Japanese or Korean book typesets
    every page as tofu — strictly worse than a non-reproducible good page.
    """
    directory = bundled_font_dir(verify_digests=verify_digests)
    if directory is None:
        return False
    token = _lang_token(target_lang)
    return bool(token) and token in bundled_script_coverage()


class FontDigestMismatch(RuntimeError):
    """A font file is not the one this build expects."""

    def __init__(self, filename: str, expected: str, actual: str) -> None:
        self.filename = filename
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"{filename}: expected sha256 {expected}, got {actual}. Re-fetch with "
            "`python scripts/fetch_fonts.py --force`, and investigate if it persists."
        )


def describe() -> dict[str, object]:
    """A report-friendly summary of the font set's state.

    ``bundled`` is the fact that matters: True means the deliverable was
    typeset against a fixed, redistributable set; False means it used ambient
    system fonts and carries the portability caveat that implies.
    """
    directory = font_dir()
    present = bundled_font_dir()
    return {
        "bundled": present is not None,
        "directory": str(directory),
        "expected_families": [face.family for face in BUNDLED_FONTS],
        "missing": [
            face.filename for face in BUNDLED_FONTS if not (directory / face.filename).is_file()
        ],
    }


__all__ = [
    "BUNDLED_FONTS",
    "BundledFont",
    "DEFAULT_FONT_DIR",
    "FONT_DIR_ENV",
    "FontDigestMismatch",
    "bundled_font_dir",
    "bundled_script_coverage",
    "can_isolate",
    "describe",
    "digest",
    "font_dir",
]
