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
        sha256="f4d53db237e74cb0be9bc0d8e3155ec50dc7b9e14b0769e0170bc8379a586c1d",
        license="OFL-1.1",
        scripts=("hans", "cjk"),
    ),
    BundledFont(
        filename="NotoSansCJKsc-Regular-subset.otf",
        family="Noto Sans CJK SC",
        sha256="5ec607bb46f0d557e07c1aa6cf3e0c6619d656536cadbfbaa9e40651878630c9",
        license="OFL-1.1",
        scripts=("hans", "cjk"),
    ),
    BundledFont(
        filename="NotoSerif-Regular.ttf",
        family="Noto Serif",
        sha256="19e72cd8d595fae5bd74a5206f5d938512e1183d4fed7abb1ec1be1d7efa5f88",
        license="OFL-1.1",
        scripts=("latin", "latin-ext"),
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
    "describe",
    "digest",
    "font_dir",
]
