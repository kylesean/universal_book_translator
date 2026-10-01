"""Ask the renderer which font families it can actually resolve.

The language profile names a font stack per target language
(``language_profile.resolve_font_config``), but the names are static while the
machine is not: Typst enumerates fonts itself, and a family that is simply not
installed is reported per compile as ``unknown font family``. Worse, when *no*
CJK-capable family is installed the CJK stack has nothing left to fall back to
and the run produces tofu without failing.

This module queries the renderer for the families it can see and prunes the
requested stack down to those, turning "a Chinese book rendered as blank
squares" into an explicit verdict a caller can act on.

Two probes, in order of trustworthiness:

1. ``typst fonts`` — the compiler's own view, so it matches what a compile will
   resolve. Keep the invocation aligned with ``typst_compile.typst_compile``
   (neither passes ``--font-path`` / ``--ignore-system-fonts``); if the compile
   ever starts passing a font flag, this probe has to gain it too or the two
   will disagree.
2. ``fc-list`` — used only when Typst cannot answer, so the caller still gets a
   stack worth emitting on a machine without the compiler.

When neither answers, ``probed`` is False and callers must change nothing: an
unavailable probe is not evidence that a font is missing.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ubt.core.env import subprocess_env

logger = logging.getLogger(__name__)

# Families that can carry Chinese/Japanese/Korean glyphs, matched on the
# normalized family name.
#
# Precision matters more than recall here: a false "available" suppresses the
# tofu warning and ships blank squares, while a false "missing" only emits an
# extra WARN that never fails a run. So no broad stems ("hei", "song",
# "notosans") that a Latin font name can also hit -- Noto Sans alone is a Latin
# family; only the "-CJK" Noto releases carry these scripts.
#
# Vendor internal names must be listed explicitly: "MSYGOTH" is how MS Gothic
# reports itself and matches no readable spelling.
_CJK_NAME_MARKERS: tuple[str, ...] = (
    "cjk",  # Noto Serif/Sans CJK *, the common Linux release
    "sourcehanserif",
    "sourcehansans",
    "simhei",
    "simsun",
    "simfang",
    "simkai",
    "heiti",
    "songti",
    "kaiti",
    "fangsong",
    "yahei",
    "msyh",
    "msygoth",
    "msgothic",
    "msmincho",
    "meiryo",
    # Bare "gothic"/"mincho" are NOT usable stems: they are style words that Latin
    # faces carry too. "Franklin Gothic" and "Century Gothic" are Windows system
    # fonts with no CJK glyph at all, and they sort alphabetically *before*
    # "Microsoft YaHei" — so as a stem they won the substitution race, shipped a
    # Chinese book as blank squares, and reported cjk_available=True while doing
    # it. The real Gothic/Mincho families are named explicitly below and via
    # "msgothic"/"msmincho"/"msyh".
    "yugothic",
    "yumincho",
    "pgothic",
    "pmincho",
    "uigothic",
    "malgun",
    "gulim",
    "batang",
    "nanum",
    "wqy",
    "wenquanyi",
    "arphic",
    "uming",
    "droidsansfallback",
    "pingfang",
    "hiragino",
    "kochi",
    "heisei",
    "unifont",
)

# Target languages whose scripts need glyphs a Latin-only stack cannot supply.
_CJK_SCRIPT_LANGS: frozenset[str] = frozenset(
    {
        "zh",
        "zh-cn",
        "zh-hans",
        "zh-hk",
        "zh-mo",
        "zh-sg",
        "zh-tw",
        "zh-hant",
        "ja",
        "ko",
    }
)

# Right-to-left targets (ADR-0001 §12 Q4): a Latin-only stack cannot shape
# Arabic/Hebrew either, so they get the same "substitute rather than ship tofu"
# treatment as CJK. Markers are script-specific strings a Latin family does not
# carry; the DejaVu tail cannot be trusted for Arabic shaping, so it is not a
# marker.
_RTL_NAME_MARKERS: tuple[str, ...] = (
    "arabic",
    "naskh",
    "kufi",
    "amiri",
    "scheherazade",
    "kacst",
    "hebrew",
    "rashi",
)
_RTL_SCRIPT_LANGS: frozenset[str] = frozenset(
    {"ar", "he", "fa", "ur", "ps", "sd", "ug", "yi", "dv"}
)

_PROBE_TIMEOUT_S = 20.0


def _norm(name: str) -> str:
    """Normalize a family name for comparison across vendor spellings."""
    return re.sub(r"[\s_-]+", "", name).casefold()


@lru_cache(maxsize=8)
def _typst_families_cached(resolved: str, mtime_ns: int) -> frozenset[str] | None:
    """Families the Typst compiler itself can see, or None when it cannot answer.

    ``mtime_ns`` is part of the cache key (see ``_typst_families``); the probe
    re-runs after a Typst upgrade instead of pruning the font stack against a
    stale family set.
    """
    try:
        proc = subprocess.run(
            [resolved, "fonts"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_PROBE_TIMEOUT_S,
            # mirror typst_compile's environment policy exactly — the probe
            # is only trustworthy if it sees the same environment as a compile.
            env=subprocess_env(),
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("typst font probe could not run: %s", exc)
        return None
    if proc.returncode != 0:
        logger.debug("typst fonts exited %s", proc.returncode)
        return None
    names = frozenset(line.strip() for line in proc.stdout.splitlines() if line.strip())
    return names or None


class _TypstFamilyProbe:
    """Callable wrapper keeping the public ``_typst_families(name)`` surface.

    A plain function cannot both accept the config's ``typst_binary`` and key the
    cache on the resolved binary's mtime, and ``lru_cache`` on a method triggers
    B019. The cache lives in the module-level :func:`_typst_families_cached`;
    this wrapper resolves the binary and exposes ``cache_clear`` for the test
    fixture.
    """

    def __call__(self, typst_binary: str) -> frozenset[str] | None:
        resolved = shutil.which(typst_binary)
        if not resolved and Path(typst_binary).is_file():
            resolved = typst_binary
        if not resolved:
            return None
        try:
            mtime_ns = Path(resolved).stat().st_mtime_ns
        except OSError:
            mtime_ns = 0
        return _typst_families_cached(resolved, int(mtime_ns))

    def cache_clear(self) -> None:
        _typst_families_cached.cache_clear()


_typst_families = _TypstFamilyProbe()


@lru_cache(maxsize=1)
def _fontconfig_families() -> frozenset[str] | None:
    """Families fontconfig knows about, or None when fontconfig is absent."""
    listing = shutil.which("fc-list")
    if not listing:
        return None
    try:
        proc = subprocess.run(
            [listing, ":", "family"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_PROBE_TIMEOUT_S,
            env=subprocess_env(),
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("fc-list could not run: %s", exc)
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    # Each line is a font file's comma-separated family list; a family can carry
    # style variants, so split on both commas and take the base names.
    found: set[str] = set()
    for line in proc.stdout.splitlines():
        _, _, families = line.partition(":")
        for name in families.split(","):
            name = name.strip()
            if name:
                found.add(name)
    return frozenset(found) or None


def available_font_families(typst_binary: str = "typst") -> frozenset[str] | None:
    """Font families the renderer can resolve, or None when no probe answered."""
    return _typst_families(typst_binary) or _fontconfig_families()


def is_cjk_capable(family: str) -> bool:
    """Whether a family name suggests glyphs for CJK scripts."""
    normalized = _norm(family)
    return any(marker in normalized for marker in _CJK_NAME_MARKERS)


def needs_cjk(target_lang: str | None) -> bool:
    """Whether a target language cannot be set from a Latin-only stack."""
    if not target_lang:
        return False
    lang = target_lang.strip().lower().replace("_", "-")
    return lang in _CJK_SCRIPT_LANGS or lang.split("-")[0] in {"zh", "ja", "ko"}


def is_rtl_capable(family: str) -> bool:
    """Whether a family name suggests glyphs for Arabic/Hebrew scripts."""
    normalized = _norm(family)
    return any(marker in normalized for marker in _RTL_NAME_MARKERS)


def needs_rtl(target_lang: str | None) -> bool:
    """Whether a target language is right-to-left (cannot use a Latin-only stack)."""
    if not target_lang:
        return False
    lang = target_lang.strip().lower().replace("_", "-")
    return lang in _RTL_SCRIPT_LANGS or lang.split("-")[0] in _RTL_SCRIPT_LANGS


def script_need(target_lang: str | None) -> str:
    """The non-Latin script a target needs: ``"cjk"``, ``"rtl"``, or ``""``."""
    if needs_cjk(target_lang):
        return "cjk"
    if needs_rtl(target_lang):
        return "rtl"
    return ""


def _is_script_capable(family: str, need: str) -> bool:
    if need == "cjk":
        return is_cjk_capable(family)
    if need == "rtl":
        return is_rtl_capable(family)
    return False


@dataclass(frozen=True, slots=True)
class FontStack:
    """A requested font stack, resolved against what the machine has."""

    families: tuple[str, ...]
    # Requested but not resolvable by the renderer; dropped from the emitted stack.
    unavailable: tuple[str, ...] = ()
    # False when no probe answered, so the stack was passed through untouched.
    probed: bool = True
    # Whether the emitted stack can render the target script at all (CJK or RTL).
    script_available: bool = True
    # Families added because nothing requested could render a non-Latin target.
    substituted: tuple[str, ...] = ()

    def as_typst_tuple(self) -> str:
        """Render as a Typst font list: ``("A", "B")``."""
        return "(" + ", ".join(f'"{f}"' for f in self.families) + ")"

    @property
    def primary(self) -> str:
        return self.families[0] if self.families else ""


def _script_of(target_lang: str | None) -> str:
    """Base code ('zh'/'ja'/'ko'/'ar'/'he') for a non-Latin target, else ''."""
    if not script_need(target_lang):
        return ""
    return (target_lang or "").strip().lower().replace("_", "-").split("-")[0]


# Families that carry one script's characters and not another's. Substituting a
# Korean face into a Chinese book produces the same blank squares this whole
# module exists to avoid, so a foreign-script name is demoted rather than merely
# sorted. Region suffixes (…CJK SC / …JP / …KR) are tested positionally: as a
# bare substring "sc" hits half the Latin catalogue.
_SCRIPT_EXCLUSIVE: dict[str, tuple[str, ...]] = {
    "zh": (
        "yahei",
        "msyh",
        "simhei",
        "simsun",
        "simfang",
        "simkai",
        "heiti",
        "songti",
        "kaiti",
        "fangsong",
        "pingfang",
        "wqy",
        "wenquanyi",
        "arphic",
        "uming",
    ),
    "ja": (
        "hiragino",
        "kochi",
        "heisei",
        "meiryo",
        "yugothic",
        "yumincho",
        "msgothic",
        "msmincho",
        "pgothic",
        "pmincho",
        "uigothic",
    ),
    "ko": ("malgun", "gulim", "batang", "nanum"),
    "ar": ("arabic", "naskh", "kufi", "amiri", "scheherazade", "kacst"),
    "he": ("hebrew", "rashi"),
}
_SCRIPT_SUFFIXES: dict[str, tuple[str, ...]] = {
    "zh": ("sc", "hans", "cn"),
    "ja": ("jp",),
    "ko": ("kr",),
}
_PAN_CJK = ("cjk", "sourcehan")


def _substitution_rank(family: str, target_lang: str | None) -> tuple[int, str]:
    """Sort key: our script first, then pan-CJK coverage, then anything, never
    another script's face if a usable one is installed."""
    normalized = _norm(family)
    ours = _script_of(target_lang)
    tagged_ours = normalized.endswith(_SCRIPT_SUFFIXES.get(ours, ()))
    named_ours = any(m in normalized for m in _SCRIPT_EXCLUSIVE.get(ours, ()))
    named_theirs = any(
        m in normalized
        for script, markers in _SCRIPT_EXCLUSIVE.items()
        if script != ours
        for m in markers
    )
    tagged_theirs = any(
        normalized.endswith(sfx)
        for script, sfxs in _SCRIPT_SUFFIXES.items()
        if script != ours
        for sfx in sfxs
    )
    pan = any(m in normalized for m in _PAN_CJK)
    if tagged_ours or named_ours:
        rank = 0
    elif pan and not tagged_theirs:
        rank = 1
    elif named_theirs or tagged_theirs:
        rank = 3
    else:
        rank = 2
    return rank, normalized


def _substitute_for(
    available: frozenset[str], kept: list[str], target_lang: str | None
) -> str | None:
    """Best same-script family on this machine that is not already in ``kept``."""
    need = script_need(target_lang)
    if not need:
        return None
    already = {_norm(k) for k in kept}
    candidates = [f for f in available if _is_script_capable(f, need) and _norm(f) not in already]
    if not candidates:
        return None
    return min(candidates, key=lambda f: _substitution_rank(f, target_lang))


def resolve_font_stack(
    configured: list[str] | tuple[str, ...],
    *,
    target_lang: str | None = None,
    typst_binary: str = "typst",
    available: frozenset[str] | None = None,
    protected: tuple[str, ...] = (),
) -> FontStack:
    """Prune ``configured`` to families the renderer can resolve.

    ``protected`` names are never dropped even when unresolvable — they carry a
    user's explicit ``font_family`` choice, and quietly replacing that is worse
    than letting Typst fall back. They still appear in ``unavailable`` so the
    caller can say the name is not installed.

    Never returns an empty stack. When a non-Latin target's stack would leave no
    family for its script (CJK or RTL), one is substituted from what the renderer
    reports and named in ``substituted``: readable text in a substitute typeface
    beats blank squares. If nothing requested resolves and there is no substitute,
    the requested stack is emitted unchanged so Typst's own fallback applies, with
    every name reported in ``unavailable`` and ``script_available`` False.
    """
    need = script_need(target_lang)
    requested = [f for f in configured if f and f.strip()]
    if not requested:
        return FontStack(families=(), script_available=not need)

    if available is None:
        available = available_font_families(typst_binary)

    if available is None:
        return FontStack(
            families=tuple(requested),
            probed=False,
            script_available=any(_is_script_capable(f, need) for f in requested) or not need,
        )

    present = {_norm(f) for f in available}
    protected_norm = {_norm(f) for f in protected}
    # Order preserved; a protected name stays even when unresolvable, and stays
    # in `dropped` so the caller still reports it as not installed.
    kept = [f for f in requested if _norm(f) in present or _norm(f) in protected_norm]
    dropped = tuple(f for f in requested if _norm(f) not in present)

    def _has_script(names: list[str]) -> bool:
        return any(_norm(f) in present and _is_script_capable(f, need) for f in names)

    substituted: tuple[str, ...] = ()
    if need and not _has_script(kept):
        # The profile's stack cannot render this script, but the machine may
        # still have a same-script family under a name the profile never heard
        # of. Appending it turns blank squares into readable text in a
        # substitute typeface; shipping tofu is the worse failure.
        spare = _substitute_for(available, kept, target_lang)
        if spare:
            kept.append(spare)
            substituted = (spare,)

    if not kept:
        # Nothing requested exists and there is no substitute. Emit the
        # requested stack anyway so Typst's own fallback applies, rather than us
        # guessing a replacement -- but report every name as unavailable and
        # claim no script coverage, because we verified none of them resolve.
        kept = list(requested)

    return FontStack(
        families=tuple(kept),
        unavailable=dropped,
        probed=True,
        script_available=_has_script(kept) or not need,
        substituted=substituted,
    )


__all__ = [
    "FontStack",
    "available_font_families",
    "is_cjk_capable",
    "is_rtl_capable",
    "needs_cjk",
    "needs_rtl",
    "resolve_font_stack",
    "script_need",
]
