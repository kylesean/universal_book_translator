"""Typst math-syntax probe shared by the PDF typesetting engines.

A span the model delimited as ``$...$`` is only emitted in math mode when a
minimal Typst document containing it compiles; anything else (``F_th,SI``
without braces, stray markup) falls back to escaped text at placement time.
``//`` is rejected without compiling (line-comment hazard); ``#`` is
legitimate here — :mod:`ubt.adapters.pdf.overlay_text` emits code-mode
strings (``#"th"``) and rejects user-supplied ``#`` upstream.

The probe is pure-cache: same body, same verdict, one compile ever.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_PROBE_DOC = "#set page(width: 60pt, height: 20pt, margin: 0pt)\n${body}$\n"


class TypstMathProbe:
    """Compile-gated verdict for one inline-math body (cached).

    ``binary`` must be the same Typst the surrounding document is compiled
    with: a probe pinned to PATH ``typst`` while the real compile honors
    ``UBT_TYPST_BINARY`` reports a false-negative verdict for every formula
    whenever typst lives off PATH, silently escaping all math to text yet
    still "succeeding" — and because the probe is pure-cache, that wrong
    verdict is then reused for the whole run.
    """

    def __init__(self, binary: str = "typst") -> None:
        self._cache: dict[str, bool] = {}
        self._binary = binary

    def check(self, body: str) -> bool:
        clean = body.strip()
        if clean.startswith("$") and clean.endswith("$") and len(clean) >= 2:
            clean = clean[1:-1].strip()
        if clean in self._cache:
            return self._cache[clean]
        ok = False
        if clean and "//" not in clean.replace(" ", ""):
            from ubt.adapters.pdf.typst_compile import typst_compile

            try:
                with tempfile.TemporaryDirectory() as tmp:
                    typ_path = f"{tmp}/probemath.typ"
                    Path(typ_path).write_text(
                        _PROBE_DOC.format(body=clean.replace("\n", " ")), encoding="utf-8"
                    )
                    ok, _ = typst_compile(typ_path, f"{tmp}/probemath.pdf", binary=self._binary)
            except Exception:  # probe must never break a render
                logger.debug("Typst math probe failed for %r", clean, exc_info=True)
                ok = False
        self._cache[clean] = ok
        return ok
