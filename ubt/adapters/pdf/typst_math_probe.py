"""Typst math-syntax probe shared by the PDF typesetting engines.

A span the model delimited as ``$...$`` is only emitted in math mode when a
minimal Typst document containing it compiles; anything else (``F_th,SI``
without braces, stray markup) falls back to escaped text at placement time.
``//`` is rejected without compiling (line-comment hazard); ``#`` is
legitimate here — :mod:`ubt.adapters.pdf.overlay_text` emits code-mode
strings (``#"th"``) and rejects user-supplied ``#`` upstream.

The probe is pure-cache: same body, same verdict, one compile ever. A whole
document's bodies can be resolved in a handful of compiles via
:meth:`TypstMathProbe.check_many`, which lays every not-yet-known body out as a
page of one document and bisects on failure — a Typst invocation costs ~0.7s of
startup, so one process per body dominated a math-dense render.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Iterable
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

    @staticmethod
    def _normalize(body: str) -> str:
        clean = (body or "").strip()
        if clean.startswith("$") and clean.endswith("$") and len(clean) >= 2:
            clean = clean[1:-1].strip()
        return clean

    def check(self, body: str) -> bool:
        clean = self._normalize(body)
        if clean in self._cache:
            return self._cache[clean]
        ok = False
        if clean and "//" not in clean.replace(" ", ""):
            ok = self._compile_source(_PROBE_DOC.format(body=clean.replace("\n", " ")))
        self._cache[clean] = ok
        return ok

    def check_many(self, bodies: Iterable[str]) -> None:
        """Resolve many bodies at once, populating the same cache ``check`` reads.

        Every not-yet-known body is laid out as a page of one document; if it
        compiles, all of them pass in a single Typst invocation. A failure is
        bisected, so a handful of bad bodies costs O(log n) extra compiles
        rather than one per body. Verdicts are identical to per-body ``check``:
        bodies never share state, so a subset compiles iff each member does.
        """
        pending = [
            clean
            for clean in dict.fromkeys(self._normalize(body) for body in bodies)
            if clean and clean not in self._cache and "//" not in clean.replace(" ", "")
        ]
        self._probe_batch(pending)

    def _probe_batch(self, bodies: list[str]) -> None:
        if not bodies:
            return
        source = "\n#pagebreak()\n".join(
            _PROBE_DOC.format(body=body.replace("\n", " ")) for body in bodies
        )
        if self._compile_source(source):
            for body in bodies:
                self._cache[body] = True
            return
        if len(bodies) == 1:
            self._cache[bodies[0]] = False
            return
        mid = len(bodies) // 2
        self._probe_batch(bodies[:mid])
        self._probe_batch(bodies[mid:])

    def _compile_source(self, source: str) -> bool:
        from ubt.adapters.pdf.typst_compile import typst_compile

        try:
            with tempfile.TemporaryDirectory() as tmp:
                typ_path = f"{tmp}/probemath.typ"
                Path(typ_path).write_text(source, encoding="utf-8")
                ok, _ = typst_compile(typ_path, f"{tmp}/probemath.pdf", binary=self._binary)
            return ok
        except Exception:  # probe must never break a render
            logger.debug("Typst math probe failed", exc_info=True)
            return False
