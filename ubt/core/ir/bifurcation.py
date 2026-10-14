"""The semantic-break marker: a delimiter for a layout collision the model spotted.

When an upstream layout parser/OCR engine erroneously glues distinct passages
(such as cross-column prose, an intervening running header, or an accidental
paragraph merge), the translation model detects the layout collision and inserts
:data:`SEMANTIC_BREAK_TOKEN` (``⟦SEMANTIC_BREAK⟧``) at the boundary in its
translation.

This module owns the marker's two constants and nothing else:

* :data:`SEMANTIC_BREAK_TOKEN` -- the delimiter a prompt asks the model to emit.
  :mod:`ubt.render.typst.artifact_guard` recognises it, so a token whose restore
  pass could not account for it is escaped as prose rather than mangled by the
  math transpiler (a leak that used to fail the whole compile).
* :data:`SEMANTIC_BREAK_FLAG` -- the ``error_flags`` marker the continuation
  heuristic consults, so it can never re-fuse blocks a break deliberately
  separated.

The IRBlock *splitting* helpers this module used to carry (``bifurcate_block`` /
``bifurcate_blocks``) had no caller once the overlay PDF engine was deleted: the
publication (Typst) renderer reflows from the IR and never re-materialized
bifurcated siblings. They were removed rather than kept as a second, untested
split path.
"""

from __future__ import annotations

SEMANTIC_BREAK_TOKEN: str = "⟦SEMANTIC_BREAK⟧"
#: ``error_flags`` marker stamped on a block a semantic break split out. The
#: continuation heuristic consults it so it can never re-fuse what the break
#: deliberately separated.
SEMANTIC_BREAK_FLAG: str = "bifurcated:semantic_break"

__all__ = ["SEMANTIC_BREAK_FLAG", "SEMANTIC_BREAK_TOKEN"]
