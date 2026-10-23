# Architecture

How UBT is put together, for an engineer joining the project. Every claim here
is checkable in the code it names; where a rule is enforced mechanically, the
enforcing tool is named too.

## The layer graph

`ubt/` is layered. The rule, stated once in the module docstring of
`ubt/core/ports.py`:

> `ubt.core` must never import a package that depends back on it at module
> scope.

Two families of reverse edge are forbidden: the adapters (`ubt.adapters` already
depends on core's IR models, so the reverse edge is a package-level cycle that
would drag docling and Typst tooling into the core import graph) and the
compiler packages (`ubt.segment` / `ubt.translate`, where a module-level edge
from core closes the `core.engine <-> pipeline` cycle).

The packages, from the leaf upward. This is not a strict total order — what
matters is the direction of the forbidden edges, stated per package below:

| Package | Role |
| --- | --- |
| `ubt/model` | The leaf document AST: `Element`/`ElementT`, `Span`/`CompositeSpan`/`CanonicalSource`, and the translation-unit types (`Segment`, `Placeholder`). Pure data: it imports nothing from the rest of `ubt`. |
| `ubt/core` | The domain: IR (`ubt/core/ir`), engine, policy, QE, validators, ledger, router, memory. **May not** import `ubt.adapters` at module level; implementations reach it through `ubt/core/ports.py` (function-level lazy imports). |
| `ubt/cache` | Content-addressed caches (`DiskCacheStore`, the Docling conversion cache's directory root) plus `ubt/cache/dirs.py`. Leaf-adjacent: it may import `ubt/core` leaves (`fs_perms`) and `ubt/core/engine/pipeline.py` imports `ubt/cache.dirs`, which makes `core` ↔ `cache` a package-level cycle. Tolerated because `fs_perms` is a leaf module with no reverse edge; do not deepen it. |
| `ubt/layout` | Theme/direction helpers. `direction_for` is read by the render and export code, and the Typst renderer also reads `Theme.leading_em` for body leading; the rest of `Theme` (`resolve_theme`, the bilingual/size/paper fields) is exercised by tests alone — do not treat this package as the single source of truth for font stacks or paper size until a renderer reads those. |
| `ubt/analyze` | Format readers turning sources into the typed AST (`read_md`, `read_html`, `read_epub`, `read_docx`), plus the shared walkers, numbering and classifier rules the adapters call. It builds on `ubt/core`'s leaf helpers (`ubt/core/xml_safety`, `zip_safety`, `ir`) and **may not** import `ubt.adapters`. |
| `ubt/adapters` | Format adapters (`ubt/adapters/factory.py`) plus the PDF engines and the render stack. |
| `ubt/segment`, `ubt/translate` | The translation-unit layer: placeholder masking/restore and the engine that drives it. |
| `ubt/render` | Lowering delivered blocks to HTML (`html_view.py`), EPUB (`epub_view.py`) and the Typst PDF document (`typst/document_renderer.py`). It **may not** import `ubt.adapters`. |
| `ubt/pipeline` | Artifact identity (`artifact.py`): the two paths the export stage keys on. |
| `ubt/cli`, `ubt/api`, `ubt/mcp` | The three entry-point surfaces, which may import anything. |

`ubt/core/ports.py` is the single sanctioned bridge back to the higher layers.
Implementations resolve through function-level lazy imports, which are not
module-level edges; `TYPE_CHECKING` imports are likewise excluded. Keep the file
to what is actually resolved through it — unused wrappers there read as endorsed
capability.

### The invariant is enforced, not just documented

- `scripts/dump_compiler_edges.py` is the acceptance tool. It exits non-zero if
  any module-level `ubt.core` import of `ubt.pipeline` / `ubt.segment` /
  `ubt.translate` exists (the inward edge to the leaf `ubt.model` is exempt by
  design), if the `ubt.core.engine <-> ubt.pipeline` package cycle exists, if
  any `ubt.render -> ubt.adapters` module-level edge exists, or if **any**
  module-level `ubt.core -> ubt.adapters` edge exists. The last check was
  missing while this section already claimed the invariant was enforced: a core
  module could import docling or typst and every gate stayed green.
- `tests/unit/analyze/test_layering.py` pins the other direction: **no**
  `ubt.adapters` import anywhere under `ubt/analyze`. The allowed set is empty
  on purpose — any new edge there is an inversion, not a seam to negotiate.
  Shared helpers belong in a leaf module below both (e.g. `ubt/core/text_decode`).
- Both run as gates: a pre-push hook and a CI step
  (`.pre-commit-config.yaml`, `.github/workflows/ci.yml`).

## The pipeline

`ubt/core/engine/pipeline.py` owns the run's *resources* — the writer lock,
ledger, adapter, router, per-run collaborators. `ubt/core/engine/plan.py` owns
*what runs in what order*. `run_stages()` is a function, not a table of
callables: the chapter-streaming branch and the export terminal hook are real
control flow.

The stage order, from `run_stages()`:

1. **ingest** (`stages/ingest.py`) — manifest registration, streaming chapter
   preprocessing, skip classification, policy verdicts, block rows into the
   ledger.
2. **extraction witness** (`stages/advisory.py`) — flags pages whose math the
   extractor already broke, before anything is translated.
3. **mode advisory** (`stages/advisory.py`) — scores the requested render mode
   against the layout the document actually has.
4. **render preflight** (`stages/preflight.py`) — compiles a source sample
   through the real PDF renderer.
5. **cost preflight** (`stages/preflight.py`) — prices the cheapest plausible
   draft pass and refuses a job already over `--budget-usd`.
6. **bible** (`stages/bible.py`) — mines the book's terms and names, seeds the
   glossary, backfills abbreviations.
7. **the draft ladder** — either `stages/draft.py` → `stages/ctext.py` (gated)
   → `stages/quality_gate.py` → `stages/repair.py`, or, when the
   `chapter_streaming` gate is on, `stages/chapter_streaming.py` replaces the
   whole ladder.
8. **consistency** (`stages/consistency.py`) — gated; detects document-level
   terminology drift and optionally repairs it.
9. **triage** (`stages/triage.py`) — MQM severity routing to the human queue.
10. **difficulty advisory** (`stages/advisory.py`) — the post-repair half.
11. **export** (`stages/export.py`) — integrity validation, render, reports,
    contract reconciliation.

Two details are load-bearing and pinned by `tests/unit/test_run_stages.py`: the
zero-token preflights run **before** the billable bible stage, and
`on_export_completed` runs on the terminal export event *before* it is yielded,
so a consumer that stops iterating cannot lose the terminal persistence to
`GeneratorExit`. TM writeback (`stages/tm_writeback.py`) is wired through that
hook from `pipeline.py`, not from the plan.

### The stage contract

A translating stage is an async generator yielding `TranslationProgressEvent`.
The zero-token judgements are plain coroutines that emit nothing: the extraction
witness, the two preflights, and the difficulty advisory (which only steps the
render mode down for the report). Because a generator has no return value, a
stage that *produces* a value for a later stage fills a plan-owned field instead
of returning it:

- `StageContext` (`ubt/core/engine/stage_context.py`) is the frozen run
  descriptor: the run's inputs (config, router, ledger, manifest, job id, paths,
  languages, profile), the driver callbacks that reach *into* a stage
  (`create_event`, `measure_run_usage`, `bill_run_usage`, `cancel_token`), and
  the decisions made before staging (short chain, chapter window, fast lane).
  `frozen=True` stops a stage from rebinding a field. The one sanctioned mutable
  channel is `ctx.manifest.run`.
- `RunFacts` (`ubt/core/engine/facts.py`) carries what one stage produces and a
  later one consumes (`terminology`, `layout`, `render`, `scoring`); the plan
  threads each to the stage that reads it.
- `RunServices` (`ubt/core/engine/services.py`) carries the collaborators the run
  *builds for itself* — QE runner, repair loop, FastPass filter, adaptive
  policy, in-flight semaphore, TM, HTML validator, and the run's
  `SourceDocumentIndex`.
- `BlockReader` (`ubt/core/engine/blocks.py`) owns the run's one piece of mutable
  state, the block snapshot, and re-reads it when the ledger's `blocks_seq`
  revision changes.
- `SourceDocumentIndex` (`ubt/core/engine/source_document.py`) holds the book's
  source text and its indexed vocabulary. `source_text` is immutable after
  ingest (no write path touches the column), so it is read **once per run** and
  shared by the draft stage and the quality gate instead of being rebuilt from a
  full-table scan per chapter; the omission gate consumes the vocabulary rather
  than re-scanning the book per block.

Stage-specific tunables stay in the stage's own signature, so they remain
greppable.

## The ledger is the single source of run state

`ubt/core/engine/ledger.py` assembles `SQLiteJobLedger` from `LedgerBase`
(connection, transactions, schema migrations) and three mixins (job lifecycle,
block writes, batch-API partitions). The database is SQLite in WAL mode under
`db_dir` (default `.ubt/ledgers`).

- `ledger_base.py` owns the flattened baseline schema and `SCHEMA_VERSION`
  (1000; `TARGET_SCHEMA_VERSION` is retained as an alias). Fresh databases are
  created directly from `_SCHEMA_SQL` and stamped with `1000`. The retired
  14-step incremental migration ladder was removed: an existing database from
  an older build without `--fresh` is refused with clear instructions rather than
  migrated in place, avoiding the risk of corrupting the run's only copy of
  translated blocks. When `--fresh` is specified, the old tables are dropped
  and the baseline is recreated cleanly. A startup check verifies
  `_REQUIRED_BLOCK_COLUMNS` are all present so a malformed database fails at
  open rather than inside `_row_to_block`.
- `ledger_mixins.py` owns the writes: `init_job_from_manifest`, `append_chapter`,
  `fetch_pending_blocks`, `save_checkpoint` / `save_checkpoints_batch`,
  `finalize_job`, the reset paths, and the batch-API partition.
- The primary key is `(job_id, block_id)` — block ids are content-derived
  (`pdf_main#b0001`) and repeat across jobs over the same document, so every
  block-id-keyed write is scoped by job.

`finalize_job` is fail-closed on `completed`: it refuses when the job row is
missing or when any non-terminal block remains. `failed` and `cancelled` stay
permissive by design, because the abort path must land a terminal write even
mid-flight — with two deliberate exceptions in each direction: a late `failed`
never overwrites a `cancelled` job, and a late `failed`/`cancelled` never
overwrites a `completed` one. A real completion *may* supersede an earlier
`cancelled` or `failed`, since by then the work demonstrably finished.

### The writer lock

There is no per-block worker lease in the ledger schema, so a **job-level** guard
is what stops two processes running the same `job_id` at once.
`ubt/core/engine/writer_lock.py` takes an exclusive, expiring lease stored in the
`job_locks` table inside the ledger SQLite database (`(job_id, owner_pid,
owner_token, lease_expires_at, acquired_at)`), replacing the obsolete
`.writer.lock` file-offset lock that failed on NFS, overlayfs, and container
filesystems. The lease duration defaults to 300 seconds (`DEFAULT_LEASE_SECONDS`);
contention raises `LedgerWriterLockConflictError` with holder details. A crashed
process's expired lease is reclaimed atomically via `UPDATE`.

Every writer takes the lock. `ubt/api/review.py`'s `apply_human_edit` is the
canonical example of the pattern: acquire, refuse with a typed conflict if
another process holds it, do the ledger write in one transaction, feed the
accepted pair back into the shared TM, release in `finally`. `re_export.py`
follows the same shape. Read-only surfaces (status, inspect, MCP fallback reads)
never take it.

## The block lifecycle

`ubt/core/ir/models.py` is the single authority. `BlockStatus` is the block's
lifecycle vocabulary: `pending` → `drafted` → (`mtqe_passed` | `repair_pending`
→ `repaired`), plus the terminal trio `failed` / `needs_human` /
`blocked_human`.

`TERMINAL_STATUSES` in the same module is the one definition of "finished":
`mtqe_passed`, `repaired`, `failed`, `needs_human`, `blocked_human`.
`IRBlock.is_finalized`, the ledger's `NON_TERMINAL_STATUSES`, and the repair
eligibility SQL all derive from that frozenset, so adding a status cannot drift
between them.

**`job_meta.status` has its own, separate vocabulary.** It is `initialized` →
`completed` / `failed` / `cancelled`, and there is deliberately **no `running`
value**: a row stays `initialized` for the whole run, so the column alone cannot
distinguish a live job from one whose process died. Every outward surface
normalizes before answering (REST's disk fallback downgrades a non-terminal row
to `failed`; MCP's `ubt_job_status` downgrades only after the ledger goes quiet).
The service-level `JobStatus` enum in `ubt/core/engine/job_queue.py` —
`submitted`, `queued`, `running`, `completed`, `failed`, `cancelled` — is the
queue's vocabulary, and the queue is a separate SQLite file. Do not conflate the
three.

## The render path

There is **one** PDF path. `ubt/render/typst/document_renderer.py` rebuilds the
document from the IR and lets Typst lay it out; it does not draw fragments back
onto the source page. Uniform body type, rebuilt tables with translated headers,
and native math via `ubt/render/typst/math_transpiler.py` all follow from
rebuilding rather than overlaying. The renderer is deterministic — same blocks
in, same bytes out. `ubt/adapters/pdf/typst_compile.py` is the subprocess
boundary; both it and the binary probe are total (a missing binary is a verdict,
never an unhandled `FileNotFoundError`).

Two companions lower the same delivered blocks for other readers:
`ubt/render/html_view.py` (`compose_html`) and `ubt/render/epub_view.py`
(`compose_epub`, which reuses the HTML view's `render_fragment` so the two
cannot lower a block differently). The export stage calls both
(`ubt/core/engine/stages/export.py`).

## Entry points

| Entry point | Module | Notes |
| --- | --- | --- |
| `ubt` (CLI) | `ubt/cli/main.py` | Typer app; subcommands include `translate`, `assess`, `status`, `verify`, `doctor`, `worker`, `console`. |
| `ubt-api` | `ubt/api/app.py` | FastAPI app plus `run_server`; `create_app` is the testable factory. |
| `ubt-mcp` | `ubt/mcp/server.py`, via `ubt/mcp/_entrypoint.py` | stdio MCP server exposing the `ubt_*` tools. |

### Two executors, two deployment modes

Two things can run a job, and they are **not two implementations of one job** —
each deployment mode drives exactly one of them:

- **Embedded** (`ubt/api/manager.py`, `JobManager`): the API process runs each
  job as an in-process asyncio task. This is the default (`job_mode =
  "embedded"` in `ubt/core/config.py`). Zero setup, single host.
- **Queue** (`ubt/core/engine/job_worker.py`, `JobWorker`): the API only enqueues
  into a durable SQLite queue (`ubt/core/engine/job_queue.py`) and separate
  `ubt worker` processes drain it. Jobs survive restarts and scale across
  processes. Selected by `job_mode = "queue"` or an explicit `queue=` argument
  to `create_app`.

The pipeline itself is reused verbatim. The queue worker owns the cross-process
concerns: it heartbeats the job lease, persists progress so another process's
SSE can serve it, honours a cancel request between events, and writes the
terminal queue status back.

What the two *do* share is `ubt/core/engine/job_runner.py`'s
`build_job_orchestrator` — one factory, because the hand-written copies had
already drifted once. The decision of whether to rehearse stays with each
executor: the API honours its own process's `dry_run`, while the worker
recomputes from its own key and overrides an API auto-decision.
