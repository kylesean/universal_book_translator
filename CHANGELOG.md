# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Pre-1.0, no tags yet. Everything below is in the current tree.

### Changed

- The ledger schema is a **single flat baseline** (`SCHEMA_VERSION = 1`). The
  historical 14-step incremental migration ladder was removed; a fresh database
  is created directly with the final shape (composite `(job_id, block_id)`
  primary key, no dead columns). A database written by an older build is
  **refused with instructions** rather than migrated in place — silently
  rewriting a half-migrated legacy ledger risked the run's only copy of its
  translations.
- The single-writer guard is now a **SQLite-backed lease** (`job_locks` table),
  replacing the `msvcrt`/`fcntl` file-offset lock on a separate `.writer.lock`
  file. The old lock could not be taken on NFS/overlayfs/K8s PVCs and behaved
  differently under Docker PID 1; the table lock works everywhere the ledger
  does, and its expiring lease lets a crashed writer be reclaimed instead of
  wedging the job forever.
- The QE rehearsal exemption is an **explicit run flag** (`rehearsal=True`
  threaded from the mock/dry-run context), not a `REHEARSAL_MARKER` substring in
  the target text. A real translation that happens to contain the marker now
  faces the full gate; a rehearsal still passes every structural invariant.
- The neural **CometKiwi QE runner was removed** (`comet_runner.py`,
  `comet_score_ipc.py`), completing the commercial-license cut for the
  CC-BY-NC-SA weights. `SubprocessQERunner`, its resident-subprocess machinery,
  and the `comet_model` / `comet_script_path` config knobs are gone; the
  zero-token `HeuristicQERunner` / `MockQERunner` moved to
  `ubt/core/qe/heuristic.py`. A config still naming `qe_engine=comet|cometkiwi|
  neural|subprocess` normalizes to `heuristic` with a warning instead of
  crashing.
- HTTP timeouts are a **forced per-phase split** on every provider client:
  `connect=15s`, `write=15s`, `pool=30s`, and `read = max(120s, api_timeout)`.
  A single-float timeout could pin a pool slot for the whole `api_timeout` on a
  dead endpoint.
- The **Anthropic Message Batches** and **Gemini `:batchGenerateContent`** APIs
  are implemented alongside the OpenAI-compatible batch path, so a book run gets
  the provider's official batch discount whichever protocol is selected. The
  provider now routes batch calls to the transport for the active `api_mode`
  instead of always delegating to the chat transport.
- The rich/macro draft prompts drop the "pink elephant" negative rules
  (anti-passive-voice, OCR-denoising) and the "placeholders keep their original
  relative order" constraint that contradicted the reorder-tolerant unmask, in
  favour of positive register guidance.
- The HTML reading console ships KaTeX **rendering** (the auto-render JS, not
  just the stylesheet), a light/dark **theme toggle**, and emits formula blocks
  as `.math-tex` spans carrying their LaTeX. The companion is now written
  bilingual by default, so its three read modes (target-only / stacked /
  side-by-side) are actually reachable.
- Chapter-level long-context drafting is the **default** (`macro_chunk_size`
  default 8, was 1): consecutive blocks are packed up to a section boundary and
  sent as one `<block id=...>` request instead of one call per block. The dry-run
  echo provider emits the macro envelope so rehearsals exercise the same path.

### Added

- Academic Typst template is now **reachable at runtime**, and its **column
  count mirrors the source** instead of being hardcoded: the run's genre profile
  is stamped onto the manifest at ingest (`genre_profile`) and selects the
  academic styling (title/abstract banner, dynamic header). The physical column
  count is resolved independently by `resolve_document_columns` from the page
  profiler's multi-column finding (`run.route_decision.multicolumn_page_share`),
  so a two-column paper stays two-column and a **single-column paper stays
  single-column** — `--profile` styles the *text* and never forces a column
  count (previously an academic profile forced two columns, and a missing
  dispatch key made the whole branch dead code).
- `--columns 1|2` (CLI), `layout_columns` (REST `JobSubmitRequest`), and
  `UBT_LAYOUT_COLUMNS` (env) give explicit control over the reflowed column
  count. Unset aligns to the source; set 1 or 2 to override (e.g. `--columns 1`
  for comfortable single-column reading of a two-column paper, or `--columns 2`
  to force two columns for print).
- The page profiler's **multi-column detector no longer counts page furniture
  as a right column**. A rotated margin stamp (the arXiv/JMLR watermark running
  vertically down the page edge) is emitted as one near-square glyph per rect,
  which `column_right_share` counted as a right-column segment — reporting
  `right_row_share` ≈ 0.44 on body pages and `multicolumn_page_share` = 0.225 on
  a paper whose body is plainly **one** column. That false signal then drove the
  renderer to lay the delivery out in two columns. Two guards fix it: narrow
  (furniture-sized) fragments never form a segment, and a right-started segment
  only counts when its row also carries left-column text. A genuine two-column
  document scores unchanged (verified 0.83 before and after); the real
  single-column paper drops from 0.225 to 0.0.
- A recorded-fixture layer (`tests/cassettes.py`, built on `httpx.MockTransport,
  no new dependency) so transport-level tests replay real recorded responses;
  a missing cassette fails loudly instead of letting a test pass vacuously.
- One PDF render path: the document is rebuilt from the IR and laid out by
  Typst, replacing the overlay engine that drew translated fragments back onto
  the source page. Tables are re-typeset with translated headers, math is
  typeset natively from LaTeX, and the output is deterministic.
- A three-state delivery verdict (`passed` / `warning` / `blocked`) shared by
  the CLI, the REST report and the web console, so they can no longer disagree
  about whether a delivery is shippable.
- The CLI prints the Job ID at job start, with the resume command, so an
  interrupted run can always be reached again; Ctrl-C now says the job was
  checkpointed and exits 130 instead of printing a bare abort.
- A quality-measurement flag: a run whose quality was never machine-scored says
  so, instead of showing an average score of 0.000 as if it were measured.
- MCP tools for re-exporting a deliverable after a human edit, and for listing
  jobs from earlier sessions; a live job id is now an idempotent resubmit rather
  than an error.
- Operator console: a queue position, real error messages from the API, and
  labels that name blocks rather than pages.
- Typed response models for the workbench and asset endpoints, so the OpenAPI
  schema describes them and the web client is type-checked against the server.
- Profile `auto` everywhere (CLI, REST, MCP, web), inferring the domain profile
  from the document instead of silently resolving to an empty seed glossary.
- Deterministic glossary enforcement is now an explicit opt-in on the long
  route (`UBT_DETERMINISTIC_GLOSSARY_LONG_ROUTE`), and re-export reads the same
  flag so a re-render enforces what the original run enforced.
- A wheel packaging smoke test: the archive, the embedded console bundle and a
  clean install are checked, so `pip install universal-book-translator` cannot
  ship a broken package silently.
- Two hygiene guards in the fast tier: a module nobody imports fails the push,
  and a comment or docstring pointing at a deleted file or symbol fails.

### Fixed

- The delivered-artifact check no longer reports a **false `[FAIL]`** for a
  heading the renderer deliberately drops as running page chrome (an arXiv or
  journal stamp, a submission date). The renderer omits those headings so the
  document opens on its title, but the check still demanded their realization —
  a `--profile paper` run reported `artifact carries 280/281` with a bogus
  `missing` id. The drop decision is now a named predicate the renderer owns
  (`dropped_page_furniture`) and the check asks of it, so the audit can never
  again disagree with what the renderer is right not to place.
- ``--fresh`` can now restart a job whose ledger file is a legacy schema. The
  run opens the ledger before `--fresh`'s row reset, so the flattened-baseline
  refusal fired first and there was no way forward without hand-deleting the
  file — testing the tool became impossible. Under `--fresh` an old-schema
  ledger is now reset to the baseline (the operator explicitly asked to discard
  prior state); without the flag it is still refused, never silently migrated.
- The writer-lock lease is **renewed for the life of the run**. The lock is an
  expiring lease; a real book run outlives one window, so without renewal a
  second process could take the job over mid-run and both would bill the
  provider. The pipeline now runs a renewer task alongside the stages.
- A **blank-corner matrix** keeps its empty leading cell (`mat(, b; c, d)`); the
  earlier lead-empty prune ate a legitimate cell because pylatexenc already
  consumes the `array` column spec. An **unstarred** `matrix`/`array` position
  argument (`[r]`) is now stripped like the starred variants.
- The HTML formula span carries its **delimited LaTeX as content** (what KaTeX
  auto-render parses) with the readable Unicode fallback in the tooltip — the
  first cut put them the other way round, so KaTeX had nothing to typeset.
- The AST math transpiler now runs **after** LaTeX normalization, so the spaced
  Docling dialect (`\begin { pmatrix }`) reaches it instead of failing closed to
  plain text; a leaf chunk it cannot represent fails the whole formula closed (to
  readable Unicode) instead of emitting raw LaTeX that fails the Typst compile
  and loses the entire document; and environment arguments (`alignedat{2}`,
  `pmatrix*[r]`) and the `array` leading spec cell no longer leak into the
  emitted math.

### Notes / known limitations

- Prompt-cache reuse: the static-prefix prompt topology is implemented and the
  run's cache-hit rate is **measured and reported** (`cache_hit_rate`), but the
  RFC's ">85% reuse" is a *design target*, not an enforced guarantee — it is
  provider-dependent (only the Anthropic transport reads cache usage today) and
  no gate asserts it.
- `ubt/render/epub_view.py` is a generic re-render *companion*; the **in-place**
  EPUB/DOCX injection lives in the format adapters (`adapters/epub`, `adapters/docx`).

### Changed

- Layering is enforced, not just documented: `ubt.analyze` may not import
  `ubt.adapters`, `ubt.render` may not reach back into `ubt.adapters`, and
  `ubt.core` may not import the compiler packages. Both a pre-push hook and CI
  run the acceptance tool.
- The `ubt.core` port bridge resolves implementations through lazy imports
  instead of a string-keyed SPI registry.
- The HTML and EPUB views lower tables, formulas and figures semantically: a
  table becomes a real `<table>`, a formula degrades to readable Unicode rather
  than raw LaTeX, and a figure carries its image bytes instead of an empty box.
- The render preamble's font stack follows the target language (Japanese kanji
  no longer typeset with Simplified-Chinese glyphs), and `font_family` is
  honoured.
- PDF engine selection is decoupled from preset quality: the probe reports
  evidence, an explicit `--pdf-engine` always wins, and a scanned document still
  escapes to the OCR-capable engine rather than extracting nothing.
- Presets select the PDF parser: `preview` / `fast` pin the lightweight fast
  path, `publication` / `standard` keep quality-first probe routing.
- The fast tier's coverage floor is 70% (was 58, which was too far under the
  baseline to guard anything), it runs under xdist, and CI installs a
  checksum-pinned Typst *before* the tier so the render-contract tests execute
  instead of silently skipping.
- Reorganized the test tree: the round-numbered review-fix files were dissolved
  into the module tests that own each behaviour.

### Fixed

- Formulas reaching the page as literal LaTeX: a pipe inside a table cell
  splitting an escaped LaTeX norm, a CJK `\text{}` annotation making a formula
  read as prose, and a punctuation-only span misaligning every later delimiter
  pair.
- Zero-signal formula misprints: `\mid` rendering as the letters "mid", `\|`
  silently halving every norm, `\sqrt[3]{x}` printing its own source, and a math
  fallback that leaked command names for most unmapped table commands.
- Re-export is a real ledger writer: it takes the writer lock and no longer
  force-fails another run's in-flight blocks or finalizes a job it does not own.
- Quarantined blocks are surfaced honestly rather than shipped raw, and the
  delivery contract classifies them explicitly instead of relying on a fallback.
- A table row that overflows after reconstruction is reported corrupt, not
  verified.
- The short route had no terminology mechanism at all: the bible's zero-token
  miners, skeleton extraction and abbreviation backfill were all gated off, so
  the glossary enforcer it enables had no rules. Measured on one document: 0
  measurable terms before, 30 at 0.98 precision after.
- Five gate families that a real-run autopsy proved wrong: table-of-contents
  ordering, script density on table-shaped blocks, length ratio for CJK,
  heading section numbers, and glossary boundary spacing.
- Term consistency is measured over the blocks that were actually translated,
  not over the whole book.
- A shared QE runner keeps its language pair per run instead of leaking one
  job's thresholds into a concurrent one.
- The provider-backoff cancellation token is per-run context, not router state,
  so cancelling one job no longer aborts another's retry wait.
- Inline math Docling strips out of a sentence is put back.
- Dead configuration knobs were deleted (`table_render_mode="overlay"`,
  `cover_mode`, several inert formula and render-mode fields), and one that was
  silently ignored (`prices_file` in a config file) is now wired.
- Ledger and report durability: the most-read reports are written atomically,
  the v13 to v14 migration keeps its data, and the schema carries the element's
  structural fields and canonical character range so a resumed run renders the
  same document as the original.
- A file-based batch that fetched a block which stayed pending now fails loudly
  instead of silently dropping paid work.

### Removed

- The overlay PDF render engine and its support code, with no compatibility
  layer. `dual_mode` collapsed from five values to three (`mono` / `dual` /
  `auto`); `render_mode`, `formula_mode`, `math_backend` and `facing_spread` are
  gone.
- The `babeldoc` PDF engine, which the factory now refuses with a clear error
  (licensing and upstream CLI-contract incompatibility). Use `docling` or
  `pdfium`.
- Configuration knobs that had no behavioural consumer, and the `overlay` value
  of `table_render_mode` — a stale `UBT_TABLE_RENDER_MODE=overlay` is now
  rejected rather than silently reinterpreted as `auto`.
