# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Pre-1.0, no tags yet. Everything below is in the current tree.

### Added

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
