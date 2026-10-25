# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Pre-1.0; tagged `v0.1.0`. Everything below is in the current tree.

### Changed

- **The wizard walks in task order, and spending has guardrails.** The form
  read as a field inventory — upload, provider, languages, presets, budget,
  advanced — while the operator's actual decision sequence is: provide a file,
  weigh quality against cost, cap the spend, go. The sections now carry step
  headers（① 提供文件 ② 质量与代价 ③ 预算与提交）; provider and languages fold
  behind a summary badge（默认 · en→zh）because their defaults are sound and any
  change re-prices automatically; each preset card now speaks in
  user-perceivable consequences（最完整的处理…最快出稿…）with the engine
  parameters that were the old copy（DOCLING · 强制富提示词）folded into a per-card
  InfoTip. Two spending guardrails: compiling without a valid quote — the
  assess failed, or a re-price landed stale — now confirms once per quote
  fingerprint instead of silently skipping the page's core promise, and the
  dialog states the estimated cost when the budget is uncapped; a one-click
  chip offers a 1.5× quote cap when a paid quote exists; and a greyed Start
  Compile finally says why.

- **The quality gate speaks human first, and every deliverable previews
  in-console.** The gate screen rendered engine vocabulary to everyone —
  `0 corrupt / 172 masked`, `fail-closed 0 · coverage 100.0%` — summarized a
  visual scan that ran four tiers over all 14 pages as「像素校验已跳过」because
  *any* tier skip zeroed the row, and offered every artifact, from the
  reader-facing PDF to the metrics JSON, as nothing but a download button. The
  page now renders two densities: a default simple view (a verdict sentence,
  five icon+chart check cards, a per-page scan grid that shows the gate really
  looked at each page) and the previous professional view behind a toggle
  persisted in `ubt.quality.mode`. The deliverables section groups its files by
  audience — reading / translation hand-off / records — with one plain-language
  line per file; PDF and HTML embed in a preview drawer, XLIFF renders as a
  source/target segment table, and the JSON sidecars render as digests with the
  raw payload behind a disclosure. Embeds ride a new `?inline=1` on the
  download endpoint (`Content-Disposition` dropped, media type resolved by
  suffix) served under a CSP `sandbox` directive — the HTML deliverable is
  built from untrusted book text, so even a fetched-directly preview runs
  scriptless in an opaque origin. Honesty repairs landed with the redesign: the
  collapsed visual pillar states which tiers ran (「4 项扫描已运行 · 视觉语义判定未
  运行」), the radar chords across an unmeasured axis instead of drawing a 0,
  the ledger pillar's copy stops claiming a reconciliation the engine never
  implemented, and the gate polls (5 s) while its report is absent so a page
  opened mid-run no longer sits on「尚无判定结果」forever.

- **The pre-flight slowness gate measures extraction cost, not a proxy.** The
  `PATHOLOGICAL_PAGE_RISK` warning had two wrong inputs in a row: the raw
  fragment count warned on every dotted table of contents (2,289 fragments over
  5,363 characters, extracts in 0.02s), and its replacement — grouped text rows
  — measured something uncorrelated with time. Across 844 corpus pages
  `corr(extract_lines time, n_text_rows)` is -0.02 while
  `corr(time, rects × chars)` is 1.000, because `extract_lines` pays one
  `get_text_bounded` + `_probe_rect_font_style` per fragment and each walks the
  page's characters. The row-count gate scored the one genuinely slow page
  *better* than an ordinary one (26 rows vs a typical 40-60) and fired on 0 of
  844 pages; the cost gate fires on that page and nothing else, with 8.1x
  headroom over the worst legitimate page. `PageFacts.n_text_rows` and the
  extra per-page `_group_rows` pass it required (1.3s on the page above) are
  gone; the profile cache version moves to 5.
- **The segments API owns mark unwrapping; the workbench's parallel parser is
  deleted.** The engine writes two `<mark>` wrappers with different meanings
  (`ubt-failed-draft` = a real draft whose repair call failed;
  `ubt-blocked-human` = quarantined source), and the console re-implemented
  half of that grammar client-side — so quarantined blocks rendered their raw
  markup in the review pane and an edit seeded from them would have written
  the wrapper back into the ledger. `serialize_segment` now delivers the
  unwrapped text (`delivered_text`) plus the facts (`repair_failed`,
  `repair_error`, `quarantined`) through the engine's own delivered-text rule:
  one owner, and a frontend that cannot drift again.
- **An absent visual report is no longer a passing gate.** The quality view
  treated a missing visual report (gate disabled, or a non-PDF deliverable) as
  a clean pixel gate — a full-score pillar and radar axis for a check that
  never ran, the exact false attestation the view's own comments promise
  cannot happen. Missing now reads as "no report", warn-scored, with the
  reason; and a `warning` verdict (deliverable, human review pending) no
  longer renders under the red "GATE BLOCKED" banner.
- **Wizard quotes stay honest.** A silent re-assessment failure kept the last
  quote but blanked the panel with no explanation (now: the quote stays,
  flagged as possibly outdated); and a preset/language/provider change made
  while a deep assessment was in flight was never re-priced — the operator
  submitted a differently-priced payload than the dollars shown (now: a
  fingerprint comparison re-prices on settle). Numeric fields
  (concurrency/chapter window) gained the budget field's client-side guard
  against hand-typed zeros the server 422s.
- **The SSE client gives up on a stream that never opens.** An embedded-mode
  terminal job (anything a CLI run completed, or after a restart) has a
  `/status` fallback but no `/stream` fallback, so the console retried a 404
  forever at one request per backoff window per open tab; six consecutive
  failures without a single successful open now stop the retry. A stream that
  connected once keeps the old retry-forever behaviour.
- **The console's long tail**: downloads from a stale delivery open the tab
  inside the click gesture (the re-render outlasts the browser's activation
  window and the popup blocker ate the old click); `loadMore`/term-refresh
  carry the same staleness guard as `load`; switching jobs inside the review
  view can no longer persist job A's drafts under job B's key; the TM panel
  pages past its first 200 rows; reparse and delete demand the typed job id
  their own comments always claimed; the confirm dialog actually traps focus;
  AuthGate cannot wedge on a network-layer reject; provider-editor loads no
  longer overwrite typing; every rendered string (the crash screen included)
  goes through i18n; and the never-used react-query provider and Card family
  are gone from the bundle.

- **One FastPassFilter per run, owned by the run.** The QE runner used to
  build a private filter that had to be taught the glossary, the document
  context and the vocabulary by hand while the draft stage bound a second
  filter — three measured gate-versus-score contradictions came from that
  split, each patched with cross-carry rebind code. The pipeline now hands the
  run's filter to the runner (`with_fast_pass`), so whatever the draft stage
  binds is, by identity, what `score_pairs` scores with; the quality gate's
  re-synchronisation block and the glossary rebind's hand-carry are deleted.
- **Recursive gutter cuts: three-column pages read column-major.** A single
  cut merged the third column into a neighbour cluster, interleaving columns
  2 and 3 line by line (and cutting a 24-line three-column scan into 17
  paragraph groups in the VLM fallback). Each region now recurses into its
  two sides under strict sub-split guards; differential testing against the
  previous implementation: 4000/4000 random one- and two-column pages
  element-for-element identical, 300/300 three-column pages column-major.
- **The pdf-oxide binding surface is contract-pinned.** UBT touches exactly
  four symbols; a drift in a young 0.3.x dependency used to surface as a
  silent runtime `None` in the visual gate. The contract test exercises
  construction, page count, DPI parity (the crop math depends on it), PNG
  output and text shape against a real document — raising the pin is now a
  review-time decision gated by a red/green test, and the stale size/API
  notes in `pyproject.toml` are corrected.

### Fixed

- **The pre-flight quote's token breakdown sums, and the duration owns its
  heuristic.** The wizard showed 源文 beside 提示 under a total of
  提示+补全, but the engine bills prompt = (per-call prefix × calls) +
  source — the source term is a *subset* of 提示, so the three figures
  over-summed by ~28% (114,244 vs 89,460 on the reference document). The
  breakdown is now the additive one — source + call-prefix overhead +
  completion — and sums to the total exactly; the prefix overhead is derived
  client-side as `prompt_tokens − estimated_tokens`, which the engine's
  `estimate_draft_cost_from_totals` guarantees is exactly the per-call prefix
  fan-out. The total is relabeled 草稿阶段 with the footnote stating that the
  repair fan-out and QE judging are not included (CostQuote's token fields
  carry the draft stage only; repair is quoted separately in
  `repair_blocks`). The duration row gains a (启发式) tag and an InfoTip
  carrying the engine's own basis text — `runtime.heuristic` is always true,
  and a bare "4m – 9m" read as a measurement.

- **A blocked delivery is no longer downloadable by accident.** `blocked` means
  blocks were quarantined (MQM Critical unresolved) or failed, so they ship a
  source-only placeholder instead of a translation — the reader meets English
  in a Chinese book. The CLI banner, the console's gate screen and Mission
  Control all said so, while `GET /jobs/{id}/download/{key}` served the file
  anyway, which made the verdict advisory at the one moment it matters. The
  endpoint now refuses `primary`/`secondary`/`epub` from a blocked job with 409
  and points at the quality report. Three deliberate narrowings keep it from
  becoming a dead end: the audit sidecars (quality report, contract, XLIFF,
  metrics, visual report) stay open, because they are how an operator finds out
  *why* the gate refused; `?inline=1` is exempt, since that is the console's
  preview pane and the operator needs to look before deciding; and `?force=1`
  overrides, mirroring `strict_contract` — safe by default, loosening must be
  asked for. The console asks first (a confirm dialog naming the consequence)
  instead of letting the 409 arrive in a blank tab, and `/deliverables` now
  carries `delivery_verdict` so the warning lands before the click.
- **The wizard's unnamed controls have names.** The hidden file input, the
  manual path field and four `<select>`s (provider, source/target language)
  carried no accessible name and no label association — reachable only by
  position, silent to a screen reader. `aria-label`s and `htmlFor`/`id` pairs
  fix the read-out, and clicking a label now focuses its control.

- **A page whose text rects are nested no longer ships its text 12x over.**
  `extract_lines` reads one `get_text_bounded` per pdfium rect, and a box that
  selects far more characters than its own width can hold is a duplication —
  but that guard only ran when the scatter-debris cleanup had fired, i.e. only
  on a marker cloud. A page can clear that precondition and still be nested:
  arXiv 2609.20519 p1 is 9,395 rects over 104,456 characters whose boxes
  intersect their neighbours' glyphs, summing to 3,220,105 characters (30.8x
  the page's own text). Its rects are 3.6pt tall, above the cloud's 2pt median,
  so 91% of them were kept and every one over-read: `extract_lines` emitted
  1,331,556 characters, one block reached the translator carrying 146,289
  characters of duplicated fragments, and the document parsed to 1,298,561
  characters against an assessed 160,848 — a wrong delivery *and* an 8x
  under-quote. The guard is now self-contained (a box that wide cannot hold
  that much text) and needs no precondition. Across the 844-page corpus it
  fires on exactly that one page; the measured text of every other page is
  unchanged.
- **The console no longer certifies a delivery its own gate refused.**
  `completed` is the pipeline finishing, not the gate passing: the ledger
  allows a terminal write with quarantined blocks in it, because `NEEDS_HUMAN`
  and `BLOCKED_HUMAN` are terminal statuses. Mission Control read neither
  counter and printed "compiler output certified" over a job whose delivery
  verdict was `blocked`; it now reads `blocked_human_blocks` and distinguishes
  certified / awaiting review / not released. On the quality gate, the READY
  banner claimed "formula dropout is 0%, terminology drift within tolerances"
  while the verdict it renders is computed from block counters alone
  (`delivery_verdict_from_blocks`) and reads neither metric — it now says what
  it can prove. The ledger pillar was a constant `pass`/100 over an unpriced
  run; it warns when the run was never priced.
- **The console can turn quality measurement on.** Every finished job reported
  "QE not measured (heuristic engine)" and stopped there: the remedy
  (`qe_engine = "tiered"` **and** `qe_judge_enabled = true` **and** a judge
  model) lived only in `ubt.toml`, so the status named a fix the reader could not
  perform on the screen they were on. System Doctor now has a QE card that reads
  the effective configuration, says whether a run can produce a score at all, and
  writes `[defaults]` — `PUT /system/qe-config`. It writes `qe_engine` and
  `qe_judge_enabled` together on purpose: the label alone selects no judge (the
  pipeline only warns), so two independent switches could save a configuration
  that reads "tiered" and scores nothing. The judge's endpoint stays a
  `[providers.<name>]` block, named from the same card; `UBT_QE_*` variables are
  reported as overrides rather than silently outvoted.
- **The console describes quality estimation in the operator's terms.** The
  badge said `QE 未测量（heuristic 引擎）` / "QE not measured (heuristic engine)",
  which named an internal engine and left the reader to guess whether something
  broke. It now reads "no quality score (rule checks only)", and the
  Quality Gate explains what those checks are (structure, glossary, dropped text,
  garbled characters → pass / warning / blocked), why that yields no averageable
  score, and that `0.0` in the report is a placeholder rather than a bad result.
  The remedy points at System Doctor → Quality estimation instead of an
  environment variable, and the audit table's `qe_score_source` token is rendered
  as "rule checks" / "model scoring" rather than the raw `heuristic` / `llm_judge`.
- **Policy-preserved blocks are no longer described as skipped list markers.**
  The Mission Control counter said "blocks skipped (images, list markers)", but
  on the reference job 30 of the 49 were bibliography entries kept verbatim for
  retrievability, 17 were image/formula assets, and the rest were a byline and an
  arXiv stamp. Both locales and the SSE tail now say "preserved as-is" and name
  the real population, so the number stops reading as work the engine refused.
- **The live console log collapses adjacent duplicate frames.** The stream emits
  a frame per engine tick and the panel appended every one, so the identical
  `118/148 blocks (+49 preserved) · QE: rules only (no score)` line repeated
  dozens of times between two engine messages and pushed the real messages out
  of the 400-line buffer. Only *adjacent* repeats are dropped; a line that
  reappears after a different one is real history.
- **Dotted tables of contents no longer warn of geometry that never runs
  slow.** The pre-flight `PATHOLOGICAL_PAGE_RISK` gate read pdfium's raw
  rect-fragment count, and a Typst/LaTeX TOC dots each leader line: the
  measured trigger (arXiv 2608.25512) was 2,289 "rows" on an ordinary
  contents page that the geometry engine regroups into 35 rows and orders in
  0.02 s — a warn-level false positive on every dotted TOC, claiming `--deep`
  or a full translation would be noticeably slower. The profiler now reports
  `n_text_rows` (fragments regrouped into rows by vertical overlap — the
  input size textgeom's quadratic stages actually see) and the gate reads
  that; the copy counts rows instead of "rectangle rows" in both locales.
  The route badge also stops showing raw enums (`mono · high_fidelity`) and
  renders localized labels.
- **Swapped checksum-less placeholder echoes fail closed instead of restoring
  crossed spans.** A model that renumbered math/code tokens while dropping the
  checksum suffix had each bare index restored verbatim wherever its digits
  claimed to belong, so two swapped spans shipped crossed math with a clean
  integrity report (verified on this code base: `x²`/`y³` interchanged,
  `clean == True`). A bare echo that also participates in an order inversion is
  now ambiguous-by-definition and lands in `mismatched`; a verified (checksummed)
  reorder stays legitimate and clean.
- **A losing writer-lock claimant no longer stalls the event loop.** The lock
  take and the ledger's schema work ran inline in async paths — up to 10 s of
  `busy_timeout` parking every slot's loop, the missed-heartbeat path that
  reclaims a still-running job and bills the book twice. Both now run under
  `asyncio.to_thread`, and a busy-timeout exhaustion surfaces as the typed
  `LedgerWriterLockConflictError` (free retry) instead of a generic
  `OperationalError` that burned a worker attempt.
- **A re-export can no longer spend money nobody bills.** The export stage's
  visual gate reads the config bound on the ctx, so an operator with
  `visual_judge_enabled` made paid VLM calls through a path that binds no usage
  sink — invisible to every budget surface. The module's zero-spend contract is
  now enforced (judge closed on a config copy, like the rehearsal path); the
  deterministic T0/T1 checks still run.
- **A re-export refuses a contradicting output suffix** the way a first run
  always did, before rendering — a PDF engine forced into `.md` also leaked
  quarantine `<mark>` wrappers verbatim into the deliverable, because the
  non-PDF branch read `target_text` raw instead of through `delivered_text`.
- **Ordered lists keep their numbering in the companion HTML/EPUB views.** The
  `ListItem` branch emitted `<ul>` unconditionally while the same job's Typst
  PDF recovered `1.`/`2.` from the marker; the companion views now ask the same
  `_list_is_ordered` the Typst renderer asks.
- **A cancel that lands after the export finalized reports completed.** It used
  to write the queue row CANCELLED against a ledger already finalized
  `completed` with the artifact on disk — two terminal surfaces in permanent
  disagreement. The worker now arbitrates with a read-only ledger probe.
- **`"Appendix C …"` headings reach the label gate.** The regex alternative
  carried a leading space under a `^` anchor and was unreachable; the survival
  label is the letter alone, matching the bare-letter form ("附录 C" keeps it).
- **Everyday words no longer trigger semiconductor detection.** `source`,
  `gate`, `channel`, `drain` and `poisson` matched ordinary prose at the raw
  4-hit threshold and injected semiconductor style instructions (plus 40 forced
  seed terms and a job-id namespace change) into unrelated books.
- **The repair prompt and the acceptance gate share one table-grid standard.**
  The prompt counted any line with two pipes as a grid row while the gate
  requires the row to open with a pipe, so prose with inline pipes was ordered
  to "preserve the exact NxM grid" the gate never checked.
- **`db_dir`/`cache_dir` defaults anchor to the discovered config file.** The
  config search walks parents while the defaults stayed CWD-relative, so a run
  from a subdirectory read the project's `ubt.toml` but wrote ledgers under
  `<subdir>/.ubt` — resume from anywhere else found no healthy ledger.
- **A refused submit leaves no ghost output directory**, the wire User-Agent
  carries the installed version, and `DELETE /jobs/{id}` also drops the job's
  SSE replay history.
- **The review workbench is guarded against stale loads** (a slow response for
  an old filter no longer clobbers the new one) and a term cascade no longer
  resets the reviewer's paged-in segments.

- **A completed job can no longer exist without its render digest.** `finalize_job`
  committed `completed` first and the export stage wrote `rendered_text_digest`
  afterwards, so any failure in between left the one state the freshness check has
  to read as "not stale". The download endpoint then served a file rendered from
  text the ledger no longer held, and nothing self-heals it (`completed` is never
  rewritten by the failure path). The digest is now an argument to
  `finalize_job` and commits in the same transaction.
- **A re-export no longer rewrites any block.** The docstring said it "doesn't own
  any block's fate" while the stage still ran the deterministic glossary
  substitution, the HTML structure checker (which can wrap a target in `<mark>` and
  set `FAILED`), the math guard (which resets a drifted formula to its source) and
  the length policy. A formatting-only re-render could therefore undo a human edit
  or quarantine a block a later resume would not pick up.
- **DIALOGUE was treated as a table by the zero-token gate.** It was missing from
  `PROSE_BLOCK_TYPES`, and FastPass reads "not prose" as "table", which switched
  off the sentence-drop gate and lowered the script-density floor to 0.02.
- **A target consisting only of symbols passed every gate.** The residue strip left
  an empty string, which the density check read as ratio 1.0, so `!!!@@@###%%%`
  scored "Flawless". A target with no letters at all is now a critical defect when
  the source had prose, and a legitimately letter-free table body still passes.
- **`skip_translate` blocks failed their own language gate.** A verbatim formula or
  code block is supposed to arrive in the source language; the density gate
  rejected every correct "leave the LaTeX alone" repair candidate at ratio 0.00.
- **Ragged and single-row tables skipped both table checks.** `markdown_grid_shape`
  returned `[]` for any non-uniform grid, so a table whose translation dropped or
  added a cell — and a header-only table — went unexamined.
- **Hangul was budgeted as half-width in table layout.** Korean fell through to
  `isalpha()` (0.557 em instead of 1.0), and the whole syllable block counted as
  one unbreakable run, so Korean tables were ~44 % under-allocated and overflowed.
- **An asset filename could break the whole compile.** A stem containing a quote
  closed `image("…")` early; names are now restricted to what is safe inside a
  Typst string literal (the content digest already makes them unique).
- **The visual gate's Typst-source tier never ran.** It looked for `<pdf>.typ`,
  while the renderer writes `.<stem>_typst/main.typ`. The tier is now listed in
  `tiers_ran`/`tiers_skipped` so "never scanned" is distinguishable from "scanned
  and clean".
- **`⟦I⟧`/`⟦/I⟧` were not registered sentinels.** An italic marker a model rewrote
  as `\llbracket I\rrbracket` was not recognised, reached the math transpiler, and
  one such formula fails the whole document.
- **The fallback chain had no total attempt budget.** `max_retries` is per model,
  so N models meant `N * (max_retries + 1)` paid calls, each with its own backoff
  (up to a clamped 375 s), for one block. The chain now shares one budget.
- **Non-429 retry backoff was not cancellable**, and the truncation-continuation
  loop never checked the cancel token, so a cancelled job kept buying completions.
  The per-run cancel token is also unbound in `run()`'s `finally`: `JobWorker` runs
  job after job on one task, and a set token leaked into the next job's backoff.
- **`low_legibility_font` had no producer.** A table laid out at the 0.60
  readability floor now emits it; the flag was declared, documented, and never
  written.
- **TOC `auto` invented a contents page from any five tagged lines.** A row now has
  to carry a page number or name a heading that exists in the document (and a row
  cannot vouch for itself).

### Removed

- `LENGTH_OVERFLOW_TO_HUMAN` / `LENGTH_POLICY_PAGE_KINDS` and
  `apply_length_policy_flags`: unreachable. No renderer ever emitted an
  `overflow*`/`spill*` skip reason, so the flip could not fire.
- **The intentional-preserve skip whitelist**
  (`render_skip:{policy,non_prose,chrome,footer}` and four `inplace_skip:` twins).
  None of the eight had a producer, so `preserved_blocks` was a permanently-zero
  KPI in real runs, and the one flag that looked like it belonged was wrong:
  `triage` tagged a NEEDS_HUMAN non-prose block `render_skip:non_prose`, but the
  renderer typesets that block's rejected *draft* (fail-open), so the delivery
  contract recorded a translation the reader received as "text missing"
  (`SOURCE_KEPT`, or `MISSING` for an asset). The advisory is now
  `non_prose_degraded: …`, outside the skip vocabulary. A block ships its source
  for exactly one reason — `skip_translate` — which yields
  `TextDisposition.VERBATIM` on its own.
- `RepairLoop.bottom_percentile`: stored and never read. The knob still drives
  `ubt assess`'s quote; it never selected a repair candidate.
- `quality_report.md` from `SidecarKind` / `RUN_REPORT_KINDS`: nothing wrote it, so
  every export deleted a file that could not exist.

### Changed

- **The book's source text is read and indexed once per run.** `source_text` is
  immutable after ingest, yet `run_draft_stage` and `run_quality_gate_stage`
  rebuilt the whole joined string from a full-table scan **per chapter** — and
  the omission gate then re-derived its single document-wide answer ("is this
  all-caps run an acronym, or a word capitalised for a table label?") from that
  raw text **per block**. `SourceDocumentIndex`
  (`ubt/core/engine/source_document.py`) loads it once through
  `fetch_source_texts` (a bare column read: `IRBlock` construction was 88 % of
  `get_all_blocks`' cost and neither caller used it) and exposes a
  `DocumentVocabulary` built the same way. `identifier_terms`' boundary-aware
  search over the whole document became a set membership test against that
  vocabulary; the verdict is identical (pinned by a property test against the
  retired implementation). Measured on a synthetic 20k-block / 3.9M-character
  book: 139.7 ms → 0.21 ms per `evaluate()`, i.e. ~5581 s → ~8.5 s of gate CPU
  across a full run.
- **Reads no longer queue behind the writer's lock.** `SQLiteJobLedger` served
  every read on the writer's single connection under one process-wide `RLock`,
  so a full scan stalled everything else: the progress event's `get_job_stats`
  went from 3 ms to 384 ms p95 while a scan loop ran (it is issued once per
  batch — 666 times on a 20k-block run). Reads now use a per-thread `mode=ro`
  connection (`_get_read_conn`) and convert results **after** releasing it; WAL
  already allows readers concurrent with the single writer, and the writer keeps
  its one connection and `BEGIN IMMEDIATE`. Measured: progress queries p50
  375.9 → 3.8 ms, p95 408.5 → 62.7 ms, throughput 16 → 154 samples.
- **`OmissionGate.evaluate(document_text=...)` honours its argument.** The
  parameter was documented as "an explicit argument wins over the bound run
  context" and then ignored: the body normalised a local and read
  `self._document_text` anyway, so every keyword caller silently got the bound
  context. It now uses the resolved value, and also accepts a prepared
  `DocumentVocabulary` so a batch does not re-index the book per block.

- The ledger schema is a **single flat baseline** (`SCHEMA_VERSION = 1000`). The
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
- **Docling ≥ 2.136** is now the floor for the `pdf` / `academic` extras. The
  formula VLM (CodeFormulaV2) crops each equation with an `expansion_factor`
  margin, and its 0.18 default reaches into the equation number and the
  neighbouring prose line on a tight academic layout, so the model transcribes
  that text into the LaTeX. Docling's own preset API now accepts the margin and
  a repetition guard; the parser sets both through it
  (`expansion_factor=UBT_FORMULA_CROP_MARGIN`, default `0.08`, plus
  `stop_on_repetition=True`). Only engaged when `formula_enrichment=on`; the
  default screenshot path is unaffected, and the conversion cache key already
  folds in the pipeline options, so a margin change or a version bump
  re-converts instead of replaying a stale entry.
- A recovered **table of contents** is now placed at a **fixed position** —
  after the title page and before the body — instead of wherever the reader
  happened to see it. On a paper whose source spelled its section list before
  the title block, the TOC was emitted on the opening page *above* the title,
  which is not where a TOC belongs. And a paper now carries **no TOC by
  default** (`UBT_TOC_MODE=auto`): a paper has none by convention, and the run a PDF
  reader picks up is the source's section list with the *source* page numbers,
  which the reflowed deliverable does not reproduce. `UBT_TOC_MODE=on` forces it,
  `off` suppresses it; a TOC the reader never recovered is never invented.
  Its typography was rebuilt too — real dot leaders, a generous row gutter, and
  a two-step size hierarchy replace the cramped two-column rows.

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

- A **password-protected PDF** now fails `translate` with an actionable
  `DocumentParseError` ("decrypt it before translating") instead of Docling's
  raw `ConversionError` plus a burst of pdf_oxide `Decryption failed ... Not
  authenticated` noise. The guard runs before engine routing and again at
  manifest extraction, so Docling and pdfium share one message; `--profile auto`
  skips the archetype probe for an encrypted file so the noise never starts.
  `assess` keeps its existing Chinese refusal.
- A table of contents now renders **as a table of contents**, not as a run of
  section headings. The PDF reader pairs each TOC title with its page number and
  tags the block (`provenance.toc_entry` / `toc_page`), but the renderer never
  read the tag, so every TOC row took the `#heading` path: the paper's contents
  became orphaned section headings with no body (a near-empty page) and the
  recovered page numbers were dropped. The renderer now groups each run of
  consecutive `toc_entry` blocks into one full-width TOC block — a `目录` heading
  with dot-leadered `title … page` rows — which also keeps the contents out of
  the body columns.
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
  provider-dependent (Anthropic, OpenAI chat/responses and Gemini all report
  cache usage; a gateway that omits it reads 0) and
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
  path, `publication` / `standard` use the lossless docling engine (the probe
  reports evidence, it no longer routes).
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
  `cover_mode`, several inert formula and render-mode fields), and `prices_file`
  is now honoured via `UBT_PRICES_FILE` (a `[defaults]` entry is rejected by the
  provider allow-list).
- Ledger and report durability: the most-read reports are written atomically,
  the schema is a single flat baseline (`SCHEMA_VERSION = 1000`) and a pre-1000
  database is refused with instructions rather than migrated in place, and the
  schema carries the element's
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
