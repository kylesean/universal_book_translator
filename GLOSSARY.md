# Glossary

Terms that mean something specific in UBT, one to three sentences each, with
where each lives. Definitions are from the code, not a general dictionary.

## Document model

**IR** — The intermediate representation a run carries: `IRBlock`s plus a `BookManifest`, persisted in the ledger and rendered from. Distinct from the typed AST: the AST is what a reader understood, the IR is what the pipeline executes. `ubt/core/ir/models.py`.

**IRBlock** — One atomic content block: a typed `element` (its structure, the single source of truth), `style` (the adapter's typography finding), and execution state (`status`, `target_text`, `mtqe_score`, `error_flags`, ...). Structural reads go through properties, so a caller cannot set a field that disagrees with the element. `ubt/core/ir/models.py`.

**element** — A member of the closed AST union `ElementT` = `Heading | Paragraph | Dialogue | ListItem | Caption | CodeBlock | Formula | Table | Figure`. An element's *type* is its structure; the region it sits in is its layout. `ubt/model/ast.py`.

**block type** — The `BlockType` enum on the IR (`narrative`, `dialogue`, `heading`, `code`, `formula`, `image`, `table`, `list_item`), derived from the element class. `ubt/core/ir/models.py`.

**flow / region** — Two different axes. `flow` (`FlowID`: `main_story`, `sidebar_aside`, `footnote`, `table_grid`, `caption`) partitions context windows, so a footnote never bleeds into the body. `region` (`RegionKind`: `body`, `title`, `header`, `footer`, `page_number`, `caption`, `footnote`) is the page-furniture role, the one layout vocabulary. `ubt/core/ir/models.py`, `ubt/model/ast.py`.

**span / CompositeSpan / CanonicalSource** — A `Span` is an element's back-reference to where it came from: page, bbox, and a character range into `CanonicalSource.text`. `CompositeSpan` is the same for an element spread across several boxes in reading order (a paragraph crossing a page break). `CanonicalSource` is the offset-stable text stream those ranges index, keyed by the source file's content hash. `ubt/model/span.py`.

**chapter** — A manifest unit (`ChapterMeta`): id, title, spine index. Markup formats chapter at level-1 headings. A PDF is ingested as **one** chapter (`pdf_main`) because a PDF has no cheap chapter structure to read; only a PDF long enough to be partitioned into out-of-core page chunks yields more than one, and those chunks are internal streaming units, not editorial chapters. That is why per-chapter resume and billing do not apply to a whole PDF. `ubt/core/ir/models.py`, `ubt/adapters/pdf/docling_parser.py`, `ubt/adapters/pdf/page_chunking.py`.

## Lifecycle

**block status** — `BlockStatus`: `pending`, `drafted`, `mtqe_passed`, `repair_pending`, `repaired`, `failed`, `needs_human`, `blocked_human`. The block's lifecycle vocabulary; `ubt/core/ir/models.py` is its single authority.

**terminal status** — Membership in `TERMINAL_STATUSES` (`mtqe_passed`, `repaired`, `failed`, `needs_human`, `blocked_human`). One frozenset drives `IRBlock.is_finalized`, the ledger's `NON_TERMINAL_STATUSES`, and repair eligibility, so the three cannot drift. `ubt/core/ir/models.py`.

**job_meta status** — A *separate* vocabulary for the job row: `initialized` -> `completed` / `failed` / `cancelled`, with deliberately no `running` value, so the column alone cannot tell a live job from a dead one. `finalize_job` is fail-closed on `completed` (a missing row or any non-terminal block refuses it) and permissive on `failed`/`cancelled`, with two guards: a late abort never overwrites a `cancelled` or `completed` job, while a real completion may supersede an earlier `cancelled`/`failed`. Outward surfaces normalize before answering. `ubt/core/engine/ledger_mixins.py`.

**ledger** — The per-job SQLite database (WAL mode) that is the single source of run state: job metadata, every block row, batch-API partitions. Resume works because "the ledger already holds blocks, so ingest skips parsing". Schema version is the flattened baseline `SCHEMA_VERSION = 1000` (`TARGET_SCHEMA_VERSION` is kept as an alias). Legacy databases below 1000 are refused unless `--fresh` is passed. `ubt/core/engine/ledger.py`, `ledger_base.py`, `ledger_mixins.py`.

**writer lock** — The job-level exclusive lock lease backed by the ledger SQLite database (`job_locks` table), held for the lifetime of a run and recording owner pid and expiry. The only thing stopping two processes from running the same `job_id` and paying the provider twice; every writer takes it, read-only surfaces never do. `ubt/core/engine/writer_lock.py`.

**checkpoint** — A durable write of one block's progress (`status`, target text, score, flags) into the ledger, singly or batched. Checkpoints are what make resume possible. `ubt/core/engine/ledger_mixins.py`.

**flusher** — `CheckpointBatchFlusher`: a write-behind buffer batching checkpoints into SQLite so high concurrency does not fight over the WAL write lock. A handed-over checkpoint is never silently dropped; after a bounded retry budget it dies loudly rather than let a job report "completed" with paid translations missing. `ubt/core/engine/ledger_flusher.py`.

## Execution

**stage** — One unit of the pipeline. A translating stage is an async generator yielding `TranslationProgressEvent`; the zero-token judgements (extraction witness, preflights, difficulty advisory) are plain coroutines. Stages are ordered by the plan, not by the orchestrator. `ubt/core/engine/stages/`, ordered by `ubt/core/engine/plan.py`.

**StageContext** — The frozen run descriptor every stage receives: inputs (config, router, ledger, manifest, job id, languages, profile), driver callbacks, and pre-decided gates. `ctx.manifest.run` is the one sanctioned mutable channel. `ubt/core/engine/stage_context.py`.

**RunServices** — The collaborators a run builds *for itself* — QE runner, repair loop, FastPass filter, adaptive policy, in-flight semaphore, TM, HTML validator — kept apart from what the run was configured with, and frozen so a stage cannot swap one out from under a later stage. `ubt/core/engine/services.py`.

**RunFacts** — What one stage produces and a later one consumes (`terminology`, `layout`, `render`, `scoring`). A producer cannot return a value (its generator has no return), so it fills the plan-owned fact and the plan hands it on. `ubt/core/engine/facts.py`.

**route (short / long)** — The adaptive execution decision made once per run: short (whole-document rewrite plus reflow, for small born-digital documents) vs long (the staged block pipeline). Forced by `exec_mode`, otherwise probed. `ubt/core/router_mode.py`.

**preset** — A user-facing quality bundle (`publication` / `standard` / `preview` / `fast`) over prompt depth, formula handling, parser and exec mode. It never picks a model, never disables a correctness gate, and never picks a render route. `ubt/core/presets.py`.

**profile** — The domain profile (`general`, `textbook`, `paper`, `fiction`, `humanities`, `semiconductor`, or `auto` to infer from the document). Selects seed glossaries and namespaces the job id; not a closed set, since operators can add glossary resource directories. `ubt/core/memory/seed_glossary.py`, `ubt/core/archetype.py`.

**dual_mode** — The reader-facing mono/dual choice: `dual` pairs each block's source line above its translation, `mono` delivers the target only, `auto` lets the layout advisor pick from the document's structure. The only mode knob users set. `ubt/core/config.py`.

**QE / qe_measured** — Quality estimation: the scoring pass deciding whether a draft is released or repaired (`ubt/core/qe/base.py`). `qe_measured` is the report boolean saying whether *any* block was actually scored; under the default `heuristic` engine nothing is, and the average-QE fields are 0.0 placeholders, so the flag stops an unmeasured run from reading as a measured zero. `ubt/core/engine/reporter.py`, `ubt/cli/commands/translate.py`.

**MTQE** — The project's shorthand for the machine-translation quality estimation score (`mtqe_score` on a block, `MTQEEvaluationError` on failure). The code never expands the acronym; it names the scoring step, not a specific engine. `ubt/core/qe/`.

**fast pass** — The zero-token deterministic screen before any paid scoring: verbatim echo, XML leakage, repetition loops, empty targets and grid damage are rejected outright. `ubt/core/qe/fast_pass.py`.

**repair loop** — The targeted-repair orchestrator that spends expensive models only on degraded blocks: it picks the worst, re-prompts with the annotated defect, and accepts a result only when it passes the same structural and QE gates. `ubt/core/engine/repair_loop.py`.

**triage** — The post-repair stage classifying unresolved blocks by MQM severity and routing them: Critical gets one escalated repair round on the flagship tier, Major goes to the human post-editing queue, Minor auto-passes when the score permits. The fail-open rule for quality verdicts is deliberate — a possibly-wrong translation is still a translation. `ubt/core/engine/stages/triage.py`.

**MQM** — Multidimensional Quality Metrics: the error taxonomy used to grade defects and pick their severity tier. UBT uses its own span annotator and severity ladder, not an external MQM tool. `ubt/core/validators/span_repair.py`, `ubt/core/qe/defect_taxonomy.py`.

**quarantine / BLOCKED_HUMAN / NEEDS_HUMAN** — `blocked_human` means the machine output must not ship (a critical defect such as prompt-scaffold leakage); the block is quarantined and its text stripped. `needs_human` means the draft is shippable but queued for human review, and it keeps its translation. The difference is what the delivery verdict keys on. `ubt/core/ir/models.py`, `ubt/core/qe/defect_taxonomy.py`.

## Memory and terminology

**bible** — The mined book-level term/name memory (`BookBible`, `BibleEntry`): glossary, characters, tone, and the system instruction the draft prompt carries. Mining is deterministic but the LLM backfill is not, so the bible is cached on a resume-stable key. `ubt/core/memory/bible.py`, `ubt/core/engine/stages/bible.py`.

**TM (translation memory)** — The shared cross-job store of accepted translations (`{db_dir}/tm.sqlite`). An exact hit skips the LLM entirely; a near-identical source becomes a terminology-only few-shot reference, and one diverging on a polarity cue is dropped rather than shown. It is the one place a run can affect *other* runs. `ubt/core/memory/tm.py`.

**glossary enforcement** — The deterministic, zero-LLM literal term substitution at export time, guarded by word boundaries, protected spans, compounds and a fail-closed tokenizer. Short-document only by default; the long route opts in via `deterministic_glossary_long_route`. `ubt/core/validators/glossary_enforcer.py`.

**placeholder masking** — Replacing protected spans (email, code, math, soup, citation — in that order) with opaque tokens before a model sees the text, then restoring them in exact reverse order under checksum verification. Order is load-bearing: an earlier family must already be opaque to a later one. `ubt/segment/placeholders.py`.

**sentinel** — A `⟦...⟧` masking token its restore pass did not consume. Sentinels must never reach the typesetter: Typst's math mode mangles them in family-specific ways and the document compiles as one unit, so a single bad token can lose the whole document. `ubt/render/typst/artifact_guard.py`.

## Rendering and delivery

**render skip** — A per-block `(block_id, reason)` entry the adapter records when a block could not be placed, stored as a `render_skip:{reason}` error flag. A fail-closed signal: the block ships its source, and the quality report counts it as a lost translation. `ubt/core/engine/stages/export.py`, `ubt/core/engine/reporter.py`.

**render flag** — A per-block *quality* flag from the render pass (`low_legibility_font`, `formula_degraded`, `sentinel_leaked`). Deliberately never a `render_skip:` flag, because the content does ship — the flag only records that it shipped imperfectly. `ubt/core/engine/stages/export.py`.

**reconstruction (verified / corrupt)** — How a non-text asset was carried: `RECONSTRUCTED` demands a passed round-trip check, `PRESERVED_OPAQUE` means the source region was copied whole and never interpreted, `corrupt` means the structural check found a shattered rebuild (an overfilled table row), and `MISSING` is a hard violation. `ubt/core/content/nodes.py`, `ubt/core/content/contract.py`.

**delivery verdict** — The three-state answer to "may this ship": `blocked` (a quarantined or failed block ships its source), `warning` (a `needs_human` block still ships its translation), `passed`. One rule, shared by the CLI banner, the REST report and the console, so they cannot disagree. `ubt/core/job_options.py`.

**contract (content contract / reconciliation)** — The engine-, language- and layout-agnostic gate asking whether a delivery accounts for every content node. Unaccounted or dropped content is a hard error; an unverified reconstruction is a warning. `ubt/core/content/contract.py`.

**companion artifact** — A second real document written beside the deliverable (the mono/dual sibling, an XLIFF view, a translated table appendix), as opposed to a report. `ubt/core/job_options.py` (`companion_path`).

**sidecar** — A derived report belonging to one deliverable: the JSON and Markdown quality reports, `metrics.json`, `visual_report.json` and `contract.json`, written beside the artifact. Keyed on the whole file name, so two deliverables with the same stem cannot overwrite each other's report. `ubt/core/job_options.py` (`sidecar_path`, `SidecarKind`).

**spill appendix** — The companion written when blocks could not fit their layout boxes: a bilingual table of the overflowing paragraphs, so a space failure is visible to the reader rather than silently dropped. `ubt/core/engine/stages/export.py`.

## Operations

**rehearsal (dry run)** — A zero-token pass through the whole pipeline against a deterministic echo provider, so a book can be validated without a key or spend. It namespaces the job id (its drafts are not translations) and never writes the shared TM. `ubt/core/engine/dry_run.py`.

**job id derivation** — The ledger key derives from the run's full identity: doc id, target language, page selection, chapter window, profile, engine signature, and whether it was a rehearsal. Anything that changes which part of a document a run covers must namespace the ledger, or a narrow run's partial ledger gets resumed by a wider one. `ubt/core/engine/pipeline.py`.

**worker lease** — The job-level lease a queue worker holds and heartbeats while it drains a claim; an expired lease is reclaimed (requeued, or failed after max attempts). It belongs to the durable queue, not the ledger, and losing it is signalled separately from cooperative cancellation so a reclaimed job cannot read as `cancelled`. `ubt/core/engine/job_queue.py`, `ubt/core/engine/job_worker.py`.

**embedded vs queue mode** — The two deployment modes, not two implementations of one job. `embedded` (default) runs each job as an in-process task inside the API; `queue` has the API enqueue into durable SQLite and separate `ubt worker` processes drain it. Both build their orchestrator through one factory. `ubt/api/manager.py`, `ubt/core/engine/job_worker.py`, `ubt/core/engine/job_runner.py`.

**MCP** — The Model Context Protocol surface: a stdio server exposing the engine's operations as `ubt_*` tools, so an agent can drive a translation without an agent loop inside the engine. Needs the optional `mcp` extra; the console-script wrapper turns a missing extra into a clean exit, not a traceback. `ubt/mcp/server.py`, `ubt/mcp/_entrypoint.py`.
