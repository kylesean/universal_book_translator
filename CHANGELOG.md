# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Pre-1.0; tagged `v0.1.0`. Everything below is in the current tree.

### Changed

- **Batch drafting prepares its blocks concurrently.** All three draft paths
  (interactive, Batch API, whole-book) prepared each block's inputs in a
  sequential `await` loop before the batch's LLM `gather`: the TM exact + fuzzy
  lookups, the chunk glossary table, the neighbor tails, the mask, and on a TM
  hit the FastPass screen plus the whole-glossary walk — measured ~2.4 ms of
  pure serialized wall clock per block (72 s per 30k-block book) on a fresh
  run, and the delivery vehicle for the fuzzy-TM scan's cost on a populated
  one. `DraftProcessor.prepare_blocks` now fans the per-block preps out with
  `asyncio.gather` (results keep spine order; the first failure in spine order
  is re-raised after the gather settles, so a batch fails the way the
  sequential loop did). One documented ordering difference: a sibling prep
  that already reached its TM-exact checkpoint enqueue completes even when
  another prep raises — that checkpoint is a terminal-correct
  `MTQE_PASSED`-with-TM-target state a resume would have produced anyway, so
  landing it early is progress, never something the failure path must undo.

- **The length gate counts CJK characters with the module-level compiled
  pattern.** `validate_structural_invariants` rebuilt
  `re.match(f"[{CJK_SCRIPT_CLASS}]", ch)`'s pattern string once per character
  of every scored target — measured 0.571 ms per 2.5 KB target against 0.033 ms
  for `findall` over the existing `_CJK_SCRIPT_RE` (17x), about 65% of the
  whole length-gate evaluate on the hottest per-block path. Same class, same
  count; the pattern is simply no longer rebuilt per character.

- **Spine-ordered full-block reads get a serving index.** `get_all_blocks` and
  `fetch_repair_eligible_blocks` both `ORDER BY spine_index` with no
  status-leading filter any existing index can serve — `EXPLAIN` on a real
  ledger showed `USE TEMP B-TREE FOR ORDER BY` for each. The baseline schema
  gains `idx_blocks_spine (job_id, spine_index)`; fresh databases carry it and
  existing ledgers keep working without it, per the no-in-place-migration
  rule.

- **A provider name may carry a dot, because the model ids it is named after
  do.** The console refused `workbuddy-ai.deepseek-v4.1-flash` by greying out
  Create, so the only clue was a hint that did not say which character was
  illegal. Dots are now accepted at both ends: the charset is
  `[A-Za-z0-9_.-]` in `provider_editor.py` and in the form. A dotted name is not
  a TOML bare key, so tomlkit quotes it — `[providers."a.b"]` — and the loader
  reads back the literal name `a.b` instead of the nested tables `a → b` that an
  unquoted `[providers.a.b]` would silently create. A slash is still refused,
  along with quotes, `]`, `#` and whitespace: the name is the `{name}` of
  `PUT /system/providers/{name}` and a `%2F` is decoded before routing, so a
  slash could never reach the handler, while the rest could close the key.

- **Three-platform reality check, and the two Windows breaks it found.** The
  backend's platform-critical layers were already sound — the single-writer
  job lock is a SQLite lease (the old ``msvcrt``/``fcntl`` file-offset lock was
  abandoned for exactly this reason), the path guard has win32 prefixes and
  casefolded filename parts, no ``uvloop``/``fork``/``shell=True`` anywhere,
  every text-mode ``open`` carries an explicit encoding, and the sidecar/OCR/
  model channels are HTTP or per-platform binaries. The audit found two real
  breaks, both fixed: ``ubt console --dev`` crashed on Windows because
  ``signal.SIGHUP`` does not exist there（the loop now reads it through
  ``getattr``, so SIGTERM — the handler that actually reaps the Vite child —
  still installs）, and the JS toolchain resolved ``node_modules/.bin/vite``,
  a POSIX shell script that ``CreateProcess`` cannot run, instead of the
  ``<tool>.cmd`` batch stub Windows installs beside it（the same stub
  preference makes the auto-build work there）. macOS needed nothing: same
  Unix semantics, and the ``/etc``→``/private/etc`` resolution was already
  handled. What the audit cannot give is runtime evidence — a Windows/macOS
  CI matrix is the honest next step, because these conclusions are read from
  the code, not run on the hardware.

- **The doctor stops measuring what the platform cannot measure.** The
  ledger-permissions probe reads POSIX mode bits; on Windows those bits are
  synthesized — a writable file always reports 0666 — so the scan would flag
  every file as group/other-readable forever, and the printed remedy
  （`chmod -R go-rwx`）would be noise. On win32 the check now SKIPs honestly
  （Windows 通过 NTFS ACL 管理权限，POSIX 权限位在此不可度量）and the
  runtime warning path stays silent; on POSIX the scan is unchanged. The
  security panel's forbidden-path row was hardcoded Unix paths, which on
  Windows described a deny list the server does not have — it now renders
  `GET /system/boundary`, the same resolved prefix tuple the request guard
  compares against（usrmerge duplicates deduped）, so the panel shows what
  the server actually refuses, per platform. Remedies that named one
  distro's packages get cross-platform wording. The path guard itself was
  already cross-platform — win32 prefixes and casefolded filename parts —
  and now the panel agrees with it.

- **The system doctor leads with a verdict.** The page's only real question —
  is anything wrong? — was answerable by scanning fifteen same-sized 正常
  badges to the bottom. Now a verdict band opens the page: a health donut
  （20/22）with a plain headline, and one chip per non-passing check that
  jumps to, expands, and flashes its row. The details follow the same
  discipline the wizard drawer got: a 200-character ledger path clamps to two
  lines behind an 展开 link, and every 修复建议 renders as a copyable command
  chip instead of wrapped amber prose. The right rail goes sticky (the
  wizard's quote panel pattern) because the diagnostics column runs several
  screens tall and the runtime panel left dead paper beside it the whole way;
  the storage facts the security panel used to repeat in terminal mono
  （WAL Mode Active, Hard refusal）are deduped into the probe list, and what
  remains — bind address, auth gate, run mode, allowed roots, forbidden
  paths — speaks in rows and badges beside the egress card it belongs next
  to.

- **The page-egress gate gets a console toggle.** Every egress-dependent
  control (render QC, cloud/VLM OCR) named a fix the reader could not perform:
  `UBT_ALLOW_PAGE_UPLOAD=true` lives in the server's environment, not on any
  screen — the same dead end the QE card existed to remove. Now the gate is
  settable where it is hit: System Doctor gains a 页面外发 card (on/off, with
  what it unlocks, what it costs in privacy, and the runtime scope stated
  plainly), and the wizard's two blocked notes carry an inline 开启页面外发
  action that confirms once — shipping rendered manuscript pages off the
  machine is worth one dialog — then re-reads the runtime so the controls
  unlock without a reload. The write is a runtime override on the config
  source (`PUT/GET /system/egress-config`, `ConfigSource.set_overrides`), not
  a file write: `ubt.toml`'s schema has no key for this field, so the honest
  contract is "in force for this service run, reverts at restart, permanent
  path is the environment variable" — pinned by a test that replays exactly
  that restart. The gate stays fail-closed by default; the toggle writes only
  with the API key, like every other config write.

- **The advanced drawer reads like a form again.** Twelve controls had
  accumulated label + visible help + scope note + placeholder each, and the
  per-book glossary asked the operator to type a server filesystem path — a
  CLI reflex wearing a web form. The drawer is now four paired-column groups
  （读者拿到什么 / 内容与解析 / 范围与试跑 / 执行与质检）; every explanation
  moved into its InfoTip, visible prose is reserved for consequences the
  operator must act on (a refused upload, a closed egress gate), and
  placeholders shrink to a format example（如 1-10）or vanish. Delivery-mode
  options speak in outcomes（智能选择（默认）/ 双语对照 / 仅译文）now that the
  details live one hover away, and the subject hint loses its placeholder
  plus the semiconductor example. The glossary field becomes 本书术语表: a
  staged upload (new `POST /assets/glossary/upload`, 10 MB cap, timestamped
  name in the managed uploads dir) rendered as a removable file chip — the
  engine's own loader reads the staged file back in tests — with the
  always-on standing glossary one link away in Language Assets.

- **The advanced drawer says what each control actually does, and the glossary
  gains an import path.** Five controls were mechanically correct but spoke in
  engine shorthands. The domain dropdown drops「半导体」— a subject sitting in a
  list of genres; its curated seed glossary now applies when auto-detection
  lands on it, and the help says so. The OCR row states its scope once,
  visibly（只影响「图像→文字」这一步；翻译仍由你的模型完成）and its options carry
  consequences（按页计费 / 独立的取文模型 / 跳过取不出文字的页）. The per-book
  glossary field no longer pretends to be the only terminology store: it names
  the Language-Assets glossary as the always-on one and links to it. "Also
  produce the other edition" becomes 附加互补版本 with a help line that names
  the concrete companion PDF for the chosen delivery mode, and the visual
  judge moves out of the delivery group into its own 质检与校验 group under a
  label that says what it inspects（渲染质检·视觉抽查·至多 3 页）. The subject
  hint states its actual mechanism — one terminology sentence injected into
  every draft prompt — and its placeholder stops example-ing semiconductors.
  The asset center's glossary tab gains a file import（.csv/.tsv/.json,
  parsed by the same header vocabulary the engine reads back, strict 422 on
  a malformed payload via the new `POST /assets/glossary/import`）.

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

- **A rehearsal actually runs at "no rate limit" now.** `create_dry_run_orchestrator`'s
  docstring promised *no rate limit* but built the limiter itself:
  `AdaptiveTokenBucket(initial_rpm=100_000, max_rpm=100_000)` raised the RPM side
  and left `initial_tpm` at the class default of 100 000, so every echo call was
  still gated by the TPM bucket — a batch of 30 blocks waited out the refill at
  ~5 s apiece. Measured on a 400-block synthetic book: 39.1 s → 0.88 s with the
  null limiter (~44x), 1 200 blocks: 129.5 s → 2.49 s; profiling attributed 38 of
  the 40 seconds to `epoll` waits, ~2 s to actual CPU. The MCP Agent path had
  already wired `NullRateLimiter` (`agent_translation.py`); the CLI, API and
  worker rehearsal paths — all 20 `create_dry_run_orchestrator` callers — go
  through this one constructor and now match it.

- **The bottom-15% QE average no longer materializes every scored row per
  progress event.** `get_job_stats` computed the bottom-15% mean by fetching
  *all* scored rows into Python and slicing — an O(scored) fetch issued once per
  progress event (the docstring's own count: 666 on a 20k-block run) and once
  per job from the console's catalog snapshot (`get_job_snapshot` →
  `get_job_stats`), which made it the ledger's one O(n²) aggregate. The mean is
  now `AVG` over a `LIMIT cutoff` subquery: `idx_blocks_job_mtqe` already serves
  the score ordering, so SQLite reads only the cutoff's index entries and Python
  materializes none. The cutoff arithmetic (`max(1, int(n * 0.15))`) and the
  row set (`QE_SCORED_SQL`, whose `mtqe_score IS NOT NULL` term makes the old
  post-fetch None filter a no-op) are unchanged, so the value is identical.

- **The console shell is never cached, so a rebuild is visible on the next
  load.** The SPA's `index.html` was served with no `Cache-Control` at all, so
  a browser fell back to heuristic caching and might not revalidate it for
  hours. That one file hid every other change: a rebuilt bundle gets a fresh
  content hash in its name (so no client ever held a stale copy of *it*), but
  the shell that points at the bundle was never re-fetched — which presents
  exactly as "rebuilt the frontend, restarted, hard-refreshed, still the old
  UI". Measured on this machine: a build completed at 07:51 while the served
  `index.html` was from 07:46, and the running server's response carried no
  cache header. The shell now sends `no-cache` (revalidate every load; the ETag
  turns the usual answer into a cheap 304) and the content-hashed files under
  `assets/` send `public, max-age=31536000, immutable` — safe precisely because
  a changed file is a changed URL. The SPA fallback is covered too, so a client
  route cannot cache the shell under its own path. One fast-tier test asserts
  both halves against the real `create_app()`; verified it fails when the policy
  is removed.

- **Rehearsals write to their own store, not the operator's ledger.** A
  `--dry-run` used to write into the same `.ubt/ledgers/` as real work,
  distinguished only by a `_mock` segment in the job id. Nothing prunes a
  ledger, so every rehearsal left a permanent job in the console — seven of
  them after one afternoon of validating the render path — and telling them
  from real work meant reading the job id. The isolation was also only
  *mostly* true: the job-id suffix, the `mock_run` flag and the skip-TM guard
  all work, but they are three mechanisms that must each remember the
  distinction, while the ledger directory never did — so every tool that
  enumerated the store had to re-apply the rule. A rehearsal now runs against
  `<parent>/ledgers-mock`, a **sibling** of the real store (a child would still
  be walked by anything scanning `ledgers/` for `*.sqlite`, which is most of
  the code this protects). Since `uploads`/`outputs`/`tm` all derive from
  `db_dir`, one override moves the rehearsal's ledger, staged uploads, TM and
  deliverables together. Verified end to end: a CLI rehearsal left the real
  directory byte-identical (same file list, 37,504 TM entries untouched) and
  wrote its own ledger and an empty TM under `ledgers-mock`. Four fast-tier
  tests pin the sibling rule, that the caller's config is not mutated, that a
  relative `db_dir` stays relative, and that the three egress closures survive
  the same config copy.

- **Every language the engine accepts now either isolates or says why not.**
  The bundled font set could typeset Chinese, Japanese, Korean, Arabic and
  Hebrew with system fonts sealed; Russian, Greek, Thai and Hindi could not, so
  they kept the "path but not sealed" middle state and their output stayed
  host-dependent. Two of those needed no new bytes at all: the bundled Noto
  Serif measures 100% of assigned Cyrillic (through ext-B, so Ukrainian,
  Bulgarian and Serbian are included) and Greek-extended — the Greek block's
  only gaps are the 14 Coptic letters Unicode files inside it — and they were
  merely absent from the script table. Thai and Devanagari needed their own
  small faces (88 KB and 312 KB, from the `full` build of their releases; the
  `hinted`/`unhinted` builds are subset-per-script and carry 2/95 and 39/95
  ASCII, which would typeset embedded Latin and digits as tofu once isolation
  is on). Two defects surfaced while wiring this up: five languages were
  aliased to another script's face (`bn`/`pa`/`gu`/`my` → Devanagari, `lo` →
  Thai), so a Bengali book asked for a face that renders none of its characters
  — worse than the Latin default it replaced, since DejaVu at least draws the
  digits; and every Latin target isolates, but the default stack named only
  host faces, so with system fonts sealed none of them resolved and Typst fell
  through to its built-in Libertinus while the quality report printed the host
  face it had been asked for (verified by reading the embedded font out of the
  PDF). `Noto Serif` is now in the default, last. Ten languages isolate that
  did not before; `bn`/`pa`/`gu`/`my`/`lo` still cannot, and now say so instead
  of naming the wrong face.

- **Body metrics follow the target's own typesetting convention.** The renderer
  emitted `first-line-indent: 2em` for every target, in both the book preamble
  and the academic template. That is the Chinese rule (首行缩进两格, GB/T
  15834-2011) and it was wrong for two of the three languages this engine
  renders most: Japanese is 一字下げ (one character) and Korean inherits one
  character from 원고지 practice (W3C klreq §4.2), so both were indented twice
  as far as their own tradition. Measured on the rendered PDF before the
  change: zh, ja and ko all came out at exactly 2.00em; after, 2.00 / 1.00 /
  1.00em, with Chinese output unchanged. Nothing owned the value — it was a
  literal in two preamble builders — so no test could catch it.
  `TypographicConvention` is now the owner, one entry per language, and each
  entry names the standard it encodes (GB/T 15834-2011, 一字下げ / JIS X 4051,
  W3C klreq §4.2, Chicago §2.11) so the next reader can check the number
  instead of trusting it. Arabic, Hebrew, Thai, Lao, Myanmar and Khmer keep the
  previous literal and are marked `pre-table` in their source: inventing values
  would make the delivery look researched while being wrong.

- **A styled run no longer swallows what follows it — as a rule, not a list.**
  Typst keeps parsing across a styled run's closing `]`, so whatever comes next
  is read in code mode: `#strong[x](y)` calls the content as a function and
  `#super[...][m].installed` is a field access. Both fail to compile and the
  whole fragment descends to the source. The old guard was a two-entry
  blacklist (`(` and `.` + identifier) that re-emitted the character as
  `#text("…")`. It happened to be correct — an exhaustive sweep of all 32 ASCII
  punctuation characters plus 14 CJK marks found no compile failure and no
  content loss — but only because an undocumented invariant held: that
  `typst_escape` had already backslash-escaped every other follower. Nothing
  asserted that, so a later change to the escaper would have broken rendering
  silently. Every styled run now ends with `/**/` — a Typst comment, which
  terminates the expression with no output, no advance and no glyph. Measured
  against Typst 0.15.1: identical glyph positions and identical extracted text
  versus the old emission, and it also neutralises the followers the blacklist
  never considered (`/`, `-`, `*`, `_`, `[`, `{`, `@`, `<`, `#`, `$`, `` ` ``).
  A new slow-tier test compiles all 32 followers to prove the class is closed.

- **The artifact check locates text instead of guessing which page it is on.**
  The check asked what fraction of a block's *characters* appeared in a page's
  token set within a ±1-page window of its source page. Both halves of that
  were wrong on a reflowing translation. The window: measured on job
  `web_mvnvgck2` (a 92-page paper that renders as 73 in Chinese), 562 of 614
  locatable blocks sat more than one page from their source page — median
  displacement -9, worst -75 — so present text was reported missing. The signal:
  in a CJK book common ideographs appear on nearly every page, so the token
  overlap had almost no discriminative power — 46 of 60 sampled *present* blocks
  also scored ≥0.9 against a page when their text was removed. The check now
  searches each block's page text for a contiguous 16-character fragment, trying
  five distributed offsets and accepting any hit; a matched run of characters
  proves they appear in order on one page, which a lost paragraph cannot
  satisfy. Result on the same job: 894/894 realizations located (was 800/894),
  five false failures gone, and the two blocks that remain unlocated are
  accounted for — one carries a `<mark class="ubt-failed-draft">` marker instead
  of prose, the other is a table whose `|` separators are not drawn. Short
  blocks (headings, `证明。`) still use token overlap, now documented as the
  weaker claim it is. The guard test that a genuinely lost paragraph must be
  caught still passes, so this widened detection rather than loosening it.

- **Chinese deliverables keep their punctuation in the text layer.** A Chinese
  page rendered with system fonts sealed drew `：`, `；` and `？` as CID 0. The
  page looked correct — the punctuation was visible — but the PDF's CIDSet
  declared only CID 0, so those characters were absent from the text layer and
  copy, search and the artifact check all failed on text a reader could see.
  Root cause, measured with Typst 0.15.1: a bundled subset holding ideographs
  plus *both* kana and hangul (33,624 glyphs) triggers it, while ideographs+kana
  (21,799), ideographs+hangul (32,597) and the un-subsetted 65,535-glyph face
  are all clean — so the fault is in the subsetting path, not the font (glyphs,
  cmap, charstrings, FDArray subrs and CID assignment all verified correct). The
  Simplified-Chinese cut no longer carries kana or hangul: they belong to the
  Japanese and Korean faces, which each own their script and were already
  separate files. That removes the trigger and 2.8 MB (serif 9.6 → 6.8 MB, sans
  6.2 → 5.0 MB) with Chinese coverage unchanged (ideographs 100%, CJK
  punctuation 100%, fullwidth forms 93.8%, ASCII 100%). Two regression tests:
  a fast-tier guard reads the shipped files with a stdlib cmap parser (fontTools
  is not a runtime dependency) and fails if either block reappears, and a
  slow-tier test extracts the produced PDF with `pypdf` and asserts no NUL —
  deliberately not with the project's own reader, because pdfium and `textgeom`
  both turn the NUL glyph into whitespace, which is why the defect was invisible
  to the audit that should have caught it.

- **The wizard earns its width on large screens.** The workbench container
  was capped at `max-w-6xl` (1152px) — a reading-page width applied to a
  two-column tool, leaving ~370px of bare paper to the right of the report
  panel on a ~2000px display. The cap moves to 1500px: the surplus feeds the
  form column (~840px, field widths still naturally bounded by the column)
  and the report column (~580px, so the token breakdown stops wrapping), with
  ~120px margins on each side. Reading pages keep their own max-w-5xl/6xl.

- **The disabled-compile hint only speaks when the cause is hidden.** The
  greyed Start Compile carried a standing reason line —「先提供要翻译的文件」—
  which in the empty state restates what the dropzone already says and
  clutters the panel's close. The line is gone; the signal now sits where the
  cause lives: budget and numeric field errors already show inline at their
  fields, and the one invisible case — an invalid field inside the collapsed
  advanced drawer — raises an amber「高级选项有误」marker on the drawer's toggle
  row.

- **The wizard's quote panel keeps up with the form, and the copy speaks
  operator.** The right column — the quote and the compile entry — scrolled
  away with the page: while the operator worked through presets, budget and
  the advanced drawer, the one panel the page exists to inform was off-screen
  (the mobile bar had a sticky summary; desktop had nothing). The column is
  now `lg:sticky`. The step headers（① ② ③）gained a dedicated `.type-step`
  style — as the page's primary structure they must not render identically to
  the advanced drawer's inner group headings. The assess button beside the
  path input is demoted to a re-run icon: auto-assess fires on every
  path/preset change, so a primary-looking button there read as a required
  step. Copy: 「硬预算熔断红线」→「预算上限（超支自动拦截）」, and the header
  subtitle drops the engine voice（底层编译流水线）.

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
