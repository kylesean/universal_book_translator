# Delivery-contract verification corpus

This corpus is the **real-document gate**: it asserts the delivery *contract* —
the two axioms — on actual PDFs, and `ubt verify --corpus` fails the build when
a document loses text or assets. It complements (not replaces) the automated
suite in `tests/`, which pins the per-module contracts on synthetic fixtures
(`docs/guides/TESTING_STRATEGY.md`). Consumers: the `ubt verify` CLI, the
slow-tier corpus harnesses (`scripts/shadow_*.py` via
`tests/integration/test_corpus_acceptance.py`), and manual acceptance runs.

> The contract is engine-, language- and layout-agnostic. A regression on any
> render path shows up the same way: as an unbalanced book.

## Layout

```
corpus/
  cases.json          # the case manifest (checked in)
  documents/          # real source documents + delivered artifacts (NOT checked in)
```

Large/real documents are deliberately **not** committed (size + licensing).
Populate `corpus/documents/` locally. `cases.json` references them by relative
path; a case whose file is absent is reported `skip` (or `fail` under
`--require-all`).

## Running

```bash
# Translate every case's source document (dry-run, no API key) and verify the
# contract each run writes. This is the full gate: the pipeline itself is the
# ground truth, not a hand-written fixture. The slow test tier runs exactly
# this command automatically (pytest -m slow), skip when documents are absent.
uv run ubt verify --corpus corpus --run --require-all

# Verify only what is already on disk (no translation).
uv run ubt verify --corpus corpus

# Verify one delivered artifact (reads its <stem>_<tag>_contract.json).
uv run ubt verify /path/to/book_mono.pdf

# Re-derive the contract from a finished job's ledger (independent cross-check).
uv run ubt verify --job job_<docid>_zh --engine rigid
```

Local run for the checked-in cases (source PDFs are not committed; drop the
matching files into `corpus/documents/`):

```bash
cp ~/Downloads/2609.20519v1.pdf ~/Downloads/2608.25512v1.pdf \
   ~/Downloads/2609.22978v1.pdf ~/Downloads/chapter-3.pdf corpus/documents/
uv run ubt verify --corpus corpus --run --require-all
```

## Thresholds and CI coverage

The `expect` floors are **regression floors, not targets**: each is set a few
points below the current dry-run baseline so a real regression trips it while
normal parser/renderer churn does not. Measured with the dry-run (echo) provider
on the reference corpus:

| case | `min_delivered_ratio` | observed (dry run) |
| --- | --- | --- |
| `twocol-paper-2609` | 0.55 | 0.601 |
| `paper-2608` | 0.57 | 0.621 |
| `paper-2609-22978` | 0.66 | 0.759 |
| `book-chapter-3` | 0.60 | 0.639 |

When a real engine change moves a case, re-derive its floor and keep a margin
larger than the run-to-run noise rather than tightening the floor onto the
observed value.

`max_source_kept` is not zero on the two maths-heavy cases, and the reason is
worth stating so it is not mistaken for slack. The dry-run (echo) provider
returns `"[模拟翻译] " + source`, ~35% wider than the source line, and the
extractor cuts a line that runs into a display formula into narrow fragments
(`Proof.` in a 25pt box, `of` in a 12pt box). A fragment box that size cannot
hold its own echoed translation at any size the compositor will draw (the floor
is 4pt; below it the element keeps its source). Most such fragments are drawn
below 6pt and flagged `low_legibility_font`; the rest stay source. Baselines:
`paper-2608` 8, `twocol-paper-2609` 1, the other two 0. A real translation of
those fragments is shorter than the echo, so this ceiling is an artifact of the
mock, not of the delivered document -- do not read a nonzero value here as lost
text. Re-derive after a renderer change, as with the ratio floors.

A *second*, unrelated mock artifact accounts for the 1 on `paper-2609-22978` and
`book-chapter-3`: a block the QE pass quarantines (`blocked_human`, which the
contract counts as source-kept). The echo is `"[模拟翻译] " + source` and cannot
emit `$...$`, so every block whose source carries a math symbol trips the
*undelimited math* rule -- `source carries math symbols but the target has no
$...$ span` -- on every rehearsal, and a critical flag surviving escalated repair
quarantines the block. Measured: one each (`pdf_main#b0222`, a bibliography entry
carrying `β0`; `pdf_main#b0169`, a paragraph carrying `Fth`/`Ag`). A real
translation carries the LaTeX and never raises this flag, so the allowance is
scoped to the mock and stays at 1 -- a real quarantine anywhere else still trips
the gate at 0. This is the same kind of allowance as the fragment above: it is
not slack for lost content, and it does not make the quarantine exempt from the
contract (a quarantined block is source-kept *by design*, see
`ubt/core/content/adapt.py`).

`max_errors` follows the same fragment: a *space* failure -- source kept though
the translation fits nowhere, ``render:no_fit`` -- is an ERROR (Axiom B), so a
fragment the mock echo cannot fit contributes to **both** `max_source_kept` and
`max_errors`. `twocol-paper-2609` therefore allows 1 error (its one such
fragment); elsewhere the fragment stays a warning (``render:no fragment``, a
typesetter defect, not a space failure) and `max_errors` stays ``0``. The
allowance is scoped to the same mock artifact as `max_source_kept`: it is not
slack for lost content, and a real space failure anywhere else still trips the
gate at 0.

Two deliberate exclusions from the automated CI gate:

- **CI does not run this gate.** The GitHub `gate` job runs `pytest -m fast`;
  this corpus is in the `slow` tier and its documents are not committed, so CI
  skips it (`_skip_without_corpus`). It runs locally, and in any CI that has the
  corpus checked out or mounted, via `pytest -m slow` or
  `ubt verify --corpus corpus --run --require-all`. The `fast`-tier ingest
  regression net that does run in CI is
  `tests/unit/adapters/test_pdf_adapter_e2e.py` (synthetic PDFs, no toolchain).
- **The `shadow_reader` coverage floor is relaxed to 0.97** in
  `tests/integration/test_corpus_acceptance.py` (`--min-coverage 0.97`) even
  though the script defaults to 0.98: the scanned-heavy `book-chapter-3` tops out
  at ~0.973 while still round-tripping losslessly. The relaxation is pinned next
  to the case in the harness, not here, so it moves with the harness.

## Case schema (`cases.json`)

```jsonc
{
  "schema_version": 1,
  "cases": [
    {
      "id": "twocol-paper-2609",
      "description": "Two-column paper: figures/tables must survive.",
      // One of:
      "document": "documents/2609.20519v1.pdf", // source to translate (needs --run)
      // "artifact": "documents/book_mono.pdf", // an already-delivered artifact
      // "job": "job_<docid>_zh",               // a finished job's ledger
      "expect": {
        "max_errors": 0,            // ERROR-severity violations allowed (default 0)
        "max_missing_assets": 0,    // lost non-text nodes allowed
        "max_source_kept": 5,       // text nodes allowed to ship source
        "min_accounted_ratio": 0.9, // (translated + verbatim) / total_text floor
        "min_delivered_ratio": 0.6  // translated / total_text floor (stricter)
      }
    }
  ]
}
```

Any key omitted is not asserted. `max_errors` defaults to `0`, so a contract
with an unaccounted text node or a missing asset fails even with no `expect`.

## Why these cases

| Case | Axis it pins |
|---|---|
| `twocol-paper-2609` | multi-column + figures/tables → rigid; no lost assets |
| `paper-2608` | figures/tables survive across a second document |
| `paper-2609-22978` | math-dense paper; tables reconstructed+verified or preserved whole |
| `book-chapter-3` | figure-heavy prose chapter; reflow delivers text, figures preserved |

Add a case whenever a bug is found; the case is the regression test. Prefer a
`document` case (the gate runs the pipeline) over a frozen `artifact` one, so the
case keeps probing the current engine rather than yesterday's output.

---

## Golden baseline: did this change make the document *worse*?

The contract answers "is this document complete". It cannot answer "is it worse
than the version we accepted", because the numbers are recomputed every run and
nothing is frozen. A fix that repairs one defect while regressing three others
passes every existing gate.

`scripts/golden_baseline.py` closes that gap over the same `cases.json`. It has
two sets of layers, because a baseline artifact is either **reproducible** or it
is not, and the difference decides what can honestly be compared.

**Content invariants** — these gate every case:

| layer | what it compares | catches |
| --- | --- | --- |
| contract | `violations`, `missing_assets` | content lost outright |
| structure | page count, delivered blocks, asset totals | the renderer stopped placing things |
| digits | reproduction of the baseline's digit characters | a dropped or altered quantity |
| number runs | reproduction of digit runs (`93`, `2024`) | a value that changed, not a moved separator |
| identifiers | reproduction of Latin terms / model names | a lost table row, a renamed framework |
| citations | reproduction of `[n]` / `[n, m]` | a swallowed bibliography entry |
| CJK share | character share, ±5 points | a page that came out untranslated |

**Byte-level layers** — these gate only cases marked `"deterministic": true`
(e.g. a `--dry-run` echo, whose render really is reproducible):

| layer | what it compares | catches |
| --- | --- | --- |
| visible text | the PDF's own text layer, word sequence | same counters, different words |
| visual | SSIM over 3 sampled pages | same words, different layout |

### Why the split exists (measured)

The byte-level layers were originally applied to every case. On a document
translated by a real model they do not measure the code — they measure sampling.
Two runs of **identical** code over arXiv 2606.23344 produced:

| comparison | result |
| --- | --- |
| word-sequence similarity, run A vs run B | **0.83** |
| word-sequence similarity, run vs frozen baseline | 0.85 (limit 0.98) |
| SSIM on the 3 sampled pages, run A vs run B | **0.23 / 0.64 / 0.23** (limit 0.985) |

So the word layer flagged every run, and a 20% text budget that would have
stopped flagging also would have stopped detecting anything. The same two runs
agreed on the content to within sampling noise:

| invariant | run A vs run B | noise |
| --- | --- | --- |
| citations | 1.0000 | 0 |
| numbers | 0.9987 | 0.13% |
| identifiers | 0.9912 | 0.88% |

Those are the layers that can gate a model-translated document, and they are what
the corpus actually exists to protect: a dropped paragraph, a lost table, a
swallowed citation and an untranslated page all move them, while a model
re-wording a sentence does not.

One trap is worth calling out, because it produced a false "number loss" twice.
The PDF text layer joins adjacent table cells with no separator, and *whether it
does* depends on the column widths the translation produces — not on the
document's content. Two runs of the same code rendered this paper's table as
`0.082 90.30` and as `0.08290.30`; a token-keyed count called the second a 4.4%
loss. The gate therefore counts **digit characters** (invariant under fusion:
2385/2385 matched) and **digit runs** (invariant under a moved decimal point),
not numeric tokens.

Verified by mutation against the real artifact: removing every citation fails
(`citations lost 100.0%`), removing every multi-digit number fails
(`digits lost 35.7%`), altering a single 3-digit value fails
(`digits lost 3.6%`), dropping half the document fails on five layers at once,
and replacing 的 with 之 throughout passes.


### Freeze a baseline

Review the artifacts first — the baseline accepts whatever it is given.

```bash
uv run python scripts/golden_baseline.py --accept --artifacts ~/Documents/UBT
```

Writes `corpus/baseline/<case-id>/baseline.json` (structure + text + base64 page
rasters). These are **checked in**: they are small (a few MB) and they *are* the
accepted state.

### Check against it

```bash
uv run python scripts/golden_baseline.py --check --artifacts ~/Documents/UBT
uv run python scripts/golden_baseline.py --check --artifacts ~/Documents/UBT --cases paper-2608
```

Exit code 1 on any regression. Typical run over four documents is ~40s
(the sampled pages are rasterised at 110 DPI).

### Reading the report

```
[FAIL]   twocol-paper-2609
           - page count dropped 15 -> 14
           - visible text shrank 129171 -> 127424 chars (-1.4%, limit 2%)
```

### Why the page count is its own ratchet

One missing page of a 15-page paper is only 1.35% of its characters, which sat
under the text-share limit and passed in testing. A lost page is an absolute
signal and is checked as one.

### Why an asset *mix* change is allowed

A repair that turns a source crop into real math moves one asset from
`preserved_assets` to `reconstructed_assets` with the total unchanged. That is
the goal, not a regression, so only the **total** and the reconstructed
**coverage** are ratchets.

### When to re-freeze

Re-run `--accept` only after reviewing the new artifacts by eye. A re-freeze
without review turns the gate into a rubber stamp, which is worse than no gate.

### Not in pre-push

The baseline needs delivered PDFs, which are not committed. Run it manually
after a change that touches extraction or rendering, and before publishing a
deliverable.

### The case that found the gate defects

`twocol-paper-2606` (arXiv 2606.23344) is not just another paper: every gate
false positive and both semantic losses fixed in the 2026-10 audit were found on
it. It is in the corpus so that a change to a gate — or to the extractor feeding
it — is measured against the document that exposed the problem, not only against
documents that already passed.
