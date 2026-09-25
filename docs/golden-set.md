# Golden Set (baseline corpora & KPI goldens)

The baseline contract for this repository. When
`tests/baselines/standard-alice/SOURCE.md` says "the baseline contract is now
owned by `tests/baselines/test_baselines.py` and `docs/golden-set.md`", this is
the second half of that sentence: what the golden set *is*, what a red run means,
and how to change it without laundering a regression into the new baseline.

This file owns both halves: the inventory and the regeneration runbook (§1–§4),
and the tier-level rules that make a red baseline readable (§5).

## 1. What is in the set

| Corpus (under `tests/baselines/`) | Shape it exercises | Golden |
| --- | --- | --- |
| `cognitive-psychology-ch03/` | textbook multi-flow: math, code, sidebars | `metrics.golden.json` |
| `dual-column-paper/` | academic two-column: LaTeX equations, algorithms, tables | `metrics.golden.json` |
| `call-of-the-wild/` | long-form prose with streaming chapter cursor | `metrics.golden.json` |
| `standard-alice/` | Standard Ebooks EPUB, 20 chapters / 45 illustrations | `metrics.golden.json` |
| `rigid/` | rigid-engine plan golden (the `rigid` render path's committed plan) | `plan.golden.json` |

Every corpus directory carries a `SOURCE.md` provenance note (origin, why it is
in the set, what was removed in the doc cleanups and where to find it in git
history).

## 2. What makes a run comparable

Baseline runs are driven by `TokenEchoMockProvider` (`tests/mock_providers.py`)
plus `MockQERunner` — the production-side double defined in
`ubt/core/qe/comet_runner.py`, **not** a live model — so a KPI delta means the
*pipeline* changed, never that a model had a different day. Consequences:

- Goldens are byte-stable across machines; a diff is reviewable evidence.
- The double owes a structural fidelity contract (keeps tables as tables,
  keeps math/identifiers/numbers, stays in the length band), checked in
  milliseconds by `tests/unit/test_mock_provider_contract.py`.
- If a baseline goes red while that contract test stays green, suspect the
  pipeline first.

## 3. Running and regenerating

```bash
# Whole baselines tier (deterministic, mock-driven)
uv run pytest tests/baselines -o addopts="" -q

# One corpus, deliberately: a whole-suite rewrite produces a diff nobody can audit
UBT_UPDATE_GOLDENS=1 uv run pytest \
    "tests/baselines/test_baselines.py::test_baseline_dual_column_paper_end_to_end" \
    -o addopts=""
git diff tests/baselines     # only the goldens you meant to change may move
```

Rules that make a regeneration honest (KPI added ⇒ re-record in the same change
under `strict_names`; review the diff; gate ratios not counts) are owned by §5 —
read them there; do not restate them here.

Measured 2026-09-25: the full baselines tier is 6 tests / 2.39 s wall (slowest
case 1.29 s) and spawns no subprocess, so it carries no `slow` marker and runs
inside the default inner loop. `slow` is reserved for the heavy-subprocess tiers
(typst/pandoc/pdftoppm) and the live-model boundary.

## 4. `standard-alice`: how a false marker left a corpus undefended

Closed 2026-09-23: the gate is wired in `test_baseline_standard_alice_epub_e2e`
and `metrics.golden.json` was recorded in `8bef284`. The post-mortem is the
reason this section exists. The corpus carried `@pytest.mark.network`, and the
comment beside the missing gate said a golden "cannot be regenerated offline".
Neither was true — the EPUB is checked in, the run uses the offline
`TokenEchoMockProvider`, and it finishes in 1.27 s like the other three. A label
inherited from an era when it did fetch something became the standing excuse for
running the biggest corpus in the set with no KPI gate at all for a week.

Lesson for any corpus that looks ungateable: measure what it actually does
before accepting the marker.

```bash
uv run pytest tests/baselines -m network --collect-only -q
# no tests collected (6 deselected) in 0.03s   <- nothing in the tier needs network
uv run pytest tests/baselines -o addopts="" -q --durations=3
# 6 passed in 2.39s                            <- and it is cheap enough to gate on
```

Its `SOURCE.md` also records what was removed in the 2026-09-17 doc cleanup
(the pre-UBT "skill" pipeline record and its `scripts/convert.py` /
`scripts/merge_and_build.py` instructions) — those scripts no longer exist; the
historical record remains in git history (`git log --follow
tests/baselines/standard-alice/SOURCE.md`).

## 5. Tier-level golden rules

Three rules make a red baseline readable:

- **The double owes a fidelity contract.** It must behave like a competent
  model: keep markdown tables as tables of the same grid shape, keep math,
  identifiers, glossary terms and numbers, and stay inside the length band. Its
  own predicates are the pipeline's (`grid_columns`,
  `target_missing_math_delimiters`, `identifier_terms`), and
  `tests/unit/test_mock_provider_contract.py` checks that in milliseconds.
  Therefore a baseline failure means the *pipeline* changed — if the double were
  at fault, the contract test would be red first.
- **Goldens gate ratios and invariants, not counts.** Absolute block counts only
  belong in a test when they are an ingestion-shape fact (`total_blocks == 408`).
  A count of quarantined or repaired blocks is a snapshot of a policy and goes
  stale silently; assert the partition (`completed + blocked_human == total`)
  and name what you expect instead.
- **Adding a KPI means re-recording.** The gate compares with `strict_names`, so
  a metric the golden does not carry is a failure rather than a silent skip, and
  `check_thresholds` runs against the golden too — a re-record cannot launder a
  defect into the new baseline. `test_goldens_satisfy_absolute_bounds` in the
  fast tier enforces both against the checked-in artifacts, so an out-of-date
  golden goes red in seconds, not on the next slow run.

Review the diff before committing regenerated files: a golden that shifts as a
side effect of an unrelated change is a regression signal.

## 6. Related documents

- [AGENTS.md](../AGENTS.md) — the repository's testing constraints, which take precedence over any other testing doc.
- [evaluation-and-comparison-guide.md](evaluation-and-comparison-guide.md) — live/real-model evaluation (the non-golden half).
- [knob-calibration-protocol.md](knob-calibration-protocol.md) — how knob changes must re-measure against these baselines.
