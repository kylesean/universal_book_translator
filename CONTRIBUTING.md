# Contributing

Environment, the gates and their exact commands, the test tiers, commit
conventions, and the two extension points you are most likely to touch.

## Environment

Python 3.12 or newer (`requires-python = ">=3.12"` in `pyproject.toml`). The
project is developed with [uv](https://docs.astral.sh/uv/), and the committed
`uv.lock` is what CI installs against.

```bash
uv sync --extra dev
uv run pre-commit install     # installs both the pre-commit and pre-push hooks
uv run ubt doctor             # self-check: key, ledger dir, render toolchain
```

The `dev` extra is what the gates need: pytest, pytest-xdist, pytest-cov,
pytest-timeout, mypy, ruff, pre-commit, `mcp` (so the type gate can check the
MCP server) and `pypdf` (the independent PDF writer the fixtures use). Other
extras exist and are deliberately absent from the dev environment: `pdf`
(Docling), `ocr` / `academic` (RapidOCR + ONNX runtime), `qe` (CometKiwi).
Install the one you need, e.g. `uv sync --extra dev --extra pdf`; `mypy` is
configured to tolerate their absence, so the default gate stays green without
them.

### The Typst prerequisite

The fast tier's render-contract tests shell out to the external `typst`
compiler, and a PDF render fails without it. CI installs a checksum-pinned
release (currently 0.15.1) plus `fonts-noto-cjk` *before* running the tier, which
is why those tests really execute there instead of silently skipping. Locally,
install it with `brew install typst` (macOS) or `cargo install typst-cli` (any
platform with a Rust toolchain). `ubt doctor` reports whether Typst and a
CJK-capable font are visible and what breaks without them. Tests that need the
compiler skip cleanly when it is missing, so the suite stays runnable — but a
green run without it has not covered the render contract.

## The gates

The local hooks in `.pre-commit-config.yaml` are the inner-loop gate; CI
(`.github/workflows/ci.yml`) repeats them and adds the slow tier. Run everything
by hand with `uv run pre-commit run --all-files`, or the individual commands,
verbatim:

```bash
# pre-commit (on every commit)
uv run ruff check --fix --force-exclude
uv run ruff format --force-exclude

# pre-push (whole tree; pass_filenames is off on purpose)
uv run mypy --strict ubt tests
uv run pytest -m fast -q -n auto --dist=loadscope --cov=ubt --cov-report=term:skip-covered --cov-fail-under=70
uv run python scripts/dump_compiler_edges.py

# CI, additionally
uv run ruff check
uv run ruff format --check
uv run pytest -m slow -q
```

- **Whole-tree, not changed-files.** `mypy`, the fast tier and the layering
  check all run with `pass_filenames: false`, so a push cannot pass just because
  the files you happened to touch are clean.
- **Coverage floor 70%.** A regression guard, not a quality target: it sits about
  3 points under the fast-tier baseline so ordinary work never trips it, but a
  large deletion-without-tests or a new untested module does. Paired with
  `tests/unit/test_no_orphaned_modules.py`, it catches the class of rot that
  lint, types and tests alone let through.
- **The layering invariant.** `scripts/dump_compiler_edges.py` is the acceptance
  tool for the rule in `ubt/core/ports.py`: no module-level `ubt.core` import of
  `ubt.pipeline` / `ubt.segment` / `ubt.translate`, no
  `core.engine <-> pipeline` cycle, no `ubt.render -> ubt.adapters` edge.
  `tests/unit/analyze/test_layering.py` pins the `ubt.analyze -> ubt.adapters`
  direction. See `ARCHITECTURE.md` for the graph.
- **CI repeats a gate that already passes.** It is never the first place a check
  runs. If CI goes red, fix it — do not widen the tolerance.

## Tests

The suite mirrors `ubt/`: `tests/unit/` for pure-logic tests (marked `fast`) and
`tests/integration/` for heavy subprocess and end-to-end tests (marked `slow`).

**Every test must carry `fast` or `slow`.** The default filter is
`-m 'not slow and not legacy_drift'`, but both CI steps pass an explicit `-m`,
and `-m` replaces the `addopts` filter rather than narrowing it. A test with
neither mark is in neither tier and is executed by no gate at all: it still
passes when run by hand, so nothing goes red — it just stops protecting the tree.
`tests/unit/test_marker_hygiene.py` enforces this with a pure-AST check, and also
rejects a test carrying *both* marks (a module-level `pytestmark` is ANDed onto
every test in the file, so a `slow` test inside a `fast`-marked module silently
joins the fast tier).

```bash
uv run pytest -m fast -q              # inner loop
uv run pytest -m slow -q              # heavy subprocess tier
uv run pytest -o addopts="" -q        # everything; also clears --timeout
```

`-o addopts=""` replaces `addopts` wholesale, so it clears both the marker filter
and `--timeout=600`. That is deliberate: it is how the multi-minute live tiers
(local MT, CometKiwi) are meant to be run.

CI installs Typst and runs the slow tier for real: roughly 21 cases must run, 8
skip and 6 parameterized cases collapse to zero. Three groups skip there on
purpose, per the "slow tier" section of `docs/guides/TESTING_STRATEGY.md`:
`test_pdf_engine_differential.py` (6 cases) needs the `pdf` extra, which CI does
not install, so the two-engine differential guard only runs locally;
`test_corpus_acceptance.py` (2 cases) and the parameterized
`test_publish_corpus_sweep.py` (6 cases) need the gitignored real-document
corpus, so they collapse to zero. If those numbers disagree with a real run,
trust the run: `uv run pytest -m slow -q -rs` prints the reasons.

`tests/conftest.py` enforces two project-wide properties so no individual test
has to: outbound socket connects raise unless the test is marked `network`, and
all `UBT_*` environment variables are cleared for the test's scope, so a stray
variable in your shell cannot change config construction. The fast tier also
carries two structural guards beyond the marker check:
`test_no_orphaned_modules.py` (a module nobody imports fails the push) and
`test_reference_resolvability.py` (a comment or docstring pointing at a deleted
file or symbol fails).

## Commits and pull requests

The log uses Conventional-Commit-style prefixes with a scope, then a subject that
reads as a sentence about the change:

```
fix(render): the preamble font stack follows the target language; font_family finally works
feat(delivery): one three-state verdict, the CLI job id, and honest Ctrl-C
refactor(core): delete the SPI string registry; ports resolve by lazy import
test(ledger): the v13->v14 migration keeps its data (W5-2)
```

Prefixes in use: `feat`, `fix`, `refactor`, `test`, `perf`, `chore`, `ci`,
`docs`. A `!` after the scope marks a breaking change.

The subject names the *change*; the body carries the **why** and the evidence —
the measurement, the failing case, the ledger id, or the specific race that was
fixed. Bodies routinely list the individual defects and the regression test that
pins each one. A body that only restates the subject is not the house style.
Commits are made directly on `main`, and there are no merge commits in the
history.

## Documentation

Design notes and guides live in `docs/` in the working tree. **`docs/` is
intentionally unversioned** (`.gitignore` has `/docs/`) because it changes too
often to track: a clone carries the code and tests, not those notes. Code and
tests must therefore treat a missing `docs/` directory as normal, and `docs/` is
excluded from the sdist too. The documents that ship are the root-level files:
`README.md`, `ARCHITECTURE.md`, `GLOSSARY.md`, `CONTRIBUTING.md`,
`CHANGELOG.md`.

`docs/guides/TESTING_STRATEGY.md` is the authority on how tests are written and
run; `docs/guides/CI_AND_QUALITY_GATES.md` is the authority on where the gates
execute. After a refactor that moves code those docs cite, run
`python scripts/check_doc_refs.py`: it reports only a referenced file that does
not exist and a line number past the end of its file, and exits non-zero when it
finds one.

## Extending the system

### Adding a document format adapter

1. Subclass `BaseDocumentAdapter` from `ubt/adapters/base.py`. The two abstract
   methods are `extract_manifest(input_path)` (returns a `BookManifest`) and
   `parse_stream(input_path, pages=None)` (yields `ChapterIR`). For a new
   *format* the plain base is right; `BasePDFEngineAdapter` is for PDF engines
   that carry page geometry.
2. Override `render_blocks` — or, if your render fits the "one sync method"
   shape, just implement `_render_blocks_sync` and the base class dispatches it
   to a worker thread for you.
3. Set `output_suffixes` to the suffixes your render actually writes; the
   pipeline refuses an `--output` that contradicts them before spending a token.
   Set `supports_page_selection = True` only if you honour the 1-based `pages`
   set — a format that ignores it would silently bill the whole book.
4. Prefer building the parse on `ubt/analyze`'s shared readers and walkers
   (`ubt/analyze/reader_*.py`, `ubt/analyze/assemble.py`, the classifier rules in
   `ubt/analyze/structure.py`), so classification and the id space have one
   owner. `ubt.analyze` must not import `ubt.adapters` — that direction is pinned
   by `tests/unit/analyze/test_layering.py`.
5. Register the extension in `ubt/adapters/factory.py`'s `_ADAPTER_REGISTRY`, or
   call the `register_adapter` decorator from your own module for an in-process
   extension.

The adapters under `ubt/adapters/{docx,epub,html,markdown}/` are the models to
copy: each is thin, because the parsing lives in `ubt/analyze`.

### Adding a pipeline stage

1. Write the stage under `ubt/core/engine/stages/`, as an async generator
   yielding `TranslationProgressEvent`. Take `StageContext` plus whatever
   `RunFacts` value or `RunServices` collaborator it reads.
2. If it *produces* a value a later stage consumes, add a dataclass to
   `ubt/core/engine/facts.py` — a generator has no return value, so the hand-off
   has to be a plan-owned fact. If it needs a per-run collaborator, add it to
   `RunServices` in `ubt/core/engine/services.py`.
3. Add it to the plan in `ubt/core/engine/plan.py`, and export it from
   `ubt/core/engine/stages/__init__.py`.
4. If it should be switchable, add a field to `RunGates` and resolve it in
   `ubt/core/engine/pipeline.py` from the config. Do not read `config` inside the
   plan: the plan states *what runs*, the gates state *why*.
5. Extend `tests/unit/test_run_stages.py`, which drives the plan with recording
   fakes and pins the order and the gate behaviour.

If your stage must fail the job, raise a `UBTError` subclass from
`ubt/core/exceptions.py`; the orchestrator's abort handler lands the terminal
ledger write for you.
