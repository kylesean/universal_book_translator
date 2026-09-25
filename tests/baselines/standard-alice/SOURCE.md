# standard-alice

Full-pipeline baseline input for this repository.

- File: `standard-alice.epub`
- Source: Standard Ebooks edition of *Alice's Adventures in Wonderland*
- Why this baseline: stable EPUB structure, chaptered prose, recurring entities, and many illustrations — exercises ingest, chunking, translation, entity/glossary consistency, and every final-format emitter.

The pre-UBT "skill" pipeline record (38-chunk run, glossary/entity tables, per-format byte sizes) and its `scripts/convert.py` / `scripts/merge_and_build.py` instructions were removed in the 2026-09-17 doc cleanup: those scripts no longer exist, and the baseline contract is now owned by `tests/baselines/test_baselines.py` and `docs/golden-set.md`. The historical record remains in git history.
