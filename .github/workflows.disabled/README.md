# Disabled workflows

Workflows parked here are **inert**: GitHub only loads `.github/workflows/*.yml`,
so nothing in this directory runs. They exist so the staged rollout is ready to
activate without re-deriving it — see
[`docs/guides/CI_AND_QUALITY_GATES.md`](../../docs/guides/CI_AND_QUALITY_GATES.md).

To activate one, move it into the live path:

```bash
git mv .github/workflows.disabled/ci.yml .github/workflows/ci.yml
```

Activate only after `uv run pre-commit run --all-files` is green on `main`.
