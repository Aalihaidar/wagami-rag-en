<!-- Branch flow: feature/* → develop, and only develop → main. -->

## What & why

<!-- What does this change do, and what problem does it solve? Link issues with "Closes #123". -->

## How to verify

<!-- Commands run, manual steps, screenshots, or sample output. -->

## Checklist

- [ ] Target branch is correct (`feature/* → develop`; `develop → main` only)
- [ ] `uv run ruff check .` and `uv run ruff format --check .` pass
- [ ] `uv run mypy scripts` passes (plus `app` once it exists)
- [ ] Tests added/updated and `uv run pytest` passes (once `tests/` exists)
- [ ] Docs updated if behaviour, schema, or data contracts changed
- [ ] No secrets or keys committed; nothing identifies the real source restaurant chain
