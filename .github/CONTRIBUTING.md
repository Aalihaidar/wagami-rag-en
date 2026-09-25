# Contributing & repository operations — `wagami-rag-en`

Source of truth for **branching, protection rules, CI/CD, and dependency
automation**. Some pieces are applied on GitHub (rulesets) rather than in the
repo. The project overview lives in the root [`README.md`](../README.md).

---

## 1. Branching model

```
feature/*  ─PR─▶  develop  ─PR─▶  main   (release)
```

| Branch      | Role                                   | Direct push? | Accepts PRs from |
| ----------- | -------------------------------------- | ------------ | ---------------- |
| `main`      | Released / deployable history          | ❌ (after bootstrap) | `develop` **only** |
| `develop`   | Integration branch, **default branch** | ❌ (PR only)  | `feature/*`, `fix/*`, `chore/*`, `hotfix/*` |
| `feature/*` | Short-lived work branches              | ✅            | —                |

- **`develop` is the default branch.** Dependabot reads `.github/dependabot.yml`
  from the default branch, and its `target-branch: develop` means every
  dependency PR opens against `develop`, never `main`.
- Deploys are driven by merges to `develop` (staging) and `main` (production)
  — see [`workflows/docker.yml`](workflows/docker.yml) §4. Both branches are
  PR-only (§2), so "merge" is the only way either deploy fires.

### One-time bootstrap sequence

```bash
# 1. First and only direct push to main
git init -b main
git add -A && git commit -m "chore: initial commit"
git remote add origin git@github.com:Aalihaidar/wagami-rag-en.git
git push -u origin main

# 2. Create develop from main and push it
git switch -c develop
git push -u origin develop
```

Then on GitHub:

3. **Settings → General → Default branch** → set to `develop`.
4. **Settings → Rules → Rulesets** → create the two rulesets in §2.
5. **Settings → Code security** → enable *Dependabot alerts*, *Dependabot
   security updates*, and *Private vulnerability reporting*.

From here on: branch off `develop`, open a PR into `develop`; periodically
open a `develop → main` release PR.

---

## 2. GitHub Rulesets (apply in the UI)

Rulesets replace classic branch protection. Create **two**.

### Ruleset: `protect-main`

| Setting | Value |
| --- | --- |
| Target branches | `main` (add by pattern) |
| Enforcement status | **Active** |
| Bypass list | *empty* (not even repo admins — use a PR) |
| **Restrict deletions** | ✅ |
| **Block force pushes** | ✅ |
| **Require linear history** | ✅ if you squash-merge; ❌ if you merge-commit `develop → main` |
| **Require a pull request before merging** | ✅ |
| &nbsp;&nbsp;Required approvals | **1** (or more) |
| &nbsp;&nbsp;Dismiss stale approvals on new commits | ✅ |
| &nbsp;&nbsp;Require review from Code Owners | ✅ (uses [`CODEOWNERS`](CODEOWNERS)) |
| &nbsp;&nbsp;Require approval of the most recent push | ✅ |
| &nbsp;&nbsp;Require conversation resolution | ✅ |
| &nbsp;&nbsp;Allowed merge methods | **Merge** (keeps `develop` history) or **Squash** — pick one |
| **Require status checks to pass** | ✅ |
| &nbsp;&nbsp;Require branches up to date before merging | ✅ |
| &nbsp;&nbsp;Required checks | **`ci-passed`** and **`verify-pr-source`** |
| **Require signed commits** | ✅ recommended (skip if contributors can't sign) |

**"Only `develop` may PR into `main`"** cannot be expressed as a ruleset —
GitHub has no "restrict PR source branch" rule. It is enforced by the
**`verify-pr-source`** job in [`workflows/ci.yml`](workflows/ci.yml), which
fails any PR targeting `main` whose head branch isn't `develop`. Listing it as
a **required status check** above is what makes that enforcement binding.

### Ruleset: `protect-develop`

| Setting | Value |
| --- | --- |
| Target branches | `develop` |
| Enforcement status | **Active** |
| Bypass list | *empty* |
| **Restrict deletions** | ✅ |
| **Block force pushes** | ✅ |
| **Require a pull request before merging** | ✅ (1 approval; Code Owners ✅) |
| &nbsp;&nbsp;Allowed merge methods | **Squash** (tidy integration history) |
| **Require status checks to pass** | ✅ → **`ci-passed`**; require up to date ✅ |

> Solo for now? Keep "Require a pull request" on but set **0** approvals on
> `develop` so you can self-merge, and keep **1** on `main`. Raise later.

---

## 3. CI — [`workflows/ci.yml`](workflows/ci.yml)

Runs on PRs and on pushes to `main` / `develop` / `v*` tags (feature-branch
pushes are covered by the PR trigger — avoids double runs).

| Job | What | Notes |
| --- | --- | --- |
| `verify-pr-source` | Blocks non-`develop` PRs into `main` | Only runs for PRs with base `main` |
| `detect` | Flags whether `app/`, `tests/`, `docker/Dockerfile.prod` exist | Lets later jobs self-activate |
| `lint` | `ruff check` + `ruff format --check` | — |
| `typecheck` | `mypy scripts` (+ `app` once present) | — |
| `test` | `pytest --cov` + Codecov | **skipped** until `tests/` exists |
| `security` | `pip-audit` | — |
| `dockerfile-check` | Trivy config scan | **skipped** until `docker/Dockerfile.prod` exists |
| `ci-passed` | Aggregate gate — the single required check | `skipped` deps count as pass |

Hardening applied: top-level `permissions: contents: read`, per-job
`timeout-minutes`, `concurrency` with PR-only cancellation,
`persist-credentials: false` on checkout, and **every action pinned to a full
commit SHA** with a `# vX.Y.Z` comment. Dependabot's `github-actions`
ecosystem (§5) bumps both the SHA and the comment. Values from the event
(branch names, PR titles) reach `run:` scripts only through `env:`, never as
`${{ }}` inside the script, so a crafted branch name can't inject shell.

Every job in every workflow runs on an **explicit runner image, `ubuntu-26.04`**,
not `ubuntu-latest`, so the OS changes only in a reviewed PR whose CI shows
whether it works, never silently when GitHub moves the `latest` label.
Dependabot does not bump runner labels: moving to the next Ubuntu LTS is a
manual PR (search `.github/workflows/` for `runs-on:`).

## 4. Build & deploy — [`workflows/docker.yml`](workflows/docker.yml)

`workflow_run` after CI succeeds on `main` **or** `develop` (or manual
`workflow_dispatch`) → build one multi-arch image → push to GHCR with
provenance + SBOM → Trivy scan → SARIF to the Security tab → trigger the
matching Render deploy hook. Because both branches are ruleset-protected
(§2 — PR-only, no direct pushes), a green CI run here is always the result
of a merged PR, never a stray push.

| Branch merged into | Image tag | Deploy target | GitHub Environment | Deploy hook secret |
| --- | --- | --- | --- | --- |
| `develop` | `staging`, `sha-<short>` | Render staging service | `staging` | `RENDER_STAGING_DEPLOY_HOOK_URL` |
| `main` | `latest`, `sha-<short>` | Render production service | `production` | `RENDER_DEPLOY_HOOK_URL` |

One `build-and-push` job resolves which branch triggered it (`workflow_run`'s
`head_branch`, or `github.ref_name` for a manual dispatch), tags the image
accordingly, and exposes that as a `target-branch` output; two downstream
jobs (`deploy-staging`, `deploy-production`) each fire only for their branch.
Both Render services are defined in [`render.yaml`](../render.yaml) as
`runtime: image` services — Render's own Blueprint spec is explicit that the
auto-deploy trigger "has no effect for services that deploy a prebuilt
Docker image," so a git push cannot deploy either one no matter how it's
configured. The deploy hook, called only by a job downstream of a green CI
run on the right branch, is the only path in — that's a structural property
of `runtime: image` services, not a setting that could drift back open.
`packages` / `security-events` / `id-token` write scopes are granted only on
the build job.

Required repo secrets when this goes live: `CODECOV_TOKEN` (private repos),
`RENDER_DEPLOY_HOOK_URL` (`production` environment),
`RENDER_STAGING_DEPLOY_HOOK_URL` (`staging` environment). `GITHUB_TOKEN` is
automatic. See
[`docs/APP_AND_DEPLOYMENT_PLAN.md`](../docs/APP_AND_DEPLOYMENT_PLAN.md)
Section 1 for applying `render.yaml` and the rest of the one-time Render
setup this depends on.

## 5. Dependency automation — [`dependabot.yml`](dependabot.yml)

Six ecosystems, all weekly (Monday 04:00 UTC), all opening PRs against **`develop`**:

| Ecosystem | Scans | Notes |
| --- | --- | --- |
| `uv` | `pyproject.toml` + `uv.lock` (`/`) | `allow: dependency-type: all` → direct **and** transitive |
| `docker` | `docker/Dockerfile.*` (`/docker`) | `FROM` lines: the python base image and the `uv` image, both digest-pinned |
| `docker-compose` | `docker-compose.yml` (`/`) | the `redis` service image |
| `github-actions` | `.github/workflows/*` | keeps SHA pins + version comments current |
| `devcontainers` | `.devcontainer/devcontainer.json` | feature versions (+ the lock file) |
| `pre-commit` | `.pre-commit-config.yaml` | hook `rev`s |

- `versioning-strategy: increase-if-necessary` on uv → only raises a
  `pyproject` floor when the current constraint can't satisfy the new version;
  otherwise just refreshes `uv.lock`.
- Minor + patch bumps are **grouped** into one PR per ecosystem; **major**
  bumps arrive as individual PRs so each breaking change is reviewed alone.
- Security updates are grouped separately and ignore the open-PR limit.
- **Cooldown of 7 days** on every ecosystem: a version update is proposed only
  once the release has been public for a week, so a compromised or broken
  release is usually caught upstream first. Security updates are never delayed.
- The `uv` binary is pulled into both Dockerfiles through its own
  `FROM ghcr.io/astral-sh/uv:<version>@sha256:… AS uv` stage rather than an
  image named inside `COPY --from=`, because Dependabot's `docker` ecosystem
  reads `FROM` lines.
- The ruff pre-commit hook's `rev` should match ruff in `uv.lock`: its bump
  (`pre-commit` PR) and ruff's (`uv` group PR) arrive the same Monday, so
  merge them together.
- Every label Dependabot and the issue forms apply is defined in
  [`labels.yml`](labels.yml); GitHub silently drops a label that doesn't exist.
  Labels are managed there, not in the GitHub UI: a push to `develop` that
  changes the file syncs it, and **deletes any label not listed** (a PR that
  changes it gets a dry run showing what would change). Rename with `from_name`
  to keep a label on the issues and PRs that carry it.
- Debian packages inside the images are outside Dependabot; the production
  image runs `apt-get upgrade` at build time so security fixes ship before the
  base-image digest catches up.

> `develop` must exist (and be the default branch) before Dependabot's first
> run, or the `target-branch: develop` configs error.

---

## 6. Files in `.github/`

| Path | Purpose |
| --- | --- |
| `CONTRIBUTING.md` | This document (linked from the "New PR / New issue" pages) |
| `CODEOWNERS` | Auto review-request routing (`@Aalihaidar`) |
| `SECURITY.md` | Private vulnerability reporting policy |
| `dependabot.yml` | Dependency update automation |
| `labels.yml` | The repository's labels as code, synced by `workflows/labels.yml` |
| `pull_request_template.md` | PR checklist (branch flow, lint/type/test, secrets) |
| `ISSUE_TEMPLATE/` | Issue forms + `config.yml` (blank issues disabled) |
| `workflows/ci.yml` | Lint, type, test, security, PR-source gate |
| `workflows/docker.yml` | Production image build + deploy |
| `workflows/labels.yml` | Syncs `labels.yml` to GitHub on `develop`; dry run on PRs |
| `copilot-instructions.md`, `instructions/` | Editor tool config (Mermaid) — not CI |

### Owner / repo slug

Wired to **`Aalihaidar/wagami-rag-en`** in [`CODEOWNERS`](CODEOWNERS),
[`ISSUE_TEMPLATE/config.yml`](ISSUE_TEMPLATE/config.yml), and the bootstrap
`git remote` command above. If you rename the repo or move it under an
organization, update those three places (and switch `CODEOWNERS` to a team).

## 7. Optional next steps (not set up yet)

- SHA-pinned actions are done; keep them current via the `github-actions`
  Dependabot ecosystem.
- `.github/workflows/scorecard.yml` — OpenSSF Scorecard, results to the
  Security tab.
- `step-security/harden-runner` as the first step of each CI job for egress
  auditing.
- `.github/labeler.yml` + `actions/labeler` for path-based PR labels.
- Release automation (`release-please` or `changesets`) once `main` cuts tags.
- The `production` GitHub Environment (§4) can take a required-reviewer rule
  once there's someone other than the sole maintainer to review a prod
  deploy — pointless for a solo project today, but the environment already
  exists as the seam to add it to later without touching the workflow.
