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
- Deploys are driven by pushes to `main` (see [`workflows/docker.yml`](workflows/docker.yml)).

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
ecosystem (§5) bumps both the SHA and the comment.

## 4. Build & deploy — [`workflows/docker.yml`](workflows/docker.yml)

`workflow_run` after CI succeeds on `main` (or manual `workflow_dispatch`) →
build multi-arch image → push to GHCR with provenance + SBOM → Trivy scan →
SARIF to the Security tab → trigger Render deploy hook. Gated on
`docker/Dockerfile.prod` existing, so it no-ops during phase 1. `packages` /
`security-events` / `id-token` write scopes are granted only on the build job.

Required repo secrets when this goes live: `CODECOV_TOKEN` (private repos),
`RENDER_DEPLOY_HOOK_URL`. `GITHUB_TOKEN` is automatic.

## 5. Dependency automation — [`dependabot.yml`](dependabot.yml)

Four ecosystems, all weekly, all opening PRs against **`develop`**:

| Ecosystem | Scans | Notes |
| --- | --- | --- |
| `uv` | `pyproject.toml` + `uv.lock` (`/`) | `allow: dependency-type: all` → direct **and** transitive |
| `docker` | `docker-compose.yml` (`/`) + `docker/Dockerfile.*` (`/docker`) | base images incl. `redis` |
| `github-actions` | `.github/workflows/*` | keeps SHA pins + comments current |
| `devcontainers` | `.devcontainer/devcontainer.json` | feature versions |

- `versioning-strategy: increase-if-necessary` on uv → only raises a
  `pyproject` floor when the current constraint can't satisfy the new version;
  otherwise just refreshes `uv.lock`.
- Minor + patch bumps are **grouped** into one PR per ecosystem; **major**
  bumps arrive as individual PRs so each breaking change is reviewed alone.
- Security updates are grouped separately and ignore the open-PR limit.

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
| `pull_request_template.md` | PR checklist (branch flow, lint/type/test, secrets) |
| `ISSUE_TEMPLATE/` | Issue forms + `config.yml` (blank issues disabled) |
| `workflows/ci.yml` | Lint, type, test, security, PR-source gate |
| `workflows/docker.yml` | Production image build + deploy |
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
- `pre-commit` autoupdate (its hooks aren't covered by Dependabot) — e.g. a
  scheduled `pre-commit autoupdate` PR, or pre-commit.ci.
- Release automation (`release-please` or `changesets`) once `main` cuts tags.
- A `release` GitHub Environment with required reviewers, referenced by the
  deploy job.
