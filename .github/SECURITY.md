# Security Policy

## Reporting a vulnerability

Report security issues **privately** using GitHub's
[Report a vulnerability](https://github.com/Aalihaidar/wagami-rag-en/security/advisories/new)
form (repository **Security → Advisories → Report a vulnerability**).

Please do **not** open a public issue or PR for anything security-sensitive.
We aim to acknowledge a report within 3 business days and to agree a
disclosure timeline with you.

Enable **Settings → Code security → Private vulnerability reporting** on the
repository so the form above is available.

## Supported versions

Pre-release (phase 1). Only the `main` branch is supported; there are no
tagged releases yet.

| Version | Supported |
| ------- | --------- |
| `main`  | ✅        |
| other   | ❌        |

## Scope notes

- `data/` is a **rebranded demo corpus**. The presence of that data is not a
  vulnerability.
- Never attach `.env` contents, real API keys, or Weaviate/LLM credentials to
  a report.
- Dependency and container-image CVEs are tracked automatically by Dependabot
  and Trivy; you don't need to file those unless you have a working exploit
  path specific to this project.
