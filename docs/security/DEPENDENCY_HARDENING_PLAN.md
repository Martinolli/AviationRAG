# Dependency Hardening Plan

Date: 2026-05-14  
Branch: `security/dependency-hardening`  
Scope: Security Sprint S.1 dependency audit baseline and safe fixes only

## 1. Purpose

This document records the npm vulnerability baseline, the result of the safe non-breaking audit fix, and the remaining major-upgrade work. It is intentionally separate from ingestion, retrieval, manifest, bridge, embedding, Astra, FAISS, and prompt work.

## 2. Baseline Vulnerability Summary

Baseline command:

```powershell
npm audit
npm audit --json > docs/security/npm-audit-baseline.json
```

Baseline counts:

| Severity | Count |
| --- | ---: |
| Critical | 2 |
| High | 21 |
| Moderate | 12 |
| Low | 4 |
| Total | 39 |

Main affected package groups:

1. `next` and `postcss`
2. `@langchain/community`, `@langchain/core`, `@langchain/openai`, `langchain`, and `langsmith`
3. Transitive HTTP/form packages including `axios`, `form-data`, and `follow-redirects`
4. Vercel/dev tooling transitive packages including `vercel`, `@vercel/*`, `undici`, `tar`, `semver`, `path-to-regexp`, `esbuild`, `ajv`, and `debug`

Several baseline findings reported fixes that require `npm audit fix --force` and major package upgrades.

## 3. Safe Fixes Applied

Safe fix command:

```powershell
npm audit fix
```

The safe fix updated `package-lock.json` only. `package.json` was not changed.

Notable safe-fix effects:

1. Critical vulnerabilities were reduced from 2 to 0.
2. Total vulnerabilities were reduced from 39 to 27.
3. Lockfile entries for packages such as `axios`, `form-data`, `follow-redirects`, `brace-expansion`, `diff`, `js-yaml`, `minimatch`, `picomatch`, `qs`, and `yaml` were updated or removed from the active vulnerable set.
4. No `npm audit fix --force` command was run.

## 4. Remaining Vulnerabilities After Safe Fix

Post-fix report:

```powershell
npm audit
npm audit --json > docs/security/npm-audit-after-safe-fix.json
```

Post-fix counts:

| Severity | Count |
| --- | ---: |
| Critical | 0 |
| High | 18 |
| Moderate | 7 |
| Low | 2 |
| Total | 27 |

Remaining affected packages reported by npm audit:

1. `next` and `postcss`
2. `@langchain/community`, `@langchain/core`, `@langchain/openai`, `@langchain/textsplitters`, `langchain`, and `langsmith`
3. `vercel` and transitive `@vercel/*` packages
4. Transitive packages including `@tootallnate/once`, `@mapbox/node-pre-gyp`, `ajv`, `debug`, `esbuild`, `path-to-regexp`, `semver`, `tar`, and `undici`

## 5. Packages Requiring Major Upgrades

The npm audit output reports major or breaking upgrade paths for:

| Package group | Reported upgrade direction | Notes |
| --- | --- | --- |
| Next.js | `next@16.2.6` | Major upgrade from the current Next.js 14 line; requires framework compatibility testing. |
| LangChain JS | `@langchain/community@1.1.28`, `@langchain/core@1.1.46`, `@langchain/openai@1.4.5`, `langchain@1.4.0` | Major API changes likely; must be handled separately from RAG behavior changes. |
| PostCSS via Next.js | `next@16.2.6` | Tied to Next.js major upgrade path. |

Some remaining Vercel/dev-tooling transitive findings still show `npm audit fix` availability, but another safe audit fix after the first pass did not clear them. Treat the remaining Vercel toolchain cluster as requiring manual dependency review rather than repeated blind audit fix loops.

## 6. Risk Classification

| Risk area | Current risk | Rationale |
| --- | --- | --- |
| Runtime web framework | High | `next` remains in the audit report with high severity findings. |
| RAG support libraries | High | LangChain and LangSmith findings include serialization, SSRF, SQL injection, redaction, and prototype pollution classes. |
| Developer/deployment tooling | Moderate to high | Vercel and build-tool transitive dependencies remain vulnerable, mostly affecting tooling paths. |
| Critical exposure | Reduced | Safe fix reduced critical findings to zero according to npm audit. |
| Regression risk from remaining fixes | High | The remaining clean fix paths require major upgrades and compatibility testing. |

## 7. Recommended Upgrade Sequence

Major upgrades should be handled in separate small branches or commits and must not be mixed with ingestion, retrieval, embedding, manifest, or prompt changes.

Recommended sequence:

1. Upgrade the Next.js family first.
2. Confirm React and React DOM compatibility if Next.js requires it.
3. Upgrade LangChain packages as a separate branch.
4. Review Vercel and Playwright/dev tooling after framework upgrades.
5. Re-run `npm audit`, `npm run build`, and `npm run test:smoke` after each group.

## 8. Validation Plan

For each future dependency group:

1. Run `npm audit` before and after the upgrade.
2. Save audit JSON if the vulnerability profile changes materially.
3. Run `npm run sanitize:check:all`.
4. Run `npm run build`.
5. Run `npm run test:smoke`.
6. Run the lightweight Python compile/model/manifest tests to catch accidental cross-stack disruption.
7. Verify no runtime data, generated embeddings, `.env` files, or `data/manifest/documents.jsonl` are staged.

## 9. Rollback Plan

1. Keep each major upgrade isolated on its own branch.
2. If build or smoke tests fail, revert the dependency group before attempting another package group.
3. Restore the prior `package-lock.json` for lockfile-only regressions.
4. Do not change application behavior to compensate for dependency upgrades unless that behavior change is explicitly scoped and reviewed.
5. Keep RAG/ingestion/manifest changes out of dependency hardening commits.

## 10. Open Questions

1. Should Next.js move directly to 16.x, or should the project first test the latest patched 14.x or 15.x line if available?
2. Is the `node` npm package needed as an application dependency, or should runtime Node version be managed outside `package.json`?
3. Which LangChain JS imports are actively used by the runtime versus legacy experiments?
4. Should Vercel CLI remain a dev dependency, or should deployments rely on GitHub/Vercel integration without local CLI packaging?
5. Should CI fail on any high npm audit finding, or should major-upgrade exceptions be tracked until compatibility work is complete?
