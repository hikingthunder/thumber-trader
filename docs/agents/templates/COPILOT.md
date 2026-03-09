# COPILOT.md (Agent Template)

## Purpose
Template for Copilot-driven tasks to align with project standards.

## Must follow
- AGENTS.md ownership boundaries
- No credential leakage in commits
- Deterministic tests over live calls

## Deliverable format
1. What changed
2. Why it changed
3. How to validate
4. Known limitations

## Pre-merge checks
- [ ] Updated docs
- [ ] Updated config examples
- [ ] No sensitive files tracked

## Release enhancement parity
- Align task proposals and implementation sequencing with `docs/release_enhancement_plan.md`.
- When adding one of the roadmap features, include explicit handoff notes for next owner (Core → Frontend → Data → QA → SecOps as applicable).
- Keep README references in sync when roadmap items are added, removed, or re-prioritized.
