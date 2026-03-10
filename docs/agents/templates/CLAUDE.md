# CLAUDE.md (Agent Template)

## Working agreement
- Keep diffs focused and auditable.
- Prefer explicit, safe defaults.
- Explain tradeoffs briefly in PR notes.

## Task intake checklist
- Intended behavior:
- Non-goals:
- Affected modules:
- Risk level:

## Implementation checklist
- [ ] Confirm ownership against AGENTS.md
- [ ] Keep sensitive values out of logs/docs
- [ ] Add/update tests for changed logic
- [ ] Update operator docs if runtime behavior changes

## QA checklist
- [ ] Happy-path validated
- [ ] Failure-path validated
- [ ] Backward compatibility considered

## Release enhancement parity
- Align task proposals and implementation sequencing with `docs/release_enhancement_plan.md`.
- When adding one of the roadmap features, include explicit handoff notes for next owner (Core → Frontend → Data → QA → SecOps as applicable).
- Keep README references in sync when roadmap items are added, removed, or re-prioritized.
- Update roadmap tracking fields (status/owner/milestone/PR links/last-updated) when delivering roadmap features.
