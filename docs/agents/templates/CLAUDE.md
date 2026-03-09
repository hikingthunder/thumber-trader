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
