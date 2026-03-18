# Next-Generation Bitcoin Bot Feature Matrix

This matrix translates the current capability review into implementation-ready backlog items so work can proceed in sequence without losing cross-agent ownership.

## How to use

- Treat each item as a delivery track with strict handoff notes.
- Keep roadmap parity with `docs/release_enhancement_plan.md` when statuses change.
- If a track adds settings, update `.env.example` and `README.md` in the same PR.

---

## Track NG-1 — Portfolio risk-budget tripwires

**Problem statement:** Add deterministic portfolio-level kill-switches (daily loss, drawdown, concentration) beyond order-level limits.

**Primary owner:** Core Trading Engine Agent  
**Status:** `not_started`  
**Follow-up owners:** Frontend & UX Agent → QA & Testing Agent

**Changed modules (planned):**
- `app/config.py`, `.env.example`
- `app/core/strategy.py`, `app/core/engine.py`, `app/core/state.py`, `app/core/metrics.py`
- `app/web/router.py`, `app/web/templates/config.html`, `app/web/templates/partials/stats.html`
- `app/utils/notifications.py`, `app/tests/`

**Risk notes:** False positives could halt trading unexpectedly; false negatives could bypass limits.

**Validation target:** deterministic tests proving no order intent passes once a risk budget breach is active.

---

## Track NG-2 — Shadow-live forensic snapshot persistence

**Problem statement:** Persist decision-time market context for every simulated shadow-live order path, and prove no live POST order path can execute.

**Primary owner:** Core Trading Engine Agent  
**Status:** `not_started`  
**Follow-up owners:** Data & Analytics Agent → QA & Testing Agent

**Changed modules (planned):**
- `app/core/strategy.py`, `app/core/engine.py`, `app/core/exchange.py`
- `app/database/models.py`, `app/database/db.py`
- `app/core/metrics.py`, `app/web/templates/partials/stats.html`
- `app/tests/`

**Risk notes:** Snapshot writes must remain async-safe and must not stall hot loops.

**Validation target:** monkeypatched tests assert `create_order` is never called in shadow mode while metrics/snapshots continue to record.

---

## Track NG-3 — Walk-forward and stability scoring

**Problem statement:** Improve robustness by adding out-of-sample walk-forward analysis and stability metrics.

**Primary owner:** Core Trading Engine Agent  
**Status:** `not_started`  
**Follow-up owners:** Data & Analytics Agent → Frontend & UX Agent → QA & Testing Agent

**Changed modules (planned):**
- `app/core/backtest.py`
- `app/web/templates/backtest.html`, `app/web/templates/partials/backtest_results.html`
- `app/utils/export.py`, `app/tests/`

**Risk notes:** Must preserve existing single-pass backtest behavior as default.

**Validation target:** fixture-based tests for rolling splits and reproducible stability metrics.

---

## Track NG-4 — Incident replay timeline

**Problem statement:** Add deterministic incident replay to support reproducible post-mortems.

**Primary owner:** Data & Analytics Agent  
**Status:** `not_started`  
**Follow-up owners:** Frontend & UX Agent → Security & Infrastructure Agent → QA & Testing Agent

**Changed modules (planned):**
- `app/database/models.py`, `app/database/db.py`
- `app/core/strategy.py`, `app/core/engine.py`
- `app/web/router.py`, new `app/web/templates/*replay*`
- `app/tests/`

**Risk notes:** Event storage/indexing design must avoid write contention in execution loops.

**Validation target:** ordering, pagination, and RBAC tests with sensitive-field redaction checks.

---

## Track NG-5 — Pre-live readiness gate

**Problem statement:** Block unsafe paper/shadow to live transitions until operational checks pass.

**Primary owner:** Security & Infrastructure Agent  
**Status:** `not_started`  
**Follow-up owners:** Frontend & UX Agent → QA & Testing Agent

**Changed modules (planned):**
- new readiness module under `app/core/` (or `app/auth/` if access-control coupled)
- `app/core/manager.py`, `app/core/engine.py`
- `app/web/templates/partials/stats.html`, `app/utils/notifications.py`
- `app/tests/`

**Risk notes:** diagnostics must avoid leaking secrets while still being actionable.

**Validation target:** pass/fail transition tests and role-gated override audit checks.

---

## Track NG-6 — Multi-venue execution failover

**Problem statement:** Add execution-layer venue failover to complement existing multi-venue pricing inputs.

**Primary owner:** Core Trading Engine Agent  
**Status:** `not_started`  
**Follow-up owners:** Security & Infrastructure Agent → QA & Testing Agent

**Changed modules (planned):**
- `app/core/exchange.py` (adapter interfaces)
- `app/core/sor.py`, `app/core/strategy.py`, `app/core/engine.py`
- `app/config.py`, `.env.example`, `app/web/templates/config.html`
- `app/tests/`

**Risk notes:** idempotency and partial-fill reconciliation are critical during failover.

**Validation target:** mocked adapter tests for route selection and fallback correctness.

---

## Track NG-7 — High-risk deterministic QA expansion

**Problem statement:** Raise confidence for money/risk paths with focused deterministic tests and coverage targets.

**Primary owner:** QA & Testing Agent  
**Status:** `not_started`  
**Follow-up owners:** Core Trading Engine Agent

**Changed modules (planned):**
- `app/tests/` (new suites for strategy/engine/SOR/risk)
- CI/test tooling config where applicable

**Risk notes:** flaky tests can block delivery and reduce confidence if fixture design is unstable.

**Validation target:** stable mocked integration tests that avoid live internet/exchanges and document reproducible seeds.
