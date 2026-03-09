# Release Enhancement Plan (Post Paper-Trading Validation)

This plan converts the recommended improvements into a staged delivery roadmap.

## Priority 1 — Pre-live Safety & Control

### 1) Shadow Live Mode (real market data + simulated execution)
**Outcome:** Validate slippage/latency/fill-quality assumptions without risking funds.

**Scope:**
- `app/core/engine.py`, `app/core/manager.py`, `app/core/sor.py`, `app/core/exchange.py`
- `app/core/metrics.py`
- `app/web/templates/partials/stats.html`
- `app/tests/`

**Acceptance criteria:**
- No live order placement in shadow mode.
- Simulated fill quality metrics shown in dashboard and `/metrics`.
- Deterministic tests assert live-order path is never called.

### 2) Config Versioning + One-click Rollback
**Outcome:** Operators can safely revert bad runtime changes.

**Scope:**
- `app/database/models.py`, `app/database/db.py`
- `app/web/router.py`, `app/web/templates/config.html`, `app/web/templates/audit_log.html`
- `app/tests/`

**Acceptance criteria:**
- Every config save creates a versioned snapshot.
- Rollback operation is role-gated and audited.
- Restore path verified by tests.

### 3) Portfolio-level Risk Budgets
**Outcome:** Add account-wide safety brakes beyond per-order limits.

**Scope:**
- `app/config.py`, `.env.example`
- `app/core/strategy.py`, `app/core/engine.py`, `app/core/state.py`, `app/core/metrics.py`
- `app/utils/notifications.py`, dashboard templates
- `app/tests/`

**Acceptance criteria:**
- Daily loss, max drawdown, and concentration caps trigger deterministic halts.
- Halt reason visible in UI and alerts.

## Priority 2 — Robustness & Explainability

### 4) Walk-forward Backtesting + Stability Scoring
**Outcome:** Reduce overfitting risk before live deployment.

**Scope:**
- `app/core/backtest.py`
- `app/web/templates/backtest.html`, `app/web/templates/partials/backtest_results.html`
- `app/utils/export.py`, `app/tests/`

**Acceptance criteria:**
- Rolling train/validate windows supported.
- Out-of-sample degradation metrics included in reports/exports.

### 5) Session Replay + Incident Timeline
**Outcome:** Faster post-incident analysis and reproducible RCA.

**Scope:**
- event persistence models in `app/database/`
- replay APIs in `app/web/router.py`
- new templates under `app/web/templates/`
- `app/tests/`

**Acceptance criteria:**
- Timeline ordering deterministic.
- Operator-visible drill-down from event to action and result.

## Priority 3 — Operational Readiness UX

### 6) In-app Pre-live Readiness Gate
**Outcome:** Prevent unsafe paper→live transitions.

**Scope:**
- readiness service module (new)
- live-mode activation path in manager/engine
- dashboard readiness panel and notifications
- `app/tests/`

**Acceptance criteria:**
- Mandatory checks block live activation until green.
- Clear, actionable failure reasons for operators.

---

## Suggested rollout sequence (cross-agent)

1. **Core Trading Engine Agent**: Shadow mode execution safety + risk budget checks.
2. **Frontend & UX Agent**: Config history UI, readiness panel, and analytics visibility.
3. **Data & Analytics Agent**: Versioning persistence, replay storage, export/report updates.
4. **QA & Testing Agent**: Deterministic regression suites for each high-risk path.
5. **Security & Infrastructure Agent**: Verify secrets handling and role-gated controls during rollout.

## Handoff template for each feature

1. Problem statement (user-visible outcome)
2. Changed files/modules
3. Risk notes (failure modes)
4. Validation performed (commands/results)
5. Follow-up owner (next agent)
