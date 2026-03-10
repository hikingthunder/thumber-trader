# Release Enhancement Plan (Post Paper-Trading Validation)

This plan converts the recommended improvements into a staged, execution-ready delivery roadmap.

## How to use this roadmap

For each feature below, update these fields in every implementing PR:
- **Status:** `not_started` | `in_progress` | `blocked` | `done`
- **Owner:** primary role from `AGENTS.md`
- **Target milestone:** release tag / sprint
- **PR link(s):** merged and active PRs
- **Last updated:** `YYYY-MM-DD`

Also include migration notes, security sign-off, and validation command outputs before moving a feature to `done`.

---

## Priority 1 — Pre-live Safety & Control

### 1) Shadow Live Mode (real market data + simulated execution)
**Outcome:** Validate slippage/latency/fill-quality assumptions without risking funds.

**Tracking**
- Status: `not_started`
- Owner: `Core Trading Engine Agent`
- Target milestone: `v2.0.1`
- PR link(s): _TBD_
- Last updated: `2026-03-10`

**Scope:**
- `app/core/engine.py`, `app/core/manager.py`, `app/core/sor.py`, `app/core/exchange.py`
- `app/core/metrics.py`
- `app/web/templates/partials/stats.html`
- `app/tests/`

**Implementation checklist:**
- [ ] Add `SHADOW_LIVE` execution mode toggle and runtime wiring.
- [ ] Ensure no exchange order POST path can execute in shadow mode.
- [ ] Persist decision-time order book snapshots for simulated fills.
- [ ] Add simulated vs theoretical fill-quality metrics for dashboard + `/metrics`.
- [ ] Add deterministic tests proving live order path is never called in shadow mode.

**Migration / backward compatibility notes:**
- No destructive schema change expected.
- Any added tables/columns must default safely and not impact existing paper/live runs.

**Security sign-off:**
- Validate no API secrets leak into shadow telemetry/logs/screenshots.
- Confirm role-gated controls for mode switching.

**Acceptance criteria:**
- No live order placement in shadow mode.
- Simulated fill quality metrics shown in dashboard and `/metrics`.
- Deterministic tests assert live-order path is never called.

---

### 2) Config Versioning + One-click Rollback
**Outcome:** Operators can safely revert bad runtime changes.

**Tracking**
- Status: `done`
- Owner: `Frontend & UX Agent` + `Data & Analytics Agent`
- Target milestone: `v2.0.1`
- PR link(s): `53b06f4`, `6776ffa`, `cf67061`, `current integration-test commit`
- Last updated: `2026-03-10`

**Scope:**
- `app/database/models.py`, `app/database/db.py`
- `app/web/router.py`, `app/web/templates/config.html`, `app/web/templates/audit_log.html`
- `app/tests/`

**Implementation checklist:**
- [x] Persist config snapshots with actor metadata.
- [x] Add role-gated rollback endpoint and audit event.
- [x] Render recent config history and rollback action in config UI.
- [x] Add audit-log UI enhancement for rollback detail filtering/visibility.
- [x] Add integration test coverage for save→rollback roundtrip through HTTP endpoints.

**Migration / backward compatibility notes:**
- New config version table must be additive and safe for existing DBs.
- Rollback should preserve `.env` file permissions and formatting constraints.

**Security sign-off:**
- Verify sensitive values remain encrypted-at-rest in `.env` updates.
- Verify rollback actions are admin-only and audited.

**Acceptance criteria:**
- Every config save creates a versioned snapshot.
- Rollback operation is role-gated and audited.
- Restore path verified by tests.

---

### 3) Portfolio-level Risk Budgets
**Outcome:** Add account-wide safety brakes beyond per-order limits.

**Tracking**
- Status: `not_started`
- Owner: `Core Trading Engine Agent`
- Target milestone: `v2.0.2`
- PR link(s): _TBD_
- Last updated: `2026-03-09`

**Scope:**
- `app/config.py`, `.env.example`
- `app/core/strategy.py`, `app/core/engine.py`, `app/core/state.py`, `app/core/metrics.py`
- `app/utils/notifications.py`, dashboard templates
- `app/tests/`

**Implementation checklist:**
- [ ] Add risk-budget config keys (daily loss, drawdown, concentration) with safe defaults.
- [ ] Enforce budget checks before every new order intent.
- [ ] Add deterministic halt-state transitions + reason codes.
- [ ] Surface halt reasons in dashboard and notifications.
- [ ] Add deterministic regression tests for each tripwire.

**Migration / backward compatibility notes:**
- Additive config-only changes must preserve existing behavior when disabled/default.
- `.env.example` and README must be updated in the same PR.

**Security sign-off:**
- Confirm alerts/messages do not include secrets or raw keys.

**Acceptance criteria:**
- Daily loss, max drawdown, and concentration caps trigger deterministic halts.
- Halt reason visible in UI and alerts.

---

## Priority 2 — Robustness & Explainability

### 4) Walk-forward Backtesting + Stability Scoring
**Outcome:** Reduce overfitting risk before live deployment.

**Tracking**
- Status: `not_started`
- Owner: `Core Trading Engine Agent` + `Data & Analytics Agent`
- Target milestone: `v2.0.2`
- PR link(s): _TBD_
- Last updated: `2026-03-09`

**Scope:**
- `app/core/backtest.py`
- `app/web/templates/backtest.html`, `app/web/templates/partials/backtest_results.html`
- `app/utils/export.py`, `app/tests/`

**Implementation checklist:**
- [ ] Add rolling train/validate window execution model.
- [ ] Add stability metrics (parameter variance, OOS degradation, drawdown drift).
- [ ] Expose controls/results in backtest UI.
- [ ] Extend exports to include walk-forward/stability outputs.
- [ ] Add deterministic fixture-based tests for walk-forward reports.

**Migration / backward compatibility notes:**
- Keep current backtest path intact; walk-forward should be opt-in.

**Security sign-off:**
- Confirm no sensitive config values exported with report artifacts.

**Acceptance criteria:**
- Rolling train/validate windows supported.
- Out-of-sample degradation metrics included in reports/exports.

---

### 5) Session Replay + Incident Timeline
**Outcome:** Faster post-incident analysis and reproducible RCA.

**Tracking**
- Status: `not_started`
- Owner: `Data & Analytics Agent` + `Frontend & UX Agent`
- Target milestone: `v2.0.3`
- PR link(s): _TBD_
- Last updated: `2026-03-09`

**Scope:**
- event persistence models in `app/database/`
- replay APIs in `app/web/router.py`
- new templates under `app/web/templates/`
- `app/tests/`

**Implementation checklist:**
- [ ] Persist decision/fill/risk events with deterministic ordering keys.
- [ ] Implement replay APIs with filter and pagination controls.
- [ ] Add UI timeline with drill-down detail for operator workflows.
- [ ] Add tests covering event ordering and replay integrity.

**Migration / backward compatibility notes:**
- Additive event tables/indexes only; avoid blocking writes in hot loops.

**Security sign-off:**
- Ensure replay views redact secrets/tokens.
- Confirm RBAC for access to incident timelines.

**Acceptance criteria:**
- Timeline ordering deterministic.
- Operator-visible drill-down from event to action and result.

---

## Priority 3 — Operational Readiness UX

### 6) In-app Pre-live Readiness Gate
**Outcome:** Prevent unsafe paper→live transitions.

**Tracking**
- Status: `not_started`
- Owner: `Security & Infrastructure Agent` + `Frontend & UX Agent`
- Target milestone: `v2.0.3`
- PR link(s): _TBD_
- Last updated: `2026-03-09`

**Scope:**
- readiness service module (new)
- live-mode activation path in manager/engine
- dashboard readiness panel and notifications
- `app/tests/`

**Implementation checklist:**
- [ ] Implement readiness checks (config completeness, limits present, health checks).
- [ ] Block live activation when required checks fail.
- [ ] Show actionable readiness failures in dashboard UX.
- [ ] Add notifications for readiness state changes.
- [ ] Add tests for pass/fail transitions and role-gated overrides.

**Migration / backward compatibility notes:**
- Readiness gate should default to non-disruptive for existing paper workflows.

**Security sign-off:**
- Validate readiness diagnostics never disclose secret values.

**Acceptance criteria:**
- Mandatory checks block live activation until green.
- Clear, actionable failure reasons for operators.

---

## Validation matrix (required before marking a feature `done`)

- Unit tests for new logic paths and boundary conditions.
- Integration tests (mocked exchange/IO) for end-to-end feature flow.
- UI validation for operator workflows where applicable.
- Audit/security checks for role-gated controls and secret handling.
- README + `.env.example` parity verification if settings are introduced/changed.

Record exact commands and outcomes in PR description.

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
