# 🤖 Agent Tasking & Delegation Strategy

This repository is organized for multi-agent development. Use this file as the shared contract for ownership, boundaries, and handoff quality.

## Mission

Build and operate **Thumber Trader v2.0** safely: preserve capital controls, avoid sensitive-data leaks, and maintain deterministic behavior across local/dev/prod environments.

---

## 1) 🧠 Core Trading Engine Agent (Quant/Backend)

**Scope:** `app/core/` and strategy execution paths.

**Primary objective:** Keep execution mathematically correct and bounded under all market conditions.

**Owns:**
- Grid math (arithmetic/geometric/ATR adaptations)
- Alpha fusion weighting logic and signal normalization
- VPIN and risk-response logic
- SOR and consolidated pricing behavior
- Execution guardrails (inventory caps, stop-losses, max notional)

**Must enforce:**
- No runaway order loops
- Explicit bounds on every auto-sizing path
- Async-safe I/O and non-blocking behavior
- Comments tied to formulas/assumptions

**Done criteria:**
- Deterministic behavior under replay/backtest fixtures
- Risk constraints validated by tests

---

## 2) 🎨 Frontend & UX Agent (Web/UI)

**Scope:** `app/web/`, `app/static/`.

**Primary objective:** Deliver clear, responsive controls for operators with safe defaults and validation.

**Owns:**
- Dashboard components and HTMX/WebSocket updates
- Config forms, input validation, UX states
- Backtest/reporting pages and partial templates

**Must enforce:**
- Mobile-friendly layout
- Explicit validation/errors for all form writes
- No secret leakage in rendered templates or logs

**Done criteria:**
- Critical tasks (configure/start/stop/export) work end-to-end
- Accessibility and readability maintained in dark mode

---

## 3) 💾 Data & Analytics Agent (Data Engineer)

**Scope:** `app/database/`, `app/utils/export.py`, metrics/reporting paths.

**Primary objective:** Preserve data integrity and fast query behavior while supporting accounting-grade exports.

**Owns:**
- SQLite schema/index tuning
- Export mappers (CSV/XLSX/ODS)
- Aggregations for metrics and dashboard analytics

**Must enforce:**
- Backward-safe migrations and schema updates
- No long-running DB operations in hot trading loops
- Accounting method correctness (FIFO/default)

**Done criteria:**
- Export output reproducible from test fixtures
- DB access patterns reviewed for contention

---

## 4) 🛡️ Security & Infrastructure Agent (SecOps)

**Scope:** `app/auth/`, root deployment files (`Dockerfile`, compose, scripts).

**Primary objective:** Protect credentials and harden runtime posture.

**Owns:**
- Auth/session/CSRF/rate-limit middleware behavior
- Secret handling and encryption toggles
- Container hardening and service deployment scripts

**Must enforce:**
- Never commit plaintext credentials
- Never log API secrets/tokens
- Principle of least privilege for services/containers

**Done criteria:**
- .env and sensitive files excluded from git
- Deployment docs include rotation/recovery guidance

---

## 5) 🧪 QA & Testing Agent

**Scope:** `app/tests/`, test tooling, CI validation design.

**Primary objective:** Catch regressions before merge, especially for money/risk logic.

**Owns:**
- Unit coverage for indicators/grid math/risk guardrails
- Integration tests with mocked exchange paths
- Deterministic paper-trading validations

**Must enforce:**
- Tests independent from live internet/exchanges
- Stable fixtures and reproducible seeds

**Done criteria:**
- Core logic coverage target >90%
- High-risk flows have regression tests

---

## Cross-agent handoff protocol

When work crosses domains, create sequential subtasks and handoff notes:

1. **Problem statement** (user-visible outcome)
2. **Changed files/modules**
3. **Risk notes** (what could break)
4. **Validation performed** (commands/results)
5. **Follow-up owner** (next agent)

### Example sequence

Feature: “Add new signal + expose UI toggle + persist state + test coverage”
1. Core: implement signal and bounds
2. Frontend: add form control and display state
3. Data: persist setting / migration impact
4. QA: tests for math, wiring, and UI submission path

---

## Documentation expectations for all agents

- Keep README and `.env.example` synchronized with actual settings.
- If adding env keys, update docs in the same change.
- If adding deploy path (Docker/Podman/systemd), include rollback/update steps.

---

## Security red lines (all agents)

- Do not commit `.env`, keys, tokens, private certs, or production dumps.
- Sanitize screenshots/log snippets before sharing.
- Treat webhook secrets and JWT keys as sensitive.
