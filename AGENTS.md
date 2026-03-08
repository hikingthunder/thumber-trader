# 🤖 Agent Tasking & Delegation Strategy

To efficiently build, scale, and maintain the **Thumber Trader v2.0** ecosystem, development taskings should be farmed out to specialized AI agents or team roles. This document defines the persona, scope of ownership, and primary constraints for each role to ensure maximum velocity and minimal context overlap.

---

## 1. 🧠 Core Trading Engine Agent (Quant/Backend Role)

**Focus Area:** `app/core/` and trading logic.
**Primary Objective:** Ensure the trading logic is mathematically sound, lightning-fast, and accurately interacts with exchange APIs.

*   **Ownership:**
    *   Grid placement logic (Arithmetic, Geometric, ATR-adaptive spacing).
    *   Alpha fusion mechanisms (RSI, MACD, Order Book Imbalances).
    *   Toxicity detection (VPIN) and dynamic risk management (Kelly Criterion).
    *   Smart Order Routing (SOR) and multi-venue price consolidation.
*   **Key Constraints:**
    *   Must prioritize execution speed and non-blocking asynchronous operations.
    *   Code must be heavily commented with references to the underlying mathematical models.
    *   Must never introduce logic that can result in a "runaway bot" (always bound orders).

## 2. 🎨 Frontend & User Experience Agent (Web/UI Role)

**Focus Area:** `app/web/` (routers & templates) and `app/static/`.
**Primary Objective:** Deliver a premium, institutional-grade, and responsive configuration dashboard.

*   **Ownership:**
    *   Dashboard layout, real-time charting (TradingView/Chart.js integration).
    *   Configuration UI (allowing users to manage `.env` equivalents via the web).
    *   Jinja2 templating and frontend vanilla CSS/JS (or framework if applicable).
*   **Key Constraints:**
    *   Strict adherence to premium design aesthetics (glassmorphism, modern dark mode, sleek typography).
    *   Ensure all configuration inputs have strict validation before submitting to the backend.
    *   Must maintain a responsive mobile-first or mobile-friendly design.

## 3. 💾 Data & Analytics Agent (Data Engineer Role)

**Focus Area:** `app/database/`, `app/utils/`, and reporting modules.
**Primary Objective:** Manage state persistence, ensure high data integrity, and generate actionable insights.

*   **Ownership:**
    *   Database schema design, migrations, and SQLite transaction management (`thumber_trader.db`, `grid_state.db`).
    *   Tax-ready financial reporting (CSV, XLSX, ODS generation).
    *   Prometheus metrics endpoint (`/metrics`) and historical data aggregation.
*   **Key Constraints:**
    *   Queries must be optimized and properly indexed so they do not block the trading engine.
    *   Financial data exports must strictly adhere to FIFO or user-selected accounting principles.

## 4. 🛡️ Security & Infrastructure Agent (SecOps Role)

**Focus Area:** `app/auth/`, `Dockerfile`, `docker-compose.yml`, and root deployment scripts.
**Primary Objective:** Protect user API keys and ensure the bot runs flawlessly across different environments.

*   **Ownership:**
    *   Web authentication, session management, and rate limiting.
    *   Symmetric encryption for sensitive `.env` variables (e.g., Exchange API Keys).
    *   Docker containerization, Proxmox LXC setup scripts, and deployment guides.
*   **Key Constraints:**
    *   API keys must *never* be logged or stored in plaintext if encryption is enabled.
    *   Containers must be heavily locked down (e.g., non-root users, minimal attack surface).

## 5. 🧪 Quality Assurance & Testing Agent (QA Role)

**Focus Area:** `app/tests/` and CI/CD pipelines.
**Primary Objective:** Prevent regressions and validate that the bot behaves correctly under extreme market volatility.

*   **Ownership:**
    *   Unit testing for mathematical correctness (grid sizes, indicators).
    *   Integration testing with mock exchange data.
    *   Paper trading mode validation.
*   **Key Constraints:**
    *   Must achieve >90% coverage on core trading logic before merging.
    *   Tests must be deterministic and not rely on live internet connections (mocking required).

---

## 🔄 Workflow for Agent Delegation

When opening a new task or issue, assign it to the corresponding persona. If a feature spans multiple domains (e.g., "Add a new indicator and a UI toggle for it"), break the task into sequential sub-tasks:

1.  **Core** builds the indicator logic.
2.  **Frontend** builds the UI toggle.
3.  **Data** updates the database schema to save the toggle state.
4.  **QA** writes tests for all three blocks.
