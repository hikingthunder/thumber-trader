import asyncio
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.main import app
import app.web.router as web_router
from app.database.db import init_db

CSRF_TOKEN = "resilience-csrf-token"


def _auth_override():
    return SimpleNamespace(id=1, role="admin", username="tester")


def test_dashboard_stats_exposes_balances_and_exchange_health(monkeypatch):
    asyncio.run(init_db())

    async def fake_stats():
        return {
            "running": True,
            "strategies": {
                "BTC-USD": {
                    "inventory_base": 0.25,
                    "grid_spacing_base": 0.01,
                    "sharpe_ratio": 1.2,
                    "max_drawdown_pct": 2.3,
                    "execution_mode": "paper",
                    "last_price": 101000,
                }
            },
            "balances": {"BTC": 0.5, "USD": 2000},
            "total_realized_pnl": 10,
            "total_unrealized_pnl": -2,
        }

    monkeypatch.setattr(web_router.manager, "get_global_stats", fake_stats)
    monkeypatch.setattr(
        web_router.manager,
        "get_exchange_health",
        lambda: {"exchange_connected": True},
    )

    app.dependency_overrides[web_router.get_current_user] = _auth_override
    with TestClient(app) as client:
        response = client.get("/dashboard/stats")
        assert response.status_code == 200
        assert "Base Balance" in response.text
        assert "Quote Balance" in response.text
        assert "Exchange Feed" in response.text
    app.dependency_overrides.clear()


def test_dashboard_depth_endpoint_returns_cached_payload(monkeypatch):
    app.dependency_overrides[web_router.get_current_user] = _auth_override
    monkeypatch.setattr(
        web_router.manager,
        "last_depth_payload",
        {
            web_router.settings.product_id: {
                "type": "depth",
                "product_id": web_router.settings.product_id,
                "density": [{"side": "bid", "price_offset": -1, "volume": 1.0}],
                "vpin": 0.2,
                "time": 100,
            }
        },
    )

    with TestClient(app) as client:
        response = client.get("/dashboard/depth")
        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "depth"
        assert data["density"]

    app.dependency_overrides.clear()


def test_config_save_returns_safe_error_on_unexpected_exception(monkeypatch):
    def boom(_updates):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(web_router, "update_env_file", boom)
    app.dependency_overrides[web_router.get_current_user] = _auth_override

    with TestClient(app) as client:
        response = client.post(
            "/config/save",
            data={"product_ids": "BTC-USD"},
            headers={"X-CSRF-Token": CSRF_TOKEN},
            cookies={"thumber_csrf_token": CSRF_TOKEN},
        )
        assert response.status_code == 500
        assert "RuntimeError" in response.text

    app.dependency_overrides.clear()
