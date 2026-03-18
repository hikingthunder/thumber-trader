import asyncio
from decimal import Decimal

from app.core.strategy import GridStrategy
import app.core.strategy as strategy_module


class DummyExchange:
    def __init__(self, order_id: str = "live-order-1"):
        self.calls = 0
        self.order_id = order_id

    async def create_order(self, product_id, side, config):
        self.calls += 1
        return {"order_id": self.order_id}


class _FakeSession:
    def __init__(self):
        self.added = []
        self.commits = 0

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        self.commits += 1


class _FakeSessionContext:
    def __init__(self, session):
        self.session = session

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeSessionFactory:
    def __init__(self):
        self.session = _FakeSession()

    def __call__(self):
        return _FakeSessionContext(self.session)


def _configure_test_settings(monkeypatch, *, execution_mode: str, paper_mode: bool):
    monkeypatch.setattr(strategy_module.settings, "execution_mode", execution_mode)
    monkeypatch.setattr(strategy_module.settings, "paper_trading_mode", paper_mode)
    monkeypatch.setattr(strategy_module.settings, "min_notional_usd", Decimal("1"))


def test_shadow_mode_never_calls_live_order_path(monkeypatch):
    _configure_test_settings(monkeypatch, execution_mode="shadow_live", paper_mode=False)
    fake_session_factory = _FakeSessionFactory()
    monkeypatch.setattr(strategy_module, "AsyncSessionLocal", fake_session_factory)

    exchange = DummyExchange(order_id="live-should-not-happen")
    strategy = GridStrategy("BTC-USD", exchange)

    asyncio.run(strategy._place_limit_order("BUY", Decimal("50000"), Decimal("0.001"), 3))

    assert exchange.calls == 0
    assert len(strategy.orders) == 1
    created_order_id = next(iter(strategy.orders))
    assert created_order_id.startswith("shadow-")
    assert fake_session_factory.session.commits == 1


def test_live_mode_calls_exchange_create_order(monkeypatch):
    _configure_test_settings(monkeypatch, execution_mode="live", paper_mode=False)
    fake_session_factory = _FakeSessionFactory()
    monkeypatch.setattr(strategy_module, "AsyncSessionLocal", fake_session_factory)

    exchange = DummyExchange(order_id="live-order-123")
    strategy = GridStrategy("BTC-USD", exchange)

    asyncio.run(strategy._place_limit_order("SELL", Decimal("51000"), Decimal("0.001"), 4))

    assert exchange.calls == 1
    assert "live-order-123" in strategy.orders
    assert fake_session_factory.session.commits == 1
