
import pytest
from decimal import Decimal
from app.core import metrics

def test_calculate_drawdown():
    # Test case 1: Steady growth
    history = [Decimal("100"), Decimal("110"), Decimal("120")]
    dd = metrics.calculate_drawdown(history)
    assert dd["current"] == 0
    assert dd["max"] == 0

    # Test case 2: Drawdown
    history = [Decimal("100"), Decimal("120"), Decimal("90"), Decimal("110")]
    # Peak was 120, current is 90. Drop is 30/120 = 0.25
    dd = metrics.calculate_drawdown(history)
    assert dd["current"] == 0.25 # 90 is 25% below 120
    assert dd["max"] == 0.25

    # Test case 3: Recovery
    history = [Decimal("100"), Decimal("120"), Decimal("90"), Decimal("130")]
    dd = metrics.calculate_drawdown(history)
    assert dd["current"] == 0
    assert dd["max"] == 0.25

def test_calculate_sharpe_ratio():
    # Returns: 1%, 2%, -1%, 3%
    returns = [0.01, 0.02, -0.01, 0.03]
    sharpe = metrics.calculate_sharpe_ratio(returns)
    assert sharpe > 0
    
    # Negative returns
    returns = [-0.01, -0.02, -0.03]
    sharpe = metrics.calculate_sharpe_ratio(returns)
    assert sharpe < 0

def test_calculate_time_to_recovery():
    # No recovery needed
    history = [Decimal("100"), Decimal("110")]
    ttr = metrics.calculate_time_to_recovery(history)
    assert ttr == 0

    # Simple recovery
    # 100 -> 80 (hour 1) -> 120 (hour 2)
    # At index 2, it recovered. Distance from peak (100) was 1 step.
    # Actually my logic is simpler: count steps since max peak was last seen if current < peak.
    # No, it's: if in drawdown, how long since peak?
    history = [Decimal("100"), Decimal("80")]
    ttr = metrics.calculate_time_to_recovery(history)
    assert ttr == 1

def test_order_book_density():
    bids = [[100, 10], [99, 20]]
    asks = [[101, 15], [102, 25]]
    mid = 100.5
    density = metrics.get_order_book_density(bids, asks, mid)
    assert isinstance(density, list)
    if density:
        assert "side" in density[0]
        assert "volume" in density[0]
        assert "price_offset" in density[0]
