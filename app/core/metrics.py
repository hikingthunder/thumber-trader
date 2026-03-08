
import math
from decimal import Decimal
from typing import List, Dict, Optional
import time

def calculate_drawdown(equity_series: List[Decimal]) -> Dict[str, Decimal]:
    """Calculate current and max drawdown from a series of equity values."""
    if not equity_series:
        return {"current": Decimal("0"), "max": Decimal("0")}
    
    max_equity = equity_series[0]
    max_dd = Decimal("0")
    current_dd = Decimal("0")
    
    for equity in equity_series:
        if equity > max_equity:
            max_equity = equity
        
        if max_equity > 0:
            dd = (max_equity - equity) / max_equity
            current_dd = dd
            if dd > max_dd:
                max_dd = dd
                
    return {"current": current_dd, "max": max_dd}

def calculate_sharpe_ratio(returns: List[Decimal], risk_free_rate: float = 0.0) -> float:
    """Calculate the Sharpe Ratio for a series of returns."""
    if len(returns) < 5:
        return 0.0
    
    # Annualized factor (assuming 1-minute returns)
    # 60 * 24 * 365 = 525,600 minutes
    annualized_factor = math.sqrt(525600)
    
    avg_return = float(sum(returns) / len(returns))
    excess_return = avg_return - (risk_free_rate / 525600)
    
    variance = sum((float(r) - avg_return)**2 for r in returns) / (len(returns) - 1)
    std_dev = math.sqrt(variance)
    
    if std_dev == 0:
        return 0.0
        
    return (excess_return / std_dev) * annualized_factor

def calculate_time_to_recovery(equity_series: List[Decimal]) -> float:
    """Estimate time to recovery from recent drawdown in hours."""
    if len(equity_series) < 10:
        return 0.0
        
    # Find the last peak
    max_equity = equity_series[0]
    peak_idx = 0
    for i, e in enumerate(equity_series):
        if e >= max_equity:
            max_equity = e
            peak_idx = i
            
    if peak_idx == len(equity_series) - 1:
        return 0.0 # No drawdown
        
    # Calculate recovery rate since peak
    current_equity = equity_series[-1]
    lost = max_equity - current_equity
    if lost <= 0:
        return 0.0
        
    # Regression for recovery slope
    y = [float(e) for e in equity_series[peak_idx:]]
    x = list(range(len(y)))
    
    if len(x) < 5:
        return 0.0
        
    # Simple linear regression
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    den = sum((xi - mean_x)**2 for xi in x)
    
    slope = num / den if den != 0 else 0
    
    if slope <= 0:
        return 999.0 # Not recovering
        
    minutes_left = float(lost) / slope
    return round(minutes_left / 60.0, 1)

def get_order_book_density(bids: List[List], asks: List[List], price: Decimal, range_pct: float = 0.01) -> List[Dict]:
    """Generate a density heatmap of the order book around the mid-price."""
    if not bids and not asks:
        return []
        
    price_f = float(price)
    # Bins of 0.1% each
    step = 0.001 
    n_steps = 15 # +/- 1.5%
    
    density = []
    
    # Process Bids
    for i in range(1, n_steps + 1):
        target_p = price_f * (1 - (i * step))
        vol = sum(float(size) for p, size in bids if abs(float(p) - target_p) < (price_f * step / 2))
        if vol > 0:
            density.append({
                "side": "bid",
                "price_offset": -i,
                "volume": vol
            })
            
    # Process Asks
    for i in range(1, n_steps + 1):
        target_p = price_f * (1 + (i * step))
        vol = sum(float(size) for p, size in asks if abs(float(p) - target_p) < (price_f * step / 2))
        if vol > 0:
            density.append({
                "side": "ask",
                "price_offset": i,
                "volume": vol
            })
            
    return density
