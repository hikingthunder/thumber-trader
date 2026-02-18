from decimal import Decimal
from typing import List, Tuple, Dict, Any
import math
from app.core import analysis

# Re-exporting basic indicators from analysis to maintain compatibility while transition occurs
ema = analysis.ema
rsi = analysis.rsi
macd_histogram = analysis.macd_histogram
atr = analysis.atr
adx = analysis.adx
vpin = analysis.vpin

def bollinger_bands(values: List[Decimal], period: int, num_std: Decimal = Decimal("2")) -> Tuple[Decimal, Decimal, Decimal]:
    """Calculate Bollinger Bands: (Upper, Middle, Lower)."""
    if len(values) < period:
        return Decimal("0"), Decimal("0"), Decimal("0")
    
    recent_vals = values[-period:]
    middle_band = sum(recent_vals) / Decimal(period)
    
    variance = sum((v - middle_band) ** 2 for v in recent_vals) / Decimal(period)
    std_dev = variance.sqrt()
    
    upper_band = middle_band + (num_std * std_dev)
    lower_band = middle_band - (num_std * std_dev)
    
    return upper_band, middle_band, lower_band

def stochastic_oscillator(candles: List[Tuple[int, Decimal, Decimal, Decimal, Decimal]], k_period: int = 14, d_period: int = 3) -> Tuple[Decimal, Decimal]:
    """Calculate Stochastic Oscillator: (%K, %D)."""
    if len(candles) < k_period:
        return Decimal("50"), Decimal("50")
    
    k_values: List[Decimal] = []
    
    # Calculate %K for the last d_period to get the average %D
    for i in range(len(candles) - d_period + 1, len(candles) + 1):
        window = candles[i-k_period : i]
        if not window:
            continue
            
        current_close = window[-1][3]
        lows = [c[2] for c in window]
        highs = [c[1] for c in window]
        
        lowest_low = min(lows)
        highest_high = max(highs)
        
        price_range = highest_high - lowest_low
        if price_range > 0:
            k = ((current_close - lowest_low) / price_range) * Decimal("100")
        else:
            k = Decimal("50")
        k_values.append(k)
        
    if not k_values:
        return Decimal("50"), Decimal("50")
        
    current_k = k_values[-1]
    current_d = sum(k_values) / Decimal(len(k_values))
    
    return current_k, current_d

def ichimoku_cloud(candles: List[Tuple[int, Decimal, Decimal, Decimal, Decimal]], conversion_period: int = 9, base_period: int = 26, span_b_period: int = 52) -> Dict[str, Decimal]:
    """Calculate Ichimoku Cloud components."""
    def get_hi_lo_avg(period_candles: List[Tuple[Any, Decimal, Decimal, Any, Any]]) -> Decimal:
        highs = [c[1] for c in period_candles]
        lows = [c[2] for c in period_candles]
        return (max(highs) + min(lows)) / Decimal("2")

    if len(candles) < span_b_period:
        return {
            "tenkan_sen": Decimal("0"),
            "kijun_sen": Decimal("0"),
            "senkou_span_a": Decimal("0"),
            "senkou_span_b": Decimal("0")
        }

    tenkan_sen = get_hi_lo_avg(candles[-conversion_period:])
    kijun_sen = get_hi_lo_avg(candles[-base_period:])
    senkou_span_a = (tenkan_sen + kijun_sen) / Decimal("2")
    senkou_span_b = get_hi_lo_avg(candles[-span_b_period:])
    
    # Note: senkou_span_a and b are usually shifted forward 26 periods. 
    # Here we return current values.
    
    return {
        "tenkan_sen": tenkan_sen,
        "kijun_sen": kijun_sen,
        "senkou_span_a": senkou_span_a,
        "senkou_span_b": senkou_span_b
    }
