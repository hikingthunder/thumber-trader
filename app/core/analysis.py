import math
import logging
from decimal import Decimal
from typing import List, Tuple, Optional, Dict

def ema(values: List[Decimal], period: int) -> Decimal:
    if period <= 0 or len(values) < period:
        return Decimal("0")
    alpha = Decimal("2") / Decimal(period + 1)
    ema_val = values[0]
    for v in values[1:]:
        ema_val = (v * alpha) + (ema_val * (Decimal("1") - alpha))
    return ema_val

def rsi(values: List[Decimal], period: int) -> Decimal:
    if period <= 0 or len(values) < period + 1:
        return Decimal("50")

    gains: List[Decimal] = []
    losses: List[Decimal] = []
    for prev, cur in zip(values, values[1:]):
        delta = cur - prev
        gains.append(max(Decimal("0"), delta))
        losses.append(max(Decimal("0"), -delta))

    avg_gain = sum(gains[-period:], Decimal("0")) / Decimal(period)
    avg_loss = sum(losses[-period:], Decimal("0")) / Decimal(period)
    if avg_loss <= 0:
        return Decimal("100") if avg_gain > 0 else Decimal("50")
    rs = avg_gain / avg_loss
    return Decimal("100") - (Decimal("100") / (Decimal("1") + rs))

def macd_histogram(values: List[Decimal], fast: int, slow: int, signal: int) -> Decimal:
    if len(values) < max(fast, slow) + signal:
        return Decimal("0")

    alpha_fast = Decimal("2") / Decimal(fast + 1)
    alpha_slow = Decimal("2") / Decimal(slow + 1)
    alpha_signal = Decimal("2") / Decimal(signal + 1)

    fast_ema = values[0]
    slow_ema = values[0]
    macd_series: List[Decimal] = []
    for val in values:
        fast_ema = (val * alpha_fast) + (fast_ema * (Decimal("1") - alpha_fast))
        slow_ema = (val * alpha_slow) + (slow_ema * (Decimal("1") - alpha_slow))
        macd_series.append(fast_ema - slow_ema)

    signal_ema = macd_series[0]
    for m in macd_series[1:]:
        signal_ema = (m * alpha_signal) + (signal_ema * (Decimal("1") - alpha_signal))
    return macd_series[-1] - signal_ema

def atr(candles: List[Tuple[int, Decimal, Decimal, Decimal]], period: int) -> Decimal:
    if period <= 0 or len(candles) <= period:
        return Decimal("0")

    true_ranges: List[Decimal] = []
    prev_close = candles[0][3]
    for _ts, high, low, close in candles[1:]:
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(tr)
        prev_close = close

    if len(true_ranges) < period:
        return Decimal("0")
    return sum(true_ranges[-period:], Decimal("0")) / Decimal(period)

def adx(candles: List[Tuple[int, Decimal, Decimal, Decimal]], period: int) -> Decimal:
    if period <= 1 or len(candles) < (period * 2):
        return Decimal("0")

    plus_dm_values: List[Decimal] = []
    minus_dm_values: List[Decimal] = []
    tr_values: List[Decimal] = []

    prev_high = candles[0][1]
    prev_low = candles[0][2]
    prev_close = candles[0][3]

    for _ts, high, low, close in candles[1:]:
        up_move = high - prev_high
        down_move = prev_low - low

        plus_dm = up_move if up_move > down_move and up_move > 0 else Decimal("0")
        minus_dm = down_move if down_move > up_move and down_move > 0 else Decimal("0")
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))

        plus_dm_values.append(plus_dm)
        minus_dm_values.append(minus_dm)
        tr_values.append(tr)

        prev_high = high
        prev_low = low
        prev_close = close

    if len(tr_values) < period:
        return Decimal("0")

    atr_val = sum(tr_values[:period], Decimal("0"))
    plus_dm_smoothed = sum(plus_dm_values[:period], Decimal("0"))
    minus_dm_smoothed = sum(minus_dm_values[:period], Decimal("0"))

    dx_values: List[Decimal] = []

    def _append_dx(current_atr: Decimal, current_plus_dm: Decimal, current_minus_dm: Decimal) -> None:
        if current_atr <= 0:
            return
        plus_di = (current_plus_dm / current_atr) * Decimal("100")
        minus_di = (current_minus_dm / current_atr) * Decimal("100")
        denominator = plus_di + minus_di
        if denominator <= 0:
            return
        dx_values.append((abs(plus_di - minus_di) / denominator) * Decimal("100"))

    _append_dx(atr_val, plus_dm_smoothed, minus_dm_smoothed)

    for idx in range(period, len(tr_values)):
        atr_val = atr_val - (atr_val / Decimal(period)) + tr_values[idx]
        plus_dm_smoothed = plus_dm_smoothed - (plus_dm_smoothed / Decimal(period)) + plus_dm_values[idx]
        minus_dm_smoothed = minus_dm_smoothed - (minus_dm_smoothed / Decimal(period)) + minus_dm_values[idx]
        _append_dx(atr_val, plus_dm_smoothed, minus_dm_smoothed)

    if not dx_values:
        return Decimal("0")

    lookback = min(period, len(dx_values))
    return sum(dx_values[-lookback:], Decimal("0")) / Decimal(lookback)

def calculate_returns(closes: List[Decimal]) -> List[float]:
    if len(closes) < 2:
        return []
    rets: List[float] = []
    prev = closes[0]
    for current in closes[1:]:
        if prev > 0 and current > 0:
            rets.append(float((current / prev) - Decimal("1")))
        prev = current
    return rets

def returns_decimal(closes: List[Decimal]) -> List[Decimal]:
    if len(closes) < 2:
        return []
    rets: List[Decimal] = []
    prev = closes[0]
    for current in closes[1:]:
        if prev > 0 and current > 0:
            rets.append((current / prev) - Decimal("1"))
        prev = current
    return rets

def fit_gaussian_hmm(
    observations: List[float],
    n_states: int,
    iterations: int,
    min_variance: Decimal,
) -> Optional[Tuple[List[float], List[List[float]], List[float], List[float]]]:
    t_len = len(observations)
    if t_len < 2 or n_states < 2:
        return None

    eps = 1e-12
    min_var = float(min_variance)
    sorted_obs = sorted(observations)

    means = [sorted_obs[int((idx + 1) * (t_len / (n_states + 1)))] for idx in range(n_states)]
    obs_mean = sum(observations) / t_len
    obs_var = sum((v - obs_mean) ** 2 for v in observations) / t_len
    variances = [max(obs_var, min_var) for _ in range(n_states)]
    init_prob = [1.0 / n_states for _ in range(n_states)]
    trans = []
    for i in range(n_states):
        row = []
        for j in range(n_states):
            row.append(0.85 if i == j else 0.15 / max(1, n_states - 1))
        trans.append(row)

    def emission(x: float, mean: float, var: float) -> float:
        var = max(var, min_var)
        coeff = 1.0 / math.sqrt(2.0 * math.pi * var)
        exponent = math.exp(-((x - mean) ** 2) / (2.0 * var))
        return max(eps, coeff * exponent)

    for _ in range(iterations):
        b = [[emission(observations[t], means[i], variances[i]) for i in range(n_states)] for t in range(t_len)]
        alpha = [[0.0 for _ in range(n_states)] for _ in range(t_len)]
        beta = [[0.0 for _ in range(n_states)] for _ in range(t_len)]
        scales = [0.0 for _ in range(t_len)]

        for i in range(n_states):
            alpha[0][i] = init_prob[i] * b[0][i]
        scales[0] = max(eps, sum(alpha[0]))
        for i in range(n_states):
            alpha[0][i] /= scales[0]

        for t in range(1, t_len):
            for j in range(n_states):
                alpha[t][j] = sum(alpha[t - 1][i] * trans[i][j] for i in range(n_states)) * b[t][j]
            scales[t] = max(eps, sum(alpha[t]))
            for j in range(n_states):
                alpha[t][j] /= scales[t]

        for i in range(n_states):
            beta[-1][i] = 1.0
        for t in range(t_len - 2, -1, -1):
            for i in range(n_states):
                beta[t][i] = sum(trans[i][j] * b[t + 1][j] * beta[t + 1][j] for j in range(n_states))
                beta[t][i] /= max(eps, scales[t + 1])

        gamma = [[0.0 for _ in range(n_states)] for _ in range(t_len)]
        xi_sum = [[0.0 for _ in range(n_states)] for _ in range(n_states)]
        for t in range(t_len):
            norm = max(eps, sum(alpha[t][i] * beta[t][i] for i in range(n_states)))
            for i in range(n_states):
                gamma[t][i] = (alpha[t][i] * beta[t][i]) / norm

        for t in range(t_len - 1):
            denom = 0.0
            for i in range(n_states):
                for j in range(n_states):
                    denom += alpha[t][i] * trans[i][j] * b[t + 1][j] * beta[t + 1][j]
            denom = max(eps, denom)
            for i in range(n_states):
                for j in range(n_states):
                    numer = alpha[t][i] * trans[i][j] * b[t + 1][j] * beta[t + 1][j]
                    xi_sum[i][j] += numer / denom

        for i in range(n_states):
            init_prob[i] = gamma[0][i]
            trans_denom = max(eps, sum(gamma[t][i] for t in range(t_len - 1)))
            for j in range(n_states):
                trans[i][j] = xi_sum[i][j] / trans_denom
            row_sum = max(eps, sum(trans[i]))
            for j in range(n_states):
                trans[i][j] /= row_sum

            gamma_sum = max(eps, sum(gamma[t][i] for t in range(t_len)))
            means[i] = sum(gamma[t][i] * observations[t] for t in range(t_len)) / gamma_sum
            var = sum(gamma[t][i] * ((observations[t] - means[i]) ** 2) for t in range(t_len)) / gamma_sum
            variances[i] = max(var, min_var)

    last_probs = [max(eps, p) for p in gamma[-1]]
    total = max(eps, sum(last_probs))
    init_out = [p / total for p in last_probs]
    trans_out = [[float(cell) for cell in row] for row in trans]
    return init_out, trans_out, means, variances

def hmm_filter_probabilities(
    observations: List[float],
    init_prob: List[float],
    trans: List[List[float]],
    means: List[float],
    variances: List[float],
) -> List[float]:
    if not observations or not init_prob or not trans or not means or not variances:
        return []
    n_states = len(init_prob)
    if n_states == 0:
        return []
    eps = 1e-12

    def emission(x: float, mean: float, var: float) -> float:
        safe_var = max(float(var), eps)
        coeff = 1.0 / math.sqrt(2.0 * math.pi * safe_var)
        exponent = math.exp(-((x - mean) ** 2) / (2.0 * safe_var))
        return max(eps, coeff * exponent)

    probs = [max(eps, float(p)) for p in init_prob[:n_states]]
    norm = max(eps, sum(probs))
    probs = [p / norm for p in probs]

    for obs in observations:
        next_probs = [0.0 for _ in range(n_states)]
        for j in range(n_states):
            transition_sum = 0.0
            for i in range(n_states):
                transition_sum += probs[i] * max(eps, float(trans[i][j]))
            next_probs[j] = transition_sum * emission(obs, means[j], variances[j])
        denom = max(eps, sum(next_probs))
        probs = [v / denom for v in next_probs]

    return probs

def kolmogorov_smirnov_statistic(sample_a: List[float], sample_b: List[float]) -> float:
    if not sample_a or not sample_b:
        return 0.0
    xs = sorted(sample_a)
    ys = sorted(sample_b)
    n = len(xs)
    m = len(ys)
    i = 0
    j = 0
    cdf_x = 0.0
    cdf_y = 0.0
    max_diff = 0.0

    while i < n and j < m:
        if xs[i] <= ys[j]:
            i += 1
            cdf_x = i / n
        else:
            j += 1
            cdf_y = j / m
        max_diff = max(max_diff, abs(cdf_x - cdf_y))

    while i < n:
        i += 1
        cdf_x = i / n
        max_diff = max(max_diff, abs(cdf_x - cdf_y))

    while j < m:
        j += 1
        cdf_y = j / m
        max_diff = max(max_diff, abs(cdf_x - cdf_y))

    return max_diff

def pearson_corr(xs: List[Decimal], ys: List[Decimal]) -> Decimal:
    n = min(len(xs), len(ys))
    if n < 2:
        return Decimal("0")
    xs = xs[-n:]
    ys = ys[-n:]
    mean_x = sum(xs, Decimal("0")) / Decimal(n)
    mean_y = sum(ys, Decimal("0")) / Decimal(n)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / Decimal(n)
    var_x = sum((x - mean_x) * (x - mean_x) for x in xs) / Decimal(n)
    var_y = sum((y - mean_y) * (y - mean_y) for y in ys) / Decimal(n)
    if var_x <= 0 or var_y <= 0:
        return Decimal("0")
    return cov / ((var_x.sqrt()) * (var_y.sqrt()))

def beta(asset_returns: List[Decimal], benchmark_returns: List[Decimal]) -> Decimal:
    n = min(len(asset_returns), len(benchmark_returns))
    if n < 2:
        return Decimal("0")
    x = benchmark_returns[-n:]
    y = asset_returns[-n:]
    mean_x = sum(x, Decimal("0")) / Decimal(n)
    mean_y = sum(y, Decimal("0")) / Decimal(n)
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / Decimal(n)
    var_x = sum((xi - mean_x) * (xi - mean_x) for xi in x) / Decimal(n)
    if var_x <= 0:
        return Decimal("0")
    return cov / var_x

def linreg_slope_intercept(xs: List[Decimal], ys: List[Decimal]) -> Tuple[Decimal, Decimal]:
    n = min(len(xs), len(ys))
    if n < 2:
        return Decimal("0"), Decimal("0")
    x = xs[-n:]
    y = ys[-n:]
    mean_x = sum(x, Decimal("0")) / Decimal(n)
    mean_y = sum(y, Decimal("0")) / Decimal(n)
    var_x = sum((xi - mean_x) * (xi - mean_x) for xi in x) / Decimal(n)
    if var_x <= 0:
        return Decimal("0"), mean_y
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / Decimal(n)
    slope = cov / var_x
    intercept = mean_y - (slope * mean_x)
    return slope, intercept

def residual_half_life_bars(residuals: List[Decimal]) -> Decimal:
    if len(residuals) < 3:
        return Decimal("999999")
    lagged = residuals[:-1]
    delta = [cur - prev for prev, cur in zip(residuals[:-1], residuals[1:])]
    slope, _ = linreg_slope_intercept(lagged, delta)
    if slope >= 0:
        return Decimal("999999")
    slope_abs = abs(slope)
    if slope_abs <= Decimal("0.00000001"):
        return Decimal("999999")
    return Decimal(str(math.log(2.0))) / slope_abs

def zscore(values: List[Decimal]) -> Decimal:
    if len(values) < 2:
        return Decimal("0")
    mean = sum(values, Decimal("0")) / Decimal(len(values))
    var = sum((v - mean) * (v - mean) for v in values) / Decimal(len(values))
    if var <= 0:
        return Decimal("0")
    std = var.sqrt()
    return (values[-1] - mean) / std

def decimal_percentile(values: List[Decimal], percentile: Decimal) -> Decimal:
    if not values:
        return Decimal("0")
    pct = max(Decimal("0"), min(Decimal("1"), percentile))
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = int((len(ordered) - 1) * float(pct))
    idx = max(0, min(len(ordered) - 1, idx))
    return ordered[idx]

def survival_probability(
    *,
    returns: List[Decimal],
    horizon_days: int,
    ruin_drawdown_pct: Decimal,
    inventory_ratio: Decimal,
) -> Dict[str, Decimal]:
    if len(returns) < 2:
        return {
            "survival_probability": Decimal("1"),
            "risk_of_ruin_probability": Decimal("0"),
            "inventory_skew": abs(inventory_ratio - Decimal("0.5")) * Decimal("2"),
        }

    mean_return = sum(returns, Decimal("0")) / Decimal(len(returns))
    variance = sum(((ret - mean_return) ** 2 for ret in returns), Decimal("0")) / Decimal(max(1, len(returns) - 1))
    vol = variance.sqrt() if variance > 0 else Decimal("0")
    horizon_steps = Decimal(max(1, horizon_days * 24 * 60))
    sigma_h = vol * Decimal(str(math.sqrt(float(horizon_steps))))
    drift_h = mean_return * horizon_steps

    inventory_skew = abs(inventory_ratio - Decimal("0.5")) * Decimal("2")
    downside_exposure = max(Decimal("0.0001"), inventory_ratio * (Decimal("1") + (inventory_skew * Decimal("0.5"))))
    ruin_boundary = -ruin_drawdown_pct / downside_exposure

    if sigma_h <= 0:
        ruin_probability = Decimal("0") if ruin_boundary < drift_h else Decimal("1")
    else:
        z = float((ruin_boundary - drift_h) / sigma_h)
        ruin_probability = Decimal(str(0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))))
    ruin_probability = min(Decimal("1"), max(Decimal("0"), ruin_probability))
    survival_probability_val = Decimal("1") - ruin_probability
    return {
        "survival_probability": survival_probability_val,
        "risk_of_ruin_probability": ruin_probability,
        "inventory_skew": inventory_skew,
    }
