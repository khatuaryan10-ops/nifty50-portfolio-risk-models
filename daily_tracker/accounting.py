"""Portfolio accounting from daily returns: drift, value, quarterly rebalance, metrics."""

import numpy as np

VOL_WINDOW = 60
TRADING_DAYS = 252


def step(weights, returns):
    """Apply one day's stock returns to yesterday's drifted weights.
    Returns the portfolio return and today's drifted weights."""
    rp = float(weights @ returns)
    return rp, weights * (1 + returns) / (1 + rp)


def is_quarter_start(date, prev_date):
    """First trading day of a quarter: the previous trading day is in another quarter."""
    return (date.year, date.quarter) != (prev_date.year, prev_date.quarter)


def turnover(before, after):
    return float(0.5 * (after - before).abs().sum())


def max_drift(weights, target):
    return float((weights - target).abs().max())


def vol_60d(returns):
    """Annualised volatility of the last 60 daily returns; NaN until 60 exist."""
    if len(returns) < VOL_WINDOW:
        return np.nan
    return float(np.std(returns[-VOL_WINDOW:], ddof=1) * np.sqrt(TRADING_DAYS))


def clean_weights(w, dust=1e-6):
    """Zero out solver dust (weights below `dust`) and renormalise to sum to 1."""
    w = w.where(w >= dust, 0.0)
    w = w[w > 0]
    return w / w.sum()
