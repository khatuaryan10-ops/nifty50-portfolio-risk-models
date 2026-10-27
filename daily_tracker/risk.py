"""Top-down GJR-GARCH-X VaR for a tracked portfolio, using gjr_fit / gjr_filter from
nifty50_risk_layers.py unchanged. Returns are in % inside the model, as there; VIX enters
as daily variance VIX²/252.

The model is fitted once at init. Each day one step of the same recursion turns today's
return and today's India VIX into tomorrow's variance, and the 99% VaR is filtered
historical simulation: the 1% quantile of the standardised training returns times
tomorrow's volatility (as in score() in nifty50_risk_layers.py).
"""

import numpy as np
import pandas as pd

from nifty50_efficient_frontier import TRADING_DAYS
from nifty50_risk_layers import VAR_LEVEL, gjr_filter, gjr_fit

from .data import VIX, download

HISTORY_START = "2019-01-01"


def vix_term(vix):
    return np.asarray(vix, dtype=float) ** 2 / TRADING_DAYS


def next_variance(params, r_pct, vix, s2_today):
    """σ²ₜ₊₁ from today's return (%), today's VIX and today's σ²ₜ: one step of gjr_filter."""
    r = np.array([r_pct, 0.0])
    X = np.array([[vix_term(vix)], [0.0]])
    return float(gjr_filter(np.asarray(params), r, X, s2_today)[1])


def var_99(z_quantile, s2):
    """Next-day 99% VaR as a positive decimal loss."""
    return float(-z_quantile * np.sqrt(s2) / 100)


def fit_history(stocks, end):
    """Daily returns of `stocks` and India VIX up to `end`, from the first date on which
    every stock trades. Not load_data(): that starts in 2019 and drops later listings
    (MAXHEALTH, ETERNAL), which the portfolios hold. Adjusted closes are fine here: this
    is the model's fitting sample, not tracked performance."""
    symbols = [s + ".NS" for s in stocks] + [VIX]
    close = download(symbols, HISTORY_START, end + pd.Timedelta(days=1), auto_adjust=True)["close"]
    close.columns = [c.replace(".NS", "") for c in close.columns]
    missing = [s for s in stocks if close[s].isna().all()]
    if missing:
        raise RuntimeError(f"no history for {', '.join(missing)}")
    start = max(close[s].first_valid_index() for s in stocks)
    df = close.loc[start:end].ffill().dropna()
    return df[stocks].pct_change().iloc[1:], df[VIX].iloc[1:]


def fit_model(rp, vix):
    """Fit on decimal portfolio returns `rp` and VIX levels (same dates). Returns what the
    daily update needs, including σ² for the day after the last fitting date."""
    r, X = rp.values * 100, vix_term(vix.values)[:, None]
    p, s0, nll = gjr_fit(r, X)
    s2 = gjr_filter(p, r, X, s0)
    z_q = float(np.quantile(r / np.sqrt(s2), VAR_LEVEL))
    s2_next = next_variance(p, r[-1], vix.values[-1], s2[-1])
    omega, alpha, gamma, beta, delta = (float(v) for v in p)
    return {
        "params": {"omega": omega, "alpha": alpha, "gamma": gamma, "beta": beta,
                   "delta": delta},
        "s0": float(s0), "neg_log_lik": float(nll),
        "persistence": alpha + gamma / 2 + beta,
        "z_quantile": z_q, "var_level": VAR_LEVEL,
        "fit_start": str(rp.index[0].date()), "fit_end": str(rp.index[-1].date()),
        "n_obs": int(len(rp)),
        "last_date": str(rp.index[-1].date()),
        "sigma2_next": s2_next, "var_99_next": var_99(z_q, s2_next),
    }


def param_vector(model):
    p = model["params"]
    return [p["omega"], p["alpha"], p["gamma"], p["beta"], p["delta"]]
