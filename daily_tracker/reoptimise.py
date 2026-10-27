"""Friday stability record: what the optimiser would choose today on the trailing 5 years.
Recorded only, never traded; turnover versus the frozen targets measures how unstable the
optimal weights are."""

import numpy as np
import pandas as pd

from nifty50_efficient_frontier import (MAX_WEIGHT, PERIOD, TICKERS, TRADING_DAYS,
                                        load_prices, max_sharpe, min_variance_for_target)

from .accounting import clean_weights, turnover
from .data import IST
from .store import REOPT_COLS


def reoptimised_weights(prices):
    """Min-variance and max-Sharpe weights from prices, solved as in
    nifty50_efficient_frontier.py (0 <= w <= MAX_WEIGHT)."""
    rets = prices.pct_change().dropna()
    mu = rets.mean().values * TRADING_DAYS
    cov = rets.cov().values * TRADING_DAYS
    n = len(mu)
    w_mv = np.clip(min_variance_for_target(mu, cov, MAX_WEIGHT, None, np.full(n, 1 / n)).x,
                   0, MAX_WEIGHT)
    w_ms = np.clip(max_sharpe(mu, cov, MAX_WEIGHT, w_mv).x, 0, MAX_WEIGHT)
    return {"min_variance": clean_weights(pd.Series(w_mv, index=prices.columns)),
            "max_sharpe": clean_weights(pd.Series(w_ms, index=prices.columns))}


def run_weekly(store, now, warnings, prices=None):
    today = pd.Timestamp(now.astimezone(IST).date())
    done = store.read("reoptimised_weights.csv", REOPT_COLS)
    if (done.date == today).any():
        return
    cfg = store.load_config()
    prices = load_prices(TICKERS, PERIOD) if prices is None else prices
    rows = []
    for key, w in reoptimised_weights(prices).items():
        target = pd.Series(cfg["portfolios"][key]["target_weights"], dtype=float)
        tickers = w.index.union(target.index)
        w, target = w.reindex(tickers, fill_value=0.0), target.reindex(tickers, fill_value=0.0)
        to = turnover(target, w)
        not_in_data = sorted(set(target[target > 0].index) - set(prices.columns))
        if not_in_data:
            warnings.append(f"re-optimisation: no 5y history for {', '.join(not_in_data)}")
        rows += [{"date": today, "portfolio": cfg["portfolios"][key]["name"], "ticker": t,
                  "weight": w[t], "target_weight": target[t], "turnover_vs_target": to,
                  "data_start": prices.index[0], "data_end": prices.index[-1]}
                 for t in tickers if w[t] > 0 or target[t] > 0]
    new = pd.DataFrame(rows, columns=REOPT_COLS)
    for c in ["data_start", "data_end"]:
        new[c] = pd.to_datetime(new[c]).dt.strftime("%Y-%m-%d")
    store.write("reoptimised_weights.csv", pd.concat([done, new], ignore_index=True))
