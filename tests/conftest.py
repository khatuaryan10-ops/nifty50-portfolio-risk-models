"""A small fake market standing in for yfinance, and a tracker initialised on it."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from daily_tracker.data import IST
from daily_tracker.init import run_init
from daily_tracker.store import Store

STOCKS = ["AAA", "BBB", "CCC"]
TARGETS = {"Min variance": [0.5, 0.5, 0.0], "Max Sharpe": [0.2, 0.3, 0.5]}
INCEPTION = "2026-09-24"


def at(day):
    """23:30 IST on `day`, when the scheduled job runs."""
    return datetime.fromisoformat(day).replace(hour=23, minute=30, tzinfo=IST)


class FakeMarket:
    """Closes as Yahoo reports them today (split-adjusted), plus dividends and splits.
    `fetch` has the signature of daily_tracker.data.download."""

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range("2026-09-01", "2026-10-30")
        symbols = [s + ".NS" for s in STOCKS] + ["^NSEI"]
        rets = rng.normal(0.0005, 0.01, (len(dates), len(symbols)))
        self.close = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), dates, symbols)
        self.close["^INDIAVIX"] = 15 + rng.normal(0, 0.5, len(dates))
        self.dividends = pd.DataFrame(0.0, dates, self.close.columns)
        self.splits = pd.DataFrame(0.0, dates, self.close.columns)
        self.calls = []

    def fetch(self, symbols, start, end, auto_adjust=False):
        self.calls.append((start, end))
        sl = slice(pd.Timestamp(start), pd.Timestamp(end) - pd.Timedelta(days=1))
        return {"close": self.close.loc[sl, symbols].copy(),
                "dividends": self.dividends.loc[sl, symbols].copy(),
                "splits": self.splits.loc[sl, symbols].copy()}

    def total_returns(self, stock):
        c = self.close[stock + ".NS"]
        return (c + self.dividends[stock + ".NS"]) / c.shift(1) - 1


def fake_history(stocks, end):
    rng = np.random.default_rng(1)
    dates = pd.bdate_range(end=end, periods=500)
    rets = pd.DataFrame(rng.normal(0.0004, 0.012, (500, len(stocks))), dates, stocks)
    vix = pd.Series(15 + rng.normal(0, 2, 500).cumsum() * 0.1, dates).clip(10)
    return rets, vix


@pytest.fixture
def market():
    return FakeMarket()


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path)
    s.optimal_csv.parent.mkdir(parents=True)
    pd.DataFrame([[0.1, 0.01, 0.1, 1.0] + w for w in TARGETS.values()],
                 index=list(TARGETS), columns=["Return", "Variance", "Volatility", "Sharpe"]
                 + STOCKS).to_csv(s.optimal_csv)
    run_init(s, at(INCEPTION), history=fake_history)
    return s
