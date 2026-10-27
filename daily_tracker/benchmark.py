"""Benchmark series.

Default: ^NSEI from the same fresh yfinance download as the stocks. It is the Nifty 50
*price* index, so it excludes dividends and is labelled that way everywhere.

Opt-in: the Nifty 50 Total Return Index from a CSV you download from niftyindices.com
(Reports > Historical data > Total Return Index) and keep at data/nifty50_tri.csv with
columns Date, TRI. niftyindices.com has no public API and blocks automated requests, so
the tracker does not scrape it. Choose the source at init; switching later would splice
two different series.
"""

import pandas as pd

SOURCES = {
    "nsei": "Nifty 50 price index (excludes dividends)",
    "tri": "Nifty 50 TRI (total return index)",
}
TRI_FILE = "nifty50_tri.csv"


def read_tri(path):
    """TRI levels indexed by date, from the manually maintained CSV."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    level = next(c for c in df.columns if c != "date")
    s = pd.Series(pd.to_numeric(df[level], errors="coerce").values,
                  index=pd.to_datetime(df["date"], dayfirst=True).dt.normalize())
    return s.dropna().sort_index()


def benchmark_returns(source, close, dates, tri_path):
    """Daily benchmark returns on `dates`; NaN where no level is available (the caller
    applies 0%, flags it, and the revision check fixes it once the level arrives)."""
    if source == "nsei":
        level = close["^NSEI"]
    else:
        level = read_tri(tri_path).reindex(close.index) if tri_path.exists() \
            else pd.Series(float("nan"), index=close.index)
    level = level.loc[dates]
    return level / level.ffill().shift(1) - 1
