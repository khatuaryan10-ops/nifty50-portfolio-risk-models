"""Fresh price downloads, daily total returns and data checks.

Returns are only ever computed from one fresh yfinance download with auto_adjust=False.
Yahoo's Close is split-adjusted across the whole download (but not dividend-adjusted), so

    r_t = (Close_t + Dividend_t) / Close_{t-1} - 1

is the total return, and a split inside the window needs no extra handling.
"""

import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf

NIFTY, VIX = "^NSEI", "^INDIAVIX"
IST = timezone(timedelta(hours=5, minutes=30))   # India has no daylight saving
CLOSE_HOUR_IST = 16          # NSE closes 15:30; before 16:00 today's prices are not final
MIN_COVERAGE = 0.9           # share of held stocks that must have a close on a real session
MAX_DAILY_MOVE = 0.20        # flag any stock moving more than this in a day
STALE_DAYS = 3               # flag a close unchanged for this many trading days


def last_complete_date(now):
    """Latest date whose NSE closing prices are final."""
    now = now.astimezone(IST)
    day = now.date() if now.hour >= CLOSE_HOUR_IST else now.date() - timedelta(days=1)
    return pd.Timestamp(day)


def now_ist():
    return datetime.now(IST)


def _yf_download(symbols, start, end, auto_adjust, threads=True):
    try:
        return yf.download(symbols, start=start, end=end, auto_adjust=auto_adjust,
                           actions=True, progress=False, threads=threads)
    except Exception:            # rate limits and network errors: treat as empty, retry
        return pd.DataFrame()


def _field(raw, field, symbols):
    if raw.empty or field not in raw.columns.get_level_values(0):
        return pd.DataFrame(index=raw.index, columns=symbols, dtype=float)
    out = raw[field].reindex(columns=symbols).astype(float)
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    out.index = out.index.normalize()
    return out


def download(symbols, start, end=None, auto_adjust=False, retries=3):
    """{"close", "dividends", "splits"}: DataFrames of dates x symbols (".NS" kept).
    Symbols that come back empty are retried one at a time with a back-off, as in
    load_prices / load_data; any still missing stay as all-NaN columns for the caller
    to flag."""
    raw = _yf_download(symbols, start, end, auto_adjust)
    frames = {f: _field(raw, f, symbols) for f in ["Close", "Dividends", "Stock Splits"]}
    for attempt in range(retries):
        failed = [s for s in symbols if frames["Close"][s].isna().all()]
        if not failed:
            break
        time.sleep(2 ** attempt)
        retry = _yf_download(failed, start, end, auto_adjust, threads=False)
        for f in frames:
            got = _field(retry, f, failed)
            frames[f] = frames[f].reindex(frames[f].index.union(got.index))
            frames[f][failed] = got.reindex(frames[f].index)
    return {"close": frames["Close"].sort_index(),
            "dividends": frames["Dividends"].fillna(0.0).sort_index(),
            "splits": frames["Stock Splits"].fillna(0.0).sort_index()}


def total_returns(close, dividends):
    """(Close_t + Dividend_t) / Close_{t-1} - 1, where Close_{t-1} is the stock's last close
    in the same download, so a stock that missed a day earns the whole move when it
    returns. NaN where the stock has no close that day (the caller applies 0%)."""
    prev = close.ffill().shift(1)
    return (close + dividends.reindex_like(close).fillna(0.0)) / prev - 1


def trading_days(close, stocks, last_complete):
    """Dates that are real NSE sessions: Nifty has a close, at least MIN_COVERAGE of the
    stocks have one, and prices moved (Yahoo sometimes repeats the previous row on a
    holiday). Returns the sessions and any dates skipped as incomplete."""
    c = close[close.index <= last_complete]
    has_nifty = c[NIFTY].notna()
    covered = c[stocks].notna().mean(axis=1) >= MIN_COVERAGE
    incomplete = c.index[has_nifty & ~covered]
    c = c[has_nifty & covered]
    cols = stocks + [NIFTY]
    same = (c[cols] == c[cols].shift(1)) | c[cols].isna()
    repeat = same.all(axis=1)
    repeat.iloc[:1] = False
    return c.index[~repeat], incomplete


def validate(close, returns, dates, stocks):
    """Warnings for the given dates: missing prices, moves above MAX_DAILY_MOVE and
    closes unchanged for STALE_DAYS trading days."""
    warnings = []
    same = close[stocks].eq(close[stocks].shift(1))
    stale = same.rolling(STALE_DAYS - 1).sum() == STALE_DAYS - 1
    for d in dates:
        day = d.date()
        for s in stocks:
            r = returns.at[d, s]
            if pd.isna(close.at[d, s]):
                warnings.append(f"{day} {s}: no price, 0% return applied")
            elif abs(r) > MAX_DAILY_MOVE:
                warnings.append(f"{day} {s}: daily move {r:+.1%} above {MAX_DAILY_MOVE:.0%}")
            if stale.at[d, s]:
                warnings.append(f"{day} {s}: close unchanged for {STALE_DAYS}+ trading days")
    return warnings
