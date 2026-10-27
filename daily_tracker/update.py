"""`python tracker.py update`: add new trading days and correct recent revisions.

Each run makes one fresh download covering the last ~10 trading days (widened after a
gap), recomputes total returns for the last REVISION_DAYS stored dates plus any new ones,
and replays the accounting from the earliest date that is new or revised. Nothing is
computed from previously stored prices; stored *returns* (data/returns.csv) are the
record, and only the revision check may overwrite them.
"""

import pandas as pd

from . import accounting as acc
from .benchmark import TRI_FILE, benchmark_returns
from .data import (IST, NIFTY, VIX, download, last_complete_date, total_returns,
                   trading_days, validate)
from .risk import next_variance, param_vector, var_99
from .store import REBALANCE_COLS, TRACKER_COLS

LOOKBACK_DAYS = 16          # calendar days, about 10 trading days
REVISION_DAYS = 5
REVISION_TOL = 0.0005       # 0.05%: larger return differences count as revisions
HOLD_SESSIONS = 2           # sessions an incomplete date waits before going in with gaps
BENCH_COL, VIX_COL = "benchmark", "india_vix"


def run_update(store, now, fetch=download, weekly=None):
    """Run the daily update and write a job_log row whatever happens. Returns the status."""
    warnings, errors, added, revisions = [], [], 0, []
    try:
        added, revisions = _update(store, now, fetch, warnings)
    except Exception as e:
        errors.append(f"{type(e).__name__}: {e}")
    if not errors and weekly is not None and now.astimezone(IST).weekday() == 4:
        try:
            weekly(store, now, warnings)
        except Exception as e:
            errors.append(f"weekly re-optimisation: {type(e).__name__}: {e}")
    status = ("error" if errors else "warning" if warnings
              else "ok" if added or revisions else "no new data")
    store.append_log({"run_time": now.isoformat(timespec="seconds"), "command": "update",
                      "status": status, "dates_added": added,
                      "revisions": " | ".join(revisions), "warnings": " | ".join(warnings),
                      "errors": " | ".join(errors)})
    return status


def _update(store, now, fetch, warnings):
    cfg, models = store.load_config(), store.load_models()
    inception = pd.Timestamp(cfg["inception_date"])
    targets = {k: pd.Series(p["target_weights"], dtype=float)
               for k, p in cfg["portfolios"].items()}
    stocks = sorted(set().union(*(t.index for t in targets.values())))
    stored = store.read_returns()
    timeline = [inception] + list(stored.index)
    check = list(stored.index[-REVISION_DAYS:])
    anchor = timeline[len(timeline) - len(check) - 1]      # day before the first check

    # ---- one fresh download ----
    last_complete = last_complete_date(now)
    need = anchor if anchor in stored.index else anchor - pd.Timedelta(days=5)
    start = min(last_complete - pd.Timedelta(days=LOOKBACK_DAYS), need)
    if need < last_complete - pd.Timedelta(days=LOOKBACK_DAYS):
        warnings.append(f"gap since {timeline[-1].date()}: download widened to start "
                        f"{start.date()}")
    symbols = [s + ".NS" for s in stocks] + [NIFTY, VIX]
    raw = fetch(symbols, start.strftime("%Y-%m-%d"),
                (last_complete + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    raw = {k: f.rename(columns=lambda c: c.replace(".NS", "")) for k, f in raw.items()}
    failed = [s for s in stocks if raw["close"][s].isna().all()]
    if failed:
        warnings.append(f"download failed after retries: {', '.join(failed)}")
    _save_raw(store, raw, last_complete, now)

    # A session with too few stock prices waits for a later run (Yahoo often fills them
    # in), and so does everything after it; after HOLD_SESSIONS more sessions it goes in
    # with 0% for the missing stocks. Stored dates stay sessions for the revision check.
    days, incomplete = trading_days(raw["close"], stocks, last_complete)
    days = days.union(incomplete[incomplete.isin(stored.index)])
    cutoff = None
    for d in incomplete[incomplete > timeline[-1]]:
        later = int((days > d).sum())
        if later < HOLD_SESSIONS:
            cutoff = d
            warnings.append(f"{d.date()}: too few stock prices yet, waiting for the next run")
            break
        days = days.union([d])
        warnings.append(f"{d.date()}: stock prices still missing after {later} sessions, "
                        "processed with 0% for the missing stocks")
    if not (days <= anchor).any():
        raise RuntimeError(f"download has no session on or before {anchor.date()}")
    close, divs = raw["close"].loc[days], raw["dividends"].loc[days]
    rets = total_returns(close[stocks], divs[stocks])
    rets[BENCH_COL] = benchmark_returns(cfg["benchmark"]["source"], close, days,
                                        store.path(TRI_FILE))
    rets[VIX_COL] = close[VIX]
    fresh = rets[rets.index > anchor]
    if cutoff is not None:
        fresh = fresh[fresh.index < cutoff]

    # ---- revision check on the last stored dates ----
    revised, revisions = [], []
    for d in check:
        if d not in fresh.index:
            warnings.append(f"{d.date()}: stored date not in fresh download, kept as stored")
            continue
        old, new = stored.loc[d], fresh.loc[d, stored.columns]
        diff = (new - old).abs()
        tol = pd.Series(REVISION_TOL, index=diff.index)
        tol[VIX_COL] = REVISION_TOL * abs(old[VIX_COL])         # VIX is a level
        changed = (diff > tol) | (old.isna() != new.isna())
        if changed.any():
            revised.append(d)
            revisions.append(f"{d.date()}: " + ", ".join(
                f"{c} {old[c]:.4%} -> {new[c]:.4%}" if c != VIX_COL
                else f"{c} {old[c]:.2f} -> {new[c]:.2f}" for c in changed.index[changed]))
    new_dates = list(fresh.index[fresh.index > timeline[-1]])
    if not revised and not new_dates:
        return 0, []
    from_date = min(revised + new_dates)

    for d in revised + new_dates:
        if pd.isna(fresh.at[d, BENCH_COL]):
            warnings.append(f"{d.date()}: no benchmark level, 0% applied")
        if pd.isna(fresh.at[d, VIX_COL]):
            warnings.append(f"{d.date()}: no India VIX, previous close used")
    warnings += validate(close, rets, revised + new_dates, stocks)

    keep = stored[(stored.index >= from_date) & ~stored.index.isin(fresh.index)]
    returns = pd.concat([stored[stored.index < from_date], fresh[fresh.index >= from_date],
                         keep]).sort_index()
    returns[VIX_COL] = returns[VIX_COL].ffill()

    # ---- replay accounting from the earliest new or revised date ----
    tracker = store.read("tracker.csv", TRACKER_COLS)
    weights = store.read("weights.csv")
    rebalances = store.read("rebalances.csv", REBALANCE_COLS)
    prev_date = max(d for d in timeline if d < from_date)
    dates = returns.index[returns.index >= from_date]
    rows, wrows, rrows = [], [], []
    for key, target in targets.items():
        name = cfg["portfolios"][key]["name"]
        w = weights[(weights.portfolio == name) & (weights.date == prev_date)]
        r, w_, rb = _replay(name, _history(tracker, name, prev_date), returns, dates,
                            prev_date, cfg["start_value"], w=w.iloc[0][target.index].astype(float),
                            target=target, model=models[key])
        rows += r
        wrows += w_
        rrows += rb
        models[key]["last_date"] = str(dates[-1].date())
        models[key]["sigma2_next"] = r[-1]["sigma2_next"]
        models[key]["var_99_next"] = r[-1]["var_99_next"]
    bench = cfg["benchmark"]["label"]
    r, _, _ = _replay(bench, _history(tracker, bench, prev_date), returns, dates, prev_date,
                      cfg["start_value"])
    rows += r

    store.write_returns(returns)
    store.write("tracker.csv", _splice(tracker, rows, from_date)[TRACKER_COLS])
    store.write("weights.csv", _splice(weights, wrows, from_date).fillna(0.0))
    store.write("rebalances.csv", _splice(rebalances, rrows, from_date)[REBALANCE_COLS])
    store.save_models(models)
    return len(new_dates), revisions


def _history(tracker, name, prev_date):
    return tracker[(tracker.portfolio == name) & (tracker.date <= prev_date)].sort_values("date")


def _splice(old, rows, from_date):
    """Rows before `from_date` as stored, then the replayed rows."""
    new = pd.DataFrame(rows)
    out = pd.concat([old[old.date < from_date], new], ignore_index=True)
    return out.sort_values("date", kind="stable").reset_index(drop=True)


def _replay(name, hist, returns, dates, prev_date, start_value, w=None, target=None,
            model=None):
    """Tracker rows for one portfolio over `dates`, starting from its row on prev_date.
    Without weights it tracks the benchmark column (no VaR, drift or rebalancing)."""
    last = hist.iloc[-1]
    value, peak, mdd = last.value, hist.value.max(), last.max_drawdown
    past = hist.daily_return.dropna().tolist()
    s2, var_prev = last.sigma2_next, last.var_99_next
    rows, wrows, rrows = [], [], []
    for d in dates:
        flags = []
        if w is None:
            rp = returns.at[d, BENCH_COL]
            if pd.isna(rp):
                flags.append("no benchmark level, 0% applied")
                rp = 0.0
        else:
            r = returns.loc[d, w.index]
            if r.isna().any():
                flags.append("no price, 0% applied: " + ", ".join(r.index[r.isna()]))
            rp, w = acc.step(w, r.fillna(0.0))
        value *= 1 + rp
        peak = max(peak, value)
        mdd = min(mdd, value / peak - 1)
        past.append(rp)
        row = {"date": d, "portfolio": name, "value": value, "daily_return": rp,
               "cumulative_return": value / start_value - 1, "vol_60d": acc.vol_60d(past),
               "drawdown": value / peak - 1, "max_drawdown": mdd}
        if w is not None:
            breach = int(-rp > var_prev)          # loss beyond last evening's VaR
            s2 = next_variance(param_vector(model), rp * 100, returns.at[d, VIX_COL], s2)
            var_prev = var_99(model["z_quantile"], s2)
            if acc.is_quarter_start(d, prev_date):  # after the day's return
                before, w = w, target.copy()
                to = acc.turnover(before, w)
                rrows += [{"date": d, "portfolio": name, "ticker": t,
                           "weight_before": before[t], "weight_after": w[t], "turnover": to}
                          for t in target.index]
                flags.append(f"rebalanced to target, turnover {to:.2%}")
            row.update(var_99_next=var_prev, breach=breach,
                       max_weight_drift=acc.max_drift(w, target), sigma2_next=s2)
            wrows.append({"date": d, "portfolio": name, **w.to_dict()})
        row["flags"] = "; ".join(flags)
        rows.append(row)
        prev_date = d
    return rows, wrows, rrows


def _save_raw(store, raw, last_complete, now):
    """Raw Close, Dividends and Stock Splits, for audit only: never read back for returns."""
    frames = [raw["close"].stack(future_stack=True).rename("close"),
              raw["dividends"].stack(future_stack=True).rename("dividends"),
              raw["splits"].stack(future_stack=True).rename("stock_splits")]
    new = pd.concat(frames, axis=1).dropna(subset=["close"]).reset_index()
    new.columns = ["date", "ticker", "close", "dividends", "stock_splits"]
    new = new[new.date <= last_complete]
    new["downloaded_at"] = now.isoformat(timespec="seconds")
    old = store.read("prices_raw.csv")
    if not old.empty:
        key = old.date.astype(str) + old.ticker
        old = old[~key.isin(new.date.astype(str) + new.ticker)]
    out = pd.concat([old, new], ignore_index=True).sort_values(["date", "ticker"])
    store.write("prices_raw.csv", out)
