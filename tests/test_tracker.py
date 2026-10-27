import numpy as np
import pandas as pd
import pytest

from daily_tracker import data
from daily_tracker.data import total_returns, validate
from daily_tracker.update import run_update

from conftest import INCEPTION, STOCKS, at

MIN_VAR = pd.Series({"AAA": 0.5, "BBB": 0.5})
MAX_SHARPE = pd.Series({"AAA": 0.2, "BBB": 0.3, "CCC": 0.5})


def stored_returns(store):
    return store.read_returns()


def tracker_row(store, day, portfolio):
    t = store.read("tracker.csv")
    return t[(t.date == day) & (t.portfolio == portfolio)].iloc[0]


def expected_values(market, target, end):
    """Independent re-computation from the market's prices: drift, quarterly rebalance."""
    w, value, prev, out = target.copy(), 100.0, pd.Timestamp(INCEPTION), {}
    for d in market.close.loc[pd.Timestamp(INCEPTION) + pd.Timedelta(days=1):end].index:
        r = pd.Series({s: market.total_returns(s)[d] for s in w.index})
        rp = float((w * r).sum())
        value *= 1 + rp
        w = w * (1 + r) / (1 + rp)
        if d.quarter != prev.quarter:
            w = target.copy()
        out[d], prev = value, d
    return out


def test_dividend_day_total_return(store, market):
    close = pd.DataFrame({"X": [100.0, 98.0]})
    divs = pd.DataFrame({"X": [0.0, 3.0]})
    assert total_returns(close, divs)["X"].iloc[1] == pytest.approx(0.01)

    # End to end: AAA goes ex-dividend on 29 Sep; Close falls but the holder gets 3.
    day = pd.Timestamp("2026-09-29")
    market.dividends.loc[day, "AAA.NS"] = 3.0
    run_update(store, at("2026-09-30"), market.fetch)
    prev = market.close.loc[pd.Timestamp("2026-09-28"), "AAA.NS"]
    expected = (market.close.loc[day, "AAA.NS"] + 3.0) / prev - 1
    assert stored_returns(store).loc[day, "AAA"] == pytest.approx(expected)


SPLIT = pd.Timestamp("2026-10-06")


def seen_on(market, run_day):
    """Yahoo as seen on `run_day`: before a 2:1 split in BBB the closes are unadjusted;
    from the split on, every close in a download is divided by 2 and Stock Splits = 2."""
    def fetch(symbols, start, end):
        out = market.fetch(symbols, start, end)
        if pd.Timestamp(run_day) < SPLIT:
            out["close"]["BBB.NS"] *= 2
        else:
            out["splits"].loc[out["splits"].index == SPLIT, "BBB.NS"] = 2.0
        return out
    return fetch


def test_split_in_window_no_fake_jump(store, market):
    run_update(store, at("2026-10-02"), seen_on(market, "2026-10-02"))
    before = stored_returns(store)
    run_update(store, at("2026-10-07"), seen_on(market, "2026-10-07"))
    after = stored_returns(store)

    split_day = after.loc[SPLIT, "BBB"]
    assert split_day == pytest.approx(market.total_returns("BBB")[SPLIT])
    assert abs(split_day) < 0.05                         # not -50%
    # earlier returns were already right in unadjusted prices: nothing revised
    pd.testing.assert_frame_equal(before, after.loc[before.index])
    log = store.read("job_log.csv").iloc[-1]
    assert pd.isna(log.revisions)
    assert "daily move" not in str(log.warnings)
    # direct check of the formula on a download that contains the split
    window = seen_on(market, "2026-10-07")(["BBB.NS"], "2026-10-01", "2026-10-08")
    r = total_returns(window["close"], window["dividends"])
    assert validate(window["close"], r, r.index[1:], ["BBB.NS"]) == []


def test_update_twice_adds_no_duplicates(store, market):
    run_update(store, at("2026-10-02"), market.fetch)
    first = store.read("tracker.csv")
    status = run_update(store, at("2026-10-02"), market.fetch)
    second = store.read("tracker.csv")

    assert status == "no new data"
    pd.testing.assert_frame_equal(first, second)
    assert not second.duplicated(["date", "portfolio"]).any()
    assert stored_returns(store).index.is_unique
    # 6 sessions after inception, 3 series, plus the 3 inception rows
    assert len(second) == 3 * 7


def test_revised_price_in_last_5_days_is_corrected(store, market):
    run_update(store, at("2026-10-02"), market.fetch)
    day = pd.Timestamp("2026-09-30")
    market.close.loc[day, "AAA.NS"] *= 1.01              # Yahoo corrects a close
    run_update(store, at("2026-10-05"), market.fetch)

    rets = stored_returns(store)
    for d in [day, pd.Timestamp("2026-10-01")]:          # both returns touch that close
        assert rets.loc[d, "AAA"] == pytest.approx(market.total_returns("AAA")[d])
    log = store.read("job_log.csv").iloc[-1]
    assert "2026-09-30" in log.revisions and log.dates_added == 1
    expected = expected_values(market, MIN_VAR, pd.Timestamp("2026-10-05"))
    for d, v in expected.items():
        assert tracker_row(store, d, "Min variance").value == pytest.approx(v)


def test_rebalance_resets_to_target_and_logs_turnover(store, market):
    run_update(store, at("2026-10-02"), market.fetch)
    q4 = pd.Timestamp("2026-10-01")                      # first trading day of Q4

    # drifted weights going into the rebalance, recomputed independently
    w = MAX_SHARPE.copy()
    for d in pd.bdate_range("2026-09-25", q4):
        r = pd.Series({s: market.total_returns(s)[d] for s in w.index})
        w = w * (1 + r) / (1 + (w * r).sum())

    reb = store.read("rebalances.csv")
    assert set(reb.date) == {q4}
    ms = reb[reb.portfolio == "Max Sharpe"].set_index("ticker")
    assert np.allclose(ms.weight_before[w.index], w)
    assert np.allclose(ms.weight_after[w.index], MAX_SHARPE)
    assert ms.turnover.iloc[0] == pytest.approx(0.5 * (w - MAX_SHARPE).abs().sum())

    weights = store.read("weights.csv")
    on_q4 = weights[(weights.date == q4) & (weights.portfolio == "Max Sharpe")].iloc[0]
    assert np.allclose(on_q4[list(MAX_SHARPE.index)].astype(float), MAX_SHARPE)
    assert tracker_row(store, q4, "Max Sharpe").max_weight_drift == pytest.approx(0)
    assert "rebalanced" in tracker_row(store, q4, "Max Sharpe")["flags"]


def test_missing_price_carries_weight_at_zero(store, market, monkeypatch):
    monkeypatch.setattr(data, "MIN_COVERAGE", 0.6)       # 2 of 3 toy stocks still a session
    gap = pd.Timestamp("2026-09-29")
    market.close.loc[gap, "CCC.NS"] = np.nan
    assert run_update(store, at("2026-09-30"), market.fetch) == "warning"

    assert pd.isna(stored_returns(store).loc[gap, "CCC"])
    assert "CCC" in tracker_row(store, gap, "Max Sharpe")["flags"]
    # the day after, CCC earns the whole move since its last close
    nxt = pd.Timestamp("2026-09-30")
    c = market.close["CCC.NS"]
    assert stored_returns(store).loc[nxt, "CCC"] == pytest.approx(
        c[nxt] / c[pd.Timestamp("2026-09-28")] - 1)


def test_gap_widens_the_download_window(store, market):
    run_update(store, at("2026-10-30"), market.fetch)    # first run 26 sessions late
    assert market.calls[-1][0] <= "2026-09-19"
    sessions = market.close.loc["2026-09-25":"2026-10-30"].index
    assert list(stored_returns(store).index) == list(sessions)
    assert "widened" in store.read("job_log.csv").iloc[-1].warnings


def test_incomplete_latest_date_waits_and_is_filled_later(store, market):
    day = pd.Timestamp("2026-10-01")
    late = market.close.loc[day].copy()
    market.close.loc[day, ["AAA.NS", "BBB.NS"]] = np.nan  # Yahoo has not filled these yet
    run_update(store, at("2026-10-01"), market.fetch)
    assert stored_returns(store).index.max() == pd.Timestamp("2026-09-30")

    market.close.loc[day] = late                         # filled in by the next evening
    run_update(store, at("2026-10-02"), market.fetch)
    rets = stored_returns(store)
    assert day in rets.index and rets.loc[day].notna().all()
    assert rets.loc[day, "AAA"] == pytest.approx(market.total_returns("AAA")[day])


def test_dashboard_embeds_the_tracker_data(store, market, tmp_path):
    import json
    import re

    from daily_tracker.dashboard import build

    run_update(store, at("2026-10-02"), market.fetch)
    page = build(store, at("2026-10-02"), tmp_path / "site").read_text(encoding="utf-8")
    data = json.loads(re.search(r'type="application/json">(.*?)</script>', page, re.S).group(1))
    t = store.read("tracker.csv")
    last = t[t.portfolio == "Max Sharpe"].iloc[-1]
    assert data["last_date"] == "2026-10-02"
    assert data["portfolios"][1]["rows"][-1]["value"] == pytest.approx(last.value)
    assert data["rebalances"][0]["date"] == "2026-10-01"
