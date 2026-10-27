"""Static dashboard: data/ CSVs embedded into one self-contained HTML page (site/index.html),
published to GitHub Pages by the daily workflow."""

import json
from pathlib import Path

import pandas as pd

from .store import LOG_COLS, REBALANCE_COLS, REOPT_COLS, TRACKER_COLS

TEMPLATE = Path(__file__).with_name("dashboard_template.html")
SHORT = {"Min variance": "Min var", "Max Sharpe": "Max Sharpe"}
REPO_URL = "https://github.com/khatuaryan10-ops/nifty50-portfolio-risk-models"


def _records(df):
    """JSON-ready rows: dates as YYYY-MM-DD, NaN as null."""
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].dt.strftime("%Y-%m-%d")
    return json.loads(df.to_json(orient="records"))


def _series(tracker, name, short):
    t = tracker[tracker.portfolio == name].sort_values("date").copy()
    t["var_prev"] = t.var_99_next.shift(1)        # the VaR each day's loss is judged against
    cols = ["date", "value", "daily_return", "cumulative_return", "vol_60d", "drawdown",
            "max_drawdown", "var_99_next", "var_prev", "breach", "max_weight_drift", "flags"]
    return {"name": name, "short": short, "rows": _records(t[cols])}


def build(store, now, out_dir):
    cfg = store.load_config()
    tracker = store.read("tracker.csv", TRACKER_COLS)
    weights = store.read("weights.csv")
    portfolios = []
    for p in cfg["portfolios"].values():
        s = _series(tracker, p["name"], SHORT.get(p["name"], p["name"]))
        w = weights[weights.portfolio == p["name"]].sort_values("date").iloc[-1]
        s["weights"] = [{"ticker": t, "target": float(v), "current": float(w[t])}
                        for t, v in p["target_weights"].items()]
        portfolios.append(s)
    bench = cfg["benchmark"]["label"]

    rebalances = store.read("rebalances.csv", REBALANCE_COLS)
    rebalances = rebalances.drop_duplicates(["date", "portfolio"])[["date", "portfolio", "turnover"]]
    reopt = store.read("reoptimised_weights.csv", REOPT_COLS)
    reopt = reopt.drop_duplicates(["date", "portfolio"])[["date", "portfolio", "turnover_vs_target"]] \
                 .rename(columns={"turnover_vs_target": "turnover"})
    log = store.read("job_log.csv", LOG_COLS).tail(30)

    payload = {
        "generated_at": now.strftime("%d %b %Y %H:%M IST"),
        "inception": cfg["inception_date"],
        "last_date": tracker.date.max().strftime("%Y-%m-%d"),
        "repo_url": REPO_URL,
        "portfolios": portfolios,
        "benchmark": _series(tracker, bench, "Nifty"),
        "rebalances": _records(rebalances.sort_values("date", ascending=False)),
        "reopt": _records(reopt.sort_values("date", ascending=False).head(20)),
        "log": json.loads(log.to_json(orient="records")),
        "table": _records(tracker[TRACKER_COLS].drop(columns="sigma2_next")),
    }
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*DATA*/", data)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(html, encoding="utf-8")
    (out / ".nojekyll").write_text("")
    return out / "index.html"
