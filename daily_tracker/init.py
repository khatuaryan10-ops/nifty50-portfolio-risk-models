"""`python tracker.py init`: freeze target weights and fit the VaR models, once."""

import pandas as pd

from .accounting import clean_weights
from .benchmark import SOURCES, TRI_FILE, read_tri
from .data import last_complete_date
from .risk import fit_history, fit_model
from .store import TRACKER_COLS

PORTFOLIOS = {"min_variance": "Min variance", "max_sharpe": "Max Sharpe"}
STATS = ["Return", "Variance", "Volatility", "Sharpe"]
START_VALUE = 100.0


def run_init(store, now, benchmark="nsei", force=False, history=fit_history):
    if store.config_path.exists() and not force:
        raise SystemExit(f"{store.config_path} exists; the targets are frozen. "
                         "Use --force to start a new tracking record.")
    if benchmark == "tri" and not store.path(TRI_FILE).exists():
        raise SystemExit(f"--benchmark tri needs {store.path(TRI_FILE)} (Date, TRI), "
                         "downloaded from niftyindices.com")
    inception = last_complete_date(now)
    if benchmark == "tri" and read_tri(store.path(TRI_FILE)).index.max() < inception:
        raise SystemExit(f"{TRI_FILE} must include the inception date {inception.date()}")

    optimal = pd.read_csv(store.optimal_csv, index_col=0).drop(columns=STATS)
    targets = {key: clean_weights(optimal.loc[name]) for key, name in PORTFOLIOS.items()}
    held = sorted(set().union(*(t.index for t in targets.values())))
    rets, vix = history(held, inception)

    models, rows, wrows = {}, [], []
    for key, w in targets.items():
        m = fit_model(rets[w.index] @ w, vix)
        m["name"] = PORTFOLIOS[key]
        m["inception_state"] = {k: m[k] for k in ["last_date", "sigma2_next", "var_99_next"]}
        models[key] = m
        rows.append({"date": inception, "portfolio": PORTFOLIOS[key], "value": START_VALUE,
                     "cumulative_return": 0.0, "drawdown": 0.0, "max_drawdown": 0.0,
                     "var_99_next": m["var_99_next"], "max_weight_drift": 0.0,
                     "sigma2_next": m["sigma2_next"], "flags": "inception"})
        wrows.append({"date": inception, "portfolio": PORTFOLIOS[key], **w.to_dict()})
    label = SOURCES[benchmark]
    rows.append({"date": inception, "portfolio": label, "value": START_VALUE,
                 "cumulative_return": 0.0, "drawdown": 0.0, "max_drawdown": 0.0,
                 "flags": "inception"})

    store.save_config({
        "inception_date": str(inception.date()),
        "start_value": START_VALUE,
        "source": str(store.optimal_csv.relative_to(store.root).as_posix()),
        "rebalance": "to target weights on the first trading day of each quarter, "
                     "after that day's return",
        "benchmark": {"source": benchmark, "label": label},
        "portfolios": {key: {"name": PORTFOLIOS[key],
                             "target_weights": {t: float(v) for t, v in w.items()}}
                       for key, w in targets.items()},
    })
    store.save_models({"units": "returns in %, variance in %^2 per day; VaR is a decimal loss",
                       "model": "top-down GJR-GARCH(1,1)-X with India VIX^2/252 "
                                "(gjr_fit from nifty50_risk_layers.py), fitted once",
                       **models})
    store.write("tracker.csv", pd.DataFrame(rows, columns=TRACKER_COLS))
    store.write("weights.csv", pd.DataFrame(wrows).fillna(0.0))
    for name in ["returns.csv", "rebalances.csv", "prices_raw.csv"]:
        store.path(name).unlink(missing_ok=True)
    store.append_log({"run_time": now.isoformat(timespec="seconds"), "command": "init",
                      "status": "ok", "dates_added": 0, "revisions": "",
                      "warnings": "", "errors": ""})
    return models, targets
