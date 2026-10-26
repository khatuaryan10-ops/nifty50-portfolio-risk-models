"""
Volatility and 99% VaR of the min-variance and max-Sharpe portfolios, 2019 to now.

Uses the risk models from nifty50_risk_layers.py, all fitted on 2019-2023 only:
  Static     constant variance from the training-window sample covariance
  Bottom-up  factor model with GJR-GARCH-X market and stock volatilities
  Top-down   GJR-GARCH-X on the portfolio's own returns
VaR is filtered historical simulation, as in the risk-layers script: the 1% quantile of the
model's standardised training returns times that day's forecast σ. A breach is a day whose
loss exceeds the VaR set the evening before. Up to 2023 is in sample; 2024 on is the test.

Outputs: nifty50_portfolio_vol.png, nifty50_portfolio_var.png, nifty50_var_breaches.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter

from nifty50_risk_layers import (
    SPLIT, HIGH_VIX_PCT, TRADING_DAYS, MAX_WEIGHT, VAR_LEVEL, COLORS, load_data,
    fit_bottom_up, portfolio_layers, kupiec_p, style, min_variance_for_target, max_sharpe)

VOL_MODELS = ["Static", "Bottom-up", "Top-down"]
VAR_MODELS = ["Static", "Top-down"]
REALISED_WIN = 21               # centred window for realised volatility, trading days
CLUSTER_GAP = 10                # breaches closer than this many trading days form one episode
# Known market shocks, used only to name breach episodes that fall near them
EVENTS = {"2020-03-23": "COVID crash", "2022-02-24": "Russia-Ukraine invasion",
          "2024-06-04": "Election result", "2024-08-05": "Global sell-off (yen carry)",
          "2025-04-07": "US tariff shock"}
BREACH_COLOR = "#d03b3b"


def var_series(rp, s2, tr):
    """99% VaR (positive number, % return) for every day, from the training-window quantile."""
    z_train = rp[tr] / np.sqrt(s2[tr])
    return -np.quantile(z_train, VAR_LEVEL) * np.sqrt(s2)


def name_episode(start, end):
    for d, label in EVENTS.items():
        d = pd.Timestamp(d)
        if start - pd.Timedelta(days=10) <= d <= end + pd.Timedelta(days=10):
            return label
    return ""


def episodes(dates, hits):
    """Group breach days into episodes: runs of breaches less than CLUSTER_GAP days apart."""
    idx = np.flatnonzero(hits)
    if len(idx) == 0:
        return []
    groups, cur = [], [idx[0]]
    for i in idx[1:]:
        if i - cur[-1] < CLUSTER_GAP:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    return [(dates[g[0]], dates[g[-1]], len(g)) for g in groups]


def plot_vol(dates, data):
    fig, axes = plt.subplots(len(data), 1, figsize=(14, 4.2 * len(data)), sharex=True)
    split = pd.Timestamp(SPLIT)
    for ax, (pname, d) in zip(axes, data.items()):
        ann = lambda h: np.sqrt(h * TRADING_DAYS) / 100
        realised = pd.Series(d["rp"], dates).rolling(REALISED_WIN, center=True).std() \
            * np.sqrt(TRADING_DAYS) / 100
        ax.fill_between(dates, 0, realised, color="#c3c2b7", alpha=0.6, lw=0,
                        label=f"Realised ({REALISED_WIN}-day, centred)")
        for model in VOL_MODELS:
            ax.plot(dates, ann(d["s2"][model]), color=COLORS[model], lw=1.1 if model != "Static" else 1.6,
                    ls="--" if model == "Static" else "-", label=f"{model} forecast")
        ax.axvline(split, color="#898781", lw=0.8, ls=":")
        top = ax.get_ylim()[1]
        ax.text(split, top * 0.62, "  test (out of sample) →", color="#52514e", fontsize=9, va="top")
        ax.text(split, top * 0.62, "← training (in sample)  ", color="#52514e", fontsize=9,
                va="top", ha="right")
        ax.set_title(f"{pname} - annualised volatility, forecast vs realised", loc="left",
                     fontsize=13, color="#0b0b0b")
        ax.set_ylabel("Annualised volatility", color="#52514e")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_ylim(0, None)
        style(ax)
        ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=2)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig("nifty50_portfolio_vol.png", dpi=150)
    plt.show()


def plot_var(dates, data):
    fig = plt.figure(figsize=(15, 4.6 * len(data)))
    gs = fig.add_gridspec(len(data), 2, width_ratios=[3.2, 1])
    split = pd.Timestamp(SPLIT)
    for row, (pname, d) in enumerate(data.items()):
        ax, axh = fig.add_subplot(gs[row, 0]), fig.add_subplot(gs[row, 1])
        r = d["rp"] / 100
        var_td, var_st = d["var"]["Top-down"] / 100, d["var"]["Static"] / 100
        hits = r < -var_td

        ax.bar(dates, r, width=1.5, color="#9a9890", lw=0, label="Daily return")
        ax.plot(dates, -var_td, color=COLORS["Top-down"], lw=1, label="−VaR 99%, top-down")
        ax.plot(dates, -var_st, color=COLORS["Static"], lw=1.4, ls="--", label="−VaR 99%, static")
        ax.scatter(dates[hits], r[hits], s=26, color=BREACH_COLOR, edgecolors="white",
                   linewidths=0.8, zorder=3, label=f"Top-down breach ({hits.sum()} since 2019)")
        for start, end, n in d["episodes"]:
            label = name_episode(start, end)
            if label:
                span = (dates >= start) & (dates <= end)
                y = r[span].min()
                ax.annotate(label, (start, y), xytext=(8, -34), textcoords="offset points",
                            fontsize=8.5, color="#52514e",
                            arrowprops=dict(arrowstyle="-", color="#898781", lw=0.7))
        ax.axvline(split, color="#898781", lw=0.8, ls=":")
        ax.set_title(f"{pname} - daily return vs 99% VaR set the evening before", loc="left",
                     fontsize=13, color="#0b0b0b")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        style(ax)
        ax.legend(frameon=False, fontsize=9, loc="lower right", ncol=2)

        # Distribution of loss as a multiple of that day's VaR: > 1 is a breach
        te = d["te"]
        ratio = -r / var_td
        bins = np.linspace(-3, 3, 61)
        for mask, color, lab in [(te & ~hits, "#9a9890", "Test, within VaR"),
                                 (te & hits, BREACH_COLOR, "Test, breach")]:
            axh.hist(np.clip(ratio[mask], -3, 3), bins=bins, color=color, lw=0, label=lab)
        axh.axvline(1, color=COLORS["Top-down"], lw=1.2)
        axh.text(1.05, axh.get_ylim()[1] * 0.9, "VaR", color=COLORS["Top-down"], fontsize=9)
        n_te, h_te = te.sum(), (te & hits).sum()
        axh.set_title(f"Test window: loss ÷ VaR\n{h_te} breaches vs {n_te * VAR_LEVEL:.1f} expected",
                      loc="left", fontsize=11, color="#0b0b0b")
        axh.set_xlabel("Loss as a multiple of top-down VaR", color="#52514e")
        axh.set_ylabel("Days", color="#52514e")
        style(axh)
    fig.tight_layout()
    fig.savefig("nifty50_portfolio_var.png", dpi=150)
    plt.show()


def main():
    rets, nifty, vix, _ = load_data()
    cols, dates = rets.columns, rets.index
    tr = dates <= SPLIT
    te = ~tr
    R, m = rets.values * 100, nifty.values * 100
    x = vix.values ** 2 / TRADING_DAYS
    vix_prev = vix.shift(1).bfill().values
    high = vix_prev >= np.quantile(vix_prev[tr], HIGH_VIX_PCT)

    mu = rets[tr].mean().values * TRADING_DAYS
    cov = rets[tr].cov().values * TRADING_DAYS
    n = len(cols)
    w_mv = np.clip(min_variance_for_target(mu, cov, MAX_WEIGHT, None,
                                           np.full(n, 1 / n)).x, 0, MAX_WEIGHT)
    w_ms = np.clip(max_sharpe(mu, cov, MAX_WEIGHT, w_mv).x, 0, MAX_WEIGHT)

    print("Fitting models on 2019-2023...")
    bu = fit_bottom_up(R, m, x, high, tr)
    cov_static = np.cov(R[tr], rowvar=False)

    data, rows, breach_rows = {}, [], []
    for pname, w in {"Max Sharpe": w_ms, "Min variance": w_mv}.items():
        rp, s2, _, _ = portfolio_layers(w, R, x, tr, cov_static, bu)
        var = {mdl: var_series(rp, s2[mdl], tr) for mdl in VAR_MODELS}
        hits_td = rp < -var["Top-down"]
        data[pname] = dict(rp=rp, s2=s2, var=var, te=te,
                           episodes=episodes(dates, hits_td))

        for mdl in VAR_MODELS:
            hits = rp < -var[mdl]
            for label, mask in [("Train 2019-23", tr), ("Test 2024-now", te)]:
                k, nd = int((hits & mask).sum()), int(mask.sum())
                rows.append({"Portfolio": pname, "Model": mdl, "Window": label, "Days": nd,
                             "Breaches": k, "Expected": nd * VAR_LEVEL,
                             "Kupiec p": kupiec_p(k, nd, VAR_LEVEL)})
        for start, end, k in data[pname]["episodes"]:
            span = (dates >= start) & (dates <= end) & hits_td
            worst = np.argmin(np.where(span, rp, np.inf))
            breach_rows.append({"Portfolio": pname, "Start": start.date(), "End": end.date(),
                                "Breaches": k, "Window": "test" if start > pd.Timestamp(SPLIT)
                                else "train", "Worst day": dates[worst].date(),
                                "Worst return": rp[worst] / 100,
                                "VaR that day": var["Top-down"][worst] / 100,
                                "Event": name_episode(start, end)})

    summary = pd.DataFrame(rows).set_index(["Portfolio", "Model", "Window"])
    print("\n1. 99% VaR breaches (Kupiec p < 0.05 = breach rate is wrong)")
    print(summary.to_string(formatters={"Expected": "{:.1f}".format, "Kupiec p": "{:.2f}".format}))

    eps = pd.DataFrame(breach_rows)
    eps.to_csv("nifty50_var_breaches.csv", index=False)
    print("\n2. Top-down VaR breach episodes (breaches < "
          f"{CLUSTER_GAP} trading days apart are grouped)")
    print(eps.to_string(index=False, formatters={"Worst return": "{:.2%}".format,
                                                 "VaR that day": "{:.2%}".format}))

    print("\n3. Top-down breaches by year")
    by_year = pd.DataFrame({p: pd.Series(d["rp"] < -d["var"]["Top-down"], dates)
                            .groupby(dates.year).sum() for p, d in data.items()})
    by_year["Expected"] = pd.Series(1, dates).groupby(dates.year).sum() * VAR_LEVEL
    print(by_year.to_string(float_format="{:.1f}".format))

    plot_vol(dates, data)
    plot_var(dates, data)
    return summary, eps


if __name__ == "__main__":
    main()
