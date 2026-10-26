"""
Where does the forecast volatility come from around the 4 June 2024 election shock?

Every GJR-GARCH-X variance is a sum of today's inputs plus β times yesterday's variance:
    σ²ₜ = ω + (α + γ·1[ε<0]) εₜ₋₁² + δ'Xₜ₋₁ + β σ²ₜ₋₁
Unrolling the β term assigns all of σ²ₜ to the input that created it, so the split is
exact and additive. The top-down + bottom-up model (blend 1) takes the bottom-up forecast
as an input, and that forecast is unrolled the same way into its Nifty and stock-level
squared-return and VIX parts. The result shows how much VIX and squared returns contribute
through every channel, which is where any double counting would appear.

It also puts the forecasts on India VIX's terms: VIX is the Nifty's expected volatility
over the next 30 days, so it is compared with the Nifty model's 22-trading-day average
forecast rather than with a portfolio's 1-day forecast.

Models are fitted on 2019-2023 (nifty50_risk_layers.py), so the window is out of sample.

Requires: pip install numpy pandas matplotlib yfinance scipy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from nifty50_risk_layers import (
    SPLIT, HIGH_VIX_PCT, TRADING_DAYS, MAX_WEIGHT, load_data, fit_bottom_up,
    portfolio_layers, min_variance_for_target, max_sharpe)

WIN_START, WIN_END = "2024-05-06", "2024-07-05"     # two months around 4 June 2024
KEY_DATES = ["2024-05-31", "2024-06-04", "2024-06-05", "2024-06-07", "2024-06-14",
             "2024-06-28"]
HORIZON = 22                                        # trading days, about VIX's 30 calendar days
CHANNELS = ["Constant", "Portfolio r² (direct)", "Nifty r² (via BU)", "Stock r² (via BU)",
            "VIX (direct)", "VIX via Nifty model", "VIX via stock models"]
GROUPS = {"Constant": ["Constant"],
          "Squared returns": ["Portfolio r² (direct)", "Nifty r² (via BU)", "Stock r² (via BU)"],
          "India VIX": ["VIX (direct)", "VIX via Nifty model", "VIX via stock models"]}
GROUP_COLORS = {"Constant": "#c3c2b7", "Squared returns": "#eb6834", "India VIX": "#2a78d6"}


def decompose(p, r, exog, s0):
    """Exact split of a GJR-GARCH-X variance by root source. exog is a list of (T, k)
    component matrices, one per exogenous input; row t-1 of each sums to that input's value
    used for day t. Columns: constant, shock, every exogenous component, starting value."""
    omega, alpha, gamma, beta = p[:4]
    width = 2 + sum(e.shape[1] for e in exog) + 1
    C = np.zeros((len(r), width))
    C[0, -1] = s0
    for t in range(1, len(r)):
        e = r[t - 1]
        new = np.concatenate([[omega, (alpha + gamma * (e < 0)) * e * e],
                              *[d * ex[t - 1] for d, ex in zip(p[4:], exog)], [0.0]])
        C[t] = new + beta * C[t - 1]
    return C


def avg_forecast(p, s2_next, extra, h=HORIZON):
    """Mean of the next h daily variance forecasts, starting from σ²ₜ, with the exogenous
    inputs held at their current values (extra = δ'x)."""
    pers = p[1] + p[2] / 2 + p[3]
    f, total = s2_next.copy(), s2_next.copy()
    for _ in range(h - 1):
        f = p[0] + extra + pers * f
        total += f
    return total / h


def channels_bottom_up(bu, w, x):
    """Bottom-up portfolio variance split into market and stock-level sources:
    (B'w)² σ²ₘ + Σ wᵢ² hᵢ, each GARCH unrolled. Columns follow decompose():
    market [const, shock, vix, start], then stocks [const, shock, vix, start]."""
    X = [x[:, None]]
    Cm = decompose(bu["pm"], bu["s2m"] * 0 + bu["m"], X, bu["s0m"])
    Ci = sum(w[i] ** 2 * decompose(bu["idio"][i], bu["E"][:, i], X, bu["idio_s0"][i])
             for i in range(len(w)) if w[i] > 1e-6)
    bw2 = (bu["B"] @ w) ** 2
    return np.column_stack([bw2[:, None] * Cm, Ci])


def label_columns(C_tb):
    """Map the TD+BU decomposition columns to named channels."""
    names = (["Constant", "Portfolio r² (direct)", "VIX (direct)"]
             + ["Constant", "Nifty r² (via BU)", "VIX via Nifty model", "Constant"]
             + ["Constant", "Stock r² (via BU)", "VIX via stock models", "Constant"]
             + ["Constant"])
    out = pd.DataFrame(0.0, index=range(len(C_tb)), columns=CHANNELS)
    for j, name in enumerate(names):
        out[name] += C_tb[:, j]
    return out


def main():
    rets, nifty, vix, _ = load_data()
    cols, dates = rets.columns, rets.index
    tr = dates <= SPLIT
    R, m = rets.values * 100, nifty.values * 100
    x = vix.values ** 2 / TRADING_DAYS
    vix_prev = vix.shift(1).bfill().values
    high = vix_prev >= np.quantile(vix_prev[tr], HIGH_VIX_PCT)

    mu = rets[tr].mean().values * TRADING_DAYS
    cov = rets[tr].cov().values * TRADING_DAYS
    n = len(cols)
    w_mv = np.clip(min_variance_for_target(mu, cov, MAX_WEIGHT, None,
                                           np.full(n, 1 / n)).x, 0, MAX_WEIGHT)
    w = np.clip(max_sharpe(mu, cov, MAX_WEIGHT, w_mv).x, 0, MAX_WEIGHT)

    print("Fitting models on 2019-2023...")
    bu = fit_bottom_up(R, m, x, high, tr)
    bu["m"] = m
    rp, s2, _, fit = portfolio_layers(w, R, x, tr, np.cov(R[tr], rowvar=False), bu)
    p_td, p_tb = fit["td"], fit["tb"]

    # ---- Exact decompositions ----
    C_bu = channels_bottom_up(bu, w, x)
    C_bu_next = np.vstack([C_bu[1:], C_bu[-1:]])        # row t-1 holds BU for day t
    C_tb = decompose(p_tb, rp, [x[:, None], C_bu_next], fit["s0_tb"])
    C_td = decompose(p_td, rp, [x[:, None]], fit["s0_td"])
    for name, C, target in [("Bottom-up", C_bu, s2["Bottom-up"]),
                            ("Top-down", C_td, s2["Top-down"]),
                            ("TD + BU", C_tb, s2["TD + BU input"])]:
        err = np.max(np.abs(C.sum(1) - target) / target)
        assert err < 1e-8, f"{name} decomposition does not add up ({err:.1e})"

    tb = label_columns(C_tb).set_index(dates)
    td = pd.DataFrame({"Constant": C_td[:, 0] + C_td[:, 3], "Portfolio r² (direct)": C_td[:, 1],
                       "VIX (direct)": C_td[:, 2]}, index=dates).reindex(columns=CHANNELS,
                                                                         fill_value=0.0)
    bu_df = pd.DataFrame({"Constant": C_bu[:, [0, 3, 4, 7]].sum(1),
                          "Nifty r² (via BU)": C_bu[:, 1], "VIX via Nifty model": C_bu[:, 2],
                          "Stock r² (via BU)": C_bu[:, 5], "VIX via stock models": C_bu[:, 6]},
                         index=dates).reindex(columns=CHANNELS, fill_value=0.0)
    win = (dates >= WIN_START) & (dates <= WIN_END)

    print(f"\nMax Sharpe, {WIN_START} to {WIN_END} ({win.sum()} days, out of sample)")
    print(f"Blend 1 coefficients: shock α+γ {p_tb[1] + p_tb[2]:.3f}, memory β {p_tb[3]:.3f}, "
          f"VIX δ {p_tb[4]:.3f}, bottom-up φ {p_tb[5]:.3f} "
          f"(top-down alone: α+γ {p_td[1] + p_td[2]:.3f}, β {p_td[3]:.3f}, δ {p_td[4]:.3f})")

    print("\n1. Share of forecast variance by source, averaged over the window")
    share = pd.DataFrame({name: df[win].sum() / df[win].sum().sum()
                          for name, df in [("Bottom-up", bu_df), ("Top-down", td),
                                           ("TD + BU", tb)]})
    for g, members in GROUPS.items():
        share.loc[f"= {g}"] = share.loc[members].sum()
    print(share.to_string(float_format="{:.1%}".format))

    print("\n2. TD + BU forecast on key days: σ and the share from each source")
    key = pd.DatetimeIndex(KEY_DATES)
    tot = tb.loc[key].sum(1)
    by_day = pd.DataFrame({"σ (1-day, annualised)": np.sqrt(tot * TRADING_DAYS) / 100,
                           **{g: tb.loc[key, mem].sum(1) / tot for g, mem in GROUPS.items()},
                           "Return that day": rp[dates.get_indexer(key)] / 100},
                          index=key.date)
    print(by_day.to_string(float_format="{:.1%}".format))

    # ---- How correlated are the inputs? ----
    r2_prev = np.r_[np.nan, rp[:-1] ** 2]
    m2_prev = np.r_[np.nan, m[:-1] ** 2]
    inputs = pd.DataFrame({"Portfolio r²(t-1)": r2_prev, "Nifty r²(t-1)": m2_prev,
                           "VIX²(t-1)": np.r_[np.nan, x[:-1]], "BU forecast(t)": s2["Bottom-up"]},
                          index=dates)
    print("\n3. Correlation between the model inputs (training window)")
    print(inputs[tr].corr().to_string(float_format="{:.2f}".format))

    # ---- Comparison with India VIX, like for like ----
    ann = lambda v: np.sqrt(v * TRADING_DAYS) / 100
    x_prev = np.r_[x[0], x[:-1]]
    pm = bu["pm"]
    nifty_22 = avg_forecast(pm, bu["s2m"], pm[4] * x_prev)
    port_22 = avg_forecast(p_tb, s2["TD + BU input"],
                           p_tb[4] * x_prev + p_tb[5] * s2["Bottom-up"])
    fwd = lambda r: pd.Series(r ** 2, dates)[::-1].rolling(HORIZON).mean()[::-1]
    comp = pd.DataFrame({
        "India VIX (prev close)": vix.shift(1).values / 100,
        "Nifty model 1-day": ann(bu["s2m"]), "Nifty model 22-day": ann(nifty_22),
        "Nifty realised next 22d": np.sqrt(fwd(m) * TRADING_DAYS).values / 100,
        "Portfolio 1-day": ann(s2["TD + BU input"]), "Portfolio 22-day": ann(port_22),
        "Portfolio realised next 22d": np.sqrt(fwd(rp) * TRADING_DAYS).values / 100},
        index=dates)
    print("\n4. India VIX vs model forecasts (all known at the previous close; annualised)")
    print(comp.loc[key].set_axis(key.date).to_string(float_format="{:.1%}".format))
    avg = comp[win].mean()
    print("Window averages: " + ", ".join(f"{k} {v:.1%}" for k, v in avg.items()))
    gap = comp["Nifty model 22-day"] - comp["India VIX (prev close)"]
    print(f"Nifty model 22-day minus VIX: window mean {gap[win].mean():+.1%}, "
          f"training mean {gap[tr].mean():+.1%}, test mean {gap[~tr].mean():+.1%}")

    plot(dates[win], tb[win], comp[win])
    return share, comp


def plot(d, tb, comp):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))
    stacks = [tb[mem].sum(1).values * TRADING_DAYS / 1e4 for mem in GROUPS.values()]
    ax1.stackplot(d, *stacks, labels=list(GROUPS), colors=list(GROUP_COLORS.values()),
                  edgecolor="white", linewidth=0.6)
    ax1.set_title("Max Sharpe 1-day forecast variance by source (TD + BU)", loc="left",
                  fontsize=13, color="#0b0b0b")
    ax1.set_ylabel("Forecast variance, annualised (σ²)", color="#52514e")

    lines = [("India VIX (prev close)", "#0b0b0b", "-"),
             ("Nifty model 22-day", "#2a78d6", "-"),
             ("Nifty realised next 22d", "#2a78d6", ":"),
             ("Portfolio 1-day", "#eb6834", "-"),
             ("Portfolio realised next 22d", "#eb6834", ":")]
    for col, color, ls in lines:
        ax2.plot(d, comp[col], color=color, ls=ls, lw=2, label=col)
    ax2.set_title("India VIX vs model forecasts", loc="left", fontsize=13, color="#0b0b0b")
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax2.set_ylabel("Annualised volatility", color="#52514e")

    for ax in (ax1, ax2):
        ax.axvline(pd.Timestamp("2024-06-04"), color="#898781", lw=0.8, ls="--")
        ax.grid(color="#e1e0d9", lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#c3c2b7")
        ax.tick_params(colors="#898781")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig("nifty50_vol_attribution.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
