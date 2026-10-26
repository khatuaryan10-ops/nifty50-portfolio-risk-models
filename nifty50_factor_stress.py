"""
Multi-factor macro stress test of the optimised Nifty 50 portfolios.

Factors (weekly returns): Nifty 50 index, a 10Y G-Sec ETF (bond value), and Brent,
gold and copper converted to INR. Each stock is regressed on all of them,
    r_i = a_i + B_i f + e_i
which splits the sample covariance into a factor part and a residual part,
    Σ = B Σ_f B' + Σ_e          (Σ_e is the full residual covariance, not just a diagonal)
A scenario shocks some factors; the others move by their expected value given those
shocks (E[f_u | f_s] = Σ_us Σ_ss⁻¹ Δf_s), so e.g. a Nifty crash also pulls copper down.
    μ_s = μ + B Δf                       Δf = one-off factor moves
    Σ_s = B K Σ_f K B' + Σ_e             K = diag(factor vol multipliers)
The min-variance and max-Sharpe portfolios are re-evaluated under μ_s / Σ_s, and
max-Sharpe is re-optimised to show how it would move.

Repo rate changes only a few times a year, so it isn't a regression factor. A repo
shock is passed through to the 10Y yield, then to bond price via duration:
    Δy = PASS_THROUGH * Δrepo,    Δf_bond = -DURATION * Δy,    rf rises by Δy
In every scenario rf moves by the yield change implied by the bond factor's move.

Weekly returns are used because the commodity futures settle in US hours, after the
NSE close, and the two markets have different holidays.

Requires: pip install numpy pandas matplotlib yfinance scipy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import yfinance as yf

from nifty50_efficient_frontier import (
    TICKERS, PERIOD, RISK_FREE_RATE, MAX_WEIGHT,
    load_prices, min_variance_for_target, max_sharpe, optimised_frontier)

# ---------------- Parameters ----------------
FACTORS = {
    "Nifty": "^NSEI",
    "Bond": "SETF10GILT.NS",    # SBI 10Y G-Sec ETF; its price is the bond value factor
    "Brent": "BZ=F",
    "Gold": "GC=F",
    "Copper": "HG=F",
}
USD_PRICED = ["Brent", "Gold", "Copper"]    # converted to INR with USDINR
FX_TICKER = "USDINR=X"
PERIODS_PER_YEAR = 52           # weekly returns
DURATION = 7.0                  # approx. modified duration of the 10Y benchmark G-Sec
PASS_THROUGH = 0.6              # change in 10Y yield per unit change in repo rate

# repo: repo-rate change (sets the bond shock); shock: factor moves; vol: factor vol
# multipliers. Factors not given a shock move by their conditional expectation.
SCENARIOS = {
    "Baseline": {},
    "Repo +50bp": {"repo": 0.005, "vol": {"Bond": 1.5}},
    "Nifty -20% crash": {"shock": {"Nifty": -0.20}, "vol": {"Nifty": 2.0}},
    "Brent +40% spike": {"shock": {"Brent": 0.40}, "vol": {"Brent": 1.5}},
    "Stagflation": {"repo": 0.02, "shock": {"Brent": 0.30, "Nifty": -0.10},
                    "vol": {"Bond": 2.0, "Brent": 1.5, "Nifty": 1.5}},
}
# Baseline in neutral ink, scenarios in categorical slots 1-4 + distinct dashes
STYLES = {
    "Baseline": ("#52514e", "-"),
    "Repo +50bp": ("#2a78d6", "--"),
    "Nifty -20% crash": ("#eb6834", "-"),
    "Brent +40% spike": ("#1baf7a", "-."),
    "Stagflation": ("#eda100", ":"),
}


def load_weekly_returns():
    """Weekly returns of the stocks and the factors, aligned on Friday closes."""
    stocks = load_prices(TICKERS, PERIOD)
    raw = yf.download(list(FACTORS.values()) + [FX_TICKER], period=PERIOD,
                      auto_adjust=True, progress=False)["Close"]
    factors = raw[list(FACTORS.values())].set_axis(list(FACTORS), axis=1)
    factors[USD_PRICED] = factors[USD_PRICED].mul(raw[FX_TICKER], axis=0)
    weekly = (pd.concat([stocks, factors], axis=1).sort_index().ffill()
              .resample("W-FRI").last().pct_change().dropna())
    return weekly[stocks.columns], weekly[list(FACTORS)]


def fit_factor_model(returns, factors):
    """OLS of each stock's weekly return on all factors. Returns the betas (stocks x
    factors) with each stock's R², and the annualised factor and residual covariances."""
    X, F = returns.values, factors.values
    F1 = np.column_stack([np.ones(len(F)), F])
    coef = np.linalg.lstsq(F1, X, rcond=None)[0]
    B = coef[1:].T
    cov_f = np.cov(F, rowvar=False) * PERIODS_PER_YEAR
    resid_cov = np.cov(X - F1 @ coef, rowvar=False) * PERIODS_PER_YEAR
    total_var = np.diag(B @ cov_f @ B.T + resid_cov)
    model = pd.DataFrame(B, index=returns.columns, columns=factors.columns)
    model["R2"] = 1 - np.diag(resid_cov) / total_var
    return model, cov_f, resid_cov


def scenario_moves(spec, cov_f):
    """Factor moves for a scenario: the given shocks, with every other factor set to its
    expected move conditional on them."""
    names = list(FACTORS)
    shocks = dict(spec.get("shock", {}))
    if "repo" in spec:
        shocks["Bond"] = -DURATION * PASS_THROUGH * spec["repo"]
    s = [names.index(k) for k in shocks]
    u = [i for i in range(len(names)) if i not in s]
    df = np.zeros(len(names))
    if s:
        df[s] = list(shocks.values())
        df[u] = cov_f[np.ix_(u, s)] @ np.linalg.solve(cov_f[np.ix_(s, s)], df[s])
    return df


def stress_inputs(mu, B, cov_f, resid_cov, spec):
    """Stressed μ, Σ and rf for one scenario, plus the factor moves and factor covariance."""
    df = scenario_moves(spec, cov_f)
    k = np.array([spec.get("vol", {}).get(name, 1.0) for name in FACTORS])
    cov_fs = cov_f * np.outer(k, k)
    dy = -df[list(FACTORS).index("Bond")] / DURATION     # yield change implied by bond move
    return mu + B @ df, B @ cov_fs @ B.T + resid_cov, RISK_FREE_RATE + dy, df, cov_fs


def weighted_avg_corr(w, cov):
    """Average pairwise correlation, each pair weighted by w_i * w_j (diagonal excluded)."""
    sd = np.sqrt(np.diag(cov))
    ww = np.outer(w, w)
    np.fill_diagonal(ww, 0)
    return (ww * cov / np.outer(sd, sd)).sum() / ww.sum()


def evaluate(w, mu, cov, B, cov_f, rf, df):
    vol = np.sqrt(w @ cov @ w)
    b_p = w @ B
    return {"Nifty beta": b_p[0], "Shock P&L": b_p @ df, "Return": w @ mu,
            "Volatility": vol, "Sharpe": (w @ mu - rf) / vol,
            "Avg corr": weighted_avg_corr(w, cov), "Factor % var": b_p @ cov_f @ b_p / vol ** 2}


def plot_stressed_frontiers(frontiers, best):
    """Efficient frontier under each scenario, with its re-optimised max-Sharpe portfolio."""
    fig, ax = plt.subplots(figsize=(11, 7))
    for name, f in frontiers.items():
        color, ls = STYLES[name]
        ax.plot(f.Volatility, f.Return, color=color, ls=ls, lw=2, label=name)
        ax.annotate(name, (f.Volatility.iloc[-1], f.Return.iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color="#52514e")
        vol, ret = best[name]
        ax.scatter(vol, ret, marker="*", s=220, color=color, edgecolors="white",
                   linewidths=1.5, zorder=3)
    ax.scatter([], [], marker="*", s=220, color="#898781", edgecolors="white",
               label="Max Sharpe (re-optimised, scenario rf)")

    ax.set_title(f"Nifty 50 - efficient frontier under macro factor stress, "
                 f"max {MAX_WEIGHT:.0%} per stock", loc="left", fontsize=14, color="#0b0b0b")
    ax.set_xlabel("Annualised volatility (σ)", color="#52514e")
    ax.set_ylabel("Annualised expected return", color="#52514e")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(color="#e1e0d9", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors="#898781")
    ax.margins(x=0.15)                       # room for the end-of-line labels
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig("nifty50_stressed_frontiers.png", dpi=150)
    plt.show()


def main():
    returns, factors = load_weekly_returns()
    print(f"{returns.shape[1]} stocks vs {len(FACTORS)} factors, {len(returns)} weeks "
          f"({returns.index[0].date()} to {returns.index[-1].date()})")

    mu = returns.mean().values * PERIODS_PER_YEAR
    model, cov_f, resid_cov = fit_factor_model(returns, factors)
    B = model[list(FACTORS)].values
    cov = B @ cov_f @ B.T + resid_cov
    sd_f = np.sqrt(np.diag(cov_f))
    print("\nFactor volatility and correlation (annualised):")
    print(pd.DataFrame(cov_f / np.outer(sd_f, sd_f), index=list(FACTORS),
                       columns=list(FACTORS)).assign(Volatility=sd_f).round(2).to_string())
    print(f"\nStock R²: median {model.R2.median():.1%}, "
          f"range {model.R2.min():.1%} to {model.R2.max():.1%}")
    for name in FACTORS:
        top = model[name].abs().nlargest(3).index
        print(f"  Largest {name} betas: "
              + ", ".join(f"{t} {model.at[t, name]:+.2f}" for t in top))

    # Baseline optimised portfolios (same constraints as the frontier script)
    n = len(mu)
    w_minvar = np.clip(min_variance_for_target(mu, cov, MAX_WEIGHT, None,
                                               np.full(n, 1 / n)).x, 0, MAX_WEIGHT)
    w_sharpe = np.clip(max_sharpe(mu, cov, MAX_WEIGHT, w_minvar).x, 0, MAX_WEIGHT)
    print("\nBaseline factor exposures:")
    print(pd.DataFrame([w_minvar @ B, w_sharpe @ B], index=["Min variance", "Max Sharpe"],
                       columns=list(FACTORS)).round(3).to_string())

    rows, moves, universe, frontiers, best = [], {}, {}, {}, {}
    for name, spec in SCENARIOS.items():
        mu_s, cov_s, rf_s, df, cov_fs = stress_inputs(mu, B, cov_f, resid_cov, spec)
        moves[name] = df
        # max_sharpe uses the fixed RISK_FREE_RATE; as weights sum to 1, shifting μ by
        # (rf_s - RISK_FREE_RATE) gives the same optimum as using rf_s
        w_reopt = np.clip(max_sharpe(mu_s - (rf_s - RISK_FREE_RATE), cov_s, MAX_WEIGHT,
                                     w_sharpe).x, 0, MAX_WEIGHT)
        for label, w in [("Min variance", w_minvar), ("Max Sharpe", w_sharpe),
                         ("Max Sharpe re-opt", w_reopt)]:
            rows.append({"Scenario": name, "Portfolio": label,
                         **evaluate(w, mu_s, cov_s, B, cov_fs, rf_s, df)})
        universe[name] = weighted_avg_corr(np.full(n, 1 / n), cov_s)
        frontiers[name], _ = optimised_frontier(mu_s, cov_s, MAX_WEIGHT)
        best[name] = (np.sqrt(w_reopt @ cov_s @ w_reopt), w_reopt @ mu_s)

    pct = lambda x: f"{x:+.2%}"
    print("\nFactor moves per scenario (shocked + conditional):")
    print(pd.DataFrame(moves, index=list(FACTORS)).T.to_string(float_format=pct))

    table = pd.DataFrame(rows).set_index(["Scenario", "Portfolio"])
    table.to_csv("nifty50_factor_stress.csv")
    pct = lambda x: f"{x:.2%}"
    print("\n" + table.to_string(formatters={
        "Nifty beta": "{:.3f}".format, "Shock P&L": pct, "Return": pct, "Volatility": pct,
        "Sharpe": "{:.2f}".format, "Avg corr": "{:.3f}".format, "Factor % var": pct}))
    print("\nAverage pairwise correlation, all stocks: "
          + ", ".join(f"{k} {v:.3f}" for k, v in universe.items()))
    plot_stressed_frontiers(frontiers, best)
    return model, table


if __name__ == "__main__":
    main()
