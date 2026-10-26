"""
Nifty 50 portfolio analysis: random portfolios, correlation matrix and an
optimised efficient frontier with a per-stock weight cap.

1. Downloads 5 years of Nifty 50 prices from Yahoo Finance (yfinance).
2. Simulates N_PORTFOLIOS random long-only portfolios, each capped at MAX_WEIGHT
   per stock (return, variance, volatility, Sharpe for each).
3. Plots the correlation matrix of the stocks' daily returns.
4. Builds the efficient frontier with SLSQP: minimum variance at each of
   N_TARGETS target returns, warm-starting each solve from the previous one,
   plus the exact minimum-variance and maximum-Sharpe portfolios.

Requires: pip install numpy pandas matplotlib yfinance scipy
"""

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import yfinance as yf
from scipy.optimize import minimize

# ---------------- Parameters ----------------
N_PORTFOLIOS = 100_000
N_TARGETS = 60              # target returns used to draw the optimised frontier
TRADING_DAYS = 252
RISK_FREE_RATE = 0.065      # approx. Indian 10Y G-Sec yield
PERIOD = "5y"
DIRICHLET_ALPHA = 0.3       # <1 concentrates weight in a few stocks; 1.0 = uniform on the simplex
SEED = 42
MAX_WEIGHT = 0.10           # no single stock above 10% of the portfolio

# Nifty 50 constituents (Yahoo Finance symbols). The index is rebalanced
# periodically - edit this list to match the current composition.
NIFTY50 = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BEL", "BHARTIARTL",
    "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT", "ETERNAL",
    "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HINDALCO",
    "HINDUNILVR", "ICICIBANK", "INDIGO", "INFY", "ITC",
    "JIOFIN", "JSWSTEEL", "KOTAKBANK", "LT", "M&M",
    "MARUTI", "MAXHEALTH", "NESTLEIND", "NTPC", "ONGC",
    "POWERGRID", "RELIANCE", "SBILIFE", "SBIN", "SHRIRAMFIN",
    "SUNPHARMA", "TATACONSUM", "TMPV", "TATASTEEL", "TCS",
    "TECHM", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO",
]
TICKERS = [t + ".NS" for t in NIFTY50]


def load_prices(tickers, period):
    data = yf.download(tickers, period=period, auto_adjust=True, progress=False)["Close"]
    # Parallel downloads sometimes fail on yfinance's cache ("database is locked");
    # retry the empty ones one at a time.
    failed = data.columns[data.isna().all()].tolist()
    if failed:
        retry = yf.download(failed, period=period, auto_adjust=True, progress=False,
                            threads=False)["Close"]
        data[retry.columns] = retry
    # Drop tickers that failed to download or have too little history
    # (recent listings / renamed symbols), then drop remaining gaps.
    data = data.dropna(axis=1, thresh=int(0.8 * len(data)))
    data.columns = [c.replace(".NS", "") for c in data.columns]
    return data.dropna()


def cap_weights(weights, cap):
    """Clip each row at `cap` and hand the excess to the uncapped stocks in
    proportion to their weights, repeating until no weight exceeds the cap."""
    w = weights.copy()
    for _ in range(100):
        excess = np.clip(w - cap, 0, None).sum(axis=1, keepdims=True)
        if excess.max() < 1e-12:
            break
        w = np.minimum(w, cap)
        share = np.where(w < cap, w, 0.0) + 1e-12 * (w < cap)
        w += excess * share / share.sum(axis=1, keepdims=True)
    return w


def portfolio_stats(weights, mean_returns, cov):
    rets = weights @ mean_returns
    variances = np.einsum("ij,jk,ik->i", weights, cov, weights)     # w' Σ w per portfolio
    vols = np.sqrt(variances)
    return pd.DataFrame({"Return": rets, "Variance": variances, "Volatility": vols,
                         "Sharpe": (rets - RISK_FREE_RATE) / vols})


def simulate_portfolios(mean_returns, cov, n, alpha, rng):
    """n random long-only portfolios; weights sum to 1, none above MAX_WEIGHT."""
    weights = rng.dirichlet(np.full(len(mean_returns), alpha), size=n)
    weights = cap_weights(weights, MAX_WEIGHT)
    return portfolio_stats(weights, mean_returns, cov), weights


def min_variance_for_target(mean_returns, cov, cap, target, x0):
    """SLSQP: minimise w'Σw subject to sum(w) = 1, μ'w = target (if given), 0 <= w <= cap."""
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones_like(w)}]
    if target is not None:
        cons.append({"type": "eq", "fun": lambda w: w @ mean_returns - target,
                     "jac": lambda w: mean_returns})
    return minimize(lambda w: w @ cov @ w, x0, jac=lambda w: 2 * cov @ w, method="SLSQP",
                    bounds=[(0, cap)] * len(x0), constraints=cons,
                    options={"maxiter": 500, "ftol": 1e-12})


def max_sharpe(mean_returns, cov, cap, x0):
    """SLSQP: maximise (μ'w - rf) / sqrt(w'Σw) subject to sum(w) = 1, 0 <= w <= cap."""
    def neg_sharpe(w):
        return -(w @ mean_returns - RISK_FREE_RATE) / np.sqrt(w @ cov @ w)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones_like(w)}]
    return minimize(neg_sharpe, x0, method="SLSQP", bounds=[(0, cap)] * len(x0),
                    constraints=cons, options={"maxiter": 500, "ftol": 1e-12})


def optimised_frontier(mean_returns, cov, cap, n_targets=N_TARGETS):
    """Efficient frontier: minimum variance at each of n_targets evenly spaced returns, from
    the minimum-variance portfolio's return up to the highest return the cap allows.
    Each solve is warm-started from the previous target's weights."""
    n = len(mean_returns)
    w0 = np.full(n, 1 / n)
    res = min_variance_for_target(mean_returns, cov, cap, None, w0)
    cold_steps = res.nit
    # highest achievable return: fill the best stocks up to the cap
    best = np.sort(mean_returns)[::-1]
    k = int(np.floor(1 / cap))
    max_ret = best[:k].sum() * cap + best[k] * (1 - k * cap) if k < n else best.sum() / n
    targets = np.linspace(res.x @ mean_returns, max_ret, n_targets)

    weights, steps, x = [], [], res.x
    for t in targets:
        r = min_variance_for_target(mean_returns, cov, cap, t, x)    # warm start
        if not r.success:
            continue
        x = np.clip(r.x, 0, cap)
        weights.append(x)
        steps.append(r.nit)
    weights = np.array(weights)
    print(f"Frontier: {len(weights)}/{n_targets} targets solved; solver steps - cold start "
          f"{cold_steps}, warm starts avg {np.mean(steps):.1f}")
    return portfolio_stats(weights, mean_returns, cov), weights


def plot_correlation(corr):
    n = len(corr)
    fig, ax = plt.subplots(figsize=(14, 12))
    # Diverging scale fixed at -1..1 so 0 (no relationship) is the neutral midpoint
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n), corr.columns, rotation=90, fontsize=7, color="#444")
    ax.set_yticks(range(n), corr.index, fontsize=7, color="#444")
    for s in ax.spines.values():
        s.set_visible(False)
    # thin surface-colored gaps between cells
    ax.set_xticks(np.arange(-0.5, n), minor=True)
    ax.set_yticks(np.arange(-0.5, n), minor=True)
    ax.grid(which="minor", color="white", lw=0.6)
    ax.tick_params(which="both", length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Correlation of daily returns", color="#444")
    cbar.outline.set_visible(False)

    ax.set_title("Nifty 50 - correlation matrix of daily returns", loc="left",
                 fontsize=14, color="#111", pad=12)
    fig.tight_layout()
    fig.savefig("nifty50_correlation_matrix.png", dpi=150)
    plt.show()


def plot_efficient_frontier(results, frontier, optimal):
    fig, ax = plt.subplots(figsize=(11, 7))
    # Single-hue sequential scale: light = lower return, dark = higher return
    sc = ax.scatter(results.Volatility, results.Return, c=results.Return, cmap="Blues",
                    vmin=results.Return.min() - 0.03, s=3, alpha=0.6, edgecolors="none",
                    rasterized=True)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02, format=PercentFormatter(1.0))
    cbar.set_label("Annualised expected return", color="#444")
    cbar.outline.set_visible(False)

    ax.plot(frontier.Volatility, frontier.Return, color="#1f2937", lw=2,
            label=f"Optimised efficient frontier (SLSQP, {len(frontier)} target returns)")
    ax.scatter(optimal.Volatility["Min variance"], optimal.Return["Min variance"], marker="D", s=90,
               color="#047857", edgecolors="white", linewidths=1.5, zorder=3, label="Min variance")
    ax.scatter(optimal.Volatility["Max Sharpe"], optimal.Return["Max Sharpe"], marker="*", s=260,
               color="#d97706", edgecolors="white", linewidths=1.5, zorder=3,
               label=f"Max Sharpe (rf {RISK_FREE_RATE:.1%})")

    ax.set_title(f"Nifty 50 - efficient frontier vs {len(results):,} random portfolios, "
                 f"max {MAX_WEIGHT:.0%} per stock", loc="left", fontsize=14, color="#111")
    ax.set_xlabel("Annualised volatility (σ)", color="#444")
    ax.set_ylabel("Annualised expected return", color="#444")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(color="#e5e7eb", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#9ca3af")
    ax.tick_params(colors="#555")
    ax.legend(frameon=False, loc="lower right")

    fig.tight_layout()
    fig.savefig("nifty50_efficient_frontier.png", dpi=150)
    
    plt.show()


def main():
    rng = np.random.default_rng(SEED)

    prices = load_prices(TICKERS, PERIOD)
    missing = sorted(set(NIFTY50) - set(prices.columns))
    print(f"Using {prices.shape[1]} stocks, {len(prices)} trading days "
          f"({prices.index[0].date()} to {prices.index[-1].date()})")
    if missing:
        print("Skipped (no/insufficient data):", ", ".join(missing))

    daily_returns = prices.pct_change().dropna()
    mean_returns = daily_returns.mean().values * TRADING_DAYS
    cov = daily_returns.cov().values * TRADING_DAYS

    # ---- Random portfolio iterations ----
    t0 = time.perf_counter()
    results, weights = simulate_portfolios(mean_returns, cov, N_PORTFOLIOS, DIRICHLET_ALPHA, rng)
    t_sample = time.perf_counter() - t0
    portfolios = results
    # stats as CSV; the 100k x n weight matrix is large, so it goes in a compressed .npz
    results.to_csv("nifty50_portfolios.csv", index_label="Portfolio")
    np.savez_compressed("nifty50_portfolio_weights.npz", weights=weights,
                        stocks=np.array(prices.columns))
    print(f"\nSimulated {N_PORTFOLIOS:,} portfolios in {t_sample:.2f}s "
          f"(stats: nifty50_portfolios.csv, weights: nifty50_portfolio_weights.npz):")
    print(results.describe().loc[["mean", "min", "max"]].round(4))

    # ---- Correlation matrix ----
    corr = daily_returns.corr()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
    print(f"\nAverage pairwise correlation: {upper.mean():.2f}")
    print("Most correlated pairs:")
    print(upper.nlargest(5).round(2).to_string())
    print("Least correlated pairs:")
    print(upper.nsmallest(5).round(2).to_string())

    plot_correlation(corr)

    # ---- Optimised efficient frontier (0 <= w <= MAX_WEIGHT) ----
    print()
    t0 = time.perf_counter()
    frontier, frontier_w = optimised_frontier(mean_returns, cov, MAX_WEIGHT)
    t_frontier = time.perf_counter() - t0
    print(f"Optimised frontier in {t_frontier:.2f}s vs random sampling {t_sample:.2f}s")

    w_minvar = frontier_w[frontier.Variance.idxmin()]
    res = max_sharpe(mean_returns, cov, MAX_WEIGHT, frontier_w[frontier.Sharpe.idxmax()])
    w_sharpe = np.clip(res.x, 0, MAX_WEIGHT)
    opt_w = np.array([w_minvar, w_sharpe])
    labels = ["Min variance", "Max Sharpe"]
    optimal = pd.concat([portfolio_stats(opt_w, mean_returns, cov).set_axis(labels),
                         pd.DataFrame(opt_w, columns=prices.columns, index=labels)], axis=1)
    optimal.to_csv("nifty50_optimal_portfolios.csv")
    pd.concat([frontier, pd.DataFrame(frontier_w, columns=prices.columns)], axis=1) \
        .to_csv("nifty50_frontier.csv", index_label="Point")

    plot_efficient_frontier(results, frontier, optimal)
    print(f"Largest weight - random portfolios: {weights.max():.2%}, "
          f"optimised: {max(frontier_w.max(), w_sharpe.max()):.2%}")
    for label in optimal.index:
        r = optimal.loc[label]
        held = r[prices.columns][r[prices.columns] > 0.001].sort_values(ascending=False)
        print(f"\n{label}: return {r.Return:.2%}, variance {r.Variance:.4f}, "
              f"volatility {r.Volatility:.2%}, Sharpe {r.Sharpe:.2f} ({len(held)} stocks)")
        print("  " + ", ".join(f"{k} {v:.1%}" for k, v in held.items()))
    return prices, daily_returns, portfolios, corr, frontier


if __name__ == "__main__":
    main()
