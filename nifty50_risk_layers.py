"""
Risk layers for the optimised Nifty 50 portfolios, compared out of sample.

Training (models, parameters, portfolios): START to SPLIT, which includes the COVID crash.
Test: after SPLIT to today, which includes the 4 June 2024 election-day shock.

  Static          sample covariance Σ from the training window (the SLSQP input today)
  Bottom-up       one-factor model with a beta per stock in each VIX regime:
                      Σₜ = σ²ₘ,ₜ βₜβₜ' + diag(hₜ)
                  σ²ₘ,ₜ: GJR-GARCH-X on Nifty with lagged India VIX in the variance equation
                  βₜ: low- or high-VIX betas, chosen by yesterday's VIX
                  hₜ: GJR-GARCH-X on each stock's residual, also driven by lagged VIX
  Top-down        GJR-GARCH-X fitted directly to each portfolio's return series
  TD + BU input   blend 1: top-down with the bottom-up forecast as an extra variance term,
                      hₜ = ω + (α + γ·1[ε<0]) εₜ₋₁² + β hₜ₋₁ + δ VIX²ₜ₋₁ + φ h_BU,ₜ
  Blend           blend 3: λ·h_TD + (1-λ)·h_BU, λ chosen on a validation year (2023) using
                  models fitted to VAL_SPLIT, never on the test window
  Blend (state)   blend 3 with separate λ for calm and high-VIX days

Blend 2 (top-down sets the level, bottom-up the structure) is used for stress attribution:
    ρ̄_cal = (h_TD - A) / (B - A),   A = Σwᵢ²σᵢ²,  B = (Σwᵢσᵢ)²
It is not scored as a forecast because its portfolio variance equals h_TD by construction.

VaR is filtered historical simulation: the 1% quantile of each model's standardised training
returns times σₜ. The stress-aware portfolio maximises Sharpe subject to its volatility under
the bottom-up stressed covariance Σₛ staying below a limit.

Requires: pip install numpy pandas matplotlib yfinance scipy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import yfinance as yf
from scipy.optimize import minimize
from scipy.stats import chi2, spearmanr

from nifty50_efficient_frontier import (
    TICKERS, TRADING_DAYS, RISK_FREE_RATE, MAX_WEIGHT,
    min_variance_for_target, max_sharpe, optimised_frontier)

# ---------------- Parameters ----------------
START = "2019-01-01"        # training starts here, so it includes the COVID crash
VAL_SPLIT = "2022-12-31"    # blend weights are chosen on the year after this
SPLIT = "2023-12-31"        # fit on data up to here, test after
HIGH_VIX_PCT = 0.80         # yesterday's VIX above this training percentile = high regime
STRESS_NIFTY = -0.04        # hypothetical scenario: Nifty -4% day ...
STRESS_VIX = 35.0           # ... with India VIX at 35
STRESS_LIMIT = 0.85         # stress-aware cap: 85% of max-Sharpe's stressed volatility
VAR_LEVEL = 0.01            # 99% VaR
EVENT = "2024-06-04"        # election-result day
LAMBDAS = np.linspace(0, 1, 21)
DYNAMIC = ["Bottom-up", "Top-down", "TD + BU input", "Blend", "Blend (state)"]
MODELS = ["Static"] + DYNAMIC
PLOT_MODELS = ["Static", "Bottom-up", "Top-down", "Blend"]
COLORS = {"Static": "#52514e", "Bottom-up": "#2a78d6", "Top-down": "#eb6834",
          "Blend": "#1baf7a"}


def load_data():
    """Daily stock, Nifty and India VIX data from START. Only stocks with prices from the
    start are kept, so every model sees the COVID period."""
    raw = yf.download(TICKERS + ["^NSEI", "^INDIAVIX"], start=START, auto_adjust=True,
                      progress=False)["Close"]
    failed = raw.columns[raw.isna().all()].tolist()
    if failed:
        retry = yf.download(failed, start=START, auto_adjust=True, progress=False,
                            threads=False)["Close"]
        raw[retry.columns] = retry
    raw = raw.rename(columns=lambda c: c.replace(".NS", "")) \
             .rename(columns={"^NSEI": "NIFTY", "^INDIAVIX": "VIX"})
    cutoff = pd.Timestamp(START) + pd.Timedelta(days=14)
    names = [t.replace(".NS", "") for t in TICKERS]
    keep = [c for c in names if c in raw and raw[c].first_valid_index() is not None
            and raw[c].first_valid_index() <= cutoff]
    dropped = sorted(set(names) - set(keep))
    df = raw[keep + ["NIFTY", "VIX"]].dropna(subset=["NIFTY"]).ffill().dropna()
    rets = df[keep].pct_change().iloc[1:]
    return rets, df.NIFTY.pct_change().iloc[1:], df.VIX.iloc[1:], dropped


# ---------------- GJR-GARCH(1,1)-X ----------------
def gjr_filter(p, r, X, s0):
    """σ²ₜ = ω + (α + γ·1[rₜ₋₁<0]) rₜ₋₁² + β σ²ₜ₋₁ + δ'Xₜ₋₁"""
    omega, alpha, gamma, beta = p[:4]
    ex = X @ p[4:]
    s2 = np.empty(len(r))
    s2[0] = s0
    for t in range(1, len(r)):
        e = r[t - 1]
        s2[t] = omega + (alpha + gamma * (e < 0)) * e * e + beta * s2[t - 1] + ex[t - 1]
    return s2


def gjr_fit(r, X):
    """Gaussian QMLE (returns in %). X holds exogenous variance terms, one per column
    (may have zero columns). Returns parameters, starting variance and negative log-lik."""
    s0, k = r.var(), X.shape[1]

    def nll(p):
        if p[1] + p[2] / 2 + p[3] >= 0.999:
            return 1e10
        s2 = gjr_filter(p, r, X, s0)
        return 0.5 * np.sum(np.log(s2) + r ** 2 / s2)

    bounds = [(1e-6, None), (0, 1), (0, 1), (0, 0.998)] + [(0, None)] * k
    res = minimize(nll, [0.05 * s0, 0.05, 0.05, 0.85] + [0.05] * k, method="L-BFGS-B",
                   bounds=bounds)
    return res.x, s0, res.fun


# ---------------- Risk layers ----------------
def fit_bottom_up(R, m, x, high, tr):
    """Market GARCH-X, regime betas and residual GARCH-X. Returns everything needed to
    build Σₜ for any weights, plus the calm and stressed covariances (daily, %²)."""
    X = x[:, None]
    pm, s0m, _ = gjr_fit(m[tr], X[tr])
    s2m = gjr_filter(pm, m, X, s0m)
    betas = {}
    for name, mask in [("low", tr & ~high), ("high", tr & high)]:
        betas[name] = R[mask].T @ m[mask] / (m[mask] ** 2).sum()
    B = np.where(high[:, None], betas["high"], betas["low"])
    E = R - B * m[:, None]
    H = np.empty_like(R)
    idio = np.empty((R.shape[1], 5))
    idio_s0 = np.empty(R.shape[1])
    for i in range(R.shape[1]):
        idio[i], idio_s0[i], _ = gjr_fit(E[tr, i], X[tr])
        H[:, i] = gjr_filter(idio[i], E[:, i], X, idio_s0[i])

    calm, hot = tr & ~high, tr & high
    x_stress = STRESS_VIX ** 2 / TRADING_DAYS
    omega, alpha, gamma, beta, delta = pm
    s2m_stress = (omega + (alpha + gamma) * (STRESS_NIFTY * 100) ** 2
                  + beta * s2m[hot].mean() + delta * x_stress)
    om, a, g, b, d = idio.T
    h_stress = om + (a + g / 2) * (E[hot] ** 2).mean(0) + b * H[hot].mean(0) + d * x_stress
    cov_calm = s2m[calm].mean() * np.outer(betas["low"], betas["low"]) + np.diag(H[calm].mean(0))
    cov_stress = s2m_stress * np.outer(betas["high"], betas["high"]) + np.diag(h_stress)
    return dict(s2m=s2m, B=B, H=H, S2=B ** 2 * s2m[:, None] + H, betas=betas,
                cov_calm=cov_calm, cov_stress=cov_stress, s2m_stress=s2m_stress,
                calm=calm, hot=hot, idio=idio, idio_s0=idio_s0, pm=pm, s0m=s0m, E=E)


def portfolio_layers(w, R, x, tr, cov_static, bu):
    """Daily variance forecasts (%²) of portfolio w: static, bottom-up, top-down and
    top-down with the bottom-up forecast as an input (blend 1)."""
    rp = R @ w
    X = x[:, None]
    h_bu = (bu["B"] @ w) ** 2 * bu["s2m"] + bu["H"] @ w ** 2
    p_td, s0, nll_td = gjr_fit(rp[tr], X[tr])
    h_td = gjr_filter(p_td, rp, X, s0)
    # row t-1 must hold the bottom-up forecast for day t (made with data to t-1)
    X2 = np.column_stack([x, np.append(h_bu[1:], h_bu[-1])])
    p_tb, s0b, nll_tb = gjr_fit(rp[tr], X2[tr])
    h_tb = gjr_filter(p_tb, rp, X2, s0b)
    # φ >= 0 sits on a boundary, so the LR p-value is halved
    lr_p = chi2.sf(max(2 * (nll_td - nll_tb), 0), 1) / 2

    omega, alpha, gamma, beta, delta = p_td
    shock = (w @ bu["betas"]["high"]) * STRESS_NIFTY * 100
    stress = {"Static": w @ cov_static @ w, "Bottom-up": w @ bu["cov_stress"] @ w,
              "Top-down": omega + (alpha + gamma) * shock ** 2 + beta * h_td[bu["hot"]].mean()
              + delta * STRESS_VIX ** 2 / TRADING_DAYS}
    s2 = {"Static": np.full(len(rp), w @ cov_static @ w), "Bottom-up": h_bu,
          "Top-down": h_td, "TD + BU input": h_tb}
    return rp, s2, stress, dict(td=p_td, tb=p_tb, lr_p=lr_p, s0_td=s0, s0_tb=s0b)


def qlike(r, h):
    return np.mean(np.log(h) + r ** 2 / h)


def best_lambda(rp, s2, mask):
    return LAMBDAS[np.argmin([qlike(rp[mask], l * s2["Top-down"][mask]
                                    + (1 - l) * s2["Bottom-up"][mask]) for l in LAMBDAS])]


def choose_lambdas(rp, s2_val, val, high_val, min_days=20):
    """Blend 3 weights from the validation year: overall, calm days, high-VIX days.
    A regime with fewer than min_days falls back to the overall λ."""
    lam = best_lambda(rp, s2_val, val)
    pick = lambda mask: best_lambda(rp, s2_val, mask) if mask.sum() >= min_days else lam
    return lam, pick(val & ~high_val), pick(val & high_val), int((val & high_val).sum())


def kupiec_p(breaches, n, p):
    if breaches == 0:
        lr = -2 * n * np.log(1 - p)
    else:
        ph = breaches / n
        lr = -2 * ((n - breaches) * np.log((1 - p) / (1 - ph)) + breaches * np.log(p / ph))
    return chi2.sf(lr, 1)


def score(rp, s2, tr, te, event_i):
    """Out-of-sample QLIKE, bias, VaR breaches (filtered historical simulation), event z."""
    z_train = rp[tr] / np.sqrt(s2[tr])
    var = -np.quantile(z_train, VAR_LEVEL) * np.sqrt(s2[te])
    r = rp[te]
    n, hits = len(r), int((r < -var).sum())
    return {"QLIKE": qlike(r, s2[te]),
            "Bias": np.mean(r ** 2) / np.mean(s2[te]),        # realised / forecast variance
            "VaR breaches": hits, "Expected": n * VAR_LEVEL,
            "Kupiec p": kupiec_p(hits, n, VAR_LEVEL),
            "4 Jun σ": np.sqrt(s2[event_i] * TRADING_DAYS) / 100,
            "4 Jun z": rp[event_i] / np.sqrt(s2[event_i])}


def newey_west_ols(y, X, lags=5):
    """OLS with Newey-West standard errors."""
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ coef
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for L in range(1, lags + 1):
        G = Xe[L:].T @ Xe[:-L]
        S += (1 - L / (lags + 1)) * (G + G.T)
    XtX = np.linalg.inv(X.T @ X)
    return coef, np.sqrt(np.diag(XtX @ S @ XtX)), 1 - e.var() / y.var()


def mincer_zarnowitz(r2, s2):
    """Regress realised r² on forecast σ²: slope 1 = calibrated, < 1 = forecasts swing more
    than realised risk (over-reaction), > 1 = too sluggish."""
    coef, se, r2_fit = newey_west_ols(r2, np.column_stack([np.ones_like(s2), s2]))
    return {"MZ slope": coef[1], "slope SE": se[1], "MZ R²": r2_fit}


def encompassing(r2, h_td, h_bu):
    """r² = a + b_TD h_TD + b_BU h_BU. A significant b_BU means bottom-up adds information
    top-down lacks, and vice versa."""
    coef, se, _ = newey_west_ols(r2, np.column_stack([np.ones_like(h_td), h_td, h_bu]))
    return {"b_TD": coef[1], "t_TD": coef[1] / se[1], "b_BU": coef[2], "t_BU": coef[2] / se[2]}


def calibrated_rho(w, sd, h):
    """Blend 2: the average correlation that makes Σwᵢ²σᵢ² + ρ̄·(cross terms) equal h."""
    A = (w ** 2 * sd ** 2).sum(-1)
    B = (w * sd).sum(-1) ** 2
    return (h - A) / (B - A)


def rho_split(w, sd_c, sd_s, rho_c, rho_s):
    """Shapley split of the variance change into volatility and correlation effects,
    under a single average correlation."""
    f = lambda sd, rho: (w ** 2 * sd ** 2).sum() + rho * ((w * sd).sum() ** 2
                                                         - (w ** 2 * sd ** 2).sum())
    vol = 0.5 * ((f(sd_s, rho_c) - f(sd_c, rho_c)) + (f(sd_s, rho_s) - f(sd_c, rho_s)))
    corr = 0.5 * ((f(sd_c, rho_s) - f(sd_c, rho_c)) + (f(sd_s, rho_s) - f(sd_s, rho_c)))
    return vol / (vol + corr), corr / (vol + corr)


def shapley_split(w, cov_c, cov_s):
    """Split the calm-to-stress change in portfolio variance into a volatility effect and
    a correlation effect using the full matrices, averaged over both orderings."""
    def parts(c):
        sd = np.sqrt(np.diag(c))
        return sd, c / np.outer(sd, sd)
    (sc, rc), (ss, rs) = parts(cov_c), parts(cov_s)
    f = lambda s, r: (w * s) @ r @ (w * s)
    vol = 0.5 * ((f(ss, rc) - f(sc, rc)) + (f(ss, rs) - f(sc, rs)))
    corr = 0.5 * ((f(sc, rs) - f(sc, rc)) + (f(ss, rs) - f(ss, rc)))
    ws_c, ws_s = w * sc, w * ss
    rho_bar = lambda ws, r: ((ws @ r @ ws - ws @ ws) / (ws.sum() ** 2 - ws @ ws))
    return vol, corr, rho_bar(ws_c, rc), rho_bar(ws_s, rs)


def max_sharpe_stress_capped(mu, cov, cov_s, limit, x0):
    """SLSQP: maximise Sharpe under the normal Σ, with stressed volatility ≤ limit."""
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones_like(w)},
            {"type": "ineq", "fun": lambda w: limit ** 2 - w @ cov_s @ w,
             "jac": lambda w: -2 * cov_s @ w}]
    neg_sharpe = lambda w: -(w @ mu - RISK_FREE_RATE) / np.sqrt(w @ cov @ w)
    return minimize(neg_sharpe, x0, method="SLSQP", bounds=[(0, MAX_WEIGHT)] * len(x0),
                    constraints=cons, options={"maxiter": 500, "ftol": 1e-12})


def realised_stats(r, nifty, event):
    cum = (1 + r).cumprod()
    stress_days = nifty < -0.02
    return {"Return": r.mean() * TRADING_DAYS, "Volatility": r.std() * np.sqrt(TRADING_DAYS),
            "Sharpe": (r.mean() * TRADING_DAYS - RISK_FREE_RATE) / (r.std() * np.sqrt(TRADING_DAYS)),
            "Worst day": r.min(), "Max drawdown": (cum / cum.cummax() - 1).min(),
            f"Avg on Nifty<-2% days ({stress_days.sum()})": r[stress_days].mean(),
            "4 Jun 2024": r.loc[event]}


def aftershock_ratios(rp, s2, te, threshold=2.5, horizon=10):
    """Event study over test days with |r| > threshold %: realised / forecast variance on
    each of the following days. Below 1 = the model over-reacted to the shock."""
    shocks = [t for t in np.where(te & (np.abs(rp) > threshold))[0] if t + horizon < len(rp)]
    return np.array([np.mean([rp[t + k] ** 2 for t in shocks]) /
                     np.mean([s2[t + k] for t in shocks]) for k in range(1, horizon + 1)]), len(shocks)


def topdown_diagnostics(rp, s2, x, td, dates, tr, te, event_i):
    """Is the top-down GARCH-X calibrated or overfitted? Calibration, in- vs out-of-sample
    fit, the spike after the event, and parameter stability across fitting windows."""
    pct = lambda v: f"{v:.2%}"
    print("\n8. Diagnostics, Max Sharpe portfolio")
    mz = pd.DataFrame({m: mincer_zarnowitz(rp[te] ** 2, s2[m][te]) for m in DYNAMIC}).T
    print("Calibration, test window (Mincer-Zarnowitz; static has no slope, it is constant):")
    print(mz.to_string(float_format="{:.2f}".format))

    fit = pd.DataFrame({w: {"QLIKE": qlike(rp[mask], s2["Top-down"][mask]),
                            "Bias": np.mean(rp[mask] ** 2) / np.mean(s2["Top-down"][mask])}
                        for w, mask in [("Train (in-sample)", tr), ("Test (out-of-sample)", te)]}).T
    print("\nTop-down in- vs out-of-sample:")
    print(fit.to_string(float_format="{:.3f}".format))

    omega, alpha, gamma, beta, delta = td
    t = event_i + 1
    e = rp[t - 1]
    parts = {"Constant ω": omega, "Shock (α+γ)·r²": (alpha + gamma * (e < 0)) * e * e,
             "Persistence β·σ²": beta * s2["Top-down"][t - 1], "VIX δ·x": delta * x[t - 1]}
    total = sum(parts.values())
    print(f"\nTop-down forecast for {dates[t].date()} = {np.sqrt(total * TRADING_DAYS) / 100:.1%} "
          f"annualised, built from:")
    for k, v in parts.items():
        print(f"  {k:<18} {v / total:6.1%}")
    pers = alpha + gamma / 2 + beta
    print(f"  persistence α+γ/2+β = {pers:.3f}, shock half-life "
          f"{np.log(0.5) / np.log(pers):.1f} days")

    win = slice(event_i - 2, event_i + 7)
    ann = lambda v: np.sqrt(v * TRADING_DAYS) / 100
    around = pd.DataFrame({"Return": rp[win] / 100,
                           "|r| annualised": np.abs(rp[win]) / 100 * np.sqrt(TRADING_DAYS),
                           **{f"{m} σ": ann(s2[m][win]) for m in PLOT_MODELS}},
                          index=dates[win].date)
    print("\nAround the event (σ = forecast made the evening before):")
    print(around.to_string(float_format=pct))

    windows = {"2019-2021 (COVID)": dates <= "2021-12-31",
               "2022-2023": (dates > "2021-12-31") & tr,
               "2019-2023 (used)": tr, "Full sample": np.ones(len(rp), bool)}
    rows = {}
    for name, mask in windows.items():
        p, _, _ = gjr_fit(rp[mask], x[mask][:, None])
        rows[name] = dict(zip(["ω", "α", "γ", "β", "δ"], p), persistence=p[1] + p[2] / 2 + p[3])
    print("\nTop-down parameters by fitting window (stable = not fitted to one episode):")
    print(pd.DataFrame(rows).T.to_string(float_format="{:.3f}".format))


def style(ax):
    ax.grid(color="#e1e0d9", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors="#898781")
    ax.legend(frameon=False, fontsize=9)


def plot_results(dates, rp, s2, frontiers, points):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))
    realised = pd.Series(rp, dates).pow(2).rolling(5).mean().pow(0.5) * np.sqrt(TRADING_DAYS) / 100
    ax1.plot(realised.index, realised, color="#c3c2b7", lw=1, label="Realised (5-day)")
    for name in PLOT_MODELS:
        ax1.plot(dates, np.sqrt(s2[name] * TRADING_DAYS) / 100, color=COLORS[name], lw=1.6,
                 ls="--" if name == "Static" else "-", label=name)
    ax1.axvline(pd.Timestamp(EVENT), color="#898781", lw=0.8, ls=":")
    ax1.annotate("4 Jun 2024", (pd.Timestamp(EVENT), ax1.get_ylim()[1] * 0.95),
                 xytext=(4, 0), textcoords="offset points", fontsize=9, color="#52514e")
    ax1.set_title("Max Sharpe - forecast vs realised volatility, test window", loc="left",
                  fontsize=13, color="#0b0b0b")
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax1.set_ylabel("Annualised volatility", color="#52514e")

    for name, (f, color, ls) in frontiers.items():
        ax2.plot(f.Volatility, f.Return, color=color, ls=ls, lw=2, label=name)
    for label, (vol, ret, color, marker) in points.items():
        ax2.scatter(vol, ret, marker=marker, s=170 if marker == "*" else 80, color=color,
                    edgecolors="white", linewidths=1.5, zorder=3, label=label)
    ax2.set_title(f"Normal vs stressed frontier (Nifty {STRESS_NIFTY:.0%}, VIX {STRESS_VIX:.0f})",
                  loc="left", fontsize=13, color="#0b0b0b")
    ax2.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax2.set_xlabel("Annualised volatility (σ)", color="#52514e")
    ax2.set_ylabel("Annualised expected return (training window)", color="#52514e")
    style(ax1)
    style(ax2)
    fig.tight_layout()
    fig.savefig("nifty50_risk_layers.png", dpi=150)
    plt.show()


def plot_diagnostics(dates, rp, s2, te, event_i):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    win = slice(event_i - 15, event_i + 25)
    d = dates[win]
    ax1.bar(d, np.abs(rp[win]) / 100 * np.sqrt(TRADING_DAYS), color="#c3c2b7", width=0.8,
            label="Realised |r| × √252")
    for m in PLOT_MODELS:
        ax1.plot(d, np.sqrt(s2[m][win] * TRADING_DAYS) / 100, color=COLORS[m], lw=2,
                 ls="--" if m == "Static" else "-", label=f"{m} forecast")
    ax1.set_title("Max Sharpe around 4 June 2024 - forecast vs realised", loc="left",
                  fontsize=13, color="#0b0b0b")
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax1.set_ylabel("Annualised volatility", color="#52514e")
    ax1.tick_params(axis="x", rotation=45)

    for m in PLOT_MODELS:
        ratios, n = aftershock_ratios(rp, s2[m], te)
        ax2.plot(range(1, len(ratios) + 1), ratios, color=COLORS[m], lw=2, marker="o", ms=5,
                 ls="--" if m == "Static" else "-", label=m)
    ax2.axhline(1, color="#898781", lw=1)
    ax2.set_ylim(bottom=0)
    ax2.annotate("↑ under-reacting", (5.2, 1.08), fontsize=9, color="#52514e")
    ax2.annotate("↓ over-reacting", (5.2, 0.15), fontsize=9, color="#52514e")
    ax2.set_title(f"After large shocks (|r| > 2.5%, {n} in test): realised ÷ forecast variance",
                  loc="left", fontsize=13, color="#0b0b0b")
    ax2.set_xlabel("Trading days after the shock", color="#52514e")
    ax2.set_ylabel("Realised ÷ forecast variance (1 = calibrated)", color="#52514e")
    ax2.set_xticks(range(1, 11))
    style(ax1)
    style(ax2)
    fig.tight_layout()
    fig.savefig("nifty50_topdown_check.png", dpi=150)
    plt.show()


def main():
    rets, nifty, vix, dropped = load_data()
    cols, dates = rets.columns, rets.index
    tr, tr_val = dates <= SPLIT, dates <= VAL_SPLIT
    te, val = ~tr, tr & ~tr_val
    event_i = dates.get_loc(pd.Timestamp(EVENT))
    print(f"{len(cols)} stocks; train {tr.sum()} days {dates[0].date()} to {SPLIT}, "
          f"test {te.sum()} days to {dates[-1].date()}")
    print(f"Dropped (no prices from {START}): {', '.join(dropped)}")

    R, m = rets.values * 100, nifty.values * 100          # daily returns in %
    x = vix.values ** 2 / TRADING_DAYS                     # VIX as daily variance, %²
    vix_prev = vix.shift(1).bfill().values
    high = vix_prev >= np.quantile(vix_prev[tr], HIGH_VIX_PCT)
    high_val = vix_prev >= np.quantile(vix_prev[tr_val], HIGH_VIX_PCT)

    # ---- Portfolios, optimised on the training window only ----
    mu = rets[tr].mean().values * TRADING_DAYS
    cov = rets[tr].cov().values * TRADING_DAYS
    n = len(cols)
    w_mv = np.clip(min_variance_for_target(mu, cov, MAX_WEIGHT, None,
                                           np.full(n, 1 / n)).x, 0, MAX_WEIGHT)
    w_ms = np.clip(max_sharpe(mu, cov, MAX_WEIGHT, w_mv).x, 0, MAX_WEIGHT)

    print("Fitting GJR-GARCH models (training and validation fits)...")
    bu = fit_bottom_up(R, m, x, high, tr)
    bu_val = fit_bottom_up(R, m, x, high_val, tr_val)
    d = bu["idio"][:, 4]
    print(f"Stock-specific VIX loading δ: {(d > 1e-3).sum()}/{len(d)} stocks use it, "
          f"median δ {np.median(d):.3f}")
    cov_static = np.cov(R[tr], rowvar=False)
    to_ann = TRADING_DAYS / 1e4                            # daily %² -> annual decimal
    cov_s, cov_c = bu["cov_stress"] * to_ann, bu["cov_calm"] * to_ann

    # ---- Stress-aware portfolio: Sharpe with a cap on bottom-up stressed volatility ----
    w_smv = np.clip(min_variance_for_target(mu, cov_s, MAX_WEIGHT, None, w_mv).x, 0, MAX_WEIGHT)
    floor, ms_stress = np.sqrt(w_smv @ cov_s @ w_smv), np.sqrt(w_ms @ cov_s @ w_ms)
    limit = max(STRESS_LIMIT * ms_stress, 0.5 * (floor + ms_stress))
    w_sa = np.clip(max_sharpe_stress_capped(mu, cov, cov_s, limit, w_smv).x, 0, MAX_WEIGHT)
    portfolios = {"Min variance": w_mv, "Max Sharpe": w_ms, "Stress-aware": w_sa}
    print(f"Stress cap: {limit:.1%} (max Sharpe stressed {ms_stress:.1%}, "
          f"lowest achievable {floor:.1%})")

    rows, scen_rows, split_rows, blend_rows, enc_rows, cal_rows = [], [], [], [], [], []
    for pname, w in portfolios.items():
        rp, s2, stress, fit = portfolio_layers(w, R, x, tr, cov_static, bu)
        _, s2_val, _, _ = portfolio_layers(w, R, x, tr_val, cov_static, bu_val)

        # Blend 3: λ from the validation year, applied to the training-window fits
        lam, lam_lo, lam_hi, n_hi = choose_lambdas(rp, s2_val, val, high_val)
        s2["Blend"] = lam * s2["Top-down"] + (1 - lam) * s2["Bottom-up"]
        lam_t = np.where(high, lam_hi, lam_lo)
        s2["Blend (state)"] = lam_t * s2["Top-down"] + (1 - lam_t) * s2["Bottom-up"]
        p_td, p_tb = fit["td"], fit["tb"]
        blend_rows.append({"Portfolio": pname, "λ (2023)": lam, "λ calm": lam_lo,
                           f"λ high ({n_hi}d)": lam_hi, "λ hindsight": best_lambda(rp, s2, te),
                           "φ (BU input)": p_tb[5], "φ LR p": fit["lr_p"],
                           "α+γ TD": p_td[1] + p_td[2], "α+γ TD+BU": p_tb[1] + p_tb[2],
                           "β TD": p_td[3], "β TD+BU": p_tb[3]})
        enc_rows.append({"Portfolio": pname, **encompassing(rp[te] ** 2, s2["Top-down"][te],
                                                            s2["Bottom-up"][te])})

        for model in MODELS:
            rows.append({"Portfolio": pname, "Model": model,
                         **score(rp, s2[model], tr, te, event_i)})
        for model in ["Static", "Bottom-up", "Top-down"]:
            scen_rows.append({"Portfolio": pname, "Model": model,
                              "Stressed σ": np.sqrt(stress[model] * TRADING_DAYS) / 100})
        vol, corr, rho_c, rho_s = shapley_split(w, cov_c, cov_s)
        split_rows.append({"Portfolio": pname, "Calm σ": np.sqrt(w @ cov_c @ w),
                           "Stressed σ": np.sqrt(w @ cov_s @ w), "ρ̄ calm": rho_c,
                           "ρ̄ stress": rho_s, "Vol effect": vol / (vol + corr),
                           "Corr effect": corr / (vol + corr)})

        # Blend 2: top-down sets the level, bottom-up the stock volatilities
        sd_t = np.sqrt(bu["S2"])
        rho_cal = calibrated_rho(w, sd_t, s2["Top-down"])[te]
        rho_bu = calibrated_rho(w, sd_t, s2["Bottom-up"])[te]
        sd_c, sd_s = np.sqrt(np.diag(bu["cov_calm"])), np.sqrt(np.diag(bu["cov_stress"]))
        h_td_c = s2["Top-down"][bu["calm"]].mean()
        rc = calibrated_rho(w, sd_c, h_td_c)
        rs = calibrated_rho(w, sd_s, stress["Top-down"])
        v_eff, c_eff = rho_split(w, sd_c, sd_s, rc, rs)
        cal_rows.append({"Portfolio": pname, "ρ̄_cal median": np.median(rho_cal),
                         "ρ̄_BU median": np.median(rho_bu),
                         "in [0,1]": np.mean((rho_cal >= 0) & (rho_cal <= 1)),
                         "> 1": np.mean(rho_cal > 1), "4 Jun": rho_cal[event_i - te.argmax()],
                         "5 Jun": rho_cal[event_i + 1 - te.argmax()],
                         "ρ̄_cal calm": rc, "ρ̄_cal stress": rs,
                         "Vol effect": v_eff, "Corr effect": c_eff})
        if pname == "Max Sharpe":
            rp_ms, s2_ms, td_ms = rp, s2, fit["td"]

    pct = lambda v: f"{v:.2%}"
    f2 = "{:.2f}".format
    scores = pd.DataFrame(rows).set_index(["Portfolio", "Model"])
    scores.to_csv("nifty50_risk_layers.csv")
    print("\n1. Risk models, test window (lower QLIKE is better; Bias = realised / forecast "
          "variance; |z| > 3 on 4 Jun = model missed it)")
    print(scores.to_string(formatters={"QLIKE": "{:.3f}".format, "Bias": f2,
                                       "Expected": "{:.1f}".format, "Kupiec p": f2,
                                       "4 Jun σ": pct, "4 Jun z": "{:+.1f}".format}))

    print("\n2. Blends: weights and the bottom-up input to top-down")
    print(pd.DataFrame(blend_rows).set_index("Portfolio").to_string(float_format=f2))
    print("\n3. Encompassing test, test window: r² = a + b_TD·h_TD + b_BU·h_BU "
          "(|t| > 2 = that model adds information)")
    print(pd.DataFrame(enc_rows).set_index("Portfolio").to_string(float_format=f2))

    print("\n4. Blend 2: correlation implied by top-down, given bottom-up stock volatilities")
    print(pd.DataFrame(cal_rows).set_index("Portfolio").to_string(formatters={
        "ρ̄_cal median": f2, "ρ̄_BU median": f2, "in [0,1]": "{:.0%}".format,
        "> 1": "{:.0%}".format, "4 Jun": f2, "5 Jun": f2, "ρ̄_cal calm": f2,
        "ρ̄_cal stress": f2, "Vol effect": "{:.0%}".format, "Corr effect": "{:.0%}".format}))

    print(f"\n5. Hypothetical stress (Nifty {STRESS_NIFTY:.0%}, VIX {STRESS_VIX:.0f}): "
          "annualised σ by model, and the bottom-up split of the increase")
    scen = pd.DataFrame(scen_rows).pivot(index="Portfolio", columns="Model", values="Stressed σ")
    split = pd.DataFrame(split_rows).set_index("Portfolio")
    print(scen[["Static", "Bottom-up", "Top-down"]].join(split[["ρ̄ calm", "ρ̄ stress",
                                                                   "Vol effect", "Corr effect"]])
          .to_string(formatters={"Static": pct, "Bottom-up": pct, "Top-down": pct,
                                 "ρ̄ calm": f2, "ρ̄ stress": f2,
                                 "Vol effect": "{:.0%}".format, "Corr effect": "{:.0%}".format}))

    # ---- Closing the loop: who did the stress-aware portfolio cut? ----
    b_lo, b_hi = bu["betas"]["low"], bu["betas"]["high"]
    held = (w_ms > 1e-3) | (w_sa > 1e-3)
    moves = pd.DataFrame({"Max Sharpe": w_ms, "Stress-aware": w_sa, "Change": w_sa - w_ms,
                          "β calm": b_lo, "β stress": b_hi}, index=cols)[held].sort_values("Change")
    rho, pval = spearmanr(moves.Change, moves["β stress"])
    print("\n6. Weight changes, max Sharpe -> stress-aware")
    print(moves.to_string(formatters={"Max Sharpe": pct, "Stress-aware": pct,
                                      "Change": "{:+.2%}".format, "β calm": f2, "β stress": f2}))
    print(f"Spearman(weight change, stress beta) = {rho:+.2f} (p = {pval:.2f}); "
          "negative = high-crisis-beta stocks were cut")

    real = pd.DataFrame({p: realised_stats(rets[te] @ w, nifty[te], EVENT)
                         for p, w in portfolios.items()}).T
    print("\n7. Realised, test window")
    print(real.to_string(formatters={c: (f2 if c == "Sharpe" else pct) for c in real.columns}))
    real.to_csv("nifty50_risk_layers_realised.csv")

    topdown_diagnostics(rp_ms, s2_ms, x, td_ms, dates, tr, te, event_i)

    # ---- Charts ----
    f_norm, _ = optimised_frontier(mu, cov, MAX_WEIGHT)
    f_str, _ = optimised_frontier(mu, cov_s, MAX_WEIGHT)
    frontiers = {"Frontier, normal Σ": (f_norm, "#52514e", "-"),
                 "Frontier, stressed Σₛ": (f_str, "#2a78d6", "-")}
    points = {"Max Sharpe, normal σ": (np.sqrt(w_ms @ cov @ w_ms), w_ms @ mu, "#52514e", "*"),
              "Max Sharpe, stressed σ": (ms_stress, w_ms @ mu, "#2a78d6", "*"),
              "Stress-aware, stressed σ": (np.sqrt(w_sa @ cov_s @ w_sa), w_sa @ mu, "#2a78d6", "D")}
    plot_results(dates[te], rp_ms[te], {k: v[te] for k, v in s2_ms.items()}, frontiers, points)
    plot_diagnostics(dates, rp_ms, s2_ms, te, event_i)
    return scores, real


if __name__ == "__main__":
    main()
