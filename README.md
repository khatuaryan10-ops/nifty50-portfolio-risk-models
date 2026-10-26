# Nifty 50 Portfolio Risk Models

How risky are optimised Nifty 50 portfolios, and how well do different risk models forecast that risk out of sample?

The project builds a capped long-only efficient frontier and stress-tests it against macro factors. It then compares six volatility models (static, bottom-up, top-down and three blends) on 2024–2026 data the models never saw, including the 4 June 2024 election-result shock.

All data comes from Yahoo Finance via `yfinance`. No API key is needed.

## Scripts

Run them in this order from the repository root. Later scripts import functions and parameters from earlier ones by file name, so keep all five in the same folder.

| # | Script | What it does | Runtime |
|---|--------|--------------|---------|
| 1 | `nifty50_efficient_frontier.py` | Simulates 100,000 random portfolios, plots the correlation matrix, and builds the efficient frontier with SLSQP, plus the exact min-variance and max-Sharpe portfolios | < 1 min |
| 2 | `nifty50_factor_stress.py` | Multi-factor macro stress test of those two portfolios, with a stressed frontier for each scenario | < 1 min |
| 3 | `nifty50_risk_layers.py` | Fits and scores static, bottom-up, top-down and blended risk models out of sample, and builds a stress-aware portfolio | several min |
| 4 | `nifty50_portfolio_var.py` | Daily volatility and 99% VaR of both portfolios from 2019 to today, with a backtest of breaches | several min |
| 5 | `nifty50_vol_attribution.py` | Case study of May–July 2024: an exact split of forecast variance by source, compared with India VIX | several min |

Scripts 3–5 each refit every GJR-GARCH model from scratch (one per stock, plus the market and each portfolio), so each takes a few minutes. Scripts 2–5 only import from 1 and 3, so you can run them on their own without running the earlier scripts first.

```bash
pip install -r requirements.txt
python nifty50_efficient_frontier.py
python nifty50_factor_stress.py
python nifty50_risk_layers.py
python nifty50_portfolio_var.py
python nifty50_vol_attribution.py
```

Each script prints its tables to the console and writes its PNGs and CSVs to the working directory. Copies from the last run are in [`figures/`](figures) and [`results/`](results).

## Method

Common settings for all scripts: long-only, at most 10% in any one stock, a 6.5% risk-free rate (about the Indian 10-year G-Sec yield), and 252 trading days a year.

### 1. Efficient frontier

- Random portfolios are drawn from a Dirichlet distribution with α = 0.3. Values below 1 concentrate weight in a few stocks, which spreads the cloud of portfolios further across the risk–return plane. Any weight above the 10% cap is clipped and the excess handed to the uncapped stocks in proportion to their weights.
- The frontier is the minimum variance portfolio at each of 60 target returns, solved with SLSQP. Each solve starts from the previous target's weights, which cuts the number of solver steps.
- Max Sharpe is solved directly, starting from the best frontier point.

![Efficient frontier](figures/nifty50_efficient_frontier.png)

### 2. Macro factor stress test

Weekly returns of each stock are regressed on five factors: Nifty 50, a 10-year G-Sec ETF (bond prices), and Brent, gold and copper converted to rupees. Weekly data avoids the timing mismatch between the NSE close and the US commodity futures settlement.

Σ = B Σ_f B′ + Σ_e, where Σ_e is the full residual covariance (not just its diagonal).

A scenario shocks some factors. Every other factor moves by its expected value given those shocks, E[f_u | f_s] = Σ_us Σ_ss⁻¹ Δf_s, so a Nifty crash also pulls copper down. Factor volatilities can be scaled too. A repo rate change becomes a bond move through an assumed 0.6 pass-through to the 10-year yield and a duration of 7, and the risk-free rate rises by the same yield change.

| Scenario | Max Sharpe shock P&L | Max Sharpe Sharpe | Avg weighted correlation |
|---|---|---|---|
| Baseline | 0.0% | 1.31 | 0.23 |
| Repo +50bp | −2.3% | 1.12 | 0.24 |
| Nifty −20% crash | −19.1% | 0.02 | 0.51 |
| Brent +40% spike | −1.8% | 1.19 | 0.23 |
| Stagflation | −11.9% | 0.32 | 0.38 |

![Stressed frontiers](figures/nifty50_stressed_frontiers.png)

### 3. Risk models compared out of sample

Models are fitted on 2019–2023, which includes the COVID crash, and scored on 2024 to today. The portfolios are re-optimised on the same training window, so they differ from those in script 1.

| Model | Variance forecast |
|---|---|
| Static | Sample covariance of the training window (the input to the SLSQP optimiser) |
| Bottom-up | One-factor model Σₜ = σ²ₘ,ₜ βₜβₜ′ + diag(hₜ). Market variance σ²ₘ,ₜ and each stock's residual variance hₜ follow a GJR-GARCH-X with lagged India VIX in the variance equation. The betas switch between a calm and a high-VIX set, depending on whether yesterday's VIX was above its 80th training percentile |
| Top-down | GJR-GARCH-X fitted directly to each portfolio's return series |
| TD + BU input (blend 1) | Top-down with the bottom-up forecast as an extra variance term, coefficient φ ≥ 0, tested with a likelihood ratio test |
| Blend (blend 3) | λ·h_TD + (1−λ)·h_BU, with λ chosen on 2023 using models fitted to 2019–2022. The test window is never used |
| Blend (state) | Blend 3 with separate λ for calm and high-VIX days |

Blend 2 (top-down sets the level, bottom-up the structure) is not scored as a forecast, since by construction it gives the same portfolio variance as top-down. It is used instead to back out the average correlation implied by the top-down level: ρ̄_cal = (h_TD − A)/(B − A), with A = Σwᵢ²σᵢ² and B = (Σwᵢσᵢ)².

**Scoring:** QLIKE, bias (realised ÷ forecast variance), 99% VaR breaches with the Kupiec test, the z-score on 4 June 2024, Mincer–Zarnowitz calibration, and an encompassing regression with Newey–West errors to check whether bottom-up adds information that top-down lacks.

**VaR** is filtered historical simulation: the 1% quantile of a model's standardised training returns times that day's forecast σ.

**Stress scenario:** Nifty −4% in a day with India VIX at 35. The rise in portfolio variance from calm to stressed is split into a volatility effect and a correlation effect using Shapley values, which average both orderings so the two parts add up exactly. A **stress-aware portfolio** maximises Sharpe subject to its volatility under the bottom-up stressed covariance staying below a limit.

Test-window results for the max-Sharpe portfolio (692 trading days, 6.9 VaR breaches expected):

| Model | QLIKE | Bias | VaR breaches | Kupiec p | 4 Jun 2024 z |
|---|---|---|---|---|---|
| Static | 1.004 | 0.63 | 2 | 0.03 | −4.7 |
| Bottom-up | 0.823 | 1.20 | 7 | 0.98 | −5.2 |
| Top-down | 0.806 | 0.99 | 6 | 0.72 | −4.7 |
| TD + BU input | 0.803 | 1.03 | 6 | 0.72 | −5.0 |
| Blend | 0.805 | 0.99 | 6 | 0.72 | −4.7 |

The static model overstates risk in the test window, as shown by its 0.63 bias and too few breaches. The bottom-up model understates it. The dynamic top-down models are close to calibrated, but every model treated the election-day loss as roughly a 5σ event. Full results for all three portfolios are in [`results/nifty50_risk_layers.csv`](results/nifty50_risk_layers.csv).

![Risk layers](figures/nifty50_risk_layers.png)
![Top-down diagnostics](figures/nifty50_topdown_check.png)

### 4. Volatility and VaR, 2019 to today

Static, bottom-up and top-down volatility forecasts against 21-day realised volatility, and a daily 99% VaR backtest. Breaches less than 10 trading days apart are grouped into episodes, and episodes near known shocks are labelled (COVID, the Russia–Ukraine invasion, the 2024 election, the August 2024 yen-carry sell-off, the April 2025 US tariffs).

![Portfolio volatility](figures/nifty50_portfolio_vol.png)
![Portfolio VaR](figures/nifty50_portfolio_var.png)

### 5. June 2024 election: where the forecast variance came from

Every GJR-GARCH-X variance is the sum of today's inputs plus β times yesterday's variance. Unrolling that recursion assigns all of σ²ₜ to the input that created it, so the split is exact and additive. The script checks that the parts sum to the forecast to 1e-8. For blend 1, the bottom-up input is unrolled the same way into its Nifty and stock-level parts. This shows how much India VIX and squared returns contribute through each channel, which is where any double counting would show up.

The forecasts are also compared with India VIX on a like-for-like basis. VIX is the Nifty's expected volatility over the next 30 calendar days, so it is set against the Nifty model's average forecast over 22 trading days, not a 1-day forecast.

![Volatility attribution](figures/nifty50_vol_attribution.png)

## Outputs

| File | From | Contents |
|---|---|---|
| `nifty50_optimal_portfolios.csv` | 1 | Min-variance and max-Sharpe weights and statistics |
| `nifty50_frontier.csv` | 1 | Every frontier point with its weights |
| `nifty50_factor_stress.csv` | 2 | Scenario results by portfolio |
| `nifty50_risk_layers.csv` | 3 | Out-of-sample scores, all models and portfolios |
| `nifty50_risk_layers_realised.csv` | 3 | Realised test-window return, volatility, drawdown |
| `nifty50_var_breaches.csv` | 4 | VaR breach episodes |

Script 1 also writes `nifty50_portfolios.csv` (statistics for all 100,000 random portfolios, about 9 MB) and `nifty50_portfolio_weights.npz` (their weights, about 35 MB). Both are excluded from the repository by `.gitignore`.

## Limitations

- **Two data windows.** Scripts 1 and 2 use a rolling 5-year window (`PERIOD = "5y"`), so their results change depending on the run date. Scripts 3–5 use a fixed start of 1 January 2019, with training ending 31 December 2023. The figures and CSVs here are from runs in October 2026.
- **Survivorship bias.** The stock list is today's Nifty 50, applied to the past. Stocks that left the index in the meantime are not included.
- **Dropped stocks.** Scripts 3–5 keep only stocks with prices from the start of 2019, so recent listings and demergers such as JIOFIN, ETERNAL (Zomato) and MAXHEALTH are excluded. Script 3 prints the exact list. Scripts 1 and 2 drop stocks with less than 80% of the 5-year history (JIOFIN in the last run).
- **Expected returns** are historical means, which are noisy, and the max-Sharpe portfolio is sensitive to them.
- **Stress assumptions.** The repo pass-through (0.6), bond duration (7) and the −4% / VIX 35 scenario are assumptions, not estimates.
- **Yahoo Finance data** is unofficial and occasionally has gaps or symbol changes. Edit `NIFTY50` in script 1 when the index composition changes.
