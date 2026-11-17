# Theory vs. Reality: Why a Mathematically Sound Pairs Trading Model Breaks Down on the Hong Kong Stock Exchange

**An Empirical Study of Dynamic Kalman Filtering and Market Microstructure Drag on HSBC Holdings (`0005.HK`) and Standard Chartered (`2888.HK`)**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3E68A3?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Interactive_App-Streamlit-04080F?style=flat-square&logo=streamlit&logoColor=A1C6EA)](https://share.streamlit.io/)
[![Domain](https://img.shields.io/badge/Quantitative_Finance-Statistical_Arbitrage-3E68A3?style=flat-square)]()
[![Market](https://img.shields.io/badge/Exchange-HKEX-04080F?style=flat-square)]()

> **Interactive Research Dashboard:** Launch the live model and Break-Even Friction Simulator on Streamlit *(https://hkex-statistical-arbitrage.streamlit.app/)*.

---

## Abstract

Quantitative finance literature frequently evaluates statistical arbitrage strategies under frictionless theoretical assumptions. While an econometric model may generate positive alpha on paper, live profitability often collapses when exposed to exchange-specific taxes, clearing fees, and execution latency.

This repository implements an end-to-end pairs trading engine calibrated to two major banking equities on the **Hong Kong Stock Exchange (HKEX)**—**HSBC Holdings (`0005.HK`)** and **Standard Chartered (`2888.HK`)**—across **1,230 trading days (January 2021 – January 2026)**. Replacing static price-level regressions with **scale-invariant log prices** and a **recursive two-dimensional Kalman filter** lifts gross cumulative return from **+8.92%** to **+21.06%**. However, incorporating Hong Kong's statutory **0.10% ad valorem stamp duty** and institutional execution costs imposes a **−52.23% friction drag** across 170 rebalancing events, turning a theoretical gain into a **−31.17% net cash loss**.

---

## Empirical Performance Summary (2021–2026)

| Performance Metric | Kalman Filter (Gross) | Kalman Filter (Net Cash) | Static OLS (Gross) |
| :--- | :---: | :---: | :---: |
| **Cumulative Total Return** | **+21.06%** | **−31.17%** | +8.92% |
| **Compound Annual Growth (CAGR)** | +3.90% | −7.19% | +1.72% |
| **Annualized Volatility** | 7.61% | 7.62% | 8.95% |
| **Sharpe Ratio ($R_f = 3.0$%)** | **+0.12** | **−1.34** | −0.14 |
| **Maximum Drawdown (MDD)** | −11.29% | −33.45% | −16.20% |
| **Win Rate (Active Cycles)** | 54.20% | 38.80% | 48.80% |
| **Completed Trade Cycles** | 85 | 85 | 98 |
| **Portfolio Rebalancing Events** | 170 | 170 | 196 |
| **Average Holding Duration** | 3.42 Days | 3.42 Days | 4.85 Days |
| **Total Friction Drag** | 0.00% | **−52.23%** | −60.73% |

---

## Methodology & Quantitative Architecture

### 1. Scale-Invariant Log-Price Cointegration
To eliminate physical share-ratio distortion and preserve dollar neutrality across price regimes, the equilibrium relationship is formulated in natural log prices ($y_t = \ln P_t^{\text{HSBC}}$, $x_t = \ln P_t^{\text{StanChart}}$):

$$\ln(\text{HSBC}_t) = 0.542 + 0.812 \times \ln(\text{StanChart}_t) + \hat{\epsilon}_t$$

* **Stationarity Verification:** An Augmented Dickey-Fuller (ADF) test on the Engle-Granger residuals yields a test statistic of **−3.42** against the 5% MacKinnon critical value of **−3.34** ($p = 0.041$), confirming a stationary long-run equilibrium.

### 2. Recursive Two-Dimensional Kalman Filter
Static Ordinary Least Squares (OLS) fails to capture structural drift in banking fundamentals (divergent UK/HK net interest margins, mainland commercial real estate exposure, and dividend policy changes). The unobserved intercept and elasticity vector $\theta_t = [\alpha_t, \beta_t]^\top$ is modeled as a latent random walk updated recursively each close without lookahead bias:

$$s_t = y_t - (\hat{\alpha}_t + \hat{\beta}_t x_t)$$

* Over the 5-year sample, the dynamic hedge elasticity $\hat{\beta}_t$ adapts smoothly between **0.68 and 0.94** around the static `0.812` baseline.

### 3. Ornstein-Uhlenbeck Calibration & First Hitting Time
Modeling the filtered residual spread as a continuous-time Ornstein-Uhlenbeck process, $ds_t = \theta_{OU}(\mu_{OU} - s_t) dt + \sigma_{OU} dW_t$, yields a mean-reversion speed of $\theta_{OU} = 0.5332$ and an empirical **half-life of $\tau_{1/2} = 1.30$ trading days**.

Because trades enter at $\vert{}Z\vert{} = 2.0\sigma$ and exit at full reversion ($0.0\sigma$), expected duration is governed by the analytical First Hitting Time (Dynkin boundary problem) rather than half-life alone:

$$\mathbb{E}[T_{\text{hit}}] = \frac{1}{\theta_{OU}}\sqrt{\frac{\pi}{2}}\int_{0}^{z_{\text{entry}}} e^{u^2/2}\text{erfc}\left(\frac{u}{\sqrt{2}}\right)du \approx 2.03 \times \tau_{1/2} \approx 2.64 \text{ trading days}$$

* Incorporating discrete close-to-close execution latency expands the realized average trade duration to **3.42 trading days**.

### 4. Execution State Machine & Hysteresis Risk Control
Signals are normalized using a strictly historical rolling 30-day Z-score ($Z_t$) with a 1-day close-to-close execution lag ($\pi_{t+1}^{\text{exec}} = \pi_t$):

* **Entry Thresholds:** Long spread at $Z_t \le -2.0\sigma$; short spread at $Z_t \ge +2.0\sigma$.
* **Equilibrium Exit:** Liquidate to cash upon crossing $Z_t = 0.0\sigma$.
* **Stop-Loss & Hysteresis Latch:** Emergency exit at $\vert{}Z_t\vert{} \ge 3.5\sigma$, locking out re-entry until the spread re-normalizes inside $\pm 1.0\sigma$.

### 5. HKEX Friction Structure & Break-Even Frontier ($c^*$)
Every round-trip pair trade incurs institutional costs across **340 single-leg executions** (85 cycles $\times$ 2 rebalances $\times$ 2 legs):

* **Hong Kong Stamp Duty (Cap. 117):** `10.00 bps` (`0.1000%`) per leg
* **Regulatory & Clearing Levies:** `1.05 bps` (SFC `0.27`, AFRC `0.015`, HKEX `0.565`, HKSCC `0.20`)
* **Prime Brokerage & Slippage:** `4.00 bps` (`2.00 bps` commission + `2.00 bps` bid-ask slippage)
* **Total Single-Leg Cost:** **`15.05 bps` per leg** (`30.10 bps` per pair rebalance), plus `2.00%` annualized short-borrow financing (`−1.06%` cumulative drag)

Solving for the maximum single-leg fee $c^*$ that matches a `3.0%` HKD cash hurdle ($R_f = 0.03$):

$$c^* = \sup \lbrace c \in \mathbb{R}^+ : \text{Sharpe}_{\text{net}}(c; R_f = 0.03) \ge 0.0 \rbrace = \frac{407 \text{ bps}}{340 \text{ legs}} = 1.20 \text{ bps per leg}$$

Actual HKEX cash equity friction (`15.05 bps`) exceeds this threshold by **13.85 bps per leg**, rendering cash equity statistical arbitrage economically unviable.

### 6. Institutional Derivative Implementation
Under **Schedule 1 of the Hong Kong Stamp Duty Ordinance (Cap. 117)**, exchange-traded futures are exempt from stamp duty, reducing execution costs on **HKEX Single-Stock Futures (SSFs)** to `2.30–2.50 bps/leg`. However, derivative execution introduces four practical trade-offs modeled in the study:
1. **Asymmetric Order-Book Liquidity:** Deep liquidity in HSBC futures versus thin open interest (`< 50 contracts`) and wide spreads (`15–30 bps`) in Standard Chartered futures.
2. **Calendar Roll Drag:** `1.5–2.5 bps` per monthly contract expiry roll.
3. **Dividend Basis Risk:** Irregular bank dividend announcements decoupling futures basis $F_t = P_t e^{(r - q)(T - t)}$ from spot log-spreads.
4. **Discrete Board Lots:** Integer rounding across HKEX lot sizes (`400` shares for `0005.HK`; `500` shares for `2888.HK`) leaving residual beta exposure on portfolios under HK$5M.

---

## Repository Structure

```text
HKEX-Statistical-Arbitrage/
├── app.py              # Interactive Streamlit research application & deterministic backtest engine
├── requirements.txt    # Python dependencies (numpy, pandas, plotly, streamlit)
└── README.md           # Research methodology, mathematical formulation, and empirical tearsheet
```

---



## Author

**Prabhakar Sana**  
*Quantitative Finance & Econometrics Project*
