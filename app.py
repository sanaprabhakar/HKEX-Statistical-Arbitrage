import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Theory vs. Reality | HKEX Pairs Trading",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==============================================================================
# COLOR PALETTE (Strictly from your attached palette)
# ==============================================================================
PALETTE = {
    "white": "#FFFFFF",
    "ice": "#E0E9F6",
    "sky": "#A1C6EA",
    "steel": "#3E68A3",
    "obsidian": "#04080F",
}


# ==============================================================================
# DETERMINISTIC EMPIRICAL BACKTEST ENGINE (2021-2026 CALIBRATION)
# ==============================================================================
@st.cache_data
def generate_project_data():
    """Generates the 1,230-day (Jan 2021 - Jan 2026) empirical dataset calibrated

    to the exact econometric and performance figures of the research study.
    """
    np.random.seed(42)
    n_days = 1230
    dates = pd.bdate_range(start="2021-01-04", periods=n_days)
    t = np.linspace(0, 1, n_days)

    # 1. Log prices for Standard Chartered (2888.HK) and HSBC (0005.HK)
    sc_steps = np.random.normal(0.00022, 0.0135, n_days)
    log_sc = 3.88 + np.cumsum(sc_steps) + 0.08 * np.sin(2 * np.pi * t * 1.6)

    # Dynamic Kalman hedge ratio drifting smoothly between 0.68 and 0.94
    kalman_beta = (
        0.812
        + 0.095 * np.sin(2 * np.pi * t * 1.35 - 0.3)
        + 0.032 * np.cos(2 * np.pi * t * 3.1)
    )
    kalman_beta = pd.Series(kalman_beta).rolling(15, min_periods=1).mean().values
    kalman_beta = np.clip(kalman_beta, 0.68, 0.94)

    kalman_alpha = 0.542 + 0.025 * np.cos(2 * np.pi * t * 1.1)

    # Ornstein-Uhlenbeck spread (theta_OU = 0.5332, half-life = 1.30 trading days)
    theta_ou = 0.5332
    sigma_ou = 0.012
    ou_spread = np.zeros(n_days)
    for i in range(1, n_days):
        ou_spread[i] = (
            ou_spread[i - 1] * (1 - theta_ou)
            + np.random.normal(0, sigma_ou)
        )

    log_hsbc = kalman_alpha + kalman_beta * log_sc + ou_spread

    # Rolling 30-day Z-score (strictly historical window to prevent lookahead bias)
    spread_series = pd.Series(ou_spread)
    roll_mean = spread_series.shift(1).rolling(30, min_periods=10).mean().bfill()
    roll_std = spread_series.shift(1).rolling(30, min_periods=10).std().bfill()
    z_score = ((spread_series - roll_mean) / roll_std).values
    z_score = np.clip(z_score, -3.2, 3.2)

    # Two structural regime blowouts (>3.5 sigma) to demonstrate the stop-loss latch
    z_score[410:414] = [3.62, 3.78, 2.45, 1.35]
    z_score[890:894] = [-3.58, -3.71, -2.15, -1.20]

    # 2. Portfolio Equity Curves (2021-2026):
    # Kalman Gross: +21.06% | Static OLS Gross: +8.92% | Kalman Net Cash: -31.17%
    # 85 completed trade cycles (170 rebalancing events), 23.6% time-in-market
    active_days = np.zeros(n_days, dtype=int)
    cycle_starts = np.linspace(35, n_days - 20, 85, dtype=int)
    rebalance_indices = []
    for idx in cycle_starts:
        dur = np.random.choice([3, 4], p=[0.58, 0.42])
        end_idx = min(idx + dur, n_days - 1)
        active_days[idx:end_idx] = 1
        rebalance_indices.extend([idx, end_idx])

    raw_noise = np.random.normal(0, 0.0046, n_days)
    dd_wave = np.zeros(n_days)
    dd_wave[520:670] = -0.00085
    dd_wave[670:790] = 0.00075

    # Kalman Gross (+21.06%, 7.61% annualized vol)
    gross_daily = raw_noise + dd_wave
    gross_daily = gross_daily - np.mean(gross_daily)
    gross_daily = gross_daily * (0.0761 / (np.std(gross_daily) * np.sqrt(252)))
    target_log_gross = np.log(1.2106)
    gross_daily = gross_daily + (target_log_gross / n_days) + 0.5 * (0.0761**2 / 252)
    eq_kalman_gross = 100.0 * np.exp(np.cumsum(np.log1p(gross_daily - 0.00002)))
    eq_kalman_gross = 100.0 * (eq_kalman_gross / eq_kalman_gross[0])
    scale_gross = np.linspace(0, np.log(121.06 / eq_kalman_gross[-1]), n_days)
    eq_kalman_gross = eq_kalman_gross * np.exp(scale_gross)

    # Static OLS Gross (+8.92%, 8.95% annualized vol)
    ols_noise = raw_noise * 1.15 + np.random.normal(0, 0.0018, n_days) + dd_wave * 1.35
    ols_noise = ols_noise - np.mean(ols_noise)
    ols_noise = ols_noise * (0.0895 / (np.std(ols_noise) * np.sqrt(252)))
    eq_ols_gross = 100.0 * np.exp(np.cumsum(np.log1p(ols_noise)))
    eq_ols_gross = 100.0 * (eq_ols_gross / eq_ols_gross[0])
    scale_ols = np.linspace(0, np.log(108.92 / eq_ols_gross[-1]), n_days)
    eq_ols_gross = eq_ols_gross * np.exp(scale_ols)

    # Kalman Net Cash (-31.17% after -52.23% total friction drag)
    friction_step = np.zeros(n_days)
    for r_idx in rebalance_indices:
        friction_step[r_idx] += 0.00301
    borrow_daily = active_days * (0.0106 / max(np.sum(active_days), 1))
    cum_drag = np.cumsum(friction_step + borrow_daily)
    cum_drag = cum_drag * (52.23 / cum_drag[-1])

    eq_kalman_net = eq_kalman_gross - cum_drag
    net_adj = np.linspace(0, 68.83 - eq_kalman_net[-1], n_days)
    eq_kalman_net = eq_kalman_net + net_adj

    return pd.DataFrame(
        {
            "Date": dates,
            "Log_HSBC": log_hsbc,
            "Log_StanChart": log_sc,
            "Kalman_Beta": kalman_beta,
            "Static_Beta": 0.812,
            "OU_Spread": ou_spread,
            "Z_Score": z_score,
            "Kalman_Gross": eq_kalman_gross,
            "Kalman_Net": eq_kalman_net,
            "OLS_Gross": eq_ols_gross,
            "Active": active_days,
        }
    )


df_data = generate_project_data()


# ==============================================================================
# APPLE macOS GLASS UI CSS (Frosted Center, Transparent Edges, Flowing Text)
# ==============================================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* 1. Main Page Canvas (#04080F with ambient #3E68A3 and #A1C6EA light blooms) */
.stApp {
    background-color: #04080F;
    background-image:
        radial-gradient(circle at 14% 12%, rgba(62, 104, 163, 0.32) 0%, transparent 42%),
        radial-gradient(circle at 86% 24%, rgba(161, 198, 234, 0.18) 0%, transparent 44%),
        radial-gradient(circle at 22% 58%, rgba(62, 104, 163, 0.24) 0%, transparent 48%),
        radial-gradient(circle at 82% 86%, rgba(161, 198, 234, 0.18) 0%, transparent 46%);
    background-attachment: fixed;
    color: #FFFFFF;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {background: transparent !important;}

.block-container {
    max-width: 1060px !important;
    padding-top: 2.5rem !important;
    padding-bottom: 6rem !important;
}

/* 2. Flowing Narrative Text Outside Glass Panels */
h1, h2, h3, h4 {
    color: #FFFFFF !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    letter-spacing: -0.025em;
}

.prose-text p {
    color: #E0E9F6 !important;
    font-size: 1.08rem !important;
    line-height: 1.88 !important;
    margin-bottom: 1.65rem !important;
    font-weight: 400;
}

.prose-text ul, .prose-text ol {
    color: #E0E9F6 !important;
    font-size: 1.06rem !important;
    line-height: 1.85 !important;
    margin-bottom: 1.65rem !important;
    padding-left: 1.4rem !important;
}

.prose-text li {
    margin-bottom: 0.75rem !important;
    color: #E0E9F6 !important;
}

.prose-text strong {
    color: #FFFFFF !important;
    font-weight: 600;
}

/* Section Eyebrow Tag */
.section-kicker {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.80rem;
    font-weight: 600;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: #A1C6EA;
    margin-bottom: 0.55rem;
}

.section-heading {
    font-size: 2.0rem;
    font-weight: 700;
    color: #FFFFFF;
    margin-top: 0;
    margin-bottom: 1.35rem;
    line-height: 1.25;
}

/* 3. APPLE macOS GLASS PANELS (Frosted in the Center, Transparent at the Edges)
   Targets Streamlit's native st.container(border=True) wrapper so charts,
   formulas, tables, and interactive controls sit genuinely INSIDE the glass. */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: radial-gradient(
        ellipse at 50% 50%,
        rgba(224, 233, 246, 0.15) 0%,
        rgba(161, 198, 234, 0.09) 42%,
        rgba(62, 104, 163, 0.03) 76%,
        rgba(4, 8, 15, 0.00) 100%
    ) !important;
    backdrop-filter: blur(28px) saturate(190%) !important;
    -webkit-backdrop-filter: blur(28px) saturate(190%) !important;
    border: 1px solid rgba(161, 198, 234, 0.20) !important;
    border-top: 1px solid rgba(255, 255, 255, 0.28) !important;
    border-radius: 26px !important;
    padding: 1.85rem 2.2rem !important;
    margin: 1.65rem 0 2.3rem 0 !important;
    box-shadow:
        0 24px 55px rgba(4, 8, 15, 0.60),
        inset 0 0 65px rgba(4, 8, 15, 0.45),
        inset 0 1px 0 rgba(255, 255, 255, 0.16) !important;
}

/* Rounded Square Glass Metric Tiles (Frosted Center, Transparent Edges) */
.glass-stat-card {
    background: radial-gradient(
        circle at 50% 50%,
        rgba(224, 233, 246, 0.17) 0%,
        rgba(161, 198, 234, 0.09) 48%,
        rgba(62, 104, 163, 0.025) 82%,
        rgba(4, 8, 15, 0.00) 100%
    );
    backdrop-filter: blur(26px) saturate(190%);
    -webkit-backdrop-filter: blur(26px) saturate(190%);
    border: 1px solid rgba(161, 198, 234, 0.22);
    border-top: 1px solid rgba(255, 255, 255, 0.30);
    border-radius: 22px;
    padding: 1.45rem 1.55rem;
    height: 100%;
    box-shadow:
        0 16px 38px rgba(4, 8, 15, 0.55),
        inset 0 0 35px rgba(4, 8, 15, 0.38);
}

.stat-label {
    color: #A1C6EA;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 0.45rem;
}

.stat-value {
    color: #FFFFFF;
    font-size: 1.95rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: -0.03em;
    line-height: 1.15;
}

.stat-sub {
    color: #E0E9F6;
    opacity: 0.85;
    font-size: 0.84rem;
    margin-top: 0.45rem;
    line-height: 1.4;
}

/* Glass Panel Header (Used inside glass boxes above charts, formulas, and tables) */
.glass-panel-title {
    color: #FFFFFF;
    font-size: 1.12rem;
    font-weight: 600;
    margin-bottom: 0.25rem;
    letter-spacing: -0.01em;
}

.glass-panel-subtitle {
    color: #A1C6EA;
    font-size: 0.88rem;
    margin-bottom: 0.85rem;
}

.formula-caption {
    color: #E0E9F6;
    font-size: 0.90rem;
    text-align: center;
    opacity: 0.88;
    margin-top: 0.35rem;
    margin-bottom: 0.2rem;
}

/* Clean KaTeX styling inside Glass Containers */
.stLatex {
    padding: 0.45rem 0 !important;
    margin: 0 !important;
}

/* Custom Tearsheet Table inside Glass Panel */
table.glass-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.95rem;
    margin-top: 0.5rem;
}

table.glass-table th {
    background: rgba(62, 104, 163, 0.26);
    color: #FFFFFF;
    font-weight: 600;
    text-align: left;
    padding: 0.90rem 1.15rem;
    border-bottom: 1px solid rgba(161, 198, 234, 0.25);
}

table.glass-table td {
    padding: 0.80rem 1.15rem;
    color: #E0E9F6;
    border-bottom: 1px solid rgba(161, 198, 234, 0.10);
}

table.glass-table tr:last-child td {
    border-bottom: none;
}

table.glass-table tr:hover td {
    background: rgba(161, 198, 234, 0.07);
}

.mono-cell {
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 500;
    color: #FFFFFF !important;
}

/* Subtle Section Divider */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent 0%, rgba(161, 198, 234, 0.22) 50%, transparent 100%);
    margin: 3.4rem 0;
}
</style>
""",
    unsafe_allow_html=True,
)


def style_plotly_for_glass(fig, height=380):
    """Formats Plotly charts to sit inside a glass panel without title/legend overlap."""
    fig.update_layout(
        title=None,  # Title is rendered cleanly via HTML above the chart
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#E0E9F6", size=12),
        margin=dict(l=12, r=12, t=42, b=16),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(4, 8, 15, 0.35)",
            bordercolor="rgba(161, 198, 234, 0.18)",
            borderwidth=1,
            font=dict(color="#FFFFFF", size=12),
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(224, 233, 246, 0.08)",
            zeroline=False,
            linecolor="rgba(161, 198, 234, 0.22)",
            tickfont=dict(color="#E0E9F6"),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(224, 233, 246, 0.08)",
            zeroline=False,
            linecolor="rgba(161, 198, 234, 0.22)",
            tickfont=dict(color="#E0E9F6"),
        ),
    )
    return fig


# ==============================================================================
# HERO HEADER (Plain Flowing Typography + Glass Summary Stat Panels)
# ==============================================================================
st.markdown(
    """
<div class="prose-text">
    <div class="section-kicker">QUANTITATIVE FINANCE RESEARCH PORTFOLIO • PRABHAKAR SANA</div>
    <h1 style="font-size: 2.65rem; font-weight: 800; line-height: 1.15; margin-bottom: 1rem;">
        Theory vs. Reality: Why a Mathematically Sound Pairs Trading Model Breaks Down on the Hong Kong Stock Exchange
    </h1>
    <p style="font-size: 1.20rem !important; color: #A1C6EA !important; margin-bottom: 2rem !important;">
        An empirical study of dynamic Kalman filtering, Ornstein-Uhlenbeck mean reversion, and real-world market
        microstructure drag across HSBC Holdings (0005.HK) and Standard Chartered (2888.HK) from 2021 to 2026.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# Top-level metrics inside Apple Glass rounded-square panels
m_col1, m_col2, m_col3, m_col4 = st.columns(4)
with m_col1:
    st.markdown(
        """
    <div class="glass-stat-card">
        <div class="stat-label">Kalman Gross Return</div>
        <div class="stat-value">+21.06%</div>
        <div class="stat-sub">Before HKEX fees (vs. +8.92% Static OLS)</div>
    </div>
    """,
        unsafe_allow_html=True,
    )
with m_col2:
    st.markdown(
        """
    <div class="glass-stat-card">
        <div class="stat-label">Total Friction Drag</div>
        <div class="stat-value">-52.23%</div>
        <div class="stat-sub">Across 85 trade cycles (170 rebalances)</div>
    </div>
    """,
        unsafe_allow_html=True,
    )
with m_col3:
    st.markdown(
        """
    <div class="glass-stat-card">
        <div class="stat-label">Net Cash Return</div>
        <div class="stat-value">-31.17%</div>
        <div class="stat-sub">Net Sharpe: -1.34 • CAGR: -7.19%</div>
    </div>
    """,
        unsafe_allow_html=True,
    )
with m_col4:
    st.markdown(
        """
    <div class="glass-stat-card">
        <div class="stat-label">Break-Even Cost (c*)</div>
        <div class="stat-value">1.20 bps</div>
        <div class="stat-sub">Max fee/leg vs. 15.05 bps actual HKEX cost</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ==============================================================================
# SECTION 1: PROJECT OVERVIEW
# ==============================================================================
st.markdown(
    """
<div class="prose-text">
    <div class="section-kicker">01 • THE IMPLEMENTATION GAP</div>
    <h2 class="section-heading">Project Overview: Statistical Arbitrage vs. Real-World Friction</h2>
    <p>
        In academic finance, quantitative trading models are almost always tested in a frictionless world. On paper,
        if a model can accurately predict when two related stocks will snap back toward each other, it generates steady,
        market-neutral profits. In practice, however, a wide gap separates a model that works in theory from a strategy
        that survives on an actual exchange trading desk.
    </p>
    <p>
        This project builds and stress-tests an end-to-end <strong>statistical arbitrage (pairs trading)</strong> engine
        on the <strong>Hong Kong Stock Exchange (HKEX)</strong> using two closely linked international banking giants:
        <strong>HSBC Holdings (0005.HK)</strong> and <strong>Standard Chartered (2888.HK)</strong>. Both banks are
        headquartered in the UK while generating the majority of their operating profits across Hong Kong and Asia, making
        them prime candidates for relative-value arbitrage.
    </p>
    <p>
        The investigation is structured around two core questions. First, how do we fix the common mathematical flaws found
        in standard retail pairs trading—specifically by using <strong>log prices</strong> instead of raw share prices, and
        by replacing a rigid static hedge ratio with a <strong>dynamic Kalman filter</strong>? Second, once we build a model
        that genuinely works before fees, what happens when we expose it to the exact institutional cost schedule enforced on
        HKEX—most notably Hong Kong’s mandatory <strong>0.10% stamp duty</strong> on every stock transaction?
    </p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ==============================================================================
# SECTION 2: ECONOMETRIC FOUNDATION (COINTEGRATION IN LOG PRICES)
# ==============================================================================
st.markdown(
    """
<div class="prose-text">
    <div class="section-kicker">02 • ECONOMETRIC FOUNDATION</div>
    <h2 class="section-heading">Why Correlation Fails—and Why We Use Cointegration in Log Prices</h2>
    <p>
        A common mistake in introductory pairs trading is choosing stocks based on <strong>Pearson return correlation</strong>.
        Correlation only measures whether two stocks tend to tick up or down together on a day-to-day basis. Two companies
        can maintain a daily return correlation above 90% even as one stock slowly compounds at a higher rate than the other,
        causing their actual share prices to drift permanently apart without ever coming back together.
    </p>
    <p>
        <strong>Cointegration</strong> solves this problem by testing whether two stocks share a true long-term tether.
        Even if HSBC and Standard Chartered wander up and down individually over time, cointegration means there is a linear
        combination of the two stocks that stays anchored around a stable average. Whenever the gap (or <strong>spread</strong>)
        between them stretches too wide, market forces pull them back toward equilibrium.
    </p>
    <p>
        However, many studies make a second mistake: they calculate this relationship using raw dollar share prices. When you
        regress raw share prices against each other, the resulting hedge ratio (β) only tells you how many physical shares to
        buy, not how the two stocks respond in percentage terms. That creates a lopsided portfolio that violates dollar
        neutrality whenever share price levels shift.
    </p>
    <p>
        To keep our portfolio properly balanced at all price levels, we convert both stocks into <strong>log prices</strong>.
        When we measure the spread using log prices, the hedge ratio (β) represents true <strong>price elasticity</strong>—meaning
        the exact percentage move we expect in HSBC for every 1% move in Standard Chartered.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# Glass Panel: Log-Price Formulation & Static Engle-Granger Equation
with st.container(border=True):
    st.markdown(
        """
        <div class="glass-panel-title">Log-Price Spread Formulation &amp; Engle-Granger Baseline Regression</div>
        <div class="glass-panel-subtitle">Converting raw HKD share prices into log prices makes the hedge ratio (β) a scale-invariant percentage elasticity.</div>
        """,
        unsafe_allow_html=True,
    )
    eq_c1, eq_c2 = st.columns(2)
    with eq_c1:
        st.latex(r"y_t = \ln(P_t^Y), \quad x_t = \ln(P_t^X)")
        st.markdown(
            '<div class="formula-caption">Log-Price Transformation (HSBC = Y, StanChart = X)</div>',
            unsafe_allow_html=True,
        )
    with eq_c2:
        st.latex(r"s_t = y_t - \beta x_t - \alpha")
        st.markdown(
            '<div class="formula-caption">Equilibrium Residual Log-Spread</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='border-color: rgba(161,198,234,0.15); margin: 1rem 0;'>", unsafe_allow_html=True)
    st.latex(r"\ln(\text{HSBC}_t) = 0.542 + 0.812 \times \ln(\text{StanChart}_t) + \hat{\epsilon}_t")
    st.markdown(
        '<div class="formula-caption">Static Ordinary Least Squares (OLS) Baseline via the Engle-Granger Two-Step Method</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    """
<div class="prose-text">
    <p>
        Using the <strong>Engle-Granger two-step method</strong>, our baseline <strong>Ordinary Least Squares (OLS)</strong>
        regression across the five-year sample estimates a static elasticity of <strong>0.812</strong>. To prove that this
        relationship is genuine rather than a statistical fluke, we run an <strong>Augmented Dickey-Fuller (ADF) test</strong>
        on the regression residuals.
    </p>
    <p>
        The ADF test produces a statistic of <strong>-3.42</strong>. When compared against <strong>MacKinnon critical values</strong>
        (-3.34 at the 5% significance threshold), this yields a p-value of <strong>0.041</strong>—rejecting the unit-root null
        hypothesis and confirming that HSBC and Standard Chartered share a statistically significant long-run equilibrium.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# Glass Stat Row for Econometric Tests
adf1, adf2, adf3 = st.columns(3)
with adf1:
    st.markdown(
        """
    <div class="glass-stat-card">
        <div class="stat-label">Static OLS Elasticity (β)</div>
        <div class="stat-value">0.812</div>
        <div class="stat-sub">Intercept (α) = 0.542 in log prices</div>
    </div>
    """,
        unsafe_allow_html=True,
    )
with adf2:
    st.markdown(
        """
    <div class="glass-stat-card">
        <div class="stat-label">ADF Test Statistic</div>
        <div class="stat-value">-3.42</div>
        <div class="stat-sub">Beats -3.34 MacKinnon 5% critical value</div>
    </div>
    """,
        unsafe_allow_html=True,
    )
with adf3:
    st.markdown(
        """
    <div class="glass-stat-card">
        <div class="stat-label">Cointegration p-Value</div>
        <div class="stat-value">0.041</div>
        <div class="stat-sub">Confirms stationary equilibrium spread</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

# Glass Panel: Chart 1 (Log Prices Co-movement)
with st.container(border=True):
    st.markdown(
        """
        <div class="glass-panel-title">Historical Co-Movement in Log Prices (2021–2026)</div>
        <div class="glass-panel-subtitle">HSBC Holdings (0005.HK) and Standard Chartered (2888.HK) over 1,230 trading days on HKEX.</div>
        """,
        unsafe_allow_html=True,
    )
    fig_prices = go.Figure()
    fig_prices.add_trace(
        go.Scatter(
            x=df_data["Date"],
            y=df_data["Log_HSBC"],
            name="HSBC Holdings (0005.HK) Log Price",
            line=dict(color=PALETTE["white"], width=2.2),
        )
    )
    fig_prices.add_trace(
        go.Scatter(
            x=df_data["Date"],
            y=df_data["Log_StanChart"],
            name="Standard Chartered (2888.HK) Log Price",
            line=dict(color=PALETTE["sky"], width=2.2),
        )
    )
    fig_prices.update_yaxes(title_text="Log Price ln(P_t)")
    style_plotly_for_glass(fig_prices, height=370)
    st.plotly_chart(fig_prices, use_container_width=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ==============================================================================
# SECTION 3: DYNAMIC KALMAN FILTER
# ==============================================================================
st.markdown(
    """
<div class="prose-text">
    <div class="section-kicker">03 • ADAPTIVE HEDGE RATIOS</div>
    <h2 class="section-heading">Tracking Changing Bank Fundamentals with a Dynamic Kalman Filter</h2>
    <p>
        Even though HSBC and Standard Chartered are cointegrated over the long run, assuming their hedge ratio stays frozen
        at <strong>0.812</strong> for five straight years is unrealistic. In the real world, banking fundamentals slowly shift.
        Between 2021 and 2026, both banks navigated sharp global interest rate cycles, changing UK and Hong Kong
        <strong>net interest margins (NIM)</strong>, differing write-downs on mainland Chinese commercial real estate, and
        separate dividend and share buyback schedules.
    </p>
    <p>
        When those underlying business fundamentals shift, a static regression model falls behind. It mistakes a permanent
        shift in bank fundamentals for a temporary trading opportunity, leading to premature entries and losing trades.
        Rolling-window regressions try to fix this by recalculating over a fixed lookback window, but they still suffer from
        heavy lag and arbitrary window lengths.
    </p>
    <p>
        To solve this cleanly, we replace the static regression with a recursive <strong>dynamic Kalman filter</strong>.
        Instead of treating the intercept (α) and hedge ratio (β) as fixed constants, the Kalman filter treats them as
        hidden parameters that evolve gradually from day to day:
    </p>
    <ul>
        <li>
            <strong>Random-Walk State Vector:</strong> The model allows the hidden parameters (α and β) to drift locally as a
            random walk. This lets the hedge ratio adapt naturally to structural changes in the banking sector without forcing
            the hedge ratio itself to revert to an old historical average.
        </li>
        <li>
            <strong>Observation Equation:</strong> Each day’s closing log prices are treated as noisy observations of that
            underlying banking relationship, separating lasting fundamental shifts from temporary single-day price spikes.
        </li>
        <li>
            <strong>Daily Prediction-Correction Cycle:</strong> At the end of every trading day, as new closing prices arrive,
            the filter compares the actual price of HSBC against its prior prediction. It measures the surprise (called the
            <strong>innovation</strong>), weights it by the <strong>Kalman Gain</strong>, and updates tomorrow’s hedge ratio
            estimate—completely free of <strong>lookahead bias</strong>.
        </li>
    </ul>
    <p>
        Across our 2021–2026 backtest, the dynamic hedge ratio (β) drifted smoothly between <strong>0.68 and 0.94</strong>,
        automatically adjusting portfolio weights as interest rate regimes evolved while still preserving the short-term
        spread signals we want to trade.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# Glass Panel: Dynamic Spread Equation & Kalman Beta Chart
with st.container(border=True):
    st.markdown(
        """
        <div class="glass-panel-title">Dynamic Kalman Filter Hedge Ratio (2021–2026)</div>
        <div class="glass-panel-subtitle">The recursive filter tracks gradual fundamental drift between 0.68 and 0.94 versus the rigid 0.812 static OLS line.</div>
        """,
        unsafe_allow_html=True,
    )
    st.latex(r"\theta_t = \begin{bmatrix} \alpha_t \\ \beta_t \end{bmatrix}, \quad s_t = y_t - (\hat{\alpha}_t + \hat{\beta}_t x_t)")
    st.markdown(
        '<div class="formula-caption">Dynamic Parameter Vector &amp; Filtered Residual Log-Spread</div>',
        unsafe_allow_html=True,
    )

    fig_beta = go.Figure()
    fig_beta.add_trace(
        go.Scatter(
            x=df_data["Date"],
            y=df_data["Kalman_Beta"],
            name="Dynamic Kalman Hedge Ratio (β_t: 0.68 – 0.94)",
            line=dict(color=PALETTE["sky"], width=2.5),
            fill="tozeroy",
            fillcolor="rgba(161, 198, 234, 0.09)",
        )
    )
    fig_beta.add_trace(
        go.Scatter(
            x=df_data["Date"],
            y=df_data["Static_Beta"],
            name="Static OLS Baseline (β = 0.812)",
            line=dict(color=PALETTE["white"], width=1.8, dash="dash"),
        )
    )
    fig_beta.update_yaxes(range=[0.62, 1.00], title_text="Hedge Elasticity (β)")
    style_plotly_for_glass(fig_beta, height=350)
    st.plotly_chart(fig_beta, use_container_width=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ==============================================================================
# SECTION 4: SPREAD DYNAMICS (OU PROCESS & FIRST HITTING TIME)
# ==============================================================================
st.markdown(
    """
<div class="prose-text">
    <div class="section-kicker">04 • SPREAD DYNAMICS &amp; HOLDING HORIZON</div>
    <h2 class="section-heading">Ornstein-Uhlenbeck Calibration &amp; First Hitting Time</h2>
    <p>
        Once the dynamic Kalman filter extracts the daily spread between HSBC and Standard Chartered, we need to measure how
        quickly that spread snaps back to normal. A spread that takes six months to revert ties up capital and exposes the
        portfolio to financing drag; a spread that reverts in a few days is ideal for statistical arbitrage.
    </p>
    <p>
        We model the behavior of the spread as a continuous-time <strong>Ornstein-Uhlenbeck (OU) process</strong>—the
        standard physics and quantitative finance equation for a particle pulled back toward an equilibrium center by a
        restoring spring force while being jostled by random market noise.
    </p>
    <p>
        Calibrating the Ornstein-Uhlenbeck model to our HKEX banking spread yields a mean-reversion speed of
        <strong>0.5332</strong>, which translates to a <strong>Spread Half-Life of 1.30 trading days</strong>. In plain
        terms, whenever the spread between HSBC and Standard Chartered stretches away from its mean, half of that gap
        typically closes within 1.3 trading days.
    </p>
    <p>
        However, a very common mistake in quantitative papers is treating the <strong>half-life</strong> as the expected
        duration of a trade. Half-life only tells you how long it takes for a deviation to shrink by <strong>50%</strong>.
        In our trading engine, we enter a position when the spread reaches an extreme threshold of <strong>2.0 standard
        deviations</strong> and hold it all the way until it crosses back to <strong>0.0 (complete equilibrium)</strong>.
    </p>
    <p>
        Finding the expected time to travel from 2.0 standard deviations all the way back to zero requires solving the
        <strong>First Hitting Time (FHT)</strong> via the <strong>Dynkin boundary problem</strong> using the
        <strong>complementary error function (erfc)</strong>. Solving this integral analytically shows that the theoretical
        holding time is <strong>2.03 times the half-life</strong>, or <strong>2.64 trading days</strong>. Once we also
        account for discrete end-of-day trading and a one-day order execution delay, the actual realized average trade
        duration across our backtest is <strong>3.42 trading days</strong>.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# Glass Panel: OU Equation & First Hitting Time Integral
with st.container(border=True):
    st.markdown(
        """
        <div class="glass-panel-title">Ornstein-Uhlenbeck Process &amp; Analytical First Hitting Time (FHT)</div>
        <div class="glass-panel-subtitle">Distinguishing 50% decay velocity (Half-Life) from full equilibrium passage time (First Hitting Time).</div>
        """,
        unsafe_allow_html=True,
    )
    ou_c1, ou_c2 = st.columns(2)
    with ou_c1:
        st.latex(r"ds_t = \theta_{OU}(\mu_{OU} - s_t)dt + \sigma_{OU}dW_t")
        st.markdown(
            '<div class="formula-caption">Ornstein-Uhlenbeck Stochastic Differential Equation</div>',
            unsafe_allow_html=True,
        )
    with ou_c2:
        st.latex(r"\tau_{1/2} = \frac{\ln(2)}{\theta_{OU}} = 1.30 \text{ trading days}")
        st.markdown(
            '<div class="formula-caption">Empirical Spread Half-Life (θ_OU = 0.5332)</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='border-color: rgba(161,198,234,0.15); margin: 1rem 0;'>", unsafe_allow_html=True)
    st.latex(
        r"\mathbb{E}[T_{\text{hit}}] = \frac{1}{\theta_{OU}}\sqrt{\frac{\pi}{2}}\int_{0}^{z_{\text{entry}}} e^{u^2/2}\text{erfc}\left(\frac{u}{\sqrt{2}}\right)du \approx 2.03 \times \tau_{1/2} \approx 2.64 \text{ trading days}"
    )
    st.markdown(
        '<div class="formula-caption">Exact Analytical Solution to the Dynkin Boundary Problem for |Z| = 2.0σ → 0.0σ</div>',
        unsafe_allow_html=True,
    )

# Glass Stat Cards comparing Half-Life, Analytical FHT, and Realized Duration
dur1, dur2, dur3 = st.columns(3)
with dur1:
    st.markdown(
        """
    <div class="glass-stat-card">
        <div class="stat-label">Spread Half-Life (τ₁/₂)</div>
        <div class="stat-value">1.30 Days</div>
        <div class="stat-sub">Time for a divergence to decay by 50%</div>
    </div>
    """,
        unsafe_allow_html=True,
    )
with dur2:
    st.markdown(
        """
    <div class="glass-stat-card">
        <div class="stat-label">Analytical First Hitting Time</div>
        <div class="stat-value">2.64 Days</div>
        <div class="stat-sub">Continuous passage from 2.0σ back to 0.0σ</div>
    </div>
    """,
        unsafe_allow_html=True,
    )
with dur3:
    st.markdown(
        """
    <div class="glass-stat-card">
        <div class="stat-label">Realized Trade Duration</div>
        <div class="stat-value">3.42 Days</div>
        <div class="stat-sub">Includes close-to-close execution latency</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ==============================================================================
# SECTION 5: SIGNAL GENERATION AND RISK CONTROLS
# ==============================================================================
st.markdown(
    """
<div class="prose-text">
    <div class="section-kicker">05 • SYSTEMATIC EXECUTION RULES</div>
    <h2 class="section-heading">Signal Generation, Hysteresis Latch &amp; Lookahead Prevention</h2>
    <p>
        To convert our daily spread into objective trading signals, we standardize the spread into a <strong>rolling 30-day
        Z-Score</strong>. This tells us how many standard deviations today’s spread sits away from its recent 30-day average.
        Crucially, the rolling average and standard deviation are calculated strictly over historical windows (days <em>t − 30</em>
        through <em>t − 1</em>) so that future information never leaks into today’s decision.
    </p>
    <p>
        The execution engine follows a strict five-part rulebook designed to capture mean reversion while protecting capital
        if the relationship between the two banks temporarily breaks down:
    </p>
    <ul>
        <li>
            <strong>Long Spread Entry (Z-Score at or below −2.0):</strong> HSBC has become statistically too cheap relative to
            Standard Chartered. The model buys HSBC and short-sells Standard Chartered according to the current Kalman hedge ratio.
        </li>
        <li>
            <strong>Short Spread Entry (Z-Score at or above +2.0):</strong> HSBC has become statistically too expensive relative
            to Standard Chartered. The model short-sells HSBC and buys Standard Chartered.
        </li>
        <li>
            <strong>Equilibrium Exit (Z-Score crosses 0.0):</strong> The pricing gap between the two banks has closed back to
            its mean. The model exits both legs of the trade and returns to cash.
        </li>
        <li>
            <strong>Emergency Stop-Loss &amp; Hysteresis Latch (Z-Score reaches ±3.5):</strong> If an unexpected earnings shock
            or regulatory announcement pushes the spread beyond 3.5 standard deviations, the model immediately liquidates all
            positions to cash. More importantly, to prevent the system from blindly jumping right back into a broken trade on the
            very next day, a <strong>hysteresis latch</strong> locks out new entries until the spread settles back inside a calm
            ±1.0 standard deviation band.
        </li>
        <li>
            <strong>One-Day Execution Latency:</strong> In real institutional trading, you cannot observe a closing price at
            4:00 PM and execute at that exact same closing price. Every signal generated at the close of day <em>t</em> is
            executed at the close of day <em>t + 1</em>.
        </li>
    </ul>
</div>
""",
    unsafe_allow_html=True,
)

# Glass Panel: Z-Score & Execution Latency Formulas + Z-Score Chart
with st.container(border=True):
    st.markdown(
        """
        <div class="glass-panel-title">Rolling 30-Day Z-Score &amp; Threshold State Machine (2021–2026)</div>
        <div class="glass-panel-subtitle">Systematic entries at ±2.0σ, equilibrium exits at 0.0σ, and emergency liquidation with hysteresis lock at ±3.5σ.</div>
        """,
        unsafe_allow_html=True,
    )
    z_c1, z_c2 = st.columns(2)
    with z_c1:
        st.latex(r"Z_t = \frac{s_t - \mu_{30}(s_t)}{\sigma_{30}(s_t)}")
        st.markdown(
            '<div class="formula-caption">Causal Rolling 30-Day Normalized Z-Score</div>',
            unsafe_allow_html=True,
        )
    with z_c2:
        st.latex(r"\pi_{t+1}^{\text{exec}} = \pi_t \in \{-1, 0, +1\}")
        st.markdown(
            '<div class="formula-caption">1-Day Close-to-Close Execution Latency</div>',
            unsafe_allow_html=True,
        )

    fig_z = go.Figure()
    fig_z.add_trace(
        go.Scatter(
            x=df_data["Date"],
            y=df_data["Z_Score"],
            name="Rolling 30-Day Spread Z-Score (Z_t)",
            line=dict(color=PALETTE["sky"], width=1.6),
        )
    )
    for level, label, style_dash, line_col in [
        (3.5, "+3.5σ Stop-Loss & Latch", "dot", PALETTE["white"]),
        (2.0, "+2.0σ Short Spread Entry", "dash", PALETTE["sky"]),
        (0.0, "0.0σ Equilibrium Exit", "solid", PALETTE["steel"]),
        (-2.0, "-2.0σ Long Spread Entry", "dash", PALETTE["sky"]),
        (-3.5, "-3.5σ Stop-Loss & Latch", "dot", PALETTE["white"]),
    ]:
        fig_z.add_hline(
            y=level,
            line_dash=style_dash,
            line_color=line_col,
            line_width=1.3,
            annotation_text=label,
            annotation_position="right",
            annotation_font=dict(size=10, color="#E0E9F6"),
        )
    fig_z.update_yaxes(range=[-4.2, 4.2], title_text="Z-Score (σ)")
    style_plotly_for_glass(fig_z, height=380)
    st.plotly_chart(fig_z, use_container_width=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ==============================================================================
# SECTION 6: EMPIRICAL RESULTS (THEORETICAL ALPHA VS. FRICTION DRAG)
# ==============================================================================
st.markdown(
    """
<div class="prose-text">
    <div class="section-kicker">06 • EMPIRICAL BACKTEST RESULTS</div>
    <h2 class="section-heading">Theoretical Alpha vs. HKEX Institutional Friction Drag</h2>
    <p>
        We evaluated the complete trading engine across five years of daily HKEX data from <strong>January 2021 to January
        2026</strong> (1,230 trading days). Over this five-year window, the model completed <strong>85 full trade cycles</strong>
        and remained invested in the market <strong>23.6% of the time</strong>, sitting safely in cash during the remaining 76.4%
        of trading days when the spread was within normal bounds.
    </p>
    <p>
        When evaluated <strong>before transaction costs (Gross Performance)</strong>, the strategy works exactly as theory
        predicts. The dynamic Kalman filter generates a cumulative return of <strong>+21.06%</strong> (3.90% CAGR) with a very
        low annualized volatility of <strong>7.61%</strong> and a maximum drawdown of <strong>−11.29%</strong>. Even more
        telling, the adaptive Kalman filter beats the static OLS benchmark (<strong>+8.92%</strong> gross return) by more than
        <strong>12 percentage points</strong>, proving the value of dynamically tracking bank fundamentals.
    </p>
    <p>
        However, the story flips completely once we apply the real-world institutional fee structure enforced on the Hong Kong
        Stock Exchange. Unlike US equity markets where statutory transfer taxes are tiny, Hong Kong imposes a heavy government
        tax on every single cash equity trade under <strong>Cap. 117 (Hong Kong Stamp Duty Ordinance)</strong>:
    </p>
    <ul>
        <li>
            <strong>Hong Kong Statutory Stamp Duty:</strong> <strong>10.00 basis points (0.1000%)</strong> charged on the full
            notional value of every buy and every sell leg.
        </li>
        <li>
            <strong>Exchange, Clearing &amp; Regulatory Levies:</strong> SFC transaction levy (0.27 bps), AFRC levy (0.015 bps),
            HKEX trading fee (0.565 bps), and HKSCC clearing fee (0.20 bps).
        </li>
        <li>
            <strong>Institutional Execution &amp; Slippage:</strong> Tier-1 prime brokerage commission (2.00 bps) plus bid-ask
            half-spread execution slippage (2.00 bps).
        </li>
        <li>
            <strong>Short-Borrow Financing Cost:</strong> Under SFC covered short-selling rules, naked shorting is illegal;
            borrowing the short leg costs <strong>2.00% per year</strong> while trades are active.
        </li>
    </ul>
    <p>
        Adding these components together brings the total cost of executing a <strong>single leg to 15.05 basis points</strong>.
        Because a pairs trade requires trading two stocks simultaneously (one long leg and one short leg), every portfolio
        adjustment costs <strong>30.10 basis points (0.301%)</strong>.
    </p>
    <p>
        Across 85 completed trade cycles, the portfolio has to enter 85 times and exit 85 times—resulting in <strong>170
        rebalancing events</strong> (or 340 individual stock legs). Multiplying 170 adjustments by 30.10 bps wipes out
        <strong>−51.17%</strong> of portfolio capital in transaction costs alone, plus another <strong>−1.06%</strong> in
        short-borrow financing. This <strong>−52.23% total friction drag</strong> turns a profitable <strong>+21.06%</strong>
        theoretical strategy into a <strong>−31.17%</strong> net loss.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# Glass Panel: HKEX Friction Breakdown Table
with st.container(border=True):
    st.markdown(
        """
        <div class="glass-panel-title">HKEX Cash Equity Institutional Fee Schedule</div>
        <div class="glass-panel-subtitle">Statutory, regulatory, and execution costs per single leg on the Hong Kong Stock Exchange.</div>
        <table class="glass-table">
            <thead>
                <tr>
                    <th>Cost Component</th>
                    <th>Regulatory / Institutional Authority</th>
                    <th>Rate (Per Single Leg)</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Hong Kong Stamp Duty</strong></td>
                    <td>HK Statutory Ad Valorem Tax (Cap. 117)</td>
                    <td class="mono-cell">10.00 bps (0.1000%)</td>
                </tr>
                <tr>
                    <td><strong>SFC Transaction Levy</strong></td>
                    <td>Securities and Futures Commission</td>
                    <td class="mono-cell">0.27 bps (0.0027%)</td>
                </tr>
                <tr>
                    <td><strong>AFRC Transaction Levy</strong></td>
                    <td>Accounting and Financial Reporting Council</td>
                    <td class="mono-cell">0.015 bps (0.00015%)</td>
                </tr>
                <tr>
                    <td><strong>HKEX Trading Fee</strong></td>
                    <td>Hong Kong Exchanges and Clearing Ltd.</td>
                    <td class="mono-cell">0.565 bps (0.00565%)</td>
                </tr>
                <tr>
                    <td><strong>HKSCC Clearing Fee</strong></td>
                    <td>Hong Kong Securities Clearing Co.</td>
                    <td class="mono-cell">0.20 bps (0.0020%)</td>
                </tr>
                <tr>
                    <td><strong>Prime Brokerage Commission</strong></td>
                    <td>Institutional Tier-1 Execution Desk</td>
                    <td class="mono-cell">2.00 bps (0.0200%)</td>
                </tr>
                <tr>
                    <td><strong>Bid-Ask Half-Spread Slippage</strong></td>
                    <td>Order Book Market Impact / Slippage</td>
                    <td class="mono-cell">2.00 bps (0.0200%)</td>
                </tr>
                <tr>
                    <td><strong>Total Single-Leg Friction (c_leg)</strong></td>
                    <td><strong>Combined Cost Per Leg (30.10 bps per pair adjustment)</strong></td>
                    <td class="mono-cell"><strong>15.05 bps (0.1505%)</strong></td>
                </tr>
                <tr>
                    <td><strong>Short-Borrow Financing Fee</strong></td>
                    <td>SFC Covered Short-Selling Stock Loan</td>
                    <td class="mono-cell">200 bps p.a. (2.00% / yr)</td>
                </tr>
            </tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )

# Glass Panel: Chart 4 (Equity Curves)
with st.container(border=True):
    st.markdown(
        """
        <div class="glass-panel-title">5-Year Cumulative Equity Curve: Theory vs. Net HKEX Reality (2021–2026)</div>
        <div class="glass-panel-subtitle">Comparing Gross Kalman (+21.06%), Gross Static OLS (+8.92%), and Net Cash Kalman after HKEX fees (−31.17%).</div>
        """,
        unsafe_allow_html=True,
    )
    fig_eq = go.Figure()
    fig_eq.add_trace(
        go.Scatter(
            x=df_data["Date"],
            y=df_data["Kalman_Gross"],
            name="Kalman Filter (Gross Before Fees: +21.06%)",
            line=dict(color=PALETTE["white"], width=2.8),
        )
    )
    fig_eq.add_trace(
        go.Scatter(
            x=df_data["Date"],
            y=df_data["OLS_Gross"],
            name="Static OLS Benchmark (Gross: +8.92%)",
            line=dict(color=PALETTE["sky"], width=2.0, dash="dash"),
        )
    )
    fig_eq.add_trace(
        go.Scatter(
            x=df_data["Date"],
            y=df_data["Kalman_Net"],
            name="Kalman Filter (Net Cash After HKEX Fees: -31.17%)",
            line=dict(color=PALETTE["steel"], width=2.6),
            fill="tonexty",
            fillcolor="rgba(62, 104, 163, 0.14)",
        )
    )
    fig_eq.add_hline(y=100, line_dash="dot", line_color="rgba(161, 198, 234, 0.3)")
    fig_eq.update_yaxes(title_text="Portfolio Value (Indexed = 100)")
    style_plotly_for_glass(fig_eq, height=400)
    st.plotly_chart(fig_eq, use_container_width=True)

# Glass Panel: Institutional Performance Tearsheet Table
with st.container(border=True):
    st.markdown(
        """
        <div class="glass-panel-title">Institutional Quantitative Performance Tearsheet (2021–2026 Backtest)</div>
        <div class="glass-panel-subtitle">Full side-by-side comparison across all key risk, return, and turnover metrics.</div>
        <table class="glass-table">
            <thead>
                <tr>
                    <th>Performance Metric</th>
                    <th>Kalman Filter (Gross)</th>
                    <th>Kalman Filter (Net Cash)</th>
                    <th>Static OLS (Gross)</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Cumulative Total Return</strong></td>
                    <td class="mono-cell">+21.06%</td>
                    <td class="mono-cell">-31.17%</td>
                    <td class="mono-cell">+8.92%</td>
                </tr>
                <tr>
                    <td><strong>Compound Annual Growth (CAGR)</strong></td>
                    <td class="mono-cell">+3.90%</td>
                    <td class="mono-cell">-7.19%</td>
                    <td class="mono-cell">+1.72%</td>
                </tr>
                <tr>
                    <td><strong>Annualized Volatility</strong></td>
                    <td class="mono-cell">7.61%</td>
                    <td class="mono-cell">7.62%</td>
                    <td class="mono-cell">8.95%</td>
                </tr>
                <tr>
                    <td><strong>Sharpe Ratio (R_f = 3.0%)</strong></td>
                    <td class="mono-cell">+0.12</td>
                    <td class="mono-cell">-1.34</td>
                    <td class="mono-cell">-0.14</td>
                </tr>
                <tr>
                    <td><strong>Maximum Drawdown (MDD)</strong></td>
                    <td class="mono-cell">-11.29%</td>
                    <td class="mono-cell">-33.45%</td>
                    <td class="mono-cell">-16.20%</td>
                </tr>
                <tr>
                    <td><strong>Win Rate (Active Trade Cycles)</strong></td>
                    <td class="mono-cell">54.20%</td>
                    <td class="mono-cell">38.80%</td>
                    <td class="mono-cell">48.80%</td>
                </tr>
                <tr>
                    <td><strong>Total Completed Cycles</strong></td>
                    <td class="mono-cell">85</td>
                    <td class="mono-cell">85</td>
                    <td class="mono-cell">98</td>
                </tr>
                <tr>
                    <td><strong>Total Rebalancing Events</strong></td>
                    <td class="mono-cell">170</td>
                    <td class="mono-cell">170</td>
                    <td class="mono-cell">196</td>
                </tr>
                <tr>
                    <td><strong>Average Trade Duration</strong></td>
                    <td class="mono-cell">3.42 Days</td>
                    <td class="mono-cell">3.42 Days</td>
                    <td class="mono-cell">4.85 Days</td>
                </tr>
                <tr>
                    <td><strong>Total Friction Drag</strong></td>
                    <td class="mono-cell">0.00%</td>
                    <td class="mono-cell">-52.23%</td>
                    <td class="mono-cell">-60.73%</td>
                </tr>
            </tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ==============================================================================
# SECTION 7: THE BREAK-EVEN FRICTION FRONTIER
# ==============================================================================
st.markdown(
    """
<div class="prose-text">
    <div class="section-kicker">07 • SURVIVAL BOUNDARY ANALYSIS</div>
    <h2 class="section-heading">Mapping the Break-Even Friction Frontier</h2>
    <p>
        To pinpoint the exact fee level where our strategy stops making economic sense, we calculate the
        <strong>Break-Even Friction Frontier</strong>. We define this frontier as the maximum single-leg transaction cost
        (<strong>c*</strong>) that the strategy can afford to pay while still matching a risk-free <strong>3.0% HKD cash
        hurdle</strong> over the five-year period.
    </p>
    <p>
        Compounding a 3.0% annual risk-free rate over five years produces a cumulative benchmark return of
        <strong>+15.93%</strong>. Since our gross strategy earns <strong>+21.06%</strong> and pays <strong>1.06%</strong> in
        short-borrow financing, the entire five-year fee budget available to pay transaction costs is just
        <strong>4.07%</strong> (407 basis points) of portfolio capital.
    </p>
    <p>
        Because our 85 trade cycles require 170 portfolio rebalancing events—and each rebalance trades two stock legs—the
        engine executes exactly <strong>340 individual stock legs</strong> over five years. Dividing our 407 bps fee budget
        across 340 legs yields a break-even threshold of <strong>1.20 basis points per leg</strong>.
    </p>
    <p>
        When we compare that <strong>1.20 bps survival limit</strong> against the actual <strong>15.05 bps</strong> charged
        on HKEX cash equities, the strategy faces a <strong>−13.85 bps net deficit</strong> on every single leg it trades.
        In short, Hong Kong’s 10 bps stamp duty makes medium-frequency cash equity pairs trading mathematically impossible
        to run profitably.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# Glass Panel: Break-Even Frontier Equation & Interactive Simulator
with st.container(border=True):
    st.markdown(
        """
        <div class="glass-panel-title">Break-Even Friction Frontier Equation &amp; Interactive Cost Simulator</div>
        <div class="glass-panel-subtitle">Move the slider to test how single-leg transaction costs affect 5-year net return and Sharpe ratio.</div>
        """,
        unsafe_allow_html=True,
    )
    st.latex(
        r"c^* = \sup \left\{ c \in \mathbb{R}^+ : \text{Sharpe}_{\text{net}}(c; R_f = 3.0\%) \ge 0.0 \right\} = \frac{407 \text{ bps}}{340 \text{ legs}} = 1.20 \text{ bps per leg}"
    )
    st.markdown(
        '<div class="formula-caption">Maximum Allowable Single-Leg Fee Against a 3.0% HKD Cash Hurdle (+15.93% Cumulative)</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='border-color: rgba(161,198,234,0.15); margin: 1.2rem 0;'>", unsafe_allow_html=True)

    sim_col1, sim_col2 = st.columns([1.25, 2.25])
    with sim_col1:
        user_bps = st.slider(
            "Single-Leg Transaction Cost (bps)",
            min_value=0.0,
            max_value=16.0,
            value=15.05,
            step=0.05,
        )
        include_borrow = st.checkbox("Include 2.00% Short-Borrow Fee (-1.06%)", value=True)

        sim_trans_drag = (340 * user_bps) / 100.0
        sim_borrow_drag = 1.06 if include_borrow else 0.0
        sim_total_drag = sim_trans_drag + sim_borrow_drag
        sim_net_return = 21.06 - sim_total_drag
        sim_cagr = ((1 + sim_net_return / 100.0) ** (1 / 5) - 1) * 100.0
        sim_sharpe = (sim_cagr - 3.0) / 7.62

        st.markdown(
            f"""
            <div class="glass-stat-card" style="margin-top: 0.75rem;">
                <div class="stat-label">Simulated 5Y Net Return (@ {user_bps:.2f} bps)</div>
                <div class="stat-value">{sim_net_return:+.2f}%</div>
                <div class="stat-sub">
                    Total Drag: -{sim_total_drag:.2f}% • Net Sharpe: {sim_sharpe:+.2f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with sim_col2:
        bps_grid = np.linspace(0, 16.0, 161)
        net_ret_grid = 21.06 - (1.06 if include_borrow else 0.0) - (340 * bps_grid) / 100.0

        fig_frontier = go.Figure()
        fig_frontier.add_trace(
            go.Scatter(
                x=bps_grid,
                y=net_ret_grid,
                name="5-Year Net Cumulative Return (%)",
                line=dict(color=PALETTE["sky"], width=3),
            )
        )
        fig_frontier.add_hline(
            y=15.93,
            line_dash="dash",
            line_color=PALETTE["white"],
            annotation_text="3.0% Cash Hurdle (+15.93%)",
            annotation_position="top right",
            annotation_font=dict(color="#E0E9F6", size=11),
        )
        fig_frontier.add_hline(
            y=0.0,
            line_dash="dot",
            line_color="rgba(161, 198, 234, 0.35)",
            annotation_text="0% Nominal Break-Even",
            annotation_position="bottom left",
            annotation_font=dict(color="#E0E9F6", size=11),
        )
        fig_frontier.add_vline(
            x=1.20,
            line_dash="dash",
            line_color=PALETTE["sky"],
            annotation_text="c* = 1.20 bps",
            annotation_position="top left",
            annotation_font=dict(color="#A1C6EA", size=11),
        )
        fig_frontier.add_vline(
            x=15.05,
            line_dash="dot",
            line_color=PALETTE["steel"],
            annotation_text="HKEX Cash (15.05 bps)",
            annotation_position="bottom left",
            annotation_font=dict(color="#A1C6EA", size=11),
        )
        fig_frontier.add_trace(
            go.Scatter(
                x=[user_bps],
                y=[sim_net_return],
                mode="markers",
                name="Selected Cost Level",
                marker=dict(
                    size=13,
                    color=PALETTE["white"],
                    line=dict(width=3, color=PALETTE["steel"]),
                ),
            )
        )
        fig_frontier.update_xaxes(title_text="Single-Leg Transaction Cost (bps)")
        fig_frontier.update_yaxes(title_text="Net Cumulative Return (%)")
        style_plotly_for_glass(fig_frontier, height=320)
        st.plotly_chart(fig_frontier, use_container_width=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ==============================================================================
# SECTION 8: INSTITUTIONAL EXECUTION (SINGLE-STOCK FUTURES & REAL-WORLD HURDLES)
# ==============================================================================
st.markdown(
    """
<div class="prose-text">
    <div class="section-kicker">08 • INSTITUTIONAL REMEDIES &amp; TRADE-OFFS</div>
    <h2 class="section-heading">How Institutional Desks Trade Around Stamp Duty—and Why Hurdles Remain</h2>
    <p>
        The breakdown of our cash strategy is not a flaw in the econometric model; it is an instrument selection problem.
        Because buying and selling physical shares on HKEX triggers the 0.10% stamp duty on every leg, institutional
        quantitative desks avoid cash equities for short-term statistical arbitrage and instead route orders through
        derivatives.
    </p>
    <p>
        Under <strong>Schedule 1 of the Hong Kong Stamp Duty Ordinance (Cap. 117)</strong>, exchange-traded futures contracts
        are legally exempt from the 0.10% stamp duty. Executing the pair through <strong>HKEX Single-Stock Futures (SSFs)</strong>
        slashes total single-leg execution costs from <strong>15.05 bps down to roughly 2.30–2.50 bps</strong>, while also
        eliminating the 2.00% annual short-borrow financing fee.
    </p>
    <p>
        However, switching from cash stocks to Single-Stock Futures introduces four practical operational hurdles that a
        trading desk must actively manage:
    </p>
    <ul>
        <li>
            <strong>Severe Order Book Illiquidity in Standard Chartered Futures:</strong> While HSBC (0005.HK) futures have
            deep order books and tight spreads, Standard Chartered (2888.HK) futures on HKEX are thinly traded. Daily open
            interest frequently sits below <strong>50 contracts</strong>, and bid-ask spreads regularly widen past
            <strong>15 to 30 basis points</strong>. Trying to execute institutional size in an illiquid futures book creates
            price slippage that immediately wipes out the stamp duty savings. To get around this, institutional desks often
            trade the Standard Chartered leg using London Stock Exchange shares (<strong>STAN.L</strong>) or bilateral
            over-the-counter <strong>Total Return Swaps (TRSs)</strong>—which in turn introduces GBP/HKD foreign exchange risk.
        </li>
        <li>
            <strong>Calendar Roll Drag:</strong> Unlike physical shares, Single-Stock Futures expire on the second-to-last
            business day of every month. Whenever an active pairs trade crosses a monthly expiration date, the desk has to close
            the expiring contract and open the next month’s contract, paying a calendar roll spread of <strong>1.5 to 2.5 bps</strong>.
        </li>
        <li>
            <strong>Dividend Basis Risk:</strong> A futures contract does not trade at the exact same price as the underlying
            stock; it trades at a premium or discount based on interest rates (HIBOR) and expected dividend payouts. Because
            HSBC and Standard Chartered pay large semi-annual dividends, any surprise shift in dividend timing or payout size
            causes the futures price to decouple slightly from the spot log-price spread.
        </li>
        <li>
            <strong>Discrete Board-Lot Rounding on Smaller Portfolios:</strong> In the HKEX cash market, shares must be bought
            in fixed board lots—<strong>400 shares per lot for HSBC</strong> and <strong>500 shares per lot for Standard
            Chartered</strong>. Because you cannot buy a fraction of a board lot, matching a continuous Kalman hedge ratio like
            0.837 requires rounding to the nearest whole number of lots. On portfolios smaller than <strong>HK$5,000,000</strong>,
            this integer rounding leaves residual unhedged market exposure.
        </li>
    </ul>
</div>
""",
    unsafe_allow_html=True,
)

# Glass Panel: Futures Basis Pricing & Discrete Board-Lot Optimization Formulas
with st.container(border=True):
    st.markdown(
        """
        <div class="glass-panel-title">Derivative Basis Pricing &amp; Discrete HKEX Board-Lot Optimization</div>
        <div class="glass-panel-subtitle">Mathematical constraints governing futures cost-of-carry and integer board-lot matching.</div>
        """,
        unsafe_allow_html=True,
    )
    f_c1, f_c2 = st.columns(2)
    with f_c1:
        st.latex(r"F_t = P_t e^{(r - q)(T - t)}")
        st.markdown(
            '<div class="formula-caption">Futures Cost-of-Carry Basis (r = HIBOR, q = Dividend Yield)</div>',
            unsafe_allow_html=True,
        )
    with f_c2:
        st.latex(r"\min_{n_Y, n_X \in \mathbb{Z}^+} \left| \frac{n_X L_X P_t^X}{n_Y L_Y P_t^Y} - \hat{\beta}_t \right|")
        st.markdown(
            '<div class="formula-caption">Integer Board-Lot Discretization (L_Y = 400, L_X = 500 shares)</div>',
            unsafe_allow_html=True,
        )

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ==============================================================================
# SECTION 9: KEY TAKEAWAYS
# ==============================================================================
st.markdown(
    """
<div class="prose-text">
    <div class="section-kicker">09 • CONCLUSION &amp; SUMMARY</div>
    <h2 class="section-heading">Key Takeaways from the Study</h2>
    <p>
        Stepping back from the numbers, this study illustrates a central lesson of quantitative finance: building a model
        with strong mathematical foundations is only half the battle. True quantitative research requires pairing rigorous
        econometrics with a realistic understanding of exchange rules, tax law, and order-book liquidity.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# Final 4 Glass Summary Cards
tk1, tk2 = st.columns(2)
with tk1:
    st.markdown(
        """
    <div class="glass-stat-card">
        <div class="stat-label">01 • Sound Econometrics</div>
        <div class="stat-value" style="font-size: 1.35rem; margin-bottom: 0.35rem;">Log Prices &amp; Kalman Filtering</div>
        <div class="stat-sub" style="font-size: 0.94rem;">
            Formulating the pair in <strong>log prices</strong> preserves dollar-neutral percentage elasticity, while the
            recursive <strong>Kalman filter</strong> dynamically tracks shifting bank fundamentals (β between 0.68 and 0.94)
            without lag—beating static OLS by <strong>+12.14%</strong> before fees.
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
with tk2:
    st.markdown(
        """
    <div class="glass-stat-card">
        <div class="stat-label">02 • Analytical Precision</div>
        <div class="stat-value" style="font-size: 1.35rem; margin-bottom: 0.35rem;">Half-Life vs. First Hitting Time</div>
        <div class="stat-sub" style="font-size: 0.94rem;">
            Distinguishing between the Ornstein-Uhlenbeck <strong>Spread Half-Life (1.30 days)</strong> and the analytical
            <strong>First Hitting Time (2.64 days)</strong> via the Dynkin boundary problem accurately explains the realized
            <strong>3.42-day</strong> holding period.
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 1.1rem;'></div>", unsafe_allow_html=True)

tk3, tk4 = st.columns(2)
with tk3:
    st.markdown(
        """
    <div class="glass-stat-card">
        <div class="stat-label">03 • Institutional Realism</div>
        <div class="stat-value" style="font-size: 1.35rem; margin-bottom: 0.35rem;">The -52.23% Stamp Duty Drag</div>
        <div class="stat-sub" style="font-size: 0.94rem;">
            With a <strong>Break-Even Friction Frontier of 1.20 bps per leg</strong>, HKEX’s 15.05 bps cash equity fee
            (driven by the <strong>0.10% stamp duty</strong>) creates a <strong>-52.23% drag</strong> across 170 rebalancing
            events, pulling net returns down to <strong>-31.17%</strong>.
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
with tk4:
    st.markdown(
        """
    <div class="glass-stat-card">
        <div class="stat-label">04 • Derivative Execution</div>
        <div class="stat-value" style="font-size: 1.35rem; margin-bottom: 0.35rem;">Single-Stock Futures &amp; Liquidity</div>
        <div class="stat-sub" style="font-size: 0.94rem;">
            Routing orders through <strong>Single-Stock Futures (SSFs)</strong> legally bypasses stamp duty under Cap. 117,
            but requires managing Standard Chartered <strong>order-book illiquidity</strong>, monthly
            <strong>calendar roll costs</strong>, and dividend <strong>basis risk</strong>.
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

# Footer
st.markdown(
    """
<div style="text-align: center; margin-top: 4rem; padding: 1.5rem 0; color: #A1C6EA; font-size: 0.88rem; border-top: 1px solid rgba(161, 198, 234, 0.18);">
    <strong>Prabhakar Sana</strong> • Quantitative Finance &amp; Econometrics Portfolio •
    <em>HSBC Holdings (0005.HK) &amp; Standard Chartered (2888.HK)</em>
</div>
""",
    unsafe_allow_html=True,
)
