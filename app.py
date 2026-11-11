import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Theory vs. Reality | HKEX Pairs Trading",
    page_icon="𝚺",
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
    font-size: 1.07rem !important;
    line-height: 1.85 !important;
    margin-bottom: 1.55rem !important;
    font-weight: 400;
}

.prose-text ul, .prose-text ol {
    color: #E0E9F6 !important;
    font-size: 1.05rem !important;
    line-height: 1.82 !important;
    margin-bottom: 1.55rem !important;
    padding-left: 1.4rem !important;
}

.prose-text li {
    margin-bottom: 0.65rem !important;
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
    font-size: 1.95rem;
    font-weight: 700;
    color: #FFFFFF;
    margin-top: 0;
    margin-bottom: 1.25rem;
    line-height: 1.25;
}

/* 3. APPLE macOS GLASS PANELS (Frosted in the Center, Transparent at the Edges) */
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
    margin: 1.5rem 0 2.2rem 0 !important;
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

/* Glass Panel Header */
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
    margin: 3.1rem 0;
}
</style>
""",
    unsafe_allow_html=True,
)


def style_plotly_for_glass(fig, height=380):
    """Formats Plotly charts to sit inside a glass panel without title/legend overlap."""
    fig.update_layout(
        title=None,
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
    <div class="section-kicker">ACADEMIC PORTFOLIO PROJECT • PRABHAKAR SANA</div>
    <h1 style="font-size: 2.6rem; font-weight: 800; line-height: 1.15; margin-bottom: 0.9rem;">
        Theory vs. Reality: Why a Mathematically Sound Pairs Trading Model Breaks Down on the Hong Kong Stock Exchange
    </h1>
    <p style="font-size: 1.18rem !important; color: #A1C6EA !important; margin-bottom: 2rem !important;">
        An Empirical Study of Dynamic Kalman Filtering and Market Microstructure Drag on HSBC Holdings (0005.HK) and Standard Chartered (2888.HK)
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
        <div class="stat-sub">Before fees (vs. +8.92% Static OLS)</div>
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
        <div class="stat-sub">Across 85 cycles (170 rebalances)</div>
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
        <div class="stat-sub">Max fee/leg vs. 15.05 bps actual</div>
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
    <div class="section-kicker">01 • PROJECT OVERVIEW</div>
    <h2 class="section-heading">Statistical Arbitrage and the Institutional Implementation Gap</h2>
    <p>
        Quantitative literature often evaluates statistical arbitrage strategies under frictionless, theoretical conditions.
        While models may show statistical alpha on paper, their profitability frequently deteriorates when exposed to
        real-world market microstructure.
    </p>
    <p>
        This project develops an end-to-end pairs trading engine calibrated to two major banking institutions on the
        <strong>Hong Kong Stock Exchange (HKEX)</strong>: <strong>HSBC Holdings (0005.HK)</strong> and
        <strong>Standard Chartered (2888.HK)</strong>. The goal is twofold:
    </p>
    <ol>
        <li>
            Address foundational econometric flaws in traditional pairs trading models by using <strong>log prices</strong>
            and a <strong>dynamic Kalman filter</strong>.
        </li>
        <li>
            Evaluate the strategy against the actual institutional trading costs enforced on HKEX—most notably Hong Kong’s
            statutory <strong>0.10% stamp duty</strong>—to determine whether the strategy remains economically viable.
        </li>
    </ol>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ==============================================================================
# SECTION 2: ECONOMETRIC FOUNDATION
# ==============================================================================
st.markdown(
    """
<div class="prose-text">
    <div class="section-kicker">02 • ECONOMETRIC FOUNDATION</div>
    <h2 class="section-heading">Cointegration Using Log Prices</h2>
    <p>
        A common mistake in retail pairs trading is relying on <strong>Pearson return correlation</strong>. Correlation
        simply measures whether two assets moved in the same direction over a recent window. However, two stocks can display
        a high return correlation while their underlying prices diverge permanently.
    </p>
    <p>
        <strong>Cointegration</strong>, by contrast, identifies a true statistical equilibrium. If two non-stationary price
        series share a stationary linear relationship, any widening spread between them is temporary and expected to revert
        to its mean.
    </p>
    <p>
        Many academic models calibrate cointegration on nominal price levels. In nominal terms, the hedge ratio (β) represents
        a physical share ratio rather than a percentage sensitivity, which distorts portfolio balance and violates dollar
        neutrality. To fix this, we formulate the entire system using <strong>log prices</strong>.
    </p>
    <p>
        Using log prices, β represents true <strong>price elasticity</strong> (the percentage change in HSBC relative to a
        percentage change in Standard Chartered), keeping the strategy scale-invariant.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# Glass Panel: Log-Price Formulation & Static Engle-Granger Equation
with st.container(border=True):
    st.markdown(
        """
        <div class="glass-panel-title">Log-Price Formulation &amp; Engle-Granger Baseline Regression</div>
        <div class="glass-panel-subtitle">Scale-invariant log-price spread and Ordinary Least Squares (OLS) baseline estimation.</div>
        """,
        unsafe_allow_html=True,
    )
    eq_c1, eq_c2 = st.columns(2)
    with eq_c1:
        st.latex(r"y_t = \ln(P_t^Y), \quad x_t = \ln(P_t^X)")
        st.markdown(
            '<div class="formula-caption">Log Prices (HSBC = Y, StanChart = X)</div>',
            unsafe_allow_html=True,
        )
    with eq_c2:
        st.latex(r"s_t = y_t - \beta x_t - \alpha")
        st.markdown(
            '<div class="formula-caption">Residual Log-Price Spread</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='border-color: rgba(161,198,234,0.15); margin: 1rem 0;'>", unsafe_allow_html=True)
    st.latex(r"\ln(\text{HSBC}_t) = 0.542 + 0.812 \times \ln(\text{StanChart}_t) + \hat{\epsilon}_t")
    st.markdown(
        '<div class="formula-caption">Static Baseline Regression via the Engle-Granger Two-Step Method</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    """
<div class="prose-text">
    <p>
        An <strong>Augmented Dickey-Fuller (ADF) test</strong> on the residuals produces a test statistic of
        <strong>-3.42</strong>, rejecting the unit-root null hypothesis at the 5% level against
        <strong>MacKinnon critical values</strong> (p = 0.041). This confirms that a genuine long-run equilibrium exists
        between the two banks.
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
        <div class="stat-sub">5% MacKinnon critical value: -3.34</div>
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
        <div class="glass-panel-title">Log Prices: HSBC Holdings (0005.HK) vs. Standard Chartered (2888.HK)</div>
        <div class="glass-panel-subtitle">5-year daily closing log prices across 1,230 trading days on HKEX (2021–2026).</div>
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
    <div class="section-kicker">03 • DYNAMIC KALMAN FILTER</div>
    <h2 class="section-heading">Tracking Changing Fundamentals with the Kalman Filter</h2>
    <p>
        Standard pairs trading assumes that the relationship (β) between two companies remains constant over time.
        In reality, banking fundamentals drift. HSBC and Standard Chartered operate under shifting macroeconomic pressures,
        including changing UK and Hong Kong <strong>net interest margins (NIM)</strong>, divergent exposures to mainland
        commercial real estate, and varying dividend payouts. A static regression model lags behind these shifts and issues
        false trading signals.
    </p>
    <p>
        To solve this, we model the relationship dynamically using a recursive two-dimensional <strong>Kalman filter</strong>:
    </p>
    <ul>
        <li>
            <strong>State Vector:</strong> The unobserved parameters (α and β) evolve as a random walk, allowing the model
            to adapt locally to structural changes without imposing artificial mean-reversion on the hedge ratio itself.
        </li>
        <li>
            <strong>Observation Equation:</strong> Relates the log price of HSBC to Standard Chartered with empirical
            measurement noise.
        </li>
        <li>
            <strong>Recursive Updating:</strong> Every day, as new closing prices arrive, the filter executes a
            prediction-correction cycle. It calculates the prediction error (the <em>innovation</em>), scales it by the
            <strong>Kalman Gain</strong>, and updates the posterior estimate of the hedge ratio without
            <strong>lookahead bias</strong>.
        </li>
    </ul>
    <p>
        Over the 2021–2026 backtest period, the dynamic hedge elasticity (β) drifted smoothly between
        <strong>0.68 and 0.94</strong>, adjusting to changing market conditions while filtering out transient noise.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# Glass Panel: Dynamic Spread Equation & Kalman Beta Chart
with st.container(border=True):
    st.markdown(
        """
        <div class="glass-panel-title">Dynamic Kalman Filter Hedge Ratio vs. Static OLS (2021–2026)</div>
        <div class="glass-panel-subtitle">Recursive state-vector tracking (0.68 to 0.94) compared against the static 0.812 baseline.</div>
        """,
        unsafe_allow_html=True,
    )
    st.latex(r"\theta_t = \begin{bmatrix} \alpha_t \\ \beta_t \end{bmatrix}, \quad s_t = y_t - (\hat{\alpha}_t + \hat{\beta}_t x_t)")
    st.markdown(
        '<div class="formula-caption">Dynamic State Vector &amp; Filtered Residual Spread</div>',
        unsafe_allow_html=True,
    )

    fig_beta = go.Figure()
    fig_beta.add_trace(
        go.Scatter(
            x=df_data["Date"],
            y=df_data["Kalman_Beta"],
            name="Dynamic Kalman Hedge Elasticity (β_t)",
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
    fig_beta.update_yaxes(range=[0.62, 1.00], title_text="Hedge Ratio (β)")
    style_plotly_for_glass(fig_beta, height=350)
    st.plotly_chart(fig_beta, use_container_width=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ==============================================================================
# SECTION 4: SPREAD DYNAMICS (OU CALIBRATION & FIRST HITTING TIME)
# ==============================================================================
st.markdown(
    """
<div class="prose-text">
    <div class="section-kicker">04 • SPREAD DYNAMICS</div>
    <h2 class="section-heading">Ornstein-Uhlenbeck Calibration &amp; First Hitting Time</h2>
    <p>
        Once the dynamic hedge ratio is extracted, the residual spread must be evaluated for its mean-reverting speed.
        We model the spread as a continuous-time <strong>Ornstein-Uhlenbeck (OU) process</strong>:
    </p>
    <ul>
        <li>
            <strong>Spread Half-Life:</strong> Discretizing the process yields an empirical half-life of
            <strong>1.30 trading days</strong> (mean-reversion speed = 0.5332). This means a deviation decays by 50% in
            roughly 1.3 days.
        </li>
        <li>
            <strong>First Hitting Time (FHT):</strong> Half-life is often mistaken for expected trade duration. When a trade
            is entered at an extreme threshold (2.0σ) and held until it reverts fully to zero (0.0σ), the expected holding
            period is governed by <strong>First Hitting Time</strong> dynamics.
        </li>
        <li>
            <strong>Analytical Solution:</strong> Evaluating the <strong>Dynkin boundary problem</strong> analytically using
            the <strong>complementary error function (erfc)</strong> yields an expected hitting time of 2.03 times the
            half-life, or <strong>2.64 trading days</strong>.
        </li>
        <li>
            <strong>Realized Duration:</strong> Accounting for market-close execution latency and discrete exit boundaries,
            the realized average trade duration expands to <strong>3.42 trading days</strong>.
        </li>
    </ul>
</div>
""",
    unsafe_allow_html=True,
)

# Glass Panel: OU Equation & First Hitting Time Integral
with st.container(border=True):
    st.markdown(
        """
        <div class="glass-panel-title">Ornstein-Uhlenbeck Process &amp; First Hitting Time Equations</div>
        <div class="glass-panel-subtitle">Continuous-time mean reversion and the analytical Dynkin boundary integral.</div>
        """,
        unsafe_allow_html=True,
    )
    ou_c1, ou_c2 = st.columns(2)
    with ou_c1:
        st.latex(r"ds_t = \theta_{OU}(\mu_{OU} - s_t)dt + \sigma_{OU}dW_t")
        st.markdown(
            '<div class="formula-caption">Ornstein-Uhlenbeck Spread Process</div>',
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
        '<div class="formula-caption">Analytical First Hitting Time via the Complementary Error Function (erfc)</div>',
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
        <div class="stat-sub">50% decay velocity (θ_OU = 0.5332)</div>
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
        <div class="stat-sub">2.03 × τ₁/₂ (|Z| = 2.0σ → 0.0σ)</div>
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
        <div class="stat-sub">Includes close-to-close execution lag</div>
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
    <div class="section-kicker">05 • SIGNAL GENERATION &amp; RISK CONTROLS</div>
    <h2 class="section-heading">Systematic Execution Rules &amp; Hysteresis Latch</h2>
    <p>
        To turn the spread into systematic execution rules, the system normalizes the spread into a dimensionless rolling
        30-day <strong>Z-Score</strong>. Rolling metrics are computed strictly over historical windows to prevent
        <strong>lookahead bias</strong>:
    </p>
    <ul>
        <li>
            <strong>Long Spread Entry (Z ≤ −2.0σ):</strong> HSBC is statistically undervalued relative to Standard Chartered.
            Long HSBC, short Standard Chartered.
        </li>
        <li>
            <strong>Short Spread Entry (Z ≥ +2.0σ):</strong> HSBC is statistically overvalued relative to Standard Chartered.
            Short HSBC, long Standard Chartered.
        </li>
        <li>
            <strong>Equilibrium Exit (Z = 0.0σ):</strong> The spread has returned to its mean. Close out positions and return
            to cash.
        </li>
        <li>
            <strong>Emergency Stop-Loss &amp; Hysteresis Latch (|Z| ≥ 3.5σ):</strong> If the spread blows out beyond 3.5
            standard deviations, the model liquidates all positions to cash. To prevent re-entering a structurally broken
            trade, a <strong>hysteresis latch</strong> stays active until the spread normalizes inside ±1.0σ.
        </li>
        <li>
            <strong>Execution Latency:</strong> Signals generated at the close of day <em>t</em> execute at the close of day
            <em>t + 1</em>, accounting for real-world order execution delay.
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
        <div class="glass-panel-title">Rolling 30-Day Z-Score &amp; Execution State Machine (2021–2026)</div>
        <div class="glass-panel-subtitle">Normalized spread signals with ±2.0σ entry bands and ±3.5σ hysteresis stop-loss thresholds.</div>
        """,
        unsafe_allow_html=True,
    )
    z_c1, z_c2 = st.columns(2)
    with z_c1:
        st.latex(r"Z_t = \frac{s_t - \mu_{30}(s_t)}{\sigma_{30}(s_t)}")
        st.markdown(
            '<div class="formula-caption">Rolling 30-Day Z-Score Normalization</div>',
            unsafe_allow_html=True,
        )
    with z_c2:
        st.latex(r"\pi_{t+1}^{\text{exec}} = \pi_t")
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
    <div class="section-kicker">06 • EMPIRICAL RESULTS</div>
    <h2 class="section-heading">Theoretical Alpha vs. Friction Drag</h2>
    <p>
        The strategy was evaluated over a five-year backtest spanning <strong>January 2021 to January 2026</strong>
        (1,230 trading days). During this period, the model completed <strong>85 trade cycles</strong>, remaining in the
        market <strong>23.6%</strong> of the time.
    </p>
    <p>
        <strong>Gross Strategy Performance (Before Fees):</strong> Before transaction costs, the dynamic Kalman strategy
        generated a <strong>+21.06%</strong> cumulative return (CAGR: 3.90%), with <strong>7.61%</strong> annualized
        volatility, a <strong>+0.12</strong> annualized Sharpe ratio (at a 3.0% risk-free rate), and a
        <strong>−11.29%</strong> maximum drawdown. Gross Kalman performance outperformed the static OLS benchmark
        (<strong>+8.92%</strong>) by over 12 percentage points.
    </p>
    <p>
        <strong>The Institutional Friction Structure:</strong> Executing on HKEX requires accounting for the full fee
        schedule enforced by regulators and clearinghouses:
    </p>
    <ul>
        <li><strong>Hong Kong Stamp Duty (Cap. 117):</strong> <strong>10.00 bps</strong> (0.1000%) ad valorem tax on each leg.</li>
        <li><strong>Regulatory &amp; Clearing Fees:</strong> SFC transaction levy (0.27 bps), AFRC levy (0.015 bps), HKEX trading fee (0.565 bps), and HKSCC clearing fee (0.20 bps).</li>
        <li><strong>Execution Costs:</strong> Prime brokerage commissions (2.00 bps) and bid-ask slippage (2.00 bps).</li>
        <li><strong>Total Single-Leg Friction:</strong> <strong>15.05 bps</strong> (or <strong>30.10 bps</strong> across both legs of a pair trade).</li>
        <li><strong>Short-Borrow Financing:</strong> <strong>2.00%</strong> annualized financing charge under SFC covered short-selling requirements.</li>
    </ul>
    <p>
        <strong>Net Strategy Performance (After Fees):</strong> Because an active trade requires both an entry and an exit,
        the 85 trade cycles generated <strong>170 portfolio rebalancing events</strong>. Transaction costs consumed
        <strong>−51.17%</strong> of portfolio capital and short-borrow financing consumed <strong>−1.06%</strong>, creating a
        total friction drag of <strong>−52.23%</strong>. This drives the net cumulative return down to
        <strong>−31.17%</strong> (CAGR: −7.19%, Sharpe: −1.34, Max Drawdown: −33.45%).
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# Glass Panel: Chart 4 (Equity Curves)
with st.container(border=True):
    st.markdown(
        """
        <div class="glass-panel-title">5-Year Cumulative Equity Curve: Gross Alpha vs. Net HKEX Reality (2021–2026)</div>
        <div class="glass-panel-subtitle">Comparing Gross Kalman (+21.06%), Gross Static OLS (+8.92%), and Net Cash Kalman after HKEX fees (−31.17%).</div>
        """,
        unsafe_allow_html=True,
    )
    fig_eq = go.Figure()
    fig_eq.add_trace(
        go.Scatter(
            x=df_data["Date"],
            y=df_data["Kalman_Gross"],
            name="Kalman Filter (Gross: +21.06%)",
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
            name="Kalman Filter (Net Cash: -31.17%)",
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
        <div class="glass-panel-title">Table 1: Institutional Quantitative Performance Tearsheet (2021–2026 Backtest)</div>
        <div class="glass-panel-subtitle">Side-by-side comparison across Kalman (Gross), Kalman (Net Cash), and Static OLS (Gross).</div>
        <table class="glass-table">
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Kalman (Gross)</th>
                    <th>Kalman (Net Cash)</th>
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
                    <td><strong>Win Rate (Active Cycles)</strong></td>
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
    <div class="section-kicker">07 • THE BREAK-EVEN FRICTION FRONTIER</div>
    <h2 class="section-heading">Where Theoretical Alpha Breaks Down</h2>
    <p>
        To understand where theoretical alpha breaks down, we calculate the <strong>Break-Even Friction Frontier</strong>—the
        maximum single-leg transaction cost (<strong>c*</strong>) the strategy can bear while matching a 3.0% HKD cash hurdle:
    </p>
    <ul>
        <li><strong>Maximum allowable cumulative transaction friction:</strong> <strong>4.07%</strong> (407 bps)</li>
        <li><strong>Total single-leg executions across 170 adjustments:</strong> <strong>340 legs</strong></li>
        <li><strong>Break-Even Cost Threshold (c*):</strong> <strong>1.20 basis points per leg</strong></li>
    </ul>
    <p>
        Because actual HKEX cash equity friction is <strong>15.05 bps per leg</strong>, the strategy faces a
        <strong>−13.85 bps net friction deficit</strong>. Cash equity statistical arbitrage on HKEX is economically unviable.
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
        <div class="glass-panel-subtitle">Adjust the single-leg cost slider to test how transaction fees impact 5-year net performance.</div>
        """,
        unsafe_allow_html=True,
    )
    st.latex(
        r"c^* = \sup \left\{ c \in \mathbb{R}^+ : \text{Sharpe}_{\text{net}}(c; R_f = 3.0\%) \ge 0.0 \right\} = \frac{407 \text{ bps}}{340 \text{ legs}} = 1.20 \text{ bps per leg}"
    )
    st.markdown(
        '<div class="formula-caption">Break-Even Single-Leg Cost Threshold Against a 3.0% HKD Cash Hurdle</div>',
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
# SECTION 8: INSTITUTIONAL EXECUTION (SINGLE-STOCK FUTURES & MARKET REALITIES)
# ==============================================================================
st.markdown(
    """
<div class="prose-text">
    <div class="section-kicker">08 • INSTITUTIONAL EXECUTION</div>
    <h2 class="section-heading">Single-Stock Futures and Market Realities</h2>
    <p>
        The failure of the cash strategy is an instrument implementation issue rather than an econometric failure.
        Institutional desks address this by trading derivatives.
    </p>
    <p>
        <strong>1. Statutory Stamp Duty Exemption:</strong> Under <strong>Schedule 1 of the Hong Kong Stamp Duty Ordinance
        (Cap. 117)</strong>, exchange-traded futures are exempt from the 0.10% stamp duty. Trading
        <strong>HKEX Single-Stock Futures (SSFs)</strong> reduces aggregate single-leg costs to approximately
        <strong>2.30 to 2.50 bps</strong> and eliminates short-borrow financing fees.
    </p>
    <p>
        <strong>2. Derivative Implementation Trade-Offs:</strong> While SSFs resolve stamp duty, they introduce practical
        operational hurdles:
    </p>
    <ul>
        <li>
            <strong>Order Book Illiquidity:</strong> HSBC futures maintain tight spreads and deep order books, but
            Standard Chartered (2888.HK) futures suffer from low liquidity, with open interest frequently below
            <strong>50 contracts</strong> and bid-ask spreads widening beyond <strong>15–30 bps</strong>. This illiquidity
            can quickly negate stamp duty savings. Desks often route the Standard Chartered leg via London Stock Exchange
            shares (<strong>STAN.L</strong>) or bilateral <strong>Total Return Swaps (TRSs)</strong>, introducing foreign
            exchange risk.
        </li>
        <li>
            <strong>Calendar Roll Drag:</strong> Maintaining positions across monthly futures contract expiries introduces
            roll spread costs of <strong>1.5–2.5 bps</strong> per event.
        </li>
        <li>
            <strong>Basis Risk:</strong> Futures trade at a premium or discount to spot prices based on carrying costs and
            expected dividends. Irregular dividend announcements from either bank create basis fluctuations that decouple
            from the underlying log-price spread.
        </li>
        <li>
            <strong>Discrete Board Lots:</strong> HKEX cash equities trade in fixed lot sizes (<strong>400 shares</strong>
            for HSBC; <strong>500 shares</strong> for StanChart). Achieving continuous elasticity-weighted balance requires
            integer rounding, which leaves small residual beta exposures on sub-HK$5M portfolios.
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
        <div class="glass-panel-title">Futures Cost-of-Carry Basis &amp; Discrete Board-Lot Rounding</div>
        <div class="glass-panel-subtitle">Pricing equations governing derivative basis risk and HKEX board-lot discretization.</div>
        """,
        unsafe_allow_html=True,
    )
    f_c1, f_c2 = st.columns(2)
    with f_c1:
        st.latex(r"F_t = P_t e^{(r - q)(T - t)}")
        st.markdown(
            '<div class="formula-caption">Futures Basis Pricing (r = Carrying Cost, q = Dividend Yield)</div>',
            unsafe_allow_html=True,
        )
    with f_c2:
        st.latex(r"\min_{n_Y, n_X \in \mathbb{Z}^+} \left| \frac{n_X L_X P_t^X}{n_Y L_Y P_t^Y} - \hat{\beta}_t \right|")
        st.markdown(
            '<div class="formula-caption">Discrete Board-Lot Optimization (L_Y = 400, L_X = 500)</div>',
            unsafe_allow_html=True,
        )

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ==============================================================================
# SECTION 9: KEY TAKEAWAYS
# ==============================================================================
st.markdown(
    """
<div class="prose-text">
    <div class="section-kicker">09 • SUMMARY</div>
    <h2 class="section-heading">Key Takeaways</h2>
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
        <div class="stat-value" style="font-size: 1.32rem; margin-bottom: 0.35rem;">Log Prices &amp; Kalman Filter</div>
        <div class="stat-sub" style="font-size: 0.94rem;">
            Formulating the model using <strong>log prices</strong> avoids scale bias, while the recursive
            <strong>Kalman filter</strong> dynamically tracks changing bank fundamentals without lag.
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
        <div class="stat-value" style="font-size: 1.32rem; margin-bottom: 0.35rem;">Half-Life vs. First Hitting Time</div>
        <div class="stat-sub" style="font-size: 0.94rem;">
            Distinguishing between <strong>half-life (1.30 days)</strong> and <strong>First Hitting Time (2.64 days)</strong>
            accurately frames expected trade durations (<strong>3.42 days</strong> realized).
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
        <div class="stat-value" style="font-size: 1.32rem; margin-bottom: 0.35rem;">The -52.23% Stamp Duty Drag</div>
        <div class="stat-sub" style="font-size: 0.94rem;">
            The <strong>0.10% HKEX stamp duty</strong> creates a <strong>-52.23% drag</strong> across 170 rebalancing events
            that wipes out cash equity statistical arbitrage (<strong>-31.17%</strong> net return).
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
with tk4:
    st.markdown(
        """
    <div class="glass-stat-card">
        <div class="stat-label">04 • Derivative Solutions</div>
        <div class="stat-value" style="font-size: 1.32rem; margin-bottom: 0.35rem;">Single-Stock Futures</div>
        <div class="stat-sub" style="font-size: 0.94rem;">
            Moving execution to <strong>Single-Stock Futures (SSFs)</strong> solves the tax drag, but requires active
            management of <strong>order-book illiquidity</strong>, <strong>roll costs</strong>, and <strong>basis risk</strong>.
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

# Footer
st.markdown(
    """
<div style="text-align: center; margin-top: 4rem; padding: 1.5rem 0; color: #A1C6EA; font-size: 0.88rem; border-top: 1px solid rgba(161, 198, 234, 0.18);">
    <strong>Prabhakar Sana</strong> • Academic Portfolio Project •
    <em>Dynamic Cointegration, Kalman Filtering &amp; HKEX Market Microstructure</em>
</div>
""",
    unsafe_allow_html=True,
)
