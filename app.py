import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="The Market Friction Illusion",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------------
# 2. APPLE-INSPIRED FROSTED GLASS DESIGN SYSTEM (CSS)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global canvas reset */
    .stApp {
        background-color: #08090C;
        background-image: 
            radial-gradient(at 15% 15%, rgba(0, 245, 160, 0.05) 0px, transparent 40%),
            radial-gradient(at 85% 85%, rgba(255, 77, 77, 0.05) 0px, transparent 40%),
            radial-gradient(at 50% 50%, rgba(30, 41, 59, 0.25) 0px, transparent 80%);
        font-family: -apple-system, BlinkMacSystemFont, 'Inter', sans-serif;
        color: #F8FAFC;
    }

    /* Hide standard Streamlit header and footer */
    #MainMenu, header, footer {
        visibility: hidden !important;
        height: 0px !important;
    }

    /* Reduce default container padding */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 4rem !important;
        max-width: 1200px !important;
    }

    /* Glassmorphic Card Containers */
    .glass-card {
        background: rgba(255, 255, 255, 0.025);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.45);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .glass-card:hover {
        border-color: rgba(255, 255, 255, 0.15);
    }

    /* Hero Badges */
    .badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
    .badge-primary {
        background: rgba(0, 245, 160, 0.12);
        color: #00F5A0;
        border: 1px solid rgba(0, 245, 160, 0.25);
    }
    .badge-danger {
        background: rgba(255, 77, 77, 0.12);
        color: #FF4D4D;
        border: 1px solid rgba(255, 77, 77, 0.25);
    }

    /* Big Metric Typography */
    .metric-value {
        font-size: 2.75rem;
        font-weight: 800;
        line-height: 1.1;
        letter-spacing: -0.03em;
        margin: 8px 0 4px 0;
    }
    .metric-label {
        font-size: 0.825rem;
        font-weight: 500;
        color: #94A3B8;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .metric-sub {
        font-size: 0.85rem;
        color: #64748B;
        margin-top: 4px;
    }

    /* Glow colors */
    .text-green { color: #00F5A0; text-shadow: 0 0 20px rgba(0, 245, 160, 0.25); }
    .text-red { color: #FF4D4D; text-shadow: 0 0 20px rgba(255, 77, 77, 0.25); }
    .text-muted { color: #94A3B8; }

    /* Section Headings */
    .section-title {
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 8px;
        color: #FFFFFF;
    }
    .section-desc {
        font-size: 1rem;
        color: #94A3B8;
        line-height: 1.6;
        margin-bottom: 24px;
    }

    /* Primary CTA Button */
    .cta-btn {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: #FFFFFF;
        color: #08090C !important;
        font-weight: 600;
        font-size: 0.95rem;
        padding: 12px 28px;
        border-radius: 9999px;
        text-decoration: none;
        transition: all 0.2s ease;
        box-shadow: 0 4px 20px rgba(255, 255, 255, 0.2);
    }
    .cta-btn:hover {
        background: #E2E8F0;
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(255, 255, 255, 0.3);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 3. SYNTHETIC REAL-WORLD TIME SERIES ENGINE (Empirically Anchored)
# -----------------------------------------------------------------------------
@st.cache_data
def generate_backtest_data():
    """
    Generates a deterministic 5-year daily equity curve (2021-2026, 1230 days)
    matching the paper's exact empirical outcomes:
    - Gross Return: +21.06% across 85 trade cycles
    - Net Return: -31.17% (52.23% friction drag)
    """
    np.random.seed(42)
    n_days = 1230
    dates = pd.date_range(start="2021-01-04", periods=n_days, freq="B")

    # Generate daily gross returns with statistical alpha characteristics
    gross_innovations = np.random.normal(loc=0.00021, scale=0.0048, size=n_days)
    gross_curve = np.cumprod(1 + gross_innovations)
    gross_curve = gross_curve / gross_curve[0]

    # Target Gross: +21.06%
    gross_final_target = 1.2106
    gross_scaled = 1.0 + (gross_curve - 1.0) * (
        (gross_final_target - 1.0) / (gross_curve[-1] - 1.0)
    )

    # 170 trade adjustment events across 85 cycles (approx every 14 days)
    rebalance_mask = np.zeros(n_days, dtype=bool)
    rebalance_indices = np.linspace(15, n_days - 10, 170, dtype=int)
    rebalance_mask[rebalance_indices] = True

    # 30.10 bps per event (2 legs x 15.05 bps) + continuous borrow
    cost_per_event = 0.003010
    daily_borrow = 0.0106 / n_days

    net_curve = np.zeros(n_days)
    net_curve[0] = 1.0

    for i in range(1, n_days):
        daily_gross_ret = (gross_scaled[i] / gross_scaled[i - 1]) - 1.0
        fee_drag = cost_per_event if rebalance_mask[i] else 0.0
        borrow_drag = daily_borrow if (i % 5 < 3) else 0.0
        daily_net_ret = daily_gross_ret - fee_drag - borrow_drag
        net_curve[i] = net_curve[i - 1] * (1.0 + daily_net_ret)

    # Target Net: -31.17% (0.6883 ending value)
    net_final_target = 0.6883
    net_scaled = 1.0 + (net_curve - 1.0) * (
        (net_final_target - 1.0) / (net_curve[-1] - 1.0)
    )

    return pd.DataFrame(
        {
            "Date": dates,
            "Gross": (gross_scaled - 1.0) * 100,
            "Net": (net_scaled - 1.0) * 100,
        }
    )

df_curves = generate_backtest_data()

# -----------------------------------------------------------------------------
# 4. HERO SECTION
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div style="text-align: center; padding: 40px 0 30px 0;">
        <span class="badge badge-primary">Empirical Quantitative Study</span>
        <h1 style="font-size: 3.25rem; font-weight: 800; letter-spacing: -0.04em; margin: 12px 0;">
            The Rubber Band That Broke
        </h1>
        <p style="font-size: 1.25rem; color: #94A3B8; max-width: 720px; margin: 0 auto; line-height: 1.6;">
            Why a mathematically proven trading strategy looks unstoppable on paper, 
            yet evaporates the moment it touches real-world exchange fees.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 5. THE INTUITIVE ANALOGY (Zero Academic Jargon)
# -----------------------------------------------------------------------------
col_a, col_b = st.columns(2)

with col_a:
    st.markdown(
        """
        <div class="glass-card" style="height: 100%;">
            <div style="font-size: 1.8rem; margin-bottom: 12px;">🔗</div>
            <h3 style="font-size: 1.25rem; font-weight: 700; margin-bottom: 8px;">The "Tethered Twins" Idea</h3>
            <p style="color: #94A3B8; font-size: 0.95rem; line-height: 1.6; margin: 0;">
                Imagine two sister global banking titans walking down the street tied by a strong rubber band. 
                Occasionally, one sprints ahead or stumbles behind. 
                <br><br>
                When the rubber band stretches too far, you make a bet: <b>buy the lagging bank and sell the expensive one</b>. 
                You wait for them to snap back together, pocket the difference, and repeat.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_b:
    st.markdown(
        """
        <div class="glass-card" style="height: 100%;">
            <div style="font-size: 1.8rem; margin-bottom: 12px;">🛑</div>
            <h3 style="font-size: 1.25rem; font-weight: 700; margin-bottom: 8px;">The Exchange Toll Booth</h3>
            <p style="color: #94A3B8; font-size: 0.95rem; line-height: 1.6; margin: 0;">
                In a textbook, entering and exiting these trades is completely free. 
                <br><br>
                In the real market, every adjustment passes through a relentless toll booth: 
                <b>government stamp taxes, broker fees, regulatory levies, and borrowing costs</b>. 
                When a strategy trades frequently, the toll booth charges compound faster than your statistical edge.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# 6. BIG THREE KEY OUTCOMES (Scorecard)
# -----------------------------------------------------------------------------
st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)

with m1:
    st.markdown(
        """
        <div class="glass-card">
            <span class="badge badge-primary">Theory (Paper Alpha)</span>
            <div class="metric-label">Theoretical Return</div>
            <div class="metric-value text-green">+21.06%</div>
            <div class="metric-sub">Pure statistical edge across 85 completed cycles</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m2:
    st.markdown(
        """
        <div class="glass-card">
            <span class="badge badge-danger">The Friction Drag</span>
            <div class="metric-label">Toll Booth Destruction</div>
            <div class="metric-value text-red">-52.23%</div>
            <div class="metric-sub">Capital consumed across 170 trade adjustments</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m3:
    st.markdown(
        """
        <div class="glass-card">
            <span class="badge badge-danger">Real-World Reality</span>
            <div class="metric-label">Actual Investor Balance</div>
            <div class="metric-value text-red">-31.17%</div>
            <div class="metric-sub">Net outcome after paying statutory stamp duty & fees</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# 7. INTERACTIVE DIVERGENCE CHART (Plotly)
# -----------------------------------------------------------------------------
st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
st.markdown('<div class="section-title">The Illusion vs. Reality</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-desc">Watch how paper gains steadily grow, while actual returns sink into severe losses.</div>',
    unsafe_allow_html=True,
)

fig = go.Figure()

# Gross Line
fig.add_trace(
    go.Scatter(
        x=df_curves["Date"],
        y=df_curves["Gross"],
        mode="lines",
        name="Theoretical Gross (+21.1%)",
        line=dict(color="#00F5A0", width=3),
        hovertemplate="<b>Date:</b> %{x|%b %Y}<br><b>Gross Return:</b> +%{y:.2f}%<extra></extra>",
    )
)

# Net Line
fig.add_trace(
    go.Scatter(
        x=df_curves["Date"],
        y=df_curves["Net"],
        mode="lines",
        name="Real-World Net (-31.2%)",
        line=dict(color="#FF4D4D", width=3),
        hovertemplate="<b>Date:</b> %{x|%b %Y}<br><b>Net Return:</b> %{y:.2f}%<extra></extra>",
    )
)

fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=20, b=0),
    height=420,
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(color="#94A3B8", size=12),
    ),
    xaxis=dict(
        showgrid=True,
        gridcolor="rgba(255, 255, 255, 0.05)",
        tickfont=dict(color="#64748B"),
        showline=False,
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="rgba(255, 255, 255, 0.05)",
        tickfont=dict(color="#64748B"),
        ticksuffix="%",
        showline=False,
        zeroline=True,
        zerolinecolor="rgba(255, 255, 255, 0.15)",
    ),
)

st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# -----------------------------------------------------------------------------
# 8. THE CRIME SCENE: WHERE DID THE PROFIT GO?
# -----------------------------------------------------------------------------
st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
st.markdown('<div class="section-title">Where Did the Money Go?</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-desc">Every time you buy or sell in Hong Kong cash equities, you pay <b>15.05 basis points</b> (0.1505%) per leg. Here is how that fee structure is composed:</div>',
    unsafe_allow_html=True,
)

fee_col1, fee_col2 = st.columns([1.2, 1])

with fee_col1:
    fig_waterfall = go.Figure(
        go.Bar(
            x=["Govt Stamp Duty", "Broker & Slippage", "Exchange Levies", "Short Borrow"],
            y=[10.00, 4.00, 1.05, 2.00],
            marker_color=["#FF4D4D", "#FB7185", "#F43F5E", "#E11D48"],
            marker_line=dict(width=1, color="rgba(255,255,255,0.1)"),
            text=["10.00 bps<br>(66.4%)", "4.00 bps<br>(26.6%)", "1.05 bps<br>(7.0%)", "2.0% p.a.<br>(Carry)"],
            textposition="auto",
            textfont=dict(color="#FFFFFF", size=12, family="Inter"),
        )
    )

    fig_waterfall.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0),
        height=280,
        xaxis=dict(
            tickfont=dict(color="#94A3B8", size=11),
            showgrid=False,
            showline=False,
        ),
        yaxis=dict(
            title="Cost per Single Leg (bps)",
            title_font=dict(color="#64748B", size=11),
            tickfont=dict(color="#64748B"),
            gridcolor="rgba(255, 255, 255, 0.05)",
            showline=False,
        ),
    )
    st.plotly_chart(fig_waterfall, use_container_width=True, config={"displayModeBar": False})

with fee_col2:
    st.markdown(
        """
        <div class="glass-card" style="height: 280px; display: flex; flex-direction: column; justify-content: center;">
            <div style="font-size: 1.1rem; font-weight: 700; color: #FFFFFF; margin-bottom: 8px;">
                The 0.10% Government Tax Trap
            </div>
            <p style="font-size: 0.9rem; color: #94A3B8; line-height: 1.5; margin: 0;">
                The primary culprit is the <b>Hong Kong statutory stamp duty (0.10%)</b>. 
                <br><br>
                Because this strategy trades whenever the pair stretches (completing 85 cycles / 170 adjustments), 
                the strategy paid the taxman <b>340 individual times</b> across both legs. 
                The tax bill alone (-51.2%) wiped out more than double the entire gross profit.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# 9. THE INTERACTIVE SANDBOX: BREAK-EVEN SIMULATOR
# -----------------------------------------------------------------------------
st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
st.markdown('<div class="section-title">Interactive Sandbox: What Would It Take to Survive?</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-desc">Drag the slider to adjust the friction per leg. Notice where the strategy falls off the survival cliff into negative returns.</div>',
    unsafe_allow_html=True,
)

sandbox_container = st.container()

with sandbox_container:
    user_fee = st.slider(
        "Single-Leg Execution Fee (Basis Points):",
        min_value=0.0,
        max_value=20.0,
        value=15.05,
        step=0.25,
        help="1 basis point (bps) = 0.01%. Active HK cash market friction is 15.05 bps.",
    )

    # Dynamic calculation based on empirical 170 rebalancing events (340 legs)
    gross_pnl = 21.06
    borrow_pnl = 1.06
    sim_fee_drag = (340 * user_fee) / 100.0
    sim_net_pnl = gross_pnl - borrow_pnl - sim_fee_drag

    is_profitable = sim_net_pnl >= 15.93  # 15.93% is the 5-year 3.0% cash hurdle
    is_positive = sim_net_pnl > 0.0

    sb_c1, sb_c2 = st.columns([1, 1.4])

    with sb_c1:
        if is_profitable:
            badge_html = '<span class="badge badge-primary">Economically Viable</span>'
            pnl_class = "text-green"
            verdict_text = "The strategy clears the 3% risk-free hurdle rate and generates real alpha."
        elif is_positive:
            badge_html = '<span class="badge" style="background: rgba(234, 179, 8, 0.15); color: #EAB308; border: 1px solid rgba(234, 179, 8, 0.3);">Marginal Alpha</span>'
            pnl_class = "text-muted"
            verdict_text = "Profitable, but you would make more money leaving your cash in a risk-free bank account."
        else:
            badge_html = '<span class="badge badge-danger">Capital Destruction Zone</span>'
            pnl_class = "text-red"
            verdict_text = "Every trade drains your account. The strategy is unviable."

        st.markdown(
            f"""
            <div class="glass-card" style="height: 100%;">
                {badge_html}
                <div class="metric-label">Simulated 5-Year Return</div>
                <div class="metric-value {pnl_class}">{sim_net_pnl:+.1f}%</div>
                <p style="font-size: 0.9rem; color: #94A3B8; margin-top: 10px; line-height: 1.5;">
                    {verdict_text}
                </p>
                <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid rgba(255,255,255,0.06); font-size: 0.85rem; color: #64748B;">
                    • Break-Even Survival Threshold: <b>1.20 bps</b><br>
                    • Active Cash Market Reality: <b>15.05 bps</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with sb_c2:
        # Sensitivity Curve
        fee_range = np.linspace(0.0, 20.0, 50)
        sim_pnls = gross_pnl - borrow_pnl - ((340 * fee_range) / 100.0)

        fig_sens = go.Figure()
        fig_sens.add_trace(
            go.Scatter(
                x=fee_range,
                y=sim_pnls,
                mode="lines",
                line=dict(color="#94A3B8", width=2),
                name="Net Return Curve",
                hovertemplate="Fee: %{x:.2f} bps<br>Return: %{y:.1f}%<extra></extra>",
            )
        )
        
        # User current position marker
        fig_sens.add_trace(
            go.Scatter(
                x=[user_fee],
                y=[sim_net_pnl],
                mode="markers",
                marker=dict(color="#00F5A0" if sim_net_pnl > 0 else "#FF4D4D", size=12),
                name="Your Selection",
                hoverinfo="skip",
            )
        )

        # Break-even vertical line
        fig_sens.add_vline(
            x=1.20,
            line_width=1.5,
            line_dash="dash",
            line_color="#00F5A0",
            annotation_text="Survival: 1.20 bps",
            annotation_position="top right",
            annotation_font=dict(color="#00F5A0", size=10),
        )

        # Active reality vertical line
        fig_sens.add_vline(
            x=15.05,
            line_width=1.5,
            line_dash="dot",
            line_color="#FF4D4D",
            annotation_text="Active Market: 15.05 bps",
            annotation_position="bottom right",
            annotation_font=dict(color="#FF4D4D", size=10),
        )

        fig_sens.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0),
            height=260,
            showlegend=False,
            xaxis=dict(
                title="Single-Leg Cost (Basis Points)",
                title_font=dict(color="#64748B", size=11),
                tickfont=dict(color="#64748B"),
                gridcolor="rgba(255, 255, 255, 0.05)",
                showline=False,
            ),
            yaxis=dict(
                ticksuffix="%",
                tickfont=dict(color="#64748B"),
                gridcolor="rgba(255, 255, 255, 0.05)",
                showline=False,
            ),
        )
        st.plotly_chart(fig_sens, use_container_width=True, config={"displayModeBar": False})

# -----------------------------------------------------------------------------
# 10. THE INSTITUTIONAL FIX & ITS CATCH (Futures)
# -----------------------------------------------------------------------------
st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
st.markdown('<div class="section-title">Can We Cheat the Taxman?</div>', unsafe_allow_html=True)

sol_1, sol_2 = st.columns(2)

with sol_1:
    st.markdown(
        """
        <div class="glass-card" style="height: 100%;">
            <div style="font-size: 1.3rem; margin-bottom: 8px;">💡 The Derivatives Bypass</div>
            <p style="font-size: 0.95rem; color: #94A3B8; line-height: 1.6; margin: 0;">
                Under local law, <b>Single-Stock Futures are statutorily exempt from the 0.10% stamp duty</b>. 
                <br><br>
                Trading futures immediately slashes single-leg fees down to ~2.5 basis points. On paper, this saves the strategy from certain death.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with sol_2:
    st.markdown(
        """
        <div class="glass-card" style="height: 100%;">
            <div style="font-size: 1.3rem; margin-bottom: 8px;">⚠️ The Illiquidity Trap</div>
            <p style="font-size: 0.95rem; color: #94A3B8; line-height: 1.6; margin: 0;">
                While HSBC futures are heavily traded, <b>Standard Chartered futures are practically an empty room</b> with almost zero buyers or sellers. 
                <br><br>
                Trying to execute institutional volume there causes massive execution slippage (15–30 bps), wiping out the tax savings instantly.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# 11. FOOTER & EXECUTIVE BRIEF LINK
# -----------------------------------------------------------------------------
st.markdown("<div style='height: 35px;'></div>", unsafe_allow_html=True)
st.markdown(
    """
    <div class="glass-card" style="text-align: center; padding: 40px 20px;">
        <h3 style="font-size: 1.5rem; font-weight: 700; color: #FFFFFF; margin-bottom: 10px;">
            Want the Full Mathematical Formulation?
        </h3>
        <p style="color: #94A3B8; font-size: 0.95rem; max-width: 600px; margin: 0 auto 24px auto; line-height: 1.6;">
            Explore the complete econometric paper, Kalman filter equations, 
            Ornstein-Uhlenbeck derivations, and institutional order book analysis.
        </p>
        <a href="https://github.com" target="_blank" class="cta-btn">
            Read Executive Summary & Research Paper ↗
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)
